#!/usr/bin/env bash
set -euo pipefail

# モデルリストは run-llama.sh 経由で MODEL_NAMES_CSV 環境変数として渡される。
# 形式: Hugging Face のリポジトリ名/ファイル名.gguf をカンマ区切りで指定。
MODEL_NAMES=()

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAMES_CSV="${MODEL_NAMES_CSV:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-300}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-11434}"
LLAMA_ROUTER_PORT="${LLAMA_ROUTER_PORT:-$((PORT + 1))}"
# CONTEXT_SIZE が指定されない場合は各モデルの GGUF メタデータから
# 推奨(学習時最大)コンテキスト長を自動検出して使用する。
# MAX_CONTEXT_SIZE を指定すると、自動検出値に上限(VRAM保護用)をかけられる。
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
MODELS_MAX="${MODELS_MAX:-1}"
# 複数 GPU の性能差(生成速度)を考慮した tensor-split を自動計算するかどうか。
# auto: 2 GPU 以上あり、各 GPU の生成速度比が十分な差(閾値以上)であれば手動計算した
#       tensor-split / n-gpu-layers を明示指定する(--fit は off にする)。
#       計算に失敗した場合や条件を満たさない場合は自動的に --fit on にフォールバックする。
# off:  常に --fit on に委ねる(従来の挙動)。
TENSOR_SPLIT_MODE="${TENSOR_SPLIT_MODE:-auto}"
# KV キャッシュの量子化タイプ (f16 既定より VRAM を大幅削減できる: q8_0 で約1/2, q4_0 で約1/4)
# allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
# 未指定の場合は、空き VRAM とモデルサイズ・コンテキスト長から必要な KV キャッシュ量を見積もり、
# 収まる範囲でなるべく精度の高い(f16に近い)タイプを自動選択する。
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"

log() { printf '[llama-wrapper] %s\n' "$*" >&2; }
die() { printf '[llama-wrapper] error: %s\n' "$*" >&2; exit 1; }

[[ "$MODEL_IDLE_SECONDS" =~ ^[0-9]+$ ]] || die "MODEL_IDLE_SECONDS must be an integer"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be an integer"
[[ "$LLAMA_ROUTER_PORT" =~ ^[0-9]+$ ]] || die "LLAMA_ROUTER_PORT must be an integer"
[[ "$LLAMA_ROUTER_PORT" != "$PORT" ]] || die "LLAMA_ROUTER_PORT must be different from PORT"
[[ -z "$CONTEXT_SIZE" || "$CONTEXT_SIZE" =~ ^[0-9]+$ ]] || die "CONTEXT_SIZE must be an integer"
[[ -z "$MAX_CONTEXT_SIZE" || "$MAX_CONTEXT_SIZE" =~ ^[0-9]+$ ]] || die "MAX_CONTEXT_SIZE must be an integer"
[[ "$N_GPU_LAYERS" == "auto" || "$N_GPU_LAYERS" == "all" || "$N_GPU_LAYERS" =~ ^-?[0-9]+$ ]] || die "N_GPU_LAYERS must be an integer, 'auto', or 'all'"
[[ "$MODELS_MAX" =~ ^[0-9]+$ ]] || die "MODELS_MAX must be an integer"
[[ "$TENSOR_SPLIT_MODE" == "auto" || "$TENSOR_SPLIT_MODE" == "off" ]] || die "TENSOR_SPLIT_MODE must be 'auto' or 'off'"
case "$KV_CACHE_TYPE" in
  ""|f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
  *) die "KV_CACHE_TYPE must be one of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1" ;;
esac

if [[ -n "$MODEL_NAMES_CSV" ]]; then
  IFS=',' read -r -a MODEL_NAMES <<< "$MODEL_NAMES_CSV"
fi
((${#MODEL_NAMES[@]} > 0)) || die "MODEL_NAMES must not be empty"

mkdir -p "$MODEL_DIR"
declare -A allowed_files=()
PRESET_FILE="$MODEL_DIR/.models-preset.ini"
PRESET_SECTION_DIR="$MODEL_DIR/.models-preset.d"
MODEL_ALIAS_FILE="$MODEL_DIR/.model-aliases.tsv"
rm -rf "$PRESET_SECTION_DIR"
mkdir -p "$PRESET_SECTION_DIR"
: > "$MODEL_ALIAS_FILE"

for model_spec in "${MODEL_NAMES[@]}"; do
  [[ "$model_spec" == */* ]] || die "invalid model spec: $model_spec"
  repo="${model_spec%/*}"
  filename="${model_spec##*/}"
  [[ "$filename" == *.gguf ]] || die "model must be a .gguf file: $model_spec"
  destination="$MODEL_DIR/$filename"
  alias_name="${filename%.gguf}"
  allowed_files["$filename"]=1

  if [[ ! -f "$destination" ]]; then
    log "Downloading $model_spec"
    temporary="$destination.part"
    rm -f "$temporary"
    curl --fail --location --retry 3 --retry-delay 2 \
      --output "$temporary" \
      "$HF_ENDPOINT/$repo/resolve/main/$filename?download=true"
    mv "$temporary" "$destination"
  else
    log "Using cached model: $filename"
  fi

  printf '%s\t%s\n' "$alias_name" "$filename" >> "$MODEL_ALIAS_FILE"
  {
    printf '[%s]\n' "$alias_name"
    printf 'model = %s\n' "$destination"
    printf 'sleep-idle-seconds = %s\n' "$MODEL_IDLE_SECONDS"
    printf 'n-gpu-layers = %s\n' "$N_GPU_LAYERS"
    if [[ -n "$CONTEXT_SIZE" ]]; then
      printf 'ctx-size = %s\n' "$CONTEXT_SIZE"
    fi
    if [[ -n "$KV_CACHE_TYPE" ]]; then
      printf 'cache-type-k = %s\n' "$KV_CACHE_TYPE"
      printf 'cache-type-v = %s\n' "$KV_CACHE_TYPE"
    fi
    printf '\n'
  } > "$PRESET_SECTION_DIR/$alias_name.ini"
done

shopt -s nullglob
for cached_file in "$MODEL_DIR"/*.gguf; do
  filename="$(basename "$cached_file")"
  if [[ -z "${allowed_files[$filename]+x}" ]]; then
    log "Removing model not in MODEL_NAMES: $filename"
    rm -f -- "$cached_file"
  fi
done
shopt -u nullglob

{
  printf '[*]\n'
  printf 'fit = on\n'
  printf '\n'
  for section_file in "$PRESET_SECTION_DIR"/*.ini; do
    [[ -f "$section_file" ]] || continue
    cat "$section_file"
  done
} > "$PRESET_FILE"

log "Starting OpenAI-compatible llama-server router on internal port $LLAMA_ROUTER_PORT"
log "Available models: ${!allowed_files[*]} (models-max=$MODELS_MAX)"
llama-server \
  --models-preset "$PRESET_FILE" \
  --models-max "$MODELS_MAX" \
  --host 127.0.0.1 \
  --port "$LLAMA_ROUTER_PORT" &
router_pid=$!

cleanup() {
  local status="$?"
  trap - EXIT INT TERM
  if [[ -n "${proxy_pid:-}" ]] && kill -0 "$proxy_pid" >/dev/null 2>&1; then
    kill "$proxy_pid" >/dev/null 2>&1 || true
    wait "$proxy_pid" 2>/dev/null || true
  fi
  if kill -0 "$router_pid" >/dev/null 2>&1; then
    kill "$router_pid" >/dev/null 2>&1 || true
    wait "$router_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

router_ready=0
for _ in $(seq 1 60); do
  if curl --fail --silent "http://127.0.0.1:$LLAMA_ROUTER_PORT/models" >/dev/null 2>&1; then
    router_ready=1
    break
  fi
  if ! kill -0 "$router_pid" >/dev/null 2>&1; then
    wait "$router_pid"
  fi
  sleep 1
done
((router_ready == 1)) || die "llama-server router did not become ready"

export MODEL_DIR MODEL_IDLE_SECONDS CONTEXT_SIZE MAX_CONTEXT_SIZE N_GPU_LAYERS MODELS_MAX KV_CACHE_TYPE TENSOR_SPLIT_MODE
export PRESET_FILE PRESET_SECTION_DIR MODEL_ALIAS_FILE
export LLAMA_ROUTER_URL="http://127.0.0.1:$LLAMA_ROUTER_PORT"

log "Starting lazy OpenAI-compatible proxy on port $PORT"
python3 /usr/local/bin/lazy-llama-proxy.py \
  --listen-host "$HOST" \
  --listen-port "$PORT" \
  --backend "$LLAMA_ROUTER_URL" &
proxy_pid=$!

wait -n "$router_pid" "$proxy_pid"
