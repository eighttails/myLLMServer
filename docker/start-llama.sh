#!/usr/bin/env bash
set -euo pipefail

# start-cluster.sh と同じ形式で、1行1モデルを指定する。
# 形式: Hugging Face のリポジトリ名/ファイル名.gguf
MODEL_NAMES=(
  "bartowski/Llama-3.2-3B-Instruct-GGUF/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
  "unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q4_K_M.gguf"
)

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAMES_CSV="${MODEL_NAMES_CSV:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-300}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
# CONTEXT_SIZE が指定されない場合は各モデルの GGUF メタデータから
# 推奨(学習時最大)コンテキスト長を自動検出して使用する。
# MAX_CONTEXT_SIZE を指定すると、自動検出値に上限(VRAM保護用)をかけられる。
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
MODELS_MAX="${MODELS_MAX:-1}"
# KV キャッシュの量子化タイプ (f16 既定より VRAM を大幅削減できる: q8_0 で約1/2, q4_0 で約1/4)
# allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
# 未指定の場合は、空き VRAM とモデルサイズ・コンテキスト長から必要な KV キャッシュ量を見積もり、
# 収まる範囲でなるべく精度の高い(f16に近い)タイプを自動選択する。
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"

log() { printf '[llama-wrapper] %s\n' "$*" >&2; }
die() { printf '[llama-wrapper] error: %s\n' "$*" >&2; exit 1; }

[[ "$MODEL_IDLE_SECONDS" =~ ^[0-9]+$ ]] || die "MODEL_IDLE_SECONDS must be an integer"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be an integer"
[[ -z "$CONTEXT_SIZE" || "$CONTEXT_SIZE" =~ ^[0-9]+$ ]] || die "CONTEXT_SIZE must be an integer"
[[ -z "$MAX_CONTEXT_SIZE" || "$MAX_CONTEXT_SIZE" =~ ^[0-9]+$ ]] || die "MAX_CONTEXT_SIZE must be an integer"
[[ "$N_GPU_LAYERS" == "auto" || "$N_GPU_LAYERS" == "all" || "$N_GPU_LAYERS" =~ ^-?[0-9]+$ ]] || die "N_GPU_LAYERS must be an integer, 'auto', or 'all'"
[[ "$MODELS_MAX" =~ ^[0-9]+$ ]] || die "MODELS_MAX must be an integer"
case "$KV_CACHE_TYPE" in
  ""|f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
  *) die "KV_CACHE_TYPE must be one of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1" ;;
esac

# GGUF ファイルのメタデータから <arch>.context_length を読み取る。
# gguf-dump (llama.cpp 同梱) を使い、取得できなければ空文字を返す。
detect_context_length() {
  local model_file="$1"
  gguf-dump --no-tensors "$model_file" 2>/dev/null \
    | awk -F'= *' '/\.context_length[[:space:]]*=/ { print $2; exit }' \
    | tr -d '[:space:]'
}

# GGUF ファイルのメタデータから KV キャッシュサイズ計算に必要な値を読み取る。
# 見つかった場合 "block_count head_count_kv key_length value_length" を空白区切りで返す。
detect_kv_cache_params() {
  local model_file="$1"
  gguf-dump --no-tensors "$model_file" 2>/dev/null | awk -F'= *' '
    /\.block_count[[:space:]]*=/               { gsub(/[[:space:]]/, "", $2); block_count=$2 }
    /\.attention\.head_count_kv[[:space:]]*=/   { gsub(/[[:space:]]/, "", $2); head_count_kv=$2 }
    /\.attention\.key_length[[:space:]]*=/      { gsub(/[[:space:]]/, "", $2); key_length=$2 }
    /\.attention\.value_length[[:space:]]*=/    { gsub(/[[:space:]]/, "", $2); value_length=$2 }
    END {
      if (block_count != "" && head_count_kv != "" && key_length != "" && value_length != "") {
        print block_count, head_count_kv, key_length, value_length
      }
    }'
}

# 指定したキャッシュタイプ 1要素あたりのバイト数(概算)を返す。
kv_cache_type_bytes_per_element() {
  case "$1" in
    f32) echo 4 ;;
    f16|bf16) echo 2 ;;
    q8_0) echo 1.0625 ;;
    q5_0|q5_1) echo 0.6875 ;;
    q4_0|q4_1|iq4_nl) echo 0.5625 ;;
    *) echo 2 ;;
  esac
}

# 空き VRAM 合計・モデルファイルサイズ・コンテキスト長から、収まる範囲でなるべく精度の高い
# KV キャッシュタイプを選ぶ。精度が高い順に f16 -> q8_0 -> q4_0 を試す。
# 引数: model_file ctx_size free_vram_bytes_total
select_kv_cache_type() {
  local model_file="$1" ctx_size="$2" free_bytes="$3"
  local params block_count head_count_kv key_length value_length
  params="$(detect_kv_cache_params "$model_file")"
  if [[ -z "$params" ]]; then
    log "Could not detect attention params for $(basename "$model_file"); skipping KV cache type auto-selection"
    echo ""
    return
  fi
  read -r block_count head_count_kv key_length value_length <<< "$params"

  local model_bytes
  model_bytes="$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)"
  # モデル重みロード後に KV キャッシュ用として残る VRAM (安全マージンとして 90% だけ使う想定)
  local budget_bytes
  budget_bytes="$(awk -v f="$free_bytes" -v m="$model_bytes" 'BEGIN { b = (f - m) * 0.9; if (b < 0) b = 0; printf "%.0f", b }')"

  local candidate bytes_per_elem needed_bytes
  for candidate in f16 q8_0 q4_0; do
    bytes_per_elem="$(kv_cache_type_bytes_per_element "$candidate")"
    # 必要バイト数 = 2(K+V) * block_count * head_count_kv * (key_length+value_length) * ctx_size * bytes_per_elem / 2
    needed_bytes="$(awk -v bc="$block_count" -v hkv="$head_count_kv" -v kl="$key_length" -v vl="$value_length" \
      -v ctx="$ctx_size" -v bpe="$bytes_per_elem" \
      'BEGIN { printf "%.0f", bc * hkv * (kl + vl) * ctx * bpe }')"
    if (( $(awk -v n="$needed_bytes" -v b="$budget_bytes" 'BEGIN { print (n <= b) ? 1 : 0 }') )); then
      log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB (budget $((budget_bytes / 1024 / 1024)) MiB) -> selected"
      echo "$candidate"
      return
    else
      log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB exceeds budget $((budget_bytes / 1024 / 1024)) MiB"
    fi
  done
  # どれも収まらない場合は最も VRAM を節約できる q4_0 にフォールバックする
  # (それでも収まらない場合は --fit による CPU オフロードに期待する)
  echo "q4_0"
}

# 全 GPU の空き VRAM 合計(バイト)を返す。取得できなければ 0 を返す。
detect_total_free_vram_bytes() {
  local free_mib_list total_mib=0
  free_mib_list="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"
  if [[ -z "$free_mib_list" ]]; then
    echo 0
    return
  fi
  while read -r mib; do
    [[ "$mib" =~ ^[0-9]+$ ]] || continue
    total_mib=$((total_mib + mib))
  done <<< "$free_mib_list"
  echo $((total_mib * 1024 * 1024))
}

if [[ -n "$MODEL_NAMES_CSV" ]]; then
  IFS=',' read -r -a MODEL_NAMES <<< "$MODEL_NAMES_CSV"
fi
((${#MODEL_NAMES[@]} > 0)) || die "MODEL_NAMES must not be empty"

mkdir -p "$MODEL_DIR"
declare -A allowed_files=()

for model_spec in "${MODEL_NAMES[@]}"; do
  [[ "$model_spec" == */* ]] || die "invalid model spec: $model_spec"
  repo="${model_spec%/*}"
  filename="${model_spec##*/}"
  [[ "$filename" == *.gguf ]] || die "model must be a .gguf file: $model_spec"
  destination="$MODEL_DIR/$filename"
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

gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible="${CUDA_VISIBLE_DEVICES//,/ }"
  gpu_count="$(awk '{print NF}' <<< "$visible")"
fi
((gpu_count > 0)) || gpu_count=1

# tensor-split や n-gpu-layers を固定で指定すると、llama-server の自動フィット
# 機能 (--fit, デフォルト有効) が「ユーザー指定済み」と判断して調整を放棄し、
# VRAM に収まらない場合にそのまま OOM で落ちてしまう。
# そのため、これらは明示指定せず --fit に委ね、VRAM に応じて
# GPU レイヤー数やコンテキストサイズなどを自動調整させる(必要なら CPU オフロード)。
log "tensor-split / n-gpu-layers is not fixed; relying on llama-server's --fit auto-adjustment to avoid OOM"

# router server 用の preset (INI) ファイルを生成する。
# ベンチマーク自体は情報収集目的で一度だけ実行し、結果ファイルをキャッシュする。
PRESET_FILE="$MODEL_DIR/.models-preset.ini"
: > "$PRESET_FILE"

for filename in "${!allowed_files[@]}"; do
  model_file="$MODEL_DIR/$filename"
  alias_name="${filename%.gguf}"

  benchmark_marker="$MODEL_DIR/.llama-bench-${filename}.done"
  if [[ ! -f "$benchmark_marker" ]]; then
    log "Running llama-bench for $filename"
    # llama-bench は -ngl に auto/all を受け付けないため、ベンチマーク専用に固定値を使う。
    llama-bench -m "$model_file" -ngl -1 -p 128 -n 32 \
      --output json > "$MODEL_DIR/llama-bench-${filename}.json"
    : > "$benchmark_marker"
  fi

  # CONTEXT_SIZE が明示指定されていればそれを全モデル共通で使う。
  # 未指定ならモデルごとに GGUF から推奨(学習時最大)コンテキスト長を検出する。
  model_ctx_size="$CONTEXT_SIZE"
  if [[ -z "$model_ctx_size" ]]; then
    detected="$(detect_context_length "$model_file")"
    if [[ "$detected" =~ ^[0-9]+$ ]]; then
      model_ctx_size="$detected"
      log "Detected context_length for $filename: $model_ctx_size"
    else
      model_ctx_size=4096
      log "Could not detect context_length for $filename, falling back to $model_ctx_size"
    fi
  fi
  if [[ -n "$MAX_CONTEXT_SIZE" ]] && ((model_ctx_size > MAX_CONTEXT_SIZE)); then
    log "Capping ctx-size for $filename from $model_ctx_size to MAX_CONTEXT_SIZE=$MAX_CONTEXT_SIZE"
    model_ctx_size="$MAX_CONTEXT_SIZE"
  fi

  # KV_CACHE_TYPE が明示指定されていればそれを使う。未指定なら空き VRAM から自動選択する。
  model_kv_cache_type="$KV_CACHE_TYPE"
  if [[ -z "$model_kv_cache_type" ]]; then
    total_free_vram_bytes="$(detect_total_free_vram_bytes)"
    if ((total_free_vram_bytes > 0)); then
      model_kv_cache_type="$(select_kv_cache_type "$model_file" "$model_ctx_size" "$total_free_vram_bytes")"
      if [[ -n "$model_kv_cache_type" ]]; then
        log "Auto-selected KV cache type for $filename: $model_kv_cache_type"
      fi
    else
      log "Could not detect free VRAM; leaving KV cache type at llama-server default (f16) for $filename"
    fi
  fi

  {
    printf '[%s]\n' "$alias_name"
    printf 'model = %s\n' "$model_file"
    printf 'ctx-size = %s\n' "$model_ctx_size"
    printf 'n-gpu-layers = %s\n' "$N_GPU_LAYERS"
    printf 'sleep-idle-seconds = %s\n' "$MODEL_IDLE_SECONDS"
    if [[ -n "$model_kv_cache_type" ]]; then
      printf 'cache-type-k = %s\n' "$model_kv_cache_type"
      printf 'cache-type-v = %s\n' "$model_kv_cache_type"
    fi
    printf '\n'
  } >> "$PRESET_FILE"
done

log "Starting OpenAI-compatible llama-server (router mode) on port $PORT"
log "Available models: ${!allowed_files[*]} (models-max=$MODELS_MAX)"
exec llama-server \
  --fit on \
  --models-preset "$PRESET_FILE" \
  --models-max "$MODELS_MAX" \
  --host "$HOST" \
  --port "$PORT"
