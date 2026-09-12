#!/usr/bin/env bash
set -euo pipefail

# モデルリストは launch.sh 経由で MODEL_NAMES_CSV 環境変数として渡される。
# 形式: Hugging Face のリポジトリ名/ファイル名.gguf をカンマ区切りで指定。
MODEL_NAMES=()

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAMES_CSV="${MODEL_NAMES_CSV:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-1800}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-11434}"
LLAMA_ROUTER_PORT="${LLAMA_ROUTER_PORT:-$((PORT + 1))}"
# CONTEXT_SIZE が指定されない場合は各モデルの GGUF メタデータから
# 推奨(学習時最大)コンテキスト長を自動検出して使用する。
# MAX_CONTEXT_SIZE を指定すると、自動検出値に上限(VRAM保護用)をかけられる。
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
# KV キャッシュが VRAM に収まらない場合の自動切り詰めの下限と丸め単位。
MIN_CONTEXT_SIZE="${MIN_CONTEXT_SIZE:-2048}"
CONTEXT_SIZE_STEP="${CONTEXT_SIZE_STEP:-1024}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
MODELS_MAX="${MODELS_MAX:-1}"
FLASH_ATTN="${FLASH_ATTN:-on}"
# llama.cpp の既定値(2048/512)より小さくし、prompt処理用の一時VRAMを抑える。
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
# --fit と手動tensor-split計算の両方で、GPUごとに残すVRAM余白。
# 大規模なUD量子化モデルでは、計算バッファに加えて生成開始時のCUDA
# ワークスペースも必要になるため4 GiBを確保する。
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-4096}"
MOE_CPU_OFFLOAD="${MOE_CPU_OFFLOAD:-auto}"
MOE_ACTIVE_RATIO_THRESHOLD="${MOE_ACTIVE_RATIO_THRESHOLD:-0.125}"
MOE_RAM_RESERVE_MIB="${MOE_RAM_RESERVE_MIB:-8192}"
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
[[ "$MIN_CONTEXT_SIZE" =~ ^[0-9]+$ ]] && ((MIN_CONTEXT_SIZE > 0)) || die "MIN_CONTEXT_SIZE must be a positive integer"
[[ "$CONTEXT_SIZE_STEP" =~ ^[0-9]+$ ]] && ((CONTEXT_SIZE_STEP > 0)) || die "CONTEXT_SIZE_STEP must be a positive integer"
[[ "$N_GPU_LAYERS" == "auto" || "$N_GPU_LAYERS" == "all" || "$N_GPU_LAYERS" =~ ^-?[0-9]+$ ]] || die "N_GPU_LAYERS must be an integer, 'auto', or 'all'"
[[ "$MODELS_MAX" =~ ^[0-9]+$ ]] || die "MODELS_MAX must be an integer"
[[ "$FLASH_ATTN" == "on" || "$FLASH_ATTN" == "off" || "$FLASH_ATTN" == "auto" ]] || die "FLASH_ATTN must be 'on', 'off', or 'auto'"
[[ "$BATCH_SIZE" =~ ^[0-9]+$ ]] && ((BATCH_SIZE > 0)) || die "BATCH_SIZE must be a positive integer"
[[ "$UBATCH_SIZE" =~ ^[0-9]+$ ]] && ((UBATCH_SIZE > 0 && UBATCH_SIZE <= BATCH_SIZE)) || die "UBATCH_SIZE must be a positive integer no greater than BATCH_SIZE"
[[ "$VRAM_RESERVE_MIB" =~ ^[0-9]+$ ]] && ((VRAM_RESERVE_MIB > 0)) || die "VRAM_RESERVE_MIB must be a positive integer"
[[ "$MOE_CPU_OFFLOAD" == "auto" || "$MOE_CPU_OFFLOAD" == "off" || "$MOE_CPU_OFFLOAD" == "all" ]] || die "MOE_CPU_OFFLOAD must be 'auto', 'off', or 'all'"
awk -v value="$MOE_ACTIVE_RATIO_THRESHOLD" 'BEGIN { exit !(value > 0 && value <= 1) }' || die "MOE_ACTIVE_RATIO_THRESHOLD must be greater than 0 and no greater than 1"
[[ "$MOE_RAM_RESERVE_MIB" =~ ^[0-9]+$ ]] || die "MOE_RAM_RESERVE_MIB must be a non-negative integer"
[[ "$TENSOR_SPLIT_MODE" == "auto" || "$TENSOR_SPLIT_MODE" == "off" ]] || die "TENSOR_SPLIT_MODE must be 'auto' or 'off'"
case "$KV_CACHE_TYPE" in
  ""|f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
  *) die "KV_CACHE_TYPE must be one of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1" ;;
esac

PRESET_FILE="$MODEL_DIR/.models-preset.ini"
PRESET_SECTION_DIR="$MODEL_DIR/.models-preset.d"
MODEL_ALIAS_FILE="$MODEL_DIR/.model-aliases.tsv"

/usr/local/bin/sync-models.sh

log "Starting OpenAI-compatible llama-server router on internal port $LLAMA_ROUTER_PORT"
log "Available models: ${!allowed_files[*]} (models-max=$MODELS_MAX)"
llama-server \
  --models-preset "$PRESET_FILE" \
  --models-max "$MODELS_MAX" \
  --flash-attn "$FLASH_ATTN" \
  --batch-size "$BATCH_SIZE" \
  --ubatch-size "$UBATCH_SIZE" \
  --kv-unified \
  --fit-target "$VRAM_RESERVE_MIB" \
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

export MODEL_DIR MODEL_IDLE_SECONDS CONTEXT_SIZE MAX_CONTEXT_SIZE MIN_CONTEXT_SIZE CONTEXT_SIZE_STEP N_GPU_LAYERS MODELS_MAX FLASH_ATTN BATCH_SIZE UBATCH_SIZE VRAM_RESERVE_MIB MOE_CPU_OFFLOAD MOE_ACTIVE_RATIO_THRESHOLD MOE_RAM_RESERVE_MIB KV_CACHE_TYPE TENSOR_SPLIT_MODE
export PRESET_FILE PRESET_SECTION_DIR MODEL_ALIAS_FILE
export LLAMA_ROUTER_URL="http://127.0.0.1:$LLAMA_ROUTER_PORT"

log "Starting lazy OpenAI-compatible proxy on port $PORT"
python3 /usr/local/bin/lazy-llama-proxy.py \
  --listen-host "$HOST" \
  --listen-port "$PORT" \
  --backend "$LLAMA_ROUTER_URL" &
proxy_pid=$!

wait -n "$router_pid" "$proxy_pid"
