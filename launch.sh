#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-my-llm-server:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-my-llm-server}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-$SCRIPT_DIR/model_list.txt}"
MODEL_LIST_EXAMPLE="${MODEL_LIST_EXAMPLE:-$SCRIPT_DIR/model_list.example}"
PUID="${PUID:-$(id -u)}"
PGID="${PGID:-$(id -g)}"

# model_list.txt がなければ model_list.example をコピーする
[[ -f "$MODEL_LIST_FILE" ]] || { echo "model list file not found: $MODEL_LIST_FILE" >&2; echo "copying from example: $MODEL_LIST_EXAMPLE" >&2; [[ -f "$MODEL_LIST_EXAMPLE" ]] && cp "$MODEL_LIST_EXAMPLE" "$MODEL_LIST_FILE"; exit 1; }

mkdir -p "$MODEL_DIR"
chmod 0755 "$MODEL_DIR"
cp "$MODEL_LIST_FILE" "$MODEL_DIR/model_list.txt"

# model_list.txt からモデルリストを読み込む (コメント行・空行は無視、カンマ区切りで結合)
MODEL_NAMES_CSV="$(awk '!/^[[:space:]]*#/ && !/^[[:space:]]*$/' "$MODEL_LIST_FILE" | tr '\n' ',' | sed 's/,$//')"
[[ -n "$MODEL_NAMES_CSV" ]] || { echo "no valid model entries in $MODEL_LIST_FILE" >&2; exit 1; }
echo "Using models from $MODEL_LIST_FILE:"
printf '  %s\n' "${MODEL_NAMES_CSV//,/$'\n  '}"

docker build --tag "$IMAGE_NAME" "$SCRIPT_DIR/docker"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm --force "$CONTAINER_NAME"
fi

gpu_args=(--gpus all)
env_args=(-e "PUID=$PUID" -e "PGID=$PGID" -e "MODEL_NAMES_CSV=$MODEL_NAMES_CSV")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  env_args+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
# 以下の環境変数が設定されていればコンテナに引き継ぐ (docker/start-llama.sh 参照)
for var in PORT HF_ENDPOINT MODEL_IDLE_SECONDS CONTEXT_SIZE MAX_CONTEXT_SIZE MIN_CONTEXT_SIZE CONTEXT_SIZE_STEP N_GPU_LAYERS MODELS_MAX FLASH_ATTN BATCH_SIZE UBATCH_SIZE VRAM_RESERVE_MIB MOE_CPU_OFFLOAD MOE_ACTIVE_RATIO_THRESHOLD MOE_RAM_RESERVE_MIB KV_CACHE_TYPE TENSOR_SPLIT_MODE LLAMA_ROUTER_PORT; do
  if [[ -n "${!var:-}" ]]; then
    env_args+=(-e "$var=${!var}")
  fi
done

docker run --detach --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  "${gpu_args[@]}" \
  "${env_args[@]}" \
  --user "$PUID:$PGID" \
  --publish "${PORT:-11434}:${PORT:-11434}" \
  --volume "$MODEL_DIR:/models" \
  "$IMAGE_NAME"
