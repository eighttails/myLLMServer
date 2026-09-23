#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-my-llm-server:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-my-llm-server}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-$SCRIPT_DIR/model_list.yml}"
MODEL_LIST_EXAMPLE="${MODEL_LIST_EXAMPLE:-$SCRIPT_DIR/model_list.example.yml}"
PUID="${PUID:-$(id -u)}"
PGID="${PGID:-$(id -g)}"

# model_list.yml がなければ model_list.example.yml をコピーする
[[ -f "$MODEL_LIST_FILE" ]] || { echo "model list file not found: $MODEL_LIST_FILE" >&2; echo "copying from example: $MODEL_LIST_EXAMPLE" >&2; [[ -f "$MODEL_LIST_EXAMPLE" ]] && cp "$MODEL_LIST_EXAMPLE" "$MODEL_LIST_FILE"; exit 1; }

mkdir -p "$MODEL_DIR"
chmod 0755 "$MODEL_DIR"
cp "$MODEL_LIST_FILE" "$MODEL_DIR/model_list.yml"

[[ -s "$MODEL_LIST_FILE" ]] || { echo "no valid model entries in $MODEL_LIST_FILE" >&2; exit 1; }
echo "Using models from $MODEL_LIST_FILE"

docker build --pull --tag "$IMAGE_NAME" "$SCRIPT_DIR/docker"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm --force "$CONTAINER_NAME"
fi

gpu_args=(--gpus all)
env_args=(-e "PUID=$PUID" -e "PGID=$PGID" -e "MODEL_LIST_FILE=/models/model_list.yml")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  env_args+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
# 以下の環境変数が設定されていればコンテナに引き継ぐ (docker/start-llama.sh 参照)
for var in PORT HF_ENDPOINT MODEL_IDLE_SECONDS CONTEXT_SIZE MAX_CONTEXT_SIZE MIN_CONTEXT_SIZE CONTEXT_SIZE_STEP N_GPU_LAYERS MODELS_MAX FLASH_ATTN BATCH_SIZE UBATCH_SIZE VRAM_RESERVE_MIB MOE_CPU_OFFLOAD MOE_ACTIVE_RATIO_THRESHOLD MOE_RAM_RESERVE_MIB KV_CACHE_TYPE TENSOR_SPLIT_MODE SPLIT_MODE THINKING_MODE GENERATION_LOOP_DETECTION GENERATION_LOOP_WINDOW_CHARS GENERATION_LOOP_MIN_PATTERN_CHARS GENERATION_LOOP_MAX_PATTERN_CHARS GENERATION_LOOP_REPEAT_COUNT GENERATION_LOOP_MIN_REPEATED_CHARS GENERATION_LOOP_LINE_REPEAT_COUNT TOOL_LOOP_DETECTION TOOL_LOOP_REPEAT_COUNT TOOL_LOOP_MAX_CYCLE_LENGTH LLAMA_ROUTER_PORT http_proxy https_proxy no_proxy HTTP_PROXY HTTPS_PROXY NO_PROXY all_proxy ALL_PROXY; do
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
