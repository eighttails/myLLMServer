#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-my-llama-server:latest}"
CONTAINER_NAME="${CONTAINER_NAME:-my-llama-server}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
PUID="${PUID:-$(id -u)}"
PGID="${PGID:-$(id -g)}"

mkdir -p "$MODEL_DIR"
chmod 0755 "$MODEL_DIR"

docker build --tag "$IMAGE_NAME" "$SCRIPT_DIR/docker"

if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  echo "Removing existing container: $CONTAINER_NAME"
  docker rm --force "$CONTAINER_NAME"
fi

gpu_args=(--gpus all)
env_args=(-e "PUID=$PUID" -e "PGID=$PGID")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  env_args+=(-e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
fi
# 以下の環境変数が設定されていればコンテナに引き継ぐ (start-llama.sh 参照)
for var in MODEL_NAMES_CSV HF_ENDPOINT MODEL_IDLE_SECONDS CONTEXT_SIZE MAX_CONTEXT_SIZE N_GPU_LAYERS MODELS_MAX KV_CACHE_TYPE; do
  if [[ -n "${!var:-}" ]]; then
    env_args+=(-e "$var=${!var}")
  fi
done

docker run --detach --name "$CONTAINER_NAME" \
  --restart unless-stopped \
  "${gpu_args[@]}" \
  "${env_args[@]}" \
  --user "$PUID:$PGID" \
  --publish "${PORT:-8080}:${PORT:-8080}" \
  --volume "$MODEL_DIR:/models" \
  "$IMAGE_NAME"

