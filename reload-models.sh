#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-my-llm-server}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-$SCRIPT_DIR/model_list.txt}"
MODEL_LIST_EXAMPLE="${MODEL_LIST_EXAMPLE:-$SCRIPT_DIR/model_list.example}"

# model_list.txt がなければ model_list.example からコピーする
if [[ ! -f "$MODEL_LIST_FILE" ]]; then
  echo "model list file not found: $MODEL_LIST_FILE" >&2
  if [[ -f "$MODEL_LIST_EXAMPLE" ]]; then
    echo "copying from example: $MODEL_LIST_EXAMPLE" >&2
    cp "$MODEL_LIST_EXAMPLE" "$MODEL_LIST_FILE"
  else
    echo "error: example model list file not found: $MODEL_LIST_EXAMPLE" >&2
    exit 1
  fi
fi

mkdir -p "$MODEL_DIR"
cp "$MODEL_LIST_FILE" "$MODEL_DIR/model_list.txt"

# コンテナが起動中であれば、コンテナ内で sync-models.sh を実行する
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && \
   [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == "true" ]]; then
  echo "Reloading model_list and syncing models in container '$CONTAINER_NAME'..."
  if ! docker exec "$CONTAINER_NAME" test -f /usr/local/bin/sync-models.sh 2>/dev/null; then
    echo "Installing sync-models.sh into running container..."
    docker cp "$SCRIPT_DIR/docker/sync-models.sh" "$CONTAINER_NAME:/usr/local/bin/sync-models.sh"
    docker exec "$CONTAINER_NAME" chmod 0755 /usr/local/bin/sync-models.sh
  fi
  docker exec "$CONTAINER_NAME" /usr/local/bin/sync-models.sh
else
  echo "Container '$CONTAINER_NAME' is not running."
  echo "Synced model list to $MODEL_DIR/model_list.txt."
  echo "Run ./launch.sh to start the container with the updated model list."
fi
