#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-my-llm-server}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-$SCRIPT_DIR/model_list.yml}"
MODEL_LIST_EXAMPLE="${MODEL_LIST_EXAMPLE:-$SCRIPT_DIR/model_list.example.yml}"

if [[ ! -f "$MODEL_LIST_FILE" ]]; then
  if [[ ! -f "$MODEL_LIST_EXAMPLE" ]]; then
    echo "error: model list file and example are both missing" >&2
    exit 1
  fi
  echo "model list file not found: $MODEL_LIST_FILE" >&2
  echo "copying from example: $MODEL_LIST_EXAMPLE" >&2
  cp "$MODEL_LIST_EXAMPLE" "$MODEL_LIST_FILE"
fi

mkdir -p "$MODEL_DIR"
cp "$MODEL_LIST_FILE" "$MODEL_DIR/model_list.yml"

echo "Reloading model list and syncing models in container '$CONTAINER_NAME'..."
docker exec "$CONTAINER_NAME" python3 /usr/local/bin/sync-model.py
