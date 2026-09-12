#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-my-llm-server}"
MODEL_DIR="${MODEL_DIR:-$SCRIPT_DIR/models}"
MODEL_LIST_FILE="${MODEL_LIST_FILE:-$SCRIPT_DIR/model_list.txt}"
MODEL_LIST_EXAMPLE="${MODEL_LIST_EXAMPLE:-$SCRIPT_DIR/model_list.example}"

log() { printf '%s\n' "$*" >&2; }
die() { printf 'error: %s\n' "$*" >&2; exit 1; }

cleanup_unused_models() {
  local model_spec repo filename cached_file
  declare -A allowed_files=()

  while IFS= read -r model_spec; do
    [[ "$model_spec" =~ ^[[:space:]]*# || "$model_spec" =~ ^[[:space:]]*$ ]] && continue
    [[ "$model_spec" == */* ]] || die "invalid model spec: $model_spec"
    repo="${model_spec%/*}"
    filename="${model_spec##*/}"
    [[ -n "$repo" ]] || die "invalid model spec: $model_spec"
    [[ "$filename" == *.gguf ]] || die "model must be a .gguf file: $model_spec"
    allowed_files["$filename"]=1
  done < "$MODEL_LIST_FILE"

  ((${#allowed_files[@]} > 0)) || die "no valid model entries found in $MODEL_LIST_FILE"

  shopt -s nullglob
  for cached_file in "$MODEL_DIR"/*.gguf; do
    filename="$(basename "$cached_file")"
    if [[ -z "${allowed_files[$filename]+x}" ]]; then
      log "Removing model not in model_list: $cached_file"
      rm -f -- "$cached_file"
      rm -f -- "$cached_file.part"
      rm -f -- "$MODEL_DIR/llama-bench-$filename.json"
    fi
  done
  shopt -u nullglob
}

# model_list.txt がなければ model_list.example からコピーする
if [[ ! -f "$MODEL_LIST_FILE" ]]; then
  log "model list file not found: $MODEL_LIST_FILE"
  if [[ -f "$MODEL_LIST_EXAMPLE" ]]; then
    log "copying from example: $MODEL_LIST_EXAMPLE"
    cp "$MODEL_LIST_EXAMPLE" "$MODEL_LIST_FILE"
  else
    die "example model list file not found: $MODEL_LIST_EXAMPLE"
  fi
fi

mkdir -p "$MODEL_DIR"
cp "$MODEL_LIST_FILE" "$MODEL_DIR/model_list.txt"
cleanup_unused_models

# コンテナが起動中であれば、コンテナ内で sync-model.sh を実行する
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1 && \
   [[ "$(docker container inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == "true" ]]; then
  echo "Reloading model_list and syncing models in container '$CONTAINER_NAME'..."
  if ! docker exec "$CONTAINER_NAME" test -f /usr/local/bin/sync-model.sh 2>/dev/null; then
    echo "Installing sync-model.sh into running container..."
    docker cp "$SCRIPT_DIR/docker/sync-model.sh" "$CONTAINER_NAME:/usr/local/bin/sync-model.sh"
    docker exec "$CONTAINER_NAME" chmod 0755 /usr/local/bin/sync-model.sh
  fi
  docker exec "$CONTAINER_NAME" /usr/local/bin/sync-model.sh
else
  echo "Container '$CONTAINER_NAME' is not running."
  echo "Synced model list to $MODEL_DIR/model_list.txt."
  echo "Run ./launch-container.sh to start the container with the updated model list."
fi
