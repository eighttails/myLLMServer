#!/usr/bin/env bash
set -euo pipefail

PORT="${PORT:-11434}"
MODEL_NAME="${1:-}"

URL="http://127.0.0.1:${PORT}/models/unload"

if [[ -n "$MODEL_NAME" ]]; then
  echo "Unloading model: $MODEL_NAME" >&2
  payload=$(jq -n --arg m "$MODEL_NAME" '{model: $m}' 2>/dev/null || printf '{"model": "%s"}' "$MODEL_NAME")
else
  echo "Unloading active model..." >&2
  payload="{}"
fi

curl --fail --silent --show-error -X POST "$URL" \
  -H "Content-Type: application/json" \
  -d "$payload"
echo
