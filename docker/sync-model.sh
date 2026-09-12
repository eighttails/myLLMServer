#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAMES_CSV="${MODEL_NAMES_CSV:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-1800}"
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"
LLAMA_ROUTER_PORT="${LLAMA_ROUTER_PORT:-11435}"

PRESET_FILE="${PRESET_FILE:-$MODEL_DIR/.models-preset.ini}"
PRESET_SECTION_DIR="${PRESET_SECTION_DIR:-$MODEL_DIR/.models-preset.d}"
MODEL_ALIAS_FILE="${MODEL_ALIAS_FILE:-$MODEL_DIR/.model-aliases.tsv}"

log() { printf '[llama-wrapper] %s\n' "$*" >&2; }
die() { printf '[llama-wrapper] error: %s\n' "$*" >&2; exit 1; }

# /models/model_list.txt が存在すればそこから最新のモデルリストを読み込む
if [[ -f "$MODEL_DIR/model_list.txt" ]]; then
  file_csv="$(awk '!/^[[:space:]]*#/ && !/^[[:space:]]*$/' "$MODEL_DIR/model_list.txt" | tr '\n' ',' | sed 's/,$//')"
  if [[ -n "$file_csv" ]]; then
    MODEL_NAMES_CSV="$file_csv"
  fi
fi

if [[ -n "$MODEL_NAMES_CSV" ]]; then
  IFS=',' read -r -a MODEL_NAMES <<< "$MODEL_NAMES_CSV"
else
  MODEL_NAMES=()
fi
((${#MODEL_NAMES[@]} > 0)) || die "no valid model entries found in model_list.txt or MODEL_NAMES_CSV"

mkdir -p "$MODEL_DIR"
PRESET_TMP_DIR="${PRESET_SECTION_DIR}.tmp"
ALIAS_TMP_FILE="${MODEL_ALIAS_FILE}.tmp"
rm -rf "$PRESET_TMP_DIR"
mkdir -p "$PRESET_TMP_DIR"
: > "$ALIAS_TMP_FILE"

declare -A allowed_files=()

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

  model_metadata="$(
    gguf-dump --no-tensors --json --json-array "$destination" 2>/dev/null | jq -r '
      (.metadata | to_entries) as $entries
      | ($entries[] | select(.key == "general.architecture") | .value.value) as $architecture
      | ($entries[] | select(.key | endswith(".context_length")) | .value.value) as $context_length
      | select(($architecture | type) == "string" and ($context_length | type) == "number")
      | [$architecture, $context_length] | @tsv
    '
  )"
  [[ "$model_metadata" == *$'\t'* ]] || die "could not detect architecture and context length: $filename"
  model_architecture="${model_metadata%%$'\t'*}"
  detected_context_size="${model_metadata#*$'\t'}"
  advertised_context_size="${CONTEXT_SIZE:-$detected_context_size}"
  if [[ -n "$MAX_CONTEXT_SIZE" ]] && ((advertised_context_size > MAX_CONTEXT_SIZE)); then
    advertised_context_size="$MAX_CONTEXT_SIZE"
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "$alias_name" "$filename" "$model_architecture" "$advertised_context_size" >> "$ALIAS_TMP_FILE"
  {
    printf '[%s]\n' "$alias_name"
    printf 'model = %s\n' "$destination"
    printf 'sleep-idle-seconds = %s\n' "$MODEL_IDLE_SECONDS"
    printf 'n-gpu-layers = %s\n' "$N_GPU_LAYERS"
    printf 'ctx-size = %s\n' "$advertised_context_size"
    kv_type="${KV_CACHE_TYPE:-f16}"
    printf 'cache-type-k = %s\n' "$kv_type"
    printf 'cache-type-v = %s\n' "$kv_type"
    printf '\n'
  } > "$PRESET_TMP_DIR/$alias_name.ini"
done

# リストにない .gguf ファイル（使わなくなったモデル）をディスクから削除
shopt -s nullglob
for cached_file in "$MODEL_DIR"/*.gguf; do
  filename="$(basename "$cached_file")"
  if [[ -z "${allowed_files[$filename]+x}" ]]; then
    log "Removing model not in model_list: $filename"
    rm -f -- "$cached_file"
    rm -f -- "$cached_file.part"
    rm -f -- "$MODEL_DIR/llama-bench-$filename.json"
  fi
done
shopt -u nullglob

rm -rf "$PRESET_SECTION_DIR"
mv "$PRESET_TMP_DIR" "$PRESET_SECTION_DIR"
mv "$ALIAS_TMP_FILE" "$MODEL_ALIAS_FILE"

{
  printf '[*]\n'
  printf 'fit = on\n'
  printf '\n'
  for section_file in "$PRESET_SECTION_DIR"/*.ini; do
    [[ -f "$section_file" ]] || continue
    cat "$section_file"
  done
} > "$PRESET_FILE"

if curl --fail --silent "http://127.0.0.1:${LLAMA_ROUTER_PORT}/models?reload=1" >/dev/null 2>&1; then
  log "Reloaded models-preset on llama-server"
fi

log "Model sync completed successfully. Active models in list: ${!allowed_files[*]}"
