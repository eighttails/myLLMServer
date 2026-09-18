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
METADATA_CACHE_FILE="${METADATA_CACHE_FILE:-$MODEL_DIR/.model-metadata-cache.tsv}"
DOWNLOAD_MAX_STALLED_ATTEMPTS="${DOWNLOAD_MAX_STALLED_ATTEMPTS:-10}"

log() { printf '[llama-wrapper] %s\n' "$*" >&2; }
die() { printf '[llama-wrapper] error: %s\n' "$*" >&2; exit 1; }

# ファイル内容が変わったことを検出するための署名（サイズ:更新時刻）
file_signature() {
  stat -c '%s:%Y' -- "$1" 2>/dev/null || printf 'unknown:unknown'
}

file_size() {
  stat -c '%s' -- "$1" 2>/dev/null || printf '0'
}

# GGUF ヘッダーを解析して "アーキテクチャ<TAB>コンテキスト長" を返す
read_gguf_metadata() {
  gguf-dump --no-tensors --json --json-array "$1" 2>/dev/null | jq -r '
    (.metadata | to_entries) as $entries
    | ($entries[] | select(.key == "general.architecture") | .value.value) as $architecture
    | ($entries[] | select(.key | endswith(".context_length")) | .value.value) as $context_length
    | select(($architecture | type) == "string" and ($context_length | type) == "number")
    | [$architecture, $context_length] | @tsv
  ' || true
}

# 再開可能なダウンロード。SSL エラーなど一時的な失敗は続きから何度でも再試行する。
download_with_resume() {
  local url="$1" temporary="$2" label="$3"
  local rc error_output stalled=0 delay=2
  local size_before size_after
  local -a curl_args

  while :; do
    size_before="$(file_size "$temporary")"
    curl_args=(
      --fail --location --show-error --silent
      --retry 5 --retry-delay 2 --retry-connrefused
      --connect-timeout 30
      --output "$temporary"
    )
    if [[ -s "$temporary" ]]; then
      curl_args+=(--continue-at -)
    fi

    set +e
    error_output="$(curl "${curl_args[@]}" "$url" 2>&1)"
    rc=$?
    set -e

    if ((rc == 0)); then
      return 0
    fi

    case "$rc" in
      33|416)
        # サーバーが範囲リクエストを拒否した場合のみ最初からやり直す
        log "Resume not supported by server; restarting download: $label"
        rm -f -- "$temporary"
        stalled=0
        delay=2
        continue
        ;;
      6|7|16|18|28|35|52|55|56|92)
        : # 一時的なネットワーク/SSL エラーとして再開リトライする
        ;;
      *)
        log "error: failed to download $label (curl exit $rc): ${error_output:-unknown error}"
        return 1
        ;;
    esac

    size_after="$(file_size "$temporary")"
    if ((size_after > size_before)); then
      stalled=0
      delay=2
    else
      stalled=$((stalled + 1))
      if ((stalled >= DOWNLOAD_MAX_STALLED_ATTEMPTS)); then
        log "error: download made no progress after ${stalled} attempts: $label (curl exit $rc): ${error_output:-unknown error}"
        return 1
      fi
      ((delay < 30)) && delay=$((delay * 2))
    fi

    log "Download interrupted (curl exit $rc); resuming from $(file_size "$temporary") bytes in ${delay}s: $label"
    sleep "$delay"
  done
}

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
METADATA_CACHE_TMP_FILE="${METADATA_CACHE_FILE}.tmp"
rm -rf "$PRESET_TMP_DIR"
mkdir -p "$PRESET_TMP_DIR"
: > "$ALIAS_TMP_FILE"
: > "$METADATA_CACHE_TMP_FILE"

# 前回の GGUF 検証結果を読み込む（ファイルが変化していなければ再解析しない）
declare -A metadata_cache=()
if [[ -f "$METADATA_CACHE_FILE" ]]; then
  while IFS=$'\t' read -r cached_name cached_signature cached_architecture cached_context || [[ -n "${cached_name:-}" ]]; do
    [[ -n "${cached_name:-}" && -n "${cached_signature:-}" ]] || continue
    [[ -n "${cached_architecture:-}" && -n "${cached_context:-}" ]] || continue
    metadata_cache["$cached_name:$cached_signature"]="$cached_architecture"$'\t'"$cached_context"
  done < "$METADATA_CACHE_FILE"
fi

declare -A allowed_files=()
model_success_count=0
skipped_models=()

# モデル1件分の準備処理（不正なURL・ダウンロード失敗・GGUF読み取り失敗などは
# 致命的エラーとせず、そのモデルだけスキップして 1 を返す）
process_model_spec() {
  local model_spec="$1"
  local download_url="" filename repo destination alias_name
  local model_metadata cache_key temporary
  local model_architecture detected_context_size advertised_context_size

  # フルURL形式 (https://huggingface.co/.../resolve/main/....gguf) と
  # 従来の短縮形式 (owner/repo/filename.gguf) の両方に対応する
  if [[ "$model_spec" == https://huggingface.co/* ]]; then
    filename=$(basename "${model_spec%%\?*}")
    # クエリパラメータ（例: ?download=true）が付いていなければ付与する
    if [[ "$model_spec" == *\?* ]]; then
      download_url="$model_spec"
    else
      download_url="${model_spec}?download=true"
    fi
    # HF_ENDPOINT が変更されている場合（ミラー等）はホスト部分を差し替える
    if [[ "$HF_ENDPOINT" != "https://huggingface.co" ]]; then
      download_url="${HF_ENDPOINT}${download_url#https://huggingface.co}"
    fi
    # ログ表示用に owner/repo 部分を抽出
    model_spec="${model_spec#https://huggingface.co/}"
    model_spec="${model_spec%%/resolve/*}"
  else
    # 従来形式: owner/repo/filename.gguf
    filename="$(basename "$model_spec")"
    repo="${model_spec%/*}"
    download_url="$HF_ENDPOINT/$repo/resolve/main/$filename?download=true"
  fi

  if [[ "$model_spec" != */* ]]; then
    log "error: invalid model spec: $model_spec (owner/repo が抜けています。Hugging Faceの「Copy download link」で取得したURL https://huggingface.co/{owner}/{repo}/resolve/main/{filename} を確認してください) -- skipping this model"
    return 1
  fi
  if [[ "$filename" != *.gguf ]]; then
    log "error: model must be a .gguf file: $filename -- skipping this model"
    return 1
  fi
  destination="$MODEL_DIR/$filename"
  alias_name="${filename%.gguf}"

  model_metadata=""
  if [[ -f "$destination" ]]; then
    cache_key="$filename:$(file_signature "$destination")"
    if [[ -n "${metadata_cache[$cache_key]+x}" ]]; then
      model_metadata="${metadata_cache[$cache_key]}"
      log "Using cached model: $filename"
    else
      model_metadata="$(read_gguf_metadata "$destination")"
      if [[ "$model_metadata" == *$'\t'* ]]; then
        log "Verified model: $filename"
      else
        log "Cached model is invalid; downloading again: $filename"
        model_metadata=""
        rm -f -- "$destination"
      fi
    fi
  fi

  if [[ ! -f "$destination" ]]; then
    log "Downloading $model_spec"
    temporary="$destination.part"
    # 途中まで取得済みのファイルは削除せず、続きから再開する
    if ! download_with_resume "$download_url" "$temporary" "$model_spec"; then
      log "error: skipping model due to download failure: $model_spec"
      return 1
    fi
    mv "$temporary" "$destination"
  fi

  if [[ -z "$model_metadata" ]]; then
    model_metadata="$(read_gguf_metadata "$destination")"
    if [[ "$model_metadata" != *$'\t'* ]]; then
      log "error: could not read GGUF metadata after download: $filename -- skipping this model and removing invalid file"
      rm -f -- "$destination"
      return 1
    fi
  fi
  model_architecture="${model_metadata%%$'\t'*}"
  detected_context_size="${model_metadata#*$'\t'}"
  advertised_context_size="${CONTEXT_SIZE:-$detected_context_size}"
  if [[ -n "$MAX_CONTEXT_SIZE" ]] && ((advertised_context_size > MAX_CONTEXT_SIZE)); then
    advertised_context_size="$MAX_CONTEXT_SIZE"
  fi

  printf '%s\t%s\t%s\t%s\n' \
    "$filename" "$(file_signature "$destination")" "$model_architecture" "$detected_context_size" \
    >> "$METADATA_CACHE_TMP_FILE"
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

  allowed_files["$filename"]=1
  return 0
}

for model_spec_raw in "${MODEL_NAMES[@]}"; do
  # Remove leading/trailing whitespace
  model_spec_trimmed="$(echo "$model_spec_raw" | xargs)"
  [[ -n "$model_spec_trimmed" ]] || continue

  if process_model_spec "$model_spec_trimmed"; then
    model_success_count=$((model_success_count + 1))
  else
    skipped_models+=("$model_spec_trimmed")
    log "Skipped model: $model_spec_trimmed"
  fi
done

if ((${#skipped_models[@]} > 0)); then
  log "Skipped ${#skipped_models[@]} model(s) due to errors: ${skipped_models[*]}"
fi

((model_success_count > 0)) || die "no models were successfully prepared (all ${#MODEL_NAMES[@]} entries failed or were skipped)"

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
mv "$METADATA_CACHE_TMP_FILE" "$METADATA_CACHE_FILE"

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

if ((${#skipped_models[@]} > 0)); then
  log "Model sync completed with ${#skipped_models[@]} model(s) skipped. Active models in list: ${!allowed_files[*]}"
else
  log "Model sync completed successfully. Active models in list: ${!allowed_files[*]}"
fi
