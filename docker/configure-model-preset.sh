#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-1800}"
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
MIN_CONTEXT_SIZE="${MIN_CONTEXT_SIZE:-2048}"
CONTEXT_SIZE_STEP="${CONTEXT_SIZE_STEP:-1024}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
MODELS_MAX="${MODELS_MAX:-1}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"
TENSOR_SPLIT_MODE="${TENSOR_SPLIT_MODE:-auto}"
VRAM_RESERVE_MIB="${VRAM_RESERVE_MIB:-4096}"
MOE_CPU_OFFLOAD="${MOE_CPU_OFFLOAD:-auto}"
MOE_ACTIVE_RATIO_THRESHOLD="${MOE_ACTIVE_RATIO_THRESHOLD:-0.125}"
MOE_RAM_RESERVE_MIB="${MOE_RAM_RESERVE_MIB:-8192}"
PRESET_FILE="${PRESET_FILE:-$MODEL_DIR/.models-preset.ini}"
PRESET_SECTION_DIR="${PRESET_SECTION_DIR:-$MODEL_DIR/.models-preset.d}"
MODEL_ALIAS_FILE="${MODEL_ALIAS_FILE:-$MODEL_DIR/.model-aliases.tsv}"

log() { printf '[llama-wrapper] %s\n' "$*" >&2; }
die() { printf '[llama-wrapper] error: %s\n' "$*" >&2; exit 1; }

requested_model="${1:-}"
[[ -n "$requested_model" ]] || die "model name is required"
[[ -f "$MODEL_ALIAS_FILE" ]] || die "model alias file not found: $MODEL_ALIAS_FILE"

resolve_model() {
  awk -F '\t' -v requested="$requested_model" '
    $1 == requested || $2 == requested {
      print $1 "\t" $2
      found = 1
      exit
    }
    END { if (!found) exit 1 }
  ' "$MODEL_ALIAS_FILE"
}

resolved="$(resolve_model)" || die "unknown model: $requested_model"
alias_name="${resolved%%$'\t'*}"
filename="${resolved#*$'\t'}"
model_file="$MODEL_DIR/$filename"
[[ -f "$model_file" ]] || die "model file not found: $model_file"

lock_dir="$MODEL_DIR/.preset-configure.lock"
lock_pid_file="$lock_dir/pid"
lock_wait_deadline=$(( SECONDS + 600 ))
while ! mkdir "$lock_dir" 2>/dev/null; do
  # ロックを保持していたプロセスが異常終了した場合、ロックディレクトリが残り続けて
  # 以降のすべてのプリセット設定が永久にブロックされてしまう。
  # PID が生存していなければ stale lock とみなして解放する。
  lock_pid="$(cat "$lock_pid_file" 2>/dev/null || true)"
  if [[ -z "$lock_pid" ]] || ! kill -0 "$lock_pid" 2>/dev/null; then
    log "removing stale preset lock (pid=${lock_pid:-unknown})"
    rm -rf "$lock_dir"
    continue
  fi
  if (( SECONDS > lock_wait_deadline )); then
    die "timed out waiting for preset lock held by pid $lock_pid"
  fi
  sleep 0.1
done
printf '%s\n' "$$" > "$lock_pid_file"
cleanup_lock() {
  rm -rf "$lock_dir"
}
trap cleanup_lock EXIT INT TERM

detect_context_length() {
  local model_file="$1"
  gguf-dump --no-tensors "$model_file" 2>/dev/null \
    | awk -F'= *' '/\.context_length[[:space:]]*=/ { print $2; exit }' \
    | tr -d '[:space:]'
}

detect_kv_cache_params() {
  local model_file="$1"
  gguf-dump --no-tensors --json --json-array "$model_file" 2>/dev/null | jq -r '
    def scalar_or_max:
      if type == "array" then (map(select(type == "number")) | max) else . end;
    (.metadata | to_entries) as $entries
    | ($entries[] | select(.key | endswith(".block_count")) | .value.value | scalar_or_max) as $block_count
    | ($entries[] | select(.key | endswith(".attention.head_count_kv")) | .value.value) as $head_count_kv
    | ($entries[] | select(.key | endswith(".attention.key_length")) | .value.value | scalar_or_max) as $key_length
    | ($entries[] | select(.key | endswith(".attention.value_length")) | .value.value | scalar_or_max) as $value_length
    | (if ($head_count_kv | type) == "array"
       then ($head_count_kv | map(select(type == "number" and . > 0)) | add // 0)
       else ($block_count * $head_count_kv)
       end) as $total_kv_heads
    | "1 \($total_kv_heads) \($key_length) \($value_length)"
  ' 2>/dev/null | head -n1
}

kv_cache_type_bytes_per_element() {
  case "$1" in
    f32) echo 4 ;;
    f16|bf16) echo 2 ;;
    q8_0) echo 1.0625 ;;
    q5_0|q5_1) echo 0.6875 ;;
    q4_0|q4_1|iq4_nl) echo 0.5625 ;;
    *) echo 2 ;;
  esac
}

# "<kv-cache-type> <ctx-size>" を返す。モデル本体すら VRAM に収まらない場合は
# 空文字を返し、呼び出し側で llama-server の --fit に委ねる。
select_kv_cache_type() {
  local model_file="$1" ctx_size="$2" free_bytes="$3" model_bytes_override="${4:-}"
  local params block_count head_count_kv key_length value_length
  params="$(detect_kv_cache_params "$model_file")"
  if [[ -z "$params" ]]; then
    log "Could not detect attention params for $(basename "$model_file"); skipping KV cache type auto-selection"
    echo ""
    return
  fi
  read -r block_count head_count_kv key_length value_length <<< "$params"

  local model_bytes budget_bytes
  model_bytes="${model_bytes_override:-$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)}"
  budget_bytes="$(awk -v f="$free_bytes" -v m="$model_bytes" 'BEGIN { b = (f - m) * 0.9; if (b < 0) b = 0; printf "%.0f", b }')"
  if ((budget_bytes <= 0)); then
    log "$(basename "$model_file"): model weights alone exceed free VRAM; falling back to --fit"
    echo ""
    return
  fi

  local candidate bytes_per_elem needed_bytes
  for candidate in f16 q8_0 q4_0; do
    bytes_per_elem="$(kv_cache_type_bytes_per_element "$candidate")"
    needed_bytes="$(awk -v bc="$block_count" -v hkv="$head_count_kv" -v kl="$key_length" -v vl="$value_length" \
      -v ctx="$ctx_size" -v bpe="$bytes_per_elem" \
      'BEGIN { printf "%.0f", bc * hkv * (kl + vl) * ctx * bpe }')"
    if (( $(awk -v n="$needed_bytes" -v b="$budget_bytes" 'BEGIN { print (n <= b) ? 1 : 0 }') )); then
      log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB (budget $((budget_bytes / 1024 / 1024)) MiB) -> selected"
      printf '%s %s\n' "$candidate" "$ctx_size"
      return
    fi
    log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB exceeds budget $((budget_bytes / 1024 / 1024)) MiB"
  done

  # 最も軽い q4_0 でも収まらない場合は、--fit に切り替えるのではなく
  # 予算に収まるところまで ctx-size を切り詰める。
  local fitted_ctx
  bytes_per_elem="$(kv_cache_type_bytes_per_element q4_0)"
  fitted_ctx="$(awk -v bc="$block_count" -v hkv="$head_count_kv" -v kl="$key_length" -v vl="$value_length" \
    -v b="$budget_bytes" -v bpe="$bytes_per_elem" -v step="$CONTEXT_SIZE_STEP" \
    'BEGIN {
       per_token = bc * hkv * (kl + vl) * bpe
       if (per_token <= 0) { print 0; exit }
       ctx = int(b / per_token)
       ctx = int(ctx / step) * step
       print ctx
     }')"
  if [[ "$fitted_ctx" =~ ^[0-9]+$ ]] && ((fitted_ctx >= MIN_CONTEXT_SIZE)); then
    log "$(basename "$model_file"): shrinking ctx-size from $ctx_size to $fitted_ctx to fit q4_0 KV cache in VRAM"
    printf '%s %s\n' q4_0 "$fitted_ctx"
    return
  fi
  log "$(basename "$model_file"): cannot fit q4_0 KV cache even at MIN_CONTEXT_SIZE=$MIN_CONTEXT_SIZE; falling back to --fit"
  echo ""
}

detect_total_free_vram_bytes() {
  local free_mib_list total_mib=0
  free_mib_list="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null)"
  if [[ -z "$free_mib_list" ]]; then
    echo 0
    return
  fi
  while read -r mib; do
    [[ "$mib" =~ ^[0-9]+$ ]] || continue
    total_mib=$((total_mib + mib))
  done <<< "$free_mib_list"
  echo $((total_mib * 1024 * 1024))
}

detect_per_gpu_free_vram_mib() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local idx i
    IFS=',' read -r -a idx <<< "$CUDA_VISIBLE_DEVICES"
    for i in "${idx[@]}"; do
      nvidia-smi --id="$i" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null
    done
  else
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null
  fi
}

gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible="${CUDA_VISIBLE_DEVICES//,/ }"
  gpu_count="$(awk '{print NF}' <<< "$visible")"
fi
((gpu_count > 0)) || gpu_count=1

BENCH_TG_CACHE="$MODEL_DIR/.gpu-tg-speed.tsv"
detect_per_gpu_tg_speed() {
  local benchmark_model="$1"
  if ((gpu_count < 2)); then
    echo ""
    return
  fi
  if [[ ! -f "$BENCH_TG_CACHE" ]]; then
    log "Benchmarking per-GPU generation speed for tensor-split calculation (one-time, on first model switch)"
    : > "$BENCH_TG_CACHE"
    local i speed
    for ((i = 0; i < gpu_count; i++)); do
      speed="$(CUDA_VISIBLE_DEVICES="$i" llama-bench -m "$benchmark_model" -ngl 99 -p 0 -n 32 \
        --output json 2>/dev/null | /usr/local/bin/model-preset-utils.py benchmark-speed 2>/dev/null)"
      [[ "$speed" =~ ^[0-9.]+$ ]] || speed=0
      printf '%s\n' "$speed" >> "$BENCH_TG_CACHE"
    done
  fi
  cat "$BENCH_TG_CACHE"
}

detect_layer_bytes() {
  local model_file="$1" mode="${2:-full}"
  local args=(layer-bytes)
  if [[ "$mode" == "exclude" ]]; then
    args+=(--exclude-moe)
  elif [[ "$mode" == "split" ]]; then
    args+=(--split-moe)
  fi
  gguf-dump --json --json-array "$model_file" 2>/dev/null | /usr/local/bin/model-preset-utils.py "${args[@]}"
}

detect_moe_profile() {
  local model_file="$1"
  gguf-dump --json --json-array "$model_file" 2>/dev/null | /usr/local/bin/model-preset-utils.py moe-profile
}

detect_available_ram_bytes() {
  awk '/^MemAvailable:/ { printf "%.0f", $2 * 1024; exit }' /proc/meminfo 2>/dev/null
}

calculate_tensor_split() {
  local layer_bytes_file="$1" free_mib_list="$2" speed_list="$3" reserve_mib="$4" moe_auto="${5:-0}" kv_bytes_per_head="${6:-0}"
  local args=(tensor-split "$layer_bytes_file" "$free_mib_list" "$speed_list" "$reserve_mib")
  if [[ "$moe_auto" == 1 ]]; then
    args+=(--moe-auto)
  fi
  args+=(--kv-bytes-per-head "$kv_bytes_per_head")
  /usr/local/bin/model-preset-utils.py "${args[@]}"
}

estimate_kv_cache_mib() {
  local params="$1" cache_type="$2" ctx_size="$3"
  if [[ -z "$params" || -z "$cache_type" ]]; then
    echo 0
    return
  fi

  local bc hkv kl vl bpe
  read -r bc hkv kl vl <<< "$params"
  bpe="$(kv_cache_type_bytes_per_element "$cache_type")"
  awk -v bc="$bc" -v hkv="$hkv" -v kl="$kl" -v vl="$vl" -v ctx="$ctx_size" -v bpe="$bpe" \
    'BEGIN { printf "%.0f", (bc * hkv * (kl + vl) * ctx * bpe) / 1024 / 1024 }'
}

render_preset() {
  local tmp
  tmp="$(mktemp "$PRESET_FILE.tmp.XXXXXX")"
  {
    printf '[*]\n'
    printf 'fit = on\n\n'
    for section_file in "$PRESET_SECTION_DIR"/*.ini; do
      [[ -f "$section_file" ]] || continue
      cat "$section_file"
    done
  } > "$tmp"
  mv "$tmp" "$PRESET_FILE"
}

model_ctx_size="$CONTEXT_SIZE"
if [[ -z "$model_ctx_size" ]]; then
  detected="$(detect_context_length "$model_file")"
  if [[ "$detected" =~ ^[0-9]+$ ]]; then
    model_ctx_size="$detected"
    log "Detected context_length for $filename: $model_ctx_size"
  else
    model_ctx_size=4096
    log "Could not detect context_length for $filename, falling back to $model_ctx_size"
  fi
fi
if [[ -n "$MAX_CONTEXT_SIZE" ]] && ((model_ctx_size > MAX_CONTEXT_SIZE)); then
  log "Capping ctx-size for $filename from $model_ctx_size to MAX_CONTEXT_SIZE=$MAX_CONTEXT_SIZE"
  model_ctx_size="$MAX_CONTEXT_SIZE"
fi

model_n_cpu_moe=""
model_moe_auto=0
model_gpu_bytes="$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)"
moe_profile="$(detect_moe_profile "$model_file" || true)"
if [[ -n "$moe_profile" ]]; then
  read -r block_count expert_count expert_used_count expert_layer_count expert_bytes <<< "$moe_profile"
  should_offload_moe=0
  if [[ "$MOE_CPU_OFFLOAD" == "all" ]] && ((expert_bytes > 0)); then
    should_offload_moe=1
  elif [[ "$MOE_CPU_OFFLOAD" == "auto" ]] && ((expert_count > 0 && expert_used_count > 0 && expert_bytes > 0)); then
    should_offload_moe="$(awk -v used="$expert_used_count" -v total="$expert_count" -v threshold="$MOE_ACTIVE_RATIO_THRESHOLD" \
      'BEGIN { print ((used / total) <= threshold) ? 1 : 0 }')"
  fi

  if ((should_offload_moe == 1)); then
    available_ram_bytes="$(detect_available_ram_bytes)"
    required_ram_bytes=$((expert_bytes + MOE_RAM_RESERVE_MIB * 1024 * 1024))
    if [[ "$available_ram_bytes" =~ ^[0-9]+$ ]] && ((available_ram_bytes >= required_ram_bytes)); then
      model_moe_auto=1
      model_gpu_bytes=$((model_gpu_bytes - expert_bytes))
      ((model_gpu_bytes < 0)) && model_gpu_bytes=0
      active_ratio="$(awk -v used="$expert_used_count" -v total="$expert_count" 'BEGIN { printf "%.1f", 100 * used / total }')"
      log "$filename: MoE uses $expert_used_count/$expert_count experts ($active_ratio%); prioritizing KV cache, then keeping as many expert layers in VRAM as fit"
      log "$filename: up to $((expert_bytes / 1024 / 1024)) MiB of expert weights can be moved to CPU for the KV cache"
    else
      log "$filename: skipping MoE CPU offload because available host RAM is below expert weights plus MOE_RAM_RESERVE_MIB=$MOE_RAM_RESERVE_MIB"
    fi
  fi
fi

model_kv_cache_type="$KV_CACHE_TYPE"
model_fit_fallback=0
total_free_vram_bytes="$(detect_total_free_vram_bytes)"
if [[ -z "$model_kv_cache_type" ]]; then
  if ((total_free_vram_bytes > 0)); then
    kv_selection="$(select_kv_cache_type "$model_file" "$model_ctx_size" "$total_free_vram_bytes" "$model_gpu_bytes")"
    if [[ -n "$kv_selection" ]]; then
      read -r model_kv_cache_type fitted_ctx_size <<< "$kv_selection"
      if [[ "$fitted_ctx_size" =~ ^[0-9]+$ ]] && ((fitted_ctx_size != model_ctx_size)); then
        log "Adjusted ctx-size for $filename: $model_ctx_size -> $fitted_ctx_size"
        model_ctx_size="$fitted_ctx_size"
      fi
      log "Auto-selected KV cache type for $filename: $model_kv_cache_type"
    else
      model_fit_fallback=1
    fi
  else
    log "Could not detect free VRAM; leaving KV cache type at llama-server default (f16) for $filename"
  fi
fi

model_tensor_split=""
model_n_gpu_layers_fixed=""
if [[ "$model_fit_fallback" == 1 ]]; then
  log "Model weights do not fit in free VRAM for $filename; relying on llama-server's --fit auto-adjustment"
elif [[ "$TENSOR_SPLIT_MODE" == "auto" && "$gpu_count" -ge 2 ]]; then
  gpu_tg_speeds="$(detect_per_gpu_tg_speed "$model_file")"
  if [[ -n "$gpu_tg_speeds" ]]; then
    log "Per-GPU generation speed (tok/s): $(tr '\n' ' ' <<< "$gpu_tg_speeds")"
    layer_bytes_file="$(mktemp)"
    layer_bytes_mode="split"
    if detect_layer_bytes "$model_file" "$layer_bytes_mode" > "$layer_bytes_file" && [[ -s "$layer_bytes_file" ]]; then
      free_mib_list="$(detect_per_gpu_free_vram_mib)"
      params="$(detect_kv_cache_params "$model_file")"
      kv_total_mib="$(estimate_kv_cache_mib "$params" "$model_kv_cache_type" "$model_ctx_size")"
      read -r _ total_kv_heads kl vl <<< "$params"
      bpe="$(kv_cache_type_bytes_per_element "$model_kv_cache_type")"
      kv_bytes_per_head="$(awk -v kl="$kl" -v vl="$vl" -v ctx="$model_ctx_size" -v bpe="$bpe" \
        'BEGIN { printf "%.0f", (kl + vl) * ctx * bpe }')"
      reserve_mib="$VRAM_RESERVE_MIB"
      result="$(calculate_tensor_split "$layer_bytes_file" "$free_mib_list" "$gpu_tg_speeds" "$reserve_mib" "$model_moe_auto" "$kv_bytes_per_head" || true)"
      if [[ -z "$result" ]]; then
        # KV 選択時の概算では収まっても、GPU ごとの重み配置と固定予約を加えると
        # tensor-split が成立しないことがある。重みだけが収まるなら --fit へ逃げず、
        # 手動配置が成立する最大のコンテキスト長を探索する。
        model_only_result="$(calculate_tensor_split "$layer_bytes_file" "$free_mib_list" "$gpu_tg_speeds" "$VRAM_RESERVE_MIB" "$model_moe_auto" 0 || true)"
        if [[ -n "$model_only_result" ]]; then
          min_step=$(( (MIN_CONTEXT_SIZE + CONTEXT_SIZE_STEP - 1) / CONTEXT_SIZE_STEP ))
          max_step=$(( model_ctx_size / CONTEXT_SIZE_STEP ))
          best_ctx=0
          best_result=""

          if ((min_step <= max_step)); then
            min_ctx=$((min_step * CONTEXT_SIZE_STEP))
            min_kv_mib="$(estimate_kv_cache_mib "$params" "$model_kv_cache_type" "$min_ctx")"
            min_kv_bytes_per_head="$(awk -v kl="$kl" -v vl="$vl" -v ctx="$min_ctx" -v bpe="$bpe" \
              'BEGIN { printf "%.0f", (kl + vl) * ctx * bpe }')"
            min_result="$(calculate_tensor_split "$layer_bytes_file" "$free_mib_list" "$gpu_tg_speeds" "$VRAM_RESERVE_MIB" "$model_moe_auto" "$min_kv_bytes_per_head" || true)"
            if [[ -n "$min_result" ]]; then
              low_step="$min_step"
              high_step="$max_step"
              while ((low_step <= high_step)); do
                mid_step=$(( (low_step + high_step) / 2 ))
                trial_ctx=$((mid_step * CONTEXT_SIZE_STEP))
                trial_kv_mib="$(estimate_kv_cache_mib "$params" "$model_kv_cache_type" "$trial_ctx")"
                trial_kv_bytes_per_head="$(awk -v kl="$kl" -v vl="$vl" -v ctx="$trial_ctx" -v bpe="$bpe" \
                  'BEGIN { printf "%.0f", (kl + vl) * ctx * bpe }')"
                trial_result="$(calculate_tensor_split "$layer_bytes_file" "$free_mib_list" "$gpu_tg_speeds" "$VRAM_RESERVE_MIB" "$model_moe_auto" "$trial_kv_bytes_per_head" || true)"
                if [[ -n "$trial_result" ]]; then
                  best_ctx="$trial_ctx"
                  best_result="$trial_result"
                  low_step=$((mid_step + 1))
                else
                  high_step=$((mid_step - 1))
                fi
              done
            fi
          fi

          if [[ -n "$best_result" ]]; then
            log "Shrinking ctx-size for $filename from $model_ctx_size to $best_ctx to enable manual tensor-split"
            model_ctx_size="$best_ctx"
            result="$best_result"
          else
            log "Manual tensor-split cannot fit at MIN_CONTEXT_SIZE=$MIN_CONTEXT_SIZE; falling back to --fit"
          fi
        else
          log "Model weights cannot fit with the per-GPU reserve; falling back to --fit"
        fi
      fi
      if [[ -n "$result" ]]; then
        read -r model_n_gpu_layers_fixed model_tensor_split model_n_cpu_moe cpu_moe_bytes <<< "$result"
        if ((model_moe_auto == 1)); then
          model_n_cpu_moe="${model_n_cpu_moe:-0}"
          cpu_moe_bytes="${cpu_moe_bytes:-0}"
          gpu_moe_bytes=$((expert_bytes - cpu_moe_bytes))
          log "$filename: keeping $((gpu_moe_bytes / 1024 / 1024)) MiB of expert weights in VRAM; offloading $((cpu_moe_bytes / 1024 / 1024)) MiB from the first $model_n_cpu_moe layers"
        fi
        log "Calculated tensor-split for $filename: n-gpu-layers=$model_n_gpu_layers_fixed tensor-split=$model_tensor_split"
      else
        log "Could not fit a manual tensor-split for $filename within free VRAM"
      fi
    else
      log "Could not determine per-layer tensor sizes for $filename; falling back to --fit"
    fi
    rm -f "$layer_bytes_file"
  else
    log "Could not benchmark per-GPU speed; tensor-split will fall back to --fit"
  fi
else
  log "TENSOR_SPLIT_MODE=$TENSOR_SPLIT_MODE or single GPU; relying on llama-server's --fit auto-adjustment"
fi

section_file="$PRESET_SECTION_DIR/$alias_name.ini"
tmp_section="$(mktemp "$section_file.tmp.XXXXXX")"
{
  printf '[%s]\n' "$alias_name"
  printf 'model = %s\n' "$model_file"
  printf 'ctx-size = %s\n' "$model_ctx_size"
  printf 'sleep-idle-seconds = %s\n' "$MODEL_IDLE_SECONDS"
  if [[ -n "$model_tensor_split" ]]; then
    printf 'fit = off\n'
    printf 'n-gpu-layers = %s\n' "$model_n_gpu_layers_fixed"
    printf 'tensor-split = %s\n' "$model_tensor_split"
  else
    printf 'n-gpu-layers = %s\n' "$N_GPU_LAYERS"
  fi
  if [[ -n "$model_n_cpu_moe" ]] && ((model_n_cpu_moe > 0)); then
    printf 'n-cpu-moe = %s\n' "$model_n_cpu_moe"
  fi
  kv_type="${model_kv_cache_type:-${KV_CACHE_TYPE:-f16}}"
  printf 'cache-type-k = %s\n' "$kv_type"
  printf 'cache-type-v = %s\n' "$kv_type"
  printf '\n'
} > "$tmp_section"
mv "$tmp_section" "$section_file"
render_preset
log "Updated lazy preset for $alias_name"
