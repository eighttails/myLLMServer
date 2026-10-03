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
MAX_PARALLEL_SLOTS="${MAX_PARALLEL_SLOTS:-4}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"
TENSOR_SPLIT_MODE="${TENSOR_SPLIT_MODE:-auto}"
SPLIT_MODE="${SPLIT_MODE:-layer}"
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

# "<全長Attention層の1トークンあたりKV要素数> <SWA層の1トークンあたりKV要素数> <sliding_window>" を返す。
# SSM/Attention ハイブリッドの SSM 層は除外し、SWA 層はウィンドウ分しか確保されないため別枠で返す。
detect_kv_cache_params() {
  local model_file="$1"
  gguf-dump --no-tensors --json --json-array "$model_file" 2>/dev/null \
    | /usr/local/bin/model-preset-utils.py kv-profile 2>/dev/null
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

# SWA 層で確保される KV のトークン数。llama-server は SWA 層にウィンドウ×シーケンス数+ubatch 分
# (ただし全体の ctx-size が上限) しか KV を確保しない。
swa_cache_tokens() {
  local params="$1" ctx="$2" slots="$3"
  local window tokens
  read -r _ _ window _ <<< "$params"
  tokens=$(( window * slots + ${UBATCH_SIZE:-256} ))
  ((tokens > ctx * slots)) && tokens=$((ctx * slots))
  echo "$tokens"
}

# 指定した KV キャッシュ型・1スロットあたりのコンテキスト長・スロット数での KV キャッシュ量 (bytes)。
kv_cache_bytes() {
  local params="$1" cache_type="$2" ctx="$3" slots="${4:-1}"
  local full swa bpe
  read -r full swa _ <<< "$params"
  bpe="$(kv_cache_type_bytes_per_element "$cache_type")"
  awk -v full="$full" -v swa="$swa" -v n=$((ctx * slots)) -v sn="$(swa_cache_tokens "$params" "$ctx" "$slots")" -v bpe="$bpe" \
    'BEGIN { printf "%.0f", (full * n + swa * sn) * bpe }'
}

# GPU 1枚あたりの Attention 作業領域 (bytes) のうち、VRAM_RESERVE_MIB の半分を超える分。
# 量子化 KV では Flash Attention が層ごとに K/V を f16 へ展開するため「1層の最大要素数 × トークン数 × 2」、
# さらに KQ マスク (ubatch × トークン数 × 2) が compute buffer に載る。総コンテキストに比例して GB 単位に
# なり得るため、VRAM_RESERVE_MIB (compute buffer 込みの余白) の半分で吸収しきれない分を KV と合わせて見積もる。
attn_scratch_bytes() {
  local params="$1" cache_type="$2" ctx="$3" slots="${4:-1}"
  local max_full max_swa dequant=1
  read -r _ _ _ max_full max_swa <<< "$params"
  [[ "$cache_type" == f16 || "$cache_type" == bf16 || "$cache_type" == f32 ]] && dequant=0
  awk -v mf="${max_full:-0}" -v ms="${max_swa:-0}" -v n=$((ctx * slots)) -v sn="$(swa_cache_tokens "$params" "$ctx" "$slots")" \
    -v ub="${UBATCH_SIZE:-256}" -v dq="$dequant" \
    -v allowance=$((VRAM_RESERVE_MIB * 1024 * 1024 / 2)) \
    'BEGIN { d = mf * n; if (ms * sn > d) d = ms * sn; x = (dq * d + ub * n) * 2 - allowance; printf "%.0f", (x > 0 ? x : 0) }'
}

# KV キャッシュ本体と全GPU分の Attention 作業領域を合わせた VRAM 所要量 (bytes)。
kv_vram_bytes() {
  local gpus=$((gpu_count > 0 ? gpu_count : 1))
  echo $(( $(kv_cache_bytes "$@") + gpus * $(attn_scratch_bytes "$@") ))
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

# GPU ごとに VRAM_RESERVE_MIB (compute buffer 等の余白) を差し引いた空き VRAM の合計 (bytes)。
detect_usable_vram_bytes() {
  local total_mib=0 mib
  while read -r mib; do
    [[ "$mib" =~ ^[0-9]+$ ]] || continue
    ((mib > VRAM_RESERVE_MIB)) && total_mib=$((total_mib + mib - VRAM_RESERVE_MIB))
  done <<< "$(detect_per_gpu_free_vram_mib)"
  echo $((total_mib * 1024 * 1024))
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
# 重みが1つのGPUのVRAMに収まらない大規模モデルは、-ngl 99 で単一GPUだけ
# を見えるようにしてベンチマークするとモデルがロードできず常に 0 tok/s を
# 返してしまう。Tensor Split は各GPU間の「相対速度」が大事なので、単一GPUに
# 確実に収まる小さなモデルで相対速度を測定し、キャッシュを再利用する。
BENCH_BENCH_MODEL=""
find_benchmark_model() {
  [[ -n "$BENCH_BENCH_MODEL" ]] && return 0
  for candidate in \
    "$MODEL_DIR/Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
    "$MODEL_DIR/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf" \
    "$MODEL_DIR/gpt-oss-20b-Q4_K_M.gguf" \
    "$MODEL_DIR/Ornith-1.5-35B-Q4_K_M.gguf" ; do
    [[ -f "$candidate" ]] || continue
    speed="$(CUDA_VISIBLE_DEVICES=0 /app/llama-bench -m "$candidate" -ngl 99 -p 0 -n 32 \
        --output json 2>/dev/null | /usr/local/bin/model-preset-utils.py benchmark-speed 2>/dev/null)"
    if [[ "$speed" =~ ^[0-9.]+$ ]] && awk -v s="$speed" 'BEGIN { exit !(s > 0) }'; then
      BENCH_BENCH_MODEL="$candidate"
      log "Using $BENCH_BENCH_MODEL as benchmark model for per-GPU speed (fits on a single GPU, $speed tok/s)"
      return 0
    fi
  done
  BENCH_BENCH_MODEL=""
  log "No single-GPU-fitting model found for speed benchmark; will fall back to --fit"
  return 1
}
detect_per_gpu_tg_speed() {
  local benchmark_model="$1"
  if ((gpu_count < 2)); then
    echo ""
    return
  fi
  # 前回の走測が単一GPUに収まらないモデルによる失敗(0)でキャッシュが汚染
  # されている場合があるので、有効な速度が出ていれば再測定の必要はない。
  if [[ -f "$BENCH_TG_CACHE" ]] && grep -Eq '^[0-9.]+' "$BENCH_TG_CACHE" \
      && [[ "$(grep -cE '^[0-9.]+' "$BENCH_TG_CACHE")" -ge "$gpu_count" ]]; then
    local ok_speed
    ok_speed="$(grep -E '^[0-9.]+' "$BENCH_TG_CACHE" | head -n 1)"
    if [[ -n "$ok_speed" && "$ok_speed" != "0" && "$ok_speed" != "0.0" ]]; then
      cat "$BENCH_TG_CACHE"
      return
    fi
    log "Previous per-GPU speed cache is invalid; re-benchmarking"
    rm -f "$BENCH_TG_CACHE"
  fi
  if ! find_benchmark_model; then
    return 1
  fi
  log "Benchmarking per-GPU generation speed for tensor-split calculation (one-time, on first model switch)"
  : > "$BENCH_TG_CACHE"
  local i speed
  for ((i = 0; i < gpu_count; i++)); do
    speed="$(CUDA_VISIBLE_DEVICES="$i" /app/llama-bench -m "$BENCH_BENCH_MODEL" -ngl 99 -p 0 -n 32 \
      --output json 2>/dev/null | /usr/local/bin/model-preset-utils.py benchmark-speed 2>/dev/null)"
    [[ "$speed" =~ ^[0-9.]+$ ]] || speed=0
    printf '%s\n' "$speed" >> "$BENCH_TG_CACHE"
  done
  cat "$BENCH_TG_CACHE"
}

detect_layer_bytes() {
  local model_file="$1" mode="${2:-full}"
  local args=(layer-bytes)
  if [[ "$mode" == "exclude" ]]; then
    args+=(--exclude-moe)
  elif [[ "$mode" == "split" ]]; then
    args+=(--split-moe)
  elif [[ "$mode" == "split-ffn" ]]; then
    args+=(--split-moe --split-ffn)
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
  local layer_bytes_file="$1" free_mib_list="$2" speed_list="$3" reserve_mib="$4" moe_auto="${5:-0}"
  local kv_tokens="${6:-0}" swa_tokens="${7:-0}" kv_bpe="${8:-0}" scratch_bytes="${9:-0}"
  local args=(tensor-split "$layer_bytes_file" "$free_mib_list" "$speed_list" "$reserve_mib")
  if [[ "$moe_auto" == 1 ]]; then
    args+=(--moe-auto)
  fi
  args+=(--kv-tokens "$kv_tokens" --swa-tokens "$swa_tokens" --kv-bytes-per-element "$kv_bpe" --scratch-bytes "$scratch_bytes")
  /usr/local/bin/model-preset-utils.py "${args[@]}"
}

render_preset() {
  local tmp
  tmp="$(mktemp "$PRESET_FILE.tmp.XXXXXX")"
  {
    printf '[*]\n'
    printf 'fit = on\n'
    printf 'split-mode = %s\n\n' "$SPLIT_MODE"
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
      active_ratio="$(awk -v used="$expert_used_count" -v total="$expert_count" 'BEGIN { printf "%.1f", 100 * used / total }')"
      log "$filename: MoE uses $expert_used_count/$expert_count experts ($active_ratio%); prioritizing KV cache, then keeping as many expert layers in VRAM as fit"
      log "$filename: up to $((expert_bytes / 1024 / 1024)) MiB of expert weights can be moved to CPU for the KV cache"
    else
      log "$filename: skipping MoE CPU offload because available host RAM is below expert weights plus MOE_RAM_RESERVE_MIB=$MOE_RAM_RESERVE_MIB"
    fi
  fi
fi

# KV キャッシュ型・並列スロット数・重みの配置を次の優先順位で決める。
#   1. 基本の KV キャッシュ型 (既定 q4_0、KV_CACHE_TYPE 指定時はその型) で、コンテキスト全量の
#      KV キャッシュを VRAM に載せる。載らない場合のみ ctx-size を切り詰める
#   2. 残りの VRAM にモデルの重みを載せる。載りきらない分は FFN / Expert の重みを CPU へ退避する
#   3. さらに余れば、並列スロット分の KV キャッシュを確保する
#   4. それでも余れば、KV キャッシュ型を f16 / q8_0 へ引き上げる
if [[ -n "$KV_CACHE_TYPE" ]]; then
  kv_base_type="$KV_CACHE_TYPE"
  kv_upgrade_types=()
else
  kv_base_type=q4_0
  kv_upgrade_types=(f16 q8_0)
fi
model_kv_cache_type="$kv_base_type"
model_parallel=1
model_fit_fallback=0
model_file_bytes="$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)"
kv_params="$(detect_kv_cache_params "$model_file")"
usable_vram_bytes="$(detect_usable_vram_bytes)"
if [[ -z "$kv_params" ]]; then
  log "Could not detect attention params for $filename; relying on llama-server's --fit auto-adjustment"
  model_fit_fallback=1
elif ((usable_vram_bytes <= 0)); then
  log "Could not detect free VRAM for $filename; relying on llama-server's --fit auto-adjustment"
  model_fit_fallback=1
else
  kv_one_slot_bytes="$(kv_vram_bytes "$kv_params" "$kv_base_type" "$model_ctx_size")"
  if ((kv_one_slot_bytes > usable_vram_bytes)); then
    # KV 所要量は ctx にほぼ比例するが SWA 分が非線形のため、CONTEXT_SIZE_STEP 単位で二分探索する
    lo=0 hi=$((model_ctx_size / CONTEXT_SIZE_STEP))
    while ((lo < hi)); do
      mid=$(((lo + hi + 1) / 2))
      if (($(kv_vram_bytes "$kv_params" "$kv_base_type" $((mid * CONTEXT_SIZE_STEP))) <= usable_vram_bytes)); then
        lo=$mid
      else
        hi=$((mid - 1))
      fi
    done
    fitted_ctx=$((lo * CONTEXT_SIZE_STEP))
    if ((fitted_ctx >= MIN_CONTEXT_SIZE)); then
      log "$filename: $kv_base_type KV cache for ctx-size=$model_ctx_size ($((kv_one_slot_bytes / 1024 / 1024)) MiB) exceeds usable VRAM ($((usable_vram_bytes / 1024 / 1024)) MiB); shrinking ctx-size to $fitted_ctx"
      model_ctx_size="$fitted_ctx"
      kv_one_slot_bytes="$(kv_vram_bytes "$kv_params" "$kv_base_type" "$model_ctx_size")"
    else
      log "$filename: cannot fit $kv_base_type KV cache even at MIN_CONTEXT_SIZE=$MIN_CONTEXT_SIZE; relying on llama-server's --fit auto-adjustment"
      model_fit_fallback=1
    fi
  fi

  if ((model_fit_fallback == 0)); then
    remaining_bytes=$((usable_vram_bytes - kv_one_slot_bytes))
    if ((model_file_bytes <= remaining_bytes)); then
      if ((MAX_PARALLEL_SLOTS > 1 && kv_one_slot_bytes > 0)); then
        model_parallel=$((1 + (remaining_bytes - model_file_bytes) / kv_one_slot_bytes))
        ((model_parallel > MAX_PARALLEL_SLOTS)) && model_parallel="$MAX_PARALLEL_SLOTS"
      fi
      for candidate in "${kv_upgrade_types[@]}"; do
        candidate_bytes="$(kv_vram_bytes "$kv_params" "$candidate" "$model_ctx_size" "$model_parallel")"
        if ((model_file_bytes + candidate_bytes <= usable_vram_bytes)); then
          model_kv_cache_type="$candidate"
          break
        fi
      done
    else
      log "$filename: model weights ($((model_file_bytes / 1024 / 1024)) MiB) do not fit next to the $kv_base_type KV cache ($((kv_one_slot_bytes / 1024 / 1024)) MiB) in usable VRAM ($((usable_vram_bytes / 1024 / 1024)) MiB); part of the weights will be kept on CPU"
    fi
    log "Estimated plan for $filename: ctx-size=$model_ctx_size, KV cache=$model_kv_cache_type, slots=$model_parallel"
  fi
fi

model_ctx_per_slot="$model_ctx_size"
model_tensor_split=""
model_n_gpu_layers_fixed=""
model_offload_kind=""
if [[ "$model_fit_fallback" == 1 ]]; then
  model_parallel=1
  model_kv_cache_type="$kv_base_type"
elif [[ "$SPLIT_MODE" != "layer" ]]; then
  # 手動 tensor-split/n-gpu-layers 計算は「レイヤーを丸ごと1枚のGPUに割り当てる」前提の
  # ロジックのため、row/tensor/none 分割では成立しない。--fit に委ねる。
  log "SPLIT_MODE=$SPLIT_MODE; skipping layer-based tensor-split calculation and relying on llama-server's --fit auto-adjustment"
elif [[ "$TENSOR_SPLIT_MODE" == "auto" && "$gpu_count" -ge 2 ]]; then
  gpu_tg_speeds="$(detect_per_gpu_tg_speed "$model_file")"
  if [[ -n "$gpu_tg_speeds" ]]; then
    log "Per-GPU generation speed (tok/s): $(tr '\n' ' ' <<< "$gpu_tg_speeds")"
    layer_bytes_file="$(mktemp)"
    # CPU へ退避できる重み: MoE は Expert (--n-cpu-moe)、Dense は FFN (--n-cpu-ffn)。
    # いずれも Attention と KV キャッシュは GPU に残るため、KV 優先の方針を維持できる。
    offload_allowed=0
    if [[ -n "${expert_bytes:-}" ]] && ((expert_bytes > 0)); then
      layer_bytes_mode="split"
      if ((model_moe_auto == 1)); then
        offload_allowed=1
        model_offload_kind="moe"
      fi
    else
      layer_bytes_mode="split-ffn"
    fi
    if detect_layer_bytes "$model_file" "$layer_bytes_mode" > "$layer_bytes_file" && [[ -s "$layer_bytes_file" ]]; then
      if [[ "$layer_bytes_mode" == "split-ffn" ]]; then
        ffn_bytes="$(awk 'NR > 1 { sum += $2 } END { printf "%.0f", sum }' "$layer_bytes_file")"
        available_ram_bytes="$(detect_available_ram_bytes)"
        if ((ffn_bytes > 0)) && [[ "$available_ram_bytes" =~ ^[0-9]+$ ]] \
            && ((available_ram_bytes >= ffn_bytes + MOE_RAM_RESERVE_MIB * 1024 * 1024)); then
          offload_allowed=1
          model_offload_kind="ffn"
        fi
      fi
      free_mib_list="$(detect_per_gpu_free_vram_mib)"
      try_split() {
        local cache_type="$1" slots="$2" ctx="${3:-$model_ctx_per_slot}"
        calculate_tensor_split "$layer_bytes_file" "$free_mib_list" "$gpu_tg_speeds" "$VRAM_RESERVE_MIB" \
          "$offload_allowed" $((ctx * slots)) "$(swa_cache_tokens "$kv_params" "$ctx" "$slots")" \
          "$(kv_cache_type_bytes_per_element "$cache_type")" "$(attn_scratch_bytes "$kv_params" "$cache_type" "$ctx" "$slots")" || true
      }
      offloaded_layers() {
        local n
        read -r _ _ n _ <<< "$1"
        echo "${n:-0}"
      }

      # 1〜2: 基本型の KV キャッシュを 1 スロット分確保したうえで、CPU へ退避する層数が最小になる配置
      result="$(try_split "$kv_base_type" 1)"
      if [[ -z "$result" ]] && ((offload_allowed == 1)); then
        # FFN / Expert をすべて CPU へ退避しても、KV キャッシュと GPU に残す重み (Attention 等) が
        # 収まらない。--fit に任せると層ごと CPU へ移り KV キャッシュも CPU 側に置かれるため、
        # KV キャッシュを VRAM に載せきれる最大の ctx-size まで切り詰める。
        low_step=$(( (MIN_CONTEXT_SIZE + CONTEXT_SIZE_STEP - 1) / CONTEXT_SIZE_STEP ))
        high_step=$(( model_ctx_per_slot / CONTEXT_SIZE_STEP ))
        best_ctx=0
        while ((low_step <= high_step)); do
          mid_step=$(( (low_step + high_step) / 2 ))
          trial="$(try_split "$kv_base_type" 1 $((mid_step * CONTEXT_SIZE_STEP)))"
          if [[ -n "$trial" ]]; then
            best_ctx=$((mid_step * CONTEXT_SIZE_STEP))
            result="$trial"
            low_step=$((mid_step + 1))
          else
            high_step=$((mid_step - 1))
          fi
        done
        if ((best_ctx > 0)); then
          log "$filename: $kv_base_type KV cache for ctx-size=$model_ctx_per_slot does not fit in VRAM next to the non-offloadable weights; shrinking ctx-size to $best_ctx"
          model_ctx_per_slot="$best_ctx"
        fi
      fi
      if [[ -z "$result" ]]; then
        log "Could not fit $kv_base_type KV cache for ctx-size=$model_ctx_per_slot with a manual tensor-split for $filename; relying on llama-server's --fit auto-adjustment"
        model_parallel=1
        model_kv_cache_type="$kv_base_type"
      else
        min_offload="$(offloaded_layers "$result")"
        # 3: 退避層数を増やさずに確保できる最大スロット数
        model_parallel=1
        for ((slots = MAX_PARALLEL_SLOTS; slots > 1; slots--)); do
          trial="$(try_split "$kv_base_type" "$slots")"
          if [[ -n "$trial" ]] && (($(offloaded_layers "$trial") == min_offload)); then
            model_parallel="$slots"
            result="$trial"
            break
          fi
        done
        # 4: 退避層数・スロット数を維持できる範囲で KV キャッシュ型を引き上げる
        model_kv_cache_type="$kv_base_type"
        for candidate in "${kv_upgrade_types[@]}"; do
          trial="$(try_split "$candidate" "$model_parallel")"
          if [[ -n "$trial" ]] && (($(offloaded_layers "$trial") == min_offload)); then
            model_kv_cache_type="$candidate"
            result="$trial"
            break
          fi
        done

        read -r model_n_gpu_layers_fixed model_tensor_split model_n_cpu_moe cpu_offload_bytes <<< "$result"
        model_n_cpu_moe="${model_n_cpu_moe:-0}"
        cpu_offload_bytes="${cpu_offload_bytes:-0}"
        if ((model_n_cpu_moe > 0)); then
          log "$filename: keeping $model_offload_kind weights of the first $model_n_cpu_moe layers ($((cpu_offload_bytes / 1024 / 1024)) MiB) on CPU to keep the KV cache in VRAM"
        fi
        log "Calculated tensor-split for $filename: n-gpu-layers=$model_n_gpu_layers_fixed tensor-split=$model_tensor_split"
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
model_ctx_size=$((model_ctx_per_slot * model_parallel))
log "Auto-selected KV cache type for $filename: $model_kv_cache_type"

section_file="$PRESET_SECTION_DIR/$alias_name.ini"
tmp_section="$(mktemp "$section_file.tmp.XXXXXX")"
if ((model_parallel > 1)); then
  log "Allocating $model_parallel parallel slots for $filename (per-slot ctx-size=$model_ctx_per_slot, total ctx-size=$model_ctx_size)"
else
  log "Serializing requests for $filename (1 slot, ctx-size=$model_ctx_per_slot)"
fi
{
  printf '[%s]\n' "$alias_name"
  printf 'model = %s\n' "$model_file"
  printf 'ctx-size = %s\n' "$model_ctx_size"
  printf 'parallel = %s\n' "$model_parallel"
  # --kv-unified では全スロットが 1 本の KV プールを共有し、各スロットには ctx-size 全量が
  # 使える (n_ctx_slot = ctx-size) と申告されてしまう。1 リクエストがプールを食い潰して
  # 他のリクエストを "Context size has been exceeded" で落とさないよう上限を明示する。
  printf 'kv-unified-per-slot = %s\n' "$model_ctx_per_slot"
  printf 'sleep-idle-seconds = %s\n' "$MODEL_IDLE_SECONDS"
  if [[ -n "$model_tensor_split" ]]; then
    printf 'fit = off\n'
    printf 'n-gpu-layers = %s\n' "$model_n_gpu_layers_fixed"
    printf 'tensor-split = %s\n' "$model_tensor_split"
  else
    printf 'n-gpu-layers = %s\n' "$N_GPU_LAYERS"
  fi
  if [[ -n "$model_n_cpu_moe" ]] && ((model_n_cpu_moe > 0)); then
    if [[ "$model_offload_kind" == "ffn" ]]; then
      printf 'n-cpu-ffn = %s\n' "$model_n_cpu_moe"
    else
      printf 'n-cpu-moe = %s\n' "$model_n_cpu_moe"
    fi
  fi
  kv_type="${model_kv_cache_type:-${KV_CACHE_TYPE:-f16}}"
  printf 'cache-type-k = %s\n' "$kv_type"
  printf 'cache-type-v = %s\n' "$kv_type"
  printf '\n'
} > "$tmp_section"
mv "$tmp_section" "$section_file"
render_preset
log "Updated lazy preset for $alias_name"
