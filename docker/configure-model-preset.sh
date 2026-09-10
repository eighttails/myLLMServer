#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-300}"
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
MODELS_MAX="${MODELS_MAX:-1}"
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"
TENSOR_SPLIT_MODE="${TENSOR_SPLIT_MODE:-auto}"
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
    | ($entries[] | select(.key | endswith(".attention.head_count_kv")) | .value.value | scalar_or_max) as $head_count_kv
    | ($entries[] | select(.key | endswith(".attention.key_length")) | .value.value | scalar_or_max) as $key_length
    | ($entries[] | select(.key | endswith(".attention.value_length")) | .value.value | scalar_or_max) as $value_length
    | "\($block_count) \($head_count_kv) \($key_length) \($value_length)"
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

select_kv_cache_type() {
  local model_file="$1" ctx_size="$2" free_bytes="$3"
  local params block_count head_count_kv key_length value_length
  params="$(detect_kv_cache_params "$model_file")"
  if [[ -z "$params" ]]; then
    log "Could not detect attention params for $(basename "$model_file"); skipping KV cache type auto-selection"
    echo ""
    return
  fi
  read -r block_count head_count_kv key_length value_length <<< "$params"

  local model_bytes budget_bytes
  model_bytes="$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)"
  budget_bytes="$(awk -v f="$free_bytes" -v m="$model_bytes" 'BEGIN { b = (f - m) * 0.9; if (b < 0) b = 0; printf "%.0f", b }')"

  local candidate bytes_per_elem needed_bytes
  for candidate in f16 q8_0 q4_0; do
    bytes_per_elem="$(kv_cache_type_bytes_per_element "$candidate")"
    needed_bytes="$(awk -v bc="$block_count" -v hkv="$head_count_kv" -v kl="$key_length" -v vl="$value_length" \
      -v ctx="$ctx_size" -v bpe="$bytes_per_elem" \
      'BEGIN { printf "%.0f", bc * hkv * (kl + vl) * ctx * bpe }')"
    if (( $(awk -v n="$needed_bytes" -v b="$budget_bytes" 'BEGIN { print (n <= b) ? 1 : 0 }') )); then
      log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB (budget $((budget_bytes / 1024 / 1024)) MiB) -> selected"
      echo "$candidate"
      return
    fi
    log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB exceeds budget $((budget_bytes / 1024 / 1024)) MiB"
  done
  echo "q4_0"
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
        --output json 2>/dev/null | python3 -c '
import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    print(0)
    sys.exit(0)
for row in data:
    if row.get("n_gen", 0) > 0:
        print(row.get("avg_ts", 0))
        break
else:
    print(0)
' 2>/dev/null)"
      [[ "$speed" =~ ^[0-9.]+$ ]] || speed=0
      printf '%s\n' "$speed" >> "$BENCH_TG_CACHE"
    done
  fi
  cat "$BENCH_TG_CACHE"
}

detect_layer_bytes() {
  local model_file="$1"
  gguf-dump --json "$model_file" 2>/dev/null | python3 -c '
import json, sys, re

QK = {
    "F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
    "Q4_0": (32, 18), "Q4_1": (32, 20), "Q5_0": (32, 22), "Q5_1": (32, 24), "Q8_0": (32, 34),
    "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210), "Q8_K": (256, 292),
    "Q2_K": (256, 84), "Q3_K": (256, 110),
    "IQ4_NL": (32, 18), "IQ4_XS": (256, 136),
}

def tensor_bytes(shape, typ):
    n = 1
    for s in shape:
        n *= s
    block_size, type_bytes = QK.get(typ, (1, 4))
    return (n // block_size) * type_bytes

try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(1)

tensors = data.get("tensors", {})
layer_bytes = {}
other_bytes = 0
for name, info in tensors.items():
    b = tensor_bytes(info["shape"], info["type"])
    m = re.match(r"^blk\.(\d+)\.", name)
    if m:
        idx = int(m.group(1))
        layer_bytes[idx] = layer_bytes.get(idx, 0) + b
    else:
        other_bytes += b

if not layer_bytes:
    sys.exit(1)

print(other_bytes)
for idx in sorted(layer_bytes.keys()):
    print(layer_bytes[idx])
'
}

calculate_tensor_split() {
  local layer_bytes_file="$1" free_mib_list="$2" speed_list="$3" reserve_mib="$4"
  python3 -c '
import sys

layer_bytes_file, free_mib_str, speed_str, reserve_mib = sys.argv[1:5]

with open(layer_bytes_file) as f:
    lines = [l.strip() for l in f if l.strip() != ""]
if not lines:
    sys.exit(1)
other_bytes = int(lines[0])
layer_bytes = [int(x) for x in lines[1:]]
n_layers = len(layer_bytes)
if n_layers == 0:
    sys.exit(1)

free_mib = [int(x) for x in free_mib_str.split() if x.strip() != ""]
speeds = [float(x) for x in speed_str.split() if x.strip() != ""]
n_gpu = len(free_mib)
if n_gpu < 2 or len(speeds) != n_gpu:
    sys.exit(1)
if any(s <= 0 for s in speeds):
    sys.exit(1)

reserve_bytes = int(reserve_mib) * 1024 * 1024
budget = [max(0, m * 1024 * 1024 - reserve_bytes) for m in free_mib]

fastest = max(range(n_gpu), key=lambda i: speeds[i])
budget[fastest] -= other_bytes
if budget[fastest] < 0:
    sys.exit(1)

total_speed = sum(speeds)
ideal = [n_layers * s / total_speed for s in speeds]
avg_layer_bytes = sum(layer_bytes) / n_layers
max_layers_by_budget = [int(budget[i] // avg_layer_bytes) if avg_layer_bytes > 0 else n_layers for i in range(n_gpu)]
assign = [min(round(ideal[i]), max_layers_by_budget[i]) for i in range(n_gpu)]

def total_assigned():
    return sum(assign)

order_fast_to_slow = sorted(range(n_gpu), key=lambda i: -speeds[i])
guard = 0
while total_assigned() < n_layers and guard < n_layers * 2:
    guard += 1
    added = False
    for i in order_fast_to_slow:
        if assign[i] < max_layers_by_budget[i]:
            assign[i] += 1
            added = True
            if total_assigned() >= n_layers:
                break
    if not added:
        break

order_slow_to_fast = sorted(range(n_gpu), key=lambda i: speeds[i])
guard = 0
while total_assigned() > n_layers and guard < n_layers * 2:
    guard += 1
    for i in order_slow_to_fast:
        if assign[i] > 0:
            assign[i] -= 1
            break
    else:
        break

if total_assigned() != n_layers or any(a < 0 for a in assign):
    sys.exit(1)

print(n_layers, ",".join(str(a) for a in assign))
' "$layer_bytes_file" "$free_mib_list" "$speed_list" "$reserve_mib"
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

model_kv_cache_type="$KV_CACHE_TYPE"
total_free_vram_bytes="$(detect_total_free_vram_bytes)"
if [[ -z "$model_kv_cache_type" ]]; then
  if ((total_free_vram_bytes > 0)); then
    model_kv_cache_type="$(select_kv_cache_type "$model_file" "$model_ctx_size" "$total_free_vram_bytes")"
    if [[ -n "$model_kv_cache_type" ]]; then
      log "Auto-selected KV cache type for $filename: $model_kv_cache_type"
    fi
  else
    log "Could not detect free VRAM; leaving KV cache type at llama-server default (f16) for $filename"
  fi
fi

model_tensor_split=""
model_n_gpu_layers_fixed=""
if [[ "$TENSOR_SPLIT_MODE" == "auto" && "$gpu_count" -ge 2 ]]; then
  gpu_tg_speeds="$(detect_per_gpu_tg_speed "$model_file")"
  if [[ -n "$gpu_tg_speeds" ]]; then
    log "Per-GPU generation speed (tok/s): $(tr '\n' ' ' <<< "$gpu_tg_speeds")"
    layer_bytes_file="$(mktemp)"
    if detect_layer_bytes "$model_file" > "$layer_bytes_file" && [[ -s "$layer_bytes_file" ]]; then
      free_mib_list="$(detect_per_gpu_free_vram_mib)"
      params="$(detect_kv_cache_params "$model_file")"
      kv_total_mib=0
      if [[ -n "$params" && -n "$model_kv_cache_type" ]]; then
        read -r bc hkv kl vl <<< "$params"
        bpe="$(kv_cache_type_bytes_per_element "$model_kv_cache_type")"
        kv_total_mib="$(awk -v bc="$bc" -v hkv="$hkv" -v kl="$kl" -v vl="$vl" -v ctx="$model_ctx_size" -v bpe="$bpe" \
          'BEGIN { printf "%.0f", (bc * hkv * (kl + vl) * ctx * bpe) / 1024 / 1024 }')"
      fi
      reserve_mib="$(awk -v kv="$kv_total_mib" -v n="$gpu_count" 'BEGIN { printf "%.0f", 1024 + (kv / n) }')"
      result="$(calculate_tensor_split "$layer_bytes_file" "$free_mib_list" "$gpu_tg_speeds" "$reserve_mib" || true)"
      if [[ -n "$result" ]]; then
        read -r model_n_gpu_layers_fixed model_tensor_split <<< "$result"
        log "Calculated tensor-split for $filename: n-gpu-layers=$model_n_gpu_layers_fixed tensor-split=$model_tensor_split"
      else
        log "Could not fit a manual tensor-split for $filename within free VRAM; falling back to --fit"
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
  if [[ -n "$model_kv_cache_type" ]]; then
    printf 'cache-type-k = %s\n' "$model_kv_cache_type"
    printf 'cache-type-v = %s\n' "$model_kv_cache_type"
  fi
  printf '\n'
} > "$tmp_section"
mv "$tmp_section" "$section_file"
render_preset
log "Updated lazy preset for $alias_name"
