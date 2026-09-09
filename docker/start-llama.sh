#!/usr/bin/env bash
set -euo pipefail

# モデルリストは run-llama.sh 経由で MODEL_NAMES_CSV 環境変数として渡される。
# 形式: Hugging Face のリポジトリ名/ファイル名.gguf をカンマ区切りで指定。
MODEL_NAMES=()

MODEL_DIR="${MODEL_DIR:-/models}"
MODEL_NAMES_CSV="${MODEL_NAMES_CSV:-}"
HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
MODEL_IDLE_SECONDS="${MODEL_IDLE_SECONDS:-300}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8080}"
# CONTEXT_SIZE が指定されない場合は各モデルの GGUF メタデータから
# 推奨(学習時最大)コンテキスト長を自動検出して使用する。
# MAX_CONTEXT_SIZE を指定すると、自動検出値に上限(VRAM保護用)をかけられる。
CONTEXT_SIZE="${CONTEXT_SIZE:-}"
MAX_CONTEXT_SIZE="${MAX_CONTEXT_SIZE:-}"
N_GPU_LAYERS="${N_GPU_LAYERS:-auto}"
MODELS_MAX="${MODELS_MAX:-1}"
# 複数 GPU の性能差(生成速度)を考慮した tensor-split を自動計算するかどうか。
# auto: 2 GPU 以上あり、各 GPU の生成速度比が十分な差(閾値以上)であれば手動計算した
#       tensor-split / n-gpu-layers を明示指定する(--fit は off にする)。
#       計算に失敗した場合や条件を満たさない場合は自動的に --fit on にフォールバックする。
# off:  常に --fit on に委ねる(従来の挙動)。
TENSOR_SPLIT_MODE="${TENSOR_SPLIT_MODE:-auto}"
# KV キャッシュの量子化タイプ (f16 既定より VRAM を大幅削減できる: q8_0 で約1/2, q4_0 で約1/4)
# allowed: f32, f16, bf16, q8_0, q4_0, q4_1, iq4_nl, q5_0, q5_1
# 未指定の場合は、空き VRAM とモデルサイズ・コンテキスト長から必要な KV キャッシュ量を見積もり、
# 収まる範囲でなるべく精度の高い(f16に近い)タイプを自動選択する。
KV_CACHE_TYPE="${KV_CACHE_TYPE:-}"

log() { printf '[llama-wrapper] %s\n' "$*" >&2; }
die() { printf '[llama-wrapper] error: %s\n' "$*" >&2; exit 1; }

[[ "$MODEL_IDLE_SECONDS" =~ ^[0-9]+$ ]] || die "MODEL_IDLE_SECONDS must be an integer"
[[ "$PORT" =~ ^[0-9]+$ ]] || die "PORT must be an integer"
[[ -z "$CONTEXT_SIZE" || "$CONTEXT_SIZE" =~ ^[0-9]+$ ]] || die "CONTEXT_SIZE must be an integer"
[[ -z "$MAX_CONTEXT_SIZE" || "$MAX_CONTEXT_SIZE" =~ ^[0-9]+$ ]] || die "MAX_CONTEXT_SIZE must be an integer"
[[ "$N_GPU_LAYERS" == "auto" || "$N_GPU_LAYERS" == "all" || "$N_GPU_LAYERS" =~ ^-?[0-9]+$ ]] || die "N_GPU_LAYERS must be an integer, 'auto', or 'all'"
[[ "$MODELS_MAX" =~ ^[0-9]+$ ]] || die "MODELS_MAX must be an integer"
[[ "$TENSOR_SPLIT_MODE" == "auto" || "$TENSOR_SPLIT_MODE" == "off" ]] || die "TENSOR_SPLIT_MODE must be 'auto' or 'off'"
case "$KV_CACHE_TYPE" in
  ""|f32|f16|bf16|q8_0|q4_0|q4_1|iq4_nl|q5_0|q5_1) ;;
  *) die "KV_CACHE_TYPE must be one of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1" ;;
esac

# GGUF ファイルのメタデータから <arch>.context_length を読み取る。
# gguf-dump (llama.cpp 同梱) を使い、取得できなければ空文字を返す。
detect_context_length() {
  local model_file="$1"
  gguf-dump --no-tensors "$model_file" 2>/dev/null \
    | awk -F'= *' '/\.context_length[[:space:]]*=/ { print $2; exit }' \
    | tr -d '[:space:]'
}

# GGUF ファイルのメタデータから KV キャッシュサイズ計算に必要な値を読み取る。
# 見つかった場合 "block_count head_count_kv key_length value_length" を空白区切りで返す。
# head_count_kv は Nemotron-H のようなハイブリッド(Mamba/Transformer混在)アーキテクチャでは
# レイヤーごとの配列(Mambaレイヤーは0、Attentionレイヤーのみ実値)として保存されている場合があるため、
# --json --json-array で全要素を取得し、実際にアテンションで使われる最大値を採用する
# (0 のままだと必要 KV キャッシュ量を過小評価し、VRAM 予算チェックが素通りして OOM の原因になる)。
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

# 指定したキャッシュタイプ 1要素あたりのバイト数(概算)を返す。
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

# 空き VRAM 合計・モデルファイルサイズ・コンテキスト長から、収まる範囲でなるべく精度の高い
# KV キャッシュタイプを選ぶ。精度が高い順に f16 -> q8_0 -> q4_0 を試す。
# 引数: model_file ctx_size free_vram_bytes_total
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

  local model_bytes
  model_bytes="$(stat -c '%s' "$model_file" 2>/dev/null || echo 0)"
  # モデル重みロード後に KV キャッシュ用として残る VRAM (安全マージンとして 90% だけ使う想定)
  local budget_bytes
  budget_bytes="$(awk -v f="$free_bytes" -v m="$model_bytes" 'BEGIN { b = (f - m) * 0.9; if (b < 0) b = 0; printf "%.0f", b }')"

  local candidate bytes_per_elem needed_bytes
  for candidate in f16 q8_0 q4_0; do
    bytes_per_elem="$(kv_cache_type_bytes_per_element "$candidate")"
    # 必要バイト数 = 2(K+V) * block_count * head_count_kv * (key_length+value_length) * ctx_size * bytes_per_elem / 2
    needed_bytes="$(awk -v bc="$block_count" -v hkv="$head_count_kv" -v kl="$key_length" -v vl="$value_length" \
      -v ctx="$ctx_size" -v bpe="$bytes_per_elem" \
      'BEGIN { printf "%.0f", bc * hkv * (kl + vl) * ctx * bpe }')"
    if (( $(awk -v n="$needed_bytes" -v b="$budget_bytes" 'BEGIN { print (n <= b) ? 1 : 0 }') )); then
      log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB (budget $((budget_bytes / 1024 / 1024)) MiB) -> selected"
      echo "$candidate"
      return
    else
      log "$(basename "$model_file"): estimated KV cache for $candidate = $((needed_bytes / 1024 / 1024)) MiB exceeds budget $((budget_bytes / 1024 / 1024)) MiB"
    fi
  done
  # どれも収まらない場合は最も VRAM を節約できる q4_0 にフォールバックする
  # (それでも収まらない場合は --fit による CPU オフロードに期待する)
  echo "q4_0"
}

# 全 GPU の空き VRAM 合計(バイト)を返す。取得できなければ 0 を返す。
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

# GPU ごとの空き VRAM (MiB) を index 順の配列として返す(1行1台)。
# CUDA_VISIBLE_DEVICES が設定されている場合は、そこで指定された順序・台数に絞る。
detect_per_gpu_free_vram_mib() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    local idx
    IFS=',' read -r -a idx <<< "$CUDA_VISIBLE_DEVICES"
    local i
    for i in "${idx[@]}"; do
      nvidia-smi --id="$i" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null
    done
  else
    nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null
  fi
}

# GPU ごとの相対的な生成速度 (tokens/sec) を llama-bench で計測し、GPU index 順に
# 改行区切りで返す。1 GPU しか無い、または計測できない場合は空文字を返す。
# 結果はコンテナ内にキャッシュし、以後は再計測しない (数分かかるため)。
BENCH_TG_CACHE="$MODEL_DIR/.gpu-tg-speed.tsv"
detect_per_gpu_tg_speed() {
  local small_model="$1"
  if ((gpu_count < 2)); then
    echo ""
    return
  fi
  if [[ ! -f "$BENCH_TG_CACHE" ]]; then
    log "Benchmarking per-GPU generation speed for tensor-split calculation (one-time, may take a while)"
    : > "$BENCH_TG_CACHE"
    local i speed
    for ((i = 0; i < gpu_count; i++)); do
      speed="$(CUDA_VISIBLE_DEVICES="$i" llama-bench -m "$small_model" -ngl 99 -p 0 -n 32 \
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

# GGUF ファイルのテンソル形状・量子化タイプから、各レイヤー(blk.N)のおおよそのバイト数と
# レイヤー以外(embedding/output等)の合計バイト数を計算する。
# 出力: 1行目にレイヤー以外の合計バイト数、以降レイヤー番号順に1行1レイヤーのバイト数。
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

# GPU 性能比(生成速度)・各 GPU の空き VRAM・レイヤーごとのバイト数から、
# OOM しない範囲で速い GPU により多くのレイヤーを割り当てる tensor-split を計算する。
# 引数: layer_bytes_file(detect_layer_bytesの出力) free_mib_list(改行区切り) speed_list(改行区切り) reserve_mib_per_gpu
# 出力: 成功時 "n_gpu_layers tensor_split_csv"、収まらない/条件を満たさない場合は空文字。
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
# 各 GPU が重み用に使える予算(空きVRAM - 予約マージン)。0未満は0に丸める。
budget = [max(0, m * 1024 * 1024 - reserve_bytes) for m in free_mib]

# 非レイヤー分(embedding/output等)は最速GPUに割り当てる想定で予算から差し引く
fastest = max(range(n_gpu), key=lambda i: speeds[i])
budget[fastest] -= other_bytes
if budget[fastest] < 0:
    # 最速GPUに埋め込み層すら乗らない場合は手動計算を諦める
    sys.exit(1)

# 速度比に応じた理想レイヤー数(合計 n_layers になるよう按分)
total_speed = sum(speeds)
ideal = [n_layers * s / total_speed for s in speeds]

# 大きい方から割り当てていき、各GPUの予算(バイト)を超えないようにレイヤーを1つずつ配る
# レイヤーサイズはほぼ均一という前提で、平均サイズを使って予算に収まる最大レイヤー数を求める。
avg_layer_bytes = sum(layer_bytes) / n_layers
max_layers_by_budget = [int(budget[i] // avg_layer_bytes) if avg_layer_bytes > 0 else n_layers for i in range(n_gpu)]

# 理想配分を四捨五入しつつ、各GPUの予算上限でクリップする
assign = [min(round(ideal[i]), max_layers_by_budget[i]) for i in range(n_gpu)]

# 端数調整: 割り当て済み合計が n_layers に届かない場合、予算に余裕がある GPU から
# (速い順に)不足分を追加する。超過している場合は逆に遅い GPU から削る。
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

if total_assigned() != n_layers:
    # 予算が全体として不足しており全レイヤーを配置しきれない -> 手動計算を諦める
    sys.exit(1)
if any(a < 0 for a in assign):
    sys.exit(1)

print(n_layers, ",".join(str(a) for a in assign))
' "$layer_bytes_file" "$free_mib_list" "$speed_list" "$reserve_mib"
}

if [[ -n "$MODEL_NAMES_CSV" ]]; then
  IFS=',' read -r -a MODEL_NAMES <<< "$MODEL_NAMES_CSV"
fi
((${#MODEL_NAMES[@]} > 0)) || die "MODEL_NAMES must not be empty"

mkdir -p "$MODEL_DIR"
declare -A allowed_files=()

for model_spec in "${MODEL_NAMES[@]}"; do
  [[ "$model_spec" == */* ]] || die "invalid model spec: $model_spec"
  repo="${model_spec%/*}"
  filename="${model_spec##*/}"
  [[ "$filename" == *.gguf ]] || die "model must be a .gguf file: $model_spec"
  destination="$MODEL_DIR/$filename"
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
done

shopt -s nullglob
for cached_file in "$MODEL_DIR"/*.gguf; do
  filename="$(basename "$cached_file")"
  if [[ -z "${allowed_files[$filename]+x}" ]]; then
    log "Removing model not in MODEL_NAMES: $filename"
    rm -f -- "$cached_file"
  fi
done
shopt -u nullglob

gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_count="$(nvidia-smi --list-gpus 2>/dev/null | wc -l)"
fi
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  visible="${CUDA_VISIBLE_DEVICES//,/ }"
  gpu_count="$(awk '{print NF}' <<< "$visible")"
fi
((gpu_count > 0)) || gpu_count=1

# tensor-split や n-gpu-layers を固定で指定すると、llama-server の自動フィット
# 機能 (--fit, デフォルト有効) が「ユーザー指定済み」と判断して調整を放棄し、
# VRAM に収まらない場合にそのまま OOM で落ちてしまう。
# そのため既定では明示指定せず --fit に委ねるが、TENSOR_SPLIT_MODE=auto かつ
# 複数 GPU の性能差が大きい場合は、性能比に応じた tensor-split を自前で計算し、
# --fit off として明示指定することで高速化を図る(計算に失敗した場合は --fit にフォールバック)。
gpu_tg_speeds=""
if [[ "$TENSOR_SPLIT_MODE" == "auto" && "$gpu_count" -ge 2 ]]; then
  # 性能計測にはダウンロード済みモデルの中で最もファイルサイズが小さいものを使う
  # (ロードが速く、全レイヤーが確実に全GPUに載る可能性が高いため)。
  bench_filename=""
  bench_size=0
  for filename in "${!allowed_files[@]}"; do
    size="$(stat -c '%s' "$MODEL_DIR/$filename" 2>/dev/null || echo 0)"
    if [[ -z "$bench_filename" ]] || ((size < bench_size)); then
      bench_filename="$filename"
      bench_size="$size"
    fi
  done
  if [[ -n "$bench_filename" ]]; then
    gpu_tg_speeds="$(detect_per_gpu_tg_speed "$MODEL_DIR/$bench_filename")"
  fi
  if [[ -n "$gpu_tg_speeds" ]]; then
    log "Per-GPU generation speed (tok/s): $(tr '\n' ' ' <<< "$gpu_tg_speeds")"
  else
    log "Could not benchmark per-GPU speed; tensor-split will fall back to --fit"
  fi
else
  log "TENSOR_SPLIT_MODE=$TENSOR_SPLIT_MODE or single GPU; relying on llama-server's --fit auto-adjustment"
fi

# router server 用の preset (INI) ファイルを生成する。
# ベンチマーク自体は情報収集目的で一度だけ実行し、結果ファイルをキャッシュする。
PRESET_FILE="$MODEL_DIR/.models-preset.ini"
{
  # コマンドライン引数はpresetより優先されるため、--fit はここでは指定せず
  # global セクションの既定値として on を置く。各モデルの preset セクションで
  # 手動 tensor-split を使う場合のみ fit = off で上書きする。
  printf '[*]\n'
  printf 'fit = on\n'
  printf '\n'
} > "$PRESET_FILE"

for filename in "${!allowed_files[@]}"; do
  model_file="$MODEL_DIR/$filename"
  alias_name="${filename%.gguf}"

  benchmark_marker="$MODEL_DIR/.llama-bench-${filename}.done"
  if [[ ! -f "$benchmark_marker" ]]; then
    log "Running llama-bench for $filename"
    # llama-bench は -ngl に auto/all を受け付けないため、ベンチマーク専用に固定値を使う。
    llama-bench -m "$model_file" -ngl -1 -p 128 -n 32 \
      --output json > "$MODEL_DIR/llama-bench-${filename}.json"
    : > "$benchmark_marker"
  fi

  # CONTEXT_SIZE が明示指定されていればそれを全モデル共通で使う。
  # 未指定ならモデルごとに GGUF から推奨(学習時最大)コンテキスト長を検出する。
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

  # KV_CACHE_TYPE が明示指定されていればそれを使う。未指定なら空き VRAM から自動選択する。
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

  # TENSOR_SPLIT_MODE=auto かつ GPU 性能差の実測に成功していれば、性能比に応じた
  # tensor-split / n-gpu-layers を計算してみる。VRAM に収まらない・計算失敗時は
  # 空文字となり、後段で --fit on にフォールバックする。
  model_tensor_split=""
  model_n_gpu_layers_fixed=""
  if [[ -n "$gpu_tg_speeds" ]]; then
    layer_bytes_file="$(mktemp)"
    if detect_layer_bytes "$model_file" > "$layer_bytes_file" && [[ -s "$layer_bytes_file" ]]; then
      free_mib_list="$(detect_per_gpu_free_vram_mib)"
      # KV キャッシュもレイヤーと同じ比率で各 GPU に分散配置されるため、重み用予算からは
      # 「KVキャッシュ総量 / GPU数」+ 固定マージン(1024MiB)を安全側に一律差し引く。
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
  fi

  {
    printf '[%s]\n' "$alias_name"
    printf 'model = %s\n' "$model_file"
    printf 'ctx-size = %s\n' "$model_ctx_size"
    printf 'sleep-idle-seconds = %s\n' "$MODEL_IDLE_SECONDS"
    if [[ -n "$model_tensor_split" ]]; then
      # 手動計算した tensor-split を使う場合、--fit は明示指定と衝突して例外になるため off にする。
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
  } >> "$PRESET_FILE"
done

log "Starting OpenAI-compatible llama-server (router mode) on port $PORT"
log "Available models: ${!allowed_files[*]} (models-max=$MODELS_MAX)"
exec llama-server \
  --models-preset "$PRESET_FILE" \
  --models-max "$MODELS_MAX" \
  --host "$HOST" \
  --port "$PORT"
