#!/usr/bin/env python3
import argparse
import json
import re
import sys


QK = {
    "F32": (1, 4),
    "F16": (1, 2),
    "BF16": (1, 2),
    "Q4_0": (32, 18),
    "Q4_1": (32, 20),
    "Q5_0": (32, 22),
    "Q5_1": (32, 24),
    "Q8_0": (32, 34),
    "Q4_K": (256, 144),
    "Q5_K": (256, 176),
    "Q6_K": (256, 210),
    "Q8_K": (256, 292),
    "Q2_K": (256, 84),
    "Q3_K": (256, 110),
    "IQ3_S": (256, 110),
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
    "MXFP4": (32, 17),
}


def read_benchmark_speed() -> int | float:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 0

    if not isinstance(data, list):
        return 0
    for row in data:
        if isinstance(row, dict) and row.get("n_gen", 0) > 0:
            return row.get("avg_ts", 0)
    return 0


def tensor_bytes(shape: list[int], tensor_type: str) -> int:
    count = 1
    for size in shape:
        count *= size
    block_size, type_bytes = QK.get(tensor_type, (1, 4))
    return ((count + block_size - 1) // block_size) * type_bytes


def metadata_value(data: dict, suffix: str):
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return None
    for key, entry in metadata.items():
        if not isinstance(key, str) or not key.endswith(suffix):
            continue
        value = entry.get("value") if isinstance(entry, dict) else None
        if isinstance(value, dict):
            value = value.get("value")
        return value
    return None


def layer_kv_profile(data: dict) -> tuple[list[tuple[int, bool]], int]:
    """層ごとの (1トークンあたりの KV 要素数, SWA層か) と sliding window 長を返す。

    - head_count_kv が層ごとの配列の場合は 0 の層 (SSM 等) を除外する
    - full_attention_interval を持つハイブリッドモデルは (i+1) % interval == 0 の層だけを Attention とする
    - sliding_window_pattern で SWA と判定された層は key/value_length_swa を使い、
      llama-server 側でウィンドウ分しか KV を確保しないため別枠で扱う
    """
    block_count = metadata_number(data, ".block_count")
    head_count_kv = metadata_value(data, ".attention.head_count_kv")
    key_length = metadata_number(data, ".attention.key_length")
    value_length = metadata_number(data, ".attention.value_length")
    key_length_swa = metadata_number(data, ".attention.key_length_swa") or key_length
    value_length_swa = metadata_number(data, ".attention.value_length_swa") or value_length
    sliding_window = metadata_number(data, ".attention.sliding_window")
    swa_pattern = metadata_value(data, ".attention.sliding_window_pattern")
    attention_interval = metadata_number(data, ".full_attention_interval")
    if not isinstance(swa_pattern, list) or sliding_window <= 0:
        swa_pattern = []

    layers: list[tuple[int, bool]] = []
    for index in range(block_count):
        if isinstance(head_count_kv, list):
            heads = head_count_kv[index] if index < len(head_count_kv) else 0
            heads = heads if isinstance(heads, int) and heads > 0 else 0
        else:
            heads = head_count_kv if isinstance(head_count_kv, int) else 0
            if attention_interval > 1 and (index + 1) % attention_interval != 0:
                heads = 0
        is_swa = index < len(swa_pattern) and swa_pattern[index] is True
        lengths = (key_length_swa + value_length_swa) if is_swa else (key_length + value_length)
        layers.append((heads * lengths, is_swa))
    return layers, sliding_window


def read_kv_profile() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1
    layers, sliding_window = layer_kv_profile(data)
    full_elements = sum(elements for elements, is_swa in layers if not is_swa)
    swa_elements = sum(elements for elements, is_swa in layers if is_swa)
    if full_elements + swa_elements <= 0:
        return 1
    # 量子化 KV の Flash Attention は層ごとに K/V を f16 へ展開する作業領域を使うため、
    # 1層あたりの最大要素数も返す。
    max_full = max((elements for elements, is_swa in layers if not is_swa), default=0)
    max_swa = max((elements for elements, is_swa in layers if is_swa), default=0)
    print(full_elements, swa_elements, sliding_window, max_full, max_swa)
    return 0


def read_layer_bytes(exclude_moe: bool = False, split_moe: bool = False, split_ffn: bool = False) -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1

    tensors = data.get("tensors", {})
    if not isinstance(tensors, dict):
        return 1
    kv_layers: list[tuple[int, bool]] = []
    if split_moe:
        kv_layers, _ = layer_kv_profile(data)
    layer_bytes: dict[int, int] = {}
    layer_moe_bytes: dict[int, int] = {}
    other_bytes = 0
    for name, info in tensors.items():
        if not isinstance(name, str) or not isinstance(info, dict):
            return 1
        shape = info.get("shape")
        tensor_type = info.get("type")
        if not isinstance(shape, list) or not all(isinstance(size, int) for size in shape):
            return 1
        if not isinstance(tensor_type, str):
            return 1
        is_moe = bool(re.search(r"\.ffn_.+_exps\.weight$", name))
        if exclude_moe and is_moe:
            continue
        # split_ffn 時は Dense FFN (--n-cpu-ffn で CPU へ退避できる重み) を退避可能分として分離する
        is_offloadable = (
            bool(re.search(r"\.ffn_(up|down|gate|gate_up)\.weight$", name))
            if split_ffn
            else is_moe
        )
        size = tensor_bytes(shape, tensor_type)
        match = re.match(r"^blk\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            if split_moe and is_offloadable:
                layer_moe_bytes[index] = layer_moe_bytes.get(index, 0) + size
            else:
                layer_bytes[index] = layer_bytes.get(index, 0) + size
        else:
            other_bytes += size

    if not layer_bytes:
        return 1

    print(other_bytes)
    for index in sorted(layer_bytes):
        if split_moe:
            kv_elements, is_swa = kv_layers[index] if index < len(kv_layers) else (0, False)
            print(layer_bytes[index], layer_moe_bytes.get(index, 0), kv_elements, int(is_swa))
        else:
            print(layer_bytes[index])
    return 0


def metadata_number(data: dict, suffix: str) -> int:
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        return 0
    for key, entry in metadata.items():
        if not isinstance(key, str) or not key.endswith(suffix):
            continue
        if not isinstance(entry, dict):
            return 0
        value = entry.get("value")
        if isinstance(value, dict):
            value = value.get("value")
        if isinstance(value, list):
            values = [item for item in value if isinstance(item, (int, float))]
            return int(max(values)) if values else 0
        return int(value) if isinstance(value, (int, float)) else 0
    return 0


def read_moe_profile() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1

    tensors = data.get("tensors", {})
    if not isinstance(tensors, dict):
        return 1

    expert_bytes = 0
    expert_layers: set[int] = set()
    for name, info in tensors.items():
        if (
            not isinstance(name, str)
            or not isinstance(info, dict)
            or not re.search(r"\.ffn_.+_exps\.weight$", name)
        ):
            continue
        shape = info.get("shape")
        tensor_type = info.get("type")
        if not isinstance(shape, list) or not all(isinstance(size, int) for size in shape):
            return 1
        if not isinstance(tensor_type, str):
            return 1
        expert_bytes += tensor_bytes(shape, tensor_type)
        match = re.match(r"^blk\.(\d+)\.", name)
        if match:
            expert_layers.add(int(match.group(1)))

    print(
        metadata_number(data, ".block_count"),
        metadata_number(data, ".expert_count"),
        metadata_number(data, ".expert_used_count"),
        len(expert_layers),
        expert_bytes,
    )
    return 0


def calculate_tensor_split(
    layer_bytes_file: str,
    free_mib_string: str,
    speed_string: str,
    reserve_mib: int,
    moe_auto: bool = False,
    kv_tokens: int = 0,
    swa_tokens: int = 0,
    kv_bytes_per_element: float = 0.0,
    scratch_bytes: int = 0,
) -> int:
    with open(layer_bytes_file, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return 1

    other_bytes = int(lines[0])
    try:
        detailed = len(lines) > 1 and len(lines[1].split()) > 1
        if detailed:
            parsed_layers = [tuple(int(value) for value in line.split()) for line in lines[1:]]
            if any(len(values) not in (2, 4) for values in parsed_layers):
                return 1
            base_layer_bytes = [values[0] for values in parsed_layers]
            moe_layer_bytes = [values[1] for values in parsed_layers]
            kv_layer_bytes = [
                int(values[2] * (swa_tokens if values[3] else kv_tokens) * kv_bytes_per_element)
                if len(values) == 4
                else 0
                for values in parsed_layers
            ]
        else:
            base_layer_bytes = [int(value) for value in lines[1:]]
            moe_layer_bytes = [0] * len(base_layer_bytes)
            kv_layer_bytes = [0] * len(base_layer_bytes)
    except ValueError:
        return 1
    layer_count = len(base_layer_bytes)
    if layer_count == 0:
        return 1

    free_mib = [int(value) for value in free_mib_string.split() if value]
    speeds = [float(value) for value in speed_string.split() if value]
    gpu_count = len(free_mib)
    if gpu_count < 2 or len(speeds) != gpu_count or any(speed <= 0 for speed in speeds):
        return 1

    def assign_layers(layer_bytes: list[int]) -> list[int] | None:
        reserve_bytes = reserve_mib * 1024 * 1024
        # scratch_bytes は Attention の作業領域 (KV の f16 展開・マスク)。各GPUの compute buffer に載る。
        budget = [max(0, mib * 1024 * 1024 - reserve_bytes - scratch_bytes) for mib in free_mib]
        fastest = max(range(gpu_count), key=lambda index: speeds[index])
        budget[fastest] -= other_bytes
        if budget[fastest] < 0:
            return None

        total_speed = sum(speeds)
        ideal = [layer_count * speed / total_speed for speed in speeds]
        prefix = [0]
        for size in layer_bytes:
            prefix.append(prefix[-1] + size)

        # layer splitはGPUごとに連続した層範囲を割り当てる。平均層サイズでは
        # MoE/SSM混在モデルの大きな層サイズ差を扱えないため、全境界を実サイズで探索する。
        states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
        for gpu_index in range(gpu_count):
            next_states: dict[int, tuple[float, list[int]]] = {}
            for start, (score, counts) in states.items():
                for end in range(start, layer_count + 1):
                    if prefix[end] - prefix[start] > budget[gpu_index]:
                        break
                    count = end - start
                    candidate_score = score + (count - ideal[gpu_index]) ** 2
                    current = next_states.get(end)
                    if current is None or candidate_score < current[0]:
                        next_states[end] = (candidate_score, counts + [count])
            states = next_states
            if not states:
                return None

        result = states.get(layer_count)
        return result[1] if result is not None else None

    max_cpu_moe = layer_count if moe_auto else 0
    for n_cpu_moe in range(max_cpu_moe + 1):
        layer_bytes = [
            base + (0 if index < n_cpu_moe else moe) + kv
            for index, (base, moe, kv) in enumerate(
                zip(base_layer_bytes, moe_layer_bytes, kv_layer_bytes)
            )
        ]
        assignments = assign_layers(layer_bytes)
        if assignments is None:
            continue
        output = [str(layer_count), ",".join(str(count) for count in assignments)]
        if moe_auto:
            cpu_moe_bytes = sum(moe_layer_bytes[:n_cpu_moe])
            output.extend((str(n_cpu_moe), str(cpu_moe_bytes)))
        print(" ".join(output))
        return 0
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("benchmark-speed")
    layer_bytes = commands.add_parser("layer-bytes")
    layer_bytes.add_argument("--exclude-moe", action="store_true")
    layer_bytes.add_argument("--split-moe", action="store_true")
    layer_bytes.add_argument("--split-ffn", action="store_true")
    commands.add_parser("moe-profile")
    commands.add_parser("kv-profile")
    tensor_split = commands.add_parser("tensor-split")
    tensor_split.add_argument("layer_bytes_file")
    tensor_split.add_argument("free_mib")
    tensor_split.add_argument("speeds")
    tensor_split.add_argument("reserve_mib", type=int)
    tensor_split.add_argument("--moe-auto", action="store_true")
    tensor_split.add_argument("--kv-tokens", type=int, default=0)
    tensor_split.add_argument("--swa-tokens", type=int, default=0)
    tensor_split.add_argument("--kv-bytes-per-element", type=float, default=0.0)
    tensor_split.add_argument("--scratch-bytes", type=int, default=0)
    args = parser.parse_args()

    if args.command == "benchmark-speed":
        print(read_benchmark_speed())
        return 0
    if args.command == "layer-bytes":
        return read_layer_bytes(args.exclude_moe, args.split_moe, args.split_ffn)
    if args.command == "moe-profile":
        return read_moe_profile()
    if args.command == "kv-profile":
        return read_kv_profile()
    return calculate_tensor_split(
        args.layer_bytes_file,
        args.free_mib,
        args.speeds,
        args.reserve_mib,
        args.moe_auto,
        args.kv_tokens,
        args.swa_tokens,
        args.kv_bytes_per_element,
        args.scratch_bytes,
    )


if __name__ == "__main__":
    raise SystemExit(main())
