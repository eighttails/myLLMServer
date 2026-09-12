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


def read_layer_bytes(exclude_moe: bool = False, split_moe: bool = False) -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1

    tensors = data.get("tensors", {})
    if not isinstance(tensors, dict):
        return 1
    head_count_kv: int | list[int] = 0
    if split_moe:
        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            for key, entry in metadata.items():
                if not isinstance(key, str) or not key.endswith(".attention.head_count_kv"):
                    continue
                value = entry.get("value") if isinstance(entry, dict) else None
                if isinstance(value, dict):
                    value = value.get("value")
                if isinstance(value, list) and all(isinstance(item, int) for item in value):
                    head_count_kv = value
                elif isinstance(value, int):
                    head_count_kv = value
                break

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
        size = tensor_bytes(shape, tensor_type)
        match = re.match(r"^blk\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            if split_moe and is_moe:
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
            kv_heads = (
                head_count_kv[index]
                if isinstance(head_count_kv, list) and index < len(head_count_kv)
                else head_count_kv
                if isinstance(head_count_kv, int)
                else 0
            )
            print(layer_bytes[index], layer_moe_bytes.get(index, 0), kv_heads)
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
    kv_bytes_per_head: int = 0,
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
            if any(len(values) not in (2, 3) for values in parsed_layers):
                return 1
            base_layer_bytes = [values[0] for values in parsed_layers]
            moe_layer_bytes = [values[1] for values in parsed_layers]
            kv_layer_bytes = [
                (values[2] if len(values) == 3 else 0) * kv_bytes_per_head
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
        budget = [max(0, mib * 1024 * 1024 - reserve_bytes) for mib in free_mib]
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
    commands.add_parser("moe-profile")
    tensor_split = commands.add_parser("tensor-split")
    tensor_split.add_argument("layer_bytes_file")
    tensor_split.add_argument("free_mib")
    tensor_split.add_argument("speeds")
    tensor_split.add_argument("reserve_mib", type=int)
    tensor_split.add_argument("--moe-auto", action="store_true")
    tensor_split.add_argument("--kv-bytes-per-head", type=int, default=0)
    args = parser.parse_args()

    if args.command == "benchmark-speed":
        print(read_benchmark_speed())
        return 0
    if args.command == "layer-bytes":
        return read_layer_bytes(args.exclude_moe, args.split_moe)
    if args.command == "moe-profile":
        return read_moe_profile()
    return calculate_tensor_split(
        args.layer_bytes_file,
        args.free_mib,
        args.speeds,
        args.reserve_mib,
        args.moe_auto,
        args.kv_bytes_per_head,
    )


if __name__ == "__main__":
    raise SystemExit(main())
