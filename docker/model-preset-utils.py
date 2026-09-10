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
    "IQ4_NL": (32, 18),
    "IQ4_XS": (256, 136),
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
    return (count // block_size) * type_bytes


def read_layer_bytes() -> int:
    try:
        data = json.load(sys.stdin)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return 1

    tensors = data.get("tensors", {})
    if not isinstance(tensors, dict):
        return 1

    layer_bytes: dict[int, int] = {}
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
        size = tensor_bytes(shape, tensor_type)
        match = re.match(r"^blk\.(\d+)\.", name)
        if match:
            index = int(match.group(1))
            layer_bytes[index] = layer_bytes.get(index, 0) + size
        else:
            other_bytes += size

    if not layer_bytes:
        return 1

    print(other_bytes)
    for index in sorted(layer_bytes):
        print(layer_bytes[index])
    return 0


def calculate_tensor_split(
    layer_bytes_file: str, free_mib_string: str, speed_string: str, reserve_mib: int
) -> int:
    with open(layer_bytes_file, encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return 1

    other_bytes = int(lines[0])
    layer_bytes = [int(value) for value in lines[1:]]
    layer_count = len(layer_bytes)
    if layer_count == 0:
        return 1

    free_mib = [int(value) for value in free_mib_string.split() if value]
    speeds = [float(value) for value in speed_string.split() if value]
    gpu_count = len(free_mib)
    if gpu_count < 2 or len(speeds) != gpu_count or any(speed <= 0 for speed in speeds):
        return 1

    reserve_bytes = reserve_mib * 1024 * 1024
    budget = [max(0, mib * 1024 * 1024 - reserve_bytes) for mib in free_mib]
    fastest = max(range(gpu_count), key=lambda index: speeds[index])
    budget[fastest] -= other_bytes
    if budget[fastest] < 0:
        return 1

    total_speed = sum(speeds)
    ideal = [layer_count * speed / total_speed for speed in speeds]
    average_layer_bytes = sum(layer_bytes) / layer_count
    max_layers_by_budget = [
        int(value // average_layer_bytes) if average_layer_bytes > 0 else layer_count
        for value in budget
    ]
    assignments = [
        min(round(ideal[index]), max_layers_by_budget[index]) for index in range(gpu_count)
    ]

    def total_assigned() -> int:
        return sum(assignments)

    fastest_to_slowest = sorted(range(gpu_count), key=lambda index: -speeds[index])
    guard = 0
    while total_assigned() < layer_count and guard < layer_count * 2:
        guard += 1
        for index in fastest_to_slowest:
            if assignments[index] < max_layers_by_budget[index]:
                assignments[index] += 1
                if total_assigned() >= layer_count:
                    break
        else:
            break

    slowest_to_fastest = sorted(range(gpu_count), key=lambda index: speeds[index])
    guard = 0
    while total_assigned() > layer_count and guard < layer_count * 2:
        guard += 1
        for index in slowest_to_fastest:
            if assignments[index] > 0:
                assignments[index] -= 1
                break
        else:
            break

    if total_assigned() != layer_count or any(count < 0 for count in assignments):
        return 1

    print(layer_count, ",".join(str(count) for count in assignments))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("benchmark-speed")
    commands.add_parser("layer-bytes")
    tensor_split = commands.add_parser("tensor-split")
    tensor_split.add_argument("layer_bytes_file")
    tensor_split.add_argument("free_mib")
    tensor_split.add_argument("speeds")
    tensor_split.add_argument("reserve_mib", type=int)
    args = parser.parse_args()

    if args.command == "benchmark-speed":
        print(read_benchmark_speed())
        return 0
    if args.command == "layer-bytes":
        return read_layer_bytes()
    return calculate_tensor_split(
        args.layer_bytes_file, args.free_mib, args.speeds, args.reserve_mib
    )


if __name__ == "__main__":
    raise SystemExit(main())
