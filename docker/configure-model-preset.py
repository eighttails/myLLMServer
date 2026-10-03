#!/usr/bin/env python3
"""モデル切替時に VRAM を見積もり、対象モデルのプリセットだけ再計算する。"""

from __future__ import annotations

import configparser
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.request import urlopen


MODEL_DIR = Path(os.environ.get("MODEL_DIR") or "/models")
PRESET_FILE = Path(os.environ.get("PRESET_FILE") or MODEL_DIR / ".models-preset.ini")
SECTION_DIR = Path(os.environ.get("PRESET_SECTION_DIR") or MODEL_DIR / ".models-preset.d")
ALIAS_FILE = Path(os.environ.get("MODEL_ALIAS_FILE") or MODEL_DIR / ".model-aliases.tsv")
UTIL = "/usr/local/bin/model-preset-utils.py"
MIB = 1024 * 1024
AUTO_CONTEXT_SIZE_MAX = 262144


def env(name: str, default: str) -> str:
    return os.environ.get(name) or default


CFG = {
    "MODEL_IDLE_SECONDS": env("MODEL_IDLE_SECONDS", "1800"),
    "CONTEXT_SIZE": os.environ.get("CONTEXT_SIZE", ""),
    "MAX_CONTEXT_SIZE": os.environ.get("MAX_CONTEXT_SIZE", ""),
    "MIN_CONTEXT_SIZE": env("MIN_CONTEXT_SIZE", "2048"),
    "CONTEXT_SIZE_STEP": env("CONTEXT_SIZE_STEP", "1024"),
    "N_GPU_LAYERS": env("N_GPU_LAYERS", "auto"),
    "MAX_PARALLEL_SLOTS": env("MAX_PARALLEL_SLOTS", "1"),
    "KV_CACHE_TYPE": os.environ.get("KV_CACHE_TYPE", ""),
    "TENSOR_SPLIT_MODE": env("TENSOR_SPLIT_MODE", "auto"),
    "SPLIT_MODE": env("SPLIT_MODE", "layer"),
    "VRAM_RESERVE_MIB": env("VRAM_RESERVE_MIB", "4096"),
    "MOE_CPU_OFFLOAD": env("MOE_CPU_OFFLOAD", "auto"),
    "MOE_ACTIVE_RATIO_THRESHOLD": env("MOE_ACTIVE_RATIO_THRESHOLD", "0.125"),
    "MOE_RAM_RESERVE_MIB": env("MOE_RAM_RESERVE_MIB", "8192"),
    "SPECULATIVE_DECODING": env("SPECULATIVE_DECODING", "off"),
    "UBATCH_SIZE": env("UBATCH_SIZE", "256"),
    "LLAMA_ROUTER_PORT": env("LLAMA_ROUTER_PORT", "11435"),
}


def log(message: str) -> None:
    print(f"[llama-wrapper] {message}", file=sys.stderr, flush=True)


def run(command: list[str], *, check: bool = True, input_text: str | None = None,
        extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command, input=input_text, text=True, capture_output=True, check=False,
        env={**os.environ, **(extra_env or {})},
    )
    if check and result.returncode:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")
    return result


def gguf_dump(path: Path, *, tensors: bool = False) -> str:
    command = ["gguf-dump"]
    if not tensors:
        command.append("--no-tensors")
    command.extend(["--json", "--json-array", str(path)])
    return run(command).stdout


def util(command: list[str], *, input_text: str = "") -> str:
    result = run([UTIL, *command], check=False, input_text=input_text)
    if result.returncode:
        return ""
    return result.stdout.strip()


def metadata(data: dict, suffix: str):
    values = data.get("metadata", {})
    if not isinstance(values, dict):
        return None
    for key, entry in values.items():
        if isinstance(key, str) and key.endswith(suffix):
            value = entry.get("value") if isinstance(entry, dict) else None
            return value.get("value") if isinstance(value, dict) else value
    return None


def parse_json_dump(content: str) -> dict:
    data = json.loads(content)
    if isinstance(data, list):
        data = next((entry for entry in data if isinstance(entry, dict)), {})
    if not isinstance(data, dict):
        return {}
    return data


def has_mtp_head(content: str) -> bool:
    tensors = parse_json_dump(content).get("tensors", {})
    if not isinstance(tensors, dict):
        return False
    return any(
        isinstance(name, str)
        and re.search(r"\.nextn\.(?:eh_proj|enorm|hnorm|shared_head_norm)\.weight$", name)
        for name in tensors
    )


def kv_params(model: Path) -> tuple[int, int, int, int, int]:
    result = util(["kv-profile"], input_text=gguf_dump(model))
    if not result:
        raise RuntimeError(f"Could not detect attention params for {model.name}")
    return tuple(map(int, result.split()))  # full, swa, window, max-full, max-swa


def bytes_per_element(cache_type: str) -> float:
    return {
        "f32": 4.0, "f16": 2.0, "bf16": 2.0, "q8_0": 1.0625,
        "q5_0": 0.6875, "q5_1": 0.6875,
        "q4_0": 0.5625, "q4_1": 0.5625, "iq4_nl": 0.5625,
    }.get(cache_type, 2.0)


def rounded_bytes(value: float) -> int:
    return math.floor(value + 0.5)


def swa_tokens(params: tuple[int, int, int, int, int], ctx: int, slots: int) -> int:
    return min(params[2] * slots + int(CFG["UBATCH_SIZE"]), ctx * slots)


def kv_cache_bytes(params: tuple[int, int, int, int, int], cache_type: str, ctx: int, slots: int = 1) -> int:
    full, swa, _, _, _ = params
    return rounded_bytes((full * ctx * slots + swa * swa_tokens(params, ctx, slots)) * bytes_per_element(cache_type))


def scratch_bytes(params: tuple[int, int, int, int, int], cache_type: str, ctx: int, slots: int) -> int:
    _, _, _, max_full, max_swa = params
    dequant = 0 if cache_type in {"f16", "bf16", "f32"} else 1
    elements = max(max_full * ctx * slots, max_swa * swa_tokens(params, ctx, slots))
    allowance = int(CFG["VRAM_RESERVE_MIB"]) * MIB // 2
    raw = (dequant * elements + int(CFG["UBATCH_SIZE"]) * ctx * slots) * 2 - allowance
    return max(0, rounded_bytes(raw))


def kv_vram_bytes(params: tuple[int, int, int, int, int], cache_type: str, ctx: int, slots: int = 1) -> int:
    return kv_cache_bytes(params, cache_type, ctx, slots) + gpu_count * scratch_bytes(params, cache_type, ctx, slots)


def gpu_ids() -> list[str]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return [item.strip() for item in visible.split(",") if item.strip()]
    result = run(["nvidia-smi", "--list-gpus"], check=False)
    count = sum(line.startswith("GPU ") for line in result.stdout.splitlines())
    return [str(index) for index in range(count)]


GPU_IDS = gpu_ids()
gpu_count = max(1, len(GPU_IDS))


def per_gpu_free_mib() -> list[int]:
    values: list[int] = []
    for gpu_id in GPU_IDS:
        result = run([
            "nvidia-smi", f"--id={gpu_id}", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits",
        ], check=False)
        try:
            values.append(int(result.stdout.strip().splitlines()[0]))
        except (ValueError, IndexError):
            continue
    return values


def usable_vram_bytes() -> int:
    reserve = int(CFG["VRAM_RESERVE_MIB"])
    return sum(max(0, memory - reserve) for memory in per_gpu_free_mib()) * MIB


def available_ram_bytes() -> int:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def run_benchmark(model: Path, gpu_id: str) -> float:
    result = run(
        ["/app/llama-bench", "-m", str(model), "-ngl", "99", "-p", "0", "-n", "32", "--output", "json"],
        check=False, extra_env={"CUDA_VISIBLE_DEVICES": gpu_id},
    )
    if result.returncode:
        return 0.0
    try:
        rows = json.loads(result.stdout)
        return float(next((row.get("avg_ts", 0) for row in rows if row.get("n_gen", 0) > 0), 0))
    except (json.JSONDecodeError, TypeError, ValueError):
        return 0.0


BENCH_CACHE = MODEL_DIR / ".gpu-tg-speed.tsv"
BENCH_MODELS = (
    "Llama-3.2-3B-Instruct-Q4_K_M.gguf",
    "gemma-4-12B-it-qat-UD-Q4_K_XL.gguf",
    "gpt-oss-20b-Q4_K_M.gguf",
    "Ornith-1.5-35B-Q4_K_M.gguf",
)


def per_gpu_speed() -> list[float]:
    if gpu_count < 2:
        return []
    if BENCH_CACHE.is_file():
        try:
            cached = [float(line) for line in BENCH_CACHE.read_text().splitlines() if re.fullmatch(r"[0-9.]+", line)]
            if len(cached) >= gpu_count and cached[0] > 0:
                return cached[:gpu_count]
            log("Previous per-GPU speed cache is invalid; re-benchmarking")
            BENCH_CACHE.unlink()
        except ValueError:
            BENCH_CACHE.unlink(missing_ok=True)
    benchmark = next((MODEL_DIR / name for name in BENCH_MODELS if (MODEL_DIR / name).is_file()), None)
    if benchmark is None:
        log("No single-GPU-fitting model found for speed benchmark; will fall back to --fit")
        return []
    log(f"Using {benchmark} as benchmark model for per-GPU speed")
    speeds = [run_benchmark(benchmark, gpu_id) for gpu_id in GPU_IDS]
    BENCH_CACHE.write_text("".join(f"{speed:g}\n" for speed in speeds))
    if not speeds or speeds[0] <= 0:
        return []
    return speeds


def layer_bytes(model: Path, mode: str) -> str:
    args = ["layer-bytes"]
    if mode == "split":
        args.append("--split-moe")
    elif mode == "split-ffn":
        args.extend(["--split-moe", "--split-ffn"])
    return util(args, input_text=gguf_dump(model, tensors=True))


def auxiliary_preset_options(alias: str) -> dict[str, str]:
    section_file = SECTION_DIR / f"{alias}.ini"
    if not section_file.is_file():
        return {}
    preset = configparser.ConfigParser(interpolation=None)
    preset.read(section_file, encoding="utf-8")
    if not preset.has_section(alias):
        return {}
    return {
        option: preset.get(alias, option)
        for option in ("model-draft", "mmproj")
        if preset.has_option(alias, option)
    }


def single_gpu_moe_plan(
    layer_data: str,
    usable_vram: int,
    params: tuple[int, int, int, int, int],
    cache_type: str,
    context: int,
    max_slots: int,
    available_ram: int,
    ram_reserve: int,
    upgrades: list[str],
) -> tuple[int, int, int, str] | None:
    """Choose the smallest CPU expert-layer prefix that fits alongside KV on one GPU."""
    try:
        lines = [line.split() for line in layer_data.splitlines() if line.strip()]
        other_bytes = int(lines[0][0])
        rows = [tuple(map(int, line)) for line in lines[1:]]
    except (IndexError, ValueError):
        return None
    if not rows or any(len(row) != 4 for row in rows):
        return None

    base_bytes = sum(row[0] for row in rows)
    expert_by_layer = [row[1] for row in rows]
    ram_budget = max(0, available_ram - ram_reserve)
    offloaded_bytes = 0

    for cpu_layers in range(len(rows) + 1):
        if cpu_layers:
            offloaded_bytes += expert_by_layer[cpu_layers - 1]
            if offloaded_bytes > ram_budget:
                break

        gpu_weight_bytes = other_bytes + base_bytes + sum(expert_by_layer[cpu_layers:])
        selected_slots = next(
            (
                slots
                for slots in range(max_slots, 0, -1)
                if gpu_weight_bytes + kv_vram_bytes(params, cache_type, context, slots) <= usable_vram
            ),
            0,
        )
        if not selected_slots:
            continue

        selected_cache = cache_type
        for candidate in upgrades:
            if gpu_weight_bytes + kv_vram_bytes(params, candidate, context, selected_slots) <= usable_vram:
                selected_cache = candidate
                break
        return cpu_layers, offloaded_bytes, selected_slots, selected_cache
    return None


def resolve_alias(requested: str) -> tuple[str, str]:
    if not ALIAS_FILE.is_file():
        raise RuntimeError(f"model alias file not found: {ALIAS_FILE}")
    for line in ALIAS_FILE.read_text(encoding="utf-8").splitlines():
        columns = line.split("\t")
        if len(columns) >= 2 and requested in columns[:2]:
            return columns[0], columns[1]
    raise RuntimeError(f"unknown model: {requested}")


def acquire_lock(lock_dir: Path) -> None:
    deadline = time.monotonic() + 600
    while True:
        try:
            lock_dir.mkdir()
            (lock_dir / "pid").write_text(f"{os.getpid()}\n")
            return
        except FileExistsError:
            try:
                pid = int((lock_dir / "pid").read_text().strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                log(f"removing stale preset lock (pid={locals().get('pid', 'unknown')})")
                shutil.rmtree(lock_dir, ignore_errors=True)
                continue
            if time.monotonic() > deadline:
                raise RuntimeError(f"timed out waiting for preset lock held by pid {pid}")
            time.sleep(0.1)


def render_preset() -> None:
    content = f"[*]\nfit = on\nsplit-mode = {CFG['SPLIT_MODE']}\n\n"
    for section in sorted(SECTION_DIR.glob("*.ini")):
        content += section.read_text(encoding="utf-8")
    temporary = PRESET_FILE.with_name(f".{PRESET_FILE.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(PRESET_FILE)


def configure(requested_model: str) -> None:
    alias, filename = resolve_alias(requested_model)
    model_file = MODEL_DIR / filename
    if not model_file.is_file():
        raise RuntimeError(f"model file not found: {model_file}")
    lock_dir = MODEL_DIR / ".preset-configure.lock"
    acquire_lock(lock_dir)
    try:
        configure_locked(alias, filename, model_file)
    finally:
        shutil.rmtree(lock_dir, ignore_errors=True)


def configure_locked(alias: str, filename: str, model_file: Path) -> None:
    auxiliary_options = auxiliary_preset_options(alias)
    auxiliary_model_bytes = sum(
        Path(path).stat().st_size
        for path in auxiliary_options.values()
        if Path(path).is_file()
    )
    model_data = parse_json_dump(gguf_dump(model_file))
    model_context = int(metadata(model_data, ".context_length") or 4096)
    context = int(CFG["CONTEXT_SIZE"] or min(model_context, AUTO_CONTEXT_SIZE_MAX))
    if not CFG["CONTEXT_SIZE"] and metadata(model_data, ".context_length"):
        if model_context > AUTO_CONTEXT_SIZE_MAX:
            log(f"Capping automatically detected ctx-size for {filename} at {AUTO_CONTEXT_SIZE_MAX}")
        else:
            log(f"Detected context_length for {filename}: {context}")
    if CFG["MAX_CONTEXT_SIZE"] and context > int(CFG["MAX_CONTEXT_SIZE"]):
        log(f"Capping ctx-size for {filename} from {context} to MAX_CONTEXT_SIZE={CFG['MAX_CONTEXT_SIZE']}")
        context = int(CFG["MAX_CONTEXT_SIZE"])

    tensor_dump = gguf_dump(model_file, tensors=True)
    moe_profile_text = util(["moe-profile"], input_text=tensor_dump)
    internal_mtp = (
        CFG["SPECULATIVE_DECODING"] == "on"
        and "model-draft" not in auxiliary_options
        and has_mtp_head(tensor_dump)
    )
    del tensor_dump
    moe_profile = [int(item) for item in moe_profile_text.split()] if moe_profile_text else []
    expert_count = moe_profile[1] if len(moe_profile) > 1 else 0
    expert_used = moe_profile[2] if len(moe_profile) > 2 else 0
    expert_bytes = moe_profile[4] if len(moe_profile) > 4 else 0
    moe_auto = False
    should_offload = expert_bytes > 0 and (
        CFG["MOE_CPU_OFFLOAD"] == "all"
        or (CFG["MOE_CPU_OFFLOAD"] == "auto" and expert_count > 0 and expert_used > 0
            and expert_used / expert_count <= float(CFG["MOE_ACTIVE_RATIO_THRESHOLD"]))
    )
    ram_available = available_ram_bytes() if should_offload else 0
    ram_reserve = int(CFG["MOE_RAM_RESERVE_MIB"]) * MIB
    if should_offload:
        required = expert_bytes + ram_reserve
        if ram_available >= required:
            moe_auto = True
            ratio = 100 * expert_used / expert_count if expert_count else 0
            log(f"{filename}: MoE uses {expert_used}/{expert_count} experts ({ratio:.1f}%); RAM is available for measured CPU-placement planning")
            log(f"{filename}: up to {expert_bytes // MIB} MiB of expert weights can be moved to CPU")
        elif gpu_count == 1:
            ratio = 100 * expert_used / expert_count if expert_count else 0
            log(f"{filename}: MoE uses {expert_used}/{expert_count} experts ({ratio:.1f}%); checking partial CPU offload against available RAM")
        else:
            log(f"{filename}: skipping MoE CPU offload because available host RAM is below expert weights plus reserve")

    kv_base = CFG["KV_CACHE_TYPE"] or "q4_0"
    upgrades: list[str] = []
    cache_type = kv_base
    parallel = 1
    model_file_bytes = model_file.stat().st_size
    try:
        params = kv_params(model_file)
    except (RuntimeError, ValueError):
        params = None
    usable = usable_vram_bytes()
    planning_usable = max(0, usable - auxiliary_model_bytes)
    fit_fallback = params is None or planning_usable <= 0
    if params is None:
        log(f"Could not detect attention params for {filename}; relying on llama-server's --fit auto-adjustment")
    elif usable <= 0:
        log(f"Could not detect free VRAM for {filename}; relying on llama-server's --fit auto-adjustment")

    if not fit_fallback and params is not None:
        one_slot = kv_vram_bytes(params, kv_base, context)
        if one_slot > planning_usable:
            low, high = 0, context // int(CFG["CONTEXT_SIZE_STEP"])
            while low < high:
                mid = (low + high + 1) // 2
                if kv_vram_bytes(params, kv_base, mid * int(CFG["CONTEXT_SIZE_STEP"])) <= planning_usable:
                    low = mid
                else:
                    high = mid - 1
            fitted = low * int(CFG["CONTEXT_SIZE_STEP"])
            if fitted >= int(CFG["MIN_CONTEXT_SIZE"]):
                log(f"{filename}: shrinking ctx-size from {context} to {fitted} to fit {kv_base} KV cache")
                context = fitted
                one_slot = kv_vram_bytes(params, kv_base, context)
            else:
                log(f"{filename}: cannot fit {kv_base} KV cache even at MIN_CONTEXT_SIZE; relying on --fit")
                fit_fallback = True
        if not fit_fallback:
            remaining = planning_usable - one_slot
            if model_file_bytes <= remaining and one_slot:
                parallel = min(
                    int(CFG["MAX_PARALLEL_SLOTS"]),
                    max(1, 1 + (remaining - model_file_bytes) // one_slot),
                )
                for candidate in upgrades:
                    if model_file_bytes + kv_vram_bytes(params, candidate, context, parallel) <= usable:
                        cache_type = candidate
                        break
            elif model_file_bytes > remaining:
                log(f"{filename}: model weights do not fit next to the {kv_base} KV cache; part will remain on CPU")
            if not (gpu_count == 1 and should_offload and CFG["SPLIT_MODE"] == "layer"):
                log(f"Estimated plan for {filename}: ctx-size={context}, KV cache={cache_type}, slots={parallel}")

    ctx_per_slot = context
    tensor_split = ""
    gpu_layers = ""
    cpu_layers = 0
    offload_kind = ""
    explicit_gpu_placement = False
    if fit_fallback:
        parallel = 1
        cache_type = kv_base
    elif CFG["SPLIT_MODE"] != "layer":
        log(f"SPLIT_MODE={CFG['SPLIT_MODE']}; skipping layer-based tensor-split and relying on --fit")
    elif CFG["TENSOR_SPLIT_MODE"] == "auto" and gpu_count >= 2:
        speeds = per_gpu_speed()
        if speeds:
            log("Per-GPU generation speed (tok/s): " + " ".join(f"{speed:g}" for speed in speeds))
            offload = False
            if expert_bytes:
                layer_mode = "split"
                offload = moe_auto
                offload_kind = "moe" if offload else ""
            else:
                layer_mode = "split-ffn"
            layer_data = layer_bytes(model_file, layer_mode)
            if layer_data:
                free_mib = per_gpu_free_mib()
                if auxiliary_model_bytes and free_mib:
                    fastest_gpu = max(range(len(speeds)), key=lambda index: speeds[index])
                    if fastest_gpu < len(free_mib):
                        free_mib[fastest_gpu] = max(
                            0, free_mib[fastest_gpu] - (auxiliary_model_bytes + MIB - 1) // MIB
                        )
                def try_split(kind: str, slots: int, ctx: int = ctx_per_slot):
                    # Tensor-split estimates use total KV tokens across the configured slots.
                    tokens = ctx * slots
                    params_with_slots = params
                    detailed = layer_data
                    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8") as handle:
                        handle.write(detailed)
                        handle.flush()
                        args = [
                            "tensor-split", handle.name, " ".join(map(str, free_mib)),
                            " ".join(f"{speed:g}" for speed in speeds), CFG["VRAM_RESERVE_MIB"],
                        ]
                        if offload:
                            args.append("--moe-auto")
                        args.extend([
                            "--kv-tokens", str(tokens),
                            "--swa-tokens", str(swa_tokens(params_with_slots, ctx, slots)),
                            "--kv-bytes-per-element", str(bytes_per_element(kind)),
                            "--scratch-bytes", str(scratch_bytes(params_with_slots, kind, ctx, slots)),
                        ])
                        result = util(args)
                    return result.split() if result else None

                result = try_split(kv_base, 1)
                if not result and offload:
                    low = (int(CFG["MIN_CONTEXT_SIZE"]) + int(CFG["CONTEXT_SIZE_STEP"]) - 1) // int(CFG["CONTEXT_SIZE_STEP"])
                    high = ctx_per_slot // int(CFG["CONTEXT_SIZE_STEP"])
                    best_ctx = 0
                    while low <= high:
                        mid = (low + high) // 2
                        trial_ctx = mid * int(CFG["CONTEXT_SIZE_STEP"])
                        trial = try_split(kv_base, 1, trial_ctx)
                        if trial:
                            best_ctx, result = trial_ctx, trial
                            low = mid + 1
                        else:
                            high = mid - 1
                    if best_ctx:
                        log(f"{filename}: shrinking ctx-size to {best_ctx} to keep its KV cache in VRAM")
                        ctx_per_slot = best_ctx
                if not result:
                    log(f"Could not fit {kv_base} KV cache with a manual tensor-split for {filename}; relying on --fit")
                    parallel, cache_type = 1, kv_base
                else:
                    min_offload = int(result[2]) if len(result) > 2 else 0
                    for slots in range(int(CFG["MAX_PARALLEL_SLOTS"]), 1, -1):
                        trial = try_split(kv_base, slots)
                        if trial and (int(trial[2]) if len(trial) > 2 else 0) == min_offload:
                            parallel, result = slots, trial
                            break
                    for candidate in upgrades:
                        trial = try_split(candidate, parallel)
                        if trial and (int(trial[2]) if len(trial) > 2 else 0) == min_offload:
                            cache_type, result = candidate, trial
                            break
                    gpu_layers, tensor_split = result[0], result[1]
                    cpu_layers = int(result[2]) if len(result) > 2 else 0
                    if cpu_layers:
                        amount = int(result[3]) // MIB if len(result) > 3 else 0
                        log(f"{filename}: keeping {offload_kind} weights of first {cpu_layers} layers ({amount} MiB) on CPU")
                    log(f"Calculated tensor-split for {filename}: n-gpu-layers={gpu_layers} tensor-split={tensor_split}")
            else:
                log(f"Could not determine per-layer tensor sizes for {filename}; falling back to --fit")
        else:
            log("Could not benchmark per-GPU speed; tensor-split will fall back to --fit")
    elif gpu_count == 1 and should_offload and params is not None:
        layer_data = layer_bytes(model_file, "split")
        plan = single_gpu_moe_plan(
            layer_data,
            planning_usable,
            params,
            kv_base,
            ctx_per_slot,
            int(CFG["MAX_PARALLEL_SLOTS"]),
            ram_available,
            ram_reserve,
            upgrades,
        )
        if plan is None:
            log(f"Could not fit MoE weights and KV cache within GPU/RAM budgets for {filename}; relying on --fit")
        else:
            cpu_layers, cpu_weight_bytes, parallel, cache_type = plan
            offload_kind = "moe" if cpu_layers else ""
            explicit_gpu_placement = True
            if cpu_layers:
                log(
                    f"{filename}: keeping MoE expert weights of first {cpu_layers} layers "
                    f"({cpu_weight_bytes // MIB} MiB) on CPU"
                )
            log(f"Estimated single-GPU plan for {filename}: ctx-size={context}, KV cache={cache_type}, slots={parallel}")
    elif (
        gpu_count == 1
        and params is not None
        and CFG["SPLIT_MODE"] == "layer"
        and model_file_bytes + auxiliary_model_bytes
        + kv_vram_bytes(params, cache_type, ctx_per_slot, parallel) <= usable
    ):
        gpu_layers = "all"
        explicit_gpu_placement = True
        log(f"Keeping all model weights on the GPU for decode speed: {filename}")
    else:
        log(f"TENSOR_SPLIT_MODE={CFG['TENSOR_SPLIT_MODE']} or single GPU; relying on llama-server's --fit")

    context_total = ctx_per_slot * parallel
    log(f"Auto-selected KV cache type for {filename}: {cache_type}")
    if parallel > 1:
        log(f"Allocating {parallel} parallel slots for {filename} (per-slot ctx-size={ctx_per_slot}, total ctx-size={context_total})")
    else:
        log(f"Serializing requests for {filename} (1 slot, ctx-size={ctx_per_slot})")
    lines = [
        f"[{alias}]",
        f"model = {model_file}",
        f"ctx-size = {context_total}",
        f"parallel = {parallel}",
        f"kv-unified-per-slot = {ctx_per_slot}",
        f"sleep-idle-seconds = {CFG['MODEL_IDLE_SECONDS']}",
    ]
    if tensor_split:
        lines.extend(["fit = off", f"n-gpu-layers = {gpu_layers}", f"tensor-split = {tensor_split}"])
    elif explicit_gpu_placement:
        lines.extend(["fit = off", "n-gpu-layers = all"])
    else:
        lines.append(f"n-gpu-layers = {CFG['N_GPU_LAYERS']}")
    if cpu_layers:
        option = "n-cpu-ffn" if offload_kind == "ffn" else "n-cpu-moe"
        lines.append(f"{option} = {cpu_layers}")
    lines.extend(f"{option} = {value}" for option, value in auxiliary_options.items())
    if CFG["SPECULATIVE_DECODING"] == "on":
        if "model-draft" in auxiliary_options:
            lines.append("spec-type = draft-simple")
            lines.extend(["spec-draft-type-k = q4_0", "spec-draft-type-v = q4_0"])
            log(f"Enabling speculative decoding with external draft model for {filename}")
        elif internal_mtp:
            lines.append("spec-type = draft-mtp")
            lines.extend(["spec-draft-type-k = q4_0", "spec-draft-type-v = q4_0"])
            log(f"Enabling built-in MTP speculative decoding for {filename}")
    lines.extend([f"cache-type-k = {cache_type}", f"cache-type-v = {cache_type}", ""])
    temporary = SECTION_DIR / f".{alias}.ini.tmp-{os.getpid()}"
    SECTION_DIR.mkdir(parents=True, exist_ok=True)
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(SECTION_DIR / f"{alias}.ini")
    render_preset()
    try:
        with urlopen(f"http://127.0.0.1:{CFG['LLAMA_ROUTER_PORT']}/models?reload=1", timeout=10):
            pass
    except OSError:
        pass
    log(f"Updated lazy preset for {alias}")


def main() -> int:
    if len(sys.argv) != 2 or not sys.argv[1]:
        print("[llama-wrapper] error: model name is required", file=sys.stderr)
        return 1
    try:
        configure(sys.argv[1])
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"[llama-wrapper] error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
