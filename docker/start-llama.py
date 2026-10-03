#!/usr/bin/env python3
"""コンテナ内の llama-server と遅延ロードプロキシを起動する。"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


def log(message: str, *, error: bool = False) -> None:
    stream = sys.stderr if error else sys.stdout
    print(f"[llama-wrapper] {message}", file=stream, flush=True)


def fail(message: str) -> None:
    log(f"error: {message}", error=True)
    raise SystemExit(1)


def setting(name: str, default: str) -> str:
    return os.environ.get(name) or default


def validate() -> dict[str, str]:
    config = {
        "MODEL_DIR": setting("MODEL_DIR", "/models"),
        "MODEL_IDLE_SECONDS": setting("MODEL_IDLE_SECONDS", "1800"),
        "HOST": setting("HOST", "0.0.0.0"),
        "PORT": setting("PORT", "11434"),
        "LLAMA_ROUTER_PORT": setting("LLAMA_ROUTER_PORT", ""),
        "CONTEXT_SIZE": os.environ.get("CONTEXT_SIZE", ""),
        "MAX_CONTEXT_SIZE": os.environ.get("MAX_CONTEXT_SIZE", ""),
        "MIN_CONTEXT_SIZE": setting("MIN_CONTEXT_SIZE", "2048"),
        "CONTEXT_SIZE_STEP": setting("CONTEXT_SIZE_STEP", "1024"),
        "N_GPU_LAYERS": setting("N_GPU_LAYERS", "auto"),
        "MODELS_MAX": setting("MODELS_MAX", "1"),
        "MAX_PARALLEL_SLOTS": setting("MAX_PARALLEL_SLOTS", "1"),
        "FLASH_ATTN": setting("FLASH_ATTN", "on"),
        "BATCH_SIZE": setting("BATCH_SIZE", "1024"),
        "UBATCH_SIZE": setting("UBATCH_SIZE", "256"),
        "VRAM_RESERVE_MIB": setting("VRAM_RESERVE_MIB", "4096"),
        "MOE_CPU_OFFLOAD": setting("MOE_CPU_OFFLOAD", "auto"),
        "MOE_ACTIVE_RATIO_THRESHOLD": setting("MOE_ACTIVE_RATIO_THRESHOLD", "0.125"),
        "MOE_RAM_RESERVE_MIB": setting("MOE_RAM_RESERVE_MIB", "8192"),
        "TENSOR_SPLIT_MODE": setting("TENSOR_SPLIT_MODE", "auto"),
        "SPLIT_MODE": setting("SPLIT_MODE", "layer"),
        "KV_CACHE_TYPE": os.environ.get("KV_CACHE_TYPE", ""),
        "SPECULATIVE_DECODING": setting("SPECULATIVE_DECODING", "off"),
        "THINKING_MODE": setting("THINKING_MODE", "auto"),
        "GENERATION_LOOP_DETECTION": setting("GENERATION_LOOP_DETECTION", "on"),
        "GENERATION_LOOP_WINDOW_CHARS": setting("GENERATION_LOOP_WINDOW_CHARS", "16384"),
        "GENERATION_LOOP_MIN_PATTERN_CHARS": setting("GENERATION_LOOP_MIN_PATTERN_CHARS", "64"),
        "GENERATION_LOOP_MAX_PATTERN_CHARS": setting("GENERATION_LOOP_MAX_PATTERN_CHARS", "2048"),
        "GENERATION_LOOP_REPEAT_COUNT": setting("GENERATION_LOOP_REPEAT_COUNT", "3"),
        "GENERATION_LOOP_MIN_REPEATED_CHARS": setting("GENERATION_LOOP_MIN_REPEATED_CHARS", "256"),
        "GENERATION_LOOP_LINE_REPEAT_COUNT": setting("GENERATION_LOOP_LINE_REPEAT_COUNT", "6"),
        "TOOL_LOOP_DETECTION": setting("TOOL_LOOP_DETECTION", "on"),
        "TOOL_LOOP_REPEAT_COUNT": setting("TOOL_LOOP_REPEAT_COUNT", "3"),
        "TOOL_LOOP_MAX_CYCLE_LENGTH": setting("TOOL_LOOP_MAX_CYCLE_LENGTH", "4"),
    }
    if not config["LLAMA_ROUTER_PORT"]:
        config["LLAMA_ROUTER_PORT"] = str(int(config["PORT"]) + 1) if config["PORT"].isdigit() else ""

    integer_fields = ("MODEL_IDLE_SECONDS", "PORT", "LLAMA_ROUTER_PORT", "MODELS_MAX", "MOE_RAM_RESERVE_MIB")
    for name in integer_fields:
        if not config[name].isdigit():
            fail(f"{name} must be an integer")
    for name in ("CONTEXT_SIZE", "MAX_CONTEXT_SIZE"):
        if config[name] and not config[name].isdigit():
            fail(f"{name} must be an integer")
    for name in ("MIN_CONTEXT_SIZE", "CONTEXT_SIZE_STEP", "MAX_PARALLEL_SLOTS", "BATCH_SIZE", "VRAM_RESERVE_MIB"):
        if not config[name].isdigit() or int(config[name]) < 1:
            fail(f"{name} must be a positive integer")
    if int(config["UBATCH_SIZE"]) < 1 or int(config["UBATCH_SIZE"]) > int(config["BATCH_SIZE"]):
        fail("UBATCH_SIZE must be a positive integer no greater than BATCH_SIZE")
    if config["LLAMA_ROUTER_PORT"] == config["PORT"]:
        fail("LLAMA_ROUTER_PORT must be different from PORT")
    if config["N_GPU_LAYERS"] not in {"auto", "all"}:
        try:
            int(config["N_GPU_LAYERS"])
        except ValueError:
            fail("N_GPU_LAYERS must be an integer, 'auto', or 'all'")
    choices = {
        "FLASH_ATTN": {"on", "off", "auto"},
        "MOE_CPU_OFFLOAD": {"auto", "off", "all"},
        "TENSOR_SPLIT_MODE": {"auto", "off"},
        "SPLIT_MODE": {"none", "layer", "row", "tensor"},
        "SPECULATIVE_DECODING": {"on", "off"},
        "THINKING_MODE": {"on", "off", "auto"},
    }
    for name, allowed in choices.items():
        if config[name] not in allowed:
            fail(f"{name} must be one of: {', '.join(sorted(allowed))}")
    try:
        moe_threshold = float(config["MOE_ACTIVE_RATIO_THRESHOLD"])
    except ValueError:
        fail("MOE_ACTIVE_RATIO_THRESHOLD must be a number")
    if not 0 < moe_threshold <= 1:
        fail("MOE_ACTIVE_RATIO_THRESHOLD must be greater than 0 and no greater than 1")
    numeric_rules = {
        "GENERATION_LOOP_WINDOW_CHARS": 1,
        "GENERATION_LOOP_MIN_PATTERN_CHARS": 1,
        "GENERATION_LOOP_MAX_PATTERN_CHARS": 1,
        "GENERATION_LOOP_REPEAT_COUNT": 2,
        "GENERATION_LOOP_MIN_REPEATED_CHARS": 1,
        "GENERATION_LOOP_LINE_REPEAT_COUNT": 2,
        "TOOL_LOOP_REPEAT_COUNT": 2,
        "TOOL_LOOP_MAX_CYCLE_LENGTH": 1,
    }
    for name, minimum in numeric_rules.items():
        if not config[name].isdigit() or int(config[name]) < minimum:
            fail(f"{name} must be an integer no less than {minimum}")
    if int(config["GENERATION_LOOP_WINDOW_CHARS"]) < int(config["GENERATION_LOOP_MIN_REPEATED_CHARS"]):
        fail("GENERATION_LOOP_WINDOW_CHARS must be no less than GENERATION_LOOP_MIN_REPEATED_CHARS")
    if int(config["GENERATION_LOOP_WINDOW_CHARS"]) < (
        int(config["GENERATION_LOOP_MAX_PATTERN_CHARS"]) * int(config["GENERATION_LOOP_REPEAT_COUNT"])
    ):
        fail("GENERATION_LOOP_WINDOW_CHARS is too small for the configured pattern size and repeat count")
    for name in ("GENERATION_LOOP_DETECTION", "TOOL_LOOP_DETECTION"):
        if config[name] not in {"on", "off"}:
            fail(f"{name} must be 'on' or 'off'")
    kv_types = {"", "f32", "f16", "bf16", "q8_0", "q4_0", "q4_1", "iq4_nl", "q5_0", "q5_1"}
    if config["KV_CACHE_TYPE"] not in kv_types:
        fail("KV_CACHE_TYPE must be one of: f32 f16 bf16 q8_0 q4_0 q4_1 iq4_nl q5_0 q5_1")
    return config


def terminate(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def main() -> int:
    config = validate()
    model_dir = Path(config["MODEL_DIR"])
    config.update({
        "MODEL_LIST_FILE": setting("MODEL_LIST_FILE", str(model_dir / "model_list.yml")),
        "PRESET_FILE": str(model_dir / ".models-preset.ini"),
        "PRESET_SECTION_DIR": str(model_dir / ".models-preset.d"),
        "MODEL_ALIAS_FILE": str(model_dir / ".model-aliases.tsv"),
        "LLAMA_ROUTER_URL": f"http://127.0.0.1:{config['LLAMA_ROUTER_PORT']}",
    })
    os.environ.update(config)
    os.environ["no_proxy"] = ",".join(filter(None, [os.environ.get("no_proxy", ""), "127.0.0.1", "localhost"]))
    os.environ["NO_PROXY"] = ",".join(filter(None, [os.environ.get("NO_PROXY", ""), "127.0.0.1", "localhost"]))

    sync_script = Path("/usr/local/bin/sync-model.py")
    try:
        subprocess.run([sys.executable, str(sync_script)], check=True)
    except subprocess.CalledProcessError as error:
        fail(f"initial model sync failed with exit code {error.returncode}")
    aliases = Path(config["MODEL_ALIAS_FILE"]).read_text(encoding="utf-8").splitlines()
    available_models = " ".join(line.split("\t", 1)[0] for line in aliases if line)
    log(f"Starting OpenAI-compatible llama-server router on internal port {config['LLAMA_ROUTER_PORT']}")
    log(f"Available models: {available_models} (models-max={config['MODELS_MAX']})")

    command = [
        "llama-server",
        "--models-preset", config["PRESET_FILE"],
        "--models-max", config["MODELS_MAX"],
        "--flash-attn", config["FLASH_ATTN"],
        "--batch-size", setting("BATCH_SIZE", "1024"),
        "--ubatch-size", setting("UBATCH_SIZE", "256"),
        "--kv-unified",
        "--fit-target", config["VRAM_RESERVE_MIB"],
        "--host", "127.0.0.1",
        "--port", config["LLAMA_ROUTER_PORT"],
    ]
    if config["THINKING_MODE"] in {"on", "off"}:
        command.extend(["--reasoning", config["THINKING_MODE"]])

    router: subprocess.Popen[bytes] | None = None
    proxy: subprocess.Popen[bytes] | None = None
    def handle_shutdown(signum: int, _frame) -> None:
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    try:
        router = subprocess.Popen(command)
        log(f"Waiting for llama-server router at {config['LLAMA_ROUTER_URL']}")
        for _ in range(60):
            if router.poll() is not None:
                return router.returncode or 1
            try:
                with urlopen(f"{config['LLAMA_ROUTER_URL']}/models", timeout=2):
                    break
            except (OSError, URLError):
                time.sleep(1)
        else:
            fail("llama-server router did not become ready")

        proxy_command = [
            sys.executable,
            "/usr/local/bin/lazy-llama-proxy.py",
            "--listen-host", config["HOST"],
            "--listen-port", config["PORT"],
            "--backend", config["LLAMA_ROUTER_URL"],
        ]
        log(f"Starting lazy OpenAI-compatible proxy on port {config['PORT']}")
        proxy = subprocess.Popen(proxy_command)
        while router.poll() is None and proxy.poll() is None:
            time.sleep(0.5)
        return router.returncode if router.poll() is not None else (proxy.returncode or 0)
    except KeyboardInterrupt:
        return 130
    finally:
        terminate(proxy)
        terminate(router)


if __name__ == "__main__":
    raise SystemExit(main())
