#!/usr/bin/env python3
"""モデルの取得・検証・プリセット同期を行う。"""

from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import urlopen


MODEL_DIR = Path(os.environ.get("MODEL_DIR") or "/models")
AUTO_CONTEXT_SIZE_MAX = 262144
MODEL_LIST_FILE = Path(os.environ.get("MODEL_LIST_FILE") or MODEL_DIR / "model_list.yml")
HF_ENDPOINT = (os.environ.get("HF_ENDPOINT") or "https://huggingface.co").rstrip("/")
PRESET_FILE = Path(os.environ.get("PRESET_FILE") or MODEL_DIR / ".models-preset.ini")
PRESET_SECTION_DIR = Path(os.environ.get("PRESET_SECTION_DIR") or MODEL_DIR / ".models-preset.d")
MODEL_ALIAS_FILE = Path(os.environ.get("MODEL_ALIAS_FILE") or MODEL_DIR / ".model-aliases.tsv")
METADATA_CACHE_FILE = Path(os.environ.get("METADATA_CACHE_FILE") or MODEL_DIR / ".model-metadata-cache.tsv")
MAX_STALLED_ATTEMPTS = int(os.environ.get("DOWNLOAD_MAX_STALLED_ATTEMPTS", "10"))


class ModelError(Exception):
    """モデル1件の処理をスキップできるエラー。"""


SHARD_FILENAME = re.compile(r"^(.*)-(\d+)-of-(\d+)\.gguf$")


def log(message: str) -> None:
    print(f"[llama-wrapper] {message}", file=sys.stderr, flush=True)


def signature(path: Path) -> str:
    stat = path.stat()
    return f"{stat.st_size}:{int(stat.st_mtime)}"


def asset_name(url: str) -> str:
    return Path(urlsplit(url).path).name


def hf_download_url(url: str) -> str:
    if "?" not in url:
        url += "?download=true"
    if url.startswith("https://huggingface.co/") and HF_ENDPOINT != "https://huggingface.co":
        url = HF_ENDPOINT + url.removeprefix("https://huggingface.co")
    return url


def parse_models() -> list[dict[str, str | list[str]]]:
    if MODEL_LIST_FILE.is_file():
        result = subprocess.run(
            ["/usr/local/bin/model-list-utils.py", str(MODEL_LIST_FILE)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stderr.strip() or "failed to read model list")
        keys = ("url", "mtp", "mmproj", "imatrix", "shards")
        return [
            {
                **dict(zip(keys[:4], fields[:4])),
                "shards": json.loads(fields[4]) if len(fields) > 4 else [],
            }
            for line in result.stdout.splitlines()
            if (fields := line.split("\t")) and fields[0].strip()
        ]
    legacy = os.environ.get("MODEL_NAMES_CSV", "")
    if not legacy:
        raise RuntimeError(f"model_list.yml not found and MODEL_NAMES_CSV is empty: {MODEL_LIST_FILE}")
    return [{"url": item.strip()} for item in legacy.split(",") if item.strip()]


def gguf_data(path: Path, *, tensors: bool = False) -> dict:
    command = ["gguf-dump"]
    if not tensors:
        command.append("--no-tensors")
    command.extend(["--json", "--json-array", str(path)])
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode:
        raise ModelError(f"gguf-dump failed for {path.name}: {result.stderr.strip()}")
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ModelError(f"invalid GGUF metadata for {path.name}: {error}") from error
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), {})
    if not isinstance(data, dict):
        raise ModelError(f"invalid GGUF metadata for {path.name}")
    return data


def metadata_value(metadata: dict, suffix: str):
    for key, entry in metadata.items():
        if key.endswith(suffix):
            value = entry.get("value") if isinstance(entry, dict) else None
            if isinstance(value, dict):
                value = value.get("value")
            return value
    return None


def model_metadata(path: Path) -> tuple[str, int]:
    data = gguf_data(path)
    metadata = data.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ModelError(f"missing GGUF metadata for {path.name}")
    architecture = metadata_value(metadata, "general.architecture")
    context_size = metadata_value(metadata, ".context_length")
    if not isinstance(architecture, str) or not isinstance(context_size, (int, float)):
        raise ModelError(f"could not read GGUF architecture/context_length: {path.name}")
    return architecture, int(context_size)


def download_with_resume(url: str, temporary: Path, label: str) -> None:
    transient_curl_codes = {6, 7, 16, 18, 28, 35, 52, 55, 56, 92}
    stalled = 0
    delay = 2
    while True:
        before = temporary.stat().st_size if temporary.exists() else 0
        command = [
            "curl", "--fail", "--location", "--show-error", "--silent",
            "--retry", "5", "--retry-delay", "2", "--retry-connrefused",
            "--connect-timeout", "30", "--output", str(temporary),
        ]
        if before:
            command.extend(["--continue-at", "-"])
        command.append(url)
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        if result.returncode == 0:
            return
        error = result.stderr.strip() or "unknown error"
        if result.returncode in {33, 416}:
            log(f"Resume not supported by server; restarting download: {label}")
            temporary.unlink(missing_ok=True)
            stalled = 0
            delay = 2
            continue
        if result.returncode not in transient_curl_codes:
            raise ModelError(f"failed to download {label} (curl exit {result.returncode}): {error}")

        after = temporary.stat().st_size if temporary.exists() else 0
        if after > before:
            stalled = 0
            delay = 2
        else:
            stalled += 1
            if stalled >= MAX_STALLED_ATTEMPTS:
                raise ModelError(
                    f"download made no progress after {stalled} attempts: {label} "
                    f"(curl exit {result.returncode}): {error}"
                )
            delay = min(delay * 2, 30)
        log(f"Download interrupted (curl exit {result.returncode}); resuming from {after} bytes in {delay}s: {label}")
        time.sleep(delay)


def download_auxiliary(url: str) -> str:
    if not url:
        return ""
    if not url.startswith("https://huggingface.co/"):
        raise ModelError(f"auxiliary URL must be a full Hugging Face URL: {url}")
    name = asset_name(url)
    if not name.endswith((".gguf", ".bin", ".dat", ".gguf.imatrix")):
        raise ModelError(f"unsupported auxiliary model file: {name}")
    destination = MODEL_DIR / name
    if destination.is_file():
        return name
    temporary = Path(f"{destination}.part")
    log(f"Downloading auxiliary file {name}")
    download_with_resume(hf_download_url(url), temporary, name)
    temporary.replace(destination)
    return name


def read_metadata_cache() -> dict[tuple[str, str], tuple[str, int]]:
    cache: dict[tuple[str, str], tuple[str, int]] = {}
    if not METADATA_CACHE_FILE.is_file():
        return cache
    for row in METADATA_CACHE_FILE.read_text(encoding="utf-8").splitlines():
        parts = row.split("\t")
        if len(parts) != 4:
            continue
        name, file_sig, architecture, context = parts
        try:
            cache[(name, file_sig)] = (architecture, int(context))
        except ValueError:
            continue
    return cache


def resolve_model_spec(spec: str) -> tuple[str, str, str]:
    if spec.startswith("https://huggingface.co/"):
        filename = asset_name(spec)
        repo = spec.removeprefix("https://huggingface.co/").split("/resolve/", 1)[0]
        return repo, filename, hf_download_url(spec)
    filename = asset_name(spec)
    repo = spec.rpartition("/")[0]
    return repo, filename, f"{HF_ENDPOINT}/{repo}/resolve/main/{filename}?download=true"


def resolve_model_shards(url: str, shard_urls: list[str]) -> list[tuple[str, str]]:
    repo, filename, _ = resolve_model_spec(url)
    match = SHARD_FILENAME.fullmatch(filename)
    if not match:
        if shard_urls:
            raise ModelError(f"shards can only be used with a numbered GGUF shard: {filename}")
        return []

    prefix, first_index, total_count = match.groups()
    first_index = int(first_index)
    total_count = int(total_count)
    if total_count < 2 or first_index != 1:
        raise ModelError(f"the model url must point to the first shard (part 1 of at least 2): {filename}")
    if len(shard_urls) != total_count - 1:
        raise ModelError(
            f"expected {total_count - 1} additional shard URL(s) for {filename}, got {len(shard_urls)}"
        )

    resolved = []
    for expected_index, shard_url in enumerate(shard_urls, start=2):
        shard_repo, shard_filename, download_url = resolve_model_spec(shard_url)
        shard_match = SHARD_FILENAME.fullmatch(shard_filename)
        if shard_repo != repo or not shard_match:
            raise ModelError(f"shard must be in the same repository and use the same numbered pattern: {shard_url}")
        shard_prefix, shard_index, shard_total = shard_match.groups()
        if (
            shard_prefix != prefix
            or int(shard_index) != expected_index
            or int(shard_total) != total_count
        ):
            raise ModelError(
                f"expected shard {expected_index:0{len(match.group(2))}d}-of-{total_count:0{len(match.group(3))}d} "
                f"for {filename}, got {shard_filename}"
            )
        resolved.append((shard_filename, download_url))
    return resolved


def write_atomic(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def process_model(
    entry: dict[str, str | list[str]],
    metadata_cache: dict[tuple[str, str], tuple[str, int]],
    allowed_files: set[str],
) -> tuple[str, str]:
    spec = entry.get("url", "").strip()
    repo, filename, download_url = resolve_model_spec(spec)
    if "/" not in repo:
        raise ModelError(
            f"invalid model spec: {spec} (owner/repo is missing; verify the Hugging Face download URL)"
        )
    if not filename.endswith(".gguf"):
        raise ModelError(f"model must be a .gguf file: {filename}")

    shard_urls = entry.get("shards", [])
    if not isinstance(shard_urls, list) or any(not isinstance(item, str) or not item.strip() for item in shard_urls):
        raise ModelError(f"shards must be a list of non-empty URLs for {filename}")
    shard_files = resolve_model_shards(spec, shard_urls)
    model_files = [filename, *(shard_filename for shard_filename, _ in shard_files)]
    allowed_files.update(model_files)

    auxiliary_names: dict[str, str] = {}
    for key in ("mtp", "mmproj", "imatrix"):
        url = entry.get(key, "").strip()
        auxiliary_names[key] = download_auxiliary(url) if url else ""
        if auxiliary_names[key]:
            allowed_files.add(auxiliary_names[key])

    destination = MODEL_DIR / filename
    cached_metadata = None
    if destination.is_file():
        cache_key = (filename, signature(destination))
        cached_metadata = metadata_cache.get(cache_key)
        if cached_metadata:
            log(f"Using cached model: {filename}")
        else:
            try:
                cached_metadata = model_metadata(destination)
                log(f"Verified model: {filename}")
            except ModelError:
                log(f"Cached model is invalid; downloading again: {filename}")
                destination.unlink(missing_ok=True)

    if not destination.is_file():
        log(f"Downloading {repo}")
        temporary = Path(f"{destination}.part")
        try:
            download_with_resume(download_url, temporary, repo)
            temporary.replace(destination)
        except ModelError as error:
            log(f"error: skipping model due to download failure: {repo}: {error}")
            raise

    for shard_filename, shard_download_url in shard_files:
        shard_destination = MODEL_DIR / shard_filename
        if shard_destination.is_file() and shard_destination.stat().st_size > 0:
            continue
        log(f"Downloading shard {shard_filename}")
        temporary = Path(f"{shard_destination}.part")
        try:
            download_with_resume(shard_download_url, temporary, shard_filename)
            temporary.replace(shard_destination)
        except ModelError as error:
            log(f"error: skipping model due to shard download failure: {shard_filename}: {error}")
            raise

    try:
        architecture, detected_context = cached_metadata or model_metadata(destination)
    except ModelError as error:
        log(f"error: could not read GGUF metadata after download: {filename}: {error}; removing invalid file")
        destination.unlink(missing_ok=True)
        raise

    configured_context = int(
        os.environ.get("CONTEXT_SIZE")
        or min(detected_context, AUTO_CONTEXT_SIZE_MAX)
    )
    max_context = os.environ.get("MAX_CONTEXT_SIZE")
    if max_context and configured_context > int(max_context):
        configured_context = int(max_context)
    # 分割GGUFは "-00001-of-0000N" を除いた名前をモデル名として公開する
    shard_match = SHARD_FILENAME.fullmatch(filename) if shard_files else None
    alias = shard_match.group(1) if shard_match else filename.removesuffix(".gguf")
    metadata_line = f"{filename}\t{signature(destination)}\t{architecture}\t{detected_context}"
    alias_line = f"{alias}\t{filename}\t{architecture}\t{configured_context}"
    cache_type = os.environ.get("KV_CACHE_TYPE") or "q4_0"
    lines = [
        f"[{alias}]",
        f"model = {destination}",
        f"sleep-idle-seconds = {os.environ.get('MODEL_IDLE_SECONDS') or '1800'}",
        f"n-gpu-layers = {os.environ.get('N_GPU_LAYERS') or 'auto'}",
        f"ctx-size = {configured_context}",
        "parallel = 1",
        f"kv-unified-per-slot = {configured_context}",
    ]
    for key, option in (("mtp", "model-draft"), ("mmproj", "mmproj")):
        if auxiliary_names[key]:
            lines.append(f"{option} = {MODEL_DIR / auxiliary_names[key]}")
    lines.extend([f"cache-type-k = {cache_type}", f"cache-type-v = {cache_type}", ""])
    return metadata_line, alias_line + "\n" + "\n".join(lines)


def cleanup_unused_models(allowed_files: set[str]) -> None:
    for path in MODEL_DIR.iterdir():
        if path.is_file() and path.suffix in {".gguf", ".bin", ".dat"} and path.name not in allowed_files:
            log(f"Removing model not in model_list: {path.name}")
            path.unlink()
            Path(f"{path}.part").unlink(missing_ok=True)
            (MODEL_DIR / f"llama-bench-{path.name}.json").unlink(missing_ok=True)


def main() -> int:
    try:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        with (MODEL_DIR / ".sync-model.lock").open("a", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            return sync_models()
    except (OSError, RuntimeError, ValueError) as error:
        log(f"error: {error}")
        return 1


def sync_models() -> int:
    preset_tmp = Path(f"{PRESET_SECTION_DIR}.tmp-{os.getpid()}")
    backup = Path(f"{PRESET_SECTION_DIR}.backup-{os.getpid()}")
    try:
        entries = parse_models()
        if not entries:
            raise RuntimeError("no valid model entries found in model_list.yml")
        if preset_tmp.exists():
            shutil.rmtree(preset_tmp)
        if backup.exists():
            shutil.rmtree(backup)
        preset_tmp.mkdir(parents=True)
        metadata_cache = read_metadata_cache()
        allowed_files: set[str] = set()
        metadata_rows: list[str] = []
        alias_rows: list[str] = []
        section_rows: list[str] = []
        skipped: list[str] = []
        for entry in entries:
            spec = entry.get("url", "")
            try:
                metadata_row, alias_and_section = process_model(entry, metadata_cache, allowed_files)
                alias_row, section = alias_and_section.split("\n", 1)
                metadata_rows.append(metadata_row)
                alias_rows.append(alias_row)
                section_rows.append(section)
                alias = alias_row.split("\t", 1)[0]
                (preset_tmp / f"{alias}.ini").write_text(section, encoding="utf-8")
            except (ModelError, OSError, ValueError) as error:
                skipped.append(spec)
                log(f"Skipped model {spec}: {error}")
        if not alias_rows:
            raise RuntimeError(f"no models were successfully prepared (all {len(entries)} entries failed or skipped)")

        cleanup_unused_models(allowed_files)
        if PRESET_SECTION_DIR.exists():
            PRESET_SECTION_DIR.replace(backup)
        try:
            preset_tmp.replace(PRESET_SECTION_DIR)
        except OSError:
            if backup.exists() and not PRESET_SECTION_DIR.exists():
                backup.replace(PRESET_SECTION_DIR)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        write_atomic(MODEL_ALIAS_FILE, "\n".join(alias_rows) + "\n")
        write_atomic(METADATA_CACHE_FILE, "\n".join(metadata_rows) + "\n")
        preset = "[*]\nfit = on\nsplit-mode = " + (os.environ.get("SPLIT_MODE") or "layer") + "\n\n"
        preset += "".join(f"{section}\n" for section in section_rows)
        write_atomic(PRESET_FILE, preset)

        router_port = os.environ.get("LLAMA_ROUTER_PORT", "11435")
        try:
            with urlopen(f"http://127.0.0.1:{router_port}/models?reload=1", timeout=10):
                log("Reloaded models-preset on llama-server")
        except OSError:
            pass
        if skipped:
            log(f"Model sync completed with {len(skipped)} model(s) skipped. Active models in list: {' '.join(sorted(allowed_files))}")
        else:
            log(f"Model sync completed successfully. Active models in list: {' '.join(sorted(allowed_files))}")
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        log(f"error: {error}")
        return 1
    finally:
        if preset_tmp.exists():
            shutil.rmtree(preset_tmp)


if __name__ == "__main__":
    raise SystemExit(main())
