#!/usr/bin/env python3
"""myLLMServer の依存しない最小 YAML モデルリスト読み取り器。"""

from __future__ import annotations

import json
import re
import sys


SCALAR_KEYS = {"url", "mtp", "mmproj", "imatrix"}
KEYS = SCALAR_KEYS | {"shards"}


def scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def read_models(path: str) -> list[dict[str, str | list[str]]]:
    models: list[dict[str, str | list[str]]] = []
    current: dict[str, str | list[str]] | None = None
    in_models = False
    shards_indent = -1
    with open(path, encoding="utf-8") as handle:
        for number, raw in enumerate(handle, 1):
            line = raw.rstrip("\n")
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            if stripped == "models:":
                in_models = True
                continue
            if not in_models:
                raise ValueError(f"{path}:{number}: expected 'models:'")
            list_item = re.match(r"^(\s*)-\s*(.*)$", line)
            if list_item:
                indent = len(list_item.group(1))
                value = list_item.group(2)
                if current is not None and shards_indent >= 0 and indent >= shards_indent:
                    if not value:
                        raise ValueError(f"{path}:{number}: shard URL must not be empty")
                    shards = current.get("shards")
                    if not isinstance(shards, list):
                        raise ValueError(f"{path}:{number}: invalid shards list")
                    shards.append(scalar(value))
                    continue
                if current is not None:
                    models.append(current)
                current = {}
                shards_indent = -1
                if value:
                    key, separator, value = value.partition(":")
                    if not separator or key.strip() not in KEYS:
                        raise ValueError(f"{path}:{number}: invalid model property")
                    key = key.strip()
                    if key == "shards":
                        if value.strip():
                            raise ValueError(f"{path}:{number}: shards must use a YAML list")
                        current["shards"] = []
                        shards_indent = indent + 2
                    elif key in SCALAR_KEYS:
                        current[key] = scalar(value)
                continue
            match = re.match(r"^(\s+)([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
            if not match or current is None or match.group(2) not in KEYS:
                raise ValueError(f"{path}:{number}: invalid model property")
            indent = len(match.group(1))
            key = match.group(2)
            value = match.group(3)
            if key == "shards":
                if value.strip():
                    raise ValueError(f"{path}:{number}: shards must use a YAML list")
                current["shards"] = []
                shards_indent = indent + 2
            elif key in SCALAR_KEYS:
                current[key] = scalar(value)
                shards_indent = -1
            else:
                raise ValueError(f"{path}:{number}: invalid model property")
    if current is not None:
        models.append(current)
    if not models:
        raise ValueError(f"{path}: no models found")
    for index, model in enumerate(models, 1):
        if not model.get("url"):
            raise ValueError(f"{path}: model {index} is missing url")
    return models


def main() -> int:
    try:
        models = read_models(sys.argv[1])
        for model in models:
            print(
                "\t".join(
                    [
                        *(str(model.get(key, "")) for key in ("url", "mtp", "mmproj", "imatrix")),
                        json.dumps(model.get("shards", []), ensure_ascii=False),
                    ]
                )
            )
    except (IndexError, OSError, ValueError) as error:
        print(f"model-list error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
