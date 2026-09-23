#!/usr/bin/env python3
"""myLLMServer の依存しない最小 YAML モデルリスト読み取り器。"""

from __future__ import annotations

import re
import sys


KEYS = {"url", "mtp", "mmproj", "imatrix"}


def scalar(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    return value


def read_models(path: str) -> list[dict[str, str]]:
    models: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    in_models = False
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
            if re.match(r"^\s*-\s*", line):
                if current is not None:
                    models.append(current)
                current = {}
                rest = re.sub(r"^\s*-\s*", "", line)
                if rest:
                    key, separator, value = rest.partition(":")
                    if not separator or key.strip() not in KEYS:
                        raise ValueError(f"{path}:{number}: invalid model property")
                    current[key.strip()] = scalar(value)
                continue
            match = re.match(r"^\s+([A-Za-z][A-Za-z0-9_-]*)\s*:\s*(.*)$", line)
            if not match or current is None or match.group(1) not in KEYS:
                raise ValueError(f"{path}:{number}: invalid model property")
            current[match.group(1)] = scalar(match.group(2))
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
            print("\t".join(model.get(key, "") for key in ("url", "mtp", "mmproj", "imatrix")))
    except (IndexError, OSError, ValueError) as error:
        print(f"model-list error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
