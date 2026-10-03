#!/usr/bin/env python3
"""コンテナ内の llama-server からモデルをアンロードする。"""

from __future__ import annotations

import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def main() -> int:
    port = os.environ.get("PORT", "11434")
    model = sys.argv[1] if len(sys.argv) > 1 else ""
    if model:
        print(f"Unloading model: {model}", file=sys.stderr)
    else:
        print("Unloading active model...", file=sys.stderr)
    request = Request(
        f"http://127.0.0.1:{port}/models/unload",
        data=json.dumps({"model": model} if model else {}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=60) as response:
            sys.stdout.buffer.write(response.read())
            sys.stdout.write("\n")
    except (HTTPError, URLError, OSError) as error:
        print(f"error: failed to unload model: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
