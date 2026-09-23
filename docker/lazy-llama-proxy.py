#!/usr/bin/env python3
import argparse
import configparser
import http.server
import json
import os
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request


HOP_BY_HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}


class LazyProxy(http.server.ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, server_address, handler_class, backend):
        super().__init__(server_address, handler_class)
        self.backend = backend.rstrip("/")
        self.active_model = None
        self.model_lock = threading.Lock()
        self.alias_file_mtime = 0
        self.model_entries = []
        self.model_aliases = {}
        self.reload_model_entries()

    def reload_model_entries(self):
        alias_file = os.environ.get("MODEL_ALIAS_FILE")
        if alias_file and os.path.exists(alias_file):
            try:
                self.alias_file_mtime = os.path.getmtime(alias_file)
            except OSError:
                self.alias_file_mtime = 0
        entries = self._load_model_entries()
        aliases = self._load_model_aliases(entries)
        self.model_entries = entries
        self.model_aliases = aliases
        if self.active_model and self.active_model not in self.model_aliases:
            self.active_model = None

    def ensure_model_entries_up_to_date(self):
        alias_file = os.environ.get("MODEL_ALIAS_FILE")
        if alias_file and os.path.exists(alias_file):
            try:
                current_mtime = os.path.getmtime(alias_file)
            except OSError:
                current_mtime = 0
            if current_mtime != self.alias_file_mtime:
                self.reload_model_entries()

    def _load_model_entries(self):
        entries = []
        alias_file = os.environ.get("MODEL_ALIAS_FILE")
        if not alias_file:
            return entries
        try:
            with open(alias_file, "r", encoding="utf-8") as handle:
                for line in handle:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 4:
                        raise RuntimeError(
                            f"invalid model alias entry in {alias_file}: expected 4 fields"
                        )
                    alias, filename, architecture, context_length = fields
                    try:
                        context_length = int(context_length)
                    except ValueError as err:
                        raise RuntimeError(
                            f"invalid context length for model {alias}: {context_length}"
                        ) from err
                    if not alias or not filename or not architecture or context_length <= 0:
                        raise RuntimeError(f"invalid model alias entry for model {alias}")
                    entries.append(
                        {
                            "alias": alias,
                            "filename": filename,
                            "architecture": architecture,
                            "context_length": context_length,
                        }
                    )
        except OSError as err:
            raise RuntimeError(f"failed to read model alias file {alias_file}: {err}") from err
        return entries

    def _load_model_aliases(self, entries):
        aliases = {}
        for entry in entries:
            alias = entry["alias"]
            filename = entry["filename"]
            aliases[alias] = alias
            aliases[filename] = alias
        return aliases

    def model_context_length(self, entry):
        section_dir = os.environ.get("PRESET_SECTION_DIR")
        if not section_dir:
            raise RuntimeError("PRESET_SECTION_DIR is not configured")

        parser = configparser.RawConfigParser()
        section_file = os.path.join(section_dir, f"{entry['alias']}.ini")
        try:
            with open(section_file, "r", encoding="utf-8") as handle:
                parser.read_file(handle)
            context_length = parser.getint(entry["alias"], "ctx-size")
        except (OSError, configparser.Error, ValueError) as err:
            raise RuntimeError(
                f"failed to read context length for model {entry['alias']}: {err}"
            ) from err
        if context_length <= 0:
            raise RuntimeError(
                f"invalid context length for model {entry['alias']}: {context_length}"
            )
        return context_length


class LazyProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        self._handle()

    def do_POST(self):
        self._handle()

    def do_PUT(self):
        self._handle()

    def do_PATCH(self):
        self._handle()

    def do_DELETE(self):
        self._handle()

    def log_message(self, fmt, *args):
        parsed_path = urllib.parse.urlsplit(self.path).path if getattr(self, "path", None) else ""
        if parsed_path == "/api/ps":
            return
        sys.stderr.write("[llama-proxy] " + fmt % args + "\n")

    def _read_body(self):
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length else b""

    def _extract_model(self, body):
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if query.get("model", [""])[0]:
            return query["model"][0]

        if self.command not in {"POST", "PUT", "PATCH"} or not body:
            return None

        content_type = self.headers.get("Content-Type", "")
        if "json" not in content_type:
            return None

        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            return None
        if isinstance(payload, dict):
            for key in ("model", "name"):
                if isinstance(payload.get(key), str):
                    return payload[key]
        return None

    def _request_backend(self, method, path, body=b"", headers=None):
        headers = headers or {}
        req = urllib.request.Request(
            self.server.backend + path,
            data=body if method not in {"GET", "HEAD"} else None,
            headers=headers,
            method=method,
        )
        return urllib.request.urlopen(req, timeout=None)

    def _backend_json(self, method, path, payload=None):
        body = b""
        headers = {}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
            headers["Content-Length"] = str(len(body))
        with self._request_backend(method, path, body=body, headers=headers) as resp:
            data = resp.read()
            if data:
                return json.loads(data.decode("utf-8"))
            return None

    def _wait_until_unloaded(self, model):
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            data = self._backend_json("GET", "/models")
            for item in data.get("data", []) if isinstance(data, dict) else []:
                if item.get("id") != model:
                    continue
                status = item.get("status", {})
                if isinstance(status, dict) and status.get("value") == "unloaded":
                    return
            time.sleep(0.5)
        raise RuntimeError(f"timed out waiting for model to unload: {model}")

    def _prepare_model_if_needed(self, model):
        if not model or model == "*":
            return

        canonical_model = self.server.model_aliases.get(model, model)
        with self.server.model_lock:
            if canonical_model == self.server.active_model:
                return

            if self.server.active_model and os.environ.get("MODELS_MAX", "1") == "1":
                self.log_message("unloading previous model before recalculating preset: %s", self.server.active_model)
                try:
                    self._backend_json("POST", "/models/unload", {"model": self.server.active_model})
                except urllib.error.HTTPError as err:
                    if err.code != 400:
                        raise
                    # モデルが既に (アイドルタイムアウトなどで) アンロード済みの場合、
                    # llama-server は 400 "model is not running" を返す。
                    # プロキシ側の active_model の記憶が古いだけなので無視して続行する。
                    self.log_message(
                        "model already unloaded (ignoring 400 from /models/unload): %s",
                        self.server.active_model,
                    )
                else:
                    self._wait_until_unloaded(self.server.active_model)

            self.log_message("preparing lazy preset for model switch: %s", canonical_model)
            subprocess.run(
                ["/usr/local/bin/configure-model-preset.sh", canonical_model],
                check=True,
                env=os.environ.copy(),
            )
            self._backend_json("GET", "/models?reload=1")
            self.server.active_model = canonical_model

    def _forward(self, body):
        parsed = urllib.parse.urlsplit(self.path)
        target_path = urllib.parse.urlunsplit(("", "", parsed.path, parsed.query, ""))
        headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() not in HOP_BY_HOP_HEADERS and key.lower() != "host"
        }
        if body:
            headers["Content-Length"] = str(len(body))
        elif "Content-Length" in headers:
            headers.pop("Content-Length", None)

        # thinking モード無効化時: /v1/chat/completions のリクエストから thinking/reasoning パラメータを削除
        thinking_mode = self._extract_thinking_mode()
        if thinking_mode == "off" and self.command == "POST" and "/v1/chat/completions" in target_path:
            try:
                payload = json.loads(body.decode("utf-8"))
                if isinstance(payload, dict):
                    payload.pop("thinking", None)
                    payload.pop("reasoning", None)
                    # messages 内の reasoning_content / thinking_content も削除
                    for msg in payload.get("messages", []):
                        if isinstance(msg, dict):
                            msg.pop("reasoning_content", None)
                            msg.pop("thinking_content", None)
                    body = json.dumps(payload).encode("utf-8")
                    headers["Content-Length"] = str(len(body))
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass

        try:
            with self._request_backend(self.command, target_path, body=body, headers=headers) as resp:
                self.send_response(resp.status)
                for key, value in resp.headers.items():
                    if key.lower() not in HOP_BY_HOP_HEADERS:
                        self.send_header(key, value)
                self.send_header("Connection", "close")
                self.end_headers()
                while True:
                    chunk = resp.read(64 * 1024)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
        except urllib.error.HTTPError as err:
            self.send_response(err.code)
            for key, value in err.headers.items():
                if key.lower() not in HOP_BY_HOP_HEADERS:
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(err.read())
        finally:
            self.close_connection = True

    def _send_error_json(self, status, message):
        self._send_json(status, {"error": message})

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def _ollama_modified_at(self, filename):
        model_dir = os.environ.get("MODEL_DIR", "")
        try:
            modified_at = os.path.getmtime(os.path.join(model_dir, filename))
        except OSError:
            modified_at = time.time()
        return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(modified_at))

    def _ollama_model_size(self, filename):
        model_dir = os.environ.get("MODEL_DIR", "")
        try:
            return os.path.getsize(os.path.join(model_dir, filename))
        except OSError:
            return 0

    def _send_ollama_tags(self):
        models = []
        for entry in self.server.model_entries:
            alias = entry["alias"]
            filename = entry["filename"]
            models.append(
                {
                    "name": alias,
                    "model": alias,
                    "modified_at": self._ollama_modified_at(filename),
                    "size": self._ollama_model_size(filename),
                    "digest": "",
                    "details": {
                        "format": "gguf",
                        "family": "",
                        "families": [],
                        "parameter_size": "",
                        "quantization_level": "",
                    },
                }
            )
        self._send_json(200, {"models": models})

    def _send_ollama_show(self, model):
        canonical_model = self.server.model_aliases.get(model, model)
        entry = next(
            (e for e in self.server.model_entries if e["alias"] == canonical_model),
            None,
        )
        if entry is None:
            self._send_json(404, {"error": f"model '{model}' not found"})
            return

        filename = entry["filename"]
        architecture = entry["architecture"]
        context_length = self.server.model_context_length(entry)
        # Copilot 等はモデル一覧表示時に全モデルへ /api/show を投げるため、
        # ここでモデルのロードやプリセット再計算を行ってはならない (タイムアウトの原因になる)。
        self._send_json(
            200,
            {
                "license": "",
                "modelfile": "",
                "parameters": "",
                "template": "",
                "details": {
                    "parent_model": "",
                    "format": "gguf",
                    "family": "",
                    "families": [],
                    "parameter_size": "",
                    "quantization_level": "",
                },
                "model_info": {
                    "general.basename": canonical_model,
                    "general.architecture": architecture,
                    "general.file_type": 15,
                    f"{architecture}.context_length": context_length,
                },
                "capabilities": ["completion", "tools"],
                "modified_at": self._ollama_modified_at(filename),
            },
        )

    def _send_ollama_ps(self):
        models = []
        active = self.server.active_model
        if active:
            entry = next(
                (e for e in self.server.model_entries if e["alias"] == active),
                None,
            )
            if entry is not None:
                models.append(
                    {
                        "name": active,
                        "model": active,
                        "size": self._ollama_model_size(entry["filename"]),
                        "digest": "",
                        "details": {
                            "parent_model": "",
                            "format": "gguf",
                            "family": "",
                            "families": [],
                            "parameter_size": "",
                            "quantization_level": "",
                        },
                        "expires_at": "0001-01-01T00:00:00Z",
                        "size_vram": 0,
                    }
                )
        self._send_json(200, {"models": models})

    def _ollama_now(self):
        return time.strftime("%Y-%m-%dT%H:%M:%S.000000Z", time.gmtime())

    def _openai_tool_call_to_ollama(self, tool_call):
        if not isinstance(tool_call, dict):
            return None

        function = tool_call.get("function")
        if not isinstance(function, dict):
            return None

        arguments = function.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments else {}
            except json.JSONDecodeError:
                pass

        return {
            "function": {
                "name": function.get("name", ""),
                "arguments": arguments,
            },
        }

    def _ollama_messages_to_openai(self, messages):
        openai_messages = []
        for message in messages if isinstance(messages, list) else []:
            if not isinstance(message, dict):
                continue
            openai_message = dict(message)
            # thinking モード無効化時は、reasoning_content / thinking_content を削除
            if self._extract_thinking_mode() == "off":
                openai_message.pop("reasoning_content", None)
                openai_message.pop("thinking_content", None)
            tool_calls = openai_message.get("tool_calls")
            if isinstance(tool_calls, list):
                converted_tool_calls = []
                for index, tool_call in enumerate(tool_calls):
                    if not isinstance(tool_call, dict):
                        continue
                    function = tool_call.get("function")
                    if not isinstance(function, dict):
                        continue
                    arguments = function.get("arguments", {})
                    if not isinstance(arguments, str):
                        arguments = json.dumps(arguments)
                    converted_tool_calls.append(
                        {
                            "id": tool_call.get("id") or f"call_{index}",
                            "type": tool_call.get("type") or "function",
                            "function": {
                                "name": function.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    )
                openai_message["tool_calls"] = converted_tool_calls
            openai_messages.append(openai_message)
        return openai_messages

    def _openai_to_ollama_chat_response(self, model, payload, done_reason="stop"):
        choice = (payload.get("choices") or [{}])[0]
        message = choice.get("message", {})
        usage = payload.get("usage", {}) or {}
        timings = payload.get("timings", {}) or {}
        ollama_message = {
            "role": message.get("role", "assistant"),
            "content": message.get("content") or "",
        }
        tool_calls = [
            converted
            for converted in (
                self._openai_tool_call_to_ollama(tool_call)
                for tool_call in message.get("tool_calls", [])
            )
            if converted is not None
        ]
        if tool_calls:
            ollama_message["tool_calls"] = tool_calls
        return {
            "model": model,
            "created_at": self._ollama_now(),
            "message": ollama_message,
            "done": True,
            "done_reason": choice.get("finish_reason") or done_reason,
            "total_duration": int(timings.get("predicted_ms", 0) * 1_000_000),
            "load_duration": 0,
            "prompt_eval_count": usage.get("prompt_tokens", 0),
            "prompt_eval_duration": int(timings.get("prompt_ms", 0) * 1_000_000),
            "eval_count": usage.get("completion_tokens", 0),
            "eval_duration": int(timings.get("predicted_ms", 0) * 1_000_000),
        }

    def _strip_thinking_from_openai_response(self, payload):
        """thinking モード無効化時に、レスポンスから推論プロセス (reasoning/thinking) を削除する."""
        if not isinstance(payload, dict):
            return payload
        for choice in payload.get("choices", []):
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if not isinstance(message, dict):
                continue
            # reasoning_content や thinking_content を削除
            for key in ("reasoning_content", "thinking_content", "reasoning", "thoughts"):
                message.pop(key, None)
            # tool_calls の中の reasoning も削除
            for tool_call in message.get("tool_calls", []):
                if isinstance(tool_call, dict):
                    tool_call.pop("reasoning", None)
                    tool_call.pop("thinking", None)
        return payload

    def _strip_thinking_from_openai_chunk(self, chunk):
        """thinking モード無効化時に、ストリーミングチャンクから推論プロセスを削除する."""
        if not isinstance(chunk, dict):
            return chunk
        for choice in chunk.get("choices", []):
            if not isinstance(choice, dict):
                continue
            delta = choice.get("delta", {})
            if not isinstance(delta, dict):
                continue
            for key in ("reasoning_content", "thinking_content", "reasoning", "thoughts"):
                delta.pop(key, None)
        return chunk

    def _openai_chunk_to_ollama_chunk(self, model, chunk):
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta", {})
        return {
            "model": model,
            "created_at": self._ollama_now(),
            "message": {
                "role": delta.get("role") or "assistant",
                "content": delta.get("content") or "",
            },
            "done": False,
        }

    def _accumulate_tool_call_delta(self, accumulator, fragment):
        """OpenAI streamingのtool_calls断片(index単位で分割されたid/name/argumentsの部分文字列)を
        indexごとに連結し、完成したtool_callを組み立てる."""
        if not isinstance(fragment, dict):
            return
        index = fragment.get("index", 0)
        entry = accumulator.setdefault(
            index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}}
        )
        if fragment.get("id"):
            entry["id"] = fragment["id"]
        if fragment.get("type"):
            entry["type"] = fragment["type"]
        function_fragment = fragment.get("function")
        if isinstance(function_fragment, dict):
            if function_fragment.get("name"):
                entry["function"]["name"] += function_fragment["name"]
            if function_fragment.get("arguments"):
                entry["function"]["arguments"] += function_fragment["arguments"]

    def _finalize_tool_calls(self, accumulator):
        return [
            converted
            for converted in (
                self._openai_tool_call_to_ollama(
                    {
                        "id": entry["id"],
                        "type": entry["type"],
                        "function": {
                            "name": entry["function"]["name"],
                            "arguments": entry["function"]["arguments"],
                        },
                    }
                )
                for _, entry in sorted(accumulator.items())
            )
            if converted is not None
        ]

    def _extract_thinking_mode(self):
        """THINKING_MODE 環境変数を取得する (デフォルト: auto)."""
        return os.environ.get("THINKING_MODE", "auto")

    def _handle_ollama_chat(self, body, model):
        try:
            request_payload = json.loads(body.decode("utf-8")) if body else {}
        except json.JSONDecodeError:
            self._send_error_json(400, "invalid JSON body")
            return

        stream = bool(request_payload.get("stream", True))
        thinking_mode = self._extract_thinking_mode()
        tools = request_payload.get("tools")
        openai_payload = {
            "model": model,
            "messages": self._ollama_messages_to_openai(request_payload.get("messages", [])),
            # tools が付いていても常にバックエンドへストリーミングでリクエストする。
            # 非ストリーミングで生成完了までブロックすると、長い生成中はクライアントへ
            # 1バイトも送られず、クライアント側のヘッダー受信タイムアウトで失敗する
            # (llama-server自体は正常に生成を継続しているにもかかわらず)。
            "stream": stream,
        }
        if tools:
            openai_payload["tools"] = tools
        if "tool_choice" in request_payload:
            openai_payload["tool_choice"] = request_payload["tool_choice"]
        options = request_payload.get("options") or {}
        if "temperature" in options:
            openai_payload["temperature"] = options["temperature"]
        if "top_p" in options:
            openai_payload["top_p"] = options["top_p"]
        if "num_predict" in options:
            openai_payload["max_tokens"] = options["num_predict"]

        # thinking / reasoning 関連パラメータを削除 (THINKING_MODE=off 時 or llama-server が未対応の場合)
        openai_payload.pop("thinking", None)
        openai_payload.pop("reasoning", None)

        openai_body = json.dumps(openai_payload).encode("utf-8")
        headers = {"Content-Type": "application/json", "Content-Length": str(len(openai_body))}

        if not stream:
            with self._request_backend("POST", "/v1/chat/completions", body=openai_body, headers=headers) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            if thinking_mode == "off":
                data = self._strip_thinking_from_openai_response(data)
            self._send_json(200, self._openai_to_ollama_chat_response(model, data))
            return

        # ストリーミング応答: OpenAI の text/event-stream (SSE) を Ollama の NDJSON に変換する。
        # tool_calls はOpenAI形式ではindexごとに断片化されて送られてくるため、
        # ここで蓄積し、[DONE] を受け取った時点でまとめて最終メッセージに含めて送る。
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Connection", "close")
        self.end_headers()
        tool_call_accumulator = {}
        finish_reason = "stop"
        with self._request_backend("POST", "/v1/chat/completions", body=openai_body, headers=headers) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue
                if thinking_mode == "off":
                    chunk = self._strip_thinking_from_openai_chunk(chunk)
                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta", {}) if isinstance(choice, dict) else {}
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]
                delta_tool_calls = delta.get("tool_calls") if isinstance(delta, dict) else None
                if isinstance(delta_tool_calls, list):
                    for fragment in delta_tool_calls:
                        self._accumulate_tool_call_delta(tool_call_accumulator, fragment)
                # reasoning や tool_calls の断片も空 content の有効な Ollama チャンクとして送る。
                # 未完成の内容は公開せず、長い推論中もクライアントの本文無通信タイムアウトを防ぐ。
                ollama_chunk = self._openai_chunk_to_ollama_chunk(model, chunk)
                self.wfile.write((json.dumps(ollama_chunk) + "\n").encode("utf-8"))
                self.wfile.flush()
        final_message = {"role": "assistant", "content": ""}
        final_tool_calls = self._finalize_tool_calls(tool_call_accumulator)
        if final_tool_calls:
            final_message["tool_calls"] = final_tool_calls
        final = {
            "model": model,
            "created_at": self._ollama_now(),
            "message": final_message,
            "done": True,
            "done_reason": finish_reason,
        }
        self.wfile.write((json.dumps(final) + "\n").encode("utf-8"))
        self.wfile.flush()
        self.close_connection = True

    def _handle_unload(self, body):
        model = self._extract_model(body)
        canonical_model = self.server.model_aliases.get(model, model) if model else None
        target_model = canonical_model or self.server.active_model

        with self.server.model_lock:
            if not target_model:
                self._send_json(200, {"status": "ok", "message": "no active model to unload"})
                return

            self.log_message("unloading model via API request: %s", target_model)
            try:
                self._backend_json("POST", "/models/unload", {"model": target_model})
            except urllib.error.HTTPError as err:
                if err.code != 400:
                    raise
                self.log_message(
                    "model already unloaded (ignoring 400 from /models/unload): %s",
                    target_model,
                )
            else:
                self._wait_until_unloaded(target_model)

            if target_model == self.server.active_model:
                self.server.active_model = None

            self._send_json(
                200,
                {
                    "status": "ok",
                    "message": f"model '{target_model}' unloaded successfully",
                    "unloaded_model": target_model,
                },
            )

    def _handle_reload(self, body):
        with self.server.model_lock:
            self.log_message("reloading model list from model_list.yml / MODEL_NAMES_CSV...")
            try:
                subprocess.run(
                    ["/usr/local/bin/sync-model.sh"],
                    check=True,
                    env=os.environ.copy(),
                )
            except subprocess.CalledProcessError as err:
                self.log_message("failed to sync models: %s", err)
                self._send_error_json(500, f"failed to sync models: {err}")
                return

            self.server.reload_model_entries()
            models_list = [entry["alias"] for entry in self.server.model_entries]
            self.log_message("model list reloaded successfully. active models: %s", models_list)
            self._send_json(
                200,
                {
                    "status": "ok",
                    "message": "model_list reloaded and synchronized successfully",
                    "models": models_list,
                },
            )

    def _handle(self):
        self.server.ensure_model_entries_up_to_date()
        body = self._read_body()
        model = self._extract_model(body)
        canonical_model = self.server.model_aliases.get(model, model) if model else None
        parsed_path = urllib.parse.urlsplit(self.path).path
        try:
            if self.command == "GET" and parsed_path == "/api/tags":
                self._send_ollama_tags()
                return
            if self.command == "GET" and parsed_path == "/api/version":
                self._send_json(200, {"version": "llama-server-proxy"})
                return
            if self.command == "GET" and parsed_path == "/api/ps":
                self._send_ollama_ps()
                return
            if self.command == "POST" and parsed_path == "/api/show":
                self._send_ollama_show(model)
                return
            if parsed_path in {"/models/reload", "/api/reload", "/v1/models/reload", "/v1/reload", "/reload"}:
                self._handle_reload(body)
                return
            if parsed_path in {"/models/unload", "/api/unload", "/v1/models/unload", "/v1/unload"}:
                self._handle_unload(body)
                return
            self._prepare_model_if_needed(model)
            if self.command == "POST" and parsed_path == "/api/chat":
                self._handle_ollama_chat(body, canonical_model or model)
                return
            self._forward(body)
        except subprocess.CalledProcessError as err:
            self.log_message("failed to prepare model preset: %s", err)
            self._send_error_json(502, f"failed to prepare model preset for {model}")
        except urllib.error.URLError as err:
            self.log_message("backend request failed: %s", err)
            self._send_error_json(502, "llama-server backend request failed")
        except RuntimeError as err:
            self.log_message("%s", err)
            self._send_error_json(502, str(err))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--listen-host", required=True)
    parser.add_argument("--listen-port", type=int, required=True)
    parser.add_argument("--backend", required=True)
    args = parser.parse_args()

    server = LazyProxy((args.listen_host, args.listen_port), LazyProxyHandler, args.backend)
    sys.stderr.write(
        f"[llama-proxy] listening on {args.listen_host}:{args.listen_port}, backend={args.backend}\n"
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
