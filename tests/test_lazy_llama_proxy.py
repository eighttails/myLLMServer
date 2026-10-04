import importlib.util
import io
import json
import pathlib
import unittest
from unittest import mock


MODULE_PATH = pathlib.Path(__file__).parents[1] / "docker" / "lazy-llama-proxy.py"
SPEC = importlib.util.spec_from_file_location("lazy_llama_proxy", MODULE_PATH)
PROXY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROXY)


class TextRepetitionDetectorTests(unittest.TestCase):
    def test_detects_repeated_block_across_chunks(self):
        detector = PROXY.TextRepetitionDetector()
        pattern = "".join(
            f"{index:02d}番目の要素には他と異なる内容が含まれています。"
            for index in range(8)
        )

        self.assertIsNone(detector.append(pattern))
        self.assertIsNone(detector.append(pattern))
        detection = detector.append(pattern)

        self.assertEqual("repeated-block", detection["kind"])
        self.assertGreaterEqual(detection["repeat_count"], 3)

    def test_does_not_flag_normal_long_text(self):
        detector = PROXY.TextRepetitionDetector()
        text = "".join(f"{index:04d}: 異なる内容の行です。\n" for index in range(1000))

        detections = [
            detector.append(text[index:index + 47])
            for index in range(0, len(text), 47)
        ]

        self.assertTrue(all(detection is None for detection in detections))

    def test_detects_repeated_line(self):
        detector = PROXY.TextRepetitionDetector()
        line = "同一の十分に長いエラー行が繰り返されています。詳細情報も毎回完全に同一です。"

        detection = detector.append((line + "\n") * 7)

        self.assertEqual("repeated-line", detection["kind"])


class ToolLoopDetectorTests(unittest.TestCase):
    def _cycle(self, name, arguments, result):
        return [
            {
                "role": "assistant",
                "tool_calls": [{"function": {"name": name, "arguments": arguments}}],
            },
            {"role": "tool", "content": result},
        ]

    def test_detects_identical_tool_call_and_result(self):
        messages = []
        for _ in range(3):
            messages.extend(self._cycle("read_file", {"path": "a.txt"}, "unchanged"))

        detection = PROXY.detect_tool_result_cycle(messages)

        self.assertEqual(1, detection["cycle_length"])
        self.assertEqual(["read_file"], detection["tool_names"])

    def test_canonicalizes_argument_key_order(self):
        messages = []
        messages.extend(self._cycle("search", {"query": "x", "limit": 10}, "same"))
        messages.extend(self._cycle("search", '{"limit":10,"query":"x"}', "same"))
        messages.extend(self._cycle("search", {"limit": 10, "query": "x"}, "same"))

        self.assertIsNotNone(PROXY.detect_tool_result_cycle(messages))

    def test_does_not_flag_changing_tool_results(self):
        messages = []
        for result in ("pending", "running", "complete"):
            messages.extend(self._cycle("poll", {"job": 1}, result))

        self.assertIsNone(PROXY.detect_tool_result_cycle(messages))

    def test_detects_two_step_cycle(self):
        messages = []
        for _ in range(3):
            messages.extend(self._cycle("read", {"path": "a"}, "A"))
            messages.extend(self._cycle("search", {"query": "b"}, "B"))

        detection = PROXY.detect_tool_result_cycle(messages)

        self.assertEqual(2, detection["cycle_length"])
        self.assertEqual(["read", "search"], detection["tool_names"])


class OllamaOptionTests(unittest.TestCase):
    def test_forwards_repetition_and_dry_sampler_options(self):
        payload = {}

        PROXY.apply_ollama_options(
            payload,
            {
                "repeat_penalty": 1.1,
                "repeat_last_n": 256,
                "dry_multiplier": 0.8,
                "dry_allowed_length": 2,
                "top_k": 40,
            },
        )

        self.assertEqual(1.1, payload["repeat_penalty"])
        self.assertEqual(256, payload["repeat_last_n"])
        self.assertEqual(0.8, payload["dry_multiplier"])
        self.assertEqual(2, payload["dry_allowed_length"])
        self.assertEqual(40, payload["top_k"])
        self.assertNotIn("max_tokens", payload)

    def test_maps_num_predict_only_when_client_supplies_it(self):
        payload = {}

        PROXY.apply_ollama_options(payload, {"num_predict": 8192})

        self.assertEqual(8192, payload["max_tokens"])


class StreamingLoopGuardTests(unittest.TestCase):
    def test_cancels_backend_before_forwarding_detected_repetition(self):
        handler = object.__new__(PROXY.LazyProxyHandler)
        handler.wfile = io.BytesIO()
        handler.send_response = lambda _status: None
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        handler.log_message = lambda _format, *_args: None
        backend_closed = []
        pattern = "".join(
            f"{index:02d}番目の要素には他と異なる内容が含まれています。"
            for index in range(8)
        )

        def backend_stream(_method, _path, body=b"", headers=None):
            del body, headers
            try:
                for _ in range(4):
                    chunk = {"choices": [{"delta": {"content": pattern}}]}
                    yield f"data: {json.dumps(chunk)}\n".encode()
            finally:
                backend_closed.append(True)

        handler._iter_backend_stream = backend_stream
        request = {
            "model": "test",
            "stream": True,
            "messages": [{"role": "user", "content": "test"}],
        }

        handler._handle_ollama_chat(json.dumps(request).encode(), "test")

        responses = [
            json.loads(line)
            for line in handler.wfile.getvalue().decode().splitlines()
        ]
        streamed_content = "".join(
            response["message"]["content"]
            for response in responses
            if not response["done"]
        )
        self.assertEqual(pattern * 2, streamed_content)
        self.assertTrue(responses[-1]["done"])
        self.assertEqual("stop", responses[-1]["done_reason"])
        self.assertEqual([True], backend_closed)


class BackendStreamingForwardTests(unittest.TestCase):
    def test_forwards_available_backend_chunks_without_filling_read_buffer(self):
        first_chunk = b"data: first\n\n"
        second_chunk = b"data: second\n\n"
        writes = io.BytesIO()

        class BackendResponse:
            status = 200
            headers = {"Content-Type": "text/event-stream"}

            def __init__(self):
                self.chunks = [first_chunk, second_chunk, b""]

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read1(self, _size):
                if len(self.chunks) == 2:
                    self.assert_forwarded(first_chunk)
                elif len(self.chunks) == 1:
                    self.assert_forwarded(first_chunk + second_chunk)
                return self.chunks.pop(0)

            @staticmethod
            def assert_forwarded(expected):
                if writes.getvalue() != expected:
                    raise AssertionError("backend chunk was not forwarded before the next read")

            def read(self, _size=-1):
                raise AssertionError("buffer-filling read must not be used for backend streams")

        response = BackendResponse()
        handler = object.__new__(PROXY.LazyProxyHandler)
        handler.command = "POST"
        handler.path = "/v1/chat/completions"
        handler.headers = {"Content-Type": "application/json"}
        handler.wfile = writes
        handler.close_connection = False
        handler.send_response = lambda _status: None
        handler.send_header = lambda _name, _value: None
        handler.end_headers = lambda: None
        handler._request_backend = lambda *_args, **_kwargs: response

        handler._forward(b'{"stream":true}')

        self.assertEqual(first_chunk + second_chunk, writes.getvalue())
        self.assertTrue(handler.close_connection)


class EmptyOllamaResponseTests(unittest.TestCase):
    def _handler(self, chunks=None, response=None):
        handler = object.__new__(PROXY.LazyProxyHandler)
        handler.command = "POST"
        handler.path = "/api/chat"
        handler.headers = {"Content-Type": "application/json"}
        handler.wfile = io.BytesIO()
        handler.send_response = mock.Mock()
        handler.send_header = mock.Mock()
        handler.end_headers = mock.Mock()
        handler.log_message = mock.Mock()
        handler.server = mock.Mock()
        handler.server.model_aliases = {}
        handler._prepare_model_if_needed = mock.Mock()
        handler._extract_thinking_mode = lambda: "auto"
        handler._read_body = lambda: json.dumps({
            "model": "test",
            "messages": [{"role": "user", "content": "test"}],
            "stream": chunks is not None,
        }).encode()
        if chunks is not None:
            def backend_stream(*_args, **_kwargs):
                for chunk in chunks:
                    if chunk is None:
                        yield None
                    else:
                        yield f"data: {json.dumps(chunk)}\n".encode()
                yield b"data: [DONE]\n"
            handler._iter_backend_stream = backend_stream
        else:
            backend_response = io.BytesIO(json.dumps(response).encode())
            handler._request_backend = lambda *_args, **_kwargs: backend_response
        return handler

    def test_reasoning_only_stream_returns_error_not_empty_success(self):
        handler = self._handler(chunks=[
            None,
            {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
            {"choices": [{"delta": {}, "finish_reason": "length"}]},
        ])

        handler._handle()

        responses = [json.loads(line) for line in handler.wfile.getvalue().splitlines()]
        self.assertIn("no assistant content or tool calls", responses[-1]["error"])
        self.assertIn("length", responses[-1]["error"])
        self.assertFalse(any(response.get("done") for response in responses))
        self.assertNotIn("private reasoning", handler.wfile.getvalue().decode())

    def test_empty_nonstream_response_returns_502(self):
        handler = self._handler(response={
            "choices": [{
                "message": {"content": "", "reasoning_content": "private reasoning"},
                "finish_reason": "stop",
            }],
        })

        handler._handle()

        handler.send_response.assert_called_once_with(502)
        response = json.loads(handler.wfile.getvalue())
        self.assertIn("no assistant content or tool calls", response["error"])

    def test_visible_stream_still_finishes_normally(self):
        handler = self._handler(chunks=[
            {"choices": [{"delta": {"reasoning_content": "private reasoning"}}]},
            {"choices": [{"delta": {"content": "answer"}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
        ])

        handler._handle()

        responses = [json.loads(line) for line in handler.wfile.getvalue().splitlines()]
        self.assertTrue(responses[-1]["done"])
        self.assertEqual("answer", "".join(r["message"]["content"] for r in responses))

    def test_tool_only_stream_still_finishes_normally(self):
        handler = self._handler(chunks=[
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"name": "read_file", "arguments": '{"path":'},
            }]}}]},
            {"choices": [{"delta": {"tool_calls": [{
                "index": 0,
                "function": {"arguments": '"a.txt"}'},
            }]}, "finish_reason": "tool_calls"}]},
        ])

        handler._handle()

        responses = [json.loads(line) for line in handler.wfile.getvalue().splitlines()]
        self.assertTrue(responses[-1]["done"])
        self.assertEqual(
            {"function": {"name": "read_file", "arguments": {"path": "a.txt"}}},
            responses[-1]["message"]["tool_calls"][0],
        )


if __name__ == "__main__":
    unittest.main()
