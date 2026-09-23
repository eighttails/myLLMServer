import importlib.util
import io
import json
import pathlib
import unittest


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


if __name__ == "__main__":
    unittest.main()
