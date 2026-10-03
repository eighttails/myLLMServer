import importlib.util
import json
import os
import pathlib
import tempfile
import unittest
from unittest.mock import patch


DOCKER_DIR = pathlib.Path(__file__).parents[1] / "docker"


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, DOCKER_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ContainerScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.start = load_script("start-llama")
        cls.sync = load_script("sync-model")
        cls.configure = load_script("configure-model-preset")

    def test_start_configuration_uses_router_port_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual("11435", self.start.validate()["LLAMA_ROUTER_PORT"])
            self.assertEqual("1", self.start.validate()["MAX_PARALLEL_SLOTS"])
            self.assertEqual("off", self.start.validate()["SPECULATIVE_DECODING"])

    def test_resolves_hugging_face_and_short_model_urls(self):
        with patch.object(self.sync, "HF_ENDPOINT", "https://mirror.example"):
            self.assertEqual(
                (
                    "owner/repo",
                    "model.gguf",
                    "https://mirror.example/owner/repo/resolve/main/model.gguf?download=true",
                ),
                self.sync.resolve_model_spec(
                    "https://huggingface.co/owner/repo/resolve/main/model.gguf"
                ),
            )
        self.assertEqual(
            (
                "owner/repo",
                "model.gguf",
                "https://huggingface.co/owner/repo/resolve/main/model.gguf?download=true",
            ),
            self.sync.resolve_model_spec("owner/repo/model.gguf"),
        )

    def test_model_sync_removes_unlisted_files_and_writes_preset(self):
        with tempfile.TemporaryDirectory() as temporary:
            model_dir = pathlib.Path(temporary)
            (model_dir / "active.gguf").write_bytes(b"valid model")
            stale_model = model_dir / "stale.gguf"
            stale_model.write_bytes(b"unused")
            pathlib.Path(f"{stale_model}.part").write_bytes(b"partial")
            self.sync.MODEL_DIR = model_dir
            self.sync.PRESET_FILE = model_dir / ".models-preset.ini"
            self.sync.PRESET_SECTION_DIR = model_dir / ".models-preset.d"
            self.sync.MODEL_ALIAS_FILE = model_dir / ".model-aliases.tsv"
            self.sync.METADATA_CACHE_FILE = model_dir / ".model-metadata-cache.tsv"
            with (
                patch.object(
                    self.sync,
                    "parse_models",
                    return_value=[{"url": "https://huggingface.co/owner/repo/resolve/main/active.gguf"}],
                ),
                patch.object(self.sync, "model_metadata", return_value=("llama", 500000)),
                patch.dict(os.environ, {
                    "CONTEXT_SIZE": "",
                    "MAX_CONTEXT_SIZE": "",
                    "KV_CACHE_TYPE": "",
                }),
            ):
                self.assertEqual(0, self.sync.main())

            self.assertFalse(stale_model.exists())
            self.assertFalse(pathlib.Path(f"{stale_model}.part").exists())
            self.assertIn("active\tactive.gguf", self.sync.MODEL_ALIAS_FILE.read_text())
            preset = self.sync.PRESET_FILE.read_text()
            self.assertIn("[active]", preset)
            self.assertIn("ctx-size = 262144", preset)
            self.assertIn("cache-type-k = q4_0", preset)

    def test_detects_builtin_mtp_tensor_names(self):
        dump = json.dumps({
            "tensors": {
                "blk.64.nextn.eh_proj.weight": {},
                "blk.64.nextn.enorm.weight": {},
            }
        })
        self.assertTrue(self.configure.has_mtp_head(dump))
        self.assertFalse(self.configure.has_mtp_head(json.dumps({"tensors": {"blk.0.ffn_up.weight": {}}})))

    def test_preset_enables_builtin_mtp_with_quantized_draft_kv(self):
        original_cfg = self.configure.CFG.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                model_dir = pathlib.Path(temporary)
                model = model_dir / "mtp.gguf"
                model.write_bytes(b"model")
                self.configure.MODEL_DIR = model_dir
                self.configure.PRESET_FILE = model_dir / ".models-preset.ini"
                self.configure.SECTION_DIR = model_dir / ".models-preset.d"
                self.configure.CFG.update({
                    "CONTEXT_SIZE": "",
                    "MAX_CONTEXT_SIZE": "",
                    "KV_CACHE_TYPE": "",
                    "MAX_PARALLEL_SLOTS": "1",
                    "MODEL_IDLE_SECONDS": "1800",
                    "SPECULATIVE_DECODING": "on",
                    "SPLIT_MODE": "layer",
                    "TENSOR_SPLIT_MODE": "auto",
                })
                dump = json.dumps({
                    "metadata": {
                        "general.architecture": {"value": {"value": "qwen35"}},
                        "qwen35.context_length": {"value": {"value": 8192}},
                    }
                })
                with (
                    patch.object(self.configure, "gguf_dump", return_value=dump),
                    patch.object(self.configure, "util", return_value=""),
                    patch.object(self.configure, "has_mtp_head", return_value=True),
                    patch.object(self.configure, "gpu_count", 1),
                    patch.object(self.configure, "usable_vram_bytes", return_value=10**12),
                    patch.object(self.configure, "kv_vram_bytes", return_value=100),
                ):
                    self.configure.configure_locked("mtp", "mtp.gguf", model)
                    section = (self.configure.SECTION_DIR / "mtp.ini").read_text()
                    self.assertIn("spec-type = draft-mtp", section)
                    self.assertIn("spec-draft-type-k = q4_0", section)
                    self.assertIn("spec-draft-type-v = q4_0", section)

                    self.configure.CFG["SPECULATIVE_DECODING"] = "off"
                    self.configure.configure_locked("mtp", "mtp.gguf", model)
                    section = (self.configure.SECTION_DIR / "mtp.ini").read_text()
                    self.assertNotIn("spec-type =", section)
        finally:
            self.configure.CFG.clear()
            self.configure.CFG.update(original_cfg)

    def test_preset_falls_back_to_server_fit_when_vram_is_unavailable(self):
        original_cfg = self.configure.CFG.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                model_dir = pathlib.Path(temporary)
                model = model_dir / "active.gguf"
                model.write_bytes(b"valid model")
                self.configure.MODEL_DIR = model_dir
                self.configure.PRESET_FILE = model_dir / ".models-preset.ini"
                self.configure.SECTION_DIR = model_dir / ".models-preset.d"
                self.configure.CFG.update({
                    "CONTEXT_SIZE": "",
                    "MAX_CONTEXT_SIZE": "",
                    "KV_CACHE_TYPE": "",
                    "N_GPU_LAYERS": "auto",
                    "MODEL_IDLE_SECONDS": "1800",
                    "SPLIT_MODE": "layer",
                })
                dump = json.dumps({
                    "metadata": {
                        "general.architecture": {"value": {"value": "llama"}},
                        "llama.context_length": {"value": {"value": 8192}},
                    }
                })
                with (
                    patch.object(self.configure, "gguf_dump", return_value=dump),
                    patch.object(self.configure, "util", return_value=""),
                    patch.object(self.configure, "usable_vram_bytes", return_value=0),
                ):
                    self.configure.configure_locked("active", "active.gguf", model)
                section = (self.configure.SECTION_DIR / "active.ini").read_text()
                self.assertIn("ctx-size = 8192", section)
                self.assertIn("cache-type-k = q4_0", section)
        finally:
            self.configure.CFG.clear()
            self.configure.CFG.update(original_cfg)

    def test_single_gpu_moe_preset_offloads_only_ram_safe_expert_layers(self):
        original_cfg = self.configure.CFG.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                model_dir = pathlib.Path(temporary)
                model = model_dir / "active.gguf"
                model.write_bytes(b"model")
                self.configure.MODEL_DIR = model_dir
                self.configure.PRESET_FILE = model_dir / ".models-preset.ini"
                self.configure.SECTION_DIR = model_dir / ".models-preset.d"
                self.configure.CFG.update({
                    "CONTEXT_SIZE": "",
                    "MAX_CONTEXT_SIZE": "",
                    "KV_CACHE_TYPE": "",
                    "MAX_PARALLEL_SLOTS": "1",
                    "MODEL_IDLE_SECONDS": "1800",
                    "MOE_CPU_OFFLOAD": "auto",
                    "MOE_ACTIVE_RATIO_THRESHOLD": "0.125",
                    "MOE_RAM_RESERVE_MIB": "400",
                    "SPLIT_MODE": "layer",
                    "TENSOR_SPLIT_MODE": "auto",
                })
                dump = json.dumps({
                    "metadata": {
                        "general.architecture": {"value": {"value": "llama"}},
                        "llama.context_length": {"value": {"value": 1000}},
                    }
                })

                def fake_util(command, input_text=""):
                    if command == ["moe-profile"]:
                        return "2 8 1 2 600"
                    if command == ["kv-profile"]:
                        return "10 0 0 10 0"
                    if command == ["layer-bytes", "--split-moe"]:
                        return "100\n100 300 0 0\n100 300 0 0"
                    return ""

                with (
                    patch.object(self.configure, "gguf_dump", return_value=dump),
                    patch.object(self.configure, "util", side_effect=fake_util),
                    patch.object(self.configure, "gpu_count", 1),
                    patch.object(self.configure, "MIB", 1),
                    patch.object(self.configure, "available_ram_bytes", return_value=700),
                    patch.object(self.configure, "usable_vram_bytes", return_value=850),
                    patch.object(
                        self.configure,
                        "kv_vram_bytes",
                        side_effect=lambda params, kind, context, slots=1: (
                            slots * {"q4_0": 100, "f16": 200, "q8_0": 120}[kind]
                        ),
                    ),
                ):
                    self.configure.configure_locked("active", "active.gguf", model)

                section = (self.configure.SECTION_DIR / "active.ini").read_text()
                self.assertIn("n-cpu-moe = 1", section)
                self.assertIn("parallel = 1", section)
                self.assertIn("cache-type-k = q4_0", section)
                self.assertIn("fit = off", section)
                self.assertIn("n-gpu-layers = all", section)
        finally:
            self.configure.CFG.clear()
            self.configure.CFG.update(original_cfg)

    def test_auto_context_is_capped_at_256k_and_full_model_stays_on_gpu(self):
        original_cfg = self.configure.CFG.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                model_dir = pathlib.Path(temporary)
                model = model_dir / "active.gguf"
                model.write_bytes(b"model")
                self.configure.MODEL_DIR = model_dir
                self.configure.PRESET_FILE = model_dir / ".models-preset.ini"
                self.configure.SECTION_DIR = model_dir / ".models-preset.d"
                self.configure.CFG.update({
                    "CONTEXT_SIZE": "",
                    "MAX_CONTEXT_SIZE": "",
                    "KV_CACHE_TYPE": "",
                    "MAX_PARALLEL_SLOTS": "1",
                    "MODEL_IDLE_SECONDS": "1800",
                    "SPECULATIVE_DECODING": "on",
                    "SPLIT_MODE": "layer",
                    "TENSOR_SPLIT_MODE": "auto",
                })
                draft_model = model_dir / "draft.gguf"
                draft_model.write_bytes(b"draft model")
                self.configure.SECTION_DIR.mkdir()
                (self.configure.SECTION_DIR / "active.ini").write_text(
                    f"[active]\nmodel = {model}\nmodel-draft = {draft_model}\n",
                    encoding="utf-8",
                )
                dump = json.dumps({
                    "metadata": {
                        "general.architecture": {"value": {"value": "llama"}},
                        "llama.context_length": {"value": {"value": 1000000}},
                    }
                })

                def fake_util(command, input_text=""):
                    if command == ["kv-profile"]:
                        return "10 0 0 10 0"
                    return ""

                with (
                    patch.object(self.configure, "gguf_dump", return_value=dump),
                    patch.object(self.configure, "util", side_effect=fake_util),
                    patch.object(self.configure, "gpu_count", 1),
                    patch.object(self.configure, "usable_vram_bytes", return_value=10**12),
                    patch.object(self.configure, "kv_vram_bytes", return_value=100),
                ):
                    self.configure.configure_locked("active", "active.gguf", model)

                section = (self.configure.SECTION_DIR / "active.ini").read_text()
                self.assertIn("ctx-size = 262144", section)
                self.assertIn("parallel = 1", section)
                self.assertIn("fit = off", section)
                self.assertIn("n-gpu-layers = all", section)
                self.assertIn(f"model-draft = {draft_model}", section)
                self.assertIn("spec-type = draft-simple", section)
                self.assertIn("spec-draft-type-k = q4_0", section)
                self.assertIn("spec-draft-type-v = q4_0", section)
        finally:
            self.configure.CFG.clear()
            self.configure.CFG.update(original_cfg)

    def test_multi_gpu_decode_plan_keeps_dense_ffn_on_gpu(self):
        original_cfg = self.configure.CFG.copy()
        try:
            with tempfile.TemporaryDirectory() as temporary:
                model_dir = pathlib.Path(temporary)
                model = model_dir / "active.gguf"
                model.write_bytes(b"model")
                self.configure.MODEL_DIR = model_dir
                self.configure.PRESET_FILE = model_dir / ".models-preset.ini"
                self.configure.SECTION_DIR = model_dir / ".models-preset.d"
                self.configure.CFG.update({
                    "CONTEXT_SIZE": "",
                    "MAX_CONTEXT_SIZE": "",
                    "MIN_CONTEXT_SIZE": "2048",
                    "CONTEXT_SIZE_STEP": "1024",
                    "KV_CACHE_TYPE": "",
                    "MAX_PARALLEL_SLOTS": "1",
                    "MODEL_IDLE_SECONDS": "1800",
                    "MOE_CPU_OFFLOAD": "auto",
                    "MOE_RAM_RESERVE_MIB": "1",
                    "N_GPU_LAYERS": "auto",
                    "SPLIT_MODE": "layer",
                    "TENSOR_SPLIT_MODE": "auto",
                    "UBATCH_SIZE": "256",
                    "VRAM_RESERVE_MIB": "4096",
                })
                dump = json.dumps({
                    "metadata": {
                        "general.architecture": {"value": {"value": "llama"}},
                        "llama.context_length": {"value": {"value": 8192}},
                    }
                })
                util_calls = []

                def fake_util(command, input_text=""):
                    util_calls.append(command)
                    if command == ["moe-profile"]:
                        return ""
                    if command == ["kv-profile"]:
                        return "10 0 0 10 0"
                    if command and command[0] == "tensor-split":
                        return "2 1,1 0 0"
                    return ""

                with (
                    patch.object(self.configure, "gguf_dump", return_value=dump),
                    patch.object(self.configure, "util", side_effect=fake_util),
                    patch.object(self.configure, "layer_bytes", return_value="100\n100 300 10 0\n100 300 10 0"),
                    patch.object(self.configure, "gpu_count", 2),
                    patch.object(self.configure, "usable_vram_bytes", return_value=100 * 1024**3),
                    patch.object(self.configure, "per_gpu_free_mib", return_value=[12000, 12000]),
                    patch.object(self.configure, "per_gpu_speed", return_value=[2.0, 1.0]),
                    patch.object(self.configure, "available_ram_bytes", return_value=100 * 1024**3),
                    patch.object(self.configure, "kv_vram_bytes", return_value=100),
                    patch.object(self.configure, "scratch_bytes", return_value=0),
                ):
                    self.configure.configure_locked("active", "active.gguf", model)

                section = (self.configure.SECTION_DIR / "active.ini").read_text()
                self.assertIn("tensor-split = 1,1", section)
                self.assertNotIn("n-cpu-ffn", section)
                self.assertFalse(any("--moe-auto" in command for command in util_calls))
        finally:
            self.configure.CFG.clear()
            self.configure.CFG.update(original_cfg)


if __name__ == "__main__":
    unittest.main()
