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
                patch.object(self.sync, "model_metadata", return_value=("llama", 8192)),
            ):
                self.assertEqual(0, self.sync.main())

            self.assertFalse(stale_model.exists())
            self.assertFalse(pathlib.Path(f"{stale_model}.part").exists())
            self.assertIn("active\tactive.gguf", self.sync.MODEL_ALIAS_FILE.read_text())
            self.assertIn("[active]", self.sync.PRESET_FILE.read_text())

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


if __name__ == "__main__":
    unittest.main()
