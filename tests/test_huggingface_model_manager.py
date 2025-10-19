import asyncio
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict


if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")
    torch_stub.float16 = "float16"
    torch_stub.bfloat16 = "bfloat16"
    cuda_stub = SimpleNamespace(
        is_available=lambda: False,
        device_count=lambda: 0,
        empty_cache=lambda: None,
        get_device_properties=lambda *args, **kwargs: SimpleNamespace(total_memory=0),
        memory_allocated=lambda *args, **kwargs: 0,
    )
    torch_stub.cuda = cuda_stub
    torch_stub.compile = lambda model: model
    sys.modules["torch"] = torch_stub

if "psutil" not in sys.modules:
    psutil_stub = types.ModuleType("psutil")
    psutil_stub.virtual_memory = lambda: SimpleNamespace(available=1024 * 1024 * 1024)
    sys.modules["psutil"] = psutil_stub

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _AutoConfig:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace(model_type="gpt2")

    class _AutoTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace()

    class _AutoModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return SimpleNamespace()

    class _BitsAndBytesConfig:
        def __init__(self, *args, **kwargs):
            pass

    class _Trainer:
        pass

    class _TrainingArguments:
        def __init__(self, *args, **kwargs):
            pass

    class _DataCollator:
        pass

    def _pipeline(*args, **kwargs):
        return lambda *a, **k: []

    transformers_stub.AutoConfig = _AutoConfig
    transformers_stub.AutoTokenizer = _AutoTokenizer
    transformers_stub.AutoModelForCausalLM = _AutoModel
    transformers_stub.BitsAndBytesConfig = _BitsAndBytesConfig
    transformers_stub.Trainer = _Trainer
    transformers_stub.TrainingArguments = _TrainingArguments
    transformers_stub.DataCollatorForLanguageModeling = _DataCollator
    transformers_stub.pipeline = _pipeline

    sys.modules["transformers"] = transformers_stub

    integrations_stub = types.ModuleType("transformers.integrations")
    deepspeed_stub = types.ModuleType("transformers.integrations.deepspeed")

    class _HfDeepSpeedConfig:
        def __init__(self, *args, **kwargs):
            pass

    deepspeed_stub.HfDeepSpeedConfig = _HfDeepSpeedConfig
    integrations_stub.deepspeed = deepspeed_stub
    sys.modules["transformers.integrations"] = integrations_stub
    sys.modules["transformers.integrations.deepspeed"] = deepspeed_stub

if "accelerate" not in sys.modules:
    accelerate_stub = types.ModuleType("accelerate")
    accelerate_stub.infer_auto_device_map = lambda *args, **kwargs: {}
    sys.modules["accelerate"] = accelerate_stub

if "huggingface_hub" not in sys.modules:
    huggingface_hub_stub = types.ModuleType("huggingface_hub")

    class _InferenceClient:
        def __init__(self, *args, **kwargs):
            pass

        def model_info(self, *args, **kwargs):
            return SimpleNamespace(pipeline_tag=None, tags=[], num_parameters=None)

    class _HfApi:
        def list_repo_files(self, *args, **kwargs):
            return []

    def _hf_hub_download(*args, **kwargs):
        return ""

    def _snapshot_download(*args, **kwargs):
        return ""

    huggingface_hub_stub.InferenceClient = _InferenceClient
    huggingface_hub_stub.HfApi = _HfApi
    huggingface_hub_stub.hf_hub_download = _hf_hub_download
    huggingface_hub_stub.snapshot_download = _snapshot_download
    sys.modules["huggingface_hub"] = huggingface_hub_stub
else:
    huggingface_hub_module = sys.modules["huggingface_hub"]

    if not hasattr(huggingface_hub_module, "InferenceClient"):
        class _InferenceClient:
            def __init__(self, *args, **kwargs):
                pass

            def model_info(self, *args, **kwargs):
                return SimpleNamespace(pipeline_tag=None, tags=[], num_parameters=None)

        huggingface_hub_module.InferenceClient = _InferenceClient

    if not hasattr(huggingface_hub_module, "HfApi"):
        class _HfApi:
            def list_repo_files(self, *args, **kwargs):
                return []

        huggingface_hub_module.HfApi = _HfApi

    if not hasattr(huggingface_hub_module, "hf_hub_download"):
        def _hf_hub_download(*args, **kwargs):
            return ""

        huggingface_hub_module.hf_hub_download = _hf_hub_download

    if not hasattr(huggingface_hub_module, "snapshot_download"):
        def _snapshot_download(*args, **kwargs):
            return ""

        huggingface_hub_module.snapshot_download = _snapshot_download

if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")

    class _Dataset:
        pass

    datasets_stub.Dataset = _Dataset
    sys.modules["datasets"] = datasets_stub

import pytest

from modules.Providers.HuggingFace.components import huggingface_model_manager as manager_module
from modules.Providers.HuggingFace.components.huggingface_model_manager import HuggingFaceModelManager
from modules.Providers.HuggingFace.config.base_config import BaseConfig
from modules.Providers.HuggingFace.config.nvme_config import NVMeConfig
from modules.Providers.HuggingFace.utils.cache_manager import CacheManager


class _DummyConfigManager:
    def __init__(self, token, cache_dir):
        self._token = token
        self._cache_dir = cache_dir
        self._generation_settings: Dict[str, Any] = {}

    def get_huggingface_api_key(self):
        return self._token

    def get_model_cache_dir(self):
        return self._cache_dir

    def get_huggingface_generation_settings(self):
        return dict(self._generation_settings)

    def set_huggingface_generation_settings(self, settings):
        self._generation_settings = dict(settings)
        return dict(self._generation_settings)


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {"input_ids": []}


class _DummyModel:
    transformer = SimpleNamespace(h=[])


def test_base_config_uses_persisted_generation_settings(tmp_path):
    config_manager = _DummyConfigManager(token=None, cache_dir=str(tmp_path))
    config_manager.set_huggingface_generation_settings({"temperature": 0.9, "top_k": 5})

    base_config = BaseConfig(config_manager)

    assert base_config.model_settings["temperature"] == 0.9
    assert base_config.model_settings["top_k"] == 5

    base_config.update_model_settings({"do_sample": True, "max_tokens": 256})

    persisted = config_manager.get_huggingface_generation_settings()
    assert persisted["do_sample"] is True
    assert persisted["max_tokens"] == 256


def test_base_config_rejects_invalid_generation_settings(tmp_path):
    config_manager = _DummyConfigManager(token=None, cache_dir=str(tmp_path))
    base_config = BaseConfig(config_manager)

    with pytest.raises(ValueError):
        base_config.update_model_settings({"top_k": -1})

    with pytest.raises(ValueError):
        base_config.update_model_settings({"temperature": 3})


@pytest.fixture(autouse=True)
def _patch_heavy_dependencies(monkeypatch):
    monkeypatch.setattr(manager_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(manager_module.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(manager_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(manager_module, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(model_type="gpt2")))
    monkeypatch.setattr(manager_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: _DummyTokenizer()))
    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *a, **k: _DummyModel()))
    monkeypatch.setattr(manager_module, "pipeline", lambda *a, **k: lambda *args, **kwargs: [])
    monkeypatch.setattr(manager_module.psutil, "virtual_memory", lambda: SimpleNamespace(available=1024 * 1024 * 1024))


def test_cache_manager_generates_key_with_binary_payload(tmp_path):
    cache_manager = CacheManager(str(tmp_path / "cache.json"))

    messages = [
        {"role": "user", "content": "hello", "audio": b"\x00\x01"},
        {"role": "assistant", "content": "world", "metadata": {"blob": b"\x02"}},
    ]
    settings = {"temperature": 0.7, "payload": b"\x03\x04"}

    key = cache_manager.generate_cache_key(messages, "test-model", settings)

    assert isinstance(key, str)
    assert key

    # Ensure sanitization is stable for identical inputs
    assert key == cache_manager.generate_cache_key(messages, "test-model", settings)

    # Changing the binary payload should produce a different key
    different_messages = [
        {"role": "user", "content": "hello", "audio": b"\x00\x01\x02"},
        {"role": "assistant", "content": "world", "metadata": {"blob": b"\x02"}},
    ]
    different_key = cache_manager.generate_cache_key(different_messages, "test-model", settings)

    assert key != different_key


def test_load_model_downloads_when_token_present(monkeypatch, tmp_path, caplog):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token="hf_token", cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    snapshot_calls = []

    def _fake_snapshot_download(repo_id, cache_dir, allow_patterns, token, max_workers):
        snapshot_calls.append(
            {
                "repo_id": repo_id,
                "cache_dir": cache_dir,
                "allow_patterns": allow_patterns,
                "token": token,
                "max_workers": max_workers,
            }
        )
        snapshot_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "fake"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = snapshot_dir / "config.json"
        dummy_file.write_bytes(b"data")
        return str(snapshot_dir)

    monkeypatch.setattr(manager_module, "snapshot_download", _fake_snapshot_download)

    config_calls = []

    def _config_from_pretrained(*args, **kwargs):
        config_calls.append((args, kwargs))
        return SimpleNamespace(model_type="gpt2")

    monkeypatch.setattr(manager_module, "AutoConfig", SimpleNamespace(from_pretrained=_config_from_pretrained))

    tokenizer_calls = []

    class _RecordingTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            tokenizer_calls.append((args, kwargs))
            return _DummyTokenizer()

    monkeypatch.setattr(manager_module, "AutoTokenizer", _RecordingTokenizer)

    model_calls = []

    class _RecordingModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            model_calls.append((args, kwargs))
            return _DummyModel()

    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", _RecordingModel)

    caplog.set_level("INFO")
    asyncio.run(manager.load_model("test/model", force_download=True))

    assert len(snapshot_calls) == 1
    snapshot_call = snapshot_calls[0]
    assert snapshot_call["repo_id"] == "test/model"
    assert snapshot_call["token"] == "hf_token"
    assert snapshot_call["allow_patterns"] == manager_module.HuggingFaceModelManager.SNAPSHOT_ALLOW_PATTERNS
    assert snapshot_call["max_workers"] == 4

    assert len(config_calls) == 1
    config_args, config_kwargs = config_calls[0]
    assert config_kwargs["local_files_only"] is True
    assert Path(config_args[0]).name == "fake"

    assert len(tokenizer_calls) == 1
    tokenizer_args, tokenizer_kwargs = tokenizer_calls[0]
    assert tokenizer_kwargs["local_files_only"] is True
    assert Path(tokenizer_args[0]).name == "fake"

    assert len(model_calls) == 1
    model_args, model_kwargs = model_calls[0]
    assert Path(model_args[0]).name == "fake"
    assert model_kwargs["local_files_only"] is True

    summary_message = next(
        (record.message for record in caplog.records if "Cache directory summary" in record.message),
        None,
    )

    assert summary_message is not None
    assert "Cache directory summary" in summary_message
    assert "1 file" in summary_message
    assert "4 B total" in summary_message


def test_load_model_uses_local_when_token_missing(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token=None, cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    model_dir = cache_dir / "models--test--model" / "snapshots" / "local"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("data")

    def _fail_snapshot(*args, **kwargs):
        raise AssertionError("snapshot_download should not be called when using local models")

    monkeypatch.setattr(manager_module, "snapshot_download", _fail_snapshot)

    asyncio.run(manager.load_model("test/model"))

    assert manager.current_model == "test/model"


def test_load_model_handles_multi_gpu_device_map(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token=None, cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    model_dir = cache_dir / "models--test--model" / "snapshots" / "local"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("data")

    monkeypatch.setattr(manager_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(manager_module.torch.cuda, "device_count", lambda: 2)

    call_kwargs = []

    class _RecordingModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            call_kwargs.append(kwargs)
            return _DummyModel()

    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", _RecordingModel)

    asyncio.run(manager.load_model("test/model"))

    assert manager.current_model == "test/model"
    assert len(call_kwargs) == 1
    kwargs = call_kwargs[0]
    assert kwargs["device_map"] == "auto"
    assert kwargs["max_memory"]["cpu"].endswith("B")
    assert kwargs["max_memory"]["cuda:0"].endswith("B")
    assert kwargs["max_memory"]["cuda:1"].endswith("B")


def test_load_model_instantiates_model_once(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token=None, cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    model_dir = cache_dir / "models--test--model" / "snapshots" / "local"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("data")

    call_count = 0

    class _RecordingModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            nonlocal call_count
            call_count += 1
            return _DummyModel()

    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", _RecordingModel)

    asyncio.run(manager.load_model("test/model"))

    assert call_count == 1


def test_unload_model_clears_onnx_sessions(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token=None, cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    model_dir = cache_dir / "models--test--model" / "snapshots" / "local"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("data")

    onnx_path = tmp_path / "model.onnx"
    onnx_path.write_bytes(b"fake")

    class _DummyOrt:
        def __init__(self):
            self.sessions = []

        def InferenceSession(self, model_path, providers=None):
            session = SimpleNamespace(model_path=model_path, providers=tuple(providers or ()))
            self.sessions.append(session)
            return session

    dummy_ort = _DummyOrt()
    monkeypatch.setattr(manager_module, "ort", dummy_ort)

    asyncio.run(
        manager.load_model("test/model", use_onnx=True, onnx_model_path=str(onnx_path))
    )

    assert "test/model" in manager.ort_sessions

    manager.unload_model()

    assert manager.ort_sessions == {}


def test_failed_load_does_not_persist_install(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token="hf_token", cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    model_name = "fail/model"
    model_dir = cache_dir / f"models--{model_name.replace('/', '--')}"

    def _fake_snapshot_download(repo_id, cache_dir, allow_patterns, token, max_workers):
        snapshot_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}" / "snapshots" / "fake"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        (snapshot_dir / "config.json").write_text("data")
        return str(snapshot_dir)

    class _FailingModel:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(manager_module, "snapshot_download", _fake_snapshot_download)
    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", _FailingModel)

    with pytest.raises(ValueError):
        asyncio.run(manager.load_model(model_name, force_download=True))

    assert not model_dir.exists()
    assert manager.current_model is None
    assert manager.installed_models == []

    with open(manager.installed_models_file, "r", encoding="utf-8") as fh:
        assert json.load(fh) == []
