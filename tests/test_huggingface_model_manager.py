import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace


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

    huggingface_hub_stub.InferenceClient = _InferenceClient
    huggingface_hub_stub.HfApi = _HfApi
    huggingface_hub_stub.hf_hub_download = _hf_hub_download
    sys.modules["huggingface_hub"] = huggingface_hub_stub

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

    def get_huggingface_api_key(self):
        return self._token

    def get_model_cache_dir(self):
        return self._cache_dir


class _DummyTokenizer:
    def __call__(self, *args, **kwargs):
        return {"input_ids": []}


class _DummyModel:
    transformer = SimpleNamespace(h=[])


@pytest.fixture(autouse=True)
def _patch_heavy_dependencies(monkeypatch):
    monkeypatch.setattr(manager_module.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(manager_module.torch.cuda, "device_count", lambda: 0)
    monkeypatch.setattr(manager_module.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(manager_module, "AutoConfig", SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace(model_type="gpt2")))
    monkeypatch.setattr(manager_module, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: _DummyTokenizer()))
    monkeypatch.setattr(manager_module, "AutoModelForCausalLM", SimpleNamespace(from_pretrained=lambda *a, **k: _DummyModel()))
    monkeypatch.setattr(manager_module, "pipeline", lambda *a, **k: lambda *args, **kwargs: [])
    monkeypatch.setattr(manager_module, "infer_auto_device_map", lambda *a, **k: {})
    monkeypatch.setattr(manager_module.psutil, "virtual_memory", lambda: SimpleNamespace(available=1024 * 1024 * 1024))


def test_load_model_downloads_when_token_present(monkeypatch, tmp_path, caplog):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    config_manager = _DummyConfigManager(token="hf_token", cache_dir=str(cache_dir))
    base_config = BaseConfig(config_manager)
    cache_manager = CacheManager(str(tmp_path / "cache.json"))
    manager = HuggingFaceModelManager(base_config, NVMeConfig(), cache_manager)

    download_calls = []

    class _DummyApi:
        def list_repo_files(self, repo_id):
            download_calls.append(("list", repo_id))
            return ["config.json"]

    monkeypatch.setattr(manager_module, "HfApi", lambda *a, **k: _DummyApi())

    def _fake_download(repo_id, filename, cache_dir):
        download_calls.append(("download", repo_id, filename))
        model_dir = Path(cache_dir) / f"models--{repo_id.replace('/', '--')}"
        model_dir.mkdir(parents=True, exist_ok=True)
        dummy_file = model_dir / filename
        dummy_file.write_bytes(b"data")
        return str(dummy_file)

    monkeypatch.setattr(manager_module, "hf_hub_download", _fake_download)

    caplog.set_level("INFO")
    asyncio.run(manager.load_model("test/model", force_download=True))

    assert ("list", "test/model") in download_calls
    assert any(call[0] == "download" for call in download_calls)

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

    model_dir = cache_dir / "models--test--model"
    model_dir.mkdir(parents=True)

    def _fail_list_repo_files(*args, **kwargs):
        raise AssertionError("Remote list_repo_files should not be called when using local models")

    monkeypatch.setattr(manager_module, "HfApi", lambda *a, **k: SimpleNamespace(list_repo_files=_fail_list_repo_files))
    def _fail_download(*args, **kwargs):
        raise AssertionError("Remote download should not occur")

    monkeypatch.setattr(manager_module, "hf_hub_download", _fail_download)

    asyncio.run(manager.load_model("test/model"))

    assert manager.current_model == "test/model"
