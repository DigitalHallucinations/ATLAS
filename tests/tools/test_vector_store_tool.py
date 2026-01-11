import asyncio
import importlib.util
import sys
import types
from pathlib import Path

import pytest


def _ensure_stub_modules() -> None:
    if "yaml" not in sys.modules:
        yaml_stub = types.ModuleType("yaml")
        yaml_stub.safe_load = lambda *_args, **_kwargs: {}
        yaml_stub.safe_dump = lambda *_args, **_kwargs: ""
        sys.modules["yaml"] = yaml_stub

    if "dotenv" not in sys.modules:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda *_args, **_kwargs: None
        dotenv_stub.set_key = lambda *_args, **_kwargs: None
        dotenv_stub.find_dotenv = lambda *_args, **_kwargs: ""
        sys.modules["dotenv"] = dotenv_stub

    if "pytz" not in sys.modules:
        pytz_stub = types.ModuleType("pytz")

        class _StubTimezone:
            def localize(self, dt):  # pragma: no cover - deterministic stub
                return dt

        pytz_stub.utc = _StubTimezone()
        pytz_stub.timezone = lambda *_args, **_kwargs: _StubTimezone()
        sys.modules["pytz"] = pytz_stub


def _ensure_package(name: str, path: Path) -> None:
    if name in sys.modules:
        return
    package = types.ModuleType(name)
    package.__path__ = [str(path)]  # type: ignore[attr-defined]
    sys.modules[name] = package


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_ensure_stub_modules()

ROOT = Path(__file__).resolve().parents[2]
_ensure_package("modules", ROOT / "modules")
_ensure_package("modules.Tools", ROOT / "modules" / "Tools")
_ensure_package("modules.Tools.Base_Tools", ROOT / "modules" / "Tools" / "Base_Tools")
_ensure_package("modules.Tools.providers", ROOT / "modules" / "Tools" / "providers")
_ensure_package("modules.Tools.providers.vector_store", ROOT / "modules" / "Tools" / "providers" / "vector_store")

vector_store_module = _load_module(
    "modules.Tools.Base_Tools.vector_store",
    ROOT / "modules" / "Tools" / "Base_Tools" / "vector_store.py",
)
providers_base_module = _load_module(
    "modules.Tools.providers.base",
    ROOT / "modules" / "Tools" / "providers" / "base.py",
)
providers_registry_module = _load_module(
    "modules.Tools.providers.registry",
    ROOT / "modules" / "Tools" / "providers" / "registry.py",
)
_load_module(
    "modules.Tools.providers.vector_store.in_memory",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "in_memory.py",
)
_load_module(
    "modules.Tools.providers.vector_store.chroma",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "chroma.py",
)
_load_module(
    "modules.Tools.providers.vector_store.faiss",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "faiss.py",
)
_load_module(
    "modules.Tools.providers.vector_store.pinecone",
    ROOT / "modules" / "Tools" / "providers" / "vector_store" / "pinecone.py",
)
router_module = _load_module(
    "modules.Tools.providers.router",
    ROOT / "modules" / "Tools" / "providers" / "router.py",
)

ToolProviderRouter = router_module.ToolProviderRouter


class _StubConfigManager:
    def __init__(self, tools_config):
        self._tools_config = tools_config

    def get_config(self, key, default=None):
        if key == "tools":
            return self._tools_config
        return default


def test_upsert_query_delete_roundtrip() -> None:
    async def _run() -> None:
        provider_spec = {
            "name": "vector_store_in_memory",
            "priority": 0,
            "config": {
                "adapter": "in_memory",
            "adapter_config": {
                "index_name": "test-index",
            },
        },
    }

        upsert_router = ToolProviderRouter(
            tool_name="upsert_vectors",
            provider_specs=[provider_spec],
        )
        query_router = ToolProviderRouter(
            tool_name="query_vectors",
            provider_specs=[provider_spec],
        )
        delete_router = ToolProviderRouter(
            tool_name="delete_namespace",
            provider_specs=[provider_spec],
        )

        upsert_result = await upsert_router.call(
            namespace="workspace",
            vectors=[
                {
                    "id": "doc-1",
                    "values": [0.1, 0.2, 0.3],
                    "metadata": {"tag": "alpha", "source": "unit-test"},
                }
            ],
        )

        assert upsert_result == {
            "namespace": "workspace",
            "ids": ["doc-1"],
            "upserted_count": 1,
        }

        query_result = await query_router.call(
            namespace="workspace",
            query=[0.1, 0.2, 0.3],
            top_k=3,
            filter={"tag": "alpha"},
            include_values=True,
        )

        assert query_result["namespace"] == "workspace"
        assert query_result["top_k"] == 3
        assert len(query_result["matches"]) == 1
        match = query_result["matches"][0]
        assert match["id"] == "doc-1"
        assert match["metadata"]["tag"] == "alpha"
        assert isinstance(match["values"], list)
        for expected, actual in zip([0.1, 0.2, 0.3], match["values"]):
            assert actual == pytest.approx(expected)

        delete_result = await delete_router.call(namespace="workspace")
        assert delete_result == {
            "namespace": "workspace",
            "deleted": True,
            "removed_ids": ["doc-1"],
        }

        delete_again = await delete_router.call(namespace="workspace")
        assert delete_again == {
            "namespace": "workspace",
            "deleted": False,
            "removed_ids": [],
        }

    asyncio.run(_run())


def test_upsert_vectors_validation_errors() -> None:
    async def _run() -> None:
        with pytest.raises(vector_store_module.VectorValidationError):
            await vector_store_module.upsert_vectors(
                namespace="",
                vectors=[],
                adapter_name="in_memory",
                adapter_config={"index_name": "validation"},
            )

        with pytest.raises(vector_store_module.VectorValidationError):
            await vector_store_module.upsert_vectors(
                namespace="validation",
                vectors=[{"id": "bad", "values": []}],
                adapter_name="in_memory",
            )

    asyncio.run(_run())


def test_available_adapters_include_registered_providers() -> None:
    adapters = set(vector_store_module.available_vector_store_adapters())
    expected = {"in_memory", "chroma", "faiss", "pinecone"}
    assert expected.issubset(adapters)


def test_build_vector_store_service_uses_config_defaults() -> None:
    async def _run() -> None:
        config = _StubConfigManager(
            {
                "vector_store": {
                    "default_adapter": "in_memory",
                    "adapters": {
                        "in_memory": {
                            "index_name": "config-index",
                        }
                    },
                }
            }
        )

        service = vector_store_module.build_vector_store_service(config_manager=config)
        result = await service.upsert_vectors(
            namespace="config-namespace",
            vectors=[{"id": "cfg", "values": [1.0]}],
        )
        assert result == {
            "namespace": "config-namespace",
            "ids": ["cfg"],
            "upserted_count": 1,
        }

    asyncio.run(_run())
