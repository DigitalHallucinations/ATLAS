import asyncio
import importlib.util
import shutil
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


ROOT = Path(__file__).resolve().parents[1]
_ensure_stub_modules()
_ensure_package("modules", ROOT / "modules")
_ensure_package("modules.Tools", ROOT / "modules" / "Tools")
_ensure_package("modules.Tools.Base_Tools", ROOT / "modules" / "Tools" / "Base_Tools")
_ensure_package("modules.Tools.providers", ROOT / "modules" / "Tools" / "providers")

kv_store_module = _load_module(
    "modules.Tools.Base_Tools.kv_store",
    ROOT / "modules" / "Tools" / "Base_Tools" / "kv_store.py",
)
providers_base_module = _load_module(
    "modules.Tools.providers.base",
    ROOT / "modules" / "Tools" / "providers" / "base.py",
)
providers_registry_module = _load_module(
    "modules.Tools.providers.registry",
    ROOT / "modules" / "Tools" / "providers" / "registry.py",
)
kv_provider_module = _load_module(
    "modules.Tools.providers.kv_store",
    ROOT / "modules" / "Tools" / "providers" / "kv_store.py",
)
router_module = _load_module(
    "modules.Tools.providers.router",
    ROOT / "modules" / "Tools" / "providers" / "router.py",
)

PG_CTL = shutil.which("pg_ctl")

KeyValueStoreService = kv_store_module.KeyValueStoreService
KeyValueStoreError = kv_store_module.KeyValueStoreError
NamespaceQuotaExceededError = kv_store_module.NamespaceQuotaExceededError

ToolProviderRouter = router_module.ToolProviderRouter


def test_available_kv_adapters_expose_sqlite():
    adapters = kv_store_module.available_kv_store_adapters()
    assert "sqlite" in adapters


def test_adapter_defaults_require_persistence_section():
    class LegacyConfigManager:
        persistence = None

    with pytest.raises(KeyValueStoreError, match="kv_store\\.get_settings"):
        kv_store_module.create_kv_store_adapter(
            "postgres",
            config_manager=LegacyConfigManager(),
        )


def _normalize_dsn(dsn: str) -> str:
    if dsn.startswith("postgresql+psycopg://"):
        return dsn
    if dsn.startswith("postgresql://"):
        return "postgresql+psycopg://" + dsn[len("postgresql://"):]
    return dsn


def _build_service(
    dsn: str,
    *,
    namespace_quota: int = 4096,
    global_quota: int = 8192,
) -> KeyValueStoreService:
    adapter_config = {
        "url": _normalize_dsn(dsn),
        "reuse_conversation_store": False,
        "namespace_quota_bytes": namespace_quota,
        "global_quota_bytes": global_quota,
    }
    return kv_store_module.build_kv_store_service(
        adapter_name="postgres",
        adapter_config=adapter_config,
        config_manager=None,
    )


@pytest.mark.skipif(PG_CTL is None, reason="pg_ctl executable not available")
def test_concurrent_increments_are_atomic(postgresql) -> None:
    async def _run() -> None:
        service = _build_service(postgresql.dsn())

        async def _worker(count: int) -> None:
            for _ in range(count):
                await service.increment_value("metrics", "counter", delta=1)

        await asyncio.gather(*(_worker(50) for _ in range(8)))
        result = await service.get_value("metrics", "counter")
        assert result["value"] == 400

    asyncio.run(_run())


@pytest.mark.skipif(PG_CTL is None, reason="pg_ctl executable not available")
def test_ttl_expiry_removes_values(postgresql) -> None:
    async def _run() -> None:
        service = _build_service(postgresql.dsn())
        await service.set_value("session", "token", {"scopes": ["a", "b"]}, ttl_seconds=0.2)
        first = await service.get_value("session", "token")
        assert first["found"] is True
        assert first["ttl_seconds"] == pytest.approx(0.2, abs=0.1)
        await asyncio.sleep(0.35)
        second = await service.get_value("session", "token")
        assert second["found"] is False

    asyncio.run(_run())


@pytest.mark.skipif(PG_CTL is None, reason="pg_ctl executable not available")
def test_quota_enforcement(postgresql) -> None:
    async def _run() -> None:
        service = _build_service(postgresql.dsn(), namespace_quota=128, global_quota=256)
        small_payload = "x" * 32
        await service.set_value("caps", "small", small_payload)
        large_payload = "x" * 512
        with pytest.raises(NamespaceQuotaExceededError):
            await service.set_value("caps", "too_big", large_payload)

    asyncio.run(_run())


@pytest.mark.skipif(PG_CTL is None, reason="pg_ctl executable not available")
def test_namespace_isolation(postgresql) -> None:
    async def _run() -> None:
        service = _build_service(postgresql.dsn())
        await service.set_value("alpha", "shared", "first")
        await service.set_value("beta", "shared", "second")

        alpha = await service.get_value("alpha", "shared")
        beta = await service.get_value("beta", "shared")

        assert alpha["value"] == "first"
        assert beta["value"] == "second"

    asyncio.run(_run())
