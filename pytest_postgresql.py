"""Stub :mod:`pytest_postgresql` plugin when the optional dependency is absent."""

from __future__ import annotations

import pytest

pytest_plugins: list[str] = []


@pytest.fixture
def postgresql():
    """Skip tests requiring PostgreSQL when the real plugin is unavailable."""

    pytest.skip("pytest-postgresql is not installed; skipping PostgreSQL-dependent tests")


def pytest_configure(config: pytest.Config) -> None:  # pragma: no cover - pytest hook
    """Register the ``postgresql`` marker to silence unknown marker warnings."""

    config.addinivalue_line(
        "markers",
        "postgresql: mark tests that require a PostgreSQL fixture",
    )
