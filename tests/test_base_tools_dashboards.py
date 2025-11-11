"""Smoke tests for Base Tools dashboard helpers."""

import asyncio

from modules.Tools.Base_Tools.analytics_dashboard import AnalyticsDashboardClient
from modules.Tools.Base_Tools.atlas_dashboard import AtlasDashboardClient
from modules.Tools.Base_Tools.dashboard_service import DashboardService


def test_dashboard_service_filters_non_numeric_metrics() -> None:
    service = DashboardService()

    result = asyncio.run(
        service.run(
            dashboard_id="main",
            metrics={"valid": "1.5", "invalid": object()},
        )
    )

    assert result["metrics"] == {"valid": 1.5}


def test_analytics_dashboard_normalizes_metrics() -> None:
    client = AnalyticsDashboardClient()

    result = asyncio.run(
        client.run(
            dashboard_id="analytics",
            summary="Daily summary",
            metrics={"converted": 7, "skip": "abc"},
            segments=[{"cohort": "A"}],
            tags=["Alpha", "alpha", "  "],
            metadata={"source": "unit"},
        )
    )

    assert result["metrics"] == {"converted": 7.0}
    assert result["tags"] == ("alpha",)
    assert result["segments"] == ({"cohort": "A"},)


def test_atlas_dashboard_accepts_optional_metrics() -> None:
    client = AtlasDashboardClient()

    result = asyncio.run(
        client.run(
            initiative="Q1",
            health="GREEN",
            summary="Status is nominal",
            metrics=None,
            stakeholders=["Ada", "Ada", "  "],
        )
    )

    assert result["metrics"] == {}
    assert result["health"] == "green"
    assert result["stakeholders"] == ("Ada",)
