"""
Tests for PersonaAnalyticsAdapter.

Tests the GTKUI bridge layer for persona analytics.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.services.personas.types import (
    PersonaPerformanceMetrics,
    TokenUsage,
    PersonaVariant,
    PromptRefinement,
    RefinementStatus,
    ImprovementArea,
)
from core.services.common.types import OperationResult

from typing import TYPE_CHECKING, cast

# Import the adapter (handle import error gracefully for test discovery)
_PersonaAnalyticsAdapterType: Any = None
try:
    from GTKUI.Persona_manager.analytics_adapter import (
        PersonaAnalyticsAdapter as _RealAdapter,
    )
    _PersonaAnalyticsAdapterType = _RealAdapter
except ImportError:
    pass

# Skip all tests in this module if the adapter couldn't be imported
pytestmark = pytest.mark.skipif(
    _PersonaAnalyticsAdapterType is None,
    reason="PersonaAnalyticsAdapter not available (GTKUI not installed)",
)

if TYPE_CHECKING:
    from GTKUI.Persona_manager.analytics_adapter import PersonaAnalyticsAdapter
else:
    PersonaAnalyticsAdapter = _PersonaAnalyticsAdapterType


@pytest.fixture
def mock_analytics_service() -> MagicMock:
    """Create a mock PersonaAnalyticsService."""
    service = MagicMock()
    service.get_metrics = AsyncMock()
    service.compare_personas = AsyncMock()
    service.identify_improvement_areas = AsyncMock()
    service.get_variants = AsyncMock()
    service.get_pending_refinements = AsyncMock()
    return service


@pytest.fixture
def sample_metrics() -> PersonaPerformanceMetrics:
    """Create sample performance metrics."""
    return PersonaPerformanceMetrics(
        persona_id="test-persona",
        period_start=datetime.now(timezone.utc) - timedelta(days=7),
        period_end=datetime.now(timezone.utc),
        total_interactions=100,
        avg_response_time_ms=1500.0,
        token_usage=TokenUsage(
            prompt_tokens=5000,
            completion_tokens=3000,
            total_tokens=8000,
        ),
        task_success_rate=0.85,
        user_satisfaction_score=4.2,
        escalation_rate=0.1,
        retry_rate=0.05,
        tools_used={"calculator": 20, "search": 30},
        skills_invoked={"math": 15, "research": 25},
        capability_gaps=["image_generation"],
    )


@pytest.mark.skipif(PersonaAnalyticsAdapter is None, reason="Adapter not available")
class TestPersonaAnalyticsAdapter:
    """Tests for PersonaAnalyticsAdapter."""
    
    def test_init(self, mock_analytics_service: MagicMock) -> None:
        """Test adapter initialization."""
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        assert adapter._service == mock_analytics_service
        assert adapter._metrics_cache == {}
    
    def test_get_persona_metrics_success(
        self,
        mock_analytics_service: MagicMock,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test getting persona metrics successfully."""
        mock_analytics_service.get_metrics.return_value = OperationResult.success(sample_metrics)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        result = adapter.get_persona_metrics("test-persona", use_cache=False)
        
        assert result["persona"] == "test-persona"
        assert result["totals"]["calls"] == 100
        assert result["success_rate"] == 0.85
        assert result["average_latency_ms"] == 1500.0
        assert result["token_usage"]["total_tokens"] == 8000
        assert len(result["tools"]["totals_by_tool"]) == 2
    
    def test_get_persona_metrics_caching(
        self,
        mock_analytics_service: MagicMock,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test that metrics are cached."""
        mock_analytics_service.get_metrics.return_value = OperationResult.success(sample_metrics)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        # First call
        result1 = adapter.get_persona_metrics("test-persona")
        
        # Second call should use cache
        result2 = adapter.get_persona_metrics("test-persona")
        
        # Service should only be called once
        assert mock_analytics_service.get_metrics.call_count == 1
        assert result1 == result2
    
    def test_get_persona_metrics_cache_bypass(
        self,
        mock_analytics_service: MagicMock,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test bypassing cache."""
        mock_analytics_service.get_metrics.return_value = OperationResult.success(sample_metrics)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        # First call
        adapter.get_persona_metrics("test-persona")
        
        # Second call with cache bypass
        adapter.get_persona_metrics("test-persona", use_cache=False)
        
        # Service should be called twice
        assert mock_analytics_service.get_metrics.call_count == 2
    
    def test_get_persona_metrics_error_handling(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test error handling returns empty metrics."""
        mock_analytics_service.get_metrics.side_effect = Exception("Service error")
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        result = adapter.get_persona_metrics("test-persona")
        
        assert result["persona"] == "test-persona"
        assert result["totals"]["calls"] == 0
        assert result["success_rate"] == 0.0
    
    def test_detect_anomalies_high_escalation(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test anomaly detection for high escalation rate."""
        metrics = PersonaPerformanceMetrics(
            persona_id="test",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_interactions=50,
            avg_response_time_ms=1000.0,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            task_success_rate=0.8,
            escalation_rate=0.4,  # High
            retry_rate=0.05,
            tools_used={},
            skills_invoked={},
            capability_gaps=[],
        )
        
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        anomalies = adapter._detect_anomalies(metrics)
        
        assert any(a["type"] == "high_escalation_rate" for a in anomalies)
    
    def test_detect_anomalies_low_success(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test anomaly detection for low success rate."""
        metrics = PersonaPerformanceMetrics(
            persona_id="test",
            period_start=datetime.now(timezone.utc),
            period_end=datetime.now(timezone.utc),
            total_interactions=50,
            avg_response_time_ms=1000.0,
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
            task_success_rate=0.5,  # Low
            escalation_rate=0.1,
            retry_rate=0.05,
            tools_used={},
            skills_invoked={},
            capability_gaps=[],
        )
        
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        anomalies = adapter._detect_anomalies(metrics)
        
        assert any(a["type"] == "low_success_rate" for a in anomalies)
    
    def test_get_comparison_summary(
        self,
        mock_analytics_service: MagicMock,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test getting comparison summary."""
        comparison = {
            "persona1": sample_metrics,
            "persona2": PersonaPerformanceMetrics(
                persona_id="persona2",
                period_start=datetime.now(timezone.utc),
                period_end=datetime.now(timezone.utc),
                total_interactions=80,
                avg_response_time_ms=1200.0,
                token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
                task_success_rate=0.9,
                escalation_rate=0.05,
                retry_rate=0.02,
                tools_used={},
                skills_invoked={},
                capability_gaps=[],
            ),
        }
        mock_analytics_service.compare_personas.return_value = OperationResult.success(comparison)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        result = adapter.get_comparison_summary(["persona1", "persona2"])
        
        assert result["top_performer"]["persona"] == "persona2"
        assert len(result["personas"]) == 2
    
    def test_get_improvement_areas(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test getting improvement areas."""
        areas = [
            ImprovementArea(
                area="response_time",
                priority="high",
                current_value=2000.0,
                target_value=1000.0,
                suggestions=["Optimize prompts", "Use faster model"],
            ),
        ]
        mock_analytics_service.identify_improvement_areas.return_value = OperationResult.success(areas)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        result = adapter.get_improvement_areas("test-persona")
        
        assert len(result) == 1
        assert result[0]["area"] == "response_time"
        assert result[0]["priority"] == "high"
        assert len(result[0]["suggestions"]) == 2
    
    def test_get_variants(
        self,
        mock_analytics_service: MagicMock,
    ) -> None:
        """Test getting A/B test variants."""
        variants = [
            PersonaVariant(
                variant_id="v1",
                base_persona_id="test-persona",
                modifications={"temperature": 0.8},
                traffic_percentage=0.2,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            ),
        ]
        mock_analytics_service.get_variants.return_value = OperationResult.success(variants)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        result = adapter.get_variants("test-persona")
        
        assert len(result) == 1
        assert result[0]["variant_id"] == "v1"
        assert result[0]["traffic_percentage"] == 0.2
    
    def test_invalidate_cache_specific_persona(
        self,
        mock_analytics_service: MagicMock,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test invalidating cache for specific persona."""
        mock_analytics_service.get_metrics.return_value = OperationResult.success(sample_metrics)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        # Populate cache
        adapter.get_persona_metrics("persona1")
        adapter.get_persona_metrics("persona2")
        
        # Invalidate only persona1
        adapter.invalidate_cache("persona1")
        
        # persona1 should need refresh, persona2 should still be cached
        adapter.get_persona_metrics("persona1")
        adapter.get_persona_metrics("persona2")
        
        # persona1 called twice, persona2 called once
        # get_metrics is called with (actor, persona_id, ...) - persona_id is second arg
        calls = [call[0][1] for call in mock_analytics_service.get_metrics.call_args_list]
        assert calls.count("persona1") == 2
        assert calls.count("persona2") == 1
    
    def test_invalidate_cache_all(
        self,
        mock_analytics_service: MagicMock,
        sample_metrics: PersonaPerformanceMetrics,
    ) -> None:
        """Test invalidating entire cache."""
        mock_analytics_service.get_metrics.return_value = OperationResult.success(sample_metrics)
        adapter = PersonaAnalyticsAdapter(mock_analytics_service)
        
        # Populate cache
        adapter.get_persona_metrics("persona1")
        adapter.get_persona_metrics("persona2")
        
        # Invalidate all
        adapter.invalidate_cache()
        
        # Both should need refresh
        adapter.get_persona_metrics("persona1")
        adapter.get_persona_metrics("persona2")
        
        assert mock_analytics_service.get_metrics.call_count == 4
