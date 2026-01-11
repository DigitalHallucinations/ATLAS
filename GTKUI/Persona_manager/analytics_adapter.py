"""
GTKUI adapter for PersonaAnalyticsService.

Bridges between the new PersonaAnalyticsService and the existing
persona management UI. Provides synchronous wrappers and format
conversion for GTK callbacks.

Author: ATLAS Team
Date: Jan 11, 2026
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from core.services.personas import (
        PersonaAnalyticsService,
        PersonaPerformanceMetrics,
    )
    from core.services.common import Actor, OperationResult

logger = logging.getLogger(__name__)


def _create_default_actor() -> "Actor":
    """Create a default system actor for UI operations."""
    from core.services.common import Actor
    return Actor(
        type="system",
        id="gtkui-adapter",
        tenant_id="default",
        permissions={"personas:read", "analytics:read"},
    )


class PersonaAnalyticsAdapter:
    """
    Adapter for PersonaAnalyticsService to work with GTKUI.
    
    Provides:
    - Synchronous wrappers for async service methods
    - Format conversion from service DTOs to UI-expected dictionaries
    - Caching for dashboard refresh
    - Comparison data aggregation
    """
    
    def __init__(
        self,
        analytics_service: "PersonaAnalyticsService",
        actor: Optional["Actor"] = None,
    ) -> None:
        """
        Initialize the adapter.
        
        Args:
            analytics_service: The PersonaAnalyticsService instance
            actor: The actor for permission checks
        """
        self._service = analytics_service
        self._actor = actor or _create_default_actor()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        
        # Cache for dashboard updates
        self._metrics_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = timedelta(seconds=30)
        self._cache_timestamps: Dict[str, datetime] = {}
    
    def _get_loop(self) -> asyncio.AbstractEventLoop:
        """Get or create an event loop for running async code."""
        if self._loop is None or self._loop.is_closed():
            try:
                self._loop = asyncio.get_running_loop()
            except RuntimeError:
                self._loop = asyncio.new_event_loop()
        return self._loop
    
    def _run_async(self, coro: Any) -> Any:
        """Run an async coroutine synchronously."""
        loop = self._get_loop()
        if loop.is_running():
            # Schedule in running loop (GTK may have one)
            future = asyncio.run_coroutine_threadsafe(coro, loop)
            return future.result(timeout=30)
        else:
            return loop.run_until_complete(coro)
    
    # =========================================================================
    # Dashboard Integration Methods
    # =========================================================================
    
    def get_persona_metrics(
        self,
        persona_id: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        use_cache: bool = True,
    ) -> Dict[str, Any]:
        """
        Get persona performance metrics for the dashboard.
        
        Args:
            persona_id: The persona to get metrics for
            start: Start of the time window
            end: End of the time window
            use_cache: Whether to use cached data if available
            
        Returns:
            Dictionary with metrics in UI-expected format:
            {
                "persona": str,
                "window": {"start": datetime, "end": datetime},
                "totals": {"calls": int, "success": int, "failure": int},
                "success_rate": float,
                "average_latency_ms": float,
                "recent": [...],
                "recent_anomalies": [...],
                "tools": {"totals_by_tool": [...]},
                "skills": {"totals_by_skill": [...]},
            }
        """
        cache_key = f"{persona_id}:{start}:{end}"
        
        if use_cache and cache_key in self._metrics_cache:
            cached_at = self._cache_timestamps.get(cache_key)
            if cached_at and (datetime.now(timezone.utc) - cached_at) < self._cache_ttl:
                return self._metrics_cache[cache_key]
        
        try:
            op_result = self._run_async(
                self._service.get_metrics(
                    self._actor,
                    persona_id,
                    period_days=7,
                    start_date=start,
                    end_date=end,
                )
            )
            
            # Extract metrics from OperationResult
            if hasattr(op_result, 'is_success') and op_result.is_success:
                metrics = op_result.value
            else:
                metrics = None
                
            result = self._format_metrics_for_ui(persona_id, metrics, start, end)
            
            # Cache the result
            self._metrics_cache[cache_key] = result
            self._cache_timestamps[cache_key] = datetime.now(timezone.utc)
            
            return result
        except Exception as e:
            logger.warning("Failed to get persona metrics: %s", e)
            return self._empty_metrics(persona_id, start, end)
    
    def _format_metrics_for_ui(
        self,
        persona_id: str,
        metrics: Optional["PersonaPerformanceMetrics"],
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> Dict[str, Any]:
        """Convert PersonaPerformanceMetrics to UI dictionary format."""
        if metrics is None:
            return self._empty_metrics(persona_id, start, end)
        
        return {
            "persona": persona_id,
            "window": {
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
            },
            "totals": {
                "calls": metrics.total_interactions,
                "success": int(metrics.total_interactions * metrics.task_success_rate),
                "failure": int(metrics.total_interactions * (1 - metrics.task_success_rate)),
            },
            "success_rate": metrics.task_success_rate,
            "average_latency_ms": metrics.avg_response_time_ms,
            "recent": [],  # Would be populated from interaction log
            "recent_anomalies": self._detect_anomalies(metrics),
            "tools": {
                "totals_by_tool": [
                    {"name": name, "calls": count}
                    for name, count in (metrics.tools_used or {}).items()
                ],
            },
            "skills": {
                "totals_by_skill": [
                    {"name": name, "calls": count}
                    for name, count in (metrics.skills_invoked or {}).items()
                ],
            },
            "token_usage": {
                "prompt_tokens": metrics.token_usage.prompt_tokens if metrics.token_usage else 0,
                "completion_tokens": metrics.token_usage.completion_tokens if metrics.token_usage else 0,
                "total_tokens": metrics.token_usage.total_tokens if metrics.token_usage else 0,
            },
            "escalation_rate": metrics.escalation_rate,
            "retry_rate": metrics.retry_rate,
            "capability_gaps": list(metrics.capability_gaps or []),
        }
    
    def _empty_metrics(
        self,
        persona_id: str,
        start: Optional[datetime],
        end: Optional[datetime],
    ) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "persona": persona_id,
            "window": {
                "start": start.isoformat() if start else None,
                "end": end.isoformat() if end else None,
            },
            "totals": {"calls": 0, "success": 0, "failure": 0},
            "success_rate": 0.0,
            "average_latency_ms": 0.0,
            "recent": [],
            "recent_anomalies": [],
            "tools": {"totals_by_tool": []},
            "skills": {"totals_by_skill": []},
            "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "escalation_rate": 0.0,
            "retry_rate": 0.0,
            "capability_gaps": [],
        }
    
    def _detect_anomalies(
        self,
        metrics: "PersonaPerformanceMetrics",
    ) -> List[Dict[str, Any]]:
        """Detect anomalies from metrics data."""
        anomalies: List[Dict[str, Any]] = []
        
        # High escalation rate
        if metrics.escalation_rate > 0.3:
            anomalies.append({
                "type": "high_escalation_rate",
                "metric": "escalation_rate",
                "value": metrics.escalation_rate,
                "threshold": 0.3,
                "severity": "warning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        # High retry rate
        if metrics.retry_rate > 0.2:
            anomalies.append({
                "type": "high_retry_rate",
                "metric": "retry_rate",
                "value": metrics.retry_rate,
                "threshold": 0.2,
                "severity": "warning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        # Low success rate
        if metrics.task_success_rate < 0.7 and metrics.total_interactions > 10:
            anomalies.append({
                "type": "low_success_rate",
                "metric": "task_success_rate",
                "value": metrics.task_success_rate,
                "threshold": 0.7,
                "severity": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        # High latency
        if metrics.avg_response_time_ms > 5000:
            anomalies.append({
                "type": "high_latency",
                "metric": "avg_response_time_ms",
                "value": metrics.avg_response_time_ms,
                "threshold": 5000,
                "severity": "warning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        
        return anomalies
    
    def get_comparison_summary(
        self,
        persona_ids: Optional[List[str]] = None,
        *,
        category: str = "tool",
        recent: int = 5,
    ) -> Dict[str, Any]:
        """
        Get comparison summary across personas for leaderboard widgets.
        
        Returns:
            Dictionary with:
            {
                "top_performer": {"persona": str, "success_rate": float},
                "highest_failure_rate": {"persona": str, "failure_rate": float},
                "fastest_latency": {"persona": str, "avg_latency_ms": float},
                "personas": [...comparison data...],
            }
        """
        try:
            # Get comparison data from service
            op_result = self._run_async(
                self._service.compare_personas(
                    self._actor,
                    persona_ids or [],
                    period_days=7,
                )
            )
            
            # Extract comparison from OperationResult
            if hasattr(op_result, 'is_success') and op_result.is_success:
                comparison = op_result.value
            else:
                comparison = None
            
            if not comparison:
                return self._empty_comparison()
            
            return self._format_comparison_for_ui(comparison)
        except Exception as e:
            logger.warning("Failed to get comparison summary: %s", e)
            return self._empty_comparison()
    
    def _format_comparison_for_ui(
        self,
        comparison: Dict[str, "PersonaPerformanceMetrics"],
    ) -> Dict[str, Any]:
        """Format comparison data for UI widgets."""
        personas: List[Dict[str, Any]] = []
        
        for persona_id, metrics in comparison.items():
            personas.append({
                "persona": persona_id,
                "success_rate": metrics.task_success_rate,
                "failure_rate": 1 - metrics.task_success_rate,
                "avg_latency_ms": metrics.avg_response_time_ms,
                "total_interactions": metrics.total_interactions,
            })
        
        # Sort for leaderboard
        by_success = sorted(personas, key=lambda x: x["success_rate"], reverse=True)
        by_failure = sorted(personas, key=lambda x: x["failure_rate"], reverse=True)
        by_latency = sorted(personas, key=lambda x: x["avg_latency_ms"])
        
        return {
            "top_performer": by_success[0] if by_success else None,
            "highest_failure_rate": by_failure[0] if by_failure else None,
            "fastest_latency": by_latency[0] if by_latency else None,
            "personas": personas,
        }
    
    def _empty_comparison(self) -> Dict[str, Any]:
        """Return empty comparison structure."""
        return {
            "top_performer": None,
            "highest_failure_rate": None,
            "fastest_latency": None,
            "personas": [],
        }
    
    # =========================================================================
    # Improvement Areas Widget
    # =========================================================================
    
    def get_improvement_areas(
        self,
        persona_id: str,
    ) -> List[Dict[str, Any]]:
        """
        Get improvement recommendations for a persona.
        
        Returns list of improvement areas with priority and suggestions.
        """
        try:
            op_result = self._run_async(
                self._service.identify_improvement_areas(
                    self._actor,
                    persona_id,
                )
            )
            
            # Extract areas from OperationResult
            if hasattr(op_result, 'is_success') and op_result.is_success:
                areas = op_result.value or []
            else:
                areas = []
            
            return [
                {
                    "area": area.get("area", "") if isinstance(area, dict) else getattr(area, "area", ""),
                    "priority": area.get("priority", "medium") if isinstance(area, dict) else getattr(area, "priority", "medium"),
                    "current_value": area.get("current_value", 0) if isinstance(area, dict) else getattr(area, "current_value", 0),
                    "target_value": area.get("target_value", 0) if isinstance(area, dict) else getattr(area, "target_value", 0),
                    "suggestions": list(area.get("suggestions", []) if isinstance(area, dict) else getattr(area, "suggestions", [])),
                }
                for area in areas
            ]
        except Exception as e:
            logger.warning("Failed to get improvement areas: %s", e)
            return []
    
    # =========================================================================
    # A/B Testing Widget Support
    # =========================================================================
    
    def get_variants(self, persona_id: str) -> List[Dict[str, Any]]:
        """Get A/B test variants for a persona."""
        try:
            op_result = self._run_async(
                self._service.get_variants(self._actor, persona_id)
            )
            
            # Extract variants from OperationResult
            if hasattr(op_result, 'is_success') and op_result.is_success:
                variants = op_result.value or []
            else:
                variants = []
                
            return [
                {
                    "variant_id": v.variant_id,
                    "base_persona_id": v.base_persona_id,
                    "modifications": dict(v.modifications or {}),
                    "traffic_percentage": v.traffic_percentage,
                    "is_active": v.is_active,
                    "created_at": v.created_at.isoformat() if v.created_at else None,
                }
                for v in variants
            ]
        except Exception as e:
            logger.warning("Failed to get variants: %s", e)
            return []

    # =========================================================================
    # Pending Refinements Widget
    # =========================================================================
    
    def get_pending_refinements(self, persona_id: str) -> List[Dict[str, Any]]:
        """Get pending prompt refinements for review."""
        try:
            op_result = self._run_async(
                self._service.get_pending_refinements(self._actor, persona_id)
            )
            
            # Extract refinements from OperationResult
            if hasattr(op_result, 'is_success') and op_result.is_success:
                refinements = op_result.value or []
            else:
                refinements = []
                
            return [
                {
                    "refinement_id": r.refinement_id,
                    "persona_id": r.persona_id,
                    "suggestion": getattr(r, 'suggested_value', ''),
                    "rationale": getattr(r, 'reason', ''),
                    "expected_improvement": getattr(r, 'impact_score', None),
                    "status": r.status if isinstance(r.status, str) else str(r.status),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                }
                for r in (refinements or [])
            ]
        except Exception as e:
            logger.warning("Failed to get pending refinements: %s", e)
            return []
    
    def invalidate_cache(self, persona_id: Optional[str] = None) -> None:
        """Invalidate cached metrics."""
        if persona_id:
            keys_to_remove = [
                k for k in self._metrics_cache if k.startswith(f"{persona_id}:")
            ]
            for key in keys_to_remove:
                self._metrics_cache.pop(key, None)
                self._cache_timestamps.pop(key, None)
        else:
            self._metrics_cache.clear()
            self._cache_timestamps.clear()
