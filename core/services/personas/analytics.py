"""
Persona analytics service for ATLAS.

Provides performance tracking, metrics collection, and
self-improvement capabilities for personas.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import PersonaError
from .permissions import PersonaPermissionChecker
from .types import (
    # Analytics types
    PersonaPerformanceMetrics,
    TokenUsage,
    PersonaVariant,
    PromptRefinement,
    # Events
    PersonaMetricsRecorded,
)

if TYPE_CHECKING:
    from core.config import ConfigManager
    from core.messaging import MessageBus


logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


class PersonaAnalyticsError(PersonaError):
    """Error related to persona analytics operations."""
    pass


class PersonaAnalyticsService:
    """
    Service for tracking and analyzing persona performance.
    
    Provides:
    - Interaction recording and metrics
    - Performance analytics per persona
    - A/B testing infrastructure
    - Prompt refinement suggestions
    - Improvement identification
    """
    
    # Default metrics retention period
    DEFAULT_RETENTION_DAYS = 90
    
    # Minimum interactions for statistical significance
    MIN_INTERACTIONS_FOR_STATS = 30
    
    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[PersonaPermissionChecker] = None,
    ) -> None:
        """
        Initialize the PersonaAnalyticsService.
        
        Args:
            config_manager: Configuration manager
            message_bus: Message bus for publishing events
            permission_checker: Permission checker for authorization
        """
        self._config_manager = config_manager
        self._message_bus = message_bus
        self._permission_checker = permission_checker or PersonaPermissionChecker()
        
        # In-memory storage (would be persisted in production)
        # Raw interaction records
        self._interactions: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Aggregated metrics by period
        self._metrics_cache: Dict[str, PersonaPerformanceMetrics] = {}
        
        # A/B testing variants
        self._variants: Dict[str, List[PersonaVariant]] = defaultdict(list)
        
        # Prompt refinements
        self._refinements: Dict[str, List[PromptRefinement]] = defaultdict(list)
    
    # =========================================================================
    # Interaction Recording
    # =========================================================================
    
    async def record_interaction(
        self,
        persona_id: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        response_time_ms: int = 0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        tools_used: Optional[List[str]] = None,
        skills_used: Optional[List[str]] = None,
        success: bool = True,
        error: Optional[str] = None,
        user_feedback: Optional[float] = None,  # 0-5 rating
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[Dict[str, Any]]:
        """
        Record an interaction with a persona.
        
        Args:
            persona_id: ID of the persona
            user_id: ID of the user
            conversation_id: Optional conversation ID
            response_time_ms: Response time in milliseconds
            prompt_tokens: Prompt tokens used
            completion_tokens: Completion tokens used
            tools_used: List of tools invoked
            skills_used: List of skills invoked
            success: Whether interaction was successful
            error: Error message if failed
            user_feedback: Optional user feedback score
            metadata: Additional metadata
            
        Returns:
            OperationResult containing the recorded interaction
        """
        try:
            interaction = {
                "persona_id": persona_id,
                "user_id": user_id,
                "conversation_id": conversation_id,
                "timestamp": _now_utc().isoformat(),
                "response_time_ms": response_time_ms,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "tools_used": tools_used or [],
                "skills_used": skills_used or [],
                "success": success,
                "error": error,
                "user_feedback": user_feedback,
                "metadata": metadata or {},
            }
            
            self._interactions[persona_id].append(interaction)
            
            # Cleanup old interactions
            await self._cleanup_old_interactions(persona_id)
            
            # Invalidate metrics cache
            self._invalidate_metrics_cache(persona_id)
            
            return OperationResult.success(interaction)
            
        except Exception as e:
            logger.exception(f"Error recording interaction for {persona_id}")
            return OperationResult.failure(f"Failed to record interaction: {e}")
    
    async def record_escalation(
        self,
        persona_id: str,
        user_id: str,
        to_persona: Optional[str] = None,
        reason: str = "",
    ) -> OperationResult[None]:
        """
        Record an escalation event.
        
        Args:
            persona_id: ID of the persona that escalated
            user_id: ID of the user
            to_persona: Persona escalated to (None = human)
            reason: Reason for escalation
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            interaction = {
                "persona_id": persona_id,
                "user_id": user_id,
                "timestamp": _now_utc().isoformat(),
                "type": "escalation",
                "to_persona": to_persona,
                "reason": reason,
            }
            
            self._interactions[persona_id].append(interaction)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error recording escalation for {persona_id}")
            return OperationResult.failure(f"Failed to record escalation: {e}")
    
    async def record_retry(
        self,
        persona_id: str,
        user_id: str,
        original_interaction_id: Optional[str] = None,
        reason: str = "",
    ) -> OperationResult[None]:
        """
        Record a retry event (user asked to regenerate response).
        
        Args:
            persona_id: ID of the persona
            user_id: ID of the user
            original_interaction_id: ID of the original interaction
            reason: Reason for retry
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            interaction = {
                "persona_id": persona_id,
                "user_id": user_id,
                "timestamp": _now_utc().isoformat(),
                "type": "retry",
                "original_interaction_id": original_interaction_id,
                "reason": reason,
            }
            
            self._interactions[persona_id].append(interaction)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error recording retry for {persona_id}")
            return OperationResult.failure(f"Failed to record retry: {e}")
    
    async def record_capability_gap(
        self,
        persona_id: str,
        user_id: str,
        requested_capability: str,
        request_context: str = "",
    ) -> OperationResult[None]:
        """
        Record when a user requests something the persona can't do.
        
        Args:
            persona_id: ID of the persona
            user_id: ID of the user
            requested_capability: What was requested
            request_context: Context of the request
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            interaction = {
                "persona_id": persona_id,
                "user_id": user_id,
                "timestamp": _now_utc().isoformat(),
                "type": "capability_gap",
                "requested_capability": requested_capability,
                "request_context": request_context,
            }
            
            self._interactions[persona_id].append(interaction)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error recording capability gap for {persona_id}")
            return OperationResult.failure(f"Failed to record capability gap: {e}")
    
    # =========================================================================
    # Metrics Retrieval
    # =========================================================================
    
    async def get_metrics(
        self,
        actor: Actor,
        persona_id: str,
        period_days: int = 7,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> OperationResult[PersonaPerformanceMetrics]:
        """
        Get performance metrics for a persona.
        
        Args:
            actor: The actor requesting metrics
            persona_id: ID of the persona
            period_days: Number of days to aggregate (default 7)
            start_date: Optional start date (overrides period_days)
            end_date: Optional end date (defaults to now)
            
        Returns:
            OperationResult containing performance metrics
        """
        try:
            if not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot read metrics for persona {persona_id}"
                )
            
            end_date = end_date or _now_utc()
            start_date = start_date or (end_date - timedelta(days=period_days))
            
            # Check cache
            cache_key = f"{persona_id}:{start_date.date()}:{end_date.date()}"
            if cache_key in self._metrics_cache:
                return OperationResult.success(self._metrics_cache[cache_key])
            
            # Calculate metrics
            metrics = await self._calculate_metrics(persona_id, start_date, end_date)
            
            # Cache result
            self._metrics_cache[cache_key] = metrics
            
            return OperationResult.success(metrics)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting metrics for {persona_id}")
            return OperationResult.failure(f"Failed to get metrics: {e}")
    
    async def _calculate_metrics(
        self,
        persona_id: str,
        start_date: datetime,
        end_date: datetime,
    ) -> PersonaPerformanceMetrics:
        """Calculate performance metrics for a period."""
        interactions = self._interactions.get(persona_id, [])
        
        # Filter by date range
        filtered = []
        for i in interactions:
            ts = datetime.fromisoformat(i["timestamp"].replace("Z", "+00:00"))
            if start_date <= ts <= end_date:
                filtered.append(i)
        
        # Separate by type
        regular = [i for i in filtered if i.get("type") is None]
        escalations = [i for i in filtered if i.get("type") == "escalation"]
        retries = [i for i in filtered if i.get("type") == "retry"]
        gaps = [i for i in filtered if i.get("type") == "capability_gap"]
        
        # Calculate metrics
        total_interactions = len(regular)
        conversations = set(i.get("conversation_id") for i in regular if i.get("conversation_id"))
        
        response_times = [i["response_time_ms"] for i in regular if i.get("response_time_ms", 0) > 0]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0.0
        
        successes = sum(1 for i in regular if i.get("success", True))
        success_rate = successes / total_interactions if total_interactions > 0 else 0.0
        
        errors = sum(1 for i in regular if not i.get("success", True))
        error_rate = errors / total_interactions if total_interactions > 0 else 0.0
        
        # User satisfaction
        feedback_scores = [i["user_feedback"] for i in regular if i.get("user_feedback") is not None]
        avg_satisfaction = sum(feedback_scores) / len(feedback_scores) if feedback_scores else None
        
        # Token usage
        prompt_tokens = sum(i.get("prompt_tokens", 0) for i in regular)
        completion_tokens = sum(i.get("completion_tokens", 0) for i in regular)
        
        # Tool/skill usage
        tools_used: Dict[str, int] = defaultdict(int)
        skills_used: Dict[str, int] = defaultdict(int)
        for i in regular:
            for tool in i.get("tools_used", []):
                tools_used[tool] += 1
            for skill in i.get("skills_used", []):
                skills_used[skill] += 1
        
        # Capability gaps
        capability_gaps = list(set(i.get("requested_capability", "") for i in gaps if i.get("requested_capability")))
        
        # Rates
        escalation_rate = len(escalations) / total_interactions if total_interactions > 0 else 0.0
        retry_rate = len(retries) / total_interactions if total_interactions > 0 else 0.0
        
        return PersonaPerformanceMetrics(
            persona_id=persona_id,
            period_start=start_date,
            period_end=end_date,
            total_interactions=total_interactions,
            total_conversations=len(conversations),
            avg_response_time_ms=avg_response_time,
            token_usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            task_success_rate=success_rate,
            user_satisfaction_score=avg_satisfaction,
            escalation_rate=escalation_rate,
            retry_rate=retry_rate,
            error_rate=error_rate,
            tools_used=dict(tools_used),
            skills_invoked=dict(skills_used),
            capability_gaps=capability_gaps,
        )
    
    async def compare_personas(
        self,
        actor: Actor,
        persona_ids: List[str],
        period_days: int = 7,
    ) -> OperationResult[Dict[str, PersonaPerformanceMetrics]]:
        """
        Compare performance metrics across multiple personas.
        
        Args:
            actor: The actor requesting comparison
            persona_ids: List of persona IDs to compare
            period_days: Period to compare over
            
        Returns:
            OperationResult containing metrics dict keyed by persona ID
        """
        try:
            results = {}
            for persona_id in persona_ids:
                result = await self.get_metrics(actor, persona_id, period_days)
                if result.is_success:
                    results[persona_id] = result.value
            
            return OperationResult.success(results)
            
        except Exception as e:
            logger.exception("Error comparing personas")
            return OperationResult.failure(f"Failed to compare personas: {e}")
    
    # =========================================================================
    # Improvement Identification
    # =========================================================================
    
    async def identify_improvement_areas(
        self,
        actor: Actor,
        persona_id: str,
        period_days: int = 30,
    ) -> OperationResult[List[Dict[str, Any]]]:
        """
        Identify areas where a persona could be improved.
        
        Args:
            actor: The actor requesting analysis
            persona_id: ID of the persona
            period_days: Period to analyze
            
        Returns:
            OperationResult containing list of improvement suggestions
        """
        try:
            if not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot analyze persona {persona_id}"
                )
            
            metrics_result = await self.get_metrics(actor, persona_id, period_days)
            if not metrics_result.is_success:
                return metrics_result
            
            metrics = metrics_result.value
            suggestions: List[Dict[str, Any]] = []
            
            # Not enough data
            if metrics.total_interactions < self.MIN_INTERACTIONS_FOR_STATS:
                suggestions.append({
                    "area": "data",
                    "severity": "info",
                    "message": f"Insufficient data ({metrics.total_interactions} interactions). "
                               f"Need at least {self.MIN_INTERACTIONS_FOR_STATS} for statistical significance.",
                    "recommendation": "Continue using this persona to gather more data.",
                })
                return OperationResult.success(suggestions)
            
            # High error rate
            if metrics.error_rate > 0.1:
                suggestions.append({
                    "area": "reliability",
                    "severity": "high",
                    "message": f"High error rate: {metrics.error_rate:.1%}",
                    "recommendation": "Review error logs and common failure patterns. "
                                     "Consider adjusting prompts or adding error handling.",
                })
            
            # High retry rate
            if metrics.retry_rate > 0.15:
                suggestions.append({
                    "area": "quality",
                    "severity": "high",
                    "message": f"High retry rate: {metrics.retry_rate:.1%}",
                    "recommendation": "Users frequently request regeneration. "
                                     "Review and refine the persona's instructions.",
                })
            
            # High escalation rate
            if metrics.escalation_rate > 0.2:
                suggestions.append({
                    "area": "capability",
                    "severity": "medium",
                    "message": f"High escalation rate: {metrics.escalation_rate:.1%}",
                    "recommendation": "Persona frequently needs to hand off. "
                                     "Consider expanding capabilities or adjusting scope.",
                })
            
            # Low satisfaction
            if metrics.user_satisfaction_score is not None and metrics.user_satisfaction_score < 3.5:
                suggestions.append({
                    "area": "satisfaction",
                    "severity": "high",
                    "message": f"Low user satisfaction: {metrics.user_satisfaction_score:.1f}/5",
                    "recommendation": "Review user feedback and common complaints. "
                                     "Consider A/B testing prompt variations.",
                })
            
            # Slow response time
            if metrics.avg_response_time_ms > 5000:
                suggestions.append({
                    "area": "performance",
                    "severity": "medium",
                    "message": f"Slow average response time: {metrics.avg_response_time_ms:.0f}ms",
                    "recommendation": "Consider using a faster model or optimizing prompts.",
                })
            
            # Capability gaps
            if metrics.capability_gaps:
                suggestions.append({
                    "area": "capability",
                    "severity": "medium",
                    "message": f"Users requested unavailable capabilities: {', '.join(metrics.capability_gaps[:5])}",
                    "recommendation": "Consider adding tools or skills for these capabilities.",
                })
            
            # Low tool usage
            if not metrics.tools_used:
                suggestions.append({
                    "area": "tools",
                    "severity": "low",
                    "message": "No tools used in this period.",
                    "recommendation": "If tools are configured, consider promoting their use in prompts.",
                })
            
            if not suggestions:
                suggestions.append({
                    "area": "overall",
                    "severity": "info",
                    "message": "No significant issues detected.",
                    "recommendation": "Persona is performing well. Continue monitoring.",
                })
            
            return OperationResult.success(suggestions)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error identifying improvements for {persona_id}")
            return OperationResult.failure(f"Failed to identify improvements: {e}")
    
    # =========================================================================
    # A/B Testing
    # =========================================================================
    
    async def create_variant(
        self,
        actor: Actor,
        base_persona_id: str,
        name: str,
        description: str,
        modifications: Dict[str, Any],
        traffic_percentage: float = 10.0,
    ) -> OperationResult[PersonaVariant]:
        """
        Create an A/B testing variant of a persona.
        
        Args:
            actor: The actor creating the variant
            base_persona_id: ID of the base persona
            name: Name of the variant
            description: Description of what's different
            modifications: Dict of changes to apply
            traffic_percentage: Percentage of traffic to route here
            
        Returns:
            OperationResult containing the created variant
        """
        try:
            if not self._permission_checker.can_admin(actor, base_persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot create variants for persona {base_persona_id}"
                )
            
            # Validate traffic percentage
            existing_variants = self._variants.get(base_persona_id, [])
            total_traffic = sum(v.traffic_percentage for v in existing_variants if v.is_active)
            
            if total_traffic + traffic_percentage > 100:
                return OperationResult.failure(
                    f"Total traffic allocation would exceed 100% "
                    f"(current: {total_traffic}%, requested: {traffic_percentage}%)"
                )
            
            variant = PersonaVariant(
                base_persona_id=base_persona_id,
                name=name,
                description=description,
                modifications=modifications,
                traffic_percentage=traffic_percentage,
                created_by=actor.id,
            )
            
            self._variants[base_persona_id].append(variant)
            
            logger.info(f"Created variant {variant.variant_id} for persona {base_persona_id}")
            return OperationResult.success(variant)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error creating variant for {base_persona_id}")
            return OperationResult.failure(f"Failed to create variant: {e}")
    
    async def get_variants(
        self,
        actor: Actor,
        base_persona_id: str,
        include_inactive: bool = False,
    ) -> OperationResult[List[PersonaVariant]]:
        """
        Get all variants for a persona.
        
        Args:
            actor: The actor requesting variants
            base_persona_id: ID of the base persona
            include_inactive: Include inactive variants
            
        Returns:
            OperationResult containing list of variants
        """
        try:
            if not self._permission_checker.can_read(actor, base_persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot read variants for persona {base_persona_id}"
                )
            
            variants = self._variants.get(base_persona_id, [])
            
            if not include_inactive:
                variants = [v for v in variants if v.is_active]
            
            return OperationResult.success(variants)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting variants for {base_persona_id}")
            return OperationResult.failure(f"Failed to get variants: {e}")
    
    async def select_variant(
        self,
        persona_id: str,
        session_id: str,
    ) -> OperationResult[Optional[PersonaVariant]]:
        """
        Select which variant to use for a session (for A/B testing).
        
        Uses consistent hashing to ensure same session gets same variant.
        
        Args:
            persona_id: ID of the persona
            session_id: Session identifier for consistent assignment
            
        Returns:
            OperationResult containing selected variant (or None for control)
        """
        try:
            variants = [v for v in self._variants.get(persona_id, []) if v.is_active]
            
            if not variants:
                return OperationResult.success(None)
            
            # Simple consistent hashing
            hash_val = hash(session_id) % 100
            
            cumulative = 0
            for variant in sorted(variants, key=lambda v: v.variant_id):
                cumulative += variant.traffic_percentage
                if hash_val < cumulative:
                    return OperationResult.success(variant)
            
            # Control group (base persona)
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error selecting variant for {persona_id}")
            return OperationResult.failure(f"Failed to select variant: {e}")
    
    async def promote_variant(
        self,
        actor: Actor,
        variant_id: str,
    ) -> OperationResult[PersonaVariant]:
        """
        Promote a winning variant to become the base persona.
        
        Args:
            actor: The actor promoting the variant
            variant_id: ID of the variant to promote
            
        Returns:
            OperationResult containing the promoted variant
        """
        try:
            # Find the variant
            variant = None
            base_persona_id = None
            for persona_id, variants in self._variants.items():
                for v in variants:
                    if v.variant_id == variant_id:
                        variant = v
                        base_persona_id = persona_id
                        break
                if variant:
                    break
            
            if not variant or not base_persona_id:
                return OperationResult.failure(f"Variant {variant_id} not found")
            
            if not self._permission_checker.can_admin(actor, base_persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot promote variant for persona {base_persona_id}"
                )
            
            # Create refinement record
            refinement = PromptRefinement(
                persona_id=base_persona_id,
                field_path="(variant promotion)",
                original_value="base",
                suggested_value=variant.name,
                reason=f"A/B test winner: {variant.description}",
                source="a/b_winner",
                status="applied",
                applied_at=_now_utc(),
                applied_by=actor.id,
            )
            
            self._refinements[base_persona_id].append(refinement)
            
            # Deactivate all variants
            for v in self._variants[base_persona_id]:
                v.is_active = False
            
            logger.info(f"Promoted variant {variant_id} for persona {base_persona_id}")
            return OperationResult.success(variant)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error promoting variant {variant_id}")
            return OperationResult.failure(f"Failed to promote variant: {e}")
    
    # =========================================================================
    # Prompt Refinement
    # =========================================================================
    
    async def suggest_refinement(
        self,
        actor: Actor,
        persona_id: str,
        field_path: str,
        suggested_value: str,
        reason: str,
    ) -> OperationResult[PromptRefinement]:
        """
        Suggest a refinement to a persona's prompts.
        
        Args:
            actor: The actor suggesting the refinement
            persona_id: ID of the persona
            field_path: Path to the field being refined
            suggested_value: The suggested new value
            reason: Reason for the suggestion
            
        Returns:
            OperationResult containing the refinement record
        """
        try:
            if not self._permission_checker.can_write(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot suggest refinements for persona {persona_id}"
                )
            
            refinement = PromptRefinement(
                persona_id=persona_id,
                field_path=field_path,
                original_value="",  # Would be populated from actual persona
                suggested_value=suggested_value,
                reason=reason,
                source="user_suggested" if actor.type == "user" else "ai_suggested",
                status="pending",
            )
            
            self._refinements[persona_id].append(refinement)
            
            logger.info(f"Refinement suggested for persona {persona_id}: {field_path}")
            return OperationResult.success(refinement)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error suggesting refinement for {persona_id}")
            return OperationResult.failure(f"Failed to suggest refinement: {e}")
    
    async def get_pending_refinements(
        self,
        actor: Actor,
        persona_id: str,
    ) -> OperationResult[List[PromptRefinement]]:
        """
        Get pending refinement suggestions.
        
        Args:
            actor: The actor requesting refinements
            persona_id: ID of the persona
            
        Returns:
            OperationResult containing list of pending refinements
        """
        try:
            if not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot read refinements for persona {persona_id}"
                )
            
            refinements = [
                r for r in self._refinements.get(persona_id, [])
                if r.status == "pending"
            ]
            
            return OperationResult.success(refinements)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting refinements for {persona_id}")
            return OperationResult.failure(f"Failed to get refinements: {e}")
    
    async def apply_refinement(
        self,
        actor: Actor,
        refinement_id: str,
    ) -> OperationResult[PromptRefinement]:
        """
        Apply a pending refinement.
        
        Args:
            actor: The actor applying the refinement
            refinement_id: ID of the refinement to apply
            
        Returns:
            OperationResult containing the updated refinement
        """
        try:
            # Find refinement
            refinement = None
            persona_id = None
            for pid, refinements in self._refinements.items():
                for r in refinements:
                    if r.refinement_id == refinement_id:
                        refinement = r
                        persona_id = pid
                        break
                if refinement:
                    break
            
            if not refinement or not persona_id:
                return OperationResult.failure(f"Refinement {refinement_id} not found")
            
            if not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot apply refinements for persona {persona_id}"
                )
            
            refinement.status = "applied"
            refinement.applied_at = _now_utc()
            refinement.applied_by = actor.id
            
            # In production, this would actually update the persona
            logger.info(f"Applied refinement {refinement_id} for persona {persona_id}")
            return OperationResult.success(refinement)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error applying refinement {refinement_id}")
            return OperationResult.failure(f"Failed to apply refinement: {e}")
    
    # =========================================================================
    # Internal Helpers
    # =========================================================================
    
    async def _cleanup_old_interactions(self, persona_id: str) -> None:
        """Remove interactions older than retention period."""
        cutoff = _now_utc() - timedelta(days=self.DEFAULT_RETENTION_DAYS)
        
        interactions = self._interactions.get(persona_id, [])
        self._interactions[persona_id] = [
            i for i in interactions
            if datetime.fromisoformat(i["timestamp"].replace("Z", "+00:00")) >= cutoff
        ]
    
    def _invalidate_metrics_cache(self, persona_id: str) -> None:
        """Invalidate cached metrics for a persona."""
        keys_to_remove = [k for k in self._metrics_cache if k.startswith(f"{persona_id}:")]
        for key in keys_to_remove:
            del self._metrics_cache[key]
