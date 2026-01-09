"""
Budget Policy Service implementation.

Provides CRUD operations for budget policies and pre-flight budget checks,
following the ATLAS service pattern.

Author: ATLAS Team
Date: Jan 8, 2026
"""

from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Dict, List, Optional, Protocol, TYPE_CHECKING

from core.services.common import (
    Actor,
    OperationResult,
    PermissionDeniedError,
    Service,
)

from modules.budget.models import (
    BudgetPolicy,
    BudgetScope,
    BudgetPeriod,
    LimitAction,
    BudgetCheckResult,
)

from .exceptions import (
    BudgetPolicyNotFoundError,
    BudgetPolicyConflictError,
    BudgetExceededError,
    BudgetValidationError,
)
from .permissions import BudgetPermissionChecker
from .types import (
    BudgetPolicyCreate,
    BudgetPolicyUpdate,
    BudgetCheckRequest,
    BudgetCheckResponse,
    BudgetPolicyCreated,
    BudgetPolicyUpdated,
    BudgetPolicyDeleted,
    BudgetCheckRequested,
)


if TYPE_CHECKING:
    from core.services.common import DomainEventPublisher


logger = logging.getLogger(__name__)


class BudgetRepository(Protocol):
    """Protocol for budget policy persistence."""
    
    async def get_policy(self, policy_id: str) -> Optional[BudgetPolicy]:
        """Get policy by ID."""
        ...
    
    async def save_policy(self, policy: BudgetPolicy) -> BudgetPolicy:
        """Save (create or update) a policy."""
        ...
    
    async def delete_policy(self, policy_id: str) -> bool:
        """Delete a policy by ID."""
        ...
    
    async def list_policies(
        self,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> List[BudgetPolicy]:
        """List policies with filtering."""
        ...
    
    async def get_current_spend(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        period: BudgetPeriod,
    ) -> Decimal:
        """Get current spending for a scope/period."""
        ...


class BudgetPolicyService(Service):
    """
    Service for managing budget policies and pre-flight checks.
    
    Provides:
    - CRUD operations for budget policies
    - Pre-flight budget checks before LLM requests
    - Configurable blocking/warning behavior per policy
    - Strict tenant isolation
    
    Example:
        service = BudgetPolicyService(
            repository=budget_repo,
            permission_checker=BudgetPermissionChecker(),
        )
        
        # Create a policy
        result = await service.create_policy(
            actor=user_actor,
            policy_data=BudgetPolicyCreate(
                name="Monthly Team Budget",
                scope=BudgetScope.TEAM,
                limit_amount=Decimal("500.00"),
            ),
        )
        
        # Check budget before LLM call
        check_result = await service.check_budget(
            actor=user_actor,
            request=BudgetCheckRequest(
                provider="openai",
                model="gpt-4o",
                estimated_input_tokens=1000,
                estimated_output_tokens=500,
            ),
        )
    """
    
    def __init__(
        self,
        *,
        repository: BudgetRepository,
        permission_checker: Optional[BudgetPermissionChecker] = None,
        event_publisher: Optional["DomainEventPublisher"] = None,
        pricing_registry: Optional[Any] = None,
    ) -> None:
        """Initialize the policy service.
        
        Args:
            repository: Budget policy persistence layer
            permission_checker: Permission checker (defaults to BudgetPermissionChecker)
            event_publisher: Optional domain event publisher
            pricing_registry: Optional pricing registry for cost estimation
        """
        self._repository = repository
        self._permissions = permission_checker or BudgetPermissionChecker()
        self._publisher = event_publisher
        self._pricing = pricing_registry
        self._enabled = True
    
    async def initialize(self) -> None:
        """Initialize the service."""
        logger.info("BudgetPolicyService initialized")
    
    async def health_check(self) -> OperationResult[Dict[str, Any]]:
        """Check service health."""
        return OperationResult.success({
            "status": "healthy",
            "enabled": self._enabled,
        })
    
    async def cleanup(self) -> None:
        """Clean up service resources."""
        logger.info("BudgetPolicyService cleanup complete")
    
    # =========================================================================
    # Policy CRUD Operations
    # =========================================================================
    
    async def create_policy(
        self,
        actor: Actor,
        policy_data: BudgetPolicyCreate,
    ) -> OperationResult[BudgetPolicy]:
        """Create a new budget policy.
        
        Args:
            actor: Actor performing the operation
            policy_data: Policy creation data
            
        Returns:
            OperationResult with created policy or error
        """
        try:
            # Check permissions for the target scope
            await self._permissions.require_scope_access(
                actor,
                policy_data.scope,
                policy_data.scope_id,
                write=True,
            )
        except PermissionDeniedError as exc:
            return OperationResult.failure(str(exc), exc.error_code)
        
        # Validate policy data
        validation_error = self._validate_policy_create(policy_data)
        if validation_error:
            return OperationResult.failure(validation_error, "VALIDATION_ERROR")
        
        # Check for conflicting policies
        conflict = await self._check_policy_conflict(policy_data)
        if conflict:
            return OperationResult.failure(
                f"Conflicting policy exists: {conflict.name}",
                "POLICY_CONFLICT",
            )
        
        # Validate against global budget ceiling
        ceiling_error = await self._validate_global_ceiling(policy_data)
        if ceiling_error:
            return OperationResult.failure(ceiling_error, "GLOBAL_CEILING_EXCEEDED")
        
        # Create policy from DTO
        policy = policy_data.to_policy()
        
        try:
            saved_policy = await self._repository.save_policy(policy)
        except Exception as exc:
            logger.error("Failed to save policy: %s", exc)
            return OperationResult.failure(str(exc), "SAVE_FAILED")
        
        # Publish event
        await self._publish_policy_created(saved_policy, actor)
        
        logger.info(
            "Created budget policy: %s (%s) by actor %s",
            saved_policy.name, saved_policy.id, actor.id
        )
        
        return OperationResult.success(saved_policy)
    
    async def get_policy(
        self,
        actor: Actor,
        policy_id: str,
    ) -> OperationResult[BudgetPolicy]:
        """Get a budget policy by ID.
        
        Args:
            actor: Actor performing the operation
            policy_id: Policy identifier
            
        Returns:
            OperationResult with policy or error
        """
        policy = await self._repository.get_policy(policy_id)
        if not policy:
            return OperationResult.failure(
                f"Policy not found: {policy_id}",
                "NOT_FOUND",
            )
        
        try:
            await self._permissions.require_policy_read(actor, policy)
        except PermissionDeniedError as exc:
            return OperationResult.failure(str(exc), exc.error_code)
        
        return OperationResult.success(policy)
    
    async def update_policy(
        self,
        actor: Actor,
        policy_id: str,
        update_data: BudgetPolicyUpdate,
    ) -> OperationResult[BudgetPolicy]:
        """Update an existing budget policy.
        
        Args:
            actor: Actor performing the operation
            policy_id: Policy identifier
            update_data: Fields to update
            
        Returns:
            OperationResult with updated policy or error
        """
        # Get existing policy
        policy = await self._repository.get_policy(policy_id)
        if not policy:
            return OperationResult.failure(
                f"Policy not found: {policy_id}",
                "NOT_FOUND",
            )
        
        try:
            await self._permissions.require_policy_write(actor, policy)
        except PermissionDeniedError as exc:
            return OperationResult.failure(str(exc), exc.error_code)
        
        # Validate update data
        validation_error = self._validate_policy_update(update_data)
        if validation_error:
            return OperationResult.failure(validation_error, "VALIDATION_ERROR")
        
        # If limit_amount is being updated, validate against global ceiling
        if update_data.limit_amount is not None:
            ceiling_check_data = BudgetPolicyCreate(
                name=policy.name,
                scope=policy.scope,
                scope_id=policy.scope_id,
                limit_amount=update_data.limit_amount,
                period=policy.period,
            )
            ceiling_error = await self._validate_global_ceiling(
                ceiling_check_data,
                excluding_policy_id=policy_id,
            )
            if ceiling_error:
                return OperationResult.failure(ceiling_error, "GLOBAL_CEILING_EXCEEDED")
        
        # Apply updates
        changed_fields = update_data.get_changed_fields()
        updated_policy = self._apply_policy_update(policy, update_data)
        
        try:
            saved_policy = await self._repository.save_policy(updated_policy)
        except Exception as exc:
            logger.error("Failed to update policy: %s", exc)
            return OperationResult.failure(str(exc), "SAVE_FAILED")
        
        # Publish event
        await self._publish_policy_updated(saved_policy, changed_fields, actor)
        
        logger.info(
            "Updated budget policy: %s (%s) by actor %s, fields: %s",
            saved_policy.name, saved_policy.id, actor.id, changed_fields
        )
        
        return OperationResult.success(saved_policy)
    
    async def delete_policy(
        self,
        actor: Actor,
        policy_id: str,
    ) -> OperationResult[bool]:
        """Delete a budget policy.
        
        Args:
            actor: Actor performing the operation
            policy_id: Policy identifier
            
        Returns:
            OperationResult with success boolean
        """
        # Get existing policy
        policy = await self._repository.get_policy(policy_id)
        if not policy:
            return OperationResult.failure(
                f"Policy not found: {policy_id}",
                "NOT_FOUND",
            )
        
        try:
            await self._permissions.require_policy_write(actor, policy)
        except PermissionDeniedError as exc:
            return OperationResult.failure(str(exc), exc.error_code)
        
        try:
            deleted = await self._repository.delete_policy(policy_id)
        except Exception as exc:
            logger.error("Failed to delete policy: %s", exc)
            return OperationResult.failure(str(exc), "DELETE_FAILED")
        
        if deleted:
            # Publish event
            await self._publish_policy_deleted(policy, actor)
            
            logger.info(
                "Deleted budget policy: %s (%s) by actor %s",
                policy.name, policy_id, actor.id
            )
        
        return OperationResult.success(deleted)
    
    async def list_policies(
        self,
        actor: Actor,
        *,
        scope: Optional[BudgetScope] = None,
        scope_id: Optional[str] = None,
        enabled_only: bool = True,
        limit: int = 100,
        offset: int = 0,
    ) -> OperationResult[List[BudgetPolicy]]:
        """List budget policies with optional filtering.
        
        Args:
            actor: Actor performing the operation
            scope: Filter by scope
            scope_id: Filter by scope ID
            enabled_only: Only return enabled policies
            limit: Maximum policies to return
            offset: Pagination offset
            
        Returns:
            OperationResult with list of policies
        """
        try:
            await self._permissions.require_read(actor)
        except PermissionDeniedError as exc:
            return OperationResult.failure(str(exc), exc.error_code)
        
        try:
            policies = await self._repository.list_policies(
                scope=scope,
                scope_id=scope_id,
                enabled_only=enabled_only,
                limit=limit,
                offset=offset,
            )
        except Exception as exc:
            logger.error("Failed to list policies: %s", exc)
            return OperationResult.failure(str(exc), "LIST_FAILED")
        
        # Filter by tenant access
        accessible_policies = self._permissions.filter_policies_for_actor(actor, policies)
        
        return OperationResult.success(accessible_policies)
    
    # =========================================================================
    # Budget Checks
    # =========================================================================
    
    async def check_budget(
        self,
        actor: Actor,
        request: BudgetCheckRequest,
    ) -> OperationResult[BudgetCheckResponse]:
        """Perform a pre-flight budget check.
        
        Checks all applicable policies and returns whether the
        estimated operation is allowed.
        
        Args:
            actor: Actor performing the operation
            request: Budget check request with cost estimates
            
        Returns:
            OperationResult with check response
        """
        if not self._enabled:
            return OperationResult.success(BudgetCheckResponse(
                allowed=True,
                action=LimitAction.WARN,
            ))
        
        try:
            await self._permissions.require_read(actor)
        except PermissionDeniedError as exc:
            return OperationResult.failure(str(exc), exc.error_code)
        
        # Calculate estimated cost if not provided
        estimated_cost = request.estimated_cost
        if estimated_cost is None and self._pricing:
            estimated_cost = self._pricing.estimate_request_cost(
                provider=request.provider,
                model=request.model,
                estimated_input_tokens=request.estimated_input_tokens,
                estimated_output_tokens=request.estimated_output_tokens,
                image_count=request.image_count,
            )
        estimated_cost = estimated_cost or Decimal("0")
        
        # Get applicable policies
        scopes_to_check = [
            (BudgetScope.GLOBAL, None),
            (BudgetScope.PROVIDER, request.provider),
            (BudgetScope.MODEL, request.model),
        ]
        
        if request.team_id:
            scopes_to_check.append((BudgetScope.TEAM, request.team_id))
        if request.user_id:
            scopes_to_check.append((BudgetScope.USER, request.user_id))
        if request.project_id:
            scopes_to_check.append((BudgetScope.PROJECT, request.project_id))
        if request.job_id:
            scopes_to_check.append((BudgetScope.JOB, request.job_id))
        if request.task_id:
            scopes_to_check.append((BudgetScope.TASK, request.task_id))
        if request.agent_id:
            scopes_to_check.append((BudgetScope.AGENT, request.agent_id))
        if request.session_id:
            scopes_to_check.append((BudgetScope.SESSION, request.session_id))
        
        warnings: List[str] = []
        most_restrictive_action = LimitAction.WARN
        blocking_policy: Optional[BudgetPolicy] = None
        current_spend = Decimal("0")
        limit_amount = Decimal("0")
        
        for scope, scope_id in scopes_to_check:
            try:
                policies_result = await self.list_policies(
                    actor,
                    scope=scope,
                    scope_id=scope_id,
                    enabled_only=True,
                )
                
                if not policies_result.success:
                    continue
                
                for policy in policies_result.data or []:
                    spend = await self._repository.get_current_spend(
                        scope=policy.scope,
                        scope_id=policy.scope_id,
                        period=policy.period,
                    )
                    
                    # Get rollover if enabled
                    effective_limit = policy.limit_amount
                    
                    remaining = effective_limit - spend
                    percent_used = float(spend / effective_limit) if effective_limit > 0 else 0.0
                    
                    # Check soft limit warning
                    if percent_used >= policy.soft_limit_percent:
                        warnings.append(
                            f"{policy.name}: {percent_used:.1%} of budget used "
                            f"(${spend:.2f} / ${effective_limit:.2f})"
                        )
                    
                    # Check if request would exceed limit
                    if estimated_cost > remaining:
                        if self._action_priority(policy.hard_limit_action) > self._action_priority(most_restrictive_action):
                            most_restrictive_action = policy.hard_limit_action
                            blocking_policy = policy
                            current_spend = spend
                            limit_amount = effective_limit
                    
            except Exception as exc:
                logger.warning("Error checking budget for scope %s: %s", scope, exc)
                continue
        
        # Determine if allowed based on action
        allowed = most_restrictive_action in (LimitAction.WARN, LimitAction.THROTTLE)
        
        # Find cheaper alternative if blocked
        alternative_model = None
        if not allowed and self._pricing:
            try:
                alt = self._pricing.get_cheaper_alternative(request.model, request.provider)
                if alt:
                    alternative_model = alt[0]
            except Exception:
                pass
        
        response = BudgetCheckResponse(
            allowed=allowed,
            action=most_restrictive_action,
            policy_id=blocking_policy.id if blocking_policy else None,
            current_spend=current_spend,
            limit_amount=limit_amount,
            estimated_cost=estimated_cost,
            remaining_after=(limit_amount - current_spend - estimated_cost) if limit_amount else Decimal("0"),
            warnings=warnings,
            alternative_model=alternative_model,
        )
        
        # Publish check event
        await self._publish_check_event(response, actor)
        
        return OperationResult.success(response)
    
    def _action_priority(self, action: LimitAction) -> int:
        """Get priority for limit action (higher = more restrictive)."""
        priorities = {
            LimitAction.WARN: 0,
            LimitAction.THROTTLE: 1,
            LimitAction.SOFT_BLOCK: 2,
            LimitAction.DEGRADE: 3,
            LimitAction.BLOCK: 4,
        }
        return priorities.get(action, 0)
    
    # =========================================================================
    # Validation
    # =========================================================================
    
    def _validate_policy_create(self, data: BudgetPolicyCreate) -> Optional[str]:
        """Validate policy creation data."""
        if not data.name or not data.name.strip():
            return "Policy name is required"
        
        if data.limit_amount <= 0:
            return "Limit amount must be positive"
        
        if not 0.0 <= data.soft_limit_percent <= 1.0:
            return "Soft limit percent must be between 0.0 and 1.0"
        
        if not 0.0 <= data.rollover_max_percent <= 1.0:
            return "Rollover max percent must be between 0.0 and 1.0"
        
        return None
    
    def _validate_policy_update(self, data: BudgetPolicyUpdate) -> Optional[str]:
        """Validate policy update data."""
        if data.name is not None and not data.name.strip():
            return "Policy name cannot be empty"
        
        if data.limit_amount is not None and data.limit_amount <= 0:
            return "Limit amount must be positive"
        
        if data.soft_limit_percent is not None:
            if not 0.0 <= data.soft_limit_percent <= 1.0:
                return "Soft limit percent must be between 0.0 and 1.0"
        
        if data.rollover_max_percent is not None:
            if not 0.0 <= data.rollover_max_percent <= 1.0:
                return "Rollover max percent must be between 0.0 and 1.0"
        
        return None
    
    async def _check_policy_conflict(
        self,
        data: BudgetPolicyCreate,
    ) -> Optional[BudgetPolicy]:
        """Check if a conflicting policy already exists."""
        try:
            # Check for policies with same scope/scope_id/period
            existing = await self._repository.list_policies(
                scope=data.scope,
                scope_id=data.scope_id,
                enabled_only=False,
            )
            
            for policy in existing:
                if policy.period == data.period:
                    return policy
                    
        except Exception as exc:
            logger.warning("Error checking policy conflicts: %s", exc)
        
        return None
    
    async def _validate_global_ceiling(
        self,
        data: BudgetPolicyCreate,
        excluding_policy_id: Optional[str] = None,
    ) -> Optional[str]:
        """Validate that a policy doesn't exceed the global budget ceiling.
        
        Rules:
        1. No scoped policy can have a limit greater than the global policy limit
        2. The sum of all scoped policies at the same level cannot exceed global
        
        Args:
            data: Policy creation data to validate
            excluding_policy_id: Policy ID to exclude (for updates)
            
        Returns:
            Error message if validation fails, None otherwise
        """
        # Global policies don't need ceiling validation
        if data.scope == BudgetScope.GLOBAL:
            return None
        
        try:
            # Get global policies for the same period
            global_policies = await self._repository.list_policies(
                scope=BudgetScope.GLOBAL,
                scope_id=None,
                enabled_only=True,
            )
            
            # Find matching period global policy
            global_policy = None
            for policy in global_policies:
                if policy.period == data.period:
                    global_policy = policy
                    break
            
            # If no global policy exists, no ceiling constraint
            if global_policy is None:
                return None
            
            global_limit = global_policy.limit_amount
            
            # Rule 1: Individual policy cannot exceed global
            if data.limit_amount > global_limit:
                return (
                    f"Policy limit ({data.limit_amount}) exceeds global budget "
                    f"ceiling ({global_limit}) for {data.period.value} period"
                )
            
            # Rule 2: Combined scoped policies cannot exceed global
            # Get all policies at the same scope level
            existing_policies = await self._repository.list_policies(
                scope=data.scope,
                enabled_only=True,
            )
            
            # Calculate total limit for existing policies (excluding the one being updated)
            existing_total = Decimal("0")
            for policy in existing_policies:
                if policy.period == data.period:
                    if excluding_policy_id and policy.id == excluding_policy_id:
                        continue
                    existing_total += policy.limit_amount
            
            # Check if adding this policy exceeds global
            combined_total = existing_total + data.limit_amount
            if combined_total > global_limit:
                return (
                    f"Combined {data.scope.value} budgets ({combined_total}) "
                    f"would exceed global ceiling ({global_limit}). "
                    f"Existing {data.scope.value} budgets: {existing_total}, "
                    f"new policy: {data.limit_amount}"
                )
            
        except Exception as exc:
            logger.warning("Error validating global ceiling: %s", exc)
            # Don't fail creation on validation errors, just log
        
        return None
    
    def _apply_policy_update(
        self,
        policy: BudgetPolicy,
        update: BudgetPolicyUpdate,
    ) -> BudgetPolicy:
        """Apply update fields to a policy."""
        updates = {}
        
        if update.name is not None:
            updates["name"] = update.name
        if update.limit_amount is not None:
            updates["limit_amount"] = update.limit_amount
        if update.soft_limit_percent is not None:
            updates["soft_limit_percent"] = update.soft_limit_percent
        if update.hard_limit_action is not None:
            updates["hard_limit_action"] = update.hard_limit_action
        if update.rollover_enabled is not None:
            updates["rollover_enabled"] = update.rollover_enabled
        if update.rollover_max_percent is not None:
            updates["rollover_max_percent"] = update.rollover_max_percent
        if update.provider_limits is not None:
            updates["provider_limits"] = update.provider_limits
        if update.model_limits is not None:
            updates["model_limits"] = update.model_limits
        if update.enabled is not None:
            updates["enabled"] = update.enabled
        if update.priority is not None:
            updates["priority"] = update.priority
        if update.metadata is not None:
            updates["metadata"] = update.metadata
        
        updates["updated_at"] = datetime.now(timezone.utc)
        
        return replace(policy, **updates)
    
    # =========================================================================
    # Event Publishing
    # =========================================================================
    
    async def _publish_policy_created(
        self,
        policy: BudgetPolicy,
        actor: Actor,
    ) -> None:
        """Publish policy created event."""
        if not self._publisher:
            return
        
        try:
            event = BudgetPolicyCreated(
                policy_id=policy.id,
                policy_name=policy.name,
                scope=policy.scope.value,
                limit_amount=policy.limit_amount,
                period=policy.period.value,
                tenant_id=actor.tenant_id,
                actor_id=actor.id,
                scope_id=policy.scope_id,
                actor_type=actor.type,
            )
            await self._publisher.publish(event)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Failed to publish policy created event: %s", exc)
    
    async def _publish_policy_updated(
        self,
        policy: BudgetPolicy,
        changed_fields: List[str],
        actor: Actor,
    ) -> None:
        """Publish policy updated event."""
        if not self._publisher:
            return
        
        try:
            event = BudgetPolicyUpdated(
                policy_id=policy.id,
                policy_name=policy.name,
                changed_fields=tuple(changed_fields),
                tenant_id=actor.tenant_id,
                actor_id=actor.id,
                actor_type=actor.type,
            )
            await self._publisher.publish(event)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Failed to publish policy updated event: %s", exc)
    
    async def _publish_policy_deleted(
        self,
        policy: BudgetPolicy,
        actor: Actor,
    ) -> None:
        """Publish policy deleted event."""
        if not self._publisher:
            return
        
        try:
            event = BudgetPolicyDeleted(
                policy_id=policy.id,
                policy_name=policy.name,
                tenant_id=actor.tenant_id,
                actor_id=actor.id,
                actor_type=actor.type,
            )
            await self._publisher.publish(event)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Failed to publish policy deleted event: %s", exc)
    
    async def _publish_check_event(
        self,
        response: BudgetCheckResponse,
        actor: Actor,
    ) -> None:
        """Publish budget check event."""
        if not self._publisher:
            return
        
        # Only publish if there are warnings or the request was blocked
        if not response.warnings and response.allowed:
            return
        
        try:
            event = BudgetCheckRequested(
                tenant_id=actor.tenant_id,
                actor_id=actor.id,
                estimated_cost=response.estimated_cost,
                allowed=response.allowed,
                action=response.action.value,
                policy_id=response.policy_id,
                warnings=tuple(response.warnings),
                actor_type=actor.type,
            )
            await self._publisher.publish(event)  # type: ignore[arg-type]
        except Exception as exc:
            logger.warning("Failed to publish budget check event: %s", exc)
