"""
Persona safety service for ATLAS.

Provides guardrails, content filtering, and HITL approval
workflows scoped to individual personas.

Author: ATLAS Team
Date: Jan 10, 2026
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from core.services.common import Actor, OperationResult
from core.services.common.exceptions import PermissionDeniedError

from .exceptions import PersonaError, PersonaNotFoundError
from .permissions import PersonaPermissionChecker
from .types import (
    # Safety types
    PersonaSafetyPolicy,
    ContentFilterRule,
    PIIPolicy,
    PIIHandlingMode,
    EscalationPath,
    HITLTrigger,
    ActionCheckResult,
    ApprovalRequest,
    ApprovalStatus,
    RiskLevel,
    # Events
    PersonaSafetyViolation,
)

if TYPE_CHECKING:
    from core.config import ConfigManager
    from core.messaging import MessageBus


logger = logging.getLogger(__name__)


def _now_utc() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)


# Common PII patterns
PII_PATTERNS = {
    "email": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "phone": r"\b(\+?1?[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
}


class PersonaSafetyError(PersonaError):
    """Error related to persona safety operations."""
    pass


class ActionBlockedError(PersonaSafetyError):
    """Raised when an action is blocked by safety policy."""
    
    def __init__(self, message: str, action: str, reason: str):
        super().__init__(message)
        self.action = action
        self.reason = reason


class ApprovalRequiredError(PersonaSafetyError):
    """Raised when an action requires human approval."""
    
    def __init__(self, message: str, approval_request: ApprovalRequest):
        super().__init__(message)
        self.approval_request = approval_request


class PersonaSafetyService:
    """
    Service for managing persona-specific safety policies and guardrails.
    
    Provides:
    - Per-persona safety policy management
    - Action pre-flight checks
    - Content filtering (input/output)
    - PII detection and handling
    - Human-in-the-loop approval workflows
    - Risk scoring and escalation
    """
    
    # Default safety policy for personas without explicit configuration
    DEFAULT_POLICY = PersonaSafetyPolicy(
        persona_id="default",
        max_actions_per_minute=60,
        max_tokens_per_conversation=100000,
        max_tool_calls_per_turn=10,
        risk_threshold=0.7,
        auto_escalation_threshold=0.9,
    )
    
    def __init__(
        self,
        config_manager: Optional["ConfigManager"] = None,
        message_bus: Optional["MessageBus"] = None,
        permission_checker: Optional[PersonaPermissionChecker] = None,
    ) -> None:
        """
        Initialize the PersonaSafetyService.
        
        Args:
            config_manager: Configuration manager
            message_bus: Message bus for publishing events
            permission_checker: Permission checker for authorization
        """
        self._config_manager = config_manager
        self._message_bus = message_bus
        self._permission_checker = permission_checker or PersonaPermissionChecker()
        
        # In-memory storage (would be persisted in production)
        self._policies: Dict[str, PersonaSafetyPolicy] = {}
        self._pending_approvals: Dict[str, ApprovalRequest] = {}
        self._action_counts: Dict[str, List[datetime]] = {}  # persona_id -> timestamps
        
        # Compiled PII patterns
        self._pii_patterns: Dict[str, re.Pattern] = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in PII_PATTERNS.items()
        }
    
    # =========================================================================
    # Policy Management
    # =========================================================================
    
    async def get_safety_policy(
        self,
        actor: Actor,
        persona_id: str,
    ) -> OperationResult[PersonaSafetyPolicy]:
        """
        Get the safety policy for a persona.
        
        Args:
            actor: The actor requesting the policy
            persona_id: ID of the persona
            
        Returns:
            OperationResult containing the safety policy
        """
        try:
            # Check permission
            if not self._permission_checker.can_read(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot read safety policy for persona {persona_id}"
                )
            
            policy = self._policies.get(persona_id)
            if policy is None:
                # Return default policy with persona_id set
                policy = PersonaSafetyPolicy(
                    persona_id=persona_id,
                    max_actions_per_minute=self.DEFAULT_POLICY.max_actions_per_minute,
                    max_tokens_per_conversation=self.DEFAULT_POLICY.max_tokens_per_conversation,
                    max_tool_calls_per_turn=self.DEFAULT_POLICY.max_tool_calls_per_turn,
                    risk_threshold=self.DEFAULT_POLICY.risk_threshold,
                    auto_escalation_threshold=self.DEFAULT_POLICY.auto_escalation_threshold,
                )
            
            return OperationResult.success(policy)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error getting safety policy for {persona_id}")
            return OperationResult.failure(f"Failed to get safety policy: {e}")
    
    async def update_safety_policy(
        self,
        actor: Actor,
        persona_id: str,
        policy: PersonaSafetyPolicy,
    ) -> OperationResult[PersonaSafetyPolicy]:
        """
        Update or create a safety policy for a persona.
        
        Args:
            actor: The actor making the update
            persona_id: ID of the persona
            policy: The new safety policy
            
        Returns:
            OperationResult containing the updated policy
        """
        try:
            # Require admin permission to modify safety policies
            if not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot modify safety policy for persona {persona_id}"
                )
            
            # Ensure persona_id matches
            if policy.persona_id != persona_id:
                policy = PersonaSafetyPolicy(
                    persona_id=persona_id,
                    allowed_actions=policy.allowed_actions,
                    blocked_actions=policy.blocked_actions,
                    require_approval_actions=policy.require_approval_actions,
                    input_filters=policy.input_filters,
                    output_filters=policy.output_filters,
                    pii_policy=policy.pii_policy,
                    max_actions_per_minute=policy.max_actions_per_minute,
                    max_tokens_per_conversation=policy.max_tokens_per_conversation,
                    max_tool_calls_per_turn=policy.max_tool_calls_per_turn,
                    risk_threshold=policy.risk_threshold,
                    auto_escalation_threshold=policy.auto_escalation_threshold,
                    hitl_triggers=policy.hitl_triggers,
                    escalation_paths=policy.escalation_paths,
                    log_all_actions=policy.log_all_actions,
                    log_retention_days=policy.log_retention_days,
                    updated_at=_now_utc(),
                )
            
            self._policies[persona_id] = policy
            
            logger.info(f"Updated safety policy for persona {persona_id}")
            return OperationResult.success(policy)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error updating safety policy for {persona_id}")
            return OperationResult.failure(f"Failed to update safety policy: {e}")
    
    async def delete_safety_policy(
        self,
        actor: Actor,
        persona_id: str,
    ) -> OperationResult[None]:
        """
        Delete a safety policy, reverting to defaults.
        
        Args:
            actor: The actor making the deletion
            persona_id: ID of the persona
            
        Returns:
            OperationResult indicating success/failure
        """
        try:
            if not self._permission_checker.can_admin(actor, persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot delete safety policy for persona {persona_id}"
                )
            
            if persona_id in self._policies:
                del self._policies[persona_id]
                logger.info(f"Deleted safety policy for persona {persona_id}")
            
            return OperationResult.success(None)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error deleting safety policy for {persona_id}")
            return OperationResult.failure(f"Failed to delete safety policy: {e}")
    
    # =========================================================================
    # Action Checking
    # =========================================================================
    
    async def check_action(
        self,
        actor: Actor,
        persona_id: str,
        action: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> OperationResult[ActionCheckResult]:
        """
        Pre-flight check for an action against persona safety policy.
        
        Args:
            actor: The actor performing the action
            persona_id: ID of the persona
            action: The action being performed (e.g., "tool:web_search", "skill:code_review")
            context: Additional context for the action
            
        Returns:
            OperationResult containing the check result
        """
        try:
            context = context or {}
            policy = self._policies.get(persona_id) or self.DEFAULT_POLICY
            
            matched_rules: List[str] = []
            warnings: List[str] = []
            
            # Calculate risk score
            risk_score = self._calculate_risk_score(action, context, policy)
            risk_level = self._get_risk_level(risk_score)
            
            # Check if action is explicitly blocked
            if self._is_action_blocked(action, policy):
                matched_rules.append("blocked_action")
                await self._emit_safety_violation(
                    persona_id, actor.id, action, "blocked", risk_score,
                    f"Action '{action}' is blocked by policy"
                )
                return OperationResult.success(ActionCheckResult(
                    allowed=False,
                    action=action,
                    persona_id=persona_id,
                    risk_level=risk_level,
                    risk_score=risk_score,
                    requires_approval=False,
                    blocked_reason=f"Action '{action}' is blocked by safety policy",
                    matched_rules=matched_rules,
                ))
            
            # Check rate limits
            if not self._check_rate_limit(persona_id, policy):
                warnings.append("Rate limit approaching")
                if self._is_rate_limit_exceeded(persona_id, policy):
                    matched_rules.append("rate_limit_exceeded")
                    return OperationResult.success(ActionCheckResult(
                        allowed=False,
                        action=action,
                        persona_id=persona_id,
                        risk_level=risk_level,
                        risk_score=risk_score,
                        blocked_reason="Rate limit exceeded",
                        matched_rules=matched_rules,
                    ))
            
            # Check if action requires approval
            requires_approval = self._requires_approval(action, risk_score, policy)
            approval_id = None
            
            if requires_approval:
                matched_rules.append("requires_approval")
                # Create approval request
                approval_request = ApprovalRequest(
                    persona_id=persona_id,
                    actor_id=actor.id,
                    action=action,
                    context=context,
                    risk_score=risk_score,
                    reason_required=f"Action '{action}' requires approval (risk: {risk_level})",
                    expires_at=_now_utc() + timedelta(seconds=600),
                )
                self._pending_approvals[approval_request.approval_id] = approval_request
                approval_id = approval_request.approval_id
            
            # Check whitelist (if specified, action must be in whitelist)
            if policy.allowed_actions and action not in policy.allowed_actions:
                # Check if any pattern matches
                if not any(re.match(pattern, action) for pattern in policy.allowed_actions):
                    matched_rules.append("not_in_whitelist")
                    warnings.append(f"Action '{action}' not in allowed actions list")
            
            # Record action for rate limiting
            self._record_action(persona_id)
            
            return OperationResult.success(ActionCheckResult(
                allowed=True,
                action=action,
                persona_id=persona_id,
                risk_level=risk_level,
                risk_score=risk_score,
                requires_approval=requires_approval,
                approval_id=approval_id,
                warnings=warnings,
                matched_rules=matched_rules,
            ))
            
        except Exception as e:
            logger.exception(f"Error checking action {action} for persona {persona_id}")
            return OperationResult.failure(f"Failed to check action: {e}")
    
    def _calculate_risk_score(
        self,
        action: str,
        context: Dict[str, Any],
        policy: PersonaSafetyPolicy,
    ) -> float:
        """Calculate risk score for an action (0.0 - 1.0)."""
        base_score = 0.1
        
        # High-risk action patterns
        high_risk_patterns = [
            r"tool:.*delete",
            r"tool:.*remove",
            r"tool:.*execute",
            r"tool:.*terminal",
            r"tool:.*system",
            r"skill:.*financial",
            r"skill:.*payment",
            r".*admin.*",
        ]
        
        medium_risk_patterns = [
            r"tool:.*write",
            r"tool:.*create",
            r"tool:.*modify",
            r"tool:.*update",
            r"tool:.*send",
            r"skill:.*external",
        ]
        
        for pattern in high_risk_patterns:
            if re.match(pattern, action, re.IGNORECASE):
                base_score = max(base_score, 0.8)
                break
        
        for pattern in medium_risk_patterns:
            if re.match(pattern, action, re.IGNORECASE):
                base_score = max(base_score, 0.5)
                break
        
        # Adjust based on context
        if context.get("affects_external", False):
            base_score = min(1.0, base_score + 0.2)
        if context.get("irreversible", False):
            base_score = min(1.0, base_score + 0.3)
        if context.get("involves_pii", False):
            base_score = min(1.0, base_score + 0.15)
        
        return base_score
    
    def _get_risk_level(self, score: float) -> str:
        """Convert risk score to risk level."""
        if score >= 0.9:
            return RiskLevel.CRITICAL
        elif score >= 0.7:
            return RiskLevel.HIGH
        elif score >= 0.4:
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    
    def _is_action_blocked(self, action: str, policy: PersonaSafetyPolicy) -> bool:
        """Check if action is in blocked list."""
        for blocked in policy.blocked_actions:
            if action == blocked or re.match(blocked, action):
                return True
        return False
    
    def _requires_approval(
        self,
        action: str,
        risk_score: float,
        policy: PersonaSafetyPolicy,
    ) -> bool:
        """Check if action requires human approval."""
        # Explicit approval list
        for approval_action in policy.require_approval_actions:
            if action == approval_action or re.match(approval_action, action):
                return True
        
        # Risk threshold
        if risk_score >= policy.risk_threshold:
            return True
        
        # HITL triggers
        for trigger in policy.hitl_triggers:
            if trigger.enabled and re.match(trigger.action_pattern, action):
                trigger_risk = {
                    RiskLevel.LOW: 0.0,
                    RiskLevel.MEDIUM: 0.4,
                    RiskLevel.HIGH: 0.7,
                    RiskLevel.CRITICAL: 0.9,
                }.get(trigger.risk_threshold, 0.7)
                if risk_score >= trigger_risk:
                    return True
        
        return False
    
    def _check_rate_limit(self, persona_id: str, policy: PersonaSafetyPolicy) -> bool:
        """Check if approaching rate limit (warning at 80%)."""
        timestamps = self._action_counts.get(persona_id, [])
        one_minute_ago = _now_utc() - timedelta(minutes=1)
        recent = [t for t in timestamps if t > one_minute_ago]
        return len(recent) < (policy.max_actions_per_minute * 0.8)
    
    def _is_rate_limit_exceeded(self, persona_id: str, policy: PersonaSafetyPolicy) -> bool:
        """Check if rate limit is exceeded."""
        timestamps = self._action_counts.get(persona_id, [])
        one_minute_ago = _now_utc() - timedelta(minutes=1)
        recent = [t for t in timestamps if t > one_minute_ago]
        return len(recent) >= policy.max_actions_per_minute
    
    def _record_action(self, persona_id: str) -> None:
        """Record an action timestamp for rate limiting."""
        if persona_id not in self._action_counts:
            self._action_counts[persona_id] = []
        self._action_counts[persona_id].append(_now_utc())
        
        # Cleanup old entries (keep last 5 minutes)
        five_minutes_ago = _now_utc() - timedelta(minutes=5)
        self._action_counts[persona_id] = [
            t for t in self._action_counts[persona_id]
            if t > five_minutes_ago
        ]
    
    # =========================================================================
    # Content Filtering
    # =========================================================================
    
    async def filter_content(
        self,
        persona_id: str,
        content: str,
        direction: str = "output",  # "input" or "output"
    ) -> OperationResult[Dict[str, Any]]:
        """
        Apply content filters to input or output.
        
        Args:
            persona_id: ID of the persona
            content: Content to filter
            direction: "input" or "output"
            
        Returns:
            OperationResult with filtered content and any violations
        """
        try:
            policy = self._policies.get(persona_id) or self.DEFAULT_POLICY
            filters = policy.input_filters if direction == "input" else policy.output_filters
            
            violations: List[Dict[str, Any]] = []
            filtered_content = content
            should_block = False
            
            # Apply content filter rules
            for rule in sorted(filters, key=lambda r: r.priority):
                if not rule.enabled:
                    continue
                    
                if rule.applies_to not in (direction, "both"):
                    continue
                
                try:
                    if re.search(rule.pattern, content, re.IGNORECASE):
                        violations.append({
                            "rule_id": rule.rule_id,
                            "rule_name": rule.name,
                            "action": rule.action,
                            "message": rule.message,
                        })
                        
                        if rule.action == "block":
                            should_block = True
                        elif rule.action == "mask":
                            filtered_content = re.sub(
                                rule.pattern,
                                "[FILTERED]",
                                filtered_content,
                                flags=re.IGNORECASE
                            )
                        elif rule.action == "warn":
                            logger.warning(
                                f"Content filter warning for persona {persona_id}: "
                                f"{rule.name} - {rule.message}"
                            )
                except re.error as e:
                    logger.warning(f"Invalid regex pattern in filter rule {rule.rule_id}: {e}")
            
            # Apply PII filtering
            pii_result = await self._filter_pii(content, policy.pii_policy)
            if pii_result["pii_detected"]:
                violations.extend(pii_result["violations"])
                if policy.pii_policy.mode == PIIHandlingMode.BLOCK:
                    should_block = True
                elif policy.pii_policy.mode == PIIHandlingMode.MASK:
                    filtered_content = pii_result["masked_content"]
            
            return OperationResult.success({
                "original_content": content,
                "filtered_content": filtered_content,
                "blocked": should_block,
                "violations": violations,
                "pii_detected": pii_result["pii_detected"],
            })
            
        except Exception as e:
            logger.exception(f"Error filtering content for persona {persona_id}")
            return OperationResult.failure(f"Failed to filter content: {e}")
    
    async def _filter_pii(
        self,
        content: str,
        pii_policy: PIIPolicy,
    ) -> Dict[str, Any]:
        """Detect and optionally mask PII in content."""
        pii_detected = False
        violations: List[Dict[str, Any]] = []
        masked_content = content
        
        pii_checks = [
            ("email", pii_policy.detect_emails, self._pii_patterns.get("email")),
            ("phone", pii_policy.detect_phones, self._pii_patterns.get("phone")),
            ("ssn", pii_policy.detect_ssn, self._pii_patterns.get("ssn")),
            ("credit_card", pii_policy.detect_credit_cards, self._pii_patterns.get("credit_card")),
        ]
        
        for pii_type, should_check, pattern in pii_checks:
            if should_check and pattern:
                matches = pattern.findall(content)
                if matches:
                    pii_detected = True
                    violations.append({
                        "type": "pii",
                        "pii_type": pii_type,
                        "count": len(matches),
                    })
                    if pii_policy.mode in (PIIHandlingMode.MASK, PIIHandlingMode.BLOCK):
                        masked_content = pattern.sub(f"[{pii_type.upper()}_REDACTED]", masked_content)
        
        # Custom patterns
        for custom_pattern in pii_policy.custom_patterns:
            try:
                regex = re.compile(custom_pattern, re.IGNORECASE)
                matches = regex.findall(content)
                if matches:
                    pii_detected = True
                    violations.append({
                        "type": "pii",
                        "pii_type": "custom",
                        "pattern": custom_pattern,
                        "count": len(matches),
                    })
                    if pii_policy.mode in (PIIHandlingMode.MASK, PIIHandlingMode.BLOCK):
                        masked_content = regex.sub("[CUSTOM_REDACTED]", masked_content)
            except re.error:
                logger.warning(f"Invalid custom PII pattern: {custom_pattern}")
        
        return {
            "pii_detected": pii_detected,
            "violations": violations,
            "masked_content": masked_content,
        }
    
    # =========================================================================
    # Approval Workflow
    # =========================================================================
    
    async def get_pending_approvals(
        self,
        actor: Actor,
        persona_id: Optional[str] = None,
    ) -> OperationResult[List[ApprovalRequest]]:
        """
        Get pending approval requests.
        
        Args:
            actor: The actor requesting the list
            persona_id: Optional filter by persona
            
        Returns:
            OperationResult containing list of pending approvals
        """
        try:
            now = _now_utc()
            
            # Clean up expired approvals
            expired = [
                aid for aid, req in self._pending_approvals.items()
                if req.expires_at and req.expires_at < now
            ]
            for aid in expired:
                self._pending_approvals[aid].status = ApprovalStatus.EXPIRED
            
            # Filter approvals
            approvals = [
                req for req in self._pending_approvals.values()
                if req.status == ApprovalStatus.PENDING
                and (persona_id is None or req.persona_id == persona_id)
            ]
            
            return OperationResult.success(approvals)
            
        except Exception as e:
            logger.exception("Error getting pending approvals")
            return OperationResult.failure(f"Failed to get pending approvals: {e}")
    
    async def approve_action(
        self,
        actor: Actor,
        approval_id: str,
        reason: Optional[str] = None,
    ) -> OperationResult[ApprovalRequest]:
        """
        Approve a pending action request.
        
        Args:
            actor: The actor approving the request
            approval_id: ID of the approval request
            reason: Optional reason for approval
            
        Returns:
            OperationResult containing the updated request
        """
        try:
            if approval_id not in self._pending_approvals:
                return OperationResult.failure(f"Approval request {approval_id} not found")
            
            request = self._pending_approvals[approval_id]
            
            # Check permission to approve
            if not self._permission_checker.can_admin(actor, request.persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot approve actions for persona {request.persona_id}"
                )
            
            if request.status != ApprovalStatus.PENDING:
                return OperationResult.failure(f"Request is already {request.status}")
            
            if request.expires_at and request.expires_at < _now_utc():
                request.status = ApprovalStatus.EXPIRED
                return OperationResult.failure("Approval request has expired")
            
            request.status = ApprovalStatus.APPROVED
            request.approver_id = actor.id
            request.approver_reason = reason
            request.resolved_at = _now_utc()
            
            logger.info(f"Action approved: {request.action} for persona {request.persona_id}")
            return OperationResult.success(request)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error approving action {approval_id}")
            return OperationResult.failure(f"Failed to approve action: {e}")
    
    async def reject_action(
        self,
        actor: Actor,
        approval_id: str,
        reason: str,
    ) -> OperationResult[ApprovalRequest]:
        """
        Reject a pending action request.
        
        Args:
            actor: The actor rejecting the request
            approval_id: ID of the approval request
            reason: Reason for rejection
            
        Returns:
            OperationResult containing the updated request
        """
        try:
            if approval_id not in self._pending_approvals:
                return OperationResult.failure(f"Approval request {approval_id} not found")
            
            request = self._pending_approvals[approval_id]
            
            if not self._permission_checker.can_admin(actor, request.persona_id):
                raise PermissionDeniedError(
                    f"Actor {actor.id} cannot reject actions for persona {request.persona_id}"
                )
            
            if request.status != ApprovalStatus.PENDING:
                return OperationResult.failure(f"Request is already {request.status}")
            
            request.status = ApprovalStatus.REJECTED
            request.approver_id = actor.id
            request.approver_reason = reason
            request.resolved_at = _now_utc()
            
            logger.info(f"Action rejected: {request.action} for persona {request.persona_id}")
            return OperationResult.success(request)
            
        except PermissionDeniedError:
            raise
        except Exception as e:
            logger.exception(f"Error rejecting action {approval_id}")
            return OperationResult.failure(f"Failed to reject action: {e}")
    
    async def check_approval_status(
        self,
        approval_id: str,
    ) -> OperationResult[ApprovalRequest]:
        """
        Check the status of an approval request.
        
        Args:
            approval_id: ID of the approval request
            
        Returns:
            OperationResult containing the request
        """
        try:
            if approval_id not in self._pending_approvals:
                return OperationResult.failure(f"Approval request {approval_id} not found")
            
            request = self._pending_approvals[approval_id]
            
            # Check if expired
            if (
                request.status == ApprovalStatus.PENDING
                and request.expires_at
                and request.expires_at < _now_utc()
            ):
                request.status = ApprovalStatus.EXPIRED
            
            return OperationResult.success(request)
            
        except Exception as e:
            logger.exception(f"Error checking approval status {approval_id}")
            return OperationResult.failure(f"Failed to check approval status: {e}")
    
    # =========================================================================
    # Escalation
    # =========================================================================
    
    async def get_escalation_path(
        self,
        persona_id: str,
        trigger: str,
    ) -> OperationResult[Optional[EscalationPath]]:
        """
        Get the escalation path for a given trigger.
        
        Args:
            persona_id: ID of the persona
            trigger: The trigger condition
            
        Returns:
            OperationResult containing the matching escalation path if any
        """
        try:
            policy = self._policies.get(persona_id)
            if not policy:
                return OperationResult.success(None)
            
            for path in sorted(policy.escalation_paths, key=lambda p: p.priority):
                if not path.enabled:
                    continue
                if path.from_persona and path.from_persona != persona_id:
                    continue
                
                for path_trigger in path.triggers:
                    if re.match(path_trigger, trigger, re.IGNORECASE):
                        return OperationResult.success(path)
            
            return OperationResult.success(None)
            
        except Exception as e:
            logger.exception(f"Error getting escalation path for {persona_id}")
            return OperationResult.failure(f"Failed to get escalation path: {e}")
    
    # =========================================================================
    # Event Emission
    # =========================================================================
    
    async def _emit_safety_violation(
        self,
        persona_id: str,
        actor_id: str,
        action: str,
        violation_type: str,
        risk_score: float,
        details: str,
    ) -> None:
        """Emit a safety violation event."""
        if self._message_bus:
            try:
                event = PersonaSafetyViolation(
                    persona_id=persona_id,
                    actor_id=actor_id,
                    action=action,
                    violation_type=violation_type,
                    risk_score=risk_score,
                    details=details,
                )
                await self._message_bus.publish(event.event_type, event.to_dict())
            except Exception as e:
                logger.warning(f"Failed to emit safety violation event: {e}")
