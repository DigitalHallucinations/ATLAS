"""Budget policy matching and resolution.

Provides utilities for matching usage contexts against budget policies
using hierarchical scope resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import List, Optional, Tuple

from modules.budget.models import BudgetPolicy, BudgetScope, LimitAction
from modules.budget.scope_hierarchy import (
    UsageContext,
    ScopeHierarchyResolver,
    get_scope_priority,
    is_hierarchical_scope,
    is_resource_scope,
    HIERARCHY_ORDER,
)


@dataclass
class PolicyMatch:
    """Result of matching a policy to a usage context.

    Attributes:
        policy: The matched budget policy.
        match_type: How the policy matched ("exact", "inherited", "resource").
        match_scope: The scope that matched.
        match_id: The ID that matched.
        priority_score: Score for ranking matches (higher = more specific).
    """

    policy: BudgetPolicy
    match_type: str
    match_scope: BudgetScope
    match_id: Optional[str]
    priority_score: int = 0

    def __post_init__(self) -> None:
        """Calculate priority score if not set."""
        if self.priority_score == 0:
            # Base priority from scope
            self.priority_score = get_scope_priority(self.match_scope) * 10

            # Exact matches get a bonus
            if self.match_type == "exact":
                self.priority_score += 5

            # Policy's own priority as tiebreaker
            self.priority_score += self.policy.priority


@dataclass
class PolicyMatchResult:
    """Aggregated result of policy matching.

    Attributes:
        matches: All matching policies, ordered by priority.
        hierarchical_matches: Matches from organizational hierarchy.
        resource_matches: Matches from resource scopes (provider, model).
        most_restrictive: The policy with the most restrictive action.
        effective_limit: The lowest limit from all matching policies.
    """

    matches: List[PolicyMatch] = field(default_factory=list)
    hierarchical_matches: List[PolicyMatch] = field(default_factory=list)
    resource_matches: List[PolicyMatch] = field(default_factory=list)

    @property
    def has_matches(self) -> bool:
        """Check if any policies matched."""
        return len(self.matches) > 0

    @property
    def most_restrictive(self) -> Optional[PolicyMatch]:
        """Get the match with the most restrictive action."""
        if not self.matches:
            return None

        action_priority = {
            LimitAction.WARN: 1,
            LimitAction.DEGRADE: 2,
            LimitAction.THROTTLE: 3,
            LimitAction.SOFT_BLOCK: 4,
            LimitAction.BLOCK: 5,
        }

        return max(
            self.matches,
            key=lambda m: action_priority.get(m.policy.hard_limit_action, 0),
        )

    @property
    def effective_limit(self) -> Optional[Decimal]:
        """Get the lowest limit from all matching policies."""
        if not self.matches:
            return None
        return min(m.policy.limit_amount for m in self.matches)

    @property
    def most_specific(self) -> Optional[PolicyMatch]:
        """Get the most specific matching policy."""
        if not self.matches:
            return None
        return max(self.matches, key=lambda m: m.priority_score)

    def get_blocking_policies(self) -> List[PolicyMatch]:
        """Get all policies that would block the request."""
        blocking_actions = {LimitAction.SOFT_BLOCK, LimitAction.BLOCK}
        return [m for m in self.matches if m.policy.hard_limit_action in blocking_actions]


class PolicyMatcher:
    """Matches usage contexts against budget policies.

    Uses hierarchical scope resolution to find all applicable policies
    and ranks them by specificity and priority.
    """

    def __init__(
        self,
        policies: List[BudgetPolicy],
        scope_resolver: Optional[ScopeHierarchyResolver] = None,
    ) -> None:
        """Initialize the matcher with available policies.

        Args:
            policies: List of budget policies to match against.
            scope_resolver: Optional resolver for scope relationships.
        """
        self._policies = policies
        self._scope_resolver = scope_resolver

        # Index policies by scope for efficient lookup
        self._by_scope: dict[BudgetScope, dict[Optional[str], list[BudgetPolicy]]] = {}
        for policy in policies:
            if policy.scope not in self._by_scope:
                self._by_scope[policy.scope] = {}
            scope_policies = self._by_scope[policy.scope]
            if policy.scope_id not in scope_policies:
                scope_policies[policy.scope_id] = []
            scope_policies[policy.scope_id].append(policy)

    def match(self, context: UsageContext) -> PolicyMatchResult:
        """Find all policies that apply to the given context.

        Args:
            context: The usage context to match.

        Returns:
            PolicyMatchResult with all matching policies.
        """
        result = PolicyMatchResult()

        # Expand context with inherited scope IDs if resolver available
        effective_context = context
        if self._scope_resolver:
            effective_context = self._scope_resolver.expand_context(context)

        # Get all applicable scopes from context
        applicable_scopes = effective_context.get_applicable_scopes()

        # Match against each scope
        for scope, scope_id in applicable_scopes:
            scope_matches = self._match_scope(scope, scope_id, effective_context)
            result.matches.extend(scope_matches)

            # Categorize matches
            for match in scope_matches:
                if is_hierarchical_scope(match.match_scope):
                    result.hierarchical_matches.append(match)
                elif is_resource_scope(match.match_scope):
                    result.resource_matches.append(match)

        # Sort by priority (highest first)
        result.matches.sort(key=lambda m: m.priority_score, reverse=True)
        result.hierarchical_matches.sort(key=lambda m: m.priority_score, reverse=True)
        result.resource_matches.sort(key=lambda m: m.priority_score, reverse=True)

        return result

    def _match_scope(
        self,
        scope: BudgetScope,
        scope_id: Optional[str],
        context: UsageContext,
    ) -> List[PolicyMatch]:
        """Match policies for a specific scope.

        Args:
            scope: The scope to match.
            scope_id: The scope identifier.
            context: The full usage context.

        Returns:
            List of matches for this scope.
        """
        matches: List[PolicyMatch] = []

        if scope not in self._by_scope:
            return matches

        scope_policies = self._by_scope[scope]

        # Match GLOBAL (no scope_id) or specific scope_id
        if scope == BudgetScope.GLOBAL:
            # Match any policy with GLOBAL scope
            for policy in scope_policies.get(None, []):
                if policy.enabled:
                    matches.append(
                        PolicyMatch(
                            policy=policy,
                            match_type="exact",
                            match_scope=scope,
                            match_id=None,
                        )
                    )
        elif scope_id:
            # Exact match on scope_id
            for policy in scope_policies.get(scope_id, []):
                if policy.enabled:
                    matches.append(
                        PolicyMatch(
                            policy=policy,
                            match_type="exact",
                            match_scope=scope,
                            match_id=scope_id,
                        )
                    )

            # Also check for wildcard policies (scope_id = None means "all")
            for policy in scope_policies.get(None, []):
                if policy.enabled:
                    matches.append(
                        PolicyMatch(
                            policy=policy,
                            match_type="inherited",
                            match_scope=scope,
                            match_id=None,
                        )
                    )

        return matches

    def get_applicable_limits(
        self, context: UsageContext
    ) -> List[Tuple[BudgetPolicy, Decimal]]:
        """Get all applicable limits for a context.

        Returns list of (policy, effective_limit) tuples, ordered by
        restrictiveness (lowest limit first).
        """
        match_result = self.match(context)
        limits = [(m.policy, m.policy.limit_amount) for m in match_result.matches]
        limits.sort(key=lambda x: x[1])
        return limits


def build_scopes_to_check(
    context: UsageContext,
    scope_resolver: Optional[ScopeHierarchyResolver] = None,
) -> List[Tuple[BudgetScope, Optional[str]]]:
    """Build the list of scope/ID tuples to check for a context.

    This is used by the policy service to know which policies might apply.

    Args:
        context: The usage context.
        scope_resolver: Optional resolver to expand inherited scopes.

    Returns:
        List of (scope, scope_id) tuples to check.
    """
    # Expand context if resolver available
    effective_context = context
    if scope_resolver:
        effective_context = scope_resolver.expand_context(context)

    return effective_context.get_applicable_scopes()
