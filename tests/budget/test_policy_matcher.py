"""Tests for budget policy matching.

Tests the policy matching logic with hierarchical scope resolution.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from modules.budget.models import BudgetPolicy, BudgetScope, BudgetPeriod, LimitAction
from modules.budget.policy_matcher import PolicyMatch, PolicyMatcher, PolicyMatchResult
from modules.budget.scope_hierarchy import ScopeHierarchyResolver, UsageContext


def make_policy(
    name: str,
    scope: BudgetScope,
    scope_id: str | None = None,
    limit: Decimal = Decimal("100.00"),
    action: LimitAction = LimitAction.WARN,
    priority: int = 0,
    enabled: bool = True,
) -> BudgetPolicy:
    """Helper to create a test policy."""
    return BudgetPolicy(
        name=name,
        scope=scope,
        scope_id=scope_id,
        limit_amount=limit,
        hard_limit_action=action,
        priority=priority,
        enabled=enabled,
    )


class TestPolicyMatch:
    """Tests for PolicyMatch dataclass."""

    def test_priority_score_from_scope(self) -> None:
        """Priority score should increase with scope specificity."""
        global_policy = make_policy("global", BudgetScope.GLOBAL)
        user_policy = make_policy("user", BudgetScope.USER, "u1")

        global_match = PolicyMatch(
            policy=global_policy,
            match_type="exact",
            match_scope=BudgetScope.GLOBAL,
            match_id=None,
        )
        user_match = PolicyMatch(
            policy=user_policy,
            match_type="exact",
            match_scope=BudgetScope.USER,
            match_id="u1",
        )

        assert user_match.priority_score > global_match.priority_score

    def test_exact_match_bonus(self) -> None:
        """Exact matches should score higher than inherited."""
        policy = make_policy("test", BudgetScope.TEAM, "t1")

        exact = PolicyMatch(
            policy=policy,
            match_type="exact",
            match_scope=BudgetScope.TEAM,
            match_id="t1",
        )
        inherited = PolicyMatch(
            policy=policy,
            match_type="inherited",
            match_scope=BudgetScope.TEAM,
            match_id="t1",
        )

        assert exact.priority_score > inherited.priority_score


class TestPolicyMatchResult:
    """Tests for PolicyMatchResult aggregation."""

    def test_has_matches(self) -> None:
        """has_matches should reflect match count."""
        result = PolicyMatchResult()
        assert not result.has_matches

        result.matches.append(
            PolicyMatch(
                policy=make_policy("test", BudgetScope.GLOBAL),
                match_type="exact",
                match_scope=BudgetScope.GLOBAL,
                match_id=None,
            )
        )
        assert result.has_matches

    def test_most_restrictive_action(self) -> None:
        """most_restrictive should return highest priority action."""
        warn_policy = make_policy("warn", BudgetScope.GLOBAL, action=LimitAction.WARN)
        block_policy = make_policy("block", BudgetScope.USER, "u1", action=LimitAction.BLOCK)

        result = PolicyMatchResult(
            matches=[
                PolicyMatch(
                    policy=warn_policy,
                    match_type="exact",
                    match_scope=BudgetScope.GLOBAL,
                    match_id=None,
                ),
                PolicyMatch(
                    policy=block_policy,
                    match_type="exact",
                    match_scope=BudgetScope.USER,
                    match_id="u1",
                ),
            ]
        )

        most_restrictive = result.most_restrictive
        assert most_restrictive is not None
        assert most_restrictive.policy.hard_limit_action == LimitAction.BLOCK

    def test_effective_limit_minimum(self) -> None:
        """effective_limit should return lowest limit."""
        low_limit = make_policy("low", BudgetScope.GLOBAL, limit=Decimal("50.00"))
        high_limit = make_policy("high", BudgetScope.USER, "u1", limit=Decimal("100.00"))

        result = PolicyMatchResult(
            matches=[
                PolicyMatch(
                    policy=high_limit,
                    match_type="exact",
                    match_scope=BudgetScope.USER,
                    match_id="u1",
                ),
                PolicyMatch(
                    policy=low_limit,
                    match_type="exact",
                    match_scope=BudgetScope.GLOBAL,
                    match_id=None,
                ),
            ]
        )

        assert result.effective_limit == Decimal("50.00")

    def test_get_blocking_policies(self) -> None:
        """Should return only blocking action policies."""
        policies = [
            make_policy("warn", BudgetScope.GLOBAL, action=LimitAction.WARN),
            make_policy("block", BudgetScope.USER, "u1", action=LimitAction.BLOCK),
            make_policy("soft", BudgetScope.TEAM, "t1", action=LimitAction.SOFT_BLOCK),
        ]

        result = PolicyMatchResult(
            matches=[
                PolicyMatch(
                    policy=p,
                    match_type="exact",
                    match_scope=p.scope,
                    match_id=p.scope_id,
                )
                for p in policies
            ]
        )

        blocking = result.get_blocking_policies()
        assert len(blocking) == 2
        names = {m.policy.name for m in blocking}
        assert "block" in names
        assert "soft" in names


class TestPolicyMatcher:
    """Tests for PolicyMatcher matching logic."""

    def test_match_global_policy(self) -> None:
        """Should match GLOBAL policies."""
        policy = make_policy("global", BudgetScope.GLOBAL)
        matcher = PolicyMatcher([policy])

        context = UsageContext()
        result = matcher.match(context)

        assert result.has_matches
        assert len(result.matches) == 1
        assert result.matches[0].policy.name == "global"

    def test_match_user_policy(self) -> None:
        """Should match USER scope by ID."""
        user_policy = make_policy("user-policy", BudgetScope.USER, "user-123")
        other_policy = make_policy("other-user", BudgetScope.USER, "user-456")
        matcher = PolicyMatcher([user_policy, other_policy])

        context = UsageContext(user_id="user-123")
        result = matcher.match(context)

        # Should match: GLOBAL (none) + USER exact
        assert len(result.matches) == 1
        assert result.matches[0].policy.name == "user-policy"

    def test_match_provider_and_model(self) -> None:
        """Should match resource scopes."""
        provider_policy = make_policy("openai", BudgetScope.PROVIDER, "openai")
        model_policy = make_policy("gpt4", BudgetScope.MODEL, "gpt-4")
        matcher = PolicyMatcher([provider_policy, model_policy])

        context = UsageContext(provider="openai", model="gpt-4")
        result = matcher.match(context)

        assert len(result.matches) == 2
        assert len(result.resource_matches) == 2

    def test_match_agent_in_team(self) -> None:
        """Should match AGENT scope."""
        agent_policy = make_policy("agent-limit", BudgetScope.AGENT, "agent-1")
        matcher = PolicyMatcher([agent_policy])

        context = UsageContext(agent_id="agent-1", user_id="user-1")
        result = matcher.match(context)

        assert result.has_matches
        assert result.matches[0].match_scope == BudgetScope.AGENT

    def test_match_team_policy(self) -> None:
        """Should match TEAM scope."""
        team_policy = make_policy("team-budget", BudgetScope.TEAM, "team-a")
        matcher = PolicyMatcher([team_policy])

        context = UsageContext(team_id="team-a")
        result = matcher.match(context)

        assert result.has_matches
        assert result.matches[0].match_scope == BudgetScope.TEAM

    def test_match_project_policy(self) -> None:
        """Should match PROJECT scope."""
        project_policy = make_policy("project-budget", BudgetScope.PROJECT, "proj-1")
        matcher = PolicyMatcher([project_policy])

        context = UsageContext(project_id="proj-1")
        result = matcher.match(context)

        assert result.has_matches
        assert result.matches[0].match_scope == BudgetScope.PROJECT

    def test_match_session_policy(self) -> None:
        """Should match SESSION scope."""
        session_policy = make_policy("session-limit", BudgetScope.SESSION, "sess-abc")
        matcher = PolicyMatcher([session_policy])

        context = UsageContext(session_id="sess-abc")
        result = matcher.match(context)

        assert result.has_matches
        assert result.matches[0].match_scope == BudgetScope.SESSION

    def test_disabled_policy_not_matched(self) -> None:
        """Disabled policies should not match."""
        policy = make_policy("disabled", BudgetScope.GLOBAL, enabled=False)
        matcher = PolicyMatcher([policy])

        context = UsageContext()
        result = matcher.match(context)

        assert not result.has_matches

    def test_match_ordering_by_priority(self) -> None:
        """Matches should be ordered by priority score."""
        global_policy = make_policy("global", BudgetScope.GLOBAL)
        user_policy = make_policy("user", BudgetScope.USER, "u1")
        agent_policy = make_policy("agent", BudgetScope.AGENT, "a1")

        matcher = PolicyMatcher([global_policy, user_policy, agent_policy])

        context = UsageContext(user_id="u1", agent_id="a1")
        result = matcher.match(context)

        # Should be ordered: USER > AGENT > GLOBAL (by specificity in hierarchy)
        # USER is at index 5, AGENT at index 4, GLOBAL at index 0
        assert len(result.matches) == 3
        assert result.matches[0].match_scope == BudgetScope.USER
        assert result.matches[1].match_scope == BudgetScope.AGENT
        assert result.matches[2].match_scope == BudgetScope.GLOBAL


class TestPolicyMatcherWithHierarchy:
    """Tests for PolicyMatcher with scope hierarchy resolution."""

    @pytest.fixture
    def resolver(self) -> ScopeHierarchyResolver:
        """Create resolver with team structure."""
        resolver = ScopeHierarchyResolver()
        # project-1 belongs to team-a
        resolver.register_relationship(
            BudgetScope.PROJECT, "proj-1", BudgetScope.TEAM, "team-a"
        )
        # agent-1 belongs to project-1
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.PROJECT, "proj-1"
        )
        return resolver

    def test_inherits_team_from_agent(self, resolver: ScopeHierarchyResolver) -> None:
        """Agent context should inherit team policies."""
        team_policy = make_policy("team-budget", BudgetScope.TEAM, "team-a")
        matcher = PolicyMatcher([team_policy], scope_resolver=resolver)

        # Context only has agent_id
        context = UsageContext(agent_id="agent-1")
        result = matcher.match(context)

        # Should match team policy via inheritance
        assert result.has_matches
        team_matches = [m for m in result.matches if m.match_scope == BudgetScope.TEAM]
        assert len(team_matches) == 1

    def test_inherits_project_from_agent(self, resolver: ScopeHierarchyResolver) -> None:
        """Agent should inherit project policies."""
        project_policy = make_policy("project-budget", BudgetScope.PROJECT, "proj-1")
        matcher = PolicyMatcher([project_policy], scope_resolver=resolver)

        # Context only has agent_id
        context = UsageContext(agent_id="agent-1")
        result = matcher.match(context)

        # Should match project policy via inheritance
        assert result.has_matches
        project_matches = [m for m in result.matches if m.match_scope == BudgetScope.PROJECT]
        assert len(project_matches) == 1

    def test_all_levels_matched(self, resolver: ScopeHierarchyResolver) -> None:
        """Should match policies at all inheritance levels."""
        policies = [
            make_policy("global", BudgetScope.GLOBAL),
            make_policy("project", BudgetScope.PROJECT, "proj-1"),
            make_policy("team", BudgetScope.TEAM, "team-a"),
            make_policy("agent", BudgetScope.AGENT, "agent-1"),
        ]
        matcher = PolicyMatcher(policies, scope_resolver=resolver)

        context = UsageContext(agent_id="agent-1")
        result = matcher.match(context)

        # Should match all 4 levels
        assert len(result.matches) == 4
        scopes = {m.match_scope for m in result.matches}
        assert scopes == {
            BudgetScope.GLOBAL,
            BudgetScope.PROJECT,
            BudgetScope.TEAM,
            BudgetScope.AGENT,
        }


class TestGetApplicableLimits:
    """Tests for getting applicable limits from matcher."""

    def test_limits_ordered_by_amount(self) -> None:
        """Limits should be ordered lowest first."""
        policies = [
            make_policy("high", BudgetScope.GLOBAL, limit=Decimal("1000.00")),
            make_policy("low", BudgetScope.USER, "u1", limit=Decimal("50.00")),
            make_policy("medium", BudgetScope.TEAM, "t1", limit=Decimal("200.00")),
        ]
        matcher = PolicyMatcher(policies)

        context = UsageContext(user_id="u1", team_id="t1")
        limits = matcher.get_applicable_limits(context)

        assert len(limits) == 3
        assert limits[0][1] == Decimal("50.00")
        assert limits[1][1] == Decimal("200.00")
        assert limits[2][1] == Decimal("1000.00")
