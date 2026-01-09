"""Tests for budget scope hierarchy resolution.

Tests the hierarchical scope relationships and policy resolution logic.
"""

from __future__ import annotations

import pytest

from modules.budget.models import BudgetScope
from modules.budget.scope_hierarchy import (
    HIERARCHY_ORDER,
    RESOURCE_ORDER,
    SCOPE_CATEGORIES,
    ScopeCategory,
    ScopeHierarchyResolver,
    UsageContext,
    get_scope_priority,
    get_scope_resolver,
    is_hierarchical_scope,
    is_resource_scope,
    reset_scope_resolver,
)


class TestScopeCategories:
    """Tests for scope categorization."""

    def test_all_scopes_categorized(self) -> None:
        """All BudgetScope values should have a category."""
        for scope in BudgetScope:
            assert scope in SCOPE_CATEGORIES, f"{scope} not categorized"

    def test_hierarchical_scopes(self) -> None:
        """Verify hierarchical scope identification."""
        hierarchical = [
            BudgetScope.GLOBAL,
            BudgetScope.TEAM,
            BudgetScope.PROJECT,
            BudgetScope.JOB,
            BudgetScope.TASK,
            BudgetScope.AGENT,
            BudgetScope.USER,
            BudgetScope.SESSION,
        ]
        for scope in hierarchical:
            assert is_hierarchical_scope(scope), f"{scope} should be hierarchical"
            assert not is_resource_scope(scope), f"{scope} should not be resource"

    def test_resource_scopes(self) -> None:
        """Verify resource scope identification."""
        resource = [BudgetScope.PROVIDER, BudgetScope.MODEL, BudgetScope.TOOL, BudgetScope.SKILL]
        for scope in resource:
            assert is_resource_scope(scope), f"{scope} should be resource"
            assert not is_hierarchical_scope(scope), f"{scope} should not be hierarchical"


class TestScopePriority:
    """Tests for scope priority ordering."""

    def test_global_is_lowest_priority(self) -> None:
        """GLOBAL should have the lowest priority (0)."""
        assert get_scope_priority(BudgetScope.GLOBAL) == 0

    def test_session_is_highest_hierarchical(self) -> None:
        """SESSION should be most specific in hierarchy."""
        assert get_scope_priority(BudgetScope.SESSION) == len(HIERARCHY_ORDER) - 1

    def test_hierarchy_order_is_increasing(self) -> None:
        """Each scope should have higher priority than its parent."""
        for i in range(1, len(HIERARCHY_ORDER)):
            parent = HIERARCHY_ORDER[i - 1]
            child = HIERARCHY_ORDER[i]
            assert get_scope_priority(child) > get_scope_priority(parent), (
                f"{child} should have higher priority than {parent}"
            )

    def test_resource_scopes_have_separate_priority(self) -> None:
        """Resource scopes should have priorities after hierarchical."""
        hierarchical_max = len(HIERARCHY_ORDER) - 1
        for scope in RESOURCE_ORDER:
            assert get_scope_priority(scope) > hierarchical_max


class TestUsageContext:
    """Tests for UsageContext functionality."""

    def test_empty_context(self) -> None:
        """Empty context should only return GLOBAL scope."""
        ctx = UsageContext()
        scopes = ctx.get_applicable_scopes()
        assert len(scopes) == 1
        assert scopes[0] == (BudgetScope.GLOBAL, None)

    def test_full_context(self) -> None:
        """Full context should return all scopes."""
        ctx = UsageContext(
            team_id="team1",
            project_id="proj1",
            job_id="job1",
            task_id="task1",
            agent_id="agent1",
            user_id="user1",
            session_id="sess1",
            provider="openai",
            model="gpt-4",
            tool_id="tool1",
            skill_id="skill1",
        )
        scopes = ctx.get_applicable_scopes()

        # Should have: GLOBAL + 7 hierarchical + 4 resource = 12
        assert len(scopes) == 12

        # Verify order (hierarchical first, then resource)
        scope_types = [s[0] for s in scopes]
        assert scope_types[0] == BudgetScope.GLOBAL
        assert scope_types[-4] == BudgetScope.PROVIDER
        assert scope_types[-3] == BudgetScope.MODEL
        assert scope_types[-2] == BudgetScope.TOOL
        assert scope_types[-1] == BudgetScope.SKILL

    def test_partial_context(self) -> None:
        """Partial context should only include populated scopes."""
        ctx = UsageContext(
            user_id="user1",
            provider="anthropic",
        )
        scopes = ctx.get_applicable_scopes()

        # Should have: GLOBAL, USER, PROVIDER = 3
        assert len(scopes) == 3
        assert (BudgetScope.GLOBAL, None) in scopes
        assert (BudgetScope.USER, "user1") in scopes
        assert (BudgetScope.PROVIDER, "anthropic") in scopes

    def test_get_scope_id(self) -> None:
        """get_scope_id should return correct values."""
        ctx = UsageContext(
            team_id="t1",
            agent_id="agent1",
            model="gpt-4o",
        )
        assert ctx.get_scope_id(BudgetScope.GLOBAL) is None
        assert ctx.get_scope_id(BudgetScope.TEAM) == "t1"
        assert ctx.get_scope_id(BudgetScope.PROJECT) is None
        assert ctx.get_scope_id(BudgetScope.AGENT) == "agent1"
        assert ctx.get_scope_id(BudgetScope.MODEL) == "gpt-4o"


class TestScopeHierarchyResolver:
    """Tests for ScopeHierarchyResolver."""

    @pytest.fixture
    def resolver(self) -> ScopeHierarchyResolver:
        """Create a fresh resolver for each test."""
        return ScopeHierarchyResolver()

    def test_register_valid_relationship(self, resolver: ScopeHierarchyResolver) -> None:
        """Should register valid parent-child relationships."""
        # Agent belongs to team
        resolver.register_relationship(
            child_scope=BudgetScope.AGENT,
            child_id="agent-1",
            parent_scope=BudgetScope.TEAM,
            parent_id="team-1",
        )

        ancestors = resolver.get_ancestors(BudgetScope.AGENT, "agent-1")
        assert (BudgetScope.TEAM, "team-1") in ancestors

    def test_reject_invalid_hierarchy(self, resolver: ScopeHierarchyResolver) -> None:
        """Should reject relationships that violate hierarchy order."""
        # Cannot make GLOBAL a child of USER
        with pytest.raises(ValueError, match="Invalid hierarchy"):
            resolver.register_relationship(
                child_scope=BudgetScope.GLOBAL,
                child_id="",
                parent_scope=BudgetScope.USER,
                parent_id="user-1",
            )

    def test_multi_level_hierarchy(self, resolver: ScopeHierarchyResolver) -> None:
        """Should handle multi-level hierarchies."""
        # Build: GLOBAL > TEAM > PROJECT > JOB > AGENT
        resolver.register_relationship(
            BudgetScope.TEAM, "team-1", BudgetScope.GLOBAL, None
        )
        resolver.register_relationship(
            BudgetScope.PROJECT, "proj-1", BudgetScope.TEAM, "team-1"
        )
        resolver.register_relationship(
            BudgetScope.JOB, "job-1", BudgetScope.PROJECT, "proj-1"
        )
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.JOB, "job-1"
        )

        # Agent should have 4 ancestors (JOB, PROJECT, TEAM, GLOBAL)
        ancestors = resolver.get_ancestors(BudgetScope.AGENT, "agent-1")
        assert len(ancestors) == 4
        assert (BudgetScope.JOB, "job-1") in ancestors
        assert (BudgetScope.PROJECT, "proj-1") in ancestors
        assert (BudgetScope.TEAM, "team-1") in ancestors
        assert (BudgetScope.GLOBAL, None) in ancestors

    def test_get_descendants(self, resolver: ScopeHierarchyResolver) -> None:
        """Should return all descendants of an entity."""
        # Team > Project > 2 Agents
        resolver.register_relationship(
            BudgetScope.PROJECT, "proj-1", BudgetScope.TEAM, "team-1"
        )
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.PROJECT, "proj-1"
        )
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-2", BudgetScope.PROJECT, "proj-1"
        )

        descendants = resolver.get_descendants(BudgetScope.TEAM, "team-1")
        assert len(descendants) == 3  # project + 2 agents

    def test_get_full_lineage(self, resolver: ScopeHierarchyResolver) -> None:
        """Should return full lineage from GLOBAL to entity."""
        resolver.register_relationship(
            BudgetScope.PROJECT, "proj-1", BudgetScope.TEAM, "team-1"
        )
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.PROJECT, "proj-1"
        )

        lineage = resolver.get_full_lineage(BudgetScope.AGENT, "agent-1")

        # Should be: GLOBAL, TEAM, PROJECT, AGENT (in order)
        assert len(lineage) == 4
        assert lineage[0][0] == BudgetScope.GLOBAL
        assert lineage[-1] == (BudgetScope.AGENT, "agent-1")

    def test_unregister_relationship(self, resolver: ScopeHierarchyResolver) -> None:
        """Should remove relationships correctly."""
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.TEAM, "team-1"
        )

        removed = resolver.unregister_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.TEAM, "team-1"
        )
        assert removed is True

        # Verify removed
        ancestors = resolver.get_ancestors(BudgetScope.AGENT, "agent-1")
        assert len(ancestors) == 0

    def test_unregister_nonexistent(self, resolver: ScopeHierarchyResolver) -> None:
        """Unregistering nonexistent relationship should return False."""
        removed = resolver.unregister_relationship(
            BudgetScope.AGENT, "agent-x", BudgetScope.TEAM, "team-x"
        )
        assert removed is False

    def test_expand_context(self, resolver: ScopeHierarchyResolver) -> None:
        """Should expand context with inherited scope IDs."""
        # Register: agent-1 belongs to project-1, which belongs to team-1
        resolver.register_relationship(
            BudgetScope.PROJECT, "proj-1", BudgetScope.TEAM, "team-1"
        )
        resolver.register_relationship(
            BudgetScope.AGENT, "agent-1", BudgetScope.PROJECT, "proj-1"
        )

        # Context only has agent_id
        ctx = UsageContext(agent_id="agent-1", provider="openai")

        expanded = resolver.expand_context(ctx)

        # Should have inherited project_id and team_id
        assert expanded.agent_id == "agent-1"
        assert expanded.project_id == "proj-1"
        assert expanded.team_id == "team-1"
        assert expanded.provider == "openai"  # Preserved

    def test_expand_context_no_relationships(
        self, resolver: ScopeHierarchyResolver
    ) -> None:
        """Expand should return same context if no relationships."""
        ctx = UsageContext(user_id="user-1")
        expanded = resolver.expand_context(ctx)
        assert expanded.user_id == "user-1"
        assert expanded.team_id is None


class TestGlobalResolver:
    """Tests for the global resolver singleton."""

    def setup_method(self) -> None:
        """Reset global resolver before each test."""
        reset_scope_resolver()

    def teardown_method(self) -> None:
        """Reset global resolver after each test."""
        reset_scope_resolver()

    def test_get_scope_resolver_singleton(self) -> None:
        """Should return the same resolver instance."""
        r1 = get_scope_resolver()
        r2 = get_scope_resolver()
        assert r1 is r2

    def test_reset_scope_resolver(self) -> None:
        """Reset should create a new instance."""
        r1 = get_scope_resolver()
        r1.register_relationship(
            BudgetScope.AGENT, "a1", BudgetScope.TEAM, "t1"
        )

        reset_scope_resolver()
        r2 = get_scope_resolver()

        assert r1 is not r2
        assert len(r2.get_ancestors(BudgetScope.AGENT, "a1")) == 0


class TestMultipleParents:
    """Tests for entities with multiple parent relationships."""

    @pytest.fixture
    def resolver(self) -> ScopeHierarchyResolver:
        return ScopeHierarchyResolver()

    def test_agent_in_multiple_teams(self, resolver: ScopeHierarchyResolver) -> None:
        """An agent can belong to multiple teams (shared resource)."""
        resolver.register_relationship(
            BudgetScope.AGENT, "shared-agent", BudgetScope.TEAM, "team-a"
        )
        resolver.register_relationship(
            BudgetScope.AGENT, "shared-agent", BudgetScope.TEAM, "team-b"
        )

        ancestors = resolver.get_ancestors(BudgetScope.AGENT, "shared-agent")
        assert len(ancestors) == 2
        assert (BudgetScope.TEAM, "team-a") in ancestors
        assert (BudgetScope.TEAM, "team-b") in ancestors

    def test_project_shared_across_teams(
        self, resolver: ScopeHierarchyResolver
    ) -> None:
        """A project can be shared across multiple teams."""
        resolver.register_relationship(
            BudgetScope.PROJECT, "cross-team-proj", BudgetScope.TEAM, "team-a"
        )
        resolver.register_relationship(
            BudgetScope.PROJECT, "cross-team-proj", BudgetScope.TEAM, "team-b"
        )

        ancestors = resolver.get_ancestors(BudgetScope.PROJECT, "cross-team-proj")
        assert (BudgetScope.TEAM, "team-a") in ancestors
        assert (BudgetScope.TEAM, "team-b") in ancestors
