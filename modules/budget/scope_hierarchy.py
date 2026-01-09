"""Budget scope hierarchy and resolution logic.

Defines the hierarchical relationships between budget scopes and provides
utilities for resolving which policies apply to a given usage context.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import BudgetScope


class ScopeCategory(Enum):
    """Category of budget scope for resolution purposes."""

    HIERARCHICAL = "hierarchical"  # Part of org hierarchy (global > tenant > team > etc.)
    RESOURCE = "resource"  # Resource-based constraint (provider, model)


# Define which category each scope belongs to
SCOPE_CATEGORIES: Dict[BudgetScope, ScopeCategory] = {
    BudgetScope.GLOBAL: ScopeCategory.HIERARCHICAL,
    BudgetScope.TEAM: ScopeCategory.HIERARCHICAL,
    BudgetScope.PROJECT: ScopeCategory.HIERARCHICAL,
    BudgetScope.JOB: ScopeCategory.HIERARCHICAL,
    BudgetScope.TASK: ScopeCategory.HIERARCHICAL,
    BudgetScope.AGENT: ScopeCategory.HIERARCHICAL,
    BudgetScope.USER: ScopeCategory.HIERARCHICAL,
    BudgetScope.SESSION: ScopeCategory.HIERARCHICAL,
    BudgetScope.PROVIDER: ScopeCategory.RESOURCE,
    BudgetScope.MODEL: ScopeCategory.RESOURCE,
    BudgetScope.TOOL: ScopeCategory.RESOURCE,
    BudgetScope.SKILL: ScopeCategory.RESOURCE,
}

# Hierarchical scope ordering (index = priority, lower = broader)
HIERARCHY_ORDER: List[BudgetScope] = [
    BudgetScope.GLOBAL,
    BudgetScope.TEAM,
    BudgetScope.PROJECT,
    BudgetScope.JOB,
    BudgetScope.TASK,
    BudgetScope.AGENT,
    BudgetScope.USER,
    BudgetScope.SESSION,
]

# Resource scope ordering
RESOURCE_ORDER: List[BudgetScope] = [
    BudgetScope.PROVIDER,
    BudgetScope.MODEL,
    BudgetScope.TOOL,
    BudgetScope.SKILL,
]


def get_scope_priority(scope: BudgetScope) -> int:
    """Get the priority of a scope (higher = more specific).

    Args:
        scope: The budget scope.

    Returns:
        Priority value where higher means more specific.
    """
    if scope in HIERARCHY_ORDER:
        return HIERARCHY_ORDER.index(scope)
    elif scope in RESOURCE_ORDER:
        # Resource scopes get base priority + their index
        return len(HIERARCHY_ORDER) + RESOURCE_ORDER.index(scope)
    return -1


def is_hierarchical_scope(scope: BudgetScope) -> bool:
    """Check if a scope is part of the organizational hierarchy."""
    return SCOPE_CATEGORIES.get(scope) == ScopeCategory.HIERARCHICAL


def is_resource_scope(scope: BudgetScope) -> bool:
    """Check if a scope is a resource-based constraint."""
    return SCOPE_CATEGORIES.get(scope) == ScopeCategory.RESOURCE


@dataclass
class UsageContext:
    """Context for a usage event to match against policies.

    This captures all the dimensional information needed to determine
    which budget policies apply to a particular usage.
    """

    # Hierarchical context (organizational)
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    job_id: Optional[str] = None
    task_id: Optional[str] = None
    agent_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None

    # Resource context
    provider: Optional[str] = None
    model: Optional[str] = None
    tool_id: Optional[str] = None
    skill_id: Optional[str] = None

    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_scope_id(self, scope: BudgetScope) -> Optional[str]:
        """Get the identifier for a specific scope from this context.

        Args:
            scope: The budget scope to get the ID for.

        Returns:
            The scope identifier, or None if not set.
        """
        mapping = {
            BudgetScope.GLOBAL: None,  # Global has no ID
            BudgetScope.TEAM: self.team_id,
            BudgetScope.PROJECT: self.project_id,
            BudgetScope.JOB: self.job_id,
            BudgetScope.TASK: self.task_id,
            BudgetScope.AGENT: self.agent_id,
            BudgetScope.USER: self.user_id,
            BudgetScope.SESSION: self.session_id,
            BudgetScope.PROVIDER: self.provider,
            BudgetScope.MODEL: self.model,
            BudgetScope.TOOL: self.tool_id,
            BudgetScope.SKILL: self.skill_id,
        }
        return mapping.get(scope)

    def get_applicable_scopes(self) -> List[Tuple[BudgetScope, Optional[str]]]:
        """Get all scopes that are populated in this context.

        Returns list of (scope, scope_id) tuples for matching policies.
        Ordered from broadest (GLOBAL) to most specific.
        """
        result: List[Tuple[BudgetScope, Optional[str]]] = []

        # Always include GLOBAL
        result.append((BudgetScope.GLOBAL, None))

        # Add hierarchical scopes in order
        for scope in HIERARCHY_ORDER[1:]:  # Skip GLOBAL, already added
            scope_id = self.get_scope_id(scope)
            if scope_id is not None:
                result.append((scope, scope_id))

        # Add resource scopes
        for scope in RESOURCE_ORDER:
            scope_id = self.get_scope_id(scope)
            if scope_id is not None:
                result.append((scope, scope_id))

        return result


@dataclass
class ScopeRelationship:
    """Defines a relationship in the scope hierarchy.

    For example: agent "agent-123" belongs to team "team-456".
    """

    child_scope: BudgetScope
    child_id: str
    parent_scope: BudgetScope
    parent_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)


class ScopeHierarchyResolver:
    """Resolves scope relationships and inheritance.

    Maintains the graph of scope relationships (which agents belong to
    which teams, which teams belong to which projects, etc.) and provides
    methods to traverse and resolve policy inheritance.
    """

    def __init__(self) -> None:
        """Initialize the resolver with empty relationship registry."""
        # Map: (child_scope, child_id) -> list of (parent_scope, parent_id)
        self._parents: Dict[Tuple[BudgetScope, str], List[Tuple[BudgetScope, str]]] = {}
        # Map: (parent_scope, parent_id) -> list of (child_scope, child_id)
        self._children: Dict[Tuple[BudgetScope, str], List[Tuple[BudgetScope, str]]] = {}

    def register_relationship(
        self,
        child_scope: BudgetScope,
        child_id: str,
        parent_scope: BudgetScope,
        parent_id: str,
    ) -> None:
        """Register a hierarchical relationship.

        Args:
            child_scope: The scope type of the child.
            child_id: The identifier of the child entity.
            parent_scope: The scope type of the parent.
            parent_id: The identifier of the parent entity.

        Raises:
            ValueError: If the relationship violates hierarchy rules.
        """
        # Validate hierarchy order
        child_priority = get_scope_priority(child_scope)
        parent_priority = get_scope_priority(parent_scope)

        if child_priority <= parent_priority:
            raise ValueError(
                f"Invalid hierarchy: {child_scope.value} cannot be child of {parent_scope.value}"
            )

        child_key = (child_scope, child_id)
        parent_key = (parent_scope, parent_id)

        # Add to parents map
        if child_key not in self._parents:
            self._parents[child_key] = []
        if parent_key not in self._parents[child_key]:
            self._parents[child_key].append(parent_key)

        # Add to children map
        if parent_key not in self._children:
            self._children[parent_key] = []
        if child_key not in self._children[parent_key]:
            self._children[parent_key].append(child_key)

    def unregister_relationship(
        self,
        child_scope: BudgetScope,
        child_id: str,
        parent_scope: BudgetScope,
        parent_id: str,
    ) -> bool:
        """Remove a hierarchical relationship.

        Returns:
            True if the relationship existed and was removed.
        """
        child_key = (child_scope, child_id)
        parent_key = (parent_scope, parent_id)

        removed = False

        if child_key in self._parents:
            if parent_key in self._parents[child_key]:
                self._parents[child_key].remove(parent_key)
                removed = True
            if not self._parents[child_key]:
                del self._parents[child_key]

        if parent_key in self._children:
            if child_key in self._children[parent_key]:
                self._children[parent_key].remove(child_key)
            if not self._children[parent_key]:
                del self._children[parent_key]

        return removed

    def get_ancestors(
        self, scope: BudgetScope, scope_id: str
    ) -> List[Tuple[BudgetScope, str]]:
        """Get all ancestors of an entity in hierarchy order.

        Args:
            scope: The scope type of the entity.
            scope_id: The identifier of the entity.

        Returns:
            List of (scope, id) tuples for all ancestors, ordered from
            immediate parent to most distant ancestor.
        """
        key = (scope, scope_id)
        if key not in self._parents:
            return []

        result: List[Tuple[BudgetScope, str]] = []
        visited: Set[Tuple[BudgetScope, str]] = set()
        queue = list(self._parents.get(key, []))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self._parents.get(current, []))

        return result

    def get_descendants(
        self, scope: BudgetScope, scope_id: str
    ) -> List[Tuple[BudgetScope, str]]:
        """Get all descendants of an entity.

        Args:
            scope: The scope type of the entity.
            scope_id: The identifier of the entity.

        Returns:
            List of (scope, id) tuples for all descendants.
        """
        key = (scope, scope_id)
        if key not in self._children:
            return []

        result: List[Tuple[BudgetScope, str]] = []
        visited: Set[Tuple[BudgetScope, str]] = set()
        queue = list(self._children.get(key, []))

        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self._children.get(current, []))

        return result

    def get_full_lineage(
        self, scope: BudgetScope, scope_id: str
    ) -> List[Tuple[BudgetScope, str]]:
        """Get the full lineage from GLOBAL to the given entity.

        Args:
            scope: The scope type of the entity.
            scope_id: The identifier of the entity.

        Returns:
            List of (scope, id) tuples from broadest ancestor to self,
            starting with (GLOBAL, None).
        """
        # Start with ancestors in reverse order (broadest first)
        ancestors = self.get_ancestors(scope, scope_id)
        ancestors.reverse()

        # Build lineage: GLOBAL + ancestors + self
        lineage: List[Tuple[BudgetScope, str]] = [(BudgetScope.GLOBAL, "")]

        # Sort ancestors by scope priority
        sorted_ancestors = sorted(ancestors, key=lambda x: get_scope_priority(x[0]))
        lineage.extend(sorted_ancestors)

        # Add self
        lineage.append((scope, scope_id))

        return lineage

    def expand_context(self, context: UsageContext) -> UsageContext:
        """Expand a usage context with inherited scope IDs.

        If the context has an agent_id but no team_id, and the agent
        is registered as belonging to a team, this will fill in the
        team_id (and any other missing ancestors).

        Args:
            context: The original usage context.

        Returns:
            A new context with inherited scope IDs populated.
        """
        # Find the most specific hierarchical scope in the context
        most_specific: Optional[Tuple[BudgetScope, str]] = None

        for scope in reversed(HIERARCHY_ORDER):
            scope_id = context.get_scope_id(scope)
            if scope_id:
                most_specific = (scope, scope_id)
                break

        if not most_specific:
            return context

        # Get ancestors
        ancestors = self.get_ancestors(most_specific[0], most_specific[1])

        # Build updated context dict
        updates: Dict[str, Optional[str]] = {}
        for ancestor_scope, ancestor_id in ancestors:
            field_name = self._scope_to_field(ancestor_scope)
            if field_name and getattr(context, field_name, None) is None:
                updates[field_name] = ancestor_id

        if not updates:
            return context

        # Create new context with updates
        return UsageContext(
            team_id=updates.get("team_id", context.team_id),
            project_id=updates.get("project_id", context.project_id),
            job_id=updates.get("job_id", context.job_id),
            task_id=updates.get("task_id", context.task_id),
            agent_id=context.agent_id,
            user_id=context.user_id,
            session_id=context.session_id,
            provider=context.provider,
            model=context.model,
            tool_id=context.tool_id,
            skill_id=context.skill_id,
            metadata=context.metadata,
        )

    @staticmethod
    def _scope_to_field(scope: BudgetScope) -> Optional[str]:
        """Map a scope to its UsageContext field name."""
        mapping = {
            BudgetScope.TEAM: "team_id",
            BudgetScope.PROJECT: "project_id",
            BudgetScope.JOB: "job_id",
            BudgetScope.TASK: "task_id",
            BudgetScope.AGENT: "agent_id",
            BudgetScope.USER: "user_id",
            BudgetScope.SESSION: "session_id",
        }
        return mapping.get(scope)


# Default global resolver instance
_default_resolver: Optional[ScopeHierarchyResolver] = None


def get_scope_resolver() -> ScopeHierarchyResolver:
    """Get the default scope hierarchy resolver."""
    global _default_resolver
    if _default_resolver is None:
        _default_resolver = ScopeHierarchyResolver()
    return _default_resolver


def reset_scope_resolver() -> None:
    """Reset the default scope resolver (for testing)."""
    global _default_resolver
    _default_resolver = None
