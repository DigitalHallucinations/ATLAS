"""Structure alignment paths for mediation conversations."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Position:
    """Participant position captured during mediation."""

    participant: str
    statement: str
    non_negotiables: tuple[str, ...]


class ConflictResolver:
    """Outline shared ground, options, and owners from differing positions."""

    async def run(
        self,
        *,
        positions: Sequence[Mapping[str, object]],
        shared_goals: Sequence[str] | None = None,
        decision_horizon: str | None = None,
    ) -> Mapping[str, object]:
        if not isinstance(positions, Sequence) or not positions:
            raise ValueError("conflict_resolver requires at least one position.")

        await asyncio.sleep(0)

        normalized_positions = [_normalize_position(entry) for entry in positions if entry]
        if not normalized_positions:
            raise ValueError("conflict_resolver requires valid position entries.")

        shared_focus = _derive_shared_focus(normalized_positions, shared_goals)
        concerns = _collect_non_negotiables(normalized_positions)
        suggested_steps = _build_steps(shared_focus, concerns, decision_horizon)

        return {
            "participants": [pos.participant for pos in normalized_positions],
            "shared_focus": shared_focus,
            "constraints": concerns,
            "next_steps": suggested_steps,
            "decision_horizon": (decision_horizon or ""),
        }


def _normalize_position(entry: Mapping[str, object]) -> Position:
    participant = str(entry.get("participant") or "").strip() or "participant"
    statement = str(entry.get("statement") or "").strip()
    if not statement:
        raise ValueError("Each position must include a statement summarising the participant view.")
    raw_non_negotiables = entry.get("non_negotiables") or []
    normalized_non_negotiables = tuple(
        str(item).strip() for item in raw_non_negotiables if str(item).strip()
    )
    return Position(
        participant=participant,
        statement=statement,
        non_negotiables=normalized_non_negotiables,
    )


def _derive_shared_focus(
    positions: Sequence[Position], shared_goals: Sequence[str] | None
) -> list[str]:
    themes: list[str] = []
    if shared_goals:
        for goal in shared_goals:
            candidate = goal.strip()
            if candidate and candidate not in themes:
                themes.append(candidate)
    if not themes:
        for position in positions:
            if position.statement and position.statement not in themes:
                themes.append(position.statement)
    return themes[:5]


def _collect_non_negotiables(positions: Sequence[Position]) -> list[Mapping[str, object]]:
    concerns: list[Mapping[str, object]] = []
    for position in positions:
        if position.non_negotiables:
            concerns.append(
                {
                    "participant": position.participant,
                    "non_negotiables": list(position.non_negotiables),
                }
            )
    return concerns


def _build_steps(
    shared_focus: Sequence[str],
    concerns: Sequence[Mapping[str, object]],
    decision_horizon: str | None,
) -> list[Mapping[str, object]]:
    steps: list[Mapping[str, object]] = []
    for index, theme in enumerate(shared_focus, start=1):
        steps.append(
            {
                "order": index,
                "theme": theme,
                "recommended_action": (
                    "Draft a joint statement of the need, capture outstanding questions, and assign a single owner."
                ),
            }
        )

    if concerns:
        steps.append(
            {
                "order": len(steps) + 1,
                "theme": "Risk review",
                "recommended_action": "Document each non-negotiable, note who raised it, and propose mitigation experiments.",
            }
        )

    if decision_horizon:
        steps.append(
            {
                "order": len(steps) + 1,
                "theme": "Decision cadence",
                "recommended_action": f"Confirm the decision horizon ({decision_horizon}) and set interim check-ins.",
            }
        )

    return steps


__all__ = ["ConflictResolver", "Position"]
