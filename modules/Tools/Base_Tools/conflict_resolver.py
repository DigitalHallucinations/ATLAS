"""Structure incremental mediation plans for conflicting positions."""

from __future__ import annotations

import asyncio
from typing import Mapping, Sequence


class ConflictResolver:
    """Guide parties through shared goal alignment and next steps."""

    async def run(
        self,
        *,
        tensions: Sequence[str],
        shared_goals: Sequence[str] | None = None,
        constraints: Sequence[str] | None = None,
        timeframe: str | None = None,
    ) -> Mapping[str, object]:
        normalized_tensions = [
            str(item).strip() for item in tensions or [] if isinstance(item, str) and str(item).strip()
        ]
        if not normalized_tensions:
            raise ValueError("ConflictResolver needs at least one tension to address.")

        await asyncio.sleep(0)

        normalized_goals = [
            str(item).strip() for item in shared_goals or [] if isinstance(item, str) and str(item).strip()
        ]
        normalized_constraints = [
            str(item).strip() for item in constraints or [] if isinstance(item, str) and str(item).strip()
        ]

        phases = [
            "Establish psychological safety and allow each side to share without interruption.",
            "Reflect back what you heard, confirming emotions and unmet needs.",
            "Identify intersections between goals and co-create small experiments.",
        ]
        if timeframe:
            phases.append(f"Agree on a check-in cadence within {timeframe.strip()} to review progress.")

        ground_rules = [
            "One speaker at a time with neutral summarisation between turns.",
            "Name impacts using first-person language and avoid attributions of intent.",
            "Pause the session if voices escalate above a calm tone.",
        ]

        if normalized_constraints:
            ground_rules.append(
                "Document the constraints explicitly so adjustments can be scoped together."
            )

        escalation_watchpoints = [
            "Repeating the same grievance without acknowledgement.",
            "Dismissive or minimising language toward personal experiences.",
        ]
        if normalized_tensions:
            escalation_watchpoints.append(
                f"Any reference to {normalized_tensions[0]} should prompt a reset and reframing."
            )

        plan_steps = []
        for index, tension in enumerate(normalized_tensions, start=1):
            step = {
                "tension": tension,
                "step": f"Invite each party to describe how {tension.lower()} impacts them and what a respectful shift looks like.",
            }
            if normalized_goals:
                step["bridge_goal"] = normalized_goals[0]
            if normalized_constraints:
                step["constraint_check"] = normalized_constraints[0]
            plan_steps.append(step)

        shared_language = "Focus on needs and shared aspirations rather than positions."
        if normalized_goals:
            shared_language = f"Anchor the dialogue on the mutual goal of {normalized_goals[0].lower()}."

        return {
            "phases": phases,
            "ground_rules": ground_rules,
            "escalation_watchpoints": escalation_watchpoints,
            "plan_steps": plan_steps,
            "shared_language": shared_language,
        }


__all__ = ["ConflictResolver"]
