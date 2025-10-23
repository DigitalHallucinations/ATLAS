"""Problem solving scaffolds for MathTutor."""

from __future__ import annotations

import asyncio
from typing import Dict, Literal, Optional

try:  # pragma: no cover - optional dependency
    import sympy as sp  # type: ignore
except Exception:  # pragma: no cover - gracefully degrade when sympy missing
    sp = None  # type: ignore


async def stepwise_solution(
    problem: str,
    focus: Literal["algebra", "geometry", "calculus"],
) -> Dict[str, object]:
    """Produce numbered reasoning steps and lightweight verification when available.

    Examples
    --------
    >>> await stepwise_solution("2*x + 3 = 7", "algebra")
    {'steps': [...], 'verification': {...}}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    steps = [
        {
            "number": 1,
            "action": "Clarify what the problem is asking and identify knowns/unknowns.",
        },
        {
            "number": 2,
            "action": f"Apply {focus} techniques to manipulate the expressions systematically.",
        },
        {
            "number": 3,
            "action": "Check the solution against constraints and interpret the result for the learner.",
        },
    ]

    verification: Optional[Dict[str, object]] = None

    if sp is not None and focus == "algebra" and "=" in problem:
        try:
            symbol = sp.symbols("x")
            left, right = problem.split("=", 1)
            solution = sp.solve(sp.Eq(sp.sympify(left), sp.sympify(right)), symbol)
            verification = {
                "tool": "sympy",
                "symbol": str(symbol),
                "solution": [sp.simplify(item) for item in solution],
                "check": [
                    sp.simplify(sp.sympify(left) - sp.sympify(right)).subs(symbol, item)
                    for item in solution
                ],
            }
        except Exception:  # pragma: no cover - maintain resilience for complex inputs
            verification = {
                "tool": "sympy",
                "note": "Failed to auto-solve; provide manual reasoning in narrative.",
            }

    return {
        "problem": problem,
        "focus": focus,
        "steps": steps,
        "verification": verification,
        "next_prompts": [
            "Ask the learner to restate the solution in their own words.",
            "Suggest a variation problem to reinforce the concept.",
        ],
    }
