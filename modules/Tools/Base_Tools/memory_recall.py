"""Curate memory cues that keep mediation conversations grounded."""

from __future__ import annotations

import asyncio
from typing import Mapping, Sequence


class MemoryRecall:
    """Summarise prior commitments and outstanding questions."""

    async def run(
        self,
        *,
        highlights: Sequence[str],
        timeframe: str | None = None,
        commitments: Sequence[str] | None = None,
    ) -> Mapping[str, object]:
        normalized_highlights = [
            str(item).strip() for item in highlights or [] if isinstance(item, str) and str(item).strip()
        ]
        if not normalized_highlights:
            raise ValueError("MemoryRecall requires at least one highlight to summarise.")

        await asyncio.sleep(0)

        summary = ", ".join(normalized_highlights)
        timeline = []
        if timeframe:
            timeline.append({"label": timeframe.strip(), "highlights": normalized_highlights})
        else:
            timeline.append({"label": "recent", "highlights": normalized_highlights})

        commitment_list = [
            str(item).strip() for item in commitments or [] if isinstance(item, str) and str(item).strip()
        ]

        open_questions = []
        if not commitment_list:
            open_questions.append("What agreements are still waiting for confirmation?")
        else:
            open_questions.append("Have the recorded commitments been honoured or do we need new check-ins?")

        next_focus = "Which highlight should guide the next agenda step?"
        if commitment_list:
            next_focus = "Confirm the ownership and due dates for each commitment before moving forward."

        return {
            "summary": summary,
            "timeline": timeline,
            "commitments": commitment_list,
            "open_questions": open_questions,
            "next_focus": next_focus,
        }


__all__ = ["MemoryRecall"]
