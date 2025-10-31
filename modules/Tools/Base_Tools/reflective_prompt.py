"""Generate reflective prompts that nudge participants toward alignment."""

from __future__ import annotations

import asyncio
from typing import Mapping, Sequence


class ReflectivePrompt:
    """Compose facilitator prompts tailored to the observed tension."""

    async def run(
        self,
        *,
        topic: str,
        tension_points: Sequence[str] | None = None,
        tone_observation: str | None = None,
        next_review: str | None = None,
    ) -> Mapping[str, object]:
        topic_normalized = (topic or "").strip()
        if not topic_normalized:
            raise ValueError("reflective_prompt requires a topic to anchor the prompts.")

        await asyncio.sleep(0)

        prompts: list[str] = []
        prompts.append(
            f"What feels most important about {topic_normalized} right now, and what outcome would feel like progress?"
        )

        if tension_points:
            for tension in tension_points:
                candidate = tension.strip()
                if not candidate:
                    continue
                prompts.append(
                    f"When you reflect on '{candidate}', what need or constraint sits underneath it?"
                )

        if tone_observation:
            observation = tone_observation.strip()
            if observation:
                prompts.append(
                    f"I'm sensing {observation}. What would help the group feel heard before we move forward?"
                )

        prompts.append(
            "What assumptions might we be holding that, if revisited, could unlock a new option?"
        )

        if next_review:
            review_window = next_review.strip()
            if review_window:
                prompts.append(
                    f"Before we break, what should we commit to revisit by {review_window}?"
                )

        unique_prompts = []
        seen = set()
        for prompt in prompts:
            if prompt not in seen:
                unique_prompts.append(prompt)
                seen.add(prompt)

        return {
            "topic": topic_normalized,
            "prompts": unique_prompts,
            "count": len(unique_prompts),
        }


__all__ = ["ReflectivePrompt"]
