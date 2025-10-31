"""Generate reflective listening scaffolds for mediators."""

from __future__ import annotations

import asyncio
from typing import Mapping, Sequence


class ReflectivePrompt:
    """Compose validation statements and clarifying questions."""

    async def run(
        self,
        *,
        statement: str,
        emotions: Sequence[str] | None = None,
        needs: Sequence[str] | None = None,
        action_hint: str | None = None,
    ) -> Mapping[str, object]:
        if not isinstance(statement, str) or not statement.strip():
            raise ValueError("ReflectivePrompt requires a statement to mirror.")

        await asyncio.sleep(0)

        clean_statement = " ".join(statement.split())
        emotion_list = [emotion.strip().lower() for emotion in emotions or [] if emotion and str(emotion).strip()]
        need_list = [need.strip().lower() for need in needs or [] if need and str(need).strip()]

        reflection_parts = ["It sounds like you're highlighting that", clean_statement]
        if emotion_list:
            reflection_parts.append(f"and feeling {', '.join(emotion_list)} about it")
        reflection = " ".join(part for part in reflection_parts if part).strip()

        validations = []
        if emotion_list:
            validations.append(
                f"Acknowledging the emotional weight: {' and '.join(emotion_list)}."
            )
        else:
            validations.append("Validating the experience even if the emotions are still forming.")

        if need_list:
            validations.append(f"Noting the needs you named: {', '.join(need_list)}.")

        follow_up_questions = [
            "What feels most important to resolve first?",
            "What support would help you feel heard right now?",
        ]
        if need_list:
            follow_up_questions.append(
                f"How can we honour the need for {need_list[0]} while moving forward?"
            )

        next_step = action_hint.strip() if isinstance(action_hint, str) and action_hint.strip() else "Suggest taking a pause to summarise what each side is asking for before proposing solutions."

        paraphrase = f"You are emphasising that {clean_statement.lower()}"
        if need_list:
            paraphrase += f" because you need {need_list[0]}"
        paraphrase += ", and you want that recognised before we decide on next steps."

        return {
            "reflection": reflection,
            "validations": validations,
            "follow_up_questions": follow_up_questions,
            "paraphrase": paraphrase,
            "next_step": next_step,
        }


__all__ = ["ReflectivePrompt"]
