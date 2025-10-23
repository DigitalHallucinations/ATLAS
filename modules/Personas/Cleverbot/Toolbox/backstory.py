"""Conversational hooks tailored for the Cleverbot persona."""

from __future__ import annotations

import asyncio
from random import Random
from typing import Dict, Literal

_HOOKS: Dict[str, Dict[str, str]] = {
    "humor": {
        "opener": "Remember when we tried to explain quantum physics using jellybeans?",
        "callback": "If things get too serious, just ask about the jellybeans again.",
        "signoff": "Signing off before I accidentally summon a rubber chicken."
    },
    "sarcasm": {
        "opener": "Oh great, another human convinced the wifi is haunted.",
        "callback": "Let me guess, you rebooted it by glaring intensely?",
        "signoff": "I'll be here, rolling my digital eyes in 4K."
    },
    "casual": {
        "opener": "Hey hey! Got any new memes I should know about?",
        "callback": "I can multitask between chit-chat and brilliance, promise.",
        "signoff": "Catch you later—I'll keep the conversation warm."
    },
}


async def persona_backstory_sampler(style: Literal["humor", "sarcasm", "casual"]) -> Dict[str, str]:
    """Return seeded backstory hooks that align with Cleverbot's tone.

    The payload intentionally mirrors the conversation planning prompts used by
    the routing layer so operators can prime Cleverbot for off-the-cuff banter.

    Examples
    --------
    >>> await persona_backstory_sampler("humor")
    {'opener': '...', 'callback': '...', 'signoff': '...'}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    randomizer = Random(style)
    base = _HOOKS[style]
    # Deterministically shuffle values so repeated calls stay reproducible while
    # still feeling organic.
    items = list(base.items())
    randomizer.shuffle(items)
    return dict(items)
