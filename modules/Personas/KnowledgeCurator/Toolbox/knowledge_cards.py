"""Knowledge card synthesis helpers for the KnowledgeCurator persona."""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional


async def knowledge_card_builder(
    query: str,
    sources: List[str],
    review_cadence: Optional[str] = None,
) -> Dict[str, object]:
    """Assemble a review-ready knowledge card payload.

    Examples
    --------
    >>> await knowledge_card_builder('LLM eval best practices', ['ARC benchmarking'])
    {'question': 'LLM eval best practices', ...}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    card = {
        "question": query,
        "sources": sources,
        "findings": [
            "Summarize the primary insight gleaned from the listed sources.",
            "Highlight conflicting viewpoints or open research threads.",
        ],
        "citations": [
            {
                "source": source,
                "note": "Capture author, publication date, and relevance tags.",
            }
            for source in sources
        ],
        "follow_ups": [
            "List new datasets or experts to consult during the next review cycle.",
        ],
        "review_cadence": review_cadence or "quarterly",
    }
    return card
