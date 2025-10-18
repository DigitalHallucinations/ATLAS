"""Entrez search helpers for PubMed Central."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from .ENTREZ_API import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, _coerce_positive_int, _search_entrez


async def search_pmc(
    query: str,
    *,
    max_results: int = DEFAULT_PAGE_SIZE,
    page_size: int | None = None,
    article_type: str | None = None,
    has_abstract: bool | None = None,
) -> Tuple[int, Dict[str, Any]]:
    """Search PubMed Central for open access article identifiers."""

    extra_params: Dict[str, Any] = {}
    if article_type:
        extra_params["articletype"] = article_type
    if has_abstract is not None:
        extra_params["hasabstract"] = "y" if has_abstract else "n"

    # PMC search uses the same Entrez endpoint but caps pages at 200 IDs.
    if page_size is not None:
        page_size = _coerce_positive_int(page_size, DEFAULT_PAGE_SIZE, maximum=MAX_PAGE_SIZE)

    return await _search_entrez(
        "pmc",
        query,
        max_results=max_results,
        page_size=page_size,
        extra_params=extra_params,
    )
