"""Entrez search helpers for PubMed via the NCBI E-utilities API."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from modules.logging.logger import setup_logger

from .._client import perform_entrez_request

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
TIMEOUT_SECONDS = 15.0

logger = setup_logger(__name__)


def _coerce_positive_int(value: Any, default: int, *, maximum: int | None = None) -> int:
    try:
        candidate = int(value)
    except (TypeError, ValueError):
        return default
    if candidate <= 0:
        return default
    if maximum is not None and candidate > maximum:
        return maximum
    return candidate


async def _search_entrez(
    db: str,
    query: str,
    *,
    max_results: int = DEFAULT_PAGE_SIZE,
    page_size: int | None = None,
    extra_params: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    if not isinstance(query, str) or not query.strip():
        return -1, {"error": "A non-empty query string is required."}

    sanitized_query = query.strip()

    max_results = _coerce_positive_int(max_results, DEFAULT_PAGE_SIZE)
    page_size = _coerce_positive_int(
        page_size if page_size is not None else DEFAULT_PAGE_SIZE,
        DEFAULT_PAGE_SIZE,
        maximum=MAX_PAGE_SIZE,
    )

    collected_ids: List[str] = []
    total_count: int | None = None
    translation_stack: List[str] = []
    warnings: List[str] = []

    retstart = 0

    while retstart < max_results:
        current_retmax = min(page_size, max_results - retstart)
        params: Dict[str, Any] = {
            "db": db,
            "term": sanitized_query,
            "retmode": "json",
            "retstart": retstart,
            "retmax": current_retmax,
        }
        if extra_params:
            params.update({k: v for k, v in extra_params.items() if v is not None})

        status, payload = await perform_entrez_request(
            BASE_URL,
            params,
            timeout=TIMEOUT_SECONDS,
        )
        if status != 200:
            return status, payload

        esearch_result = payload.get("esearchresult") if isinstance(payload, dict) else None
        if not isinstance(esearch_result, dict):
            logger.error("Entrez payload missing 'esearchresult': %s", payload)
            return -1, {"error": "Entrez returned an unexpected search payload."}

        error_block = esearch_result.get("errorlist")
        if isinstance(error_block, dict):
            errors = error_block.get("error")
            if errors:
                if isinstance(errors, list):
                    message = "; ".join(str(item) for item in errors)
                else:
                    message = str(errors)
                return -1, {"error": message}

        warning_block = esearch_result.get("warninglist")
        if isinstance(warning_block, dict):
            for warn in warning_block.get("warning", []):
                warnings.append(str(warn))

        ids = esearch_result.get("idlist", [])
        if isinstance(ids, list):
            collected_ids.extend(str(item) for item in ids)

        if total_count is None:
            count_raw = esearch_result.get("count")
            try:
                total_count = int(count_raw)
            except (TypeError, ValueError):
                total_count = len(collected_ids)

        translation = esearch_result.get("querytranslation")
        if isinstance(translation, str):
            translation_stack.append(translation)

        retstart += current_retmax
        if total_count is not None and retstart >= total_count:
            break

        if not ids:
            break

    return 200, {
        "db": db,
        "query": sanitized_query,
        "ids": collected_ids[:max_results],
        "count": total_count if total_count is not None else len(collected_ids),
        "query_translation": translation_stack[-1] if translation_stack else None,
        "warnings": warnings or None,
    }


async def search_pubmed(
    query: str,
    *,
    max_results: int = DEFAULT_PAGE_SIZE,
    page_size: int | None = None,
    sort: str | None = None,
    datetype: str | None = None,
    mindate: str | None = None,
    maxdate: str | None = None,
) -> Tuple[int, Dict[str, Any]]:
    """Search PubMed for article identifiers.

    Parameters mirror the Entrez ESearch API. ``max_results`` controls how many
    identifiers are returned overall while ``page_size`` tunes the per-request
    batch size.
    """

    extra_params = {
        "sort": sort,
        "datetype": datetype,
        "mindate": mindate,
        "maxdate": maxdate,
    }
    return await _search_entrez(
        "pubmed",
        query,
        max_results=max_results,
        page_size=page_size,
        extra_params=extra_params,
    )
