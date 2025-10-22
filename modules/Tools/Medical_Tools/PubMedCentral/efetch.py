"""Entrez EFetch helpers for retrieving detailed PubMed records."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Sequence, Tuple

from modules.logging.logger import setup_logger

from .._client import perform_entrez_request

BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
DEFAULT_CHUNK_SIZE = 50
MAX_CHUNK_SIZE = 200
TIMEOUT_SECONDS = 20.0

logger = setup_logger(__name__)


def _normalize_ids(ids: Sequence[str] | str | Iterable[str]) -> List[str]:
    """Normalize a collection of PubMed identifiers into a clean list."""

    if isinstance(ids, str):
        # Support comma or whitespace separated identifiers.
        raw_tokens = ids.replace(";", " ").replace(",", " ").split()
    else:
        raw_tokens = list(ids)

    normalized: List[str] = []
    seen = set()
    for token in raw_tokens:
        candidate = str(token).strip()
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)

    return normalized


def _coerce_chunk_size(value: Any) -> int:
    try:
        chunk = int(value)
    except (TypeError, ValueError):
        return DEFAULT_CHUNK_SIZE

    if chunk <= 0:
        return DEFAULT_CHUNK_SIZE

    return min(chunk, MAX_CHUNK_SIZE)


async def fetch_pubmed_details(
    pubmed_ids: Sequence[str] | str | Iterable[str],
    *,
    rettype: str = "abstract",
    retmode: str = "json",
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    extra_params: Dict[str, Any] | None = None,
) -> Tuple[int, Dict[str, Any]]:
    """Return detailed PubMed records for one or more identifiers.

    The Entrez EFetch API only accepts a limited number of identifiers per
    request.  This helper transparently batches calls and merges the results
    into a single payload.  The response contains both the raw batched
    payloads (``batches``) and a flattened ``records`` list with each entry
    tagged by its PubMed identifier.
    """

    normalized_ids = _normalize_ids(pubmed_ids)
    if not normalized_ids:
        return -1, {"error": "At least one PubMed identifier is required."}

    chunk_size = _coerce_chunk_size(chunk_size)

    batches: List[Dict[str, Any]] = []
    records: List[Dict[str, Any]] = []

    for start in range(0, len(normalized_ids), chunk_size):
        current_ids = normalized_ids[start : start + chunk_size]
        params: Dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(current_ids),
            "retmode": retmode,
            "rettype": rettype,
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

        batches.append(payload)

        if isinstance(payload, dict):
            result_block = payload.get("result")
            if isinstance(result_block, dict):
                for identifier in current_ids:
                    entry = result_block.get(identifier)
                    if isinstance(entry, dict):
                        record = dict(entry)
                        record.setdefault("pmid", identifier)
                        records.append(record)
                        continue
                    logger.debug(
                        "Entrez EFetch payload missing structured entry for %s", identifier
                    )
                    records.append({"pmid": identifier, "payload": payload})
                continue

        # Fallback for unstructured payloads
        for identifier in current_ids:
            records.append({"pmid": identifier, "payload": payload})

    return 200, {
        "db": "pubmed",
        "ids": normalized_ids,
        "records": records,
        "batches": batches,
    }


__all__ = ["fetch_pubmed_details"]
