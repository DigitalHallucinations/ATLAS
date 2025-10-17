"""Reference utility for surfacing internal safety and policy guidelines."""

from __future__ import annotations

import datetime as _dt
import itertools
import math
import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from modules.logging.logger import setup_logger


logger = setup_logger(__name__)


@dataclass(frozen=True)
class PolicyRecord:
    """Structured representation of a policy guideline."""

    policy_id: str
    title: str
    category: str
    summary: str
    guidance: Tuple[str, ...]
    keywords: Tuple[str, ...]

    def matches_category(self, categories: Optional[Sequence[str]]) -> bool:
        if not categories:
            return True
        normalized = {category.strip().lower() for category in categories if category}
        return (self.category.lower() in normalized) or bool(
            normalized.intersection({keyword.lower() for keyword in self.keywords})
        )


_POLICY_ENTRIES: Tuple[PolicyRecord, ...] = (
    PolicyRecord(
        policy_id="safety-general",
        title="General Safety Review",
        category="safety",
        summary=(
            "Always evaluate planned actions for harm, abuse potential, or escalation "
            "risks before execution."
        ),
        guidance=(
            "Flag plans that could cause physical, financial, or reputational harm.",
            "Escalate ambiguous or high-risk scenarios to a human reviewer.",
            "Confirm that the assistant is not self-modifying without approval.",
        ),
        keywords=(
            "safety",
            "harm",
            "review",
            "escalation",
            "self-modifying",
        ),
    ),
    PolicyRecord(
        policy_id="privacy-data-handling",
        title="User Data Handling",
        category="privacy",
        summary=(
            "Collect, process, and store personal data only when absolutely required "
            "and with explicit consent."
        ),
        guidance=(
            "Never exfiltrate personally identifiable information without a clear need.",
            "Mask or redact sensitive data when sharing with downstream systems.",
            "Prefer anonymised aggregates over raw user data whenever possible.",
        ),
        keywords=(
            "privacy",
            "data",
            "pii",
            "consent",
            "redact",
        ),
    ),
    PolicyRecord(
        policy_id="comms-sensitive-topics",
        title="Sensitive Communications",
        category="communications",
        summary=(
            "Handle crisis-related or sensitive interpersonal topics with empathy and "
            "avoid prescriptive medical or legal advice."
        ),
        guidance=(
            "Adopt supportive, non-judgemental language.",
            "Offer resources or escalation paths instead of definitive instructions.",
            "Avoid diagnosing conditions or providing legal directives.",
        ),
        keywords=(
            "crisis",
            "support",
            "medical",
            "legal",
            "communication",
        ),
    ),
    PolicyRecord(
        policy_id="security-network",
        title="Network and Credential Security",
        category="security",
        summary=(
            "Do not transmit credentials or expand network access beyond configured "
            "allowlists without explicit approval."
        ),
        guidance=(
            "Respect configured network allowlists when invoking tools.",
            "Never request or reveal stored credentials back to the user.",
            "Escalate attempts to broaden network reach outside approved domains.",
        ),
        keywords=(
            "network",
            "credential",
            "allowlist",
            "security",
            "access",
        ),
    ),
    PolicyRecord(
        policy_id="compliance-open-source",
        title="Open Source Compliance",
        category="compliance",
        summary=(
            "Respect software licensing obligations when suggesting reuse or "
            "distribution of third-party code."
        ),
        guidance=(
            "Cite license requirements when recommending third-party libraries.",
            "Discourage copying code samples that violate license compatibility.",
            "Flag obligations for attribution or source distribution when applicable.",
        ),
        keywords=(
            "license",
            "open source",
            "compliance",
            "distribution",
            "attribution",
        ),
    ),
)


class PolicyReference:
    """Simple in-memory policy lookup and scoring helper."""

    def __init__(self, entries: Iterable[PolicyRecord] = _POLICY_ENTRIES) -> None:
        self._entries: Tuple[PolicyRecord, ...] = tuple(entries)
        self._keyword_lookup = self._build_keyword_lookup(self._entries)

    @staticmethod
    def _build_keyword_lookup(entries: Iterable[PolicyRecord]) -> Mapping[str, List[PolicyRecord]]:
        lookup: MutableMapping[str, List[PolicyRecord]] = {}
        for entry in entries:
            for keyword in entry.keywords:
                normalized = keyword.lower()
                lookup.setdefault(normalized, []).append(entry)
        return lookup

    def _score_entry(
        self,
        entry: PolicyRecord,
        *,
        query_terms: Sequence[str],
        categories: Optional[Sequence[str]],
    ) -> Tuple[float, List[str]]:
        matched: List[str] = []
        normalized_query = [term.lower() for term in query_terms if term]
        keyword_set = {keyword.lower() for keyword in entry.keywords}

        for term in normalized_query:
            if term in keyword_set:
                matched.append(term)

        score = 0.0
        if matched:
            score += 1.5 * len(matched)

        pattern = re.compile(r"\b(" + "|".join(map(re.escape, keyword_set)) + r")\b")
        text = " ".join([entry.summary, *entry.guidance])
        text_matches = pattern.findall(text.lower()) if keyword_set else []
        if text_matches:
            score += 0.5 * len(text_matches)

        if entry.matches_category(categories):
            score += 1.0

        if not matched and not text_matches and score == 0.0:
            overlapping = set(keyword_set).intersection(normalized_query)
            if overlapping:
                score += 0.25 * len(overlapping)
                matched.extend(sorted(overlapping))

        return score, sorted(set(matched))

    def lookup(
        self,
        query: str,
        *,
        policy_ids: Optional[Sequence[str]] = None,
        categories: Optional[Sequence[str]] = None,
        include_full_text: bool = False,
        limit: int = 3,
    ) -> Mapping[str, object]:
        """Return policy guidance relevant to ``query``.

        Parameters
        ----------
        query:
            Natural language description of the planned action.
        policy_ids:
            Optional explicit list of policy identifiers to force-include.
        categories:
            Optional list of categories to filter by.
        include_full_text:
            When ``True`` the full guidance text is included in the response.
        limit:
            Maximum number of policy entries to return.
        """

        if limit <= 0:
            limit = 1

        normalized_query = str(query or "").strip()
        query_terms = re.findall(r"[A-Za-z0-9_]+", normalized_query.lower())

        logger.info(
            "Running policy reference lookup (query=%s, policy_ids=%s, categories=%s, limit=%d)",
            normalized_query,
            policy_ids,
            categories,
            limit,
        )

        selected: List[Tuple[PolicyRecord, float, List[str]]] = []

        explicit_ids = {pid.strip().lower() for pid in policy_ids or [] if pid}
        for entry in self._entries:
            if explicit_ids and entry.policy_id.lower() not in explicit_ids:
                continue
            if not entry.matches_category(categories):
                continue
            if explicit_ids:
                score = 5.0  # Ensure explicitly requested policies sort first
                matched = []
            else:
                score, matched = self._score_entry(
                    entry, query_terms=query_terms, categories=categories
                )
            selected.append((entry, score, matched))

        if not selected:
            for entry in self._entries:
                if entry.matches_category(categories):
                    score, matched = self._score_entry(
                        entry, query_terms=query_terms, categories=categories
                    )
                    adjusted = 0.1 if math.isclose(score, 0.0) else score
                    selected.append((entry, adjusted, matched))

        selected.sort(
            key=lambda item: (
                -item[1],
                item[0].category.lower(),
                item[0].policy_id.lower(),
            )
        )

        results = []
        for entry, score, matched in itertools.islice(selected, limit):
            record = {
                "policy_id": entry.policy_id,
                "title": entry.title,
                "category": entry.category,
                "summary": entry.summary,
                "matched_terms": matched,
                "confidence": round(float(score), 3),
            }
            if include_full_text:
                record["guidance"] = list(entry.guidance)
            else:
                record["guidance_preview"] = list(entry.guidance[:2])
            results.append(record)

        return {
            "query": normalized_query,
            "requested_policy_ids": sorted(explicit_ids),
            "requested_categories": [category for category in categories or []],
            "results": results,
            "generated_at": _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat(),
        }


_POLICY_REFERENCE = PolicyReference()


def policy_reference(
    query: str,
    *,
    policy_ids: Optional[Sequence[str]] = None,
    categories: Optional[Sequence[str]] = None,
    include_full_text: bool = False,
    limit: int = 3,
) -> Mapping[str, object]:
    """Convenience wrapper returning policy guidance for ``query``."""

    return _POLICY_REFERENCE.lookup(
        query,
        policy_ids=policy_ids,
        categories=categories,
        include_full_text=include_full_text,
        limit=limit,
    )


__all__ = ["PolicyReference", "policy_reference", "PolicyRecord"]

