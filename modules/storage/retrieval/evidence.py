"""Evidence gating and citation verification.

This module provides tools for verifying LLM responses against source evidence:

1. CitationExtractor: Parses [1], [2] style citations from responses
2. FaithfulnessScorer: Verifies claims against source chunks using NLI
3. EvidenceGate: Combines extraction and verification for response validation

Usage:
    >>> from modules.storage.retrieval.evidence import (
    ...     CitationExtractor,
    ...     FaithfulnessScorer,
    ...     EvidenceGate,
    ... )
    >>> gate = EvidenceGate()
    >>> await gate.initialize()
    >>> result = await gate.verify_response(response_text, source_chunks)
    >>> print(result.overall_score)  # 0.85
    >>> print(result.unsupported_claims)  # ["Claim not in sources"]
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from modules.logging.logger import setup_logger

if TYPE_CHECKING:
    from modules.storage.knowledge import SearchResult, KnowledgeChunk

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class Citation:
    """A citation reference in the response text.

    Attributes:
        index: Citation number (1-based).
        text: Text span containing the citation.
        start: Start position in response.
        end: End position in response.
        claim: The claim being cited (sentence or phrase).
    """

    index: int
    text: str
    start: int
    end: int
    claim: str = ""


@dataclass
class ClaimVerification:
    """Verification result for a single claim.

    Attributes:
        claim: The claim text.
        citations: Citation indices referenced.
        source_chunks: Matched source chunk IDs.
        entailment_score: NLI entailment probability.
        is_supported: Whether the claim is supported.
        confidence: Confidence in the verification.
    """

    claim: str
    citations: List[int] = field(default_factory=list)
    source_chunks: List[str] = field(default_factory=list)
    entailment_score: float = 0.0
    is_supported: bool = False
    confidence: float = 0.0


@dataclass
class VerificationResult:
    """Full verification result for a response.

    Attributes:
        response: Original response text.
        claims: List of claim verifications.
        overall_score: Aggregate faithfulness score (0-1).
        supported_claims: Claims with evidence support.
        unsupported_claims: Claims without evidence.
        citations_found: Total citations found.
        citations_valid: Citations that match sources.
    """

    response: str
    claims: List[ClaimVerification] = field(default_factory=list)
    overall_score: float = 0.0
    supported_claims: List[str] = field(default_factory=list)
    unsupported_claims: List[str] = field(default_factory=list)
    citations_found: int = 0
    citations_valid: int = 0


class SupportLevel(str, Enum):
    """Level of source support for a claim."""

    STRONG = "strong"  # entailment > 0.8
    MODERATE = "moderate"  # entailment 0.5-0.8
    WEAK = "weak"  # entailment 0.2-0.5
    NONE = "none"  # entailment < 0.2


# -----------------------------------------------------------------------------
# Citation Extractor
# -----------------------------------------------------------------------------


class CitationExtractor:
    """Extract citations from LLM responses.

    Supports multiple citation formats:
    - Bracketed numbers: [1], [2], [1, 2, 3]
    - Superscript style: ^1, ^2 (rendered as superscripts)
    - Parenthetical: (1), (2)
    """

    # Citation patterns
    BRACKET_PATTERN = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
    PAREN_PATTERN = re.compile(r"\((\d+(?:\s*,\s*\d+)*)\)")
    SUPERSCRIPT_PATTERN = re.compile(r"\^(\d+)")

    def __init__(
        self,
        *,
        enable_bracket: bool = True,
        enable_paren: bool = False,
        enable_superscript: bool = False,
    ) -> None:
        """Initialize citation extractor.

        Args:
            enable_bracket: Enable [n] style citations.
            enable_paren: Enable (n) style citations.
            enable_superscript: Enable ^n style citations.
        """
        self._patterns: List[Tuple[re.Pattern, str]] = []
        if enable_bracket:
            self._patterns.append((self.BRACKET_PATTERN, "bracket"))
        if enable_paren:
            self._patterns.append((self.PAREN_PATTERN, "paren"))
        if enable_superscript:
            self._patterns.append((self.SUPERSCRIPT_PATTERN, "superscript"))

    def extract(self, text: str) -> List[Citation]:
        """Extract all citations from text.

        Args:
            text: Response text to parse.

        Returns:
            List of Citation objects.
        """
        citations: List[Citation] = []
        seen_positions: set = set()

        for pattern, style in self._patterns:
            for match in pattern.finditer(text):
                # Avoid duplicates at same position
                if match.start() in seen_positions:
                    continue
                seen_positions.add(match.start())

                # Parse citation indices
                indices_str = match.group(1)
                indices = [int(i.strip()) for i in indices_str.split(",")]

                # Get surrounding context (sentence containing citation)
                claim = self._extract_sentence(text, match.start(), match.end())

                for idx in indices:
                    citations.append(
                        Citation(
                            index=idx,
                            text=match.group(0),
                            start=match.start(),
                            end=match.end(),
                            claim=claim,
                        )
                    )

        # Sort by position
        citations.sort(key=lambda c: c.start)
        return citations

    def _extract_sentence(self, text: str, cite_start: int, cite_end: int) -> str:
        """Extract the sentence containing a citation.

        Args:
            text: Full text.
            cite_start: Citation start position.
            cite_end: Citation end position.

        Returns:
            Sentence containing the citation.
        """
        # Find sentence boundaries
        sentence_ends = re.compile(r"[.!?]\s+")

        # Find start of sentence
        start = 0
        for match in sentence_ends.finditer(text):
            if match.end() <= cite_start:
                start = match.end()
            else:
                break

        # Find end of sentence
        end = len(text)
        for match in sentence_ends.finditer(text):
            if match.start() >= cite_end:
                end = match.end()
                break

        return text[start:end].strip()

    def get_unique_indices(self, citations: List[Citation]) -> List[int]:
        """Get unique citation indices in order.

        Args:
            citations: List of citations.

        Returns:
            Unique indices sorted.
        """
        return sorted(set(c.index for c in citations))

    def remove_citations(self, text: str) -> str:
        """Remove all citation markers from text.

        Args:
            text: Text with citations.

        Returns:
            Text without citation markers.
        """
        result = text
        for pattern, _ in self._patterns:
            result = pattern.sub("", result)
        return result


# -----------------------------------------------------------------------------
# Faithfulness Scorer
# -----------------------------------------------------------------------------


class FaithfulnessScorer:
    """Score claim faithfulness using Natural Language Inference (NLI).

    Uses an NLI model to determine if claims are entailed by source evidence.
    """

    def __init__(
        self,
        model: str = "microsoft/deberta-base-mnli",
        device: Optional[str] = None,
        entailment_threshold: float = 0.5,
    ) -> None:
        """Initialize faithfulness scorer.

        Args:
            model: HuggingFace NLI model.
            device: Device to use (cuda, cpu, mps).
            entailment_threshold: Minimum entailment score for support.
        """
        self._model_name = model
        self._device = device
        self._entailment_threshold = entailment_threshold
        self._classifier: Optional[Any] = None
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if scorer is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Load the NLI model."""
        if self._initialized:
            return

        def _load() -> Any:
            try:
                from transformers import pipeline
            except ImportError:
                raise RuntimeError(
                    "FaithfulnessScorer requires transformers. "
                    "Install with: pip install transformers"
                )

            device = self._device
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = -1
                except ImportError:
                    device = -1

            return pipeline(
                "text-classification",
                model=self._model_name,
                device=device,
            )

        self._classifier = await asyncio.to_thread(_load)
        self._initialized = True
        logger.info(f"FaithfulnessScorer initialized with {self._model_name}")

    async def shutdown(self) -> None:
        """Cleanup model resources."""
        self._classifier = None
        self._initialized = False

    async def score_claim(
        self,
        claim: str,
        evidence: str,
    ) -> Tuple[float, SupportLevel]:
        """Score a single claim against evidence.

        Args:
            claim: The claim to verify.
            evidence: Source evidence text.

        Returns:
            Tuple of (entailment_score, support_level).
        """
        if not self._initialized:
            await self.initialize()

        if not self._classifier:
            return 0.0, SupportLevel.NONE

        # Format as NLI premise-hypothesis pair
        # Premise: evidence, Hypothesis: claim
        input_text = f"{evidence} [SEP] {claim}"

        classifier = self._classifier

        def _score() -> Dict[str, Any]:
            result = classifier(input_text)
            return result[0] if isinstance(result, list) else result

        result = await asyncio.to_thread(_score)

        # Parse NLI labels
        label = result.get("label", "").lower()
        score = result.get("score", 0.0)

        # Map to entailment score
        if "entail" in label:
            entailment_score = score
        elif "contradict" in label:
            entailment_score = 0.0
        else:  # neutral
            entailment_score = 0.5 * score

        # Determine support level
        if entailment_score >= 0.8:
            level = SupportLevel.STRONG
        elif entailment_score >= 0.5:
            level = SupportLevel.MODERATE
        elif entailment_score >= 0.2:
            level = SupportLevel.WEAK
        else:
            level = SupportLevel.NONE

        return entailment_score, level

    async def score_claims_batch(
        self,
        claims: List[str],
        evidence: str,
    ) -> List[Tuple[float, SupportLevel]]:
        """Score multiple claims against same evidence.

        Args:
            claims: List of claims.
            evidence: Source evidence text.

        Returns:
            List of (score, level) tuples.
        """
        tasks = [self.score_claim(claim, evidence) for claim in claims]
        return await asyncio.gather(*tasks)


# -----------------------------------------------------------------------------
# Evidence Gate
# -----------------------------------------------------------------------------


class EvidenceGate:
    """Combined citation extraction and faithfulness verification.

    Provides end-to-end response verification against source evidence.
    """

    def __init__(
        self,
        *,
        citation_extractor: Optional[CitationExtractor] = None,
        faithfulness_scorer: Optional[FaithfulnessScorer] = None,
        min_support_score: float = 0.5,
        flag_unsupported: bool = True,
    ) -> None:
        """Initialize evidence gate.

        Args:
            citation_extractor: Custom citation extractor.
            faithfulness_scorer: Custom faithfulness scorer.
            min_support_score: Minimum score to consider supported.
            flag_unsupported: Whether to flag unsupported claims.
        """
        self._extractor = citation_extractor or CitationExtractor()
        self._scorer = faithfulness_scorer or FaithfulnessScorer()
        self._min_support_score = min_support_score
        self._flag_unsupported = flag_unsupported
        self._initialized = False

    @property
    def is_initialized(self) -> bool:
        """Check if gate is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Initialize components."""
        if self._initialized:
            return
        await self._scorer.initialize()
        self._initialized = True

    async def shutdown(self) -> None:
        """Cleanup resources."""
        await self._scorer.shutdown()
        self._initialized = False

    async def verify_response(
        self,
        response: str,
        source_chunks: List["SearchResult"],
        *,
        verify_all_sentences: bool = False,
    ) -> VerificationResult:
        """Verify a response against source chunks.

        Args:
            response: LLM response text.
            source_chunks: Retrieved source chunks.
            verify_all_sentences: Verify sentences without citations too.

        Returns:
            VerificationResult with claim verifications.
        """
        if not self._initialized:
            await self.initialize()

        # Extract citations
        citations = self._extractor.extract(response)

        # Build index -> chunk mapping
        chunk_map: Dict[int, "SearchResult"] = {}
        for i, chunk in enumerate(source_chunks, start=1):
            chunk_map[i] = chunk

        # Group citations by claim
        claim_citations: Dict[str, List[Citation]] = {}
        for citation in citations:
            if citation.claim not in claim_citations:
                claim_citations[citation.claim] = []
            claim_citations[citation.claim].append(citation)

        # Verify each claim
        verifications: List[ClaimVerification] = []
        supported: List[str] = []
        unsupported: List[str] = []
        valid_citations = 0

        for claim, claim_cites in claim_citations.items():
            cite_indices = list(set(c.index for c in claim_cites))

            # Get evidence from cited chunks
            evidence_texts: List[str] = []
            chunk_ids: List[str] = []
            for idx in cite_indices:
                if idx in chunk_map:
                    evidence_texts.append(chunk_map[idx].chunk.content)
                    chunk_ids.append(chunk_map[idx].chunk.id)
                    valid_citations += 1

            if not evidence_texts:
                # No valid sources for this claim
                verifications.append(
                    ClaimVerification(
                        claim=claim,
                        citations=cite_indices,
                        source_chunks=[],
                        entailment_score=0.0,
                        is_supported=False,
                        confidence=1.0,
                    )
                )
                unsupported.append(claim)
                continue

            # Combine evidence and score
            combined_evidence = " ".join(evidence_texts)
            score, level = await self._scorer.score_claim(claim, combined_evidence)

            is_supported = score >= self._min_support_score
            verifications.append(
                ClaimVerification(
                    claim=claim,
                    citations=cite_indices,
                    source_chunks=chunk_ids,
                    entailment_score=score,
                    is_supported=is_supported,
                    confidence=score,
                )
            )

            if is_supported:
                supported.append(claim)
            else:
                unsupported.append(claim)

        # Calculate overall score
        if verifications:
            overall_score = sum(v.entailment_score for v in verifications) / len(verifications)
        else:
            overall_score = 1.0  # No claims to verify

        return VerificationResult(
            response=response,
            claims=verifications,
            overall_score=overall_score,
            supported_claims=supported,
            unsupported_claims=unsupported,
            citations_found=len(citations),
            citations_valid=valid_citations,
        )

    def format_with_confidence(
        self,
        response: str,
        verification: VerificationResult,
        *,
        show_scores: bool = False,
        warn_unsupported: bool = True,
    ) -> str:
        """Format response with confidence annotations.

        Args:
            response: Original response.
            verification: Verification result.
            show_scores: Whether to show entailment scores.
            warn_unsupported: Whether to add warnings for unsupported claims.

        Returns:
            Formatted response text.
        """
        result = response

        if warn_unsupported and verification.unsupported_claims:
            warnings = "\n\n⚠️ **Note**: The following claims may not be fully supported by sources:\n"
            for claim in verification.unsupported_claims:
                warnings += f"- {claim[:100]}...\n" if len(claim) > 100 else f"- {claim}\n"
            result += warnings

        if show_scores:
            result += f"\n\n📊 **Evidence Score**: {verification.overall_score:.2f}"

        return result
