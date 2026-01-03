"""Query router for intent-aware retrieval.

Classifies queries into intent categories and provides retrieval parameters
optimized for each intent type:

- FACTUAL: Precise, fact-based queries (narrow focus, high precision)
- ANALYTICAL: Complex queries requiring synthesis (broader search, more context)
- CREATIVE: Open-ended or generative queries (minimal RAG, high temperature)
- NAVIGATIONAL: Looking for specific documents/resources

Usage:
    >>> from modules.storage.retrieval.query_router import QueryRouter, QueryIntent
    >>> router = QueryRouter()
    >>> await router.initialize()
    >>> result = await router.classify("What is the capital of France?")
    >>> print(result.intent)  # QueryIntent.FACTUAL
    >>> print(result.confidence)  # 0.95
    >>> print(result.retrieval_params)  # {"top_k": 3, "rerank": True, ...}
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from modules.logging.logger import setup_logger

logger = setup_logger(__name__)


# -----------------------------------------------------------------------------
# Enums
# -----------------------------------------------------------------------------


class QueryIntent(str, Enum):
    """Query intent categories for retrieval optimization."""

    FACTUAL = "factual"  # Precise fact lookup (e.g., "What is X?")
    ANALYTICAL = "analytical"  # Complex analysis (e.g., "Compare X and Y")
    CREATIVE = "creative"  # Open-ended generation (e.g., "Write a poem")
    NAVIGATIONAL = "navigational"  # Looking for resources (e.g., "Find docs on X")
    UNKNOWN = "unknown"  # Could not classify


# Intent labels for zero-shot classification
INTENT_LABELS = [
    "factual question requiring precise information",
    "analytical question requiring comparison or synthesis",
    "creative or open-ended request",
    "navigational request to find specific documents or resources",
]

# Map labels to intents
LABEL_TO_INTENT = {
    "factual question requiring precise information": QueryIntent.FACTUAL,
    "analytical question requiring comparison or synthesis": QueryIntent.ANALYTICAL,
    "creative or open-ended request": QueryIntent.CREATIVE,
    "navigational request to find specific documents or resources": QueryIntent.NAVIGATIONAL,
}


# Default retrieval parameters for each intent
DEFAULT_RETRIEVAL_PARAMS: Dict[QueryIntent, Dict[str, Any]] = {
    QueryIntent.FACTUAL: {
        "top_k": 3,
        "rerank": True,
        "use_hybrid": True,
        "min_score": 0.7,
        "max_context_chunks": 3,
    },
    QueryIntent.ANALYTICAL: {
        "top_k": 10,
        "rerank": True,
        "use_hybrid": True,
        "min_score": 0.5,
        "max_context_chunks": 8,
    },
    QueryIntent.CREATIVE: {
        "top_k": 2,
        "rerank": False,
        "use_hybrid": False,
        "min_score": 0.8,
        "max_context_chunks": 2,
        "skip_rag": True,  # Suggest skipping RAG entirely
    },
    QueryIntent.NAVIGATIONAL: {
        "top_k": 15,
        "rerank": True,
        "use_hybrid": True,
        "min_score": 0.3,
        "max_context_chunks": 10,
    },
    QueryIntent.UNKNOWN: {
        "top_k": 5,
        "rerank": True,
        "use_hybrid": True,
        "min_score": 0.6,
        "max_context_chunks": 5,
    },
}


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------


@dataclass
class ClassificationResult:
    """Result from query intent classification.

    Attributes:
        query: Original query text.
        intent: Classified intent category.
        confidence: Confidence score (0-1).
        all_scores: Scores for all intent categories.
        retrieval_params: Recommended retrieval parameters.
    """

    query: str
    intent: QueryIntent
    confidence: float
    all_scores: Dict[QueryIntent, float] = field(default_factory=dict)
    retrieval_params: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Query Router
# -----------------------------------------------------------------------------


class QueryRouter:
    """Zero-shot query intent classifier.

    Uses a zero-shot classification model to determine query intent
    and provide optimized retrieval parameters.
    """

    def __init__(
        self,
        model: str = "facebook/bart-large-mnli",
        device: Optional[str] = None,
        confidence_threshold: float = 0.5,
        custom_params: Optional[Dict[QueryIntent, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize the query router.

        Args:
            model: HuggingFace model for zero-shot classification.
            device: Device to use (cuda, cpu, mps). Auto-detected if None.
            confidence_threshold: Minimum confidence for intent assignment.
            custom_params: Custom retrieval params per intent (overrides defaults).
        """
        self._model_name = model
        self._device = device
        self._confidence_threshold = confidence_threshold
        self._classifier: Optional[Any] = None
        self._initialized = False

        # Merge custom params with defaults
        self._retrieval_params = DEFAULT_RETRIEVAL_PARAMS.copy()
        if custom_params:
            for intent, params in custom_params.items():
                if intent in self._retrieval_params:
                    self._retrieval_params[intent].update(params)
                else:
                    self._retrieval_params[intent] = params

    @property
    def is_initialized(self) -> bool:
        """Check if router is initialized."""
        return self._initialized

    async def initialize(self) -> None:
        """Load the classification model."""
        if self._initialized:
            return

        def _load() -> Any:
            try:
                from transformers import pipeline
            except ImportError:
                raise RuntimeError(
                    "QueryRouter requires transformers. "
                    "Install with: pip install transformers"
                )

            device = self._device
            if device is None:
                try:
                    import torch
                    if torch.cuda.is_available():
                        device = 0  # CUDA device index
                    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                        device = "mps"
                    else:
                        device = -1  # CPU
                except ImportError:
                    device = -1

            return pipeline(
                "zero-shot-classification",
                model=self._model_name,
                device=device,
            )

        self._classifier = await asyncio.to_thread(_load)
        self._initialized = True
        logger.info(f"QueryRouter initialized with {self._model_name}")

    async def shutdown(self) -> None:
        """Cleanup model resources."""
        self._classifier = None
        self._initialized = False

    async def classify(
        self,
        query: str,
        *,
        hypothesis_template: str = "This query is a {}.",
    ) -> ClassificationResult:
        """Classify query intent.

        Args:
            query: Query text to classify.
            hypothesis_template: Template for zero-shot hypothesis.

        Returns:
            ClassificationResult with intent and parameters.
        """
        if not self._initialized:
            await self.initialize()

        if not self._classifier:
            # Fallback if model failed to load
            return ClassificationResult(
                query=query,
                intent=QueryIntent.UNKNOWN,
                confidence=0.0,
                retrieval_params=self._retrieval_params[QueryIntent.UNKNOWN],
            )

        # Capture classifier in local variable for type safety
        classifier = self._classifier

        def _classify() -> Dict[str, Any]:
            return classifier(
                query,
                INTENT_LABELS,
                hypothesis_template=hypothesis_template,
                multi_label=False,
            )

        result = await asyncio.to_thread(_classify)

        # Parse results
        all_scores: Dict[QueryIntent, float] = {}
        for label, score in zip(result["labels"], result["scores"]):
            intent = LABEL_TO_INTENT.get(label, QueryIntent.UNKNOWN)
            all_scores[intent] = score

        # Get top intent
        top_label = result["labels"][0]
        top_score = result["scores"][0]
        top_intent = LABEL_TO_INTENT.get(top_label, QueryIntent.UNKNOWN)

        # Apply confidence threshold
        if top_score < self._confidence_threshold:
            top_intent = QueryIntent.UNKNOWN

        return ClassificationResult(
            query=query,
            intent=top_intent,
            confidence=top_score,
            all_scores=all_scores,
            retrieval_params=self._retrieval_params.get(
                top_intent,
                self._retrieval_params[QueryIntent.UNKNOWN],
            ),
        )

    async def classify_batch(
        self,
        queries: List[str],
        *,
        hypothesis_template: str = "This query is a {}.",
    ) -> List[ClassificationResult]:
        """Classify multiple queries.

        Args:
            queries: List of query texts.
            hypothesis_template: Template for zero-shot hypothesis.

        Returns:
            List of ClassificationResult for each query.
        """
        if not queries:
            return []

        # Classify in parallel (within reason)
        tasks = [
            self.classify(q, hypothesis_template=hypothesis_template)
            for q in queries
        ]
        return await asyncio.gather(*tasks)

    def get_retrieval_params(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get retrieval parameters for a specific intent.

        Args:
            intent: Query intent category.

        Returns:
            Dict of retrieval parameters.
        """
        return self._retrieval_params.get(
            intent,
            self._retrieval_params[QueryIntent.UNKNOWN],
        ).copy()


# -----------------------------------------------------------------------------
# Simple Rule-Based Router (Fallback)
# -----------------------------------------------------------------------------


class SimpleQueryRouter:
    """Rule-based query router (no ML dependencies).

    Uses keyword heuristics to classify queries. Useful as a fallback
    or for environments where ML models are not available.
    """

    # Keyword patterns for each intent
    FACTUAL_KEYWORDS = {
        "what is", "what are", "who is", "who are", "when did",
        "when was", "where is", "where are", "how many", "how much",
        "define", "definition", "meaning of", "explain",
    }
    ANALYTICAL_KEYWORDS = {
        "compare", "contrast", "analyze", "analyse", "difference between",
        "similarities", "pros and cons", "advantages", "disadvantages",
        "why does", "why is", "how does", "relationship between",
        "impact of", "effect of", "implications",
    }
    CREATIVE_KEYWORDS = {
        "write", "create", "generate", "compose", "draft",
        "make up", "imagine", "story about", "poem about",
        "help me write", "brainstorm", "ideas for",
    }
    NAVIGATIONAL_KEYWORDS = {
        "find", "search for", "look for", "locate", "where can i find",
        "show me", "list of", "documentation", "docs for",
        "guide to", "tutorial", "how to find",
    }

    def __init__(
        self,
        custom_params: Optional[Dict[QueryIntent, Dict[str, Any]]] = None,
    ) -> None:
        """Initialize simple router.

        Args:
            custom_params: Custom retrieval params per intent.
        """
        self._retrieval_params = DEFAULT_RETRIEVAL_PARAMS.copy()
        if custom_params:
            for intent, params in custom_params.items():
                if intent in self._retrieval_params:
                    self._retrieval_params[intent].update(params)
                else:
                    self._retrieval_params[intent] = params

    @property
    def is_initialized(self) -> bool:
        """Simple router needs no initialization."""
        return True

    async def initialize(self) -> None:
        """No-op for simple router."""
        pass

    async def shutdown(self) -> None:
        """No-op for simple router."""
        pass

    async def classify(self, query: str) -> ClassificationResult:
        """Classify query using keyword heuristics.

        Args:
            query: Query text to classify.

        Returns:
            ClassificationResult with intent and parameters.
        """
        query_lower = query.lower()

        # Check each category
        scores: Dict[QueryIntent, float] = {
            QueryIntent.FACTUAL: 0.0,
            QueryIntent.ANALYTICAL: 0.0,
            QueryIntent.CREATIVE: 0.0,
            QueryIntent.NAVIGATIONAL: 0.0,
            QueryIntent.UNKNOWN: 0.1,  # Base score
        }

        for keyword in self.FACTUAL_KEYWORDS:
            if keyword in query_lower:
                scores[QueryIntent.FACTUAL] += 0.3

        for keyword in self.ANALYTICAL_KEYWORDS:
            if keyword in query_lower:
                scores[QueryIntent.ANALYTICAL] += 0.3

        for keyword in self.CREATIVE_KEYWORDS:
            if keyword in query_lower:
                scores[QueryIntent.CREATIVE] += 0.3

        for keyword in self.NAVIGATIONAL_KEYWORDS:
            if keyword in query_lower:
                scores[QueryIntent.NAVIGATIONAL] += 0.3

        # Normalize and find top
        max_score = max(scores.values())
        if max_score > 0:
            scores = {k: v / max_score for k, v in scores.items()}

        top_intent = max(scores, key=lambda k: scores[k])
        top_score = scores[top_intent]

        # If no strong signal, default to UNKNOWN
        if top_score < 0.3:
            top_intent = QueryIntent.UNKNOWN
            top_score = 0.5

        return ClassificationResult(
            query=query,
            intent=top_intent,
            confidence=min(top_score, 1.0),
            all_scores=scores,
            retrieval_params=self._retrieval_params.get(
                top_intent,
                self._retrieval_params[QueryIntent.UNKNOWN],
            ),
        )

    def get_retrieval_params(self, intent: QueryIntent) -> Dict[str, Any]:
        """Get retrieval parameters for a specific intent."""
        return self._retrieval_params.get(
            intent,
            self._retrieval_params[QueryIntent.UNKNOWN],
        ).copy()
