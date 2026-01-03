"""RAG Evaluation Framework.

Implements RAGAS-style metrics for evaluating RAG pipeline quality:
- Faithfulness: Are generated answers grounded in retrieved context?
- Answer Relevancy: How relevant is the answer to the question?
- Context Precision: How precise is the retrieved context for the question?
- Context Recall: Does the context contain information needed for the answer?

Example:
    >>> from modules.storage.retrieval.evaluation import RAGEvaluator
    >>> 
    >>> evaluator = RAGEvaluator()
    >>> result = evaluator.evaluate_sample(
    ...     question="What is Python?",
    ...     answer="Python is a programming language.",
    ...     context="Python is a high-level programming language...",
    ...     ground_truth="Python is a general-purpose programming language."
    ... )
    >>> print(f"Faithfulness: {result.faithfulness:.2f}")
"""

from __future__ import annotations

import asyncio
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)


@dataclass
class EvaluationSample:
    """A single evaluation sample for RAG assessment.
    
    Attributes:
        question: The query/question posed to the system.
        answer: The generated answer from the RAG system.
        contexts: List of retrieved context chunks.
        ground_truth: Optional reference answer for comparison.
        metadata: Optional additional metadata for the sample.
    """
    
    question: str
    answer: str
    contexts: List[str]
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetricResult:
    """Result of computing a single metric.
    
    Attributes:
        name: Metric name.
        score: Computed score (0.0 to 1.0).
        details: Optional breakdown/explanation of the score.
    """
    
    name: str
    score: float
    details: Optional[Dict[str, Any]] = None


@dataclass
class EvaluationResult:
    """Complete evaluation result for a sample.
    
    Attributes:
        sample: The original evaluation sample.
        faithfulness: Score for answer faithfulness to context.
        answer_relevancy: Score for answer relevancy to question.
        context_precision: Score for context precision.
        context_recall: Score for context recall (requires ground_truth).
        overall_score: Weighted average of all metrics.
        metric_details: Detailed breakdown per metric.
    """
    
    sample: EvaluationSample
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: Optional[float]
    overall_score: float
    metric_details: Dict[str, MetricResult] = field(default_factory=dict)
    
    @property
    def to_dict(self) -> Dict[str, Any]:
        """Convert to serializable dictionary."""
        return {
            "question": self.sample.question,
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall_score": self.overall_score,
        }


@dataclass
class BatchEvaluationResult:
    """Aggregated results from evaluating multiple samples.
    
    Attributes:
        results: Individual evaluation results.
        mean_faithfulness: Mean faithfulness across samples.
        mean_answer_relevancy: Mean answer relevancy.
        mean_context_precision: Mean context precision.
        mean_context_recall: Mean context recall (if available).
        mean_overall: Mean overall score.
        sample_count: Number of samples evaluated.
    """
    
    results: List[EvaluationResult]
    mean_faithfulness: float
    mean_answer_relevancy: float
    mean_context_precision: float
    mean_context_recall: Optional[float]
    mean_overall: float
    sample_count: int
    
    @property
    def summary(self) -> Dict[str, float]:
        """Get summary metrics."""
        result = {
            "faithfulness": self.mean_faithfulness,
            "answer_relevancy": self.mean_answer_relevancy,
            "context_precision": self.mean_context_precision,
            "overall": self.mean_overall,
            "sample_count": self.sample_count,
        }
        if self.mean_context_recall is not None:
            result["context_recall"] = self.mean_context_recall
        return result


class RAGMetric(ABC):
    """Abstract base class for RAG evaluation metrics."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Metric name."""
        pass
    
    @abstractmethod
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute metric for a sample.
        
        Args:
            sample: The evaluation sample.
            
        Returns:
            MetricResult with score and optional details.
        """
        pass


class FaithfulnessMetric(RAGMetric):
    """Measures if the answer is grounded in the provided context.
    
    Uses NLI (Natural Language Inference) to check if claims in the answer
    can be inferred from the context. A faithfulness score of 1.0 means
    all claims are supported by context.
    
    Attributes:
        nli_model: Optional pre-loaded NLI pipeline.
        sentence_splitter: Function to split text into sentences.
    """
    
    def __init__(
        self,
        nli_model: Optional[Any] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    ):
        self._nli_model = nli_model
        self._nli_pipeline: Optional[Any] = None
        self._sentence_splitter = sentence_splitter or self._default_sentence_split
    
    @property
    def name(self) -> str:
        return "faithfulness"
    
    def _default_sentence_split(self, text: str) -> List[str]:
        """Split text into sentences using regex."""
        # Simple sentence splitting - handles common cases
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_nli_pipeline(self) -> Optional[Any]:
        """Lazy-load NLI pipeline."""
        if self._nli_pipeline is not None:
            return self._nli_pipeline
        
        if self._nli_model is not None:
            self._nli_pipeline = self._nli_model
            return self._nli_pipeline
        
        try:
            from transformers import pipeline
            self._nli_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,  # CPU
            )
            return self._nli_pipeline
        except Exception as exc:
            logger.warning("Failed to load NLI model for faithfulness: %s", exc)
            return None
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute faithfulness score.
        
        Splits answer into claims (sentences) and checks how many
        are entailed by the context using NLI.
        """
        if not sample.answer or not sample.contexts:
            return MetricResult(name=self.name, score=0.0)
        
        # Split answer into claims
        claims = self._sentence_splitter(sample.answer)
        if not claims:
            return MetricResult(name=self.name, score=1.0)
        
        # Combine contexts
        context = " ".join(sample.contexts)
        
        # Try NLI-based faithfulness
        nli = self._get_nli_pipeline()
        if nli:
            try:
                supported_count = 0
                claim_results = []
                
                for claim in claims:
                    # Check if claim is entailed by context
                    result = nli(
                        claim,
                        candidate_labels=["supported", "not_supported"],
                        hypothesis_template="This claim is {} by the context: " + context[:1000],
                    )
                    is_supported = result["labels"][0] == "supported"
                    confidence = result["scores"][0]
                    
                    if is_supported and confidence > 0.5:
                        supported_count += 1
                    
                    claim_results.append({
                        "claim": claim[:100],
                        "supported": is_supported,
                        "confidence": confidence,
                    })
                
                score = supported_count / len(claims)
                return MetricResult(
                    name=self.name,
                    score=score,
                    details={"claims": claim_results},
                )
            except Exception as exc:
                logger.warning("NLI faithfulness check failed: %s", exc)
        
        # Fallback: lexical overlap
        return self._lexical_faithfulness(claims, context)
    
    def _lexical_faithfulness(
        self,
        claims: List[str],
        context: str,
    ) -> MetricResult:
        """Fallback faithfulness using lexical overlap."""
        context_lower = context.lower()
        context_words = set(context_lower.split())
        
        supported_count = 0
        for claim in claims:
            claim_words = set(claim.lower().split())
            overlap = len(claim_words & context_words) / max(len(claim_words), 1)
            if overlap > 0.5:  # More than half of words in context
                supported_count += 1
        
        score = supported_count / len(claims) if claims else 1.0
        return MetricResult(
            name=self.name,
            score=score,
            details={"method": "lexical_overlap"},
        )


class AnswerRelevancyMetric(RAGMetric):
    """Measures if the answer is relevant to the question.
    
    Uses semantic similarity between question and answer to determine
    if the answer addresses what was asked.
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
    ):
        self._embedding_model = embedding_model
        self._similarity_model: Optional[Any] = None
    
    @property
    def name(self) -> str:
        return "answer_relevancy"
    
    def _get_similarity_model(self) -> Optional[Any]:
        """Lazy-load sentence similarity model."""
        if self._similarity_model is not None:
            return self._similarity_model
        
        if self._embedding_model is not None:
            self._similarity_model = self._embedding_model
            return self._similarity_model
        
        try:
            from sentence_transformers import SentenceTransformer
            self._similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._similarity_model
        except Exception as exc:
            logger.warning("Failed to load embedding model for relevancy: %s", exc)
            return None
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute answer relevancy score.
        
        Measures semantic similarity between question and answer.
        Higher similarity indicates the answer addresses the question.
        """
        if not sample.answer or not sample.question:
            return MetricResult(name=self.name, score=0.0)
        
        model = self._get_similarity_model()
        if model:
            try:
                # Encode question and answer
                embeddings = model.encode([sample.question, sample.answer])
                
                # Compute cosine similarity
                from numpy import dot
                from numpy.linalg import norm
                
                similarity = dot(embeddings[0], embeddings[1]) / (
                    norm(embeddings[0]) * norm(embeddings[1])
                )
                
                # Convert to 0-1 score (similarity can be negative)
                score = max(0.0, min(1.0, (similarity + 1) / 2))
                
                return MetricResult(
                    name=self.name,
                    score=score,
                    details={"cosine_similarity": float(similarity)},
                )
            except Exception as exc:
                logger.warning("Semantic relevancy failed: %s", exc)
        
        # Fallback: word overlap
        return self._lexical_relevancy(sample.question, sample.answer)
    
    def _lexical_relevancy(self, question: str, answer: str) -> MetricResult:
        """Fallback relevancy using word overlap."""
        question_words = set(question.lower().split())
        answer_words = set(answer.lower().split())
        
        # Remove common stop words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "when", "where"}
        question_words -= stop_words
        answer_words -= stop_words
        
        if not question_words:
            return MetricResult(name=self.name, score=1.0)
        
        overlap = len(question_words & answer_words)
        score = min(1.0, overlap / len(question_words))
        
        return MetricResult(
            name=self.name,
            score=score,
            details={"method": "lexical_overlap"},
        )


class ContextPrecisionMetric(RAGMetric):
    """Measures how precise the retrieved context is for the question.
    
    Evaluates whether retrieved chunks are actually relevant to answering
    the question, penalizing irrelevant chunks.
    """
    
    def __init__(
        self,
        embedding_model: Optional[Any] = None,
        relevance_threshold: float = 0.5,
    ):
        self._embedding_model = embedding_model
        self._similarity_model: Optional[Any] = None
        self._relevance_threshold = relevance_threshold
    
    @property
    def name(self) -> str:
        return "context_precision"
    
    def _get_similarity_model(self) -> Optional[Any]:
        """Lazy-load sentence similarity model."""
        if self._similarity_model is not None:
            return self._similarity_model
        
        if self._embedding_model is not None:
            self._similarity_model = self._embedding_model
            return self._similarity_model
        
        try:
            from sentence_transformers import SentenceTransformer
            self._similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
            return self._similarity_model
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s", exc)
            return None
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute context precision.
        
        Measures what fraction of retrieved contexts are relevant to the question.
        Uses rank-weighted precision (earlier chunks weighted more heavily).
        """
        if not sample.contexts or not sample.question:
            return MetricResult(name=self.name, score=0.0)
        
        model = self._get_similarity_model()
        if model:
            try:
                # Encode question and all contexts
                texts = [sample.question] + sample.contexts
                embeddings = model.encode(texts)
                
                question_emb = embeddings[0]
                context_embs = embeddings[1:]
                
                from numpy import dot
                from numpy.linalg import norm
                
                relevances = []
                for ctx_emb in context_embs:
                    similarity = dot(question_emb, ctx_emb) / (
                        norm(question_emb) * norm(ctx_emb)
                    )
                    relevances.append(float(similarity))
                
                # Compute precision@k with rank weighting
                relevant_count = 0
                weighted_sum = 0.0
                
                for rank, rel in enumerate(relevances, 1):
                    is_relevant = rel > self._relevance_threshold
                    if is_relevant:
                        relevant_count += 1
                        # Precision at this rank
                        precision_at_k = relevant_count / rank
                        weighted_sum += precision_at_k
                
                # Average precision
                if relevant_count > 0:
                    score = weighted_sum / len(relevances)
                else:
                    score = 0.0
                
                return MetricResult(
                    name=self.name,
                    score=min(1.0, score),
                    details={
                        "relevances": relevances,
                        "relevant_count": relevant_count,
                    },
                )
            except Exception as exc:
                logger.warning("Semantic context precision failed: %s", exc)
        
        # Fallback: assume all contexts are relevant
        return MetricResult(
            name=self.name,
            score=1.0,
            details={"method": "default"},
        )


class ContextRecallMetric(RAGMetric):
    """Measures if the context contains all information needed for the ground truth.
    
    Requires ground_truth answer to compute. Checks what fraction of the
    ground truth information can be found in the retrieved context.
    """
    
    def __init__(
        self,
        nli_model: Optional[Any] = None,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
    ):
        self._nli_model = nli_model
        self._nli_pipeline: Optional[Any] = None
        self._sentence_splitter = sentence_splitter or self._default_sentence_split
    
    @property
    def name(self) -> str:
        return "context_recall"
    
    def _default_sentence_split(self, text: str) -> List[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in sentences if s.strip()]
    
    def _get_nli_pipeline(self) -> Optional[Any]:
        """Lazy-load NLI pipeline."""
        if self._nli_pipeline is not None:
            return self._nli_pipeline
        
        if self._nli_model is not None:
            self._nli_pipeline = self._nli_model
            return self._nli_pipeline
        
        try:
            from transformers import pipeline
            self._nli_pipeline = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=-1,
            )
            return self._nli_pipeline
        except Exception as exc:
            logger.warning("Failed to load NLI model for context recall: %s", exc)
            return None
    
    def compute(self, sample: EvaluationSample) -> MetricResult:
        """Compute context recall.
        
        Measures what fraction of ground truth claims are present in context.
        Returns None if no ground_truth is available.
        """
        if not sample.ground_truth:
            return MetricResult(name=self.name, score=0.0, details={"error": "no ground_truth"})
        
        if not sample.contexts:
            return MetricResult(name=self.name, score=0.0)
        
        # Split ground truth into claims
        claims = self._sentence_splitter(sample.ground_truth)
        if not claims:
            return MetricResult(name=self.name, score=1.0)
        
        context = " ".join(sample.contexts)
        
        # Try NLI-based recall
        nli = self._get_nli_pipeline()
        if nli:
            try:
                found_count = 0
                claim_results = []
                
                for claim in claims:
                    result = nli(
                        claim,
                        candidate_labels=["present", "absent"],
                        hypothesis_template="This information is {} in the context: " + context[:1000],
                    )
                    is_present = result["labels"][0] == "present"
                    confidence = result["scores"][0]
                    
                    if is_present and confidence > 0.5:
                        found_count += 1
                    
                    claim_results.append({
                        "claim": claim[:100],
                        "found": is_present,
                        "confidence": confidence,
                    })
                
                score = found_count / len(claims)
                return MetricResult(
                    name=self.name,
                    score=score,
                    details={"claims": claim_results},
                )
            except Exception as exc:
                logger.warning("NLI context recall failed: %s", exc)
        
        # Fallback: lexical recall
        return self._lexical_recall(claims, context)
    
    def _lexical_recall(self, claims: List[str], context: str) -> MetricResult:
        """Fallback recall using lexical overlap."""
        context_lower = context.lower()
        context_words = set(context_lower.split())
        
        found_count = 0
        for claim in claims:
            claim_words = set(claim.lower().split())
            overlap = len(claim_words & context_words) / max(len(claim_words), 1)
            if overlap > 0.5:
                found_count += 1
        
        score = found_count / len(claims) if claims else 1.0
        return MetricResult(
            name=self.name,
            score=score,
            details={"method": "lexical_overlap"},
        )


class RAGEvaluator:
    """Main evaluator class for RAG pipeline assessment.
    
    Computes multiple metrics for evaluating RAG quality:
    - Faithfulness: Is the answer grounded in context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are retrieved chunks relevant?
    - Context Recall: Does context cover ground truth? (optional)
    
    Example:
        >>> evaluator = RAGEvaluator()
        >>> sample = EvaluationSample(
        ...     question="What is the capital of France?",
        ...     answer="The capital of France is Paris.",
        ...     contexts=["Paris is the capital and largest city of France."],
        ...     ground_truth="Paris is the capital of France."
        ... )
        >>> result = evaluator.evaluate_sample(sample)
        >>> print(f"Overall score: {result.overall_score:.2f}")
    """
    
    def __init__(
        self,
        metrics: Optional[List[RAGMetric]] = None,
        weights: Optional[Dict[str, float]] = None,
    ):
        """Initialize the evaluator.
        
        Args:
            metrics: List of metrics to compute. Uses defaults if None.
            weights: Weight for each metric in overall score.
        """
        if metrics is None:
            metrics = [
                FaithfulnessMetric(),
                AnswerRelevancyMetric(),
                ContextPrecisionMetric(),
                ContextRecallMetric(),
            ]
        
        self._metrics = {m.name: m for m in metrics}
        self._weights = weights or {
            "faithfulness": 0.3,
            "answer_relevancy": 0.25,
            "context_precision": 0.25,
            "context_recall": 0.2,
        }
    
    def evaluate_sample(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample.
        
        Args:
            sample: The evaluation sample to assess.
            
        Returns:
            EvaluationResult with all metric scores.
        """
        results = {}
        for name, metric in self._metrics.items():
            try:
                results[name] = metric.compute(sample)
            except Exception as exc:
                logger.error("Metric %s failed: %s", name, exc)
                results[name] = MetricResult(name=name, score=0.0, details={"error": str(exc)})
        
        # Extract scores
        faithfulness = results.get("faithfulness", MetricResult("faithfulness", 0.0)).score
        answer_relevancy = results.get("answer_relevancy", MetricResult("answer_relevancy", 0.0)).score
        context_precision = results.get("context_precision", MetricResult("context_precision", 0.0)).score
        
        context_recall: Optional[float] = None
        if sample.ground_truth and "context_recall" in results:
            context_recall = results["context_recall"].score
        
        # Compute weighted overall score
        total_weight = 0.0
        weighted_sum = 0.0
        
        for name, score_val in [
            ("faithfulness", faithfulness),
            ("answer_relevancy", answer_relevancy),
            ("context_precision", context_precision),
        ]:
            weight = self._weights.get(name, 0.0)
            weighted_sum += score_val * weight
            total_weight += weight
        
        if context_recall is not None:
            weight = self._weights.get("context_recall", 0.0)
            weighted_sum += context_recall * weight
            total_weight += weight
        
        overall = weighted_sum / total_weight if total_weight > 0 else 0.0
        
        return EvaluationResult(
            sample=sample,
            faithfulness=faithfulness,
            answer_relevancy=answer_relevancy,
            context_precision=context_precision,
            context_recall=context_recall,
            overall_score=overall,
            metric_details=results,
        )
    
    def evaluate_batch(
        self,
        samples: Sequence[EvaluationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEvaluationResult:
        """Evaluate multiple samples.
        
        Args:
            samples: Sequence of evaluation samples.
            progress_callback: Optional callback(current, total) for progress.
            
        Returns:
            BatchEvaluationResult with aggregated metrics.
        """
        results: List[EvaluationResult] = []
        total = len(samples)
        
        for i, sample in enumerate(samples):
            result = self.evaluate_sample(sample)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        if not results:
            return BatchEvaluationResult(
                results=[],
                mean_faithfulness=0.0,
                mean_answer_relevancy=0.0,
                mean_context_precision=0.0,
                mean_context_recall=None,
                mean_overall=0.0,
                sample_count=0,
            )
        
        # Compute means
        mean_faithfulness = sum(r.faithfulness for r in results) / len(results)
        mean_relevancy = sum(r.answer_relevancy for r in results) / len(results)
        mean_precision = sum(r.context_precision for r in results) / len(results)
        mean_overall = sum(r.overall_score for r in results) / len(results)
        
        # Context recall only if any samples have ground_truth
        recall_results = [r.context_recall for r in results if r.context_recall is not None]
        mean_recall = sum(recall_results) / len(recall_results) if recall_results else None
        
        return BatchEvaluationResult(
            results=results,
            mean_faithfulness=mean_faithfulness,
            mean_answer_relevancy=mean_relevancy,
            mean_context_precision=mean_precision,
            mean_context_recall=mean_recall,
            mean_overall=mean_overall,
            sample_count=len(results),
        )
    
    async def evaluate_batch_async(
        self,
        samples: Sequence[EvaluationSample],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> BatchEvaluationResult:
        """Evaluate samples asynchronously.
        
        Runs evaluation in thread pool to avoid blocking.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.evaluate_batch(samples, progress_callback),
        )


def create_evaluator(
    *,
    use_nli: bool = True,
    use_embeddings: bool = True,
    weights: Optional[Dict[str, float]] = None,
) -> RAGEvaluator:
    """Factory function to create a RAG evaluator.
    
    Args:
        use_nli: Whether to use NLI models (requires transformers).
        use_embeddings: Whether to use embedding models (requires sentence-transformers).
        weights: Custom metric weights.
        
    Returns:
        Configured RAGEvaluator instance.
    """
    metrics: List[RAGMetric] = []
    
    # Faithfulness - uses NLI or falls back to lexical
    faithfulness = FaithfulnessMetric()
    metrics.append(faithfulness)
    
    # Answer relevancy - uses embeddings or falls back to lexical
    relevancy = AnswerRelevancyMetric()
    metrics.append(relevancy)
    
    # Context precision - uses embeddings or falls back to default
    precision = ContextPrecisionMetric()
    metrics.append(precision)
    
    # Context recall - uses NLI or falls back to lexical
    recall = ContextRecallMetric()
    metrics.append(recall)
    
    return RAGEvaluator(metrics=metrics, weights=weights)


__all__ = [
    # Data classes
    "EvaluationSample",
    "MetricResult",
    "EvaluationResult",
    "BatchEvaluationResult",
    # Metrics
    "RAGMetric",
    "FaithfulnessMetric",
    "AnswerRelevancyMetric",
    "ContextPrecisionMetric",
    "ContextRecallMetric",
    # Evaluator
    "RAGEvaluator",
    "create_evaluator",
]
