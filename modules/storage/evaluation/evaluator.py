"""RAG Evaluator implementation.

Provides RAGAS-style evaluation metrics for measuring RAG system quality:
- Faithfulness: Measures if generated answers are grounded in context
- Relevancy: Measures if retrieved context is relevant to the query
- Context Precision: Measures the precision of retrieved chunks
- Context Recall: Measures if important information is retrieved
- Answer Correctness: Measures similarity to ground truth answers
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple
import logging
import math
import re

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of evaluation metrics."""
    
    FAITHFULNESS = "faithfulness"
    RELEVANCY = "relevancy"
    CONTEXT_PRECISION = "context_precision"
    CONTEXT_RECALL = "context_recall"
    ANSWER_CORRECTNESS = "answer_correctness"
    ANSWER_SIMILARITY = "answer_similarity"


@dataclass
class EvaluationSample:
    """A single evaluation sample with question, context, answer, and optional ground truth.
    
    Attributes:
        question: The query/question.
        contexts: List of retrieved context chunks.
        answer: The generated answer.
        ground_truth: Optional expected/correct answer.
        metadata: Additional metadata about the sample.
    """
    
    question: str
    contexts: List[str]
    answer: str
    ground_truth: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationMetrics:
    """Container for evaluation metric scores.
    
    Attributes:
        faithfulness: Score 0-1 measuring grounding in context.
        relevancy: Score 0-1 measuring context relevance.
        context_precision: Score 0-1 measuring precision of retrieval.
        context_recall: Score 0-1 measuring recall of relevant info.
        answer_correctness: Score 0-1 measuring answer quality.
        answer_similarity: Score 0-1 measuring similarity to ground truth.
    """
    
    faithfulness: Optional[float] = None
    relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    answer_correctness: Optional[float] = None
    answer_similarity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary representation."""
        return {
            "faithfulness": self.faithfulness,
            "relevancy": self.relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "answer_correctness": self.answer_correctness,
            "answer_similarity": self.answer_similarity,
        }
    
    @property
    def mean_score(self) -> float:
        """Calculate mean of all non-None metrics."""
        values = [v for v in self.to_dict().values() if v is not None]
        return sum(values) / len(values) if values else 0.0


@dataclass
class EvaluationResult:
    """Result of evaluating a single sample.
    
    Attributes:
        sample: The evaluation sample.
        metrics: Computed metrics for this sample.
        details: Additional details about metric computation.
        error: Any error that occurred during evaluation.
    """
    
    sample: EvaluationSample
    metrics: EvaluationMetrics
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    @property
    def is_success(self) -> bool:
        """Check if evaluation completed without error."""
        return self.error is None


class MetricCalculator(ABC):
    """Abstract base class for metric calculators."""
    
    @property
    @abstractmethod
    def metric_type(self) -> MetricType:
        """Return the type of metric this calculator computes."""
        ...
    
    @abstractmethod
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate metric for a sample.
        
        Args:
            sample: The evaluation sample.
            
        Returns:
            Tuple of (score, details_dict).
        """
        ...


class FaithfulnessCalculator(MetricCalculator):
    """Calculates faithfulness score using NLI or heuristic approach.
    
    Faithfulness measures whether claims in the answer are supported by the context.
    Uses NLI (Natural Language Inference) when available, falls back to heuristics.
    """
    
    def __init__(self):
        """Initialize faithfulness calculator."""
        self._nli_model: Optional[Any] = None
        self._nli_tokenizer: Optional[Any] = None
        self._initialized = False
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.FAITHFULNESS
    
    def _initialize_nli(self) -> bool:
        """Initialize NLI model for faithfulness scoring."""
        if self._initialized:
            return self._nli_model is not None
        
        self._initialized = True
        
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            
            model_name = "microsoft/deberta-base-mnli"
            self._nli_tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._nli_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info("Loaded NLI model for faithfulness calculation")
            return True
        except ImportError:
            logger.warning("transformers not available, using heuristic faithfulness")
            return False
        except Exception as exc:
            logger.warning("Failed to load NLI model: %s", exc)
            return False
    
    def _extract_claims(self, text: str) -> List[str]:
        """Extract individual claims/sentences from text."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        
        # Filter out very short fragments
        claims = [s.strip() for s in sentences if len(s.strip()) > 10]
        return claims
    
    def _score_claim_nli(self, claim: str, context: str) -> float:
        """Score a claim against context using NLI."""
        if not self._nli_model or not self._nli_tokenizer:
            return 0.0
        
        try:
            import torch
            
            # Prepare input: premise (context) -> hypothesis (claim)
            inputs = self._nli_tokenizer(
                context,
                claim,
                return_tensors="pt",
                truncation=True,
                max_length=512,
            )
            
            with torch.no_grad():
                outputs = self._nli_model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # DeBERTa MNLI: 0=contradiction, 1=neutral, 2=entailment
                entailment_prob = probs[0][2].item()
                return entailment_prob
        except Exception as exc:
            logger.debug("NLI scoring failed: %s", exc)
            return 0.0
    
    def _score_claim_heuristic(self, claim: str, context: str) -> float:
        """Score a claim using keyword overlap heuristic."""
        # Tokenize and normalize
        claim_tokens = set(claim.lower().split())
        context_tokens = set(context.lower().split())
        
        # Remove common stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                     "being", "have", "has", "had", "do", "does", "did", "will",
                     "would", "could", "should", "may", "might", "must", "shall",
                     "can", "need", "dare", "ought", "used", "to", "of", "in",
                     "for", "on", "with", "at", "by", "from", "as", "into",
                     "through", "during", "before", "after", "above", "below",
                     "between", "under", "again", "further", "then", "once",
                     "and", "but", "or", "nor", "so", "yet", "both", "either",
                     "neither", "not", "only", "own", "same", "than", "too",
                     "very", "just", "also", "now", "here", "there", "when",
                     "where", "why", "how", "all", "each", "every", "some",
                     "any", "no", "most", "other", "such", "this", "that",
                     "these", "those", "it", "its"}
        
        claim_content = claim_tokens - stopwords
        context_content = context_tokens - stopwords
        
        if not claim_content:
            return 0.5  # Neutral if no content words
        
        # Calculate overlap
        overlap = claim_content & context_content
        overlap_ratio = len(overlap) / len(claim_content)
        
        return overlap_ratio
    
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate faithfulness score for a sample."""
        if not sample.answer or not sample.contexts:
            return 0.0, {"error": "Missing answer or contexts"}
        
        # Combine contexts
        full_context = " ".join(sample.contexts)
        
        # Extract claims from answer
        claims = self._extract_claims(sample.answer)
        if not claims:
            return 0.5, {"claims": [], "note": "No claims extracted"}
        
        # Initialize NLI if possible
        use_nli = self._initialize_nli()
        
        # Score each claim
        claim_scores = []
        for claim in claims:
            if use_nli:
                score = self._score_claim_nli(claim, full_context)
            else:
                score = self._score_claim_heuristic(claim, full_context)
            claim_scores.append({"claim": claim, "score": score})
        
        # Average score across claims
        avg_score = sum(cs["score"] for cs in claim_scores) / len(claim_scores)
        
        return avg_score, {
            "claim_scores": claim_scores,
            "method": "nli" if use_nli else "heuristic",
            "num_claims": len(claims),
        }


class RelevancyCalculator(MetricCalculator):
    """Calculates relevancy score measuring context relevance to query.
    
    Uses semantic similarity or keyword overlap to measure how relevant
    the retrieved contexts are to the original question.
    """
    
    def __init__(self):
        """Initialize relevancy calculator."""
        self._model: Optional[Any] = None
        self._initialized = False
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.RELEVANCY
    
    def _initialize_embeddings(self) -> bool:
        """Initialize embedding model for similarity calculation."""
        if self._initialized:
            return self._model is not None
        
        self._initialized = True
        
        try:
            from sentence_transformers import SentenceTransformer
            
            self._model = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("Loaded embedding model for relevancy calculation")
            return True
        except ImportError:
            logger.warning("sentence-transformers not available, using heuristic relevancy")
            return False
        except Exception as exc:
            logger.warning("Failed to load embedding model: %s", exc)
            return False
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _score_relevancy_embeddings(self, question: str, contexts: List[str]) -> float:
        """Score relevancy using embedding similarity."""
        if not self._model:
            return 0.0
        
        try:
            # Encode question and contexts
            question_embedding = self._model.encode([question])[0].tolist()
            context_embeddings = self._model.encode(contexts)
            
            # Calculate similarity for each context
            similarities = []
            for ctx_emb in context_embeddings:
                sim = self._cosine_similarity(question_embedding, ctx_emb.tolist())
                similarities.append(sim)
            
            # Return average similarity
            return sum(similarities) / len(similarities) if similarities else 0.0
        except Exception as exc:
            logger.debug("Embedding relevancy scoring failed: %s", exc)
            return 0.0
    
    def _score_relevancy_heuristic(self, question: str, contexts: List[str]) -> float:
        """Score relevancy using keyword overlap heuristic."""
        # Extract content words from question
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "what", "who",
                     "where", "when", "why", "how", "which", "whom", "whose",
                     "do", "does", "did", "can", "could", "would", "should",
                     "will", "shall", "may", "might", "must", "have", "has",
                     "had", "be", "been", "being", "and", "or", "but", "if",
                     "then", "else", "for", "of", "to", "in", "on", "at", "by",
                     "with", "from", "as", "into", "through", "during", "before",
                     "after", "above", "below", "between", "under", "again",
                     "further", "once", "here", "there", "all", "each", "every",
                     "some", "any", "no", "most", "other", "such", "this", "that",
                     "these", "those", "it", "its", "?"}
        
        question_tokens = set(question.lower().split()) - stopwords
        
        if not question_tokens:
            return 0.5  # Neutral if no content words
        
        # Score each context
        context_scores = []
        for ctx in contexts:
            ctx_tokens = set(ctx.lower().split())
            overlap = question_tokens & ctx_tokens
            score = len(overlap) / len(question_tokens) if question_tokens else 0
            context_scores.append(min(score, 1.0))  # Cap at 1.0
        
        return sum(context_scores) / len(context_scores) if context_scores else 0.0
    
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate relevancy score for a sample."""
        if not sample.question or not sample.contexts:
            return 0.0, {"error": "Missing question or contexts"}
        
        # Initialize embeddings if possible
        use_embeddings = self._initialize_embeddings()
        
        if use_embeddings:
            score = self._score_relevancy_embeddings(sample.question, sample.contexts)
            method = "embeddings"
        else:
            score = self._score_relevancy_heuristic(sample.question, sample.contexts)
            method = "heuristic"
        
        return score, {
            "method": method,
            "num_contexts": len(sample.contexts),
        }


class ContextPrecisionCalculator(MetricCalculator):
    """Calculates context precision score.
    
    Measures the precision of retrieved contexts - what proportion of
    retrieved chunks are actually relevant/useful for answering the question.
    """
    
    def __init__(self):
        """Initialize context precision calculator."""
        self._relevancy_calc = RelevancyCalculator()
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.CONTEXT_PRECISION
    
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate context precision for a sample."""
        if not sample.contexts:
            return 0.0, {"error": "No contexts provided"}
        
        # Initialize the relevancy calculator's model
        self._relevancy_calc._initialize_embeddings()
        
        # Score each context individually
        context_relevance = []
        threshold = 0.3  # Relevance threshold
        
        for i, ctx in enumerate(sample.contexts):
            # Create a single-context sample
            single_sample = EvaluationSample(
                question=sample.question,
                contexts=[ctx],
                answer=sample.answer,
            )
            
            score, _ = self._relevancy_calc.calculate(single_sample)
            is_relevant = score >= threshold
            context_relevance.append({
                "context_idx": i,
                "score": score,
                "is_relevant": is_relevant,
            })
        
        # Precision = relevant / total
        num_relevant = sum(1 for cr in context_relevance if cr["is_relevant"])
        precision = num_relevant / len(context_relevance)
        
        return precision, {
            "context_relevance": context_relevance,
            "num_relevant": num_relevant,
            "total_contexts": len(context_relevance),
            "threshold": threshold,
        }


class ContextRecallCalculator(MetricCalculator):
    """Calculates context recall score.
    
    Measures whether the important information from the ground truth
    answer is present in the retrieved contexts. Requires ground truth.
    """
    
    def __init__(self):
        """Initialize context recall calculator."""
        self._faithfulness_calc = FaithfulnessCalculator()
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.CONTEXT_RECALL
    
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate context recall for a sample."""
        if not sample.ground_truth:
            return 0.0, {"error": "No ground truth provided for recall calculation"}
        
        if not sample.contexts:
            return 0.0, {"error": "No contexts provided"}
        
        # Extract claims from ground truth
        claims = self._faithfulness_calc._extract_claims(sample.ground_truth)
        if not claims:
            return 0.5, {"claims": [], "note": "No claims extracted from ground truth"}
        
        # Initialize NLI if possible
        self._faithfulness_calc._initialize_nli()
        
        # Check which claims are supported by contexts
        full_context = " ".join(sample.contexts)
        supported_claims = []
        
        for claim in claims:
            if self._faithfulness_calc._nli_model:
                score = self._faithfulness_calc._score_claim_nli(claim, full_context)
            else:
                score = self._faithfulness_calc._score_claim_heuristic(claim, full_context)
            
            is_supported = score >= 0.5
            supported_claims.append({
                "claim": claim,
                "score": score,
                "is_supported": is_supported,
            })
        
        # Recall = supported / total claims
        num_supported = sum(1 for sc in supported_claims if sc["is_supported"])
        recall = num_supported / len(claims)
        
        return recall, {
            "supported_claims": supported_claims,
            "num_supported": num_supported,
            "total_claims": len(claims),
        }


class AnswerCorrectnessCalculator(MetricCalculator):
    """Calculates answer correctness score.
    
    Measures how correct the generated answer is compared to the ground truth.
    Uses a combination of semantic similarity and factual overlap.
    """
    
    def __init__(self):
        """Initialize answer correctness calculator."""
        self._faithfulness_calc = FaithfulnessCalculator()
        self._relevancy_calc = RelevancyCalculator()
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.ANSWER_CORRECTNESS
    
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate answer correctness for a sample."""
        if not sample.ground_truth:
            return 0.0, {"error": "No ground truth provided for correctness calculation"}
        
        if not sample.answer:
            return 0.0, {"error": "No answer provided"}
        
        # Calculate semantic similarity
        self._relevancy_calc._initialize_embeddings()
        
        if self._relevancy_calc._model:
            try:
                gt_emb = self._relevancy_calc._model.encode([sample.ground_truth])[0]
                ans_emb = self._relevancy_calc._model.encode([sample.answer])[0]
                semantic_sim = self._relevancy_calc._cosine_similarity(
                    gt_emb.tolist(), ans_emb.tolist()
                )
            except Exception:
                semantic_sim = 0.0
        else:
            # Heuristic semantic similarity via word overlap
            gt_words = set(sample.ground_truth.lower().split())
            ans_words = set(sample.answer.lower().split())
            overlap = gt_words & ans_words
            union = gt_words | ans_words
            semantic_sim = len(overlap) / len(union) if union else 0.0
        
        # Calculate factual overlap using faithfulness logic
        # Check if answer claims are supported by ground truth
        self._faithfulness_calc._initialize_nli()
        
        claims = self._faithfulness_calc._extract_claims(sample.answer)
        if claims:
            claim_scores = []
            for claim in claims:
                if self._faithfulness_calc._nli_model:
                    score = self._faithfulness_calc._score_claim_nli(claim, sample.ground_truth)
                else:
                    score = self._faithfulness_calc._score_claim_heuristic(claim, sample.ground_truth)
                claim_scores.append(score)
            factual_score = sum(claim_scores) / len(claim_scores)
        else:
            factual_score = semantic_sim
        
        # Combine scores (weighted average)
        combined_score = 0.6 * factual_score + 0.4 * semantic_sim
        
        return combined_score, {
            "semantic_similarity": semantic_sim,
            "factual_score": factual_score,
            "weights": {"factual": 0.6, "semantic": 0.4},
        }


class AnswerSimilarityCalculator(MetricCalculator):
    """Calculates answer similarity score.
    
    Pure semantic similarity between generated answer and ground truth.
    Simpler than correctness - just measures how similar the texts are.
    """
    
    def __init__(self):
        """Initialize answer similarity calculator."""
        self._relevancy_calc = RelevancyCalculator()
    
    @property
    def metric_type(self) -> MetricType:
        return MetricType.ANSWER_SIMILARITY
    
    def calculate(self, sample: EvaluationSample) -> Tuple[float, Dict[str, Any]]:
        """Calculate answer similarity for a sample."""
        if not sample.ground_truth:
            return 0.0, {"error": "No ground truth provided"}
        
        if not sample.answer:
            return 0.0, {"error": "No answer provided"}
        
        # Calculate semantic similarity
        self._relevancy_calc._initialize_embeddings()
        
        if self._relevancy_calc._model:
            try:
                gt_emb = self._relevancy_calc._model.encode([sample.ground_truth])[0]
                ans_emb = self._relevancy_calc._model.encode([sample.answer])[0]
                similarity = self._relevancy_calc._cosine_similarity(
                    gt_emb.tolist(), ans_emb.tolist()
                )
                method = "embeddings"
            except Exception:
                similarity = 0.0
                method = "error"
        else:
            # Fallback to Jaccard similarity
            gt_words = set(sample.ground_truth.lower().split())
            ans_words = set(sample.answer.lower().split())
            overlap = gt_words & ans_words
            union = gt_words | ans_words
            similarity = len(overlap) / len(union) if union else 0.0
            method = "jaccard"
        
        return similarity, {"method": method}


class RAGEvaluator:
    """Main evaluator class for RAG system evaluation.
    
    Coordinates multiple metric calculators to evaluate RAG system quality.
    Supports both individual sample evaluation and batch evaluation.
    
    Example:
        evaluator = RAGEvaluator()
        
        sample = EvaluationSample(
            question="What is the capital of France?",
            contexts=["Paris is the capital and largest city of France."],
            answer="The capital of France is Paris.",
            ground_truth="Paris is the capital of France.",
        )
        
        result = evaluator.evaluate(sample)
        print(f"Faithfulness: {result.metrics.faithfulness}")
        print(f"Mean score: {result.metrics.mean_score}")
    """
    
    def __init__(
        self,
        *,
        calculate_faithfulness: bool = True,
        calculate_relevancy: bool = True,
        calculate_context_precision: bool = True,
        calculate_context_recall: bool = True,
        calculate_answer_correctness: bool = True,
        calculate_answer_similarity: bool = True,
    ):
        """Initialize RAG evaluator.
        
        Args:
            calculate_faithfulness: Whether to calculate faithfulness metric.
            calculate_relevancy: Whether to calculate relevancy metric.
            calculate_context_precision: Whether to calculate context precision.
            calculate_context_recall: Whether to calculate context recall.
            calculate_answer_correctness: Whether to calculate answer correctness.
            calculate_answer_similarity: Whether to calculate answer similarity.
        """
        self._calculators: Dict[MetricType, MetricCalculator] = {}
        
        if calculate_faithfulness:
            self._calculators[MetricType.FAITHFULNESS] = FaithfulnessCalculator()
        
        if calculate_relevancy:
            self._calculators[MetricType.RELEVANCY] = RelevancyCalculator()
        
        if calculate_context_precision:
            self._calculators[MetricType.CONTEXT_PRECISION] = ContextPrecisionCalculator()
        
        if calculate_context_recall:
            self._calculators[MetricType.CONTEXT_RECALL] = ContextRecallCalculator()
        
        if calculate_answer_correctness:
            self._calculators[MetricType.ANSWER_CORRECTNESS] = AnswerCorrectnessCalculator()
        
        if calculate_answer_similarity:
            self._calculators[MetricType.ANSWER_SIMILARITY] = AnswerSimilarityCalculator()
    
    def evaluate(self, sample: EvaluationSample) -> EvaluationResult:
        """Evaluate a single sample.
        
        Args:
            sample: The evaluation sample.
            
        Returns:
            EvaluationResult with computed metrics.
        """
        metrics = EvaluationMetrics()
        details: Dict[str, Any] = {}
        errors: List[str] = []
        
        for metric_type, calculator in self._calculators.items():
            try:
                score, metric_details = calculator.calculate(sample)
                
                # Set the appropriate metric
                if metric_type == MetricType.FAITHFULNESS:
                    metrics.faithfulness = score
                elif metric_type == MetricType.RELEVANCY:
                    metrics.relevancy = score
                elif metric_type == MetricType.CONTEXT_PRECISION:
                    metrics.context_precision = score
                elif metric_type == MetricType.CONTEXT_RECALL:
                    metrics.context_recall = score
                elif metric_type == MetricType.ANSWER_CORRECTNESS:
                    metrics.answer_correctness = score
                elif metric_type == MetricType.ANSWER_SIMILARITY:
                    metrics.answer_similarity = score
                
                details[metric_type.value] = metric_details
                
            except Exception as exc:
                logger.error("Error calculating %s: %s", metric_type.value, exc)
                errors.append(f"{metric_type.value}: {exc}")
        
        return EvaluationResult(
            sample=sample,
            metrics=metrics,
            details=details,
            error="; ".join(errors) if errors else None,
        )
    
    def evaluate_batch(
        self,
        samples: List[EvaluationSample],
        *,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[EvaluationResult]:
        """Evaluate a batch of samples.
        
        Args:
            samples: List of evaluation samples.
            progress_callback: Optional callback(current, total) for progress.
            
        Returns:
            List of EvaluationResult for each sample.
        """
        results = []
        total = len(samples)
        
        for i, sample in enumerate(samples):
            result = self.evaluate(sample)
            results.append(result)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    def aggregate_results(
        self,
        results: List[EvaluationResult],
    ) -> Dict[str, Any]:
        """Aggregate results from batch evaluation.
        
        Args:
            results: List of evaluation results.
            
        Returns:
            Dictionary with aggregated statistics.
        """
        if not results:
            return {"error": "No results to aggregate"}
        
        # Collect all metric values
        metric_values: Dict[str, List[float]] = {
            "faithfulness": [],
            "relevancy": [],
            "context_precision": [],
            "context_recall": [],
            "answer_correctness": [],
            "answer_similarity": [],
        }
        
        successful = 0
        failed = 0
        
        for result in results:
            if result.is_success:
                successful += 1
                metrics = result.metrics.to_dict()
                for key, value in metrics.items():
                    if value is not None:
                        metric_values[key].append(value)
            else:
                failed += 1
        
        # Calculate statistics
        aggregated: Dict[str, Any] = {
            "total_samples": len(results),
            "successful": successful,
            "failed": failed,
            "metrics": {},
        }
        
        for metric_name, values in metric_values.items():
            if values:
                aggregated["metrics"][metric_name] = {
                    "mean": sum(values) / len(values),
                    "min": min(values),
                    "max": max(values),
                    "std": self._std_dev(values),
                    "count": len(values),
                }
        
        # Overall mean score
        all_means = [
            stats["mean"]
            for stats in aggregated["metrics"].values()
            if stats.get("mean") is not None
        ]
        if all_means:
            aggregated["overall_mean"] = sum(all_means) / len(all_means)
        
        return aggregated
    
    def _std_dev(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)


__all__ = [
    "MetricType",
    "EvaluationSample",
    "EvaluationMetrics",
    "EvaluationResult",
    "MetricCalculator",
    "FaithfulnessCalculator",
    "RelevancyCalculator",
    "ContextPrecisionCalculator",
    "ContextRecallCalculator",
    "AnswerCorrectnessCalculator",
    "AnswerSimilarityCalculator",
    "RAGEvaluator",
]
