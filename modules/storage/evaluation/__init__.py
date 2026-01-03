"""RAG evaluation module.

Provides evaluation metrics and harness for measuring RAG system quality.
Implements RAGAS-style metrics for faithfulness, relevancy, and context precision.
"""

from modules.storage.evaluation.evaluator import (
    RAGEvaluator,
    EvaluationResult,
    EvaluationMetrics,
    EvaluationSample,
)
from modules.storage.evaluation.dataset import (
    EvaluationDataset,
    DatasetLoader,
    load_dataset_from_file,
    save_dataset_to_file,
    create_sample_dataset,
)

__all__ = [
    # Evaluator
    "RAGEvaluator",
    "EvaluationResult",
    "EvaluationMetrics",
    "EvaluationSample",
    # Dataset
    "EvaluationDataset",
    "DatasetLoader",
    "load_dataset_from_file",
    "save_dataset_to_file",
    "create_sample_dataset",
]
