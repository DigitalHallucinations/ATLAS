"""Evaluation dataset loader.

Provides utilities for loading and managing evaluation datasets
containing question/answer/context triples for RAG evaluation.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union
import csv
import json
import logging

from modules.storage.evaluation.evaluator import EvaluationSample

logger = logging.getLogger(__name__)


@dataclass
class EvaluationDataset:
    """Container for evaluation samples.
    
    Attributes:
        name: Dataset name/identifier.
        samples: List of evaluation samples.
        metadata: Additional dataset metadata.
    """
    
    name: str
    samples: List[EvaluationSample] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)
    
    def __iter__(self) -> Iterator[EvaluationSample]:
        """Iterate over samples."""
        return iter(self.samples)
    
    def __getitem__(self, index: int) -> EvaluationSample:
        """Get sample by index."""
        return self.samples[index]
    
    def add_sample(self, sample: EvaluationSample) -> None:
        """Add a sample to the dataset."""
        self.samples.append(sample)
    
    def filter(
        self,
        *,
        has_ground_truth: Optional[bool] = None,
        min_contexts: Optional[int] = None,
        max_contexts: Optional[int] = None,
    ) -> "EvaluationDataset":
        """Filter samples based on criteria.
        
        Args:
            has_ground_truth: Filter by ground truth presence.
            min_contexts: Minimum number of contexts required.
            max_contexts: Maximum number of contexts allowed.
            
        Returns:
            New filtered dataset.
        """
        filtered_samples = []
        
        for sample in self.samples:
            # Check ground truth filter
            if has_ground_truth is not None:
                sample_has_gt = sample.ground_truth is not None
                if sample_has_gt != has_ground_truth:
                    continue
            
            # Check context count filters
            num_contexts = len(sample.contexts)
            if min_contexts is not None and num_contexts < min_contexts:
                continue
            if max_contexts is not None and num_contexts > max_contexts:
                continue
            
            filtered_samples.append(sample)
        
        return EvaluationDataset(
            name=f"{self.name}_filtered",
            samples=filtered_samples,
            metadata={**self.metadata, "filtered": True},
        )
    
    def split(
        self,
        ratio: float = 0.8,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ) -> tuple["EvaluationDataset", "EvaluationDataset"]:
        """Split dataset into train/test sets.
        
        Args:
            ratio: Proportion for first split (default 0.8).
            shuffle: Whether to shuffle before splitting.
            seed: Random seed for reproducibility.
            
        Returns:
            Tuple of (train_dataset, test_dataset).
        """
        samples = self.samples.copy()
        
        if shuffle:
            import random
            if seed is not None:
                random.seed(seed)
            random.shuffle(samples)
        
        split_idx = int(len(samples) * ratio)
        
        train_samples = samples[:split_idx]
        test_samples = samples[split_idx:]
        
        return (
            EvaluationDataset(
                name=f"{self.name}_train",
                samples=train_samples,
                metadata={**self.metadata, "split": "train"},
            ),
            EvaluationDataset(
                name=f"{self.name}_test",
                samples=test_samples,
                metadata={**self.metadata, "split": "test"},
            ),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "name": self.name,
            "samples": [
                {
                    "question": s.question,
                    "contexts": s.contexts,
                    "answer": s.answer,
                    "ground_truth": s.ground_truth,
                    "metadata": s.metadata,
                }
                for s in self.samples
            ],
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EvaluationDataset":
        """Create from dictionary representation."""
        samples = [
            EvaluationSample(
                question=s["question"],
                contexts=s.get("contexts", []),
                answer=s.get("answer", ""),
                ground_truth=s.get("ground_truth"),
                metadata=s.get("metadata", {}),
            )
            for s in data.get("samples", [])
        ]
        
        return cls(
            name=data.get("name", "unnamed"),
            samples=samples,
            metadata=data.get("metadata", {}),
        )


class DatasetLoader:
    """Loader for evaluation datasets from various formats.
    
    Supports:
    - JSON files with question/context/answer/ground_truth fields
    - JSONL files (one sample per line)
    - CSV files with appropriate columns
    - RAGAS-style datasets
    - Custom format mappings
    """
    
    def __init__(
        self,
        *,
        question_field: str = "question",
        contexts_field: str = "contexts",
        answer_field: str = "answer",
        ground_truth_field: str = "ground_truth",
        metadata_fields: Optional[List[str]] = None,
    ):
        """Initialize dataset loader.
        
        Args:
            question_field: Field name for questions.
            contexts_field: Field name for contexts.
            answer_field: Field name for answers.
            ground_truth_field: Field name for ground truth.
            metadata_fields: Additional fields to include in metadata.
        """
        self.question_field = question_field
        self.contexts_field = contexts_field
        self.answer_field = answer_field
        self.ground_truth_field = ground_truth_field
        self.metadata_fields = metadata_fields or []
    
    def load_json(self, path: Union[str, Path]) -> EvaluationDataset:
        """Load dataset from JSON file.
        
        Expected format:
        {
            "name": "dataset_name",
            "samples": [
                {
                    "question": "...",
                    "contexts": ["..."],
                    "answer": "...",
                    "ground_truth": "..."
                }
            ]
        }
        
        Or just a list of samples:
        [
            {"question": "...", "contexts": [...], ...}
        ]
        """
        path = Path(path)
        
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Handle list format
        if isinstance(data, list):
            data = {"name": path.stem, "samples": data}
        
        # Handle samples list
        samples = []
        raw_samples = data.get("samples", data.get("data", []))
        
        for item in raw_samples:
            sample = self._parse_sample(item)
            if sample:
                samples.append(sample)
        
        return EvaluationDataset(
            name=data.get("name", path.stem),
            samples=samples,
            metadata=data.get("metadata", {"source": str(path)}),
        )
    
    def load_jsonl(self, path: Union[str, Path]) -> EvaluationDataset:
        """Load dataset from JSONL file (one JSON object per line)."""
        path = Path(path)
        samples = []
        
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    item = json.loads(line)
                    sample = self._parse_sample(item)
                    if sample:
                        samples.append(sample)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse line %d: %s", line_num, exc)
        
        return EvaluationDataset(
            name=path.stem,
            samples=samples,
            metadata={"source": str(path), "format": "jsonl"},
        )
    
    def load_csv(
        self,
        path: Union[str, Path],
        *,
        delimiter: str = ",",
        context_separator: str = "|||",
    ) -> EvaluationDataset:
        """Load dataset from CSV file.
        
        Args:
            path: Path to CSV file.
            delimiter: CSV delimiter character.
            context_separator: Separator for multiple contexts in one field.
        """
        path = Path(path)
        samples = []
        
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            
            for row in reader:
                # Parse contexts (may be separated or single)
                contexts_raw = row.get(self.contexts_field, "")
                if context_separator in contexts_raw:
                    contexts = [c.strip() for c in contexts_raw.split(context_separator)]
                else:
                    contexts = [contexts_raw] if contexts_raw else []
                
                # Build metadata from additional fields
                metadata = {}
                for field in self.metadata_fields:
                    if field in row:
                        metadata[field] = row[field]
                
                sample = EvaluationSample(
                    question=row.get(self.question_field, ""),
                    contexts=contexts,
                    answer=row.get(self.answer_field, ""),
                    ground_truth=row.get(self.ground_truth_field),
                    metadata=metadata,
                )
                
                if sample.question:  # Only add if has question
                    samples.append(sample)
        
        return EvaluationDataset(
            name=path.stem,
            samples=samples,
            metadata={"source": str(path), "format": "csv"},
        )
    
    def load_ragas_format(self, path: Union[str, Path]) -> EvaluationDataset:
        """Load dataset in RAGAS format.
        
        RAGAS format uses:
        - question: The query
        - contexts: List of context strings
        - answer: Generated answer
        - ground_truth: Expected answer (optional)
        """
        # RAGAS uses standard field names, just load as JSON
        original_fields = (
            self.question_field,
            self.contexts_field,
            self.answer_field,
            self.ground_truth_field,
        )
        
        try:
            # Set RAGAS field names
            self.question_field = "question"
            self.contexts_field = "contexts"
            self.answer_field = "answer"
            self.ground_truth_field = "ground_truth"
            
            return self.load_json(path)
        finally:
            # Restore original field names
            (
                self.question_field,
                self.contexts_field,
                self.answer_field,
                self.ground_truth_field,
            ) = original_fields
    
    def load(self, path: Union[str, Path]) -> EvaluationDataset:
        """Auto-detect format and load dataset.
        
        Args:
            path: Path to dataset file.
            
        Returns:
            Loaded dataset.
        """
        path = Path(path)
        suffix = path.suffix.lower()
        
        if suffix == ".json":
            return self.load_json(path)
        elif suffix == ".jsonl":
            return self.load_jsonl(path)
        elif suffix == ".csv":
            return self.load_csv(path)
        else:
            # Try JSON as default
            try:
                return self.load_json(path)
            except Exception:
                raise ValueError(f"Unsupported file format: {suffix}")
    
    def _parse_sample(self, item: Dict[str, Any]) -> Optional[EvaluationSample]:
        """Parse a single sample from a dictionary."""
        question = item.get(self.question_field, "")
        if not question:
            return None
        
        # Parse contexts - may be string or list
        contexts_raw = item.get(self.contexts_field, [])
        if isinstance(contexts_raw, str):
            contexts = [contexts_raw] if contexts_raw else []
        else:
            contexts = list(contexts_raw)
        
        # Build metadata
        metadata = {}
        for field in self.metadata_fields:
            if field in item:
                metadata[field] = item[field]
        
        return EvaluationSample(
            question=question,
            contexts=contexts,
            answer=item.get(self.answer_field, ""),
            ground_truth=item.get(self.ground_truth_field),
            metadata=metadata,
        )


def load_dataset_from_file(
    path: Union[str, Path],
    *,
    question_field: str = "question",
    contexts_field: str = "contexts",
    answer_field: str = "answer",
    ground_truth_field: str = "ground_truth",
) -> EvaluationDataset:
    """Convenience function to load a dataset from file.
    
    Args:
        path: Path to dataset file.
        question_field: Field name for questions.
        contexts_field: Field name for contexts.
        answer_field: Field name for answers.
        ground_truth_field: Field name for ground truth.
        
    Returns:
        Loaded EvaluationDataset.
    """
    loader = DatasetLoader(
        question_field=question_field,
        contexts_field=contexts_field,
        answer_field=answer_field,
        ground_truth_field=ground_truth_field,
    )
    return loader.load(path)


def save_dataset_to_file(
    dataset: EvaluationDataset,
    path: Union[str, Path],
    *,
    format: str = "json",
    indent: int = 2,
) -> None:
    """Save dataset to file.
    
    Args:
        dataset: Dataset to save.
        path: Output path.
        format: Output format ("json" or "jsonl").
        indent: JSON indentation (for json format).
    """
    path = Path(path)
    
    if format == "json":
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=indent, ensure_ascii=False)
    
    elif format == "jsonl":
        with open(path, "w", encoding="utf-8") as f:
            for sample in dataset.samples:
                line = json.dumps({
                    "question": sample.question,
                    "contexts": sample.contexts,
                    "answer": sample.answer,
                    "ground_truth": sample.ground_truth,
                    "metadata": sample.metadata,
                }, ensure_ascii=False)
                f.write(line + "\n")
    
    else:
        raise ValueError(f"Unsupported format: {format}")


def create_sample_dataset() -> EvaluationDataset:
    """Create a sample evaluation dataset for testing.
    
    Returns:
        A small sample dataset with example Q&A pairs.
    """
    samples = [
        EvaluationSample(
            question="What is the capital of France?",
            contexts=[
                "Paris is the capital and largest city of France. "
                "It is situated on the river Seine, in northern France.",
                "France is a country in Western Europe with several overseas regions."
            ],
            answer="The capital of France is Paris.",
            ground_truth="Paris is the capital of France.",
        ),
        EvaluationSample(
            question="Who wrote Romeo and Juliet?",
            contexts=[
                "Romeo and Juliet is a tragedy written by William Shakespeare early in his career.",
                "The play was first published in 1597."
            ],
            answer="Romeo and Juliet was written by William Shakespeare.",
            ground_truth="William Shakespeare wrote Romeo and Juliet.",
        ),
        EvaluationSample(
            question="What is photosynthesis?",
            contexts=[
                "Photosynthesis is a process used by plants and other organisms to convert "
                "light energy into chemical energy that can be stored and later released.",
                "The process occurs in chloroplasts and uses chlorophyll to absorb sunlight."
            ],
            answer="Photosynthesis is the process by which plants convert light energy "
                   "into chemical energy using chlorophyll.",
            ground_truth="Photosynthesis is the process plants use to convert sunlight "
                        "into chemical energy for storage.",
        ),
    ]
    
    return EvaluationDataset(
        name="sample_dataset",
        samples=samples,
        metadata={
            "description": "Sample evaluation dataset for testing",
            "created": "2025-01-03",
        },
    )


__all__ = [
    "EvaluationDataset",
    "DatasetLoader",
    "load_dataset_from_file",
    "save_dataset_to_file",
    "create_sample_dataset",
]
