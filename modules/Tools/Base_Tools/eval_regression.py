"""Regression evaluation tool comparing baseline and candidate artifacts.

The tool computes textual diffs and similarity metrics, applies optional
thresholds, and publishes a structured payload to the analytics event bus.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher, unified_diff
import hashlib
from typing import Any, Mapping, Sequence
from collections.abc import Mapping as MappingABC, Sequence as SequenceABC

from modules.Tools.tool_event_system import publish_bus_event

__all__ = ["eval_regression"]


_DEFAULT_EVENT_NAME = "atlas.evaluation.regression"
_DEFAULT_DIFF_CONTEXT = 3
_PREVIEW_LENGTH = 256


@dataclass
class _MetricsConfig:
    include: set[str]
    thresholds: dict[str, Any]
    required: set[str]


def eval_regression(
    *,
    baseline: Mapping[str, Any] | str,
    candidate: Mapping[str, Any] | str,
    artifact_store: Mapping[str, Mapping[str, Any]] | None = None,
    diff: Mapping[str, Any] | int | None = None,
    metrics: Mapping[str, Any] | Sequence[Any] | None = None,
    metadata: Mapping[str, Any] | None = None,
    event_name: str | None = None,
) -> Mapping[str, Any]:
    """Compare *baseline* and *candidate* artifacts and emit regression metrics."""

    baseline_artifact = _resolve_artifact(baseline, artifact_store, role="baseline")
    candidate_artifact = _resolve_artifact(candidate, artifact_store, role="candidate")

    diff_config = _normalize_diff_config(diff)
    metrics_config = _normalize_metrics_config(metrics)

    baseline_text = _extract_artifact_text(baseline_artifact, role="baseline")
    candidate_text = _extract_artifact_text(candidate_artifact, role="candidate")

    diff_lines = _compute_diff_lines(
        baseline_text,
        candidate_text,
        diff_config["context"],
        baseline_artifact.get("id") or baseline_artifact.get("artifact_id") or "baseline",
        candidate_artifact.get("id") or candidate_artifact.get("artifact_id") or "candidate",
    )
    diff_text = "\n".join(diff_lines)

    metrics_payload = _compute_metrics(
        baseline_text,
        candidate_text,
        diff_lines,
        config=metrics_config,
    )
    thresholds_payload = _evaluate_thresholds(metrics_payload, config=metrics_config)
    summary_payload = {
        "identical": bool(metrics_payload.get("identical")),
        "regressed": any(not entry["passed"] for entry in thresholds_payload.values()),
        "thresholds": thresholds_payload,
    }

    normalized_metadata = _ensure_mapping(metadata, path="metadata")

    payload = {
        "baseline": _summarize_artifact(baseline_artifact, baseline_text),
        "candidate": _summarize_artifact(candidate_artifact, candidate_text),
        "diff": {
            "format": diff_config["format"],
            "context": diff_config["context"],
            "lines": diff_lines,
            "text": diff_text,
        },
        "metrics": metrics_payload,
        "summary": summary_payload,
        "metadata": normalized_metadata,
    }

    topic = _normalize_event_name(event_name)
    correlation = publish_bus_event(topic, payload)

    return {
        "event": topic,
        "correlation_id": correlation,
        "baseline": payload["baseline"],
        "candidate": payload["candidate"],
        "metrics": metrics_payload,
        "diff": payload["diff"],
        "summary": summary_payload,
        "metadata": normalized_metadata,
    }


def _normalize_event_name(event_name: str | None) -> str:
    if isinstance(event_name, str) and event_name.strip():
        return event_name.strip()
    return _DEFAULT_EVENT_NAME


def _resolve_artifact(
    source: Mapping[str, Any] | str,
    artifact_store: Mapping[str, Mapping[str, Any]] | None,
    *,
    role: str,
) -> dict[str, Any]:
    if isinstance(source, MappingABC):
        artifact = {str(key): value for key, value in source.items()}
    elif isinstance(source, str):
        if not artifact_store:
            raise ValueError(f"artifact_store is required when {role} is a string identifier")
        try:
            stored = artifact_store[source]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise KeyError(f"Unknown {role} artifact identifier '{source}'") from exc
        if not isinstance(stored, MappingABC):
            raise TypeError(f"artifact_store entry for '{source}' must be a mapping")
        artifact = {str(key): value for key, value in stored.items()}
        artifact.setdefault("id", source)
    else:
        raise TypeError(f"{role} must be a mapping or string identifier")

    if "id" not in artifact and "artifact_id" in artifact:
        artifact["id"] = artifact.get("artifact_id")

    return artifact


def _extract_artifact_text(artifact: Mapping[str, Any], *, role: str) -> str:
    for key in ("content", "text", "value", "data"):
        if key in artifact and artifact[key] is not None:
            value = artifact[key]
            if not isinstance(value, str):
                raise TypeError(f"{role}.{key} must be a string")
            return value
    raise ValueError(f"{role} artifact does not contain textual content")


def _normalize_diff_config(config: Mapping[str, Any] | int | None) -> dict[str, Any]:
    if config is None:
        return {"format": "unified", "context": _DEFAULT_DIFF_CONTEXT}

    if isinstance(config, int):
        if config < 0:
            raise ValueError("diff context must be non-negative")
        return {"format": "unified", "context": config}

    if not isinstance(config, MappingABC):
        raise TypeError("diff configuration must be a mapping or integer")

    format_name = config.get("format", "unified")
    if format_name != "unified":
        raise ValueError("Only 'unified' diff format is supported")

    context_value = config.get("context", config.get("n", _DEFAULT_DIFF_CONTEXT))
    if context_value is None:
        context = _DEFAULT_DIFF_CONTEXT
    else:
        if not isinstance(context_value, int):
            raise TypeError("diff context must be an integer")
        if context_value < 0:
            raise ValueError("diff context must be non-negative")
        context = context_value

    return {"format": "unified", "context": context}


def _normalize_metrics_config(config: Mapping[str, Any] | Sequence[Any] | None) -> _MetricsConfig:
    include: set[str] = {"similarity", "added_lines", "removed_lines", "total_changes", "identical"}
    thresholds: dict[str, Any] = {}

    if config is None:
        required = {"identical"}
    elif isinstance(config, SequenceABC) and not isinstance(config, (str, bytes, bytearray)):
        include = {str(name) for name in config}
        required = {"identical"}
    elif isinstance(config, MappingABC):
        required = {"identical"}
        includes = config.get("include")
        if includes is not None:
            if not isinstance(includes, SequenceABC) or isinstance(includes, (str, bytes, bytearray)):
                raise TypeError("metrics.include must be a sequence of names")
            include = {str(name) for name in includes}
        thresholds_value = config.get("thresholds")
        if thresholds_value is not None:
            if not isinstance(thresholds_value, MappingABC):
                raise TypeError("metrics.thresholds must be a mapping")
            thresholds = {str(key): value for key, value in thresholds_value.items()}
    else:
        raise TypeError("metrics must be a mapping, sequence, or None")

    required.update(_metrics_required_by_thresholds(thresholds))
    include.update(required)
    return _MetricsConfig(include=include, thresholds=thresholds, required=required)


def _metrics_required_by_thresholds(thresholds: Mapping[str, Any]) -> set[str]:
    required: set[str] = set()
    for name in thresholds:
        metric_name, _ = _parse_threshold_name(name)
        required.add(metric_name)
    return required


def _compute_diff_lines(
    baseline_text: str,
    candidate_text: str,
    context: int,
    baseline_label: str,
    candidate_label: str,
) -> list[str]:
    return list(
        unified_diff(
            baseline_text.splitlines(),
            candidate_text.splitlines(),
            fromfile=baseline_label,
            tofile=candidate_label,
            n=context,
            lineterm="",
        )
    )


def _compute_metrics(
    baseline_text: str,
    candidate_text: str,
    diff_lines: Sequence[str],
    *,
    config: _MetricsConfig,
) -> dict[str, Any]:
    matcher = SequenceMatcher(None, baseline_text, candidate_text)
    similarity = matcher.ratio()
    added, removed = _count_diff_changes(diff_lines)
    baseline_lines = baseline_text.splitlines()
    candidate_lines = candidate_text.splitlines()

    all_metrics = {
        "similarity": similarity,
        "added_lines": added,
        "removed_lines": removed,
        "total_changes": added + removed,
        "baseline_lines": len(baseline_lines),
        "candidate_lines": len(candidate_lines),
        "identical": baseline_text == candidate_text,
    }

    metrics: dict[str, Any] = {}
    for name in config.include:
        if name in all_metrics:
            metrics[name] = all_metrics[name]

    return metrics


def _count_diff_changes(diff_lines: Sequence[str]) -> tuple[int, int]:
    added = 0
    removed = 0
    for line in diff_lines:
        if not line:
            continue
        if line.startswith("+++") or line.startswith("---") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def _evaluate_thresholds(
    metrics: Mapping[str, Any],
    *,
    config: _MetricsConfig,
) -> dict[str, Mapping[str, Any]]:
    normalized: dict[str, Mapping[str, Any]] = {}
    for name, expected in config.thresholds.items():
        metric_name, mode = _parse_threshold_name(name)
        if metric_name not in metrics:
            raise ValueError(f"Metric '{metric_name}' required for threshold '{name}' is missing")
        value = metrics[metric_name]
        if not isinstance(expected, (int, float)):
            raise TypeError(f"Threshold '{name}' must be a numeric value")
        if mode == "min":
            passed = value >= expected
            comparator = ">="
        else:
            passed = value <= expected
            comparator = "<="
        normalized[name] = {
            "metric": metric_name,
            "expected": float(expected),
            "comparison": comparator,
            "value": value,
            "passed": bool(passed),
        }
    return normalized


def _parse_threshold_name(name: str) -> tuple[str, str]:
    if not isinstance(name, str) or not name:
        raise TypeError("Threshold names must be non-empty strings")
    if name.startswith("min_"):
        return name[4:], "min"
    if name.startswith("max_"):
        return name[4:], "max"
    raise ValueError("Threshold names must begin with 'min_' or 'max_'")


def _summarize_artifact(artifact: Mapping[str, Any], text: str) -> dict[str, Any]:
    identifier = artifact.get("id") or artifact.get("artifact_id")
    label = artifact.get("label") or artifact.get("name")
    checksum = hashlib.sha256(text.encode("utf-8")).hexdigest()
    preview = text if len(text) <= _PREVIEW_LENGTH else text[:_PREVIEW_LENGTH] + "\u2026"
    summary = {
        "id": identifier,
        "label": label,
        "length": len(text),
        "checksum": checksum,
        "preview": preview,
    }
    return summary


def _ensure_mapping(value: Mapping[str, Any] | None, *, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, MappingABC):
        raise TypeError(f"{path} must be a mapping")
    return {str(key): item for key, item in value.items()}
