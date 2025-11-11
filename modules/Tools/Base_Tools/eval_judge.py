"""Evaluation orchestration tool for candidate completions.

The tool accepts candidate outputs, reference solutions, and rubric guidance
and then coordinates one or more evaluator backends. Each evaluator receives
normalized payloads and returns a structured result that is aggregated into a
summary. The tool also emits an analytics event via :func:`publish_bus_event`
and forwards the aggregated payload to any provided analytics hooks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, MutableMapping, Sequence

from modules.Tools.tool_event_system import publish_bus_event

from ._evaluation_utils import _normalize_event_name

__all__ = ["eval_judge", "register_evaluator", "EvaluatorNotFoundError"]


_DEFAULT_EVENT_NAME = "atlas.evaluation.judged"


class EvaluatorNotFoundError(RuntimeError):
    """Raised when an evaluator backend cannot be resolved."""


@dataclass(frozen=True)
class _EvaluatorSpec:
    name: str
    options: Mapping[str, Any]
    threshold: float | None


_EVALUATOR_REGISTRY: dict[str, Callable[..., Mapping[str, Any]]] = {}


def register_evaluator(name: str, evaluator: Callable[..., Mapping[str, Any]]) -> None:
    """Register an evaluator callable for lookup by :func:`eval_judge`.

    Args:
        name: Name used when referencing the evaluator from tool invocations.
        evaluator: Callable that accepts keyword arguments ``candidates``,
            ``references``, ``rubrics``, ``options``, ``metadata``, and
            ``context`` and returns a mapping describing the evaluation result.
    """

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Evaluator name must be a non-empty string")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable")
    key = name.strip().casefold()
    _EVALUATOR_REGISTRY[key] = evaluator


def eval_judge(
    *,
    candidates: Sequence[Any] | Mapping[str, Any],
    references: Sequence[Any] | Mapping[str, Any] | None = None,
    rubrics: Sequence[Any] | Mapping[str, Any] | None = None,
    evaluators: Sequence[Any] | Mapping[str, Any] | str | None = None,
    thresholds: Mapping[str, Any] | float | int | None = None,
    metadata: Mapping[str, Any] | None = None,
    analytics: Any | None = None,
    context: Mapping[str, Any] | None = None,
    event_name: str | None = None,
) -> Mapping[str, Any]:
    """Execute configured evaluators and aggregate their results.

    Args:
        candidates: Sequence of candidate outputs. Each entry can be a string or
            a mapping with ``output`` and optional metadata fields.
        references: Optional reference answers that evaluators can use as
            ground truth.
        rubrics: Optional rubric descriptors guiding evaluator scoring.
        evaluators: List of evaluator names or configuration mappings. When
            omitted the tool attempts to use the evaluators provided in
            ``context['evaluators']``.
        thresholds: Global and/or per-evaluator thresholds used when
            determining pass/fail outcomes.
        metadata: Additional metadata forwarded to evaluators and analytics.
        analytics: Optional callable or nested collection of callables invoked
            with the aggregated evaluation payload.
        context: Optional mapping containing helper objects or evaluator
            overrides. If ``context['evaluators']`` is a mapping it will be used
            for evaluator resolution before consulting the shared registry.
        event_name: Optional event topic override for the published analytics
            payload. Defaults to ``atlas.evaluation.judged``.

    Returns:
        Mapping describing aggregated evaluator results, applied thresholds, and
        associated analytics metadata. The mapping includes the correlation
        identifier returned by :func:`publish_bus_event`.
    """

    if candidates is None:
        raise ValueError("candidates must be provided")

    normalized_candidates = _normalize_records(candidates, kind="candidate")
    normalized_references = _normalize_records(references, kind="reference")
    normalized_rubrics = _normalize_rubrics(rubrics)
    normalized_metadata = _ensure_mapping(metadata, path="metadata")
    normalized_context = _ensure_mapping(context, path="context")

    evaluator_specs = _normalize_evaluators(evaluators, normalized_context)
    if not evaluator_specs:
        raise ValueError("At least one evaluator must be specified")

    threshold_lookup, applied_defaults = _resolve_thresholds(
        thresholds, evaluator_specs
    )

    analytics_hooks = _collect_analytics_hooks(analytics)

    results: list[dict[str, Any]] = []
    applied_thresholds: MutableMapping[str, float | None] = {}
    passed_count = 0

    for spec in evaluator_specs:
        evaluator = _resolve_evaluator(spec.name, normalized_context)
        try:
            outcome = evaluator(
                candidates=normalized_candidates,
                references=normalized_references,
                rubrics=normalized_rubrics,
                options=spec.options,
                metadata=normalized_metadata,
                context=normalized_context,
            )
        except Exception as exc:  # pylint: disable=broad-except
            normalized_result = {
                "name": spec.name,
                "score": None,
                "details": {},
                "error": str(exc),
                "threshold": None,
                "passed": False,
            }
        else:
            normalized_result = _normalize_outcome(outcome)
            threshold = spec.threshold
            if threshold is None:
                threshold = threshold_lookup.get(spec.name)
            applied_thresholds[spec.name] = threshold
            passed = _passes_threshold(normalized_result["score"], threshold)
            if passed:
                passed_count += 1
            normalized_result.update(
                {
                    "name": spec.name,
                    "threshold": threshold,
                    "passed": passed,
                }
            )
            if spec.options:
                normalized_result["options"] = dict(spec.options)
        results.append(normalized_result)

    total = len(results)
    failed_count = total - passed_count
    summary = {
        "total": total,
        "passed": passed_count,
        "failed": failed_count,
        "thresholds": dict(applied_thresholds),
        "defaults": applied_defaults,
    }
    overall_passed = failed_count == 0

    payload = {
        "candidates": normalized_candidates,
        "references": normalized_references,
        "rubrics": normalized_rubrics,
        "metadata": normalized_metadata,
        "evaluators": [spec.name for spec in evaluator_specs],
        "results": results,
        "summary": summary,
        "passed": overall_passed,
    }

    topic = _normalize_event_name(event_name, default=_DEFAULT_EVENT_NAME)
    correlation = publish_bus_event(topic, payload)

    for hook in analytics_hooks:
        try:
            hook(_clone_payload(payload))
        except Exception:  # pragma: no cover - analytics hooks are best effort
            continue

    return {
        "event": topic,
        "correlation_id": correlation,
        "results": results,
        "summary": summary,
        "metadata": normalized_metadata,
        "passed": overall_passed,
    }
def _normalize_records(
    entries: Sequence[Any] | Mapping[str, Any] | None,
    *,
    kind: str,
) -> list[dict[str, Any]]:
    if entries is None:
        return []
    if isinstance(entries, Mapping):
        sequence: Sequence[Any] = [entries]
    elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
        sequence = entries
    else:
        raise TypeError(f"{kind}s must be provided as a mapping or sequence")

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(sequence):
        if isinstance(entry, Mapping):
            record = {str(key): value for key, value in entry.items()}
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                identifier = f"{kind}-{index}"
            record["id"] = identifier
            output = record.get("output")
            if output is None:
                for fallback in ("text", "value", "content"):
                    if fallback in record:
                        output = record[fallback]
                        break
            if output is None:
                output = ""
            record["output"] = output
            record["metadata"] = _ensure_mapping(record.get("metadata"), path=f"{kind}[{index}].metadata")
            normalized.append(record)
        else:
            normalized.append(
                {
                    "id": f"{kind}-{index}",
                    "output": entry,
                    "metadata": {},
                }
            )
    return normalized


def _normalize_rubrics(entries: Sequence[Any] | Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if entries is None:
        return []
    if isinstance(entries, Mapping):
        sequence: Sequence[Any] = [entries]
    elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
        sequence = entries
    else:
        return [
            {
                "id": "rubric-0",
                "description": str(entries),
            }
        ]

    normalized: list[dict[str, Any]] = []
    for index, entry in enumerate(sequence):
        if isinstance(entry, Mapping):
            record = {str(key): value for key, value in entry.items()}
            identifier = record.get("id")
            if not isinstance(identifier, str) or not identifier.strip():
                identifier = f"rubric-{index}"
            record["id"] = identifier
            if "description" not in record and "text" in record:
                record["description"] = record["text"]
            normalized.append(record)
        else:
            normalized.append(
                {
                    "id": f"rubric-{index}",
                    "description": str(entry),
                }
            )
    return normalized


def _normalize_evaluators(
    raw: Sequence[Any] | Mapping[str, Any] | str | None,
    context: Mapping[str, Any],
) -> list[_EvaluatorSpec]:
    if raw is None:
        raw = context.get("evaluators")
    if raw is None:
        return []
    if isinstance(raw, str):
        items: Sequence[Any] = [raw]
    elif isinstance(raw, Mapping):
        items = [raw]
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        items = raw
    else:
        raise TypeError("evaluators must be a string, mapping, or sequence")

    specs: list[_EvaluatorSpec] = []
    for index, entry in enumerate(items):
        if isinstance(entry, str):
            name = entry.strip()
            if not name:
                raise ValueError(f"Evaluator at position {index} is empty")
            specs.append(_EvaluatorSpec(name=name, options={}, threshold=None))
            continue
        if not isinstance(entry, Mapping):
            raise TypeError("Evaluator entries must be strings or mappings")
        name_value = entry.get("name")
        if not isinstance(name_value, str) or not name_value.strip():
            raise ValueError("Evaluator mappings must include a non-empty 'name'")
        options = _ensure_mapping(entry.get("options"), path=f"evaluators[{index}].options")
        threshold = _coerce_threshold(entry.get("threshold"))
        specs.append(
            _EvaluatorSpec(
                name=name_value.strip(),
                options=options,
                threshold=threshold,
            )
        )
    return specs


def _resolve_evaluator(name: str, context: Mapping[str, Any]) -> Callable[..., Mapping[str, Any]]:
    normalized_key = name.strip().casefold()
    context_evaluators = context.get("evaluators")
    if isinstance(context_evaluators, Mapping):
        candidate = context_evaluators.get(name)
        if callable(candidate):
            return candidate
        candidate = context_evaluators.get(normalized_key)
        if callable(candidate):
            return candidate
    if normalized_key in _EVALUATOR_REGISTRY:
        return _EVALUATOR_REGISTRY[normalized_key]
    raise EvaluatorNotFoundError(f"Evaluator '{name}' is not registered")


def _resolve_thresholds(
    thresholds: Mapping[str, Any] | float | int | None,
    specs: Iterable[_EvaluatorSpec],
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    default: float | None = None
    per_evaluator: MutableMapping[str, float | None] = {}
    if thresholds is None:
        pass
    elif isinstance(thresholds, (int, float)):
        default = float(thresholds)
    elif isinstance(thresholds, Mapping):
        if "default" in thresholds:
            default = _coerce_threshold(thresholds.get("default"))
        if "global" in thresholds and default is None:
            default = _coerce_threshold(thresholds.get("global"))
        per_mapping = thresholds.get("per_evaluator")
        if isinstance(per_mapping, Mapping):
            for key, value in per_mapping.items():
                per_evaluator[str(key)] = _coerce_threshold(value)
        for key, value in thresholds.items():
            if key in {"default", "global", "per_evaluator"}:
                continue
            per_evaluator[str(key)] = _coerce_threshold(value)
    else:
        raise TypeError("thresholds must be numeric or a mapping")

    applied: dict[str, float | None] = {}
    for spec in specs:
        if spec.threshold is not None:
            applied[spec.name] = spec.threshold
        elif spec.name in per_evaluator:
            applied[spec.name] = per_evaluator[spec.name]
        else:
            applied[spec.name] = default
    defaults_snapshot = {"default": default, "per_evaluator": dict(per_evaluator)}
    return applied, defaults_snapshot


def _normalize_outcome(value: Mapping[str, Any] | Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        data = dict(value)
    else:
        data = {"score": value}
    score = _coerce_score(data.get("score"))
    details = data.get("details")
    if not isinstance(details, Mapping):
        details = {}
    else:
        details = {str(key): item for key, item in details.items()}
    normalized = {
        "score": score,
        "details": details,
    }
    if "explanation" in data and isinstance(data["explanation"], str):
        normalized["explanation"] = data["explanation"]
    if "justification" in data and isinstance(data["justification"], str):
        normalized["justification"] = data["justification"]
    if "evidence" in data:
        normalized["evidence"] = data["evidence"]
    if "error" in data and isinstance(data["error"], str):
        normalized["error"] = data["error"]
    return normalized


def _passes_threshold(score: float | None, threshold: float | None) -> bool:
    if threshold is None:
        return True if score is None else True
    if score is None:
        return False
    return score >= threshold


def _ensure_mapping(value: Any, *, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    raise TypeError(f"{path} must be a mapping when provided")


def _coerce_threshold(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise TypeError("threshold values must be numeric") from None


def _coerce_score(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _collect_analytics_hooks(source: Any) -> list[Callable[[Mapping[str, Any]], None]]:
    hooks: list[Callable[[Mapping[str, Any]], None]] = []
    if source is None:
        return hooks
    if callable(source):
        hooks.append(source)
        return hooks
    if isinstance(source, Mapping):
        for value in source.values():
            hooks.extend(_collect_analytics_hooks(value))
        return hooks
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        for item in source:
            hooks.extend(_collect_analytics_hooks(item))
    return hooks


def _clone_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    import copy

    return copy.deepcopy(payload)
