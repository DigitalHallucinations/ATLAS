"""Shared helpers for the conversation store modules."""

from __future__ import annotations

import hashlib
import struct
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, List, Mapping, Optional, Sequence


_DEFAULT_MESSAGE_TYPE = "text"
_DEFAULT_STATUS = "sent"


def _coerce_uuid(value: Any) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if value is None:
        raise ValueError("UUID value cannot be None")
    if isinstance(value, bytes):
        return uuid.UUID(bytes=value)
    text = str(value)
    try:
        return uuid.UUID(text)
    except ValueError:
        return uuid.UUID(hex=text.replace("-", ""))


def _normalize_tenant_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _coerce_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, str):
        text = value.strip()
        normalised = text
        if normalised.endswith("Z"):
            normalised = normalised[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(normalised)
        except ValueError:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed
    raise TypeError(f"Unsupported datetime value: {value!r}")


def _dt_to_iso(moment: Optional[datetime]) -> Optional[str]:
    if moment is None:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    moment = moment.astimezone(timezone.utc).replace(microsecond=0)
    return moment.isoformat().replace("+00:00", "Z")


def _normalize_message_type(value: Any) -> str:
    if value is None:
        return _DEFAULT_MESSAGE_TYPE
    text = str(value).strip()
    return text or _DEFAULT_MESSAGE_TYPE


def _normalize_status(value: Any) -> str:
    if value is None:
        return _DEFAULT_STATUS
    text = str(value).strip()
    return text or _DEFAULT_STATUS


def _tenant_filter(column, tenant_id: Optional[str]):  # type: ignore[no-untyped-def]
    if tenant_id is None:
        return column.is_(None)
    return column == tenant_id


def _normalize_episode_tags(tags: Optional[Sequence[Any]]) -> List[str]:
    normalized: List[str] = []
    seen: set[str] = set()
    if not tags:
        return normalized
    for tag in tags:
        if tag is None:
            continue
        text = str(tag).strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(text)
    return normalized


def _normalize_json_like(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _normalize_json_like(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_normalize_json_like(item) for item in value]
    return value


def _normalize_node_key(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Graph node key must be a non-empty string")
    return text


def _normalize_edge_key(value: Any | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_edge_type(value: Any | None) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _extract_text_content(payload: Any) -> str:
    if isinstance(payload, Mapping):
        value = payload.get("text")
        if isinstance(value, str):
            return value
    if isinstance(payload, str):
        return payload
    return ""


def _normalize_attempts(attempts: Sequence[Any]) -> List[str]:
    normalised: List[str] = []
    for attempt in attempts:
        if isinstance(attempt, str):
            text = attempt.strip()
            if text:
                normalised.append(text)
                continue
        if isinstance(attempt, datetime):
            normalised.append(_dt_to_iso(_coerce_dt(attempt)) or "")
            continue
        candidate = str(attempt).strip()
        if candidate:
            normalised.append(candidate)
    return normalised


def _hash_vector(values: Sequence[float]) -> str:
    hasher = hashlib.sha1()
    for component in values:
        hasher.update(struct.pack("!d", float(component)))
    return hasher.hexdigest()


@dataclass(frozen=True)
class _CoercedVectorPayload:
    embedding: List[float]
    vector_key: str
    checksum: str
    metadata: dict[str, Any]
    provider: Optional[str]
    model: Optional[str]
    version: Optional[str]


def _coerce_vector_payload(
    message_id: uuid.UUID,
    conversation_id: uuid.UUID,
    raw_vector: Mapping[str, Any],
) -> _CoercedVectorPayload:
    if not isinstance(raw_vector, Mapping):
        raise TypeError("Vector payloads must be mappings containing embedding data.")

    provider = raw_vector.get("provider")
    if provider is not None:
        provider_str = str(provider).strip() or None
    else:
        provider_str = None

    model = raw_vector.get("model") or raw_vector.get("embedding_model")
    if model is not None:
        model_str = str(model).strip() or None
    else:
        model_str = None

    version = raw_vector.get("model_version") or raw_vector.get("embedding_model_version")
    if version is not None:
        version_str = str(version).strip() or None
    else:
        version_str = None

    values = raw_vector.get("embedding")
    if values is None:
        values = raw_vector.get("values")
    if values is None:
        raise ValueError("Vector payloads must include 'embedding' or 'values'.")
    if not isinstance(values, Sequence) or isinstance(values, (bytes, bytearray, str)):
        raise TypeError("Vector embeddings must be provided as a numeric sequence.")

    embedding: List[float] = [float(component) for component in values]
    if not embedding:
        raise ValueError("Vector embeddings must contain at least one component.")

    checksum = raw_vector.get("checksum") or raw_vector.get("embedding_checksum")
    checksum_str = str(checksum).strip() if checksum is not None else ""
    if not checksum_str:
        checksum_str = _hash_vector(embedding)

    raw_vector_key = raw_vector.get("vector_key") or raw_vector.get("id")
    vector_key = str(raw_vector_key).strip() if raw_vector_key is not None else ""
    if not vector_key:
        vector_key = _default_vector_key(message_id, provider_str, model_str, version_str)

    metadata = dict(raw_vector.get("metadata") or {})

    namespace = metadata.get("namespace")
    if namespace is not None:
        metadata["namespace"] = str(namespace).strip() or str(conversation_id)
    else:
        metadata["namespace"] = str(conversation_id)
    metadata.setdefault("conversation_id", str(conversation_id))
    metadata.setdefault("message_id", str(message_id))

    if provider_str:
        metadata.setdefault("provider", provider_str)
    if model_str:
        metadata.setdefault("model", model_str)
        metadata.setdefault("embedding_model", model_str)
    if version_str:
        metadata.setdefault("model_version", version_str)
        metadata.setdefault("embedding_model_version", version_str)

    metadata.setdefault("vector_key", vector_key)
    metadata.setdefault("checksum", checksum_str)
    metadata.setdefault("embedding_checksum", checksum_str)
    metadata.setdefault("dimensions", len(embedding))

    return _CoercedVectorPayload(
        embedding=embedding,
        vector_key=vector_key,
        checksum=checksum_str,
        metadata=metadata,
        provider=provider_str,
        model=model_str,
        version=version_str,
    )


def _default_vector_key(
    message_id: uuid.UUID,
    provider: Optional[str],
    model: Optional[str],
    version: Optional[str],
) -> str:
    provider_part = (provider or "conversation").strip() or "conversation"
    model_part = (model or "default").strip() or "default"
    version_part = (version or "v0").strip() or "v0"
    return f"{message_id}:{provider_part}:{model_part}:{version_part}"


__all__ = [
    "_coerce_vector_payload",
    "_coerce_dt",
    "_coerce_uuid",
    "_default_vector_key",
    "_dt_to_iso",
    "_extract_text_content",
    "_hash_vector",
    "_normalize_attempts",
    "_normalize_edge_key",
    "_normalize_edge_type",
    "_normalize_episode_tags",
    "_normalize_json_like",
    "_normalize_message_type",
    "_normalize_node_key",
    "_normalize_status",
    "_normalize_tenant_id",
    "_tenant_filter",
]
