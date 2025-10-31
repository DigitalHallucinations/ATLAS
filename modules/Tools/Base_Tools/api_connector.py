"""HTTP connector utility for orchestrating authenticated API calls.

The connector normalises outbound request metadata so personas can reason
about ingest operations before execution. Calls are executed with
``requests`` in a background thread to avoid blocking the event loop.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence
from urllib.parse import urlparse

import requests

from modules.logging.logger import setup_logger

__all__ = ["APIConnector", "APIConnectorError"]


logger = setup_logger(__name__)

_ALLOWED_SCHEMES = {"http", "https"}
_MAX_RESPONSE_PREVIEW = 32 * 1024  # 32 KiB


class APIConnectorError(RuntimeError):
    """Raised when the connector cannot execute the requested operation."""


@dataclass(frozen=True)
class APIRequestSummary:
    method: str
    url: str
    params: Mapping[str, Any]
    headers: Mapping[str, str]
    has_body: bool
    timeout: float | None


@dataclass(frozen=True)
class APIResponseSummary:
    status_code: int
    headers: Mapping[str, str]
    body_preview: str
    truncated: bool


def _normalise_headers(headers: Mapping[str, str] | None) -> Mapping[str, str]:
    if not headers:
        return {}
    cleaned: MutableMapping[str, str] = {}
    for key, value in headers.items():
        if not isinstance(key, str):
            continue
        if value is None:
            continue
        cleaned[key.strip()] = str(value)
    return dict(cleaned)


def _serialise_body(body: Any) -> tuple[str | None, Mapping[str, str]]:
    if body is None:
        return None, {}
    if isinstance(body, (bytes, bytearray)):
        return body.decode("utf-8", errors="replace"), {
            "Content-Length": str(len(body)),
        }
    if isinstance(body, str):
        return body, {"Content-Length": str(len(body.encode("utf-8")))}
    try:
        payload = json.dumps(body)
    except TypeError as exc:  # pragma: no cover - defensive branch
        raise APIConnectorError("Unable to serialise request body to JSON.") from exc
    return payload, {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload.encode("utf-8"))),
    }


class APIConnector:
    """Lightweight HTTP client with domain allow-listing."""

    def __init__(
        self,
        *,
        allowed_domains: Sequence[str] | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._allowed_domains = tuple(domain.lower() for domain in (allowed_domains or ()))
        self._session = session or requests.Session()

    def _validate_url(self, endpoint: str) -> str:
        parsed = urlparse(endpoint)
        if parsed.scheme.lower() not in _ALLOWED_SCHEMES:
            raise APIConnectorError("Endpoint must use http or https.")
        host = (parsed.hostname or "").lower()
        if self._allowed_domains and host not in self._allowed_domains:
            raise APIConnectorError(
                f"Endpoint '{endpoint}' is not part of the configured allow list."
            )
        if not parsed.netloc:
            raise APIConnectorError("Endpoint must include a hostname.")
        return parsed.geturl()

    async def run(
        self,
        *,
        endpoint: str,
        method: str = "GET",
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        body: Any = None,
        timeout: float | None = 30.0,
        dry_run: bool = True,
    ) -> Mapping[str, Any]:
        """Execute or stage an API request.

        When ``dry_run`` is ``True`` the connector returns the normalised
        request without issuing it. Setting ``dry_run`` to ``False`` executes
        the HTTP call and records a bounded response preview.
        """

        url = self._validate_url(endpoint)
        method_normalised = method.upper()
        if method_normalised not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD"}:
            raise APIConnectorError(f"Unsupported HTTP method: {method}")

        safe_headers = _normalise_headers(headers)
        body_payload, inferred_headers = _serialise_body(body)
        merged_headers = {**safe_headers, **inferred_headers}

        summary = APIRequestSummary(
            method=method_normalised,
            url=url,
            params=dict(params or {}),
            headers=merged_headers,
            has_body=body_payload is not None,
            timeout=timeout,
        )

        result: dict[str, Any] = {
            "request": summary.__dict__,
            "dry_run": dry_run,
        }

        if dry_run:
            logger.info("Planned API call to %s with method %s", url, method_normalised)
            return result

        response = await asyncio.to_thread(
            self._dispatch,
            summary,
            body_payload,
        )
        result["response"] = response.__dict__
        return result

    def _dispatch(
        self,
        summary: APIRequestSummary,
        body_payload: str | None,
    ) -> APIResponseSummary:
        try:
            logger.info("Executing API call to %s", summary.url)
            response = self._session.request(
                summary.method,
                summary.url,
                params=summary.params or None,
                headers=summary.headers or None,
                data=body_payload,
                timeout=summary.timeout,
            )
        except requests.RequestException as exc:
            raise APIConnectorError(str(exc)) from exc

        truncated = False
        body_preview = ""
        try:
            content = response.content
            if len(content) > _MAX_RESPONSE_PREVIEW:
                truncated = True
            body_preview = content[:_MAX_RESPONSE_PREVIEW].decode(
                "utf-8", errors="replace"
            )
        except Exception:  # pragma: no cover - defensive guard
            body_preview = ""

        return APIResponseSummary(
            status_code=response.status_code,
            headers=dict(response.headers),
            body_preview=body_preview,
            truncated=truncated,
        )
