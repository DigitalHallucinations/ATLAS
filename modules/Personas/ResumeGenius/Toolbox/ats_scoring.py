"""Client for the internal ATS scoring service."""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import requests

from modules.logging.logger import setup_logger

try:  # ConfigManager is optional in some test contexts
    from core.config import ConfigManager
except Exception:  # pragma: no cover - fallback when ConfigManager is unavailable
    ConfigManager = None  # type: ignore


logger = setup_logger(__name__)

_DEFAULT_TIMEOUT = 15.0
_ENDPOINT_PATH = "/score"


class ATSScoringError(RuntimeError):
    """Base class for ATS scoring failures."""


class ATSScoringConfigurationError(ATSScoringError):
    """Raised when the service cannot be configured."""


@dataclass(frozen=True)
class ATSScoreResult:
    """Normalized response returned by :class:`ATSScoringService`."""

    score: Optional[float]
    verdict: str
    matched_keywords: tuple[str, ...]
    missing_keywords: tuple[str, ...]
    optimization_steps: tuple[str, ...]
    raw: Mapping[str, Any]

    def to_dict(self) -> Mapping[str, Any]:
        """Return a JSON-serialisable representation of the result."""

        payload: MutableMapping[str, Any] = {
            "score": self.score,
            "verdict": self.verdict,
            "matched_keywords": list(self.matched_keywords),
            "missing_keywords": list(self.missing_keywords),
            "optimization_steps": list(self.optimization_steps),
            "raw": dict(self.raw),
        }
        return MappingProxyType(payload)


class ATSScoringService:
    """Small wrapper around the ATS scoring HTTP API."""

    def __init__(
        self,
        *,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        request_callable: Optional[Any] = None,
        config_manager: Optional[ConfigManager] = None,
    ) -> None:
        self._config_manager = config_manager or (ConfigManager() if ConfigManager is not None else None)
        settings = self._load_settings(
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout_seconds,
        )
        self._base_url = settings["base_url"]
        self._api_key = settings.get("api_key")
        self._timeout = settings["timeout_seconds"]
        self._request = request_callable or requests.post

    def _load_settings(
        self,
        *,
        base_url: Optional[str],
        api_key: Optional[str],
        timeout_seconds: Optional[float],
    ) -> Mapping[str, Any]:
        block: Mapping[str, Any] = {}
        if self._config_manager is not None:
            try:
                candidate = self._config_manager.get_config("ats_scoring_service", {})
            except Exception:  # pragma: no cover - defensive fallback
                candidate = {}
            if isinstance(candidate, Mapping):
                block = candidate

        resolved_url = self._coerce_string(
            base_url
            or block.get("base_url")
            or self._get_config_value("ATS_SCORING_SERVICE_URL")
            or os.getenv("ATS_SCORING_SERVICE_URL"),
        )
        if not resolved_url:
            raise ATSScoringConfigurationError(
                "ATS scoring service URL is not configured. Set ATS_SCORING_SERVICE_URL or "
                "configure ats_scoring_service.base_url in the application settings."
            )

        resolved_key = self._coerce_string(
            api_key
            or block.get("api_key")
            or self._get_config_value("ATS_SCORING_SERVICE_API_KEY")
            or os.getenv("ATS_SCORING_SERVICE_API_KEY"),
        )

        resolved_timeout = self._coerce_timeout(
            timeout_seconds,
            block.get("timeout_seconds") or block.get("timeout"),
            self._get_config_value("ATS_SCORING_SERVICE_TIMEOUT"),
        )

        return {
            "base_url": resolved_url,
            "api_key": resolved_key,
            "timeout_seconds": resolved_timeout,
        }

    def _get_config_value(self, key: str) -> Optional[Any]:
        if self._config_manager is None:
            return None
        try:
            return self._config_manager.get_config(key)
        except Exception:  # pragma: no cover - guard against misconfigured managers
            return None

    def _coerce_timeout(self, *candidates: Optional[Any]) -> float:
        for candidate in candidates:
            try:
                if candidate is None:
                    continue
                value = float(candidate)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                continue
        return _DEFAULT_TIMEOUT

    def _coerce_string(self, value: Optional[Any]) -> Optional[str]:
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    async def score_resume(
        self,
        *,
        resume_text: str,
        job_description: str,
        job_title: Optional[str] = None,
        focus_skills: Optional[Sequence[str]] = None,
        experience_level: Optional[str] = None,
        locale: Optional[str] = None,
    ) -> ATSScoreResult:
        """Submit resume and job description text to the scoring service."""

        if not isinstance(resume_text, str) or not resume_text.strip():
            raise ValueError("'resume_text' must be a non-empty string")
        if not isinstance(job_description, str) or not job_description.strip():
            raise ValueError("'job_description' must be a non-empty string")

        payload: dict[str, Any] = {
            "resume_text": resume_text,
            "job_description": job_description,
        }
        if job_title:
            payload["job_title"] = job_title
        if experience_level:
            payload["experience_level"] = experience_level
        if locale:
            payload["locale"] = locale
        if focus_skills:
            payload["focus_skills"] = [skill for skill in self._normalize_sequence(focus_skills)]

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        url = self._compose_endpoint(self._base_url, _ENDPOINT_PATH)

        try:
            response = await asyncio.to_thread(
                self._request,
                url,
                json=payload,
                headers=headers,
                timeout=self._timeout,
            )
        except requests.RequestException as exc:  # pragma: no cover - network errors are rare in tests
            logger.error("ATS scoring request failed: %s", exc)
            raise ATSScoringError("ATS scoring request failed") from exc

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            logger.error("ATS scoring service returned error status: %s", exc)
            raise ATSScoringError("ATS scoring service returned an error response") from exc

        try:
            data = response.json()
        except ValueError as exc:
            logger.error("ATS scoring service returned invalid JSON: %s", exc)
            raise ATSScoringError("ATS scoring service returned invalid JSON") from exc

        if not isinstance(data, Mapping):
            logger.error("ATS scoring service returned unexpected payload: %s", data)
            raise ATSScoringError("ATS scoring service returned an unexpected payload")

        return self._normalise_response(data)

    def _normalise_response(self, payload: Mapping[str, Any]) -> ATSScoreResult:
        score_value = payload.get("score")
        numeric_score: Optional[float]
        if score_value is None:
            numeric_score = None
        else:
            try:
                numeric_score = float(score_value)
            except (TypeError, ValueError) as exc:
                raise ATSScoringError("ATS scoring response contained an invalid score value") from exc

        verdict = self._coerce_string(
            payload.get("verdict")
            or payload.get("summary")
            or payload.get("message")
            or payload.get("status")
        ) or ""

        matched = tuple(self._normalize_sequence(
            payload.get("matched_keywords")
            or payload.get("matched_terms")
            or payload.get("highlights")
        ))
        missing = tuple(self._normalize_sequence(
            payload.get("missing_keywords")
            or payload.get("gaps")
            or payload.get("missing_terms")
        ))
        recommendations = tuple(self._normalize_sequence(
            payload.get("recommendations")
            or payload.get("recommended_actions")
            or payload.get("optimization_steps")
        ))

        return ATSScoreResult(
            score=numeric_score,
            verdict=verdict,
            matched_keywords=matched,
            missing_keywords=missing,
            optimization_steps=recommendations,
            raw=MappingProxyType(dict(payload)),
        )

    def _normalize_sequence(self, value: Any) -> Sequence[str]:
        if value is None:
            return ()
        if isinstance(value, str):
            text = value.strip()
            return (text,) if text else ()
        if isinstance(value, Mapping):
            return tuple(
                entry
                for entry in (
                    self._coerce_string(item)
                    for item in value.values()
                )
                if entry
            )
        if isinstance(value, Sequence):  # type: ignore[redundant-expr]
            items: list[str] = []
            for item in value:
                candidate = self._coerce_string(item)
                if candidate and candidate not in items:
                    items.append(candidate)
            return tuple(items)
        candidate = self._coerce_string(value)
        return (candidate,) if candidate else ()

    def _compose_endpoint(self, base_url: str, path: str) -> str:
        cleaned_base = base_url.rstrip("/")
        cleaned_path = path.lstrip("/")
        return f"{cleaned_base}/{cleaned_path}"


__all__ = [
    "ATSScoringError",
    "ATSScoringConfigurationError",
    "ATSScoreResult",
    "ATSScoringService",
]
