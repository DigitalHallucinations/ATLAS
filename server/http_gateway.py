"""FastAPI gateway exposing :class:`AtlasServer` routes over HTTP."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncIterator, Dict, Iterable, Mapping, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse

from ATLAS.ATLAS import ATLAS
from modules.Server.conversation_routes import (
    ConversationAuthorizationError,
    ConversationRouteError,
    RequestContext,
)
from modules.Server.job_routes import JobRouteError
from modules.Server.routes import AtlasServer
from modules.Server.task_routes import TaskRouteError
from modules.Personas import PersonaBundleError, PersonaValidationError
from modules.orchestration.message_bus import shutdown_message_bus
from modules.user_accounts.user_account_service import AccountLockedError

LOGGER = logging.getLogger("atlas.http_gateway")

_HEADER_TENANT = "x-atlas-tenant"
_HEADER_USER = "x-atlas-user"
_HEADER_SESSION = "x-atlas-session"
_HEADER_ROLES = "x-atlas-roles"
_HEADER_METADATA = "x-atlas-metadata"

_HEADER_API_KEY = "x-api-key"
_ENV_API_KEYS = "ATLAS_HTTP_API_KEYS"
_ENV_API_KEY_FILE = "ATLAS_HTTP_API_KEY_FILE"
_ENV_API_KEY_PUBLIC_PATHS = "ATLAS_HTTP_API_KEY_PUBLIC_PATHS"


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Authenticated user context details resolved from credentials."""

    username: str
    roles: tuple[str, ...]
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class ApiKeyConfig:
    """Configuration for API key enforcement."""

    enabled: bool
    valid_tokens: frozenset[str]
    public_paths: frozenset[str]


def _parse_roles(raw_value: Optional[str]) -> tuple[str, ...]:
    if not raw_value:
        return ()
    parts = [token.strip() for token in raw_value.split(",")]
    return tuple(token for token in parts if token)


def _parse_metadata(raw_value: Optional[str]) -> Optional[Mapping[str, Any]]:
    if not raw_value:
        return None
    try:
        data = json.loads(raw_value)
    except json.JSONDecodeError as exc:  # pragma: no cover - FastAPI handles
        raise HTTPException(status_code=400, detail="Invalid X-Atlas-Metadata header") from exc
    if not isinstance(data, Mapping):
        raise HTTPException(status_code=400, detail="Metadata header must encode an object")
    return data


def _normalize_optional_text(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    return token or None


def _read_api_key_file(file_path: str) -> set[str]:
    tokens: set[str] = set()
    try:
        with open(file_path, "r", encoding="utf-8") as handle:
            for line in handle.readlines():
                normalized = line.strip()
                if normalized:
                    tokens.add(normalized)
    except FileNotFoundError:
        LOGGER.warning("API key file %s does not exist", file_path)
    except OSError:
        LOGGER.exception("Unable to read API key file %s", file_path)
    return tokens


def _load_api_key_config() -> ApiKeyConfig:
    raw_tokens = os.getenv(_ENV_API_KEYS, "")
    tokens = {token.strip() for token in raw_tokens.split(",") if token.strip()}

    file_path = _normalize_optional_text(os.getenv(_ENV_API_KEY_FILE))
    if file_path:
        tokens.update(_read_api_key_file(file_path))

    public_paths_env = os.getenv(_ENV_API_KEY_PUBLIC_PATHS, "/healthz")
    public_paths = {
        path.strip() for path in public_paths_env.split(",") if path.strip()
    }

    return ApiKeyConfig(
        enabled=bool(tokens),
        valid_tokens=frozenset(tokens),
        public_paths=frozenset(public_paths),
    )


def _merge_roles(target: list[str], additional: Iterable[str]) -> None:
    for role in additional:
        if role not in target:
            target.append(role)


def _normalize_role_tokens(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        candidates: Iterable[Any] = [value]
    elif isinstance(value, Mapping):
        return ()
    elif isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        candidates = value
    else:
        candidates = [value]

    roles: list[str] = []
    for candidate in candidates:
        if candidate is None:
            continue
        token = str(candidate).strip()
        if token and token not in roles:
            roles.append(token)
    return tuple(roles)


def _parse_authorization_header(
    value: str, *, fallback_username: Optional[str]
) -> tuple[str, str]:
    scheme, _, credentials = value.strip().partition(" ")
    if not scheme or not credentials:
        raise ValueError("Invalid Authorization header")

    scheme = scheme.lower()
    if scheme == "basic":
        try:
            decoded = base64.b64decode(credentials).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError) as exc:
            raise ValueError("Invalid basic authorization credentials") from exc
        if ":" not in decoded:
            raise ValueError("Invalid basic authorization credentials")
        username, password = decoded.split(":", 1)
        normalized_username = _normalize_optional_text(username)
        if normalized_username is None:
            raise ValueError("Invalid basic authorization credentials")
        return normalized_username, password

    if scheme == "bearer":
        token = credentials.strip()
        if not token:
            raise ValueError("Invalid bearer token")
        try:
            decoded = base64.b64decode(token).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError):
            decoded = token
        if ":" in decoded:
            username, secret = decoded.split(":", 1)
            normalized_username = _normalize_optional_text(username)
            if normalized_username is None:
                raise ValueError("Invalid bearer token")
            return normalized_username, secret
        normalized_username = _normalize_optional_text(fallback_username)
        if normalized_username is None:
            raise ValueError("Bearer tokens must include a username")
        return normalized_username, decoded

    raise ValueError("Unsupported authorization scheme")


async def _collect_user_metadata(
    atlas: ATLAS,
    service: Any,
    username: str,
    tenant_id: str,
) -> tuple[tuple[str, ...], Dict[str, Any]]:
    roles: list[str] = []
    metadata: Dict[str, Any] = {}

    if hasattr(service, "get_user_details"):
        try:
            details = await run_in_threadpool(service.get_user_details, username)
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.debug("Failed to load user account details for %s", username, exc_info=True)
        else:
            if isinstance(details, Mapping):
                cleaned = {k: v for k, v in details.items() if v is not None}
                if cleaned:
                    metadata["user"] = cleaned
                    display_name = cleaned.get("display_name") or cleaned.get("name")
                    display_text = _normalize_optional_text(display_name)
                    if display_text:
                        metadata.setdefault("user_display_name", display_text)
                roles.extend(_normalize_role_tokens(details.get("roles")))

    repository = getattr(atlas, "conversation_repository", None)
    if repository is not None:
        try:
            profile = await run_in_threadpool(
                repository.get_user_profile,
                username,
                tenant_id=tenant_id,
            )
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.debug("Failed to resolve conversation profile for %s", username, exc_info=True)
        else:
            if isinstance(profile, Mapping):
                profile_section = profile.get("profile")
                if isinstance(profile_section, Mapping) and profile_section:
                    metadata.setdefault("profile", dict(profile_section))
                    for key in ("roles", "Roles", "user_roles"):
                        roles.extend(_normalize_role_tokens(profile_section.get(key)))
                documents_section = profile.get("documents")
                if isinstance(documents_section, Mapping) and documents_section:
                    metadata.setdefault("documents", dict(documents_section))
                display_name = _normalize_optional_text(profile.get("display_name"))
                if display_name:
                    metadata.setdefault("user_display_name", display_name)
                roles.extend(_normalize_role_tokens(profile.get("roles")))

    resolver = getattr(atlas, "_resolve_active_user_roles", None)
    if callable(resolver):
        try:
            resolved_roles = resolver()
        except Exception:  # pragma: no cover - defensive logging only
            LOGGER.debug("Active user role resolution failed", exc_info=True)
        else:
            roles.extend(_normalize_role_tokens(resolved_roles))

    metadata.setdefault("tenant_id", tenant_id)

    merged_roles: list[str] = []
    _merge_roles(merged_roles, roles)

    return tuple(merged_roles), metadata


async def _authenticate_request(
    request: Request,
    atlas: ATLAS,
    tenant_id: str,
) -> AuthenticatedPrincipal:
    header = request.headers.get("Authorization")
    if not header:
        raise HTTPException(status_code=401, detail="Authentication credentials were not provided")

    fallback_username = _normalize_optional_text(request.headers.get(_HEADER_USER))
    try:
        username, secret = _parse_authorization_header(
            header,
            fallback_username=fallback_username,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc

    if secret is None:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    facade = getattr(atlas, "user_account_facade", None)
    if facade is None:
        raise HTTPException(status_code=503, detail="User account services are unavailable")

    service = facade._get_user_account_service()  # type: ignore[attr-defined]

    try:
        valid = await run_in_threadpool(service.authenticate_user, username, secret)
    except AccountLockedError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc

    if not valid:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

    roles, metadata = await _collect_user_metadata(atlas, service, username, tenant_id)

    return AuthenticatedPrincipal(username=username, roles=roles, metadata=metadata)


async def _build_request_context(request: Request, atlas: ATLAS) -> RequestContext:
    headers = request.headers
    tenant_id = headers.get(_HEADER_TENANT)
    if tenant_id is None or not tenant_id.strip():
        raise HTTPException(status_code=403, detail="X-Atlas-Tenant header is required")
    tenant_id = tenant_id.strip()

    configured_tenant = _normalize_optional_text(getattr(atlas, "tenant_id", None))
    if configured_tenant and tenant_id != configured_tenant:
        raise HTTPException(status_code=403, detail="X-Atlas-Tenant does not match configured tenant")

    header_metadata = _parse_metadata(headers.get(_HEADER_METADATA))
    header_roles = _parse_roles(headers.get(_HEADER_ROLES))

    principal = await _authenticate_request(request, atlas, tenant_id)

    header_user = _normalize_optional_text(headers.get(_HEADER_USER))
    if header_user and header_user.lower() != principal.username.lower():
        raise HTTPException(status_code=401, detail="Authenticated username mismatch")

    combined_roles: list[str] = list(principal.roles)
    _merge_roles(combined_roles, header_roles)

    metadata_payload: Dict[str, Any] = dict(principal.metadata)
    if header_metadata:
        metadata_payload.update(dict(header_metadata))

    if not metadata_payload:
        metadata_value: Optional[Dict[str, Any]] = None
    else:
        metadata_value = metadata_payload

    return RequestContext(
        tenant_id=tenant_id,
        user_id=principal.username,
        session_id=_normalize_optional_text(headers.get(_HEADER_SESSION)),
        roles=tuple(combined_roles),
        metadata=metadata_value,
    )


def _to_http_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, HTTPException):
        return exc
    if isinstance(exc, (ConversationRouteError, TaskRouteError, JobRouteError)):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, ConversationAuthorizationError):
        return HTTPException(status_code=exc.status_code, detail=str(exc))
    if isinstance(exc, (PersonaValidationError,)):
        return HTTPException(status_code=422, detail=str(exc))
    if isinstance(exc, PersonaBundleError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, KeyError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    LOGGER.exception("Unhandled AtlasServer error", exc_info=exc)
    return HTTPException(status_code=500, detail="Internal server error")


async def _call_route(
    method: Any,
    *,
    context: RequestContext,
    args: Iterable[Any] | None = None,
    kwargs: Optional[Dict[str, Any]] = None,
) -> Any:
    call_args = list(args or [])
    call_kwargs = dict(kwargs or {})
    call_kwargs.setdefault("context", context)
    try:
        return await run_in_threadpool(method, *call_args, **call_kwargs)
    except Exception as exc:  # noqa: BLE001 - translate for HTTP layer
        raise _to_http_exception(exc) from exc


def _stream_events_response(iterator: AsyncIterator[Mapping[str, Any]]) -> StreamingResponse:
    async def event_stream() -> AsyncIterator[bytes]:
        try:
            async for payload in iterator:
                chunk = json.dumps(payload, default=str)
                yield f"data: {chunk}\n\n".encode("utf-8")
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - report in stream and log
            LOGGER.exception("Streaming route terminated due to error: %s", exc)
            detail = json.dumps({"detail": str(exc)})
            yield f"event: error\ndata: {detail}\n\n".encode("utf-8")

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@asynccontextmanager
async def lifespan(app: FastAPI):
    atlas = ATLAS()
    await atlas.initialize()
    config_manager = getattr(atlas, "config_manager", None)
    task_service = None
    if config_manager is not None and hasattr(config_manager, "get_task_service"):
        try:
            task_service = config_manager.get_task_service()
        except Exception:  # pragma: no cover - optional dependency
            LOGGER.exception("Task service initialization failed during startup")
    job_service = getattr(atlas, "job_service", None)
    server = AtlasServer(
        config_manager=atlas.config_manager,
        conversation_repository=atlas.conversation_repository,
        message_bus=atlas.message_bus,
        task_service=task_service,
        job_service=job_service,
        job_manager=getattr(atlas, "job_manager", None),
    )
    if getattr(atlas, "job_scheduler", None) is not None:
        server._job_scheduler = atlas.job_scheduler  # type: ignore[attr-defined]
    app.state.atlas = atlas  # type: ignore[attr-defined]
    app.state.server = server  # type: ignore[attr-defined]
    try:
        yield
    finally:
        try:
            await atlas.close()
        except Exception:  # pragma: no cover - best effort shutdown
            LOGGER.exception("ATLAS close() failed")
        bus = getattr(atlas, "message_bus", None)
        if bus is not None:
            try:
                await bus.close()
            except Exception:  # pragma: no cover
                LOGGER.exception("Message bus shutdown failed")
        try:
            await shutdown_message_bus()
        except Exception:  # pragma: no cover
            LOGGER.exception("Global message bus shutdown failed")


def _get_atlas(request: Request) -> ATLAS:
    atlas = getattr(request.app.state, "atlas", None)
    if atlas is None:
        raise HTTPException(status_code=503, detail="ATLAS service is unavailable")
    return atlas


def _get_server(request: Request) -> AtlasServer:
    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="AtlasServer is unavailable")
    return server


app = FastAPI(title="ATLAS HTTP Gateway", lifespan=lifespan)
app.state.api_key_config = _load_api_key_config()  # type: ignore[attr-defined]


@app.middleware("http")
async def enforce_api_key(request: Request, call_next):
    config: ApiKeyConfig | None = getattr(request.app.state, "api_key_config", None)
    if config is None or not config.enabled:
        return await call_next(request)

    path = request.url.path
    if path in config.public_paths:
        return await call_next(request)

    token = _normalize_optional_text(request.headers.get(_HEADER_API_KEY))
    if token is None:
        header_value = request.headers.get("Authorization")
        if header_value:
            scheme, _, credentials = header_value.partition(" ")
            if scheme.lower() == "bearer" and credentials.strip():
                token = credentials.strip()

    if token is None:
        return JSONResponse(
            status_code=401,
            content={"detail": "API key or bearer token is required"},
        )

    if token not in config.valid_tokens:
        return JSONResponse(status_code=403, content={"detail": "Invalid API token"})

    return await call_next(request)


@app.get("/healthz")
async def health_check(request: Request) -> JSONResponse:
    atlas = _get_atlas(request)
    payload = {"status": "ok", "initialized": atlas.is_initialized()}
    return JSONResponse(payload)


@app.get("/conversations")
async def list_conversations(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(
        server.list_conversations,
        context=context,
        args=[],
        kwargs={"params": request.query_params},
    )


@app.post("/conversations/{conversation_id}/reset")
async def reset_conversation(conversation_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.reset_conversation, context=context, args=[conversation_id])


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.delete_conversation, context=context, args=[conversation_id])


@app.get("/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(
        server.list_messages,
        context=context,
        args=[conversation_id],
        kwargs={"params": request.query_params},
    )


@app.post("/conversations/search")
async def search_conversations(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.search_conversations, context=context, args=[payload])


@app.post("/messages")
async def create_message(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.create_message, context=context, args=[payload])


@app.patch("/messages/{message_id}")
async def update_message(message_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(
        server.update_message,
        context=context,
        args=[message_id, payload],
    )


@app.delete("/messages/{message_id}")
async def delete_message(message_id: str, request: Request) -> Any:
    try:
        payload = await request.json()
    except Exception:  # pragma: no cover - empty payloads
        payload = {}
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(
        server.delete_message,
        context=context,
        args=[message_id, payload],
    )


@app.get("/conversations/{conversation_id}/events")
async def stream_conversation_events(conversation_id: str, request: Request, after: Optional[str] = None) -> StreamingResponse:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        iterator = server.stream_conversation_events(conversation_id, context=context, after=after)
    except Exception as exc:  # noqa: BLE001 - translate setup errors
        raise _to_http_exception(exc) from exc
    return _stream_events_response(iterator)


@app.post("/conversations/retention")
async def run_conversation_retention(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(server.run_conversation_retention, context=context)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.get("/tasks")
async def list_tasks(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(
        server.list_tasks,
        context=context,
        kwargs={"params": request.query_params},
    )


@app.post("/tasks")
async def create_task(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.create_task, context=context, args=[payload])


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request, include_events: Optional[bool] = None) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs: Dict[str, Any] = {}
    if include_events is not None:
        kwargs["include_events"] = include_events
    return await _call_route(server.get_task, context=context, args=[task_id], kwargs=kwargs)


@app.patch("/tasks/{task_id}")
async def update_task(task_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.update_task, context=context, args=[task_id, payload])


@app.post("/tasks/{task_id}/transition")
async def transition_task(task_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    target_status = payload.get("target_status")
    if target_status is None:
        raise HTTPException(status_code=400, detail="target_status is required")
    kwargs = {
        "target_status": target_status,
        "expected_updated_at": payload.get("expected_updated_at"),
    }
    return await _call_route(
        server.transition_task,
        context=context,
        args=[task_id, kwargs.pop("target_status")],
        kwargs=kwargs,
    )


@app.delete("/tasks/{task_id}")
async def delete_task(task_id: str, request: Request, expected_updated_at: Optional[str] = None) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": expected_updated_at} if expected_updated_at else {}
    return await _call_route(server.delete_task, context=context, args=[task_id], kwargs=kwargs)


@app.post("/tasks/search")
async def search_tasks(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.search_tasks, context=context, args=[payload])


@app.get("/tasks/{task_id}/events")
async def stream_task_events(task_id: str, request: Request, after: Optional[str] = None) -> StreamingResponse:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        iterator = server.stream_task_events(task_id, context=context, after=after)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return _stream_events_response(iterator)


@app.get("/jobs")
async def list_jobs(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(
        server.list_jobs,
        context=context,
        kwargs={"params": request.query_params},
    )


@app.post("/jobs")
async def create_job(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.create_job, context=context, args=[payload])


@app.get("/jobs/{job_id}")
async def get_job(
    job_id: str,
    request: Request,
    include_schedule: Optional[bool] = None,
    include_runs: Optional[bool] = None,
    include_events: Optional[bool] = None,
) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs: Dict[str, Any] = {}
    if include_schedule is not None:
        kwargs["include_schedule"] = include_schedule
    if include_runs is not None:
        kwargs["include_runs"] = include_runs
    if include_events is not None:
        kwargs["include_events"] = include_events
    return await _call_route(server.get_job, context=context, args=[job_id], kwargs=kwargs)


@app.patch("/jobs/{job_id}")
async def update_job(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.update_job, context=context, args=[job_id, payload])


@app.post("/jobs/{job_id}/transition")
async def transition_job(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    target_status = payload.get("target_status")
    if target_status is None:
        raise HTTPException(status_code=400, detail="target_status is required")
    kwargs = {
        "target_status": target_status,
        "expected_updated_at": payload.get("expected_updated_at"),
    }
    return await _call_route(
        server.transition_job,
        context=context,
        args=[job_id, kwargs.pop("target_status")],
        kwargs=kwargs,
    )


@app.post("/jobs/{job_id}/schedule/pause")
async def pause_job_schedule(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.pause_job_schedule, context=context, args=[job_id], kwargs=kwargs)


@app.post("/jobs/{job_id}/schedule/resume")
async def resume_job_schedule(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.resume_job_schedule, context=context, args=[job_id], kwargs=kwargs)


@app.post("/jobs/{job_id}/rerun")
async def rerun_job(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.rerun_job, context=context, args=[job_id], kwargs=kwargs)


@app.post("/jobs/{job_id}/run-now")
async def run_job_now(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.run_job_now, context=context, args=[job_id], kwargs=kwargs)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, request: Request, expected_updated_at: Optional[str] = None) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": expected_updated_at} if expected_updated_at else {}
    return await _call_route(server.delete_job, context=context, args=[job_id], kwargs=kwargs)


@app.get("/jobs/{job_id}/tasks")
async def list_job_tasks(job_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.list_job_tasks, context=context, args=[job_id])


@app.post("/jobs/{job_id}/tasks")
async def link_job_task(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    return await _call_route(server.link_job_task, context=context, args=[job_id, payload])


@app.delete("/jobs/{job_id}/tasks")
async def unlink_job_task(job_id: str, request: Request) -> Any:
    try:
        payload = await request.json()
    except Exception:  # pragma: no cover - empty payloads
        payload = {}
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    kwargs = {
        "link_id": payload.get("link_id"),
        "task_id": payload.get("task_id"),
    }
    return await _call_route(server.unlink_job_task, context=context, args=[job_id], kwargs=kwargs)


@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request, after: Optional[str] = None) -> StreamingResponse:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        iterator = server.stream_job_events(job_id, context=context, after=after)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return _stream_events_response(iterator)


@app.get("/tools")
async def get_tools(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/tools",
            method="GET",
            query=request.query_params,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.get("/skills")
async def get_skills(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/skills",
            method="GET",
            query=request.query_params,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.get("/personas/{persona_name}/analytics")
async def get_persona_metrics(persona_name: str, request: Request) -> Any:
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/analytics",
            method="GET",
            query=request.query_params,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.get("/personas/analytics/comparison")
async def get_persona_comparison(request: Request) -> Any:
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/personas/analytics/comparison",
            method="GET",
            query=request.query_params,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.get("/personas/{persona_name}/review")
async def get_persona_review(persona_name: str, request: Request) -> Any:
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/review",
            method="GET",
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/review")
async def attest_persona_review(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/review",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/tools")
async def update_persona_tools(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/tools",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/skills")
async def update_persona_skills(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/skills",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/export")
async def export_persona_bundle(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/export",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/import")
async def import_persona_bundle(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/personas/import",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/tasks/{task_name}/export")
async def export_task_bundle(task_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/tasks/{task_name}/export",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/tasks/import")
async def import_task_bundle(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/tasks/import",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/tools/{tool_name}/export")
async def export_tool_bundle(tool_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/tools/{tool_name}/export",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/tools/import")
async def import_tool_bundle(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/tools/import",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/skills/{skill_name}/export")
async def export_skill_bundle(skill_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/skills/{skill_name}/export",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/skills/import")
async def import_skill_bundle(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/skills/import",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/jobs/{job_name}/export")
async def export_job_bundle(job_name: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/jobs/{job_name}/export",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/jobs/import")
async def import_job_bundle(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = await _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/jobs/import",
            method="POST",
            query=payload,
            context=context,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result
