"""FastAPI gateway exposing :class:`AtlasServer` routes over HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
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

LOGGER = logging.getLogger("atlas.http_gateway")

_HEADER_TENANT = "x-atlas-tenant"
_HEADER_USER = "x-atlas-user"
_HEADER_SESSION = "x-atlas-session"
_HEADER_ROLES = "x-atlas-roles"
_HEADER_METADATA = "x-atlas-metadata"


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


def _build_request_context(request: Request, atlas: ATLAS) -> RequestContext:
    headers = request.headers
    tenant_id = headers.get(_HEADER_TENANT)
    if tenant_id is None or not tenant_id.strip():
        tenant_id = getattr(atlas, "tenant_id", None) or "default"
    metadata = _parse_metadata(headers.get(_HEADER_METADATA))
    roles = _parse_roles(headers.get(_HEADER_ROLES))
    return RequestContext(
        tenant_id=tenant_id,
        user_id=headers.get(_HEADER_USER),
        session_id=headers.get(_HEADER_SESSION),
        roles=roles,
        metadata=metadata,
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


@app.get("/healthz")
async def health_check(request: Request) -> JSONResponse:
    atlas = _get_atlas(request)
    payload = {"status": "ok", "initialized": atlas.is_initialized()}
    return JSONResponse(payload)


@app.get("/conversations")
async def list_conversations(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    return await _call_route(server.reset_conversation, context=context, args=[conversation_id])


@app.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    return await _call_route(server.delete_conversation, context=context, args=[conversation_id])


@app.get("/conversations/{conversation_id}/messages")
async def list_messages(conversation_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    return await _call_route(server.search_conversations, context=context, args=[payload])


@app.post("/messages")
async def create_message(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    return await _call_route(server.create_message, context=context, args=[payload])


@app.patch("/messages/{message_id}")
async def update_message(message_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    return await _call_route(
        server.delete_message,
        context=context,
        args=[message_id, payload],
    )


@app.get("/conversations/{conversation_id}/events")
async def stream_conversation_events(conversation_id: str, request: Request, after: Optional[str] = None) -> StreamingResponse:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    try:
        iterator = server.stream_conversation_events(conversation_id, context=context, after=after)
    except Exception as exc:  # noqa: BLE001 - translate setup errors
        raise _to_http_exception(exc) from exc
    return _stream_events_response(iterator)


@app.post("/conversations/retention")
async def run_conversation_retention(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    try:
        result = await run_in_threadpool(server.run_conversation_retention, context=context)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.get("/tasks")
async def list_tasks(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    return await _call_route(server.create_task, context=context, args=[payload])


@app.get("/tasks/{task_id}")
async def get_task(task_id: str, request: Request, include_events: Optional[bool] = None) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    kwargs: Dict[str, Any] = {}
    if include_events is not None:
        kwargs["include_events"] = include_events
    return await _call_route(server.get_task, context=context, args=[task_id], kwargs=kwargs)


@app.patch("/tasks/{task_id}")
async def update_task(task_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    return await _call_route(server.update_task, context=context, args=[task_id, payload])


@app.post("/tasks/{task_id}/transition")
async def transition_task(task_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": expected_updated_at} if expected_updated_at else {}
    return await _call_route(server.delete_task, context=context, args=[task_id], kwargs=kwargs)


@app.post("/tasks/search")
async def search_tasks(request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    return await _call_route(server.search_tasks, context=context, args=[payload])


@app.get("/tasks/{task_id}/events")
async def stream_task_events(task_id: str, request: Request, after: Optional[str] = None) -> StreamingResponse:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    try:
        iterator = server.stream_task_events(task_id, context=context, after=after)
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return _stream_events_response(iterator)


@app.get("/jobs")
async def list_jobs(request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    return await _call_route(server.update_job, context=context, args=[job_id, payload])


@app.post("/jobs/{job_id}/transition")
async def transition_job(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    context = _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.pause_job_schedule, context=context, args=[job_id], kwargs=kwargs)


@app.post("/jobs/{job_id}/schedule/resume")
async def resume_job_schedule(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.resume_job_schedule, context=context, args=[job_id], kwargs=kwargs)


@app.post("/jobs/{job_id}/rerun")
async def rerun_job(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.rerun_job, context=context, args=[job_id], kwargs=kwargs)


@app.post("/jobs/{job_id}/run-now")
async def run_job_now(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": payload.get("expected_updated_at")}
    return await _call_route(server.run_job_now, context=context, args=[job_id], kwargs=kwargs)


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str, request: Request, expected_updated_at: Optional[str] = None) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    kwargs = {"expected_updated_at": expected_updated_at} if expected_updated_at else {}
    return await _call_route(server.delete_job, context=context, args=[job_id], kwargs=kwargs)


@app.get("/jobs/{job_id}/tasks")
async def list_job_tasks(job_id: str, request: Request) -> Any:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    return await _call_route(server.list_job_tasks, context=context, args=[job_id])


@app.post("/jobs/{job_id}/tasks")
async def link_job_task(job_id: str, request: Request) -> Any:
    payload = await request.json()
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    return await _call_route(server.link_job_task, context=context, args=[job_id, payload])


@app.delete("/jobs/{job_id}/tasks")
async def unlink_job_task(job_id: str, request: Request) -> Any:
    try:
        payload = await request.json()
    except Exception:  # pragma: no cover - empty payloads
        payload = {}
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
    kwargs = {
        "link_id": payload.get("link_id"),
        "task_id": payload.get("task_id"),
    }
    return await _call_route(server.unlink_job_task, context=context, args=[job_id], kwargs=kwargs)


@app.get("/jobs/{job_id}/events")
async def stream_job_events(job_id: str, request: Request, after: Optional[str] = None) -> StreamingResponse:
    atlas = _get_atlas(request)
    server = _get_server(request)
    context = _build_request_context(request, atlas)
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
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/review",
            method="POST",
            query=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/tools")
async def update_persona_tools(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/tools",
            method="POST",
            query=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/skills")
async def update_persona_skills(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/skills",
            method="POST",
            query=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/{persona_name}/export")
async def export_persona_bundle(persona_name: str, request: Request) -> Any:
    payload = await request.json()
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            f"/personas/{persona_name}/export",
            method="POST",
            query=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result


@app.post("/personas/import")
async def import_persona_bundle(request: Request) -> Any:
    payload = await request.json()
    server = _get_server(request)
    try:
        result = await run_in_threadpool(
            server.handle_request,
            "/personas/import",
            method="POST",
            query=payload,
        )
    except Exception as exc:  # noqa: BLE001
        raise _to_http_exception(exc) from exc
    return result
