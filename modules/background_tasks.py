"""Utility helpers for running asynchronous tasks in dedicated threads."""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from typing import Awaitable, Callable, TypeVar


T = TypeVar("T")


def run_async_in_thread(
    coroutine_factory: Callable[[], Awaitable[T]],
    *,
    on_success: Callable[[T], None] | None = None,
    on_error: Callable[[Exception], None] | None = None,
    logger: logging.Logger | None = None,
    thread_name: str | None = None,
) -> Future[T]:
    """Execute an awaitable produced by ``coroutine_factory`` in a background thread.

    Args:
        coroutine_factory: Callable that returns the coroutine to execute.
        on_success: Optional callback invoked with the coroutine result.
        on_error: Optional callback invoked with the raised exception.
        logger: Optional logger used for reporting callback failures.
        thread_name: Optional name for the spawned thread.

    Returns:
        A :class:`concurrent.futures.Future` containing the coroutine result.
    """

    future: Future[T] = Future()

    def log_callback_error(callback: Callable[..., None], exc: Exception) -> None:
        if logger is not None:
            logger.error("Callback %s raised an exception: %s", callback, exc, exc_info=True)

    def runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coroutine_factory())
        except Exception as exc:  # noqa: BLE001 - we want to propagate arbitrary exceptions
            future.set_exception(exc)
            if on_error is not None:
                try:
                    on_error(exc)
                except Exception as callback_exc:  # noqa: BLE001 - log callback failures
                    log_callback_error(on_error, callback_exc)
        else:
            future.set_result(result)
            if on_success is not None:
                try:
                    on_success(result)
                except Exception as callback_exc:  # noqa: BLE001 - log callback failures
                    log_callback_error(on_success, callback_exc)
        finally:
            try:
                loop.run_until_complete(loop.shutdown_asyncgens())
            except Exception:
                pass
            finally:
                loop.close()

    thread = threading.Thread(target=runner, name=thread_name, daemon=True)
    thread.start()
    return future

