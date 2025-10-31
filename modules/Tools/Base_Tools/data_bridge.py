"""Composable ingestion orchestrator that stitches connector tool outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from modules.Tools.Base_Tools.api_connector import APIConnector
from modules.Tools.Base_Tools.file_ingest import FileIngestor
from modules.Tools.Base_Tools.schema_infer import SchemaInference
from modules.Tools.Base_Tools.stream_monitor import StreamMonitor
from modules.logging.logger import setup_logger

__all__ = ["DataBridge", "DataBridgeError"]


logger = setup_logger(__name__)


class DataBridgeError(RuntimeError):
    """Raised when a pipeline step cannot be executed."""


@dataclass(frozen=True)
class BridgeStepResult:
    tool: str
    index: int
    payload: Mapping[str, Any]


class DataBridge:
    """Coordinate ingestion primitives into a repeatable workflow."""

    def __init__(
        self,
        *,
        api_connector: APIConnector | None = None,
        file_ingest: FileIngestor | None = None,
        stream_monitor: StreamMonitor | None = None,
        schema_infer: SchemaInference | None = None,
    ) -> None:
        self._api = api_connector or APIConnector()
        self._files = file_ingest or FileIngestor()
        self._streams = stream_monitor or StreamMonitor()
        self._schema = schema_infer or SchemaInference()

    async def run(
        self,
        *,
        source: str,
        operations: Sequence[Mapping[str, Any]],
        dry_run: bool = True,
    ) -> Mapping[str, Any]:
        if not source:
            raise DataBridgeError("source must be provided")
        if not operations:
            raise DataBridgeError("operations must contain at least one step")

        results = []
        for index, operation in enumerate(operations):
            tool_name = str(operation.get("tool", "")).strip()
            if not tool_name:
                raise DataBridgeError(f"Step {index} is missing a tool identifier")
            params = operation.get("params", {})
            if not isinstance(params, Mapping):
                raise DataBridgeError(f"Step {index} params must be a mapping")
            execute = operation.get("dry_run", dry_run)
            payload = await self._dispatch(tool_name, params, execute)
            results.append(
                BridgeStepResult(tool=tool_name, index=index, payload=payload).__dict__
            )

        logger.info("DataBridge completed %d steps for %s", len(results), source)
        return {
            "source": source,
            "dry_run": dry_run,
            "steps": results,
        }

    async def _dispatch(
        self,
        tool_name: str,
        params: Mapping[str, Any],
        dry_run: bool,
    ) -> Mapping[str, Any]:
        if tool_name == "api_connector":
            return await self._api.run(dry_run=dry_run, **params)
        if tool_name == "file_ingest":
            return await self._files.run(**params)
        if tool_name == "stream_monitor":
            return await self._streams.run(**params)
        if tool_name == "schema_infer":
            return await self._schema.run(**params)
        if tool_name == "data_bridge":
            raise DataBridgeError("Nested data_bridge calls are not supported")
        raise DataBridgeError(f"Unknown tool '{tool_name}' referenced in pipeline")
