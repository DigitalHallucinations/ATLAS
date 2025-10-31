"""Toolbox wiring for the Hermes persona."""

from modules.Tools.Base_Tools.api_connector import APIConnector
from modules.Tools.Base_Tools.file_ingest import FileIngestor
from modules.Tools.Base_Tools.stream_monitor import StreamMonitor
from modules.Tools.Base_Tools.schema_infer import SchemaInference
from modules.Tools.Base_Tools.data_bridge import DataBridge
from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.priority_queue import priority_queue
from .helpers import compose_ingestion_playbook, stage_pipeline

api_connector_tool = APIConnector()
file_ingest_tool = FileIngestor()
stream_monitor_tool = StreamMonitor()
schema_infer_tool = SchemaInference()
data_bridge_tool = DataBridge(
    api_connector=api_connector_tool,
    file_ingest=file_ingest_tool,
    stream_monitor=stream_monitor_tool,
    schema_infer=schema_infer_tool,
)


function_map = {
    "context_tracker": context_tracker,
    "get_current_info": get_current_info,
    "policy_reference": policy_reference,
    "priority_queue": priority_queue,
    "api_connector": api_connector_tool.run,
    "file_ingest": file_ingest_tool.run,
    "stream_monitor": stream_monitor_tool.run,
    "schema_infer": schema_infer_tool.run,
    "data_bridge": data_bridge_tool.run,
    "hermes.compose_playbook": compose_ingestion_playbook,
    "hermes.stage_pipeline": stage_pipeline,
}
