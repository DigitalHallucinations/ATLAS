"""Tool bindings for the Specter persona."""

from modules.Tools.Base_Tools.context_tracker import context_tracker
from modules.Tools.Base_Tools.time import get_current_info
from modules.Tools.Base_Tools.log_event import log_event
from modules.Tools.Base_Tools.policy_reference import policy_reference
from modules.Tools.Base_Tools.log_parser import LogParser
from modules.Tools.Base_Tools.threat_scanner import ThreatScanner
from modules.Tools.Base_Tools.audit_reporter import AuditReporter
from modules.Tools.Base_Tools.sys_snapshot import SysSnapshot
from modules.Tools.Base_Tools.incident_summarizer import IncidentSummarizer

_log_parser = LogParser()
_threat_scanner = ThreatScanner()
_audit_reporter = AuditReporter()
_sys_snapshot = SysSnapshot()
_incident_summarizer = IncidentSummarizer()

function_map = {
    "context_tracker": context_tracker,
    "get_current_info": get_current_info,
    "log_parser": _log_parser.run,
    "threat_scanner": _threat_scanner.run,
    "audit_reporter": _audit_reporter.run,
    "sys_snapshot": _sys_snapshot.run,
    "incident_summarizer": _incident_summarizer.run,
    "log.event": log_event,
    "policy_reference": policy_reference,
}
