from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple


@dataclass(frozen=True)
class AuditTemplate:
    """Describe a preconfigured audit sink and retention posture."""

    key: str
    label: str
    intent: str
    persona_sink: str
    skill_sink: str
    retention_days: int
    retention_history_limit: int
    tooltip: str


_TEMPLATES: Tuple[AuditTemplate, ...] = (
    AuditTemplate(
        key="siem_30d",
        label="SIEM handoff (30d / 500 msgs)",
        intent=(
            "Keep a 30-day JSONL buffer while forwarding persona and skill audit events "
            "to an external SIEM for continuous monitoring."
        ),
        persona_sink="audit/siem_persona.jsonl",
        skill_sink="audit/siem_skill.jsonl",
        retention_days=30,
        retention_history_limit=500,
        tooltip="Routes audit trails to SIEM-friendly files and keeps a 30 day buffer for investigations.",
    ),
    AuditTemplate(
        key="privacy_14d",
        label="Privacy minimised (14d / 200 msgs)",
        intent=(
            "Minimise local exposure for privacy-sensitive tenants with shorter buffers and "
            "tighter history caps."
        ),
        persona_sink="audit/privacy_persona.jsonl",
        skill_sink="audit/privacy_skill.jsonl",
        retention_days=14,
        retention_history_limit=200,
        tooltip="Short-lived audit spool aimed at data minimisation programs.",
    ),
    AuditTemplate(
        key="extended_90d",
        label="Extended review (90d / 1500 msgs)",
        intent=(
            "Retain longer audit evidence for quarterly control reviews while keeping persona and "
            "skill changes isolated per file."
        ),
        persona_sink="audit/extended_persona.jsonl",
        skill_sink="audit/extended_skill.jsonl",
        retention_days=90,
        retention_history_limit=1500,
        tooltip="Longer retention for regulated teams that need quarterly evidence windows.",
    ),
)


def get_audit_templates() -> Tuple[AuditTemplate, ...]:
    """Return the available audit templates."""

    return _TEMPLATES


def get_audit_template(key: Optional[str]) -> Optional[AuditTemplate]:
    """Return the template matching *key*, if present."""

    if key is None:
        return None
    for template in _TEMPLATES:
        if template.key == key:
            return template
    return None


def audit_template_choices() -> Dict[str, str]:
    """Return a mapping of template keys to human-friendly labels."""

    return {template.key: template.label for template in _TEMPLATES}


def describe_templates() -> Iterable[AuditTemplate]:
    """Iterate over known templates for UI or documentation rendering."""

    return list(_TEMPLATES)
