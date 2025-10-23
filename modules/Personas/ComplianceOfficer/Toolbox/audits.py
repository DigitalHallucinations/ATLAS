"""Compliance assessment utilities for the ComplianceOfficer persona."""

from __future__ import annotations

import asyncio
from typing import Dict, List

_CHECKLISTS: Dict[str, List[str]] = {
    "gdpr": [
        "data_processing_inventory",
        "dpo_appointed",
        "breach_response_plan",
        "data_subject_request_workflow",
        "privacy_by_design_reviews",
    ],
    "hipaa": [
        "risk_analysis",
        "employee_training",
        "incident_response_plan",
        "business_associate_agreements",
        "audit_logging",
    ],
    "iso27001": [
        "asset_register",
        "access_control_policy",
        "incident_reporting",
        "business_continuity_plan",
        "vendor_risk_management",
    ],
}


async def regulatory_gap_audit(domain: str, controls: List[str]) -> Dict[str, object]:
    """Identify control gaps relative to compliance checklists.

    Examples
    --------
    >>> await regulatory_gap_audit('gdpr', ['data_processing_inventory'])
    {'domain': 'gdpr', 'missing': [...]}  # doctest: +SKIP
    """

    await asyncio.sleep(0)
    normalized_domain = domain.lower()
    expected = _CHECKLISTS.get(normalized_domain, [])
    normalized_controls = {item.lower() for item in controls}
    missing = [control for control in expected if control not in normalized_controls]

    return {
        "domain": normalized_domain,
        "expected_controls": expected,
        "provided_controls": controls,
        "missing": missing,
        "status": "complete" if not missing else "gaps_identified",
        "next_steps": [
            "Document remediation owners for each missing control.",
            "Schedule follow-up audit to verify implementation.",
        ],
    }
