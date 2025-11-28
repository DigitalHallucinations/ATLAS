"""Input validation helpers for the setup wizard forms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class EnterpriseOrg:
    company_name: str
    company_domain: str
    primary_contact: str
    tenant_id: str


def parse_required_int(value: str, field: str) -> int:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} is required")
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc


def parse_optional_int(value: str, field: str) -> Optional[int]:
    text = value.strip()
    if not text:
        return None
    try:
        return int(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be an integer") from exc


def parse_required_positive_int(value: str, field: str) -> int:
    parsed = parse_required_int(value, field)
    if parsed <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return parsed


def parse_required_float(value: str, field: str) -> float:
    text = value.strip()
    if not text:
        raise ValueError(f"{field} is required")
    try:
        return float(text)
    except ValueError as exc:
        raise ValueError(f"{field} must be a number") from exc


def validate_enterprise_org(
    *,
    company_name: str,
    company_domain: str,
    primary_contact: str,
    tenant_id: str,
) -> EnterpriseOrg:
    name = company_name.strip()
    if not name:
        raise ValueError("Company name is required for enterprise setups")

    domain = company_domain.strip()
    if not domain:
        raise ValueError("Company domain is required for enterprise setups")

    contact = primary_contact.strip()
    if not contact:
        raise ValueError("Primary contact is required for enterprise setups")

    tenant = tenant_id.strip()
    if not tenant:
        raise ValueError("Tenant ID is required for enterprise setups")

    return EnterpriseOrg(
        company_name=name,
        company_domain=domain,
        primary_contact=contact,
        tenant_id=tenant,
    )
