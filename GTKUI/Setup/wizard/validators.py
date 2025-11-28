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
    contact_email: str
    address_line1: str
    address_line2: str | None
    city: str
    state: str
    postal_code: str
    country: str
    phone_number: str | None


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
    contact_email: str,
    address_line1: str,
    address_line2: str = "",
    city: str = "",
    state: str = "",
    postal_code: str = "",
    country: str = "",
    phone_number: str = "",
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

    email = contact_email.strip().lower()
    if not email:
        raise ValueError("Contact email is required for enterprise setups")
    if "@" not in email or " " in email:
        raise ValueError("Contact email must be a valid email address")

    line1 = address_line1.strip()
    if not line1:
        raise ValueError("Address line 1 is required for enterprise setups")

    line2 = address_line2.strip() if isinstance(address_line2, str) else ""

    city_value = city.strip()
    if not city_value:
        raise ValueError("City or town is required for enterprise setups")

    state_value = state.strip()
    if not state_value:
        raise ValueError("State or province is required for enterprise setups")

    postal_value = postal_code.strip()
    if not postal_value:
        raise ValueError("Postal code is required for enterprise setups")

    country_value = country.strip()
    if not country_value:
        raise ValueError("Country is required for enterprise setups")

    phone_value = phone_number.strip()
    if phone_value and not any(char.isdigit() for char in phone_value):
        raise ValueError("Phone number must include at least one digit")

    phone_normalized = phone_value or ""

    return EnterpriseOrg(
        company_name=name,
        company_domain=domain,
        primary_contact=contact,
        tenant_id=tenant,
        contact_email=email,
        address_line1=line1,
        address_line2=line2 or None,
        city=city_value,
        state=state_value,
        postal_code=postal_value,
        country=country_value,
        phone_number=phone_normalized or None,
    )
