import pytest

from GTKUI.Setup.wizard.validators import (
    parse_optional_int,
    parse_required_float,
    parse_required_int,
    parse_required_positive_int,
    validate_enterprise_org,
)


@pytest.mark.parametrize(
    "value,field,expected",
    [("10", "Workers", 10), ("   -2", "Workers", -2)],
)
def test_parse_required_int_success(value, field, expected):
    assert parse_required_int(value, field) == expected


@pytest.mark.parametrize("value", ["", "  "])
def test_parse_required_int_empty(value):
    with pytest.raises(ValueError):
        parse_required_int(value, "Field")


def test_parse_optional_int_blank_returns_none():
    assert parse_optional_int("", "Optional") is None


@pytest.mark.parametrize(
    "value,expected",
    [("5", 5), ("7.0", 7.0)],
)
def test_parse_required_float_success(value, expected):
    assert parse_required_float(value, "Field") == expected


def test_parse_required_positive_int_rejects_non_positive():
    with pytest.raises(ValueError):
        parse_required_positive_int("0", "Positive")


def test_validate_enterprise_org_returns_cleaned_dataclass():
    result = validate_enterprise_org(
        company_name=" Atlas ",
        company_domain=" example.com ",
        primary_contact=" Owner ",
        tenant_id=" tenant123 ",
    )

    assert result.company_name == "Atlas"
    assert result.company_domain == "example.com"
    assert result.primary_contact == "Owner"
    assert result.tenant_id == "tenant123"


@pytest.mark.parametrize(
    "kwargs,expected_message",
    [
        (
            {
                "company_name": "",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
            },
            "Company name is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
            },
            "Company domain is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "",
                "tenant_id": "tenant123",
            },
            "Primary contact is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": " ",
            },
            "Tenant ID is required",
        ),
    ],
)
def test_validate_enterprise_org_errors(kwargs, expected_message):
    with pytest.raises(ValueError) as excinfo:
        validate_enterprise_org(**kwargs)

    assert expected_message in str(excinfo.value)
