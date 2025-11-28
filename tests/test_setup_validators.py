import pytest

from GTKUI.Setup.wizard.validators import (
    parse_optional_int,
    parse_required_float,
    parse_required_int,
    parse_required_positive_int,
    validate_company_identity,
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


def test_validate_company_identity_returns_cleaned_dataclass():
    result = validate_company_identity(
        company_name=" Atlas ",
        company_domain=" example.com ",
        primary_contact=" Owner ",
        contact_email=" CONTACT@example.com ",
        address_line1=" 123 Example Street ",
        address_line2=" Suite 100 ",
        city=" Gotham ",
        state=" NY ",
        postal_code=" 10001 ",
        country=" USA ",
        phone_number=" +1 (212) 555-1234 ",
    )

    assert result.company_name == "Atlas"
    assert result.company_domain == "example.com"
    assert result.primary_contact == "Owner"
    assert result.contact_email == "contact@example.com"
    assert result.address_line1 == "123 Example Street"
    assert result.address_line2 == "Suite 100"
    assert result.city == "Gotham"
    assert result.state == "NY"
    assert result.postal_code == "10001"
    assert result.country == "USA"
    assert result.phone_number == "+1 (212) 555-1234"


def test_validate_enterprise_org_returns_cleaned_dataclass():
    result = validate_enterprise_org(
        company_name=" Atlas ",
        company_domain=" example.com ",
        primary_contact=" Owner ",
        tenant_id=" tenant123 ",
        contact_email=" CONTACT@example.com ",
        address_line1=" 123 Example Street ",
        address_line2=" Suite 100 ",
        city=" Gotham ",
        state=" NY ",
        postal_code=" 10001 ",
        country=" USA ",
        phone_number=" +1 (212) 555-1234 ",
    )

    assert result.company_name == "Atlas"
    assert result.company_domain == "example.com"
    assert result.primary_contact == "Owner"
    assert result.tenant_id == "tenant123"
    assert result.contact_email == "contact@example.com"
    assert result.address_line1 == "123 Example Street"
    assert result.address_line2 == "Suite 100"
    assert result.city == "Gotham"
    assert result.state == "NY"
    assert result.postal_code == "10001"
    assert result.country == "USA"
    assert result.phone_number == "+1 (212) 555-1234"


@pytest.mark.parametrize(
    "kwargs,expected_message",
    [
        (
            {
                "company_name": "",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Company name is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "",
                "primary_contact": "Owner",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Company domain is required",
        ),
    ],
)
def test_validate_company_identity_errors(kwargs, expected_message):
    with pytest.raises(ValueError) as excinfo:
        validate_company_identity(**kwargs)

    assert expected_message in str(excinfo.value)


@pytest.mark.parametrize(
    "kwargs,expected_message",
    [
        (
            {
                "company_name": "",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Company name is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Company domain is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Primary contact is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": " ",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Tenant ID is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Contact email is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "contact-at-example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Contact email must be a valid email address",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": " ",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "Address line 1 is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": " ",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
            },
            "City or town is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": " ",
                "postal_code": "10001",
                "country": "USA",
            },
            "State or province is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": " ",
                "country": "USA",
            },
            "Postal code is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": " ",
            },
            "Country is required",
        ),
        (
            {
                "company_name": "Atlas",
                "company_domain": "example.com",
                "primary_contact": "Owner",
                "tenant_id": "tenant123",
                "contact_email": "owner@example.com",
                "address_line1": "123 Example Street",
                "city": "Gotham",
                "state": "NY",
                "postal_code": "10001",
                "country": "USA",
                "phone_number": "invalid",
            },
            "Phone number must include at least one digit",
        ),
    ],
)
def test_validate_enterprise_org_errors(kwargs, expected_message):
    with pytest.raises(ValueError) as excinfo:
        validate_enterprise_org(**kwargs)

    assert expected_message in str(excinfo.value)
