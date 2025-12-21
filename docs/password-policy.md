---
audience: Administrators and security
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/user_accounts/user_account_service.py
---

# Password Policy Configuration

ATLAS enforces a password policy for local user accounts. Deployments can
customise the requirements by setting the following configuration keys in
`config.yaml` or via environment variables. Any values omitted fall back to the
secure defaults bundled with the application.

| Key | Description | Default |
| --- | --- | --- |
| `ACCOUNT_PASSWORD_MIN_LENGTH` | Minimum password length in characters. Provide a positive integer. | `10` |
| `ACCOUNT_PASSWORD_REQUIRE_UPPERCASE` | Require at least one uppercase letter (`A-Z`). Accepts truthy/falsey strings such as `"true"`/`"false"`, numeric flags, or booleans. | `true` |
| `ACCOUNT_PASSWORD_REQUIRE_LOWERCASE` | Require at least one lowercase letter (`a-z`). | `true` |
| `ACCOUNT_PASSWORD_REQUIRE_DIGIT` | Require at least one number (`0-9`). | `true` |
| `ACCOUNT_PASSWORD_REQUIRE_SYMBOL` | Require at least one symbol (for example `!`, `@`, or `#`). | `true` |
| `ACCOUNT_PASSWORD_FORBID_WHITESPACE` | Disallow spaces or other whitespace characters in passwords. | `true` |

Values supplied for boolean options are interpreted case-insensitively. Accepted
truthy values are `true`, `yes`, `on`, and `1`; falsey values include `false`,
`no`, `off`, and `0`. Any invalid values are ignored with a warning and the
default behaviour is preserved.

After updating the configuration, restart ATLAS to apply the new policy. The UI
will automatically refresh its password guidance text the next time the account
dialog is opened.
