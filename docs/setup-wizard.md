# ATLAS Setup Wizard

ATLAS ships with a guided GTK experience for first-time configuration. The
desktop shell launches a multi-step window that walks through the key decisions
required to bring the platform online. The administrator profile now opens the
wizard so that environment-specific defaults can be staged before any
infrastructure settings are applied.

### Introduction and branching

The wizard opens with an introduction followed by a setup type choice. Picking
**User (Personal)** advances directly to the **Admin** step, where you stage the
first account. Choosing **Company (Enterprise)** routes you through a combined
company step: tenancy, retention, and scheduler defaults are captured first,
then the initial administrative account is staged. Either path retains the
staged credentials so the remaining infrastructure pages can reuse them without
repeated prompts. Once the environment is ready, the staged profile is
registered and the setup marker written. Review the
[user account management guide](./user-accounts.md) and the
[developer setup runbook](./ops/developer-setup.md) for the onboarding material
that follows this branching step.

### Step-by-step configuration

> **Callout:** Prefer to script the bootstrap? Run `python3 scripts/setup_atlas.py`
> to mirror the wizard's prompts from the terminal. Larger rollouts often
> pre-seed configuration with the CLI before handing the final review to an
> administrator in the GTK flow—see the [Standalone CLI utility](#standalone-cli-utility)
> section for the full walkthrough.

1. **Admin or Company** – Personal setups land on the admin page to collect a
   full name, username, email address, organization domain, date of birth,
   password, and privileged sudo credentials. Enterprise setups route through a
   company overview that captures tenant, retention, scheduler, and HTTP
   defaults before presenting the same admin credentials form. The wizard stages
   this information instead of registering the account immediately. Subsequent
   pages reuse the staged data; for example the database user defaults to the
   staged username and the tenant field inherits the normalized domain from the
   earlier company page. Privileged database credentials captured later in the
   flow are also stored so they can be reused by the CLI and GTK flows.
2. **Database** – Collect PostgreSQL connection details and run the conversation
   store bootstrap helpers.
3. **Job scheduling** – Enable durable scheduling, configure retry policy, and
   provide job store details when needed.
4. **Message bus** – Choose between the in-memory or Redis backends and supply
   any Redis connection details.
5. **Key-value store** – Decide whether to reuse the conversation store and set
   an alternate DSN when required.
6. **Providers** – Store default model/provider selections along with any API
   keys that should be written to the configuration.
7. **Speech** – Toggle text-to-speech and speech-to-text integrations and record
   provider defaults and API keys.

Progress and any validation errors are surfaced inline, and the wizard invokes
the same controller methods used by the CLI to persist settings. The final step
registers the staged administrator once all configuration has been applied and
the setup marker written.

### User setup

Smaller rollouts often stop at the administrator profile, but the wizard also
streamlines onboarding for up to five local users. Once the environment is
provisioned, open **Settings → Accounts** in the GTK shell to launch the same
registration panel used during bootstrap. Selecting **Add user** walks through a
condensed version of the administrator form: enter the person's full name,
username, email address, and password, then confirm to register the account.
Behind the scenes the wizard reuses `UserAccountService.register_user`, so the
new credential is hashed with PBKDF2, policy checks run, and the profile is tied
to the conversation store automatically. Review the broader
[user account management guide](./user-accounts.md) for API examples and
automation workflows.

Automation scripts can register additional users directly with
`register_user`—pass the normalized username, email, password, and optional
profile metadata to mirror the GTK experience. This is the recommended path when
bootstrapping small teams that prefer to seed accounts during provisioning
rather than waiting for users to sign in interactively.

* **Shared safeguards** – Local onboarding and automation reuse the same PBKDF2
  hashing, login attempt auditing, and lockout handling provided by
  `UserAccountService`, so small teams inherit enterprise controls even before
  directory integrations come online.【F:modules/user_accounts/user_account_service.py†L65-L195】【F:docs/user-accounts.md†L6-L20】
* **Tighten password complexity when scaling** – Raise
  `ACCOUNT_PASSWORD_MIN_LENGTH` and confirm the uppercase, lowercase, digit, and
  symbol requirements surfaced by `UserAccountService` before inviting larger
  cohorts. The service reads these toggles from configuration (see
  `ACCOUNT_PASSWORD_REQUIRE_UPPERCASE`, `ACCOUNT_PASSWORD_REQUIRE_LOWERCASE`,
  `ACCOUNT_PASSWORD_REQUIRE_DIGIT`, `ACCOUNT_PASSWORD_REQUIRE_SYMBOL`, and
  `ACCOUNT_PASSWORD_FORBID_WHITESPACE`), and the [password policy
  reference](./password-policy.md) outlines enforcement guidance for each flag.【F:modules/user_accounts/user_account_service.py†L473-L518】【F:docs/password-policy.md†L3-L24】

#### Password requirements

The wizard mirrors the configurable password policy defined in
`config.yaml`. Administrators can adjust the following keys to tune enforcement:

| Key | Purpose | Default |
| --- | --- | --- |
| `ACCOUNT_PASSWORD_MIN_LENGTH` | Minimum password length in characters. | `10` |
| `ACCOUNT_PASSWORD_REQUIRE_UPPERCASE` | Require at least one uppercase letter. | `true` |
| `ACCOUNT_PASSWORD_REQUIRE_LOWERCASE` | Require at least one lowercase letter. | `true` |
| `ACCOUNT_PASSWORD_REQUIRE_DIGIT` | Require at least one number. | `true` |
| `ACCOUNT_PASSWORD_REQUIRE_SYMBOL` | Require at least one symbol. | `true` |
| `ACCOUNT_PASSWORD_FORBID_WHITESPACE` | Disallow whitespace characters. | `true` |

Regardless of how an account is created, passwords are salted and hashed before
storage, and repeated failures trigger lockouts that mirror the behaviour
documented in the [user account service reference](./user-accounts.md). Reset
tokens issued through the automation or GTK flows inherit those protections so
small teams get the same auditing, recovery, and credential hygiene as larger
deployments. These safeguards remain important even when operating near the
five-user ceiling because they mitigate credential reuse, shoulder surfing, and
accidental lockouts in lightweight deployments.

### Step details

Each step mirrors the questions asked by the CLI setup utility:

* **Job scheduling** presents toggles and numeric inputs for retry attempts,
  backoff values, worker counts, queue size, and timezone. Invalid numeric
  entries are rejected just as they are at the command line.
* **Message bus** exposes a backend selector. Choosing Redis enables fields for
  the Redis DSN and stream prefix, while the in-memory option clears them.
* **Key-value store** allows you to reuse the conversation database or provide
  a dedicated SQLAlchemy-compatible DSN.
* **Speech** collects provider defaults and API keys for ElevenLabs, OpenAI, and
  Google, along with toggles for TTS and STT support.
* **Company defaults (enterprise path)** capture tenant IDs, conversation
  retention limits, scheduler overrides, and the HTTP auto-start flag before you
  move into the admin credentials form.

### Company setup for enterprise rollouts

Selecting the **Company (Enterprise)** path front-loads tenancy, retention, and
scheduler defaults before fleets begin connecting. The **Tenant identifier**
field writes directly to the root `tenant_id` setting so API calls inherit the
same context enforced by server routes that require the `X-Atlas-Tenant` header
for every request.【F:ATLAS/setup/controller.py†L433-L451】【F:docs/server/api.md†L34-L59】 Choose a **Primary data region** and **Residency
requirement** so downstream services can honor regional handling expectations;
both values are stored under the `data_residency` block for reuse by scheduling
and storage modules.【F:ATLAS/setup/controller.py†L430-L453】【F:ATLAS/config/atlas_config.yaml†L100-L126】 Use the retention inputs to populate the
 `conversation_database.retention` block (`days` and `history_message_limit`),
ensuring pruning behaviour is consistent with the retention worker documented in
the [conversation retention runbook](./conversation_retention.md).【F:ATLAS/setup/controller.py†L433-L451】【F:ATLAS/config/atlas_config.yaml†L108-L118】【F:docs/conversation_retention.md†L1-L34】
Timezone and queue size controls map to `job_scheduling.timezone` and
`job_scheduling.queue_size`, allowing operators to override the shared scheduler
defaults that also drive task queue sizing.【F:ATLAS/setup/controller.py†L433-L451】【F:ATLAS/config/config_manager.py†L600-L720】

Export controls and data residency considerations captured in this step are also
summarised in the [export controls reference](./export-controls.md), which
outlines regional handling expectations and how the HTTP gateway's DLP policies
protect regulated data before it is stored.【F:docs/export-controls.md†L1-L34】【F:modules/Server/routes.py†L3177-L3240】

For multi-tenant and auditing-heavy environments, complete the wizard and then
review the configuration in `atlas_config.yaml` alongside the logging/audit
modules. Tighten policies by setting stricter retention windows, enabling
`tenant_limits` through the conversation store, and configuring audit sinks via
`modules.logging.audit` to capture persona, skill, and API activity with tenant
context.【F:modules/conversation_store/repository.py†L2622-L2765】【F:modules/logging/audit.py†L33-L126】 Server-side enforcement continues to require tenant-scoped contexts, so routing
modules remain aligned with the staged identifier even as you expand to
additional tenants.【F:modules/Server/conversation_routes.py†L30-L209】【F:modules/Server/task_routes.py†L33-L252】

* **Shared safeguards** – Tenancy defaults, retention windows, and scheduler
  overrides surfaced in the company step write directly to the same
  configuration read by CLI automation and background workers, keeping fleet and
  bootstrap flows in lockstep as environments grow.【F:ATLAS/setup/controller.py†L433-L451】【F:ATLAS/config/config_manager.py†L600-L720】
* **Tighten retention and residency policies** – Increase
  `conversation_database.retention.days`, lower
  `conversation_database.retention.history_message_limit`, and align these
  values with the [conversation retention runbook](./conversation_retention.md)
  when serving regulated departments.【F:ATLAS/config/atlas_config.yaml†L102-L116】【F:docs/conversation_retention.md†L1-L34】
* **Enable richer audit hooks** – Connect `modules.logging.audit` sinks to your
  preferred SIEM or compliance pipeline after completing the wizard so persona,
  skill, and API activity is captured with tenant context beyond the default
  local logging path.【F:modules/logging/audit.py†L33-L126】【F:modules/conversation_store/repository.py†L2622-L2765】

#### Audit and retention templates

The company step also exposes audit/retention templates so operators can align
defaults with their compliance posture without memorising every field. Pick an
option to pre-fill sink destinations and retention expectations while tooltips
capture the intended reviewer audience:

* **SIEM handoff (30d / 500 msgs)** – Keeps a 30-day JSONL buffer and writes
  persona/skill audit events to SIEM-friendly files so security teams can ingest
  them continuously.【F:modules/logging/audit_templates.py†L15-L37】【F:modules/logging/audit_templates.py†L58-L66】
* **Privacy minimised (14d / 200 msgs)** – Shortens local exposure for
  privacy-focused tenants with smaller buffers and history caps.【F:modules/logging/audit_templates.py†L38-L50】
* **Extended review (90d / 1500 msgs)** – Retains longer audit evidence for
  quarterly or regulated review cycles while keeping persona and skill changes
  isolated per file.【F:modules/logging/audit_templates.py†L51-L66】

The selection is saved alongside the tenant ID, scheduler defaults, and HTTP
startup preference so server bootstrapping can configure audit sinks before any
requests are handled.【F:ATLAS/setup/controller.py†L730-L749】【F:modules/Server/routes.py†L131-L170】

## Preparing the Python environment

Before running the setup wizard, create or update the virtual environment and
install dependencies:

```bash
python3 scripts/install_environment.py
```

The helper accepts an optional `--python` flag if you need to target a specific
interpreter. It reuses the wizard's existing bootstrap logic to manage
`.venv/` and install `requirements.txt`.

## Using the GTK wizard

Launching `main.py` without an existing configuration automatically presents the
GTK wizard. You can also start it manually by running the desktop shell and
choosing **Run setup wizard** from the application menu.

When the final step succeeds the wizard now writes the completion marker and
returns you to the shell immediately, so there's no longer a need to restart
ATLAS before signing in with the administrator account.

If the wizard cannot connect to PostgreSQL it now mirrors the CLI utility by
prompting for a privileged username and password. Supplying credentials lets
ATLAS create the database and role automatically; leave the fields blank to
retry without privileged provisioning. Validation failures in any step keep you
on the current page so you can fix the input without losing progress.

### Debug log window

Operators can monitor setup activity in real time through the debug log window.
The header bar includes a circular bug icon on the left edge; selecting it
opens a dedicated inspector that streams log output from the wizard controller
and its supporting bootstrap helpers. The window mirrors the wizard's lighter
header styling, presents log lines in a monospace view, and highlights
timestamps, log levels, logger names, and messages. Use it to verify connection
attempts, database provisioning, and any validation errors that are logged but
not surfaced inline. Selecting the icon again hides the window and detaches the
temporary log handler.

## Standalone CLI utility

The original CLI workflow remains available at `scripts/setup_atlas.py`. It
mirrors the wizard's state machine, making it useful for headless servers or
automated provisioning workflows.

```bash
python3 scripts/setup_atlas.py
```

## Testing

Unit tests covering the GTK wizard live in `tests/test_setup_wizard.py`. They
exercise the happy path and validation rules for every step by faking the
controller layer. The CLI helpers remain covered by `tests/test_setup_cli.py`.
