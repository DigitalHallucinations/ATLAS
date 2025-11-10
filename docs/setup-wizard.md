# ATLAS Setup Wizard

ATLAS ships with a guided GTK experience for first-time configuration. The
desktop shell launches a multi-step window that walks through the key decisions
required to bring the platform online. The administrator profile now opens the
wizard so that environment-specific defaults can be staged before any
infrastructure settings are applied.

### Administrator bootstrap

The first screen captures the primary administrator's profile so the wizard can
stage identity data before any infrastructure steps load. Full name, username,
email address, organization domain, and birth date are written to a temporary
state object that seeds later forms: the PostgreSQL step proposes the staged
username for the database role, optional tenancy fields reuse the normalized
domain, and password policy hints pull from the captured email address. The
same staging process stores the administrator's chosen password and privileged
sudo credentials so the database bootstrapper and follow-on CLI helpers can
reuse them without asking the operator twice. Once the environment is ready,
the staged profile is registered and the setup marker written. Review the
[user account management guide](./user-accounts.md) and the
[developer setup runbook](./ops/developer-setup.md) for the onboarding
material that follows this administrator bootstrap.

### Step-by-step configuration

1. **Administrator** – Collect the profile for the first user. The form requires
   a full name, username, email address, organization domain, date of birth,
   password, and privileged sudo credentials. The wizard stages this
   information instead of registering the account immediately. Subsequent pages
   reuse the staged data; for example the database user defaults to the staged
   username and the optional settings tenant field inherits the normalized
   domain. Privileged database credentials captured later in the flow are also
   stored so they can be reused by the CLI and GTK flows.
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
8. **Optional settings** – Capture tenancy, retention, scheduler overrides, and
   whether the HTTP server should auto-start.

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
* **Optional settings** records tenant IDs, conversation retention limits,
  scheduler overrides, and the HTTP auto-start flag.

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
