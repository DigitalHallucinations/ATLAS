# ATLAS Setup Wizard

ATLAS ships with a guided GTK experience for first-time configuration. The
desktop shell launches a multi-step window that walks through the key decisions
required to bring the platform online:

1. **Database** – Collect PostgreSQL connection details and run the conversation
   store bootstrap helpers.
2. **Providers** – Store default model/provider selections along with any API
   keys that should be written to the configuration.
3. **Administrator** – Register the first user account so you can sign in after
   the wizard finishes.

Progress and any validation errors are surfaced inline, and the wizard invokes
the same controller methods used by the CLI to persist settings.

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
retry without privileged provisioning.

## Standalone CLI utility

The original CLI workflow remains available at `scripts/setup_atlas.py`. It
mirrors the wizard's state machine, making it useful for headless servers or
automated provisioning workflows.

```bash
python3 scripts/setup_atlas.py
```

## Testing

Unit tests covering the GTK wizard live in `tests/test_setup_wizard.py`. They
exercise the happy path and validation rules by faking the controller layer.
The CLI helpers remain covered by `tests/test_setup_cli.py`.
