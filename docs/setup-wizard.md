# ATLAS Standalone Setup Utility

The first-run experience for ATLAS is now driven by the standalone CLI located
at `scripts/setup_atlas.py`. Operators should execute the setup utility before
launching `main.py`; the GTK desktop shell will refuse to start until the setup
sentinel is written.

## Preparing the Python environment

Before running the setup wizard, create or update the virtual environment and
install dependencies:

```bash
python3 scripts/install_environment.py
```

The helper accepts an optional `--python` flag if you need to target a specific
interpreter. It reuses the wizard's existing bootstrap logic to manage
`.venv/` and install `requirements.txt`.

## Running the utility

```bash
python3 scripts/setup_atlas.py
```

Once dependencies are installed, the wizard performs the following tasks:

1. **PostgreSQL provisioning** – Offers platform-specific commands to install
   PostgreSQL, then uses the existing bootstrap helpers to create the database
   and role. The confirmed DSN is persisted to `config.yaml`.
2. **Messaging and services** – Collects configuration for the key-value store,
   durable scheduling, and the message bus, reusing the same validation logic as
   the former GTK wizard.
3. **Providers & speech** – Prompts for API keys, default model/provider
   selections, and text/speech services.
4. **Administrator user** – Registers the first user through
   `UserAccountService` once the conversation store is reachable.
5. **Optional tuning** – Captures tenant identifiers, retention policies, and
   HTTP auto-start preferences.

After applying the collected settings, the utility writes
`ATLAS/setup/config/setup_complete.json`. `main.py` verifies the sentinel before
initialising the GTK UI.

## GTK front-end

A lightweight GTK window remains available for desktop environments. It simply
launches the standalone utility and surfaces the exit status, keeping the UI in
sync with the CLI workflow.

## Testing

Unit tests covering the CLI helpers live in `tests/test_setup_cli.py`. They mock
subprocess and network calls to assert DSN persistence, user registration, and
sentinel writing behaviour.
