# ATLAS Standalone Setup Utility

The first-run experience for ATLAS is now driven by the standalone CLI located
at `scripts/setup_atlas.py`. Operators should execute the setup utility before
launching `main.py`; the GTK desktop shell will refuse to start until the setup
sentinel is written.

## Running the utility

```bash
python3 scripts/setup_atlas.py
```

The utility performs the following tasks:

1. **Python environment** – Creates/updates a `.venv` inside the repository and
   installs dependencies from `requirements.txt`.
2. **PostgreSQL provisioning** – Offers platform-specific commands to install
   PostgreSQL, then uses the existing bootstrap helpers to create the database
   and role. The confirmed DSN is persisted to `config.yaml`.
3. **Messaging and services** – Collects configuration for the key-value store,
   durable scheduling, and the message bus, reusing the same validation logic as
   the former GTK wizard.
4. **Providers & speech** – Prompts for API keys, default model/provider
   selections, and text/speech services.
5. **Administrator user** – Registers the first user through
   `UserAccountService` once the conversation store is reachable.
6. **Optional tuning** – Captures tenant identifiers, retention policies, and
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
