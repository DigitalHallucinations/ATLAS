# Developer Environment Setup

This runbook walks through preparing a local development environment for ATLAS. It covers creating the Python virtual environment, installing dependencies, provisioning configuration through the CLI helper, and choosing when to rely on the GTK setup wizard instead.

## Prerequisites
- Python 3.10 or newer available on your path.
- Access to the PostgreSQL instance you intend to use for development (local or remote).
- Optional: Redis if you plan to exercise the Redis-backed message bus.

## Bootstrap the virtual environment
Run the environment installer from the repository root to create `.venv/` and install Python dependencies:

```bash
python3 scripts/install_environment.py
```

Pass `--python` when you need to target a specific interpreter. For example, on systems with multiple Python installations:

```bash
python3 scripts/install_environment.py --python=/usr/bin/python3.11
```

The script is idempotent and safe to re-run whenever `requirements.txt` changes.

### Activating the environment
After the installer completes, activate the virtual environment before running tooling, tests, or the GTK shell:

```bash
source .venv/bin/activate
```

The prompt will gain a `(.venv)` prefix. Use `deactivate` to exit the environment when you finish working.

## Run the CLI setup helper
With the virtual environment active, execute the CLI setup workflow to populate configuration and database defaults:

```bash
python scripts/setup_atlas.py
```

Follow the prompts to supply PostgreSQL credentials, message-bus preferences, and any optional services. The CLI mirrors the GTK wizard’s state machine but is optimized for headless or automated environments.

### When to use the GTK setup wizard
Launch the GTK setup wizard instead of the CLI when you:
- Prefer a guided graphical workflow that validates each step inline.
- Need to capture configuration interactively during first-run of the desktop shell.
- Are provisioning a workstation where GTK is already available and you want to verify UI integrations.

Start the wizard by running `python3 main.py` without an existing configuration, or choose **Run setup wizard** from the shell’s menu after launch.

## Troubleshooting
- **Missing system dependencies** – Re-run `scripts/install_environment.py` after installing build tools or headers that were previously absent. The script will reinstall any failed wheels.
- **Broken virtual environment** – Remove the `.venv/` directory and re-run the installer if activation fails or packages are inconsistent.
- **Database or Redis connection errors** – Confirm services are reachable and credentials are correct, then re-run `scripts/setup_atlas.py`. The helper only writes configuration once the validation step succeeds.
- **Switching Python versions** – Recreate the virtual environment with the appropriate interpreter using the `--python` flag, then re-run setup to ensure compiled dependencies match the new version.

After resolving issues, re-run the relevant installer or setup helper to verify that configuration and dependencies were written successfully.
