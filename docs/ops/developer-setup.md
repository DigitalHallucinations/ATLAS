---
audience: Contributors and developers
status: in_review
last_verified: 2025-12-21
source_of_truth: scripts/install_environment.py; scripts/setup_atlas.py; main.py
---

# Developer Environment Setup

This runbook walks through preparing a local development environment for ATLAS. It covers creating the Python virtual environment, installing dependencies, provisioning configuration through the CLI helper, and choosing when to rely on the GTK setup wizard instead.

## Prerequisites
- Python 3.10 or newer available on your path.
- GTK build headers and introspection libraries installed locally.
  - **Debian/Ubuntu**
    ```bash
    sudo apt install libgtk-4-dev libadwaita-1-dev gobject-introspection gir1.2-gtk-4.0
    ```
  - **Fedora**
    ```bash
    sudo dnf install gtk4-devel libadwaita-devel gobject-introspection-devel
    ```
  - **macOS (Homebrew)**
    ```bash
    brew install gtk4 libadwaita gobject-introspection
    ```
  - Verify the bindings are visible to Python:
    ```bash
    python -c "import gi"
    ```
- Access to the PostgreSQL instance you intend to use for development (local or remote).
- Optional: Redis if you plan to exercise the Redis-backed message bus.

## Bootstrap the virtual environment
Run the environment installer from the repository root to create `.venv/` and install Python dependencies:

```bash
python3 scripts/install_environment.py
```

Running without additional flags installs the base runtime and skips GPU toolchains. Pass
`--with-accelerators` if you plan to fine-tune Hugging Face models or run local Whisper
speech-to-text and need the optional `requirements-accelerators.txt` dependencies installed
automatically:

```bash
python3 scripts/install_environment.py --with-accelerators
```

You can also install the extras manually later with `pip install -r requirements-accelerators.txt`
when you need Torch, Hugging Face fine-tuning stacks, or local Whisper tooling.

> 🔈 GPT-4o live speech-to-text capture depends on the `sounddevice` and `soundfile` Python
> packages that ship with the base requirements. Ensure your OS provides the PortAudio
> runtime (for example, `sudo apt install libportaudio2` on Debian/Ubuntu) so microphone
> recording works end-to-end.

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

When prompted for a preset, pick the profile that matches your target:

- **Personal** – local-only experimentation with minimal retention and in-memory queues.
- **Enterprise** – shared Redis/PostgreSQL backends, 30-day retention, and SIEM-friendly auditing defaults.
- **Regulatory** – in-region data residency with year-long retention and Azure OpenAI defaults for export-controlled environments.

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

## Manage local user accounts
Account creation, password policies, and lockout recovery are handled by the user account service. Review the [user account management guide](../user-accounts.md) for details on how credentials are stored, how reset tokens are issued, and how operators should respond to duplicate or locked accounts before onboarding new teammates.【F:docs/user-accounts.md†L1-L45】

## Learn the GTK shell
Review the [GTK UI overview](../ui/gtk-overview.md) for a tour of the main window layout, embedded managers, and extension points once your environment is ready.
