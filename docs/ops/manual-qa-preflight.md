# Manual QA – GTK preflight checks (Debian/Ubuntu)

These manual checks confirm the new GTK setup wizard preflight workflow on Debian- or Ubuntu-based systems.

## Prerequisites

- GNOME dependencies for the GTK wizard installed (`python3-gi`, `gir1.2-gtk-4.0`).
- The `atlas` repository cloned and `python3 -m venv .venv` created in the project root.
- PostgreSQL (`postgresql` package) and Redis (`redis-server` package) installed locally, both managed through `systemd`.
- A sudo-capable account for running privileged commands; the password should be known for testing.

## Test steps

1. Start PostgreSQL and Redis (`sudo systemctl start postgresql redis-server`) and ensure `.venv` exists.
2. Launch the GTK setup wizard (`python3 main.py`) and open the **Run preflight checks** button on the Introduction screen.
3. Trigger the scan while services are running. All rows should report success and the dialog should close normally.
4. Stop PostgreSQL and Redis (`sudo systemctl stop postgresql redis-server`) and temporarily move `.venv` aside (`mv .venv .venv.bak`).
5. Run the preflight scan again. Each missing dependency should render a failure row with actionable guidance and a Fix button for PostgreSQL and Redis.
6. Click the Fix button for PostgreSQL, supply the sudo password prompt, and verify the row re-checks to success. Repeat for Redis.
7. Restore the virtual environment (`mv .venv.bak .venv`) and re-run the scan to confirm the Project virtualenv row returns to success without needing a fix.
8. Observe that the wizard status text updates throughout (running, failures found, all passed) without crashing.

## Cleanup

- Ensure PostgreSQL and Redis are left running if they were active before the test (`sudo systemctl start postgresql redis-server`).
- Exit the setup wizard.
