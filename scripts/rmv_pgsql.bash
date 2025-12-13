# 0) Become root (optional; or prefix sudo on each line)
sudo -v

# 1) Stop any running PostgreSQL services (all versions/instances)
sudo systemctl stop postgresql.service 2>/dev/null || true
sudo systemctl stop 'postgresql@*.service' 2>/dev/null || true

# 2) Drop ALL clusters (this deletes all DB data managed by postgresql-common)
#    If pg_dropcluster isn't installed, this will just no-op.
sudo pg_dropcluster --stop --all 2>/dev/null || true

# 3) Purge all PostgreSQL-related packages (server, clients, libs, extras)
sudo apt-get update
sudo apt-get purge -y \
  'postgresql*' \
  'libpq*' \
  'postgresql-client*' \
  'postgresql-common' \
  'pgadmin4*' \
  'odbc-postgresql*' \
  'postgis*'

# 4) Remove automatically installed dependencies and clean apt caches
sudo apt-get autoremove -y --purge
sudo apt-get autoclean -y

# 5) Delete remaining data, configs, logs, sockets, and shared files
#    (These paths cover common locations for multiple versions)
sudo rm -rf \
  /var/lib/postgresql \
  /var/log/postgresql \
  /var/run/postgresql \
  /etc/postgresql \
  /etc/postgresql-common \
  /usr/lib/postgresql \
  /usr/share/postgresql

# 6) Remove the postgres system user and group if they still exist
#    (Safe to ignore errors if they're already gone.)
sudo deluser --quiet --system postgres 2>/dev/null || true
sudo groupdel postgres 2>/dev/null || true

# 7) Ensure systemd state is up to date
sudo systemctl daemon-reload

# 8) Quick sanity checks (should show nothing or "no such user")
dpkg -l | grep -i postgres || echo "No PostgreSQL packages found."
getent passwd postgres || echo "No 'postgres' user found."
