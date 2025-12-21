---
audience: Operators and backend developers
status: in_review
last_verified: 2025-12-21
source_of_truth: modules/conversation_store/repository.py; modules/user_accounts/user_account_service.py
---

# User Account Management

ATLAS stores local user accounts alongside conversations in the PostgreSQL conversation store. The user account service wraps that persistence layer with helpers for password policy enforcement, reset tokens, login attempt tracking, and synchronisation with the active user configuration used by the shell and automation services.

## Storage model
- **Conversation store tables** – User credentials, reset tokens, and login history live in the conversation store via `ConversationStoreRepository`. Each account persists the username, PBKDF2 password hash, email, optional profile fields, and lockout metadata when `create_user_account` writes a `UserCredential` row.【F:modules/conversation_store/repository.py†L623-L714】
- **Cross-linking to profiles** – Credentials are associated with the broader conversation user profile by attaching the credential row to the matching `User` entity. `ensure_user` creates or updates the profile metadata and `attach_credential` pins the credential to that user record so activity and personalization data stay aligned.【F:modules/conversation_store/repository.py†L623-L678】【F:modules/conversation_store/repository.py†L1040-L1120】
- **Lockout and login history** – Consecutive failures are stored on the credential row (`failed_attempts`, `lockout_until`) and individual events are appended to the `UserLoginAttempt` table so administrators can audit authentication patterns.【F:modules/conversation_store/repository.py†L780-L911】

## Password hashing and verification
- **PBKDF2 with SHA-256** – Passwords are salted and hashed with 100 000 PBKDF2 iterations when new accounts are registered or passwords are updated. Hash strings include the algorithm, iteration count, salt, and digest for future verification.【F:modules/user_accounts/user_account_service.py†L65-L88】
- **Server-side verification** – Authentication requests re-hash the candidate password and compare it to the stored digest via `verify_user_password`, ensuring credential checks are handled in the service layer rather than the UI.【F:modules/user_accounts/user_account_service.py†L186-L190】【F:modules/user_accounts/user_account_service.py†L1143-L1221】
- **Password policy integration** – `UserAccountService` applies configurable requirements (minimum length, character classes, whitespace rules) before accepting a password. Administrators can tune these rules as described in the [password policy reference](password-policy.md).【F:modules/user_accounts/user_account_service.py†L1010-L1093】【F:docs/password-policy.md†L1-L24】

## Password reset tokens
- **Token creation** – Reset tokens are generated with `secrets.token_urlsafe`, hashed with SHA-256, and persisted alongside their expiration timestamp when `create_password_reset_token` delegates to the repository.【F:modules/user_accounts/user_account_service.py†L1322-L1363】【F:modules/user_accounts/user_account_service.py†L91-L101】【F:modules/conversation_store/repository.py†L939-L993】
- **Verification flow** – `verify_password_reset_token` re-hashes the candidate token, verifies the stored expiration, and clears expired entries to prevent reuse. Successful resets update the password hash, delete the token, and reset failure counters.【F:modules/user_accounts/user_account_service.py†L1364-L1421】

## Login attempt tracking and lockouts
- **Recording attempts** – Every authentication attempt captures the username (or email), timestamp, success flag, and failure reason. These rows can be queried or pruned to support auditing and rate limiting policies.【F:modules/conversation_store/repository.py†L865-L938】【F:modules/user_accounts/user_account_service.py†L1143-L1221】
- **Adaptive lockouts** – In-memory caches mirror the stored failure counts so `authenticate_user` can enforce lockout windows and report retry times via `AccountLockedError`. Successful logins clear both the persistent and in-memory failure tracking, keeping credentials responsive once the user recovers access.【F:modules/user_accounts/user_account_service.py†L1143-L1218】【F:modules/user_accounts/user_account_service.py†L1322-L1421】
- **Operational review** – Administrators can inspect recent failures with `get_login_attempts` or reset the counters with `clear_lockout_state` when resolving account lockouts. Each credential record exposes both aggregated (`failed_attempts`) and detailed event history.【F:modules/conversation_store/repository.py†L780-L911】

## Active user configuration hooks
The GTK shell and orchestration services rely on the “active user” selection in `ConfigManager`. `UserAccountService` keeps that setting in sync by validating the username, updating the stored configuration entry, and clearing it automatically when the active account is deleted.【F:modules/user_accounts/user_account_service.py†L1229-L1273】 Synchronisation with the conversation store also keeps profile metadata aligned after login updates.【F:modules/user_accounts/user_account_service.py†L1127-L1217】

## Operational guidance
### Creating and updating accounts
- Use the account dialog or automation that calls `register_user` to create accounts. The service normalises identifiers, validates emails, and synchronises profile metadata in the conversation store after each registration.【F:modules/user_accounts/user_account_service.py†L1098-L1141】
- When updating credentials, provide the existing password so the service can verify ownership before hashing and persisting the new secret. `update_user` enforces this requirement and records the audit trail via the logger.【F:modules/user_accounts/user_account_service.py†L1275-L1320】

### Enforcing password policies
- Review and adjust password requirements in `config.yaml` using the keys listed in the [password policy reference](password-policy.md). Changes are reflected the next time the service initialises its policy cache.【F:modules/user_accounts/user_account_service.py†L996-L1093】【F:docs/password-policy.md†L1-L24】
- Surface policy guidance to users through UI messaging or onboarding materials so resets and registrations succeed on the first attempt.

### Troubleshooting duplicates and locks
- Attempts to create an account with an existing username or email raise `DuplicateUserError`. Catch this exception in automation or review the logs for the bundled message (“A user with the same username or email already exists.”) to identify conflicts.【F:modules/user_accounts/user_account_service.py†L38-L155】
- When operators receive lockout reports, inspect the aggregated state via `get_all_lockout_states` and clear credentials with `clear_lockout_state` after verifying the user’s identity. For persistent issues, prune historical attempts to remove stale failures.【F:modules/conversation_store/repository.py†L808-L938】

### Managing reset flows and stale tokens
- If a user reports an invalid reset token, use `verify_password_reset_token` to confirm expiry and `delete_password_reset_token` to issue a fresh challenge. Expired tokens are automatically removed when verification runs, but administrators can proactively clear tokens if they suspect compromise.【F:modules/user_accounts/user_account_service.py†L1364-L1421】【F:modules/conversation_store/repository.py†L939-L1001】

### Handling locked or duplicate active users
- Changing or removing the active user setting uses `set_active_user` and `delete_user`. If a locked account is the active user, clearing the configuration entry lets operators switch personas or credentials without editing the YAML manually.【F:modules/user_accounts/user_account_service.py†L1229-L1273】
- When removing duplicate credentials, delete the redundant account and rely on the automatic configuration cleanup to drop the active selection if it referenced that user.【F:modules/user_accounts/user_account_service.py†L1248-L1273】
