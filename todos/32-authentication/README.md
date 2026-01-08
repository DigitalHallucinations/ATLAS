# Authentication & Session Management (SOTA)

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 1-2 weeks  
> **Created**: 2026-01-08

---

## Overview

Implement state-of-the-art authentication, session management, and identity federation capabilities:

### Research References

- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)
- [OWASP Session Management Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)
- [NIST Digital Identity Guidelines (SP 800-63B)](https://pages.nist.gov/800-63-3/sp800-63b.html)
- [WebAuthn Specification](https://www.w3.org/TR/webauthn-2/)

---

## Core Concepts

### 1. Password Hashing Modernization

Migrate from PBKDF2 to Argon2id (memory-hard, side-channel resistant):
- Winner of Password Hashing Competition
- OWASP recommended default
- Configurable memory/time/parallelism parameters

### 2. Multi-Factor Authentication (MFA)

Support multiple second factors:
- **TOTP** (RFC 6238) - Time-based one-time passwords
- **WebAuthn/FIDO2** - Hardware keys, biometrics
- **Recovery codes** - Backup access method

### 3. Session Security

Modern session management:
- Short-lived access tokens with refresh rotation
- Device fingerprinting and binding
- Concurrent session limits
- Anomaly-based session invalidation

### 4. Identity Federation

Enterprise SSO integration:
- OAuth 2.0 / OpenID Connect (OIDC)
- SAML 2.0 (optional)
- Provider support: Azure AD, Okta, Keycloak, Google Workspace

---

## Phases

### Phase 1: Argon2id Migration

- [ ] **1.1** Add `argon2-cffi` dependency
- [ ] **1.2** Implement `Argon2Hasher` in `core/services/auth/`:
  - `hash_password(password)` → hash string
  - `verify_password(password, hash)` → bool
  - `needs_rehash(hash)` → bool (for gradual migration)
- [ ] **1.3** Configure Argon2id parameters:
  ```yaml
  auth:
    password_hashing:
      algorithm: argon2id  # argon2id | pbkdf2 (legacy)
      argon2:
        time_cost: 3        # iterations
        memory_cost: 65536  # KiB (64 MB)
        parallelism: 4      # threads
  ```
- [ ] **1.4** Gradual migration on login (rehash PBKDF2 → Argon2id)
- [ ] **1.5** Write migration tests

### Phase 2: TOTP Multi-Factor Authentication

- [ ] **2.1** Create `core/services/auth/mfa/` package
- [ ] **2.2** Implement `TOTPService`:
  - `generate_secret(user_id)` → secret, provisioning URI
  - `verify_code(user_id, code)` → bool
  - `get_recovery_codes(user_id)` → list[str]
  - `use_recovery_code(user_id, code)` → bool
- [ ] **2.3** Secret storage (encrypted at rest)
- [ ] **2.4** QR code generation for authenticator apps
- [ ] **2.5** MFA enrollment flow in UI
- [ ] **2.6** MFA challenge during login
- [ ] **2.7** Write MFA tests

### Phase 3: WebAuthn/FIDO2 Support

- [ ] **3.1** Add `py_webauthn` dependency
- [ ] **3.2** Implement `WebAuthnService`:
  - `generate_registration_options(user)` → options
  - `verify_registration(user, response)` → credential
  - `generate_authentication_options(user)` → options
  - `verify_authentication(user, response)` → bool
- [ ] **3.3** Credential storage schema
- [ ] **3.4** Browser integration (JavaScript)
- [ ] **3.5** Hardware key enrollment UI
- [ ] **3.6** Passkey support (resident credentials)

### Phase 4: Session Management

- [ ] **4.1** Create `core/services/auth/sessions/` package
- [ ] **4.2** Implement `SessionService`:
  - `create_session(user, device_info)` → SessionToken
  - `validate_session(token)` → Session | None
  - `refresh_session(refresh_token)` → SessionToken
  - `revoke_session(session_id)` → bool
  - `revoke_all_sessions(user_id)` → int
  - `list_sessions(user_id)` → list[Session]
- [ ] **4.3** Token structure:
  ```python
  @dataclass
  class SessionToken:
      access_token: str      # Short-lived (15 min)
      refresh_token: str     # Longer-lived (7 days), rotates on use
      expires_at: datetime
      token_type: str = "Bearer"
  ```
- [ ] **4.4** Device fingerprinting:
  - User agent, IP geolocation, timezone
  - Binding validation on refresh
- [ ] **4.5** Concurrent session limits (configurable per user/role)
- [ ] **4.6** Session activity tracking
- [ ] **4.7** Anomaly detection (new device, location change)

### Phase 5: OAuth2/OIDC Federation

- [ ] **5.1** Add `authlib` dependency
- [ ] **5.2** Create `core/services/auth/federation/` package
- [ ] **5.3** Implement `OIDCProvider` abstraction:
  - `get_authorization_url(state, nonce)` → URL
  - `exchange_code(code)` → TokenResponse
  - `get_userinfo(access_token)` → UserInfo
  - `validate_id_token(token)` → Claims
- [ ] **5.4** Provider configurations:
  ```yaml
  auth:
    federation:
      enabled: true
      providers:
        azure_ad:
          client_id: "${AZURE_CLIENT_ID}"
          client_secret: "${AZURE_CLIENT_SECRET}"
          tenant_id: "${AZURE_TENANT_ID}"
          scopes: ["openid", "profile", "email"]
        google:
          client_id: "${GOOGLE_CLIENT_ID}"
          client_secret: "${GOOGLE_CLIENT_SECRET}"
  ```
- [ ] **5.5** User provisioning (JIT creation on first login)
- [ ] **5.6** Role/group mapping from IdP claims
- [ ] **5.7** SSO logout (front-channel/back-channel)
- [ ] **5.8** UI integration (social login buttons)

### Phase 6: Privileged Access Management

- [ ] **6.1** Implement Just-In-Time (JIT) elevation:
  - `request_elevation(user, role, reason, duration)` → request
  - `approve_elevation(request_id, approver)` → grant
  - `check_elevation(user, role)` → bool
- [ ] **6.2** Time-limited privilege grants
- [ ] **6.3** Elevation audit logging
- [ ] **6.4** Auto-expiration of elevated privileges
- [ ] **6.5** Re-authentication for sensitive operations

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Authentication Service                               │
│                        (core/services/auth/)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Password       │  │  MFA            │  │  Session        │             │
│  │  (Argon2id)     │  │  (TOTP/WebAuthn)│  │  Manager        │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Federation     │  │  Privilege      │  │  Device         │             │
│  │  (OIDC/OAuth2)  │  │  Manager (JIT)  │  │  Fingerprint    │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         Token & Credential Store                             │
│  • Encrypted secrets • Refresh token rotation • Revocation lists            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Database Schema Additions

```sql
-- MFA secrets (encrypted)
CREATE TABLE user_mfa_secrets (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES user_credentials(id),
    mfa_type VARCHAR(20) NOT NULL,  -- 'totp', 'webauthn'
    secret_encrypted BYTEA NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ,
    UNIQUE(user_id, mfa_type)
);

-- Recovery codes (hashed)
CREATE TABLE user_recovery_codes (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES user_credentials(id),
    code_hash VARCHAR(64) NOT NULL,
    used_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- WebAuthn credentials
CREATE TABLE user_webauthn_credentials (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES user_credentials(id),
    credential_id BYTEA UNIQUE NOT NULL,
    public_key BYTEA NOT NULL,
    sign_count INTEGER DEFAULT 0,
    device_name VARCHAR(255),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_used_at TIMESTAMPTZ
);

-- Active sessions
CREATE TABLE user_sessions (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES user_credentials(id),
    refresh_token_hash VARCHAR(64) UNIQUE NOT NULL,
    device_fingerprint JSONB,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_activity_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);

-- Federated identities
CREATE TABLE user_federated_identities (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES user_credentials(id),
    provider VARCHAR(50) NOT NULL,
    provider_user_id VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(provider, provider_user_id)
);

-- Privilege elevations
CREATE TABLE privilege_elevations (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES user_credentials(id),
    role VARCHAR(50) NOT NULL,
    reason TEXT,
    approved_by UUID REFERENCES user_credentials(id),
    granted_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    revoked_at TIMESTAMPTZ
);
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `auth.mfa_enrolled` | `MFAEvent` | TOTPService/WebAuthnService |
| `auth.mfa_verified` | `MFAEvent` | TOTPService/WebAuthnService |
| `auth.mfa_failed` | `MFAEvent` | TOTPService/WebAuthnService |
| `auth.session_created` | `SessionEvent` | SessionService |
| `auth.session_refreshed` | `SessionEvent` | SessionService |
| `auth.session_revoked` | `SessionEvent` | SessionService |
| `auth.federation_login` | `FederationEvent` | OIDCProvider |
| `auth.elevation_requested` | `ElevationEvent` | PrivilegeManager |
| `auth.elevation_granted` | `ElevationEvent` | PrivilegeManager |
| `auth.elevation_expired` | `ElevationEvent` | PrivilegeManager |

---

## Configuration

```yaml
auth:
  password_hashing:
    algorithm: argon2id
    argon2:
      time_cost: 3
      memory_cost: 65536
      parallelism: 4
    pbkdf2:  # Legacy, for migration
      iterations: 100000
      algorithm: sha256

  mfa:
    enabled: true
    required_for_roles: ["admin", "operator"]
    totp:
      issuer: "ATLAS"
      digits: 6
      period: 30
    webauthn:
      rp_id: "atlas.local"
      rp_name: "ATLAS"
      allowed_origins: ["https://atlas.local"]
    recovery_codes:
      count: 10
      length: 8

  sessions:
    access_token_ttl_minutes: 15
    refresh_token_ttl_days: 7
    max_concurrent_sessions: 5
    require_fingerprint_match: true
    revoke_on_password_change: true

  federation:
    enabled: false
    auto_provision_users: true
    default_role: "user"
    providers: {}

  elevation:
    enabled: true
    max_duration_hours: 4
    require_approval: true
    require_reason: true
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/auth/__init__.py` | Package exports |
| `core/services/auth/types.py` | Auth types and events |
| `core/services/auth/password.py` | Argon2id password hashing |
| `core/services/auth/mfa/__init__.py` | MFA package |
| `core/services/auth/mfa/totp.py` | TOTP implementation |
| `core/services/auth/mfa/webauthn.py` | WebAuthn implementation |
| `core/services/auth/mfa/recovery.py` | Recovery codes |
| `core/services/auth/sessions/__init__.py` | Session package |
| `core/services/auth/sessions/service.py` | Session management |
| `core/services/auth/sessions/fingerprint.py` | Device fingerprinting |
| `core/services/auth/federation/__init__.py` | Federation package |
| `core/services/auth/federation/oidc.py` | OIDC provider |
| `core/services/auth/federation/providers.py` | Provider configs |
| `core/services/auth/elevation.py` | JIT privilege elevation |
| `scripts/migrations/XXXX_add_auth_tables.py` | Schema migration |
| `tests/services/auth/` | Auth service tests |

---

## Dependencies

### Phase 1
- None (Python stdlib)

### Phase 2
- `pyotp>=2.9.0` - TOTP implementation
- `qrcode[pil]>=7.4.0` - QR code generation

### Phase 3
- `py_webauthn>=2.0.0` - WebAuthn support

### Phase 4
- `pyjwt>=2.8.0` - JWT handling (already likely present)

### Phase 5
- `authlib>=1.3.0` - OAuth2/OIDC client

---

## Security Considerations

1. **Secret Storage**: MFA secrets must be encrypted at rest using envelope encryption
2. **Timing Attacks**: Use constant-time comparison for all credential verification
3. **Rate Limiting**: Apply aggressive rate limits to MFA verification endpoints
4. **Backup Codes**: Hash recovery codes; never store or log plaintext
5. **Session Binding**: Bind refresh tokens to device fingerprint to prevent theft
6. **Token Revocation**: Maintain revocation list or use short-lived tokens with refresh
7. **Federation Trust**: Validate IdP signatures; pin trusted issuers

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| MFA enforcement scope? | All users / Admins only / Configurable | TBD |
| WebAuthn attestation? | None / Self / Direct | TBD |
| Session storage backend? | PostgreSQL / Redis / In-memory | TBD |
| Federation SAML support? | Yes / OIDC only | TBD |
