# Security Infrastructure (SOTA)

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: High  
> **Effort**: 2-3 weeks  
> **Created**: 2026-01-08

---

## Overview

Implement comprehensive security infrastructure covering cryptography, secrets management, supply chain security, and agentic AI safety:

### Research References

- [OpenAI: Practices for Governing Agentic AI Systems](https://openai.com/index/practices-for-governing-agentic-ai-systems/)
- [Anthropic: Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)
- [OWASP Top 10 for LLM Applications](https://owasp.org/www-project-top-10-for-large-language-model-applications/)
- [NIST Cybersecurity Framework](https://www.nist.gov/cyberframework)
- [SLSA Supply Chain Security](https://slsa.dev/)

---

## Core Domains

### 1. Cryptographic Services
- Envelope encryption (DEK/KEK hierarchy)
- Key rotation and management
- Secure random generation
- Cryptographic audit trails

### 2. Secrets Management
- Integration with external vaults (HashiCorp Vault, keyring/libsecret)
- Environment-based secret injection
- Secret rotation workflows
- Credential lifecycle management

### 3. Agentic AI Security
- Prompt injection defenses
- Tool capability restrictions
- Output validation and classifiers
- Agent supervision and kill switches

### 4. Supply Chain Security
- Dependency scanning and SBOMs
- Container hardening
- Runtime protection
- Integrity verification

---

## Phases

### Phase 1: Cryptographic Services

- [ ] **1.1** Create `core/services/crypto/` package
- [ ] **1.2** Implement `CryptoService`:
  - `generate_key(purpose, algorithm)` → KeyMaterial
  - `encrypt(plaintext, key_id)` → CipherText
  - `decrypt(ciphertext, key_id)` → bytes
  - `rotate_key(key_id)` → new_key_id
  - `derive_key(master_key, context)` → derived_key
- [ ] **1.3** Envelope encryption implementation:
  ```python
  @dataclass
  class EncryptedBlob:
      ciphertext: bytes
      encrypted_dek: bytes  # DEK encrypted with KEK
      kek_id: str           # Key Encryption Key identifier
      algorithm: str        # e.g., "AES-256-GCM"
      nonce: bytes
      auth_tag: bytes
  ```
- [ ] **1.4** Key hierarchy:
  - Master Key (KEK) → stored in secure backend
  - Data Encryption Keys (DEK) → per-resource, encrypted
- [ ] **1.5** Algorithm support:
  - AES-256-GCM (default symmetric)
  - ChaCha20-Poly1305 (alternative)
  - X25519 (key exchange)
  - Ed25519 (signatures)
- [ ] **1.6** Write crypto tests with known test vectors

### Phase 2: Secrets Management Integration

- [ ] **2.1** Create `core/services/secrets/` package
- [ ] **2.2** Implement `SecretsService` abstraction:
  - `get_secret(path)` → SecretValue
  - `set_secret(path, value, metadata)` → bool
  - `delete_secret(path)` → bool
  - `list_secrets(prefix)` → list[str]
  - `rotate_secret(path, generator)` → SecretValue
- [ ] **2.3** Backend implementations:
  - `EnvironmentSecretsBackend` - Environment variables
  - `KeyringSecretsBackend` - OS keyring (libsecret/Keychain)
  - `VaultSecretsBackend` - HashiCorp Vault
  - `EncryptedFileSecretsBackend` - Encrypted local file
- [ ] **2.4** Configuration:
  ```yaml
  secrets:
    backend: keyring  # environment | keyring | vault | encrypted_file
    vault:
      address: "${VAULT_ADDR}"
      auth_method: token  # token | approle | kubernetes
      mount_path: secret/atlas
    encrypted_file:
      path: ~/.atlas/secrets.enc
      key_derivation: argon2id
  ```
- [ ] **2.5** Secret references in config:
  ```yaml
  providers:
    openai:
      api_key: "${secret:providers/openai/api_key}"
  ```
- [ ] **2.6** Secret rotation automation
- [ ] **2.7** Audit logging for secret access

### Phase 3: Agentic AI Security

- [ ] **3.1** Create `core/services/agent_security/` package
- [ ] **3.2** Implement `PromptInjectionDefense`:
  - `sanitize_input(text)` → sanitized_text
  - `detect_injection(text)` → InjectionAnalysis
  - `validate_structured_output(output, schema)` → bool
  - `escape_for_context(text, context_type)` → str
- [ ] **3.3** Injection detection strategies:
  - Delimiter-based detection
  - Instruction pattern matching
  - Semantic analysis (optional LLM-based)
  - Input/output length ratio checks
- [ ] **3.4** Implement `ToolCapabilityManager`:
  - `register_tool(tool, capabilities)` → None
  - `check_permission(tool, action, context)` → bool
  - `get_allowed_actions(tool, context)` → list[Action]
  - `sandbox_execution(tool, action)` → SandboxedResult
- [ ] **3.5** Capability-based security model:
  ```yaml
  tools:
    file_operations:
      capabilities:
        read: true
        write: true
        delete: false  # Requires elevation
        execute: false
      allowed_paths:
        - "${WORKSPACE}/**"
        - "!${WORKSPACE}/.git/**"
    shell_executor:
      capabilities:
        commands:
          allow: ["ls", "cat", "grep", "find"]
          deny: ["rm", "sudo", "chmod"]
        network: false
        max_runtime_seconds: 30
  ```
- [ ] **3.6** Implement `OutputValidator`:
  - `validate_response(response, policy)` → ValidationResult
  - `classify_content(text)` → ContentClassification
  - `detect_data_exfiltration(output)` → bool
  - `check_consistency(input, output)` → bool
- [ ] **3.7** Implement `AgentSupervisor`:
  - `register_agent(agent_id, parent_id)` → None
  - `check_agent_health(agent_id)` → HealthStatus
  - `terminate_agent(agent_id, reason)` → bool
  - `pause_agent(agent_id)` → bool
  - `get_agent_hierarchy()` → AgentTree
- [ ] **3.8** Kill switch implementation:
  - Global emergency stop
  - Per-agent termination
  - Graceful vs immediate shutdown
  - State preservation on termination

### Phase 4: Authorization & Policy Engine

- [ ] **4.1** Create `core/services/authz/` package
- [ ] **4.2** Implement `PolicyEngine`:
  - `evaluate(principal, action, resource)` → Decision
  - `load_policies(source)` → None
  - `add_policy(policy)` → policy_id
  - `remove_policy(policy_id)` → bool
- [ ] **4.3** Policy language support:
  - Built-in DSL (YAML-based)
  - Optional: Open Policy Agent (OPA/Rego)
  - Optional: AWS Cedar
- [ ] **4.4** RBAC implementation:
  ```yaml
  authz:
    roles:
      admin:
        permissions: ["*"]
      operator:
        permissions:
          - "conversations:*"
          - "jobs:*"
          - "tasks:*"
          - "personas:read"
      user:
        permissions:
          - "conversations:own:*"
          - "jobs:own:*"
          - "tasks:own:*"
  ```
- [ ] **4.5** ABAC extensions (attribute-based):
  - Time-based conditions
  - Resource attribute matching
  - Environmental context
- [ ] **4.6** Permission caching and invalidation

### Phase 5: Supply Chain Security

- [ ] **5.1** Create `scripts/security/` tooling
- [ ] **5.2** Dependency scanning integration:
  - `scan_dependencies.py` - Run vulnerability scan
  - Integration with Dependabot/Snyk/Safety
  - CI/CD gate for critical vulnerabilities
- [ ] **5.3** SBOM generation:
  - CycloneDX format
  - SPDX format (optional)
  - Automated on release
- [ ] **5.4** Container hardening checklist:
  - [ ] Non-root user
  - [ ] Read-only filesystem
  - [ ] No unnecessary capabilities
  - [ ] Seccomp/AppArmor profiles
  - [ ] Minimal base image
  - [ ] No secrets in image layers
- [ ] **5.5** Integrity verification:
  - Code signing for releases
  - Checksum verification for dependencies
  - Reproducible builds (best effort)

### Phase 6: Security Audit Logging

- [ ] **6.1** Create `core/services/security_audit/` package
- [ ] **6.2** Implement `SecurityAuditService`:
  - `log_event(event)` → None
  - `query_events(filters)` → list[AuditEvent]
  - `export_events(format, destination)` → bool
- [ ] **6.3** Event format (OCSF-compatible):
  ```python
  @dataclass
  class SecurityAuditEvent:
      timestamp: datetime
      event_type: str          # e.g., "authentication", "authorization"
      category: str            # e.g., "iam", "data_access"
      severity: int            # 0-5
      actor: ActorInfo
      action: str
      resource: ResourceInfo
      outcome: str             # "success", "failure", "error"
      reason: Optional[str]
      metadata: Dict[str, Any]
      correlation_id: str
  ```
- [ ] **6.4** SIEM integration:
  - Syslog output (CEF/LEEF)
  - Webhook delivery
  - File-based with rotation
- [ ] **6.5** Tamper-evident logging:
  - Hash chaining (optional)
  - Write-once storage option
- [ ] **6.6** Retention and archival policies

### Phase 7: Runtime Protection

- [ ] **7.1** Input validation framework:
  - Schema validation for all API inputs
  - Size limits enforcement
  - Character set restrictions
- [ ] **7.2** Resource exhaustion protection:
  - Memory limits per operation
  - CPU time limits
  - Connection pool limits
  - Queue depth limits
- [ ] **7.3** Anomaly detection hooks:
  - Unusual access patterns
  - Privilege escalation attempts
  - Data exfiltration indicators
- [ ] **7.4** Incident response automation:
  - Auto-lockdown triggers
  - Alert escalation
  - Evidence preservation

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Security Infrastructure                               │
│                       (core/services/security/)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Crypto         │  │  Secrets        │  │  Policy         │             │
│  │  Service        │  │  Manager        │  │  Engine         │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Agent          │  │  Output         │  │  Tool           │             │
│  │  Supervisor     │  │  Validator      │  │  Capability     │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐             │
│  │  Prompt         │  │  Security       │  │  Runtime        │             │
│  │  Injection      │  │  Audit          │  │  Protection     │             │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘             │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                         External Integrations                                │
│  • HashiCorp Vault • OS Keyring • SIEM • Vulnerability Scanners             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agentic Security Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Agent Security Layers                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT                    EXECUTION                   OUTPUT                │
│  ─────                    ─────────                   ──────                │
│                                                                             │
│  ┌──────────────┐        ┌──────────────┐        ┌──────────────┐          │
│  │ Prompt       │        │ Tool         │        │ Output       │          │
│  │ Injection    │───────▶│ Capability   │───────▶│ Validator    │          │
│  │ Defense      │        │ Sandbox      │        │ Classifier   │          │
│  └──────────────┘        └──────────────┘        └──────────────┘          │
│         │                       │                       │                   │
│         ▼                       ▼                       ▼                   │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    Agent Supervisor                           │          │
│  │  • Health monitoring • Kill switches • Hierarchy management  │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                │                                            │
│                                ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐          │
│  │                    Security Audit Log                         │          │
│  │  • Immutable • Tamper-evident • SIEM-compatible              │          │
│  └──────────────────────────────────────────────────────────────┘          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Prompt Injection Defense Strategies

| Strategy | Description | Implementation |
|----------|-------------|----------------|
| **Delimiter Enforcement** | Use special tokens to separate system/user content | XML tags, unique delimiters |
| **Input Sanitization** | Remove/escape instruction-like patterns | Regex + heuristics |
| **Output Parsing** | Structured output only, no free-form execution | JSON schema validation |
| **Dual LLM** | Separate model for input classification | Lightweight classifier |
| **Canary Tokens** | Detect if system prompt leaked | Hidden markers |
| **Length Limits** | Restrict input/output ratios | Configurable thresholds |

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `security.key_rotated` | `KeyRotationEvent` | CryptoService |
| `security.secret_accessed` | `SecretAccessEvent` | SecretsService |
| `security.injection_detected` | `InjectionEvent` | PromptInjectionDefense |
| `security.tool_blocked` | `ToolBlockEvent` | ToolCapabilityManager |
| `security.output_filtered` | `OutputFilterEvent` | OutputValidator |
| `security.agent_terminated` | `AgentTerminationEvent` | AgentSupervisor |
| `security.policy_violation` | `PolicyViolationEvent` | PolicyEngine |
| `security.anomaly_detected` | `AnomalyEvent` | RuntimeProtection |

---

## Configuration

```yaml
security:
  crypto:
    default_algorithm: AES-256-GCM
    key_rotation_days: 90
    kek_backend: keyring  # keyring | vault | environment

  secrets:
    backend: keyring
    cache_ttl_seconds: 300
    audit_access: true

  agent_security:
    prompt_injection:
      enabled: true
      mode: strict  # permissive | strict | paranoid
      max_input_length: 32000
      suspicious_patterns_file: null
    output_validation:
      enabled: true
      classify_content: true
      block_pii: false
    supervisor:
      health_check_interval_seconds: 30
      max_agent_depth: 5
      global_kill_switch: true

  authz:
    engine: builtin  # builtin | opa
    cache_decisions: true
    cache_ttl_seconds: 60
    default_deny: true

  audit:
    enabled: true
    format: ocsf
    destinations:
      - type: file
        path: /var/log/atlas/security.log
        rotation: daily
      - type: syslog
        address: localhost:514
        format: cef
    retention_days: 90
    tamper_evident: false

  supply_chain:
    scan_on_startup: false
    block_critical_vulns: true
    sbom_path: null
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/crypto/__init__.py` | Crypto package |
| `core/services/crypto/service.py` | CryptoService |
| `core/services/crypto/envelope.py` | Envelope encryption |
| `core/services/crypto/keys.py` | Key management |
| `core/services/secrets/__init__.py` | Secrets package |
| `core/services/secrets/service.py` | SecretsService |
| `core/services/secrets/backends/` | Backend implementations |
| `core/services/agent_security/__init__.py` | Agent security package |
| `core/services/agent_security/injection.py` | Prompt injection defense |
| `core/services/agent_security/capabilities.py` | Tool capabilities |
| `core/services/agent_security/validator.py` | Output validation |
| `core/services/agent_security/supervisor.py` | Agent supervisor |
| `core/services/authz/__init__.py` | Authorization package |
| `core/services/authz/engine.py` | Policy engine |
| `core/services/authz/rbac.py` | RBAC implementation |
| `core/services/security_audit/__init__.py` | Audit package |
| `core/services/security_audit/service.py` | Audit service |
| `core/services/security_audit/formats.py` | OCSF/CEF formatters |
| `scripts/security/scan_dependencies.py` | Vulnerability scanner |
| `scripts/security/generate_sbom.py` | SBOM generation |
| `tests/services/crypto/` | Crypto tests |
| `tests/services/secrets/` | Secrets tests |
| `tests/services/agent_security/` | Agent security tests |
| `tests/services/authz/` | Authorization tests |

---

## Dependencies

### Phase 1 (Crypto)
- `cryptography>=42.0.0` - Core crypto primitives (likely already present)

### Phase 2 (Secrets)
- `keyring>=25.0.0` - OS keyring integration
- `hvac>=2.0.0` - HashiCorp Vault client (optional)

### Phase 3 (Agent Security)
- None (built on existing infrastructure)

### Phase 5 (Supply Chain)
- `cyclonedx-bom>=4.0.0` - SBOM generation
- `safety>=3.0.0` - Vulnerability scanning (optional, can use external)

---

## Security Considerations

1. **Key Storage**: KEKs must never be stored alongside encrypted data
2. **Memory Safety**: Clear sensitive data from memory after use
3. **Side Channels**: Use constant-time comparisons for all secrets
4. **Audit Integrity**: Security logs must be protected from tampering
5. **Defense in Depth**: Multiple layers; no single point of failure
6. **Least Privilege**: Default deny; explicit grants required
7. **Fail Secure**: Errors should deny access, not grant it

---

## Integration Points

| Component | Integration |
|-----------|-------------|
| User Accounts | Password hashing via CryptoService |
| Providers | API keys via SecretsService |
| Tools | Capability checks via ToolCapabilityManager |
| Agents | Supervision via AgentSupervisor |
| Chat | Input/output validation |
| HTTP Gateway | Policy enforcement via PolicyEngine |
| All Services | Security audit events |

---

## Relationship to Other Todos

- **[22-guardrails](../22-guardrails/)** - HITL, content filtering, rate limiting
- **[32-authentication](../32-authentication/)** - AuthN, MFA, sessions
- **[30-observability](../30-observability/)** - General telemetry (security audit is specialized)

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Policy engine choice? | Built-in / OPA / Cedar | TBD |
| Secrets backend default? | Environment / Keyring / Vault | TBD |
| Agent sandbox technology? | Process isolation / Container / WASM | TBD |
| SBOM format preference? | CycloneDX / SPDX / Both | TBD |
| Tamper-evident logs? | Hash chain / Append-only DB / External | TBD |
