# Model Context Protocol (MCP) Support

> **Status**: 📋 Planning  
> **Priority**: High  
> **Complexity**: Medium  
> **Effort**: 1 week  
> **Created**: 2026-01-07

---

## Overview

Implement Model Context Protocol (MCP) support for integrating external tool servers:

### What is MCP?

MCP is an open protocol for connecting AI models to external data sources and tools. It enables:

- **Tool Discovery**: Agents can discover available tools from MCP servers
- **Standardized Interface**: Common protocol for tool invocation
- **Context Management**: Efficient context passing between client and server
- **Security**: Scoped permissions and authentication

### References

- [Model Context Protocol Specification](https://modelcontextprotocol.io/)
- [MCP TypeScript SDK](https://github.com/anthropics/mcp-typescript)
- [MCP Python SDK](https://github.com/anthropics/mcp-python)

---

## Phases

### Phase 1: MCP Client Core

- [ ] **1.1** Create `core/services/mcp/` package
- [ ] **1.2** Implement MCP client protocol:
  - Connection management (stdio, HTTP)
  - Message serialization/deserialization
  - Request/response handling
- [ ] **1.3** Server discovery and registration
- [ ] **1.4** Health checking for MCP servers

### Phase 2: Tool Integration

- [ ] **2.1** Implement `MCPToolService`:
  - `discover_tools(server_id)` - List available tools
  - `invoke_tool(server_id, tool_name, params)` - Execute tool
  - `get_tool_schema(server_id, tool_name)` - Get tool definition
- [ ] **2.2** Tool manifest conversion (MCP ↔ ATLAS format)
- [ ] **2.3** Register MCP tools with ToolService
- [ ] **2.4** Permission mapping

### Phase 3: Resource Integration

- [ ] **3.1** Implement resource access:
  - `list_resources(server_id)` - Available resources
  - `get_resource(server_id, resource_uri)` - Fetch resource
  - `subscribe_resource(server_id, resource_uri)` - Watch changes
- [ ] **3.2** Resource caching strategy
- [ ] **3.3** Resource change notifications

### Phase 4: Prompt Templates

- [ ] **4.1** Implement prompt template support:
  - `list_prompts(server_id)` - Available prompts
  - `get_prompt(server_id, prompt_name, args)` - Execute prompt
- [ ] **4.2** Prompt injection into conversations
- [ ] **4.3** Prompt versioning

### Phase 5: Server Management UI

- [ ] **5.1** Create `GTKUI/MCP/` package
- [ ] **5.2** Server configuration panel:
  - Add/remove MCP servers
  - Connection settings
  - Authentication
- [ ] **5.3** Tool browser:
  - List tools from all connected servers
  - Tool details and documentation
  - Enable/disable per-persona
- [ ] **5.4** Server status monitoring

### Phase 6: Security & Sandboxing

- [ ] **6.1** Permission scoping per server
- [ ] **6.2** Credential management for servers
- [ ] **6.3** Request signing/verification
- [ ] **6.4** Sandboxed execution environment

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MCP Service                                     │
│                         (core/services/mcp/)                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         MCP Client                                    │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │   Tools      │  │  Resources   │  │   Prompts    │               │   │
│  │  │   Handler    │  │   Handler    │  │   Handler    │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│  ┌─────────────────────────────────┴─────────────────────────────────┐     │
│  │                     Server Registry                                │     │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐                  │     │
│  │  │ Server  │ │ Server  │ │ Server  │ │ Server  │                  │     │
│  │  │ (Local) │ │ (HTTP)  │ │ (stdio) │ │  (...)  │                  │     │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘                  │     │
│  └───────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
          ┌─────────────────┐            ┌──────────────────┐
          │   ToolService   │            │  External MCP    │
          │   (integrated)  │            │    Servers       │
          └─────────────────┘            └──────────────────┘
```

---

## MCP Server Configuration

```yaml
mcp:
  servers:
    - id: "filesystem"
      name: "Filesystem Server"
      transport: "stdio"
      command: "npx"
      args: ["-y", "@anthropics/mcp-server-filesystem"]
      env:
        MCP_ROOT: "/home/user/documents"
      permissions:
        - "files:read"
        - "files:write"
      enabled: true
    
    - id: "github"
      name: "GitHub Server"
      transport: "http"
      url: "http://localhost:3001"
      auth:
        type: "bearer"
        token_env: "GITHUB_TOKEN"
      permissions:
        - "repos:read"
        - "issues:write"
      enabled: true
    
    - id: "database"
      name: "Database Server"
      transport: "stdio"
      command: "python"
      args: ["-m", "mcp_server_postgres"]
      env:
        DATABASE_URL: "postgres://..."
      permissions:
        - "db:query"
      enabled: false
```

---

## MessageBus Events

| Event Type | Payload | Emitted By |
|------------|---------|------------|
| `mcp.server_connected` | `MCPServerEvent` | MCPService |
| `mcp.server_disconnected` | `MCPServerEvent` | MCPService |
| `mcp.tool_discovered` | `MCPToolEvent` | MCPService |
| `mcp.tool_invoked` | `MCPInvocationEvent` | MCPService |
| `mcp.resource_updated` | `MCPResourceEvent` | MCPService |

---

## Files to Create

| File | Purpose |
|------|---------|
| `core/services/mcp/__init__.py` | Package exports |
| `core/services/mcp/types.py` | Types and events |
| `core/services/mcp/client.py` | MCP protocol client |
| `core/services/mcp/service.py` | MCPService |
| `core/services/mcp/tools.py` | Tool integration |
| `core/services/mcp/resources.py` | Resource access |
| `core/services/mcp/prompts.py` | Prompt templates |
| `core/services/mcp/registry.py` | Server registry |
| `GTKUI/MCP/` | UI components |
| `tests/services/mcp/` | Service tests |

---

## Dependencies

- **Prerequisite**: [00-foundation](../00-foundation/) - Common types and patterns
- [08-skills-tools](../08-skills-tools/) - Tool integration
- MCP Python SDK (`mcp` package)

---

## Success Criteria

1. MCP servers connectable
2. Tools discoverable and invocable
3. Resources accessible
4. Permission scoping working
5. UI for server management
6. >90% test coverage

---

## Open Questions

| Question | Options | Decision |
|----------|---------|----------|
| Supported transports? | stdio only / HTTP only / Both | TBD |
| Server process management? | ATLAS manages / External / Both | TBD |
| Tool caching strategy? | None / Per-session / Persistent | TBD |
| Authentication storage? | Config file / Keyring / Environment | TBD |
| Default MCP servers to include? | None / Filesystem / Common set | TBD |
| MCP SDK version? | Latest stable / Pinned version | TBD |
