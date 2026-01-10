# 🌐 ATLAS

## Adaptive Thinking Layers for Agentic Systems

**Your Intelligence. Your Rules. Your Data.**

[![CI](https://github.com/Digital-Hallucinations/ATLAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Digital-Hallucinations/ATLAS/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Proprietary-orange.svg)](LICENSE)
[![GTK4](https://img.shields.io/badge/GTK-4.0-green.svg)](https://gtk.org/)

*A modular, multi-provider, multi-persona agentic framework*  
*By Digital Hallucinations — Jeremy Shows*

---

[Quick Start](#-5-minute-quickstart) • [Philosophy](#-why-atlas-exists) • [Architecture](#-architecture-at-a-glance) • [Personas](#-the-persona-ecosystem) • [Setup Tiers](#️-setup-tiers) • [Docs](#-documentation)

---

## 🎯 Why ATLAS Exists

Most AI tools are powerful—but brittle, opaque, and locked into a single provider, mindset, or workflow.

As someone who works across technical, operational, and service-oriented roles, I needed an assistant that could **adapt**—not just respond. I wanted a system that could reason across domains, switch providers without breaking workflows, run locally or online, and remain transparent enough that I could understand *why* it behaved the way it did.

**Most assistants optimize for conversation. ATLAS was built to optimize for usefulness under constraint.**

That meant:

- ✅ No hard dependency on a single model or vendor
- ✅ Clear separation between personas, tools, memory, and orchestration
- ✅ Graceful degradation when services fail
- ✅ A system that respects user control over data, configuration, and behavior

> ATLAS is not meant to replace human judgment.  
> It is meant to **augment it**—reliably, inspectably, and without pretending to be more than it is.

---

## 🔮 The Bigger Picture

Sooner or later, the major labs will consolidate intelligent systems—and the everyday user will be left renting access on someone else's terms.

**ATLAS exists to change that equation.**

```Text
┌─────────────────────────────────────────────────────────────────┐
│                     THE ATLAS DIFFERENCE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   Traditional AI Services          ATLAS                        │
│   ─────────────────────           ─────                         │
│   🔒 Vendor lock-in          →    🔓 Provider-agnostic          │
│   ☁  Cloud-only              →    🏠 Local-first option         │
│   📊 Your data, their profit →    🛡  Your data, your control   │
│   🤖 One-size-fits-all       →    🎭 Persona-driven adaptation  │
│   💸 Subscription treadmill  →    ⚡ Own your infrastructure    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

Whether you're a student learning to code, a developer shipping products, or an enterprise managing compliance—ATLAS scales to your needs while keeping you in control.

---

## ✨ What Makes ATLAS Different

### 🎭 Persona-Driven Intelligence

Not one assistant—a **team** of specialized personas. CodeGenius for development. DocGenius for documentation. FitnessCoach for wellness. Each with their own tools, skills, and behavioral patterns.

Coming soon: Persona Builder & Marketplace

### 🔄 Provider Freedom

OpenAI today? Anthropic tomorrow? Local Llama next week? Switch providers **without rewriting workflows**. Your personas, tools, and memory stay intact.

### 🤖 Built-in Agentic Systems

Native support for agents and sub-agents. Task decomposition, job scheduling, capability routing—all orchestrated through a unified runtime.

### 🏠 True Data Sovereignty

PostgreSQL for persistence. Redis for messaging. Run it all locally, or deploy to your own cloud. **Your data never leaves unless you choose.**

---

## 🏗 Architecture at a Glance

ATLAS combines a GTK desktop shell, configurable personas, and an orchestration backend to coordinate multi-agent work across conversations, scheduled jobs, and automation services.

```mermaid
flowchart TD
    subgraph User["👤 User Interfaces"]
        U1[GTK Desktop Shell]
        U2[REST API Clients]
        U3[Automation Scripts]
    end

    subgraph Core["🧠 ATLAS Core Runtime"]
        C1[ConfigManager]
        C2[Provider Manager]
        C3[Persona Manager]
        C4[Tool Manager]
        C5[Skill Manager]
    end

    subgraph Data["💾 Data Layer"]
        D1[(PostgreSQL)]
        D2[(Redis Streams)]
        D3[Vector Store]
    end

    subgraph Orchestration["⚡ Orchestration Engine"]
        O1[Task Dispatcher]
        O2[Job Scheduler]
        O3[Capability Registry]
        O4[Message Bus]
    end

    subgraph Providers["🌐 AI Providers"]
        P1[OpenAI]
        P2[Anthropic]
        P3[xAI]
        P4[Local LLMs]
    end

    User --> Core
    Core --> Data
    Core --> Orchestration
    Core --> Providers
    Orchestration --> Data
```

---

## 🎭 The Persona Ecosystem

Personas are more than prompts—they're complete cognitive profiles with their own tools, skills, permissions, and behavioral patterns.

```mermaid
flowchart LR
    subgraph System["System Personas"]
        S1[🌐 ATLAS<br/>Flagship Orchestrator]
        S2[🔍 Echo<br/>Diagnostic & Debug]
        S3[⚖ ComplianceOfficer<br/>Policy Enforcement]
    end

    subgraph Domain["Domain Specialists"]
        D1[💻 CodeGenius<br/>Development]
        D2[📚 DocGenius<br/>Documentation]
        D3[🌐 WebDev<br/>Full-Stack]
        D4[🧮 MathTutor<br/>Education]
    end

    subgraph Personal["Personal Assistants"]
        P1[🏃 FitnessCoach<br/>Wellness]
        P2[📅 FocusPlanner<br/>Productivity]
        P3[📓 DailyJournal<br/>Reflection]
        P4[🗣 LanguageTutor<br/>Learning]
    end

    subgraph Creative["Creative & Research"]
        C1[💡 IdeaSpark<br/>Brainstorming]
        C2[🎨 Muse<br/>Creative Writing]
        C3[📖 KnowledgeCurator<br/>Research]
    end

    S1 -.->|delegates to| Domain
    S1 -.->|delegates to| Personal
    S1 -.->|delegates to| Creative
```

### Persona Capabilities

Each persona can be configured with:

- **Tools**: What actions can it take? (calendar, terminal, web search, code execution)
- **Skills**: What compound behaviors can it perform? (research briefs, daily digests, safety audits)
- **Permissions**: Read-only terminal? Write access to calendar? Code sandbox?
- **Provider**: Which AI backend powers this persona?

**25+ personas included** • CodeGenius • DocGenius • WebDev • FitnessCoach • HealthCoach • MathTutor • ScienceTutor • LanguageTutor • FrenchPracticePartner • FocusPlanner • DailyJournal • KnowledgeCurator • IdeaSpark • Muse • WeatherGenius • ResumeGenius • ComplianceOfficer • Einstein • Nikola Tesla • Hermes • Specter • MEDIC • and more...

---

## ⚙️ Setup Tiers

ATLAS adapts to your scale and requirements—from free learning environments to enterprise compliance.

```mermaid
flowchart LR
    subgraph Tiers["Choose Your Path"]
        T1["🎓 Student<br/>───────<br/>Free tier<br/>Guidance-focused<br/>Usage limits"]
        T2["👤 Personal<br/>───────<br/>Up to 5 profiles<br/>No preset limits<br/>Local control"]
        T3["⚡ Enthusiast<br/>───────<br/>All features<br/>Experimental access<br/>Power user mode"]
        T4["🏢 Enterprise<br/>───────<br/>Team rollout<br/>Redis + schedulers<br/>Strict retention"]
        T5["📋 Regulatory<br/>───────<br/>Extended retention<br/>Residency controls<br/>Compliance auditing"]
    end

    T1 --> T2 --> T3 --> T4 --> T5
```

| Feature | Student | Personal | Enthusiast | Enterprise | Regulatory |
| ------- | ------- | -------- | ---------- | ---------- | ---------- |
| **Message Bus** | In-memory | In-memory | Redis Streams | Shared Redis | Shared Redis |
| **Job Scheduling** | ❌ | ❌ | ✅ PostgreSQL | ✅ Dedicated | ✅ Dedicated |
| **Retention** | 7 days / 100 msgs | No limits | 90 days / 1000 msgs | 30 days / 500 msgs | Extended |
| **HTTP Server** | Auto-start | Auto-start | Auto-start | Manual | Manual |
| **Pricing** | Free | Tiered | Paid | Team license | Compliance license |

**Developer Mode**: Available on any tier—enables local Redis, PostgreSQL, and verbose logging for production-like testing.

---

## ⚡ The 100x Multiplier

ATLAS isn't just an assistant—it's an **orchestration engine** that multiplies your effectiveness.

```mermaid
sequenceDiagram
    participant You
    participant ATLAS
    participant CodeGenius
    participant DocGenius
    participant Tools

    You->>ATLAS: "Refactor auth module and update docs"
    ATLAS->>ATLAS: Decompose task
    
    par Parallel Execution
        ATLAS->>CodeGenius: Analyze & refactor code
        CodeGenius->>Tools: Read files, run tests
        Tools-->>CodeGenius: Results
        CodeGenius-->>ATLAS: Refactored code
    and
        ATLAS->>DocGenius: Draft documentation updates
        DocGenius->>Tools: Fetch current docs
        Tools-->>DocGenius: Content
        DocGenius-->>ATLAS: Updated docs
    end
    
    ATLAS->>ATLAS: Synthesize results
    ATLAS-->>You: "Done. Here's what changed..."
```

**One request. Multiple specialists. Parallel execution. Unified result.**

---

## 🚀 5-Minute Quickstart

```bash
# Clone ATLAS
git clone https://github.com/DigitalHallucinations/ATLAS.git
cd ATLAS

# Install GTK prerequisites (choose your OS)
# Debian/Ubuntu
sudo apt install libgtk-4-dev libadwaita-1-dev gobject-introspection gir1.2-gtk-4.0
# Fedora
sudo dnf install gtk4-devel libadwaita-devel gobject-introspection-devel
# macOS
brew install gtk4 libadwaita gobject-introspection

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch ATLAS
python3 main.py
```

> 💡 **Pro tip**: Use `python3 scripts/install_environment.py --with-accelerators` to automate virtualenv creation and install GPU/ML extras (Torch, Hugging Face, Whisper). Skip the flag on CPU-only systems.

### Runtime Requirements

| Component | Version | Purpose |
| --------- | ------- | ------- |
| Python | 3.10+ | Modern type syntax support |
| PostgreSQL | 14+ | Conversations, state, scheduling |
| Redis | Optional | Durable message bus (in-memory available) |
| GTK | 4.0 | Native desktop interface |

---

## 🧰 Core Capabilities

### Tools (60+ Built-in)

From web search to code execution, from calendar management to threat scanning:

```Text
📡 Web & Research        💾 Data & Storage       🔧 System & Dev
─────────────────       ─────────────────       ─────────────────
• Google Search         • Vector Store          • Terminal Command
• Webpage Fetch         • KV Store              • Filesystem I/O
• Browser (Lite)        • Memory Graph          • Code Execution
• API Connector         • Content Repository    • Schema Infer
                        • Spreadsheet           • Log Parser

📅 Productivity         🛡 Governance           🎨 Creative
─────────────────       ─────────────────       ─────────────────
• Calendar Service      • Policy Reference      • Lyricist
• Task Queue            • Audit Reporter        • Story Weaver
• Priority Queue        • Threat Scanner        • Visual Prompt
• Planner Decompose     • HITL Approval         • Mood Map
• Notification          • Vault Secrets         • Emotive Tagger
```

### Skills (Compound Behaviors)

Skills combine tools and reasoning into higher-order capabilities:

- **ContextualSummarizer** – Snapshot recaps of goals, blockers, and commitments
- **Sentinel** – Policy-aware safety review before risky actions
- **ResearchBrief** – Rapid web research with citations and follow-up questions
- **DailyDigest** – Morning briefing fusing news with work context
- **SevereWeatherAlert** – NOAA-integrated monitoring with escalation guidance

---

## 🔌 Provider Flexibility

Connect to any provider—or run models locally:

```mermaid
flowchart TD
    subgraph ATLAS["ATLAS Runtime"]
        PM[Provider Manager]
    end

    subgraph Cloud["☁ Cloud Providers"]
        O[OpenAI<br/>GPT-4, GPT-4o]
        A[Anthropic<br/>Claude 3.5, Opus]
        X[xAI<br/>Grok]
        G[Google<br/>Gemini]
    end

    subgraph Local["🏠 Local Options"]
        L[Ollama]
        LM[LM Studio]
        HF[Hugging Face<br/>Transformers]
    end

    PM --> Cloud
    PM --> Local
```

**Switch providers per-persona or globally**—your workflows and memory persist regardless.

---

## 📚 Documentation

### By Audience

| Path | For |
| ---- | --- |
| [User Docs](docs/user/README.md) | Setup wizard, GTK navigation, daily workflows |
| [Developer Docs](docs/developer/README.md) | Environment setup, APIs, extending ATLAS |
| [Enterprise Docs](docs/enterprise/README.md) | Retention, policies, compliance, backups |

### Deep Dives

- [Architecture Overview](docs/architecture-overview.md) – Runtime, personas, orchestration
- [Architecture Strategy](docs/architecture/) – Technical design decisions and evolution
- [Persona Definitions](docs/Personas.md) – Schema, validation, tooling
- [Task Lifecycle](docs/tasks/overview.md) – Routing, analytics, UI integration
- [Job Services](docs/jobs/api.md) – APIs and scheduling
- [Tool Manifest](docs/tool-manifest.md) – Metadata and discovery
- [AtlasServer API](docs/server/api.md) – REST endpoints and semantics
- [GTK UI Overview](docs/ui/gtk-overview.md) – Shell architecture

---

## 🗺 Roadmap

| Status | Phase | Focus |
| ------ | ----- | ----- |
| ✅ | **Core Framework** | Multi-persona, multi-provider orchestration |
| ✅ | **Tool Ecosystem** | 60+ built-in tools and skill framework |
| ✅ | **GTK Shell** | Native desktop with setup wizard |
| 🔨 | **Persona Builder** | Visual persona creation and editing |
| 🔨 | **Persona Marketplace** | Share, download, and purchase personas |
| 📋 | **Plugin Registry** | Public index for community modules |
| 📋 | **Mobile Companion** | Lightweight mobile interface |
| 📋 | **Team Collaboration** | Shared workspaces and blackboard sync |

---

## 🤝 Contributing

Contributions welcome! Whether you're building a persona, tool, or provider adapter:

1. Fork the repo
2. Read `AGENTS.md` from root to your target directory
3. Follow [Agent Workflow](docs/contributing/agent-workflow.md) guidelines
4. Ensure tests pass: `pytest`
5. Open a pull request

### Agent Roles

| Role | Scope |
| ---- | ----- |
| UI Agent | `GTKUI/`, `Icons/`, UI entry points |
| Backend Agent | `core/`, `modules/`, orchestration |
| Data/DB Agent | Stores, migrations, persistence |
| Infra/Config Agent | `server/`, `config.yaml`, runtime scripts |
| Docs Agent | `docs/` only |
| Testing Agent | `tests/` only |
| Security Agent | Configuration and policy review |

---

## 🙏 Credits

Created and maintained by **Jeremy Shows**  
Part of the **Digital Hallucinations** ecosystem

---

"Systems should serve users, not enclose them."

---

`#AI` `#AgenticFramework` `#OpenSource` `#GTK` `#MultiProvider` `#DataSovereignty` `#Python`
