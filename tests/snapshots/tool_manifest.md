# Tool Manifest

## Persona: ATLAS

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| atlas_dashboard | 1.0.0 | atlas_operations, status_reporting | — | No | Capture persona-specific initiative health updates for the ATLAS dashboard. |
| browser | 1.0.0 | web_navigation, research | — | No | Record a virtual browsing session including annotations and metadata for downstream summarisation. |
| calendar_service | 1.0.0 | calendar_read, calendar_write | — | No | Book lightweight calendar slots and retrieve existing reservations for personas without native calendar access. |
| context_tracker | 1.0.0 | conversation_state, status_reporting | — | No | Compile a normalized snapshot of the active conversation including recent highlights and participants. |
| crm_service | 1.0.0 | crm_logging, engagement_tracking | — | No | Log a structured customer interaction with optional tags and metadata. |
| dashboard_service | 1.0.0 | metrics_reporting, analysis | — | No | Store a dashboard snapshot with numeric metrics and supporting commentary. |
| debian12_calendar | 1.1.0 | calendar_read, calendar_write | — | No | Interact with the Debian 12 on-device calendar to list, search, inspect, and manage events. |
| email_service | 1.0.0 | email_delivery, communications | — | No | Dispatch a structured email with support for CC and BCC recipients. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| labor_market_feed | 1.0.0 | labor_market_research, trend_analysis | — | No | Generate a synthetic labour market snapshot for the requested regions and skills. |
| notebook | 1.0.0 | note_taking, knowledge_management | — | No | Capture a structured research note inside a collaborative notebook. |
| notification_service | 1.0.0 | notification_delivery, engagement | — | No | Send a notification message to the requested channel and recipients. |
| policy_reference | 1.0.0 | policy_lookup, risk_assessment_support | — | No | Retrieve internal safety and governance policy guidance relevant to a proposed action. |
| priority_queue | 1.0.0 | task_management, status_reporting | — | No | Score and sort operational tasks to produce a prioritized execution list. |
| roadmap_service | 1.0.0 | program_management, planning | — | No | Update a roadmap initiative with owner, status, and upcoming milestones. |
| spreadsheet | 1.0.0 | data_management, tabular_editing | — | No | Append or replace rows within a lightweight spreadsheet. |
| task_catalog_snapshot | 1.0.0 | task_catalog, status_reporting | — | No | Retrieve a filtered snapshot of the runtime task manifest catalog. |
| ticketing_system | 1.0.0 | issue_tracking, escalations | — | No | Create a ticket for follow-up work with assignee and tag support. |
| workspace_publisher | 1.0.0 | workspace_publishing, brief_distribution | — | No | Publish a structured brief payload into a collaborative workspace channel. |

## Persona: Cleverbot

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| persona_backstory_sampler | 1.0.0 | conversation_design, persona_memory | — | No | Return seeded conversation hooks that reinforce Cleverbot's banter styles. |

## Persona: CodeGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| execute_python | 1.0.0 | code_execution, analysis | high | No | Execute Python code inside a sandboxed interpreter with stdout capture and timeout enforcement. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| terminal_command | 1.0.0 | — | high | No | Execute a terminal command and return the output, error, and status code. |

## Persona: ComplianceOfficer

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| atlas_dashboard | 1.0.0 | atlas_operations, status_reporting | — | No | Capture persona-specific initiative health updates for the ATLAS dashboard. |
| browser | 1.0.0 | web_navigation, research | — | No | Record a virtual browsing session including annotations and metadata for downstream summarisation. |
| crm_service | 1.0.0 | crm_logging, engagement_tracking | — | No | Log a structured customer interaction with optional tags and metadata. |
| dashboard_service | 1.0.0 | metrics_reporting, analysis | — | No | Store a dashboard snapshot with numeric metrics and supporting commentary. |
| email_service | 1.0.0 | email_delivery, communications | — | No | Dispatch a structured email with support for CC and BCC recipients. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| labor_market_feed | 1.0.0 | labor_market_research, trend_analysis | — | No | Generate a synthetic labour market snapshot for the requested regions and skills. |
| notebook | 1.0.0 | note_taking, knowledge_management | — | No | Capture a structured research note inside a collaborative notebook. |
| notification_service | 1.0.0 | notification_delivery, engagement | — | No | Send a notification message to the requested channel and recipients. |
| policy_reference | 1.0.0 | policy_lookup, risk_assessment_support | — | No | Retrieve internal safety and governance policy guidance relevant to a proposed action. |
| regulatory_gap_audit | 1.0.0 | compliance, risk_assessment | — | No | Compare provided controls against compliance checklists and flag gaps. |
| roadmap_service | 1.0.0 | program_management, planning | — | No | Update a roadmap initiative with owner, status, and upcoming milestones. |
| spreadsheet | 1.0.0 | data_management, tabular_editing | — | No | Append or replace rows within a lightweight spreadsheet. |
| ticketing_system | 1.0.0 | issue_tracking, escalations | — | No | Create a ticket for follow-up work with assignee and tag support. |

## Persona: DocGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| generate_doc_outline | 1.0.0 | documentation, analysis | — | No | Parse a definition signature and return DocGenius docstring scaffolding. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| search_pmc | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Search PubMed Central (PMC) for open access articles using the Entrez API. |
| search_pubmed | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Search PubMed for biomedical literature using the NCBI Entrez API. |

## Persona: Einstein

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| relativity_scenario | 1.0.0 | education, physics_reasoning | — | No | Surface curated relativity thought experiments aligned with Einstein's pedagogy. |

## Persona: FitnessCoach

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| microcycle_plan | 1.0.0 | fitness_programming, planning | — | No | Compose a weekly microcycle plan tuned to the stated fitness goal. |

## Persona: HealthCoach

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| habit_stack_planner | 1.0.0 | wellness_planning, habit_design | — | No | Design habit stacks combining nutrition, stress, and sleep anchors. |

## Persona: KnowledgeCurator

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| atlas_dashboard | 1.0.0 | atlas_operations, status_reporting | — | No | Capture persona-specific initiative health updates for the ATLAS dashboard. |
| browser | 1.0.0 | web_navigation, research | — | No | Record a virtual browsing session including annotations and metadata for downstream summarisation. |
| context_tracker | 1.0.0 | conversation_state, status_reporting | — | No | Compile a normalized snapshot of the active conversation including recent highlights and participants. |
| crm_service | 1.0.0 | crm_logging, engagement_tracking | — | No | Log a structured customer interaction with optional tags and metadata. |
| dashboard_service | 1.0.0 | metrics_reporting, analysis | — | No | Store a dashboard snapshot with numeric metrics and supporting commentary. |
| email_service | 1.0.0 | email_delivery, communications | — | No | Dispatch a structured email with support for CC and BCC recipients. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| knowledge_card_builder | 1.0.0 | knowledge_management, summarization | — | No | Build standardized knowledge cards from curated sources. |
| labor_market_feed | 1.0.0 | labor_market_research, trend_analysis | — | No | Generate a synthetic labour market snapshot for the requested regions and skills. |
| notebook | 1.0.0 | note_taking, knowledge_management | — | No | Capture a structured research note inside a collaborative notebook. |
| notification_service | 1.0.0 | notification_delivery, engagement | — | No | Send a notification message to the requested channel and recipients. |
| roadmap_service | 1.0.0 | program_management, planning | — | No | Update a roadmap initiative with owner, status, and upcoming milestones. |
| spreadsheet | 1.0.0 | data_management, tabular_editing | — | No | Append or replace rows within a lightweight spreadsheet. |
| ticketing_system | 1.0.0 | issue_tracking, escalations | — | No | Create a ticket for follow-up work with assignee and tag support. |

## Persona: LanguageTutor

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| dialogue_drill | 1.0.0 | language_instruction, conversation_design | — | No | Generate conversational practice drills with correction cues. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: MEDIC

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| fetch_pubmed_details | 1.0.0 | medical_research, literature_retrieval | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Retrieve detailed PubMed records, including abstracts and metadata, via the Entrez EFetch API. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| search_pmc | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Search PubMed Central (PMC) for open access articles using the Entrez API. |
| search_pubmed | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Search PubMed for biomedical literature using the NCBI Entrez API. |

## Persona: MathTutor

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| stepwise_solution | 1.0.0 | education, mathematics | — | No | Produce numbered reasoning steps and optional symbolic verification. |

## Persona: Muse

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| context_tracker | 1.0.0 | conversation_history, memory_management | low | No | Record conversation context snapshots to ground creative follow-ups. |
| emotive_tagger | 1.0.0 | emotion_analysis, creative_quality_control | low | No | Annotate passages with emotional tags and optional intensity values. |
| get_current_info | 1.0.0 | time_information, date_information | low | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| lyricist | 1.0.0 | lyric_writing, creative_iteration | low | No | Generate structured lyric sections anchored on the supplied concept. |
| mood_map | 1.0.0 | tone_mapping, story_analysis | low | No | Translate beats into an emotion arc and color palette guidance. |
| storyweaver | 1.0.0 | story_development, narrative_design | low | No | Assemble a multi-beat story outline with tonal notes and focus prompts. |
| visual_prompt | 1.0.0 | visual_prompting, creative_direction | low | No | Craft a descriptive art prompt with palette, lighting, and medium cues. |

## Persona: Nikola Tesla

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| wireless_power_brief | 1.0.0 | innovation, concept_development | — | No | Draft Nikola Tesla styled wireless power solution briefs. |

## Persona: ResumeGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| analytics_dashboard | 1.0.0 | analytics_reporting, segment_tracking | — | No | Refresh an analytics dashboard with numeric metrics, cohort breakdowns, and contextual metadata. |
| atlas_dashboard | 1.0.0 | atlas_operations, status_reporting | — | No | Capture persona-specific initiative health updates for the ATLAS dashboard. |
| ats_scoring_service | 1.0.0 | resume_analysis, ats_compliance | — | Yes (docs: Configure ATS_SCORING_SERVICE_API_KEY or ats_scoring_service.api_key for authenticated requests., env: ATS_SCORING_SERVICE_API_KEY, type: api_key) | Submit resume and job description text to the ATS scoring service and return compatibility insights and optimization suggestions. |
| browser | 1.0.0 | web_navigation, research | — | No | Record a virtual browsing session including annotations and metadata for downstream summarisation. |
| content_repository | 1.0.0 | content_management, knowledge_base | — | No | Store or refresh reusable content blocks including metadata, tags, and attachments. |
| crm_service | 1.0.0 | crm_logging, engagement_tracking | — | No | Log a structured customer interaction with optional tags and metadata. |
| dashboard_service | 1.0.0 | metrics_reporting, analysis | — | No | Store a dashboard snapshot with numeric metrics and supporting commentary. |
| email_service | 1.0.0 | email_delivery, communications | — | No | Dispatch a structured email with support for CC and BCC recipients. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| labor_market_feed | 1.0.0 | labor_market_research, trend_analysis | — | No | Generate a synthetic labour market snapshot for the requested regions and skills. |
| notebook | 1.0.0 | note_taking, knowledge_management | — | No | Capture a structured research note inside a collaborative notebook. |
| notification_service | 1.0.0 | notification_delivery, engagement | — | No | Send a notification message to the requested channel and recipients. |
| policy_reference | 1.0.0 | policy_lookup, risk_assessment_support | — | No | Retrieve internal safety and governance policy guidance relevant to a proposed action. |
| roadmap_service | 1.0.0 | program_management, planning | — | No | Update a roadmap initiative with owner, status, and upcoming milestones. |
| spreadsheet | 1.0.0 | data_management, tabular_editing | — | No | Append or replace rows within a lightweight spreadsheet. |
| ticketing_system | 1.0.0 | issue_tracking, escalations | — | No | Create a ticket for follow-up work with assignee and tag support. |

## Persona: Shared

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| analytics_dashboard | 1.0.0 | analytics_reporting, segment_tracking | medium | No | Refresh an analytics dashboard with numeric metrics, cohort breakdowns, and contextual metadata. |
| atlas_dashboard | 1.0.0 | atlas_operations, status_reporting | medium | No | Capture an initiative update for the ATLAS persona dashboard, including health, metrics, and stakeholders. |
| audit_reporter | 0.1.0 | governance, audit_trail | medium | No | Aggregate security control evidence, severity counts, and analyst notes into an audit-ready summary. |
| browser | 1.0.0 | web_navigation, research | medium | No | Record a virtual browsing session including annotations and metadata for downstream summarisation. |
| browser_lite | 1.0.0 | browser_automation | high | No | Navigate a limited number of allowlisted pages with robots.txt enforcement, throttling, and optional screenshots. |
| budget.limiter | 1.0.0 | runtime_management, conversation_budgeting | medium | No | Inspect or adjust tracked tool runtime budgets for a conversation. |
| calculator | 1.0.0 | calculator | low | No | Evaluate sanitized mathematical expressions with Decimal precision and optional unit conversions. |
| calendar_service | 1.0.0 | calendar_read, calendar_write | medium | No | Book lightweight calendar slots and retrieve existing reservations for personas without native calendar access. |
| consensus.vote | 1.0.0 | orchestration, coordination | low | No | Run the negotiation vote protocol across a set of participants and return the serialized trace and decision. |
| content_repository | 1.0.0 | content_management, knowledge_base | medium | No | Store or refresh reusable content blocks including metadata, tags, and attachments. |
| context_tracker | 1.0.0 | conversation_state, status_reporting | — | No | Compile a normalized snapshot of the active conversation including recent highlights and participants. |
| crm_service | 1.0.0 | crm_logging, engagement_tracking | medium | No | Log a structured customer interaction with optional tags and metadata. |
| dashboard_service | 1.0.0 | metrics_reporting, analysis | medium | No | Store a dashboard snapshot with numeric metrics and supporting commentary. |
| debian12_calendar | 1.1.0 | calendar_read, calendar_write | — | No | Interact with the Debian 12 on-device calendar to list, search, inspect, and manage events. |
| delete_namespace | 1.0.0 | vector_store | — | Yes (docs: Configure credentials for the selected vector store provider via ConfigManager., type: api_key) | Remove all stored embeddings associated with the specified namespace. |
| email_service | 1.0.0 | email_delivery, communications | medium | No | Dispatch a structured email with support for CC and BCC recipients. |
| emotive_tagger | 1.0.0 | emotion_analysis, creative_quality_control | low | No | Tag passages with qualitative emotions and provide intensity hints. |
| eval.judge | 1.0.0 | evaluation, analytics | — | No | Run evaluator backends across candidate completions, apply configurable thresholds, and publish scoring analytics. |
| eval.regression | 1.0.0 | evaluation, analytics | — | No | Compare baseline and candidate artifacts, compute diffs and regression metrics, and publish structured analytics. |
| execute_javascript | 1.0.0 | javascript_exec, code_execution | high | No | Execute JavaScript with a sandboxed runtime that captures stdout, stderr, and generated files. |
| execute_python | 1.0.0 | code_execution, analysis | high | No | Execute Python code inside a sandboxed interpreter with stdout capture and timeout enforcement. |
| filesystem_list | 1.0.0 | — | low | No | List directory contents within the sandbox, returning metadata for each entry. |
| filesystem_read | 1.0.0 | — | medium | No | Read a file from the sandboxed filesystem, returning MIME metadata and optionally truncated contents. |
| filesystem_write | 1.0.0 | — | high | No | Write or overwrite a file within the sandboxed filesystem while enforcing byte and total storage quotas. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| get_current_location | 1.0.0 | geolocation, context_awareness | — | No | Retrieve the caller's approximate location using the IP-API geolocation service. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| hitl.approval | 1.0.0 | workflow_control, human_approval | high | No | Request human-in-the-loop approvals, poll for status, and resolve HITL review tasks. |
| incident_summarizer | 0.1.0 | incident_response, executive_reporting | medium | No | Transform incident timelines, impact assessments, and remediation actions into an executive-ready brief. |
| kv_delete | 1.0.0 | state_store | — | No | Remove a key from the namespaced state store if present. |
| kv_get | 1.0.0 | state_store | — | No | Retrieve a JSON-serializable value from the namespaced state store, honoring key TTL. |
| kv_increment | 1.0.0 | state_store | — | No | Atomically increment an integer counter within the namespaced state store, creating it when missing. |
| kv_set | 1.0.0 | state_store | — | No | Persist a JSON-serializable value within the namespaced state store with optional TTL enforcement. |
| labor_market_feed | 1.0.0 | labor_market_research, trend_analysis | low | No | Generate a synthetic labour market snapshot for the requested regions and skills. |
| log.event | 1.0.0 | telemetry, observability | — | No | Validate and emit structured telemetry events with optional message bus persistence hints. |
| log_parser | 0.1.0 | log_analysis, observability | medium | No | Normalize newline-delimited logs, enrich entries with severity and timestamps, and filter by pattern or time window. |
| lyricist | 1.0.0 | lyric_writing, creative_iteration | low | No | Draft a structured lyric sheet with sectioned lines and thematic hooks. |
| memory.graph | 1.0.0 | graph_memory, conversation_context | — | No | Manage tenant-scoped memory graph nodes and edges stored in the conversation repository. |
| memory_episodic_prune | 1.0.0 | episodic_memory, conversation_context | — | No | Delete episodic memories for a tenant using occurrence and expiration filters. |
| memory_episodic_query | 1.0.0 | episodic_memory, conversation_context | — | No | Retrieve episodic memories for a tenant filtered by tags, time bounds, and conversation scope. |
| memory_episodic_store | 1.0.0 | episodic_memory, conversation_context | — | No | Persist an episodic memory snapshot tied to a tenant, optionally linked to a conversation message. |
| mood_map | 1.0.0 | tone_mapping, story_analysis | low | No | Translate story beats into tonal transitions and color palettes. |
| notebook | 1.0.0 | note_taking, knowledge_management | low | No | Capture a structured research note inside a collaborative notebook. |
| notification_service | 1.0.0 | notification_delivery, engagement | medium | No | Send a notification message to the requested channel and recipients. |
| planner.decompose | 1.0.0 | workflow_control, planning | low | No | Produce a dependency graph snapshot for a skill's execution plan without running any tools. |
| policy_reference | 1.0.0 | policy_lookup, risk_assessment_support | — | No | Retrieve internal safety and governance policy guidance relevant to a proposed action. |
| priority_queue | 1.0.0 | task_management, status_reporting | — | No | Score and sort operational tasks to produce a prioritized execution list. |
| query_vectors | 1.0.0 | vector_store | — | Yes (docs: Configure credentials for the selected vector store provider via ConfigManager., type: api_key) | Return the most similar vectors for the supplied query embedding from the configured namespace. |
| registry.capability | 1.0.0 | orchestration, introspection | low | No | Inspect cached capability metadata, apply persona-aware filters, and optionally refresh the shared registry snapshot. |
| roadmap_service | 1.0.0 | program_management, planning | medium | No | Update a roadmap initiative with owner, status, and upcoming milestones. |
| spreadsheet | 1.0.0 | data_management, tabular_editing | low | No | Append or replace rows within a lightweight spreadsheet. |
| storyweaver | 1.0.0 | story_development, narrative_design | low | No | Assemble a structured story outline with tonal guidance and beat-level prompts. |
| structured_parse | 1.0.0 | document_parsing, ocr | low | No | Extract normalized text, tables, and metadata from PDF, DOCX, HTML, CSV, or image documents. |
| sys_snapshot | 0.1.0 | situational_awareness, observability | medium | No | Normalize host inventories, running services, and metrics into a compact situational snapshot. |
| task_queue_cancel | 1.0.0 | task_queue | — | No | Cancel a queued or scheduled task. |
| task_queue_enqueue | 1.0.0 | task_queue | — | No | Enqueue a one-off task into the durable task queue with optional execution delay. |
| task_queue_schedule | 1.0.0 | task_queue | — | No | Register or replace a cron-style recurring task in the durable queue. |
| task_queue_status | 1.0.0 | task_queue | — | No | Retrieve the current state and retry metadata for a queued task. |
| terminal_command | 1.0.0 | terminal_execution, system_inspection | high | No | Execute a constrained terminal command inside the ATLAS sandbox and return stdout, stderr, and exit status. |
| threat_scanner | 0.1.0 | threat_detection, security_analytics | high | No | Score structured events against severity heuristics and threat indicators to surface the riskiest findings. |
| ticketing_system | 1.0.0 | issue_tracking, escalations | medium | No | Create a ticket for follow-up work with assignee and tag support. |
| trace.explain | 1.0.0 | observability, diagnostics | — | No | Merge the tool activity log and recorded negotiation traces with optional filters for conversation, trace ID, and time range. |
| upsert_vectors | 1.0.0 | vector_store | — | Yes (docs: Configure credentials for the selected vector store provider via ConfigManager., type: api_key) | Insert or update vector embeddings within the configured namespace of the active vector store backend. |
| vault.secrets | 1.0.0 | credential_management, configuration_access | — | No | Inspect or update stored credentials for configured tools with masked previews of secret values. |
| visual_prompt | 1.0.0 | visual_prompting, creative_direction | low | No | Compose a descriptive visual art prompt with palette and lighting cues. |
| webpage_fetch | 1.0.0 | web_content, web_research, document_ingestion | — | No | Download an allowlisted webpage, strip scripts and ads, and return clean text with the resolved title and URL. |
| workspace_publisher | 1.0.0 | workspace_publishing, brief_distribution | medium | No | Publish a structured brief payload into a collaborative workspace channel. |

## Persona: Specter

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| audit_reporter | 0.1.0 | governance, audit_trail | medium | No | Aggregate control evidence, map findings to compliance tags, and generate an audit-friendly status payload. |
| context_tracker | 1.0.0 | conversation_state, status_reporting | medium | No | Capture a normalized snapshot of the current investigation thread so Specter can reference prior findings and open actions. |
| get_current_info | 1.0.0 | time_information, date_information | low | No | Return the current timestamp or formatted clock reference for incident timelines. |
| incident_summarizer | 0.1.0 | incident_response, executive_reporting | medium | No | Transform incident timelines, impact assessments, and remediation actions into an executive-ready brief. |
| log.event | 1.0.0 | telemetry, observability | medium | No | Emit structured telemetry to the security event bus with optional metadata for routing. |
| log_parser | 0.1.0 | log_analysis, observability | medium | No | Normalize raw log lines, filter by severity or pattern, and surface the events most relevant to the investigation. |
| policy_reference | 1.0.0 | policy_lookup, governance | low | No | Retrieve the latest security or compliance policy excerpts relevant to the incident. |
| sys_snapshot | 0.1.0 | situational_awareness, observability | medium | No | Normalize host inventories, running services, and metric samples into a compact situational snapshot. |
| threat_scanner | 0.1.0 | threat_detection, security_analytics | high | No | Score parsed events against threat indicators and return the highest-risk findings with rationale. |

## Persona: WeatherGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| geocode_location | 1.0.0 | geolocation, mapping | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Look up latitude and longitude coordinates for a user-specified location string using OpenWeather's geocoding API. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| get_current_weather | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get current and forecast weather data for a location |
| get_daily_weather_summary | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get daily aggregated historical weather data for a location |
| get_historical_weather | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get historical weather data for a location |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| weather_alert_feed | 1.0.0 | weather_alerts, monitoring | — | No | Retrieve active NOAA/NWS weather alerts for a geographic point. |

## Persona: WebDev

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| terminal_command | 1.0.0 | — | high | No | Execute a terminal command and return the output, error, and status code. |

## Persona: genius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| metaphor_palette | 1.0.0 | storytelling, rhetoric | — | No | Generate poetic metaphor sets tuned to the audience. |
