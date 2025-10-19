# Tool Manifest

## Persona: ATLAS

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| context_tracker | 1.0.0 | conversation_state, status_reporting | — | No | Compile a normalized snapshot of the active conversation including recent highlights and participants. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| priority_queue | 1.0.0 | task_management, status_reporting | — | No | Score and sort operational tasks to produce a prioritized execution list. |

## Persona: Cleverbot

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: CodeGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| execute_python | 1.0.0 | code_execution, analysis | high | No | Execute Python code inside a sandboxed interpreter with stdout capture and timeout enforcement. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| terminal_command | 1.0.0 | — | high | No | Execute a terminal command and return the output, error, and status code. |

## Persona: DocGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: Einstein

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: MEDIC

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| search_pmc | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Search PubMed Central (PMC) for open access articles using the Entrez API. |
| search_pubmed | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY) | Search PubMed for biomedical literature using the NCBI Entrez API. |

## Persona: Nikola Tesla

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: ResumeGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: Shared

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| browser_lite | 1.0.0 | browser_automation | high | No | Navigate a limited number of allowlisted pages with robots.txt enforcement, throttling, and optional screenshots. |
| calculator | 1.0.0 | calculator | low | No | Evaluate sanitized mathematical expressions with Decimal precision and optional unit conversions. |
| context_tracker | 1.0.0 | conversation_state, status_reporting | — | No | Compile a normalized snapshot of the active conversation including recent highlights and participants. |
| debian12_calendar | 1.1.0 | calendar_read, calendar_write | — | No | Interact with the Debian 12 on-device calendar to list, search, inspect, and manage events. |
| delete_namespace | 1.0.0 | vector_store | — | Yes (docs: Configure credentials for the selected vector store provider via ConfigManager., type: api_key) | Remove all stored embeddings associated with the specified namespace. |
| execute_javascript | 1.0.0 | javascript_exec, code_execution | high | No | Execute JavaScript with a sandboxed runtime that captures stdout, stderr, and generated files. |
| execute_python | 1.0.0 | code_execution, analysis | high | No | Execute Python code inside a sandboxed interpreter with stdout capture and timeout enforcement. |
| filesystem_list | 1.0.0 | — | low | No | List directory contents within the sandbox, returning metadata for each entry. |
| filesystem_read | 1.0.0 | — | medium | No | Read a file from the sandboxed filesystem, returning MIME metadata and optionally truncated contents. |
| filesystem_write | 1.0.0 | — | high | No | Write or overwrite a file within the sandboxed filesystem while enforcing byte and total storage quotas. |
| geocode_location | 1.0.0 | geolocation, mapping | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Look up latitude and longitude coordinates for a user-specified location string using OpenWeather's geocoding API. |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| get_current_location | 1.0.0 | geolocation, context_awareness | — | No | Retrieve the caller's approximate location using the IP-API geolocation service. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| kv_delete | 1.0.0 | state_store | — | No | Remove a key from the namespaced state store if present. |
| kv_get | 1.0.0 | state_store | — | No | Retrieve a JSON-serializable value from the namespaced state store, honoring key TTL. |
| kv_increment | 1.0.0 | state_store | — | No | Atomically increment an integer counter within the namespaced state store, creating it when missing. |
| kv_set | 1.0.0 | state_store | — | No | Persist a JSON-serializable value within the namespaced state store with optional TTL enforcement. |
| policy_reference | 1.0.0 | policy_lookup, risk_assessment_support | — | No | Retrieve internal safety and governance policy guidance relevant to a proposed action. |
| priority_queue | 1.0.0 | task_management, status_reporting | — | No | Score and sort operational tasks to produce a prioritized execution list. |
| query_vectors | 1.0.0 | vector_store | — | Yes (docs: Configure credentials for the selected vector store provider via ConfigManager., type: api_key) | Return the most similar vectors for the supplied query embedding from the configured namespace. |
| search_pmc | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY, type: api_key) | Search PubMed Central for open access biomedical literature and return PMC IDs. |
| search_pubmed | 1.0.0 | medical_research, literature_search | — | No (docs: Optional Entrez API key for higher rate limits., env: NCBI_API_KEY, type: api_key) | Search PubMed for biomedical literature and return a list of PubMed IDs. |
| structured_parse | 1.0.0 | document_parsing, ocr | low | No | Extract normalized text, tables, and metadata from PDF, DOCX, HTML, CSV, or image documents. |
| task_queue_cancel | 1.0.0 | task_queue | — | No | Cancel a queued or scheduled task. |
| task_queue_enqueue | 1.0.0 | task_queue | — | No | Enqueue a one-off task into the durable task queue with optional execution delay. |
| task_queue_schedule | 1.0.0 | task_queue | — | No | Register or replace a cron-style recurring task in the durable queue. |
| task_queue_status | 1.0.0 | task_queue | — | No | Retrieve the current state and retry metadata for a queued task. |
| terminal_command | 1.0.0 | terminal_execution, system_inspection | high | No | Execute a constrained terminal command inside the ATLAS sandbox and return stdout, stderr, and exit status. |
| upsert_vectors | 1.0.0 | vector_store | — | Yes (docs: Configure credentials for the selected vector store provider via ConfigManager., type: api_key) | Insert or update vector embeddings within the configured namespace of the active vector store backend. |
| webpage_fetch | 1.0.0 | web_content, web_research, document_ingestion | — | No | Download an allowlisted webpage, strip scripts and ads, and return clean text with the resolved title and URL. |

## Persona: WeatherGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| get_current_weather | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get current and forecast weather data for a location |
| get_daily_weather_summary | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get daily aggregated historical weather data for a location |
| get_historical_weather | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get historical weather data for a location |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (docs: Set GOOGLE_API_KEY and GOOGLE_CSE_ID to use Google Programmable Search. When those are missing, configure SERPAPI_KEY as a fallback., env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

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
