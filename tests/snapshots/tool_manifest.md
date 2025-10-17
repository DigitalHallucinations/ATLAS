# Tool Manifest

## Persona: ATLAS

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: Cleverbot

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: CodeGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| execute_python | 1.0.0 | — | high | No | Execute Python code and return the result |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| terminal_command | 1.0.0 | — | high | No | Execute a terminal command and return the output, error, and status code. |

## Persona: DocGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: Einstein

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: MEDIC

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| search_pmc | 1.0.0 | — | high | No | Searches PMC for a specific article using its PMCID, and downloads the article package to the Workspace directory. |
| search_pubmed | 1.0.0 | — | — | No | Searches PubMed for articles related to the given query. |

## Persona: Nikola Tesla

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: ResumeGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: Shared

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| policy_reference | 1.0.0 | policy_lookup, risk_assessment_support | — | No | Retrieve internal safety and governance policy guidance relevant to a proposed action. |

## Persona: WeatherGenius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| get_current_weather | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get current and forecast weather data for a location |
| get_daily_weather_summary | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get daily aggregated historical weather data for a location |
| get_historical_weather | 1.0.0 | — | — | Yes (env: OPENWEATHERMAP_API_KEY, type: api_key) | Get historical weather data for a location |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |

## Persona: WebDev

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
| terminal_command | 1.0.0 | — | high | No | Execute a terminal command and return the output, error, and status code. |

## Persona: genius

| Name | Version | Capabilities | Safety Level | Auth Required | Description |
| --- | --- | --- | --- | --- | --- |
| get_current_info | 1.0.0 | time_information, date_information | — | No | Get the current time, date, day, month and year, or timestamp based on the format specified. |
| google_search | 1.0.0 | web_search, knowledge_lookup | — | Yes (env: GOOGLE_API_KEY, type: api_key) | A Google search result API. When you need a short and clear answer to a specific question, you can use it. The input should be a search query. |
