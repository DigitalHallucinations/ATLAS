# ATLAS First-Run Setup Wizard

The GTK desktop experience now includes a guided assistant that walks operators
through the essential configuration required to bootstrap ATLAS. The wizard is
presented automatically when the application cannot complete its normal start
up sequence.

## Launching the wizard

The `FirstRunCoordinator` attempts to construct and initialise the main ATLAS
application. When those steps raise an error, a `SetupWizardWindow` window is
presented instead. Completing the wizard re-runs the same bootstrap path and
returns the user to the main interface.

## Step overview

1. **Conversation database** – Pre-fills the PostgreSQL credentials from the
   `_DEFAULT_CONVERSATION_STORE_DSN`. Operators can adjust the host, port,
   database, username, and password. The wizard composes a DSN, validates it
   with `bootstrap_conversation_store`, and persists the confirmed value via
   `ConfigManager._persist_conversation_database_url`.

2. **Key-value store** – Allows choosing whether the KV adapter reuses the
   conversation store engine or connects to a dedicated PostgreSQL database.
   When a dedicated DSN is provided it is saved with
   `ConfigManager.set_kv_store_settings`.

3. **Job scheduling** – Provides a toggle for durable scheduling, a job-store
   DSN, worker concurrency, and retry policy defaults. When enabled, the
   settings are stored in both the `job_scheduling` and `task_queue` blocks
   through `ConfigManager.set_job_scheduling_settings`. The global
   `_initialize_job_scheduling` routine honours the toggle.

4. **Message bus** – Allows choosing between the in-memory backend or Redis.
   Redis configuration captures the URL and stream prefix, persisting them with
   `ConfigManager.set_messaging_settings`.

5. **Providers & defaults** – Lists known LLM providers, surfaces their API key
   environment variables, and records any submitted credentials with
   `ConfigManager.update_api_key`. Operators can pick the default provider and
   model, saved using `set_default_provider` and `set_default_model`.

6. **Speech** – Enables or disables TTS/STT, chooses providers, and captures
   credentials for ElevenLabs, OpenAI, or Google. Speech options are persisted
   using `set_tts_enabled`, `set_default_speech_providers`,
   `set_elevenlabs_api_key`, `set_openai_speech_config`, and
   `set_google_credentials`.

7. **User account** – Collects an administrator username, email, password, and
   display name. The wizard calls `UserAccountService.register_user` and marks
   the created account active via `ConfigManager.set_active_user`.

   User profiles are now persisted directly in the PostgreSQL conversation
   store. When the wizard provisions the administrator account it also seeds
   the corresponding `User.meta` payload through the
   `ConversationStoreRepository`. Downstream services, such as
   `UserDataManager`, resolve profile metadata and EMR/history text from the
   same PostgreSQL database instead of the legacy `user_profiles/` filesystem
   directory.

8. **Optional adjustments** – Exposes tenant identifiers, conversation
   retention, scheduler timezone/queue limits, and an option to auto-start the
   HTTP server. These are stored using `set_tenant_id`,
   `set_conversation_retention`, `set_job_scheduling_settings`, and
   `set_http_server_autostart`.

9. **Summary** – Presents a consolidated view of the captured settings before
   applying changes. Activating *Apply* runs the configured persistence helpers
   and closes the wizard, allowing the coordinator to retry the normal
   initialisation flow.

## Testing

Integration-oriented tests in `tests/test_setup_wizard.py` mock the bootstrap
helpers and assert that each step writes through to the configuration manager.
These tests ensure that every persistence method is invoked with the expected
payload and that configuration blocks receive the new values.
