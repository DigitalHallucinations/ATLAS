---
audience: all users
status: published
last_verified: 2025-12-21
source_of_truth: Release notes source only
---

# Release Notes

## Unreleased

- No unreleased changes at this time.

## 2025-12-21

- Chat history exports now automatically create any missing parent directories
  before writing the export file. This makes it possible to export directly to
  new, nested folders without preparing them ahead of time.
- ElevenLabs speech synthesis now resolves its cache directory from the
  configured application root (or explicit speech cache settings), ensuring
  custom installations store generated audio in the expected location and log
  permission issues clearly.
- User account password policies can now be customised via configuration keys
  for minimum length and character requirements. See `docs/password-policy.md`
  for the full list of options.
- Core HTTP client dependencies now include `aiohttp`. Make sure local
  environments install it alongside the other runtime requirements listed in
  `requirements.txt`.
- Configuration helpers now depend on `platformdirs` (with `appdirs` remaining
  an optional fallback). Ensure local environments install it with the other
  runtime requirements in `requirements.txt`.
- The conversation store now requires a PostgreSQL DSN. Startup fails fast when
  `CONVERSATION_DATABASE_URL` is unset or uses a non-PostgreSQL dialect, and
  the sample configuration has been updated accordingly.
