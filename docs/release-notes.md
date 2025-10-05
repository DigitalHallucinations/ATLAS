# Release Notes

## Unreleased

- Chat history exports now automatically create any missing parent directories
  before writing the export file. This makes it possible to export directly to
  new, nested folders without preparing them ahead of time.
- ElevenLabs speech synthesis now resolves its cache directory from the
  configured application root (or explicit speech cache settings), ensuring
  custom installations store generated audio in the expected location and log
  permission issues clearly.

