# Evaluation captures

Capture schema `1` stores an ordered multi-turn manifest with provider, model, endpoint, request and
response digests, reasoning-settings digest, usage provenance, redaction version, and explicit
prompt-persistence policy. Content locations must be relative descendants of the capture root.
Metadata-only turns omit content locations; provider-reported usage still requires bound request and
response digests.

Use `llmcut capture inspect`, `verify`, `redact`, `replay`, and `delete`. Replay performs digest
verification and never contacts a provider. Captures are sensitive: authorization, cookies,
credential-like keys, and reasoning-chain fields are deterministically removed; persisted files use
restrictive permissions. Prompt/source persistence must be explicitly enabled. Do not capture
headers, environment dumps, Git credentials, or private prompts for committed fixtures.
