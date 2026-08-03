# Security and privacy model

Transport content, persisted evidence, and diagnostics are separate boundaries. Redaction applies
only to persistence and diagnostics; it never mutates the provider-bound body. Credential values are
read at request time from configured standard environment variables and are not included in stored
configuration, logs, exceptions, health, or metrics.

The proxy accepts only configured provider names and derives the destination from an allowlisted
HTTP(S) origin. Request-controlled absolute URLs, origin changes, credentials in base URLs,
hop-by-hop headers, oversized bodies, redirects, and unlimited timeouts are rejected or disabled.
Loopback is the default. External binding requires an explicit setting and produces a warning.

Local state directories are mode 0700 and database/config/index files mode 0600. SQLite enables
foreign keys and transactional migrations and never performs a destructive reset. Evidence reads
recompute hashes. Garbage collection preserves checkpoint references. Repository indexing does not
follow symlinks or read Git-ignored/untracked files by default, known secret files, credential stores,
private keys, vendored trees, or local state.

Tests use local ASGI transports and mock servers. CI needs no network or provider credentials.
