# Security and privacy model

Adaptive Codex plan state is bounded, mode 0600, stored below a mode-0700 disposable evaluation
root, and digest-bound. The MCP server verifies its repository allowlist. Cleanup removes it on
success, failure, cancellation, and timeout. Only the task digest is stored; task text is not placed
in MCP process arguments, routine diagnostics, or captures.

## Captures, MCP, and coding agents

Captures are sensitive, digest-verified artifacts. Content persistence is explicit, locations are
root-relative, redaction removes credentials and reasoning fields, replay is offline, and deletion
requires an explicit verified capture directory. Routine metrics contain provenance and counts, not
prompts.

MCP executes outside a coding agent's shell sandbox. The stdio server therefore fixes a repository
root at startup, excludes secret paths and symlink escapes, verifies indexed digests, bounds every
range/search/result, and exposes neither command execution, environment values, nor network access.
Codex configuration changes happen only on explicit `init`, retain a restrictive backup, preserve
unrelated TOML, validate, and replace atomically.

Codex authentication preflight invokes supported login-status diagnostics and never reads the auth
cache. Existing-session discovery variables reach only the Codex process. Validation gets only its
suite allowlist plus safe runtime defaults, and llmcut MCP receives no authentication variables.
Explicit API-key or access-token modes accept an environment-variable name, never a secret value.

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

Native JSON is body-size bounded, UTF-8 decoded, required to be an object, and rejected above 64
levels of nesting. Fixed fallback reasons contain no prompt content or evidence identifiers. Request
metrics contain numeric/accounting metadata only. Transparent mode never injects retrieval tools.

Tests use local ASGI transports and mock servers. CI needs no network or provider credentials.

Managed endpoints share the proxy body and nesting bounds, use unguessable run identifiers, retain
only bounded completed results in memory, and accept no request-provided credential fields. Optional
bearer authentication compares values in constant time. Provider destinations still come only from
configuration. Metrics contain modes, provider/model identifiers, numeric usage, timings, fallback,
and quality state—not prompts, source, logs, retrieval content, or credentials.

Managed evidence is untrusted text: it is never interpreted as configuration or executed. Retrieval
resolves content-addressed identifiers rather than paths, rejects secret-marked/named evidence and
stale revisions, bounds line ranges and result volume, rejects traversal by construction, and limits
regex length and high-risk backtracking constructs. Identical calls are cached; repeated model calls
are stopped. Cancellation and provider failures close the bounded loop without executing external
capabilities.
## Codex hooks

The exec evaluator launches through an argv array and sends task text only on stdin. It bounds JSONL
lines, stderr, and event counts and terminates the process group on timeout or cancellation. User
config, rules, MCP, and baseline hooks are disabled. The credential store is never scanned or copied;
validation and hook metrics receive no credential contents.

Codex hook output is untrusted data. Exact hook evidence is kept outside repositories and captures,
under restrictive permissions, without environment or authentication values. Hook failures pass
through the original tool result. Persistent installation never grants trust; automated trust
bypass is explicit, one-off, and limited to controlled evaluation.

The user-level bridge candidate is static and inert without both a random lease ID and separate
lease token. Leases are restrictive, expiring, repository/cwd/run bound, and outside repositories.
Changes to `$CODEX_HOME/hooks.json` are locked, atomically merged, journaled without file contents,
and restored without overwriting concurrent external additions. Ambiguous cleanup fails closed with
an explicit recovery command. A loopback-only bounded OTLP/HTTP receiver retains only allowlisted
model/settings metadata and discards prompts, source, reasoning, commands, outputs, and unexpected
fields; it has not been used as authenticated release evidence because activation stopped first.
The conformance harness creates unpredictable canaries outside task text and hook configuration,
stores only comparison metadata, and deletes scripts and hook state during cleanup. Capability
claims expire on a runtime-version mismatch.
