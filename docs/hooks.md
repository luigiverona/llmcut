# Codex hook output compaction

`v0.6.0` changes the primary Codex intervention from voluntary MCP retrieval to the built-in Bash
result path Codex already uses. A `PostToolUse` hook classifies an exact result. Small, unknown,
mutating, network, interactive, package-manager, build, source-read, malformed, or unsupported
results pass through unchanged. Recognized large pytest, type-check, lint, search, and listing output
may become an exact bounded projection only after the complete stdout, stderr, and exit code have
been persisted by digest.

The model-facing replacement prominently preserves exit status, labels output as untrusted data,
states parser limitations, and supplies `llmcut hook show`, `range`, and literal `search` recovery
commands. Projections are deterministic and are not LLM summaries. Recovery output is classified
separately and is never recursively compacted.

## Lifecycle and trust

`llmcut agent codex hooks config` prints the exact proposed `PostToolUse` definition. `install`
requires explicit invocation, preserves unrelated JSON hooks, writes atomically with restrictive
permissions, creates a backup when needed, and never installs persistent trust. Review installed
hooks through Codex `/hooks`. `remove` removes only the matching llmcut handler.

Automated disposable-fixture evaluation may explicitly pass `--allow-hook-trust-bypass`. This uses
Codex's documented one-off `--dangerously-bypass-hook-trust` launch option. It is never enabled by
default and is recorded as an intervention difference. Normal installations must not use managed
enterprise configuration as a trust bypass.

## Evidence and security

Hook evidence is separate from managed evidence because exact recovery cannot use persistence
redaction. It is content-addressed, run-scoped, stored outside repositories under `0700` parents
and `0600` files, digest-verified on every read, size/age bounded, and removable with `llmcut hook
gc`. Metrics contain classifications, digests, sizes, parser versions, timing, and recovery counts,
not command output, prompt text, environment values, or credentials.

The hook never executes output, follows paths found in output, changes repository files, changes an
exit code, or transmits evidence. Hook parsing and patterns are bounded. Any parse, persistence,
parser, timeout, or protocol failure defaults to Codex's original result behavior.

## Parser guarantees and measurement

- Pytest compaction requires a recognized terminal summary. A failing result is compacted only if
  failure/error sections are found and the complete retained tail fits the hard output bound.
- Mypy, basedpyright, TypeScript, and ruff-like diagnostics retain recognized diagnostic lines and
  the final status tail. Ambiguous formats pass through.
- Search/listing compaction only removes byte-identical duplicate lines, preserves first-occurrence
  order and error text, and records the exact duplicate count. It performs no relevance filtering.

Component byte/token estimates diagnose hook behavior. Authenticated agent-reported total input is
the only authority for release savings. Hook support and wire formats can change with Codex; run
`llmcut agent codex hooks doctor` before evaluation.

## Current runtime blocker

The authenticated `codex-cli 0.146.0` probe executed the hook and persisted exact evidence, but the
next model response demonstrated access to both the original Bash result and compact hook feedback.
Exclusive model-facing replacement was therefore not proven. Hook-based live pilots and release
statistics are blocked; byte estimates from this probe are not token-savings evidence.
