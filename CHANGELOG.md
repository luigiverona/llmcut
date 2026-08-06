# Changelog

## 0.6.0 - Unreleased

- Added real `off`, `orientation`, `guided`, `adaptive`, and deprecated `legacy-passive` Codex
  context strategies, task-aware orientation, compact retrieval, adaptive opt-out, private run
  state, discovery instrumentation, and release-ineligible pilot reporting.
- Version 0.6.0 remains unreleased until the authenticated live gates pass.
- Recorded the first adaptive pilot as a permanent negative result: all 24 runs preserved quality,
  orientation measured a +2.23% representative median reduction, guided MCP measured -0.26%, and
  Codex made no llmcut MCP calls.
- Added an unreleased Codex `PostToolUse` hook candidate that conservatively compacts supported large
  Bash results while preserving exact digest-verified local recovery.
- The installed-runtime probe did not demonstrate exclusive model-facing replacement, so the hook
  pilot and v0.6 release remain blocked and no hook savings are claimed.

## 0.5.0 - 2026-08-03

- Made the official pinned Python Codex SDK the default live automation backend while retaining the
  direct App Server client as an explicit compatibility backend and the fake runtime for CI.
- Added authentication preflight and separated Codex, MCP, and validation environments without
  reading, copying, logging, or persisting credential contents.
- Replaced the artificial full-repository Codex baseline with the same ordinary task prompt and
  repository access used by optimized runs; synthetic planner measurement remains separately labeled.
- Added explicit payload/agent/provider/subscription claim states, SDK-backed captures, representative
  live suites, and small-sample paired statistics.
- Completed an authenticated 36-run release suite with all deterministic outcomes passing. The suite
  measured no overall agent-input reduction, so no Codex token or subscription saving is claimed.

## 0.4.0 - 2026-08-03

- Replaced fixture-authored release token gates with digest-bound counts of exact provider-bound
  payloads and explicit measurement quality, trust, layer, counter version, and capture provenance.
- Added verified multi-turn capture inspection, offline replay, deterministic redaction, and
  deletion; recorded mock usage is now explicitly untrusted and release-ineligible.
- Added materialized benchmark repositories, isolated paired worktrees, executable acceptance
  commands, patch-scope checks, and all-case release statistics including negative controls.
- Added a standards-compliant MCP stdio server using the maintained official SDK with compact
  repository retrieval tools/resources and independent path, secret, digest, and volume controls.
- Added an experimental isolated Codex App Server/MCP integration with capability detection,
  safe reversible TOML configuration, parity-preserving run settings, and unavailable usage labels.
- Added complete repeated baseline/optimized Codex App Server execution with isolated worktrees,
  deterministic validation, bounded event/usage normalization, safe cleanup, and JSON/text reports.
- Added redacted agent-evaluation capture generation and a lockfile-controlled executable
  TypeScript compiler/runtime benchmark; fake usage remains untrusted integration evidence.

## 0.3.0 - 2026-08-03

- Added versioned provider-neutral managed requests, explicit retention/provenance semantics, and
  separate canonical, model-bound, evidence, provider, and diagnostic serializations.
- Added dependency-aware planning, stable-prefix construction, exact managed retrieval, checkpoint
  history compaction, and task-scoped virtualization for large tool catalogs.
- Added bounded multi-turn execution with complete usage accounting, sync/async Python clients,
  managed CLI and local HTTP APIs, and provider-native continuation for existing adapters.
- Added provider-aware counter registration, managed-only privacy-safe metrics, deterministic
  managed evaluation, realistic saving/no-saving/retrieval-heavy controls, and security tests.

## 0.2.0 - 2026-08-03

- Connected supported proxy routes to native conversion, optimization, semantic validation,
  size-aware reconstruction, and fail-open original-body replay.
- Added distinct strict/parity/extreme policies, safe diagnostic headers, request-level metrics,
  opt-in managed retrieval schemas, and size-aware command-output virtualization.
- Added Tree-sitter JavaScript/TypeScript parsing, Python symbol ranges, dependency/test expansion,
  and a persistent Git-blob keyed incremental repository index.
- Made `llmcut eval` execute deterministic baseline and optimized paths and added a ten-case offline
  benchmark corpus with honest no-savings fallback reporting.

## 0.1.0 - 2026-08-03

- Initial provider-neutral canonical request model and fail-open optimization engine.
- Exact deduplication, content-addressed evidence, recovery, checkpoints, repository packing,
  command-output virtualization, stable-prefix planning, and labeled token accounting.
- OpenAI-compatible, Anthropic Messages, and Gemini generateContent adapters.
- Bounded allowlisted ASGI proxy, complete CLI, SQLite migrations/metrics, offline evaluation harness,
  documentation, tests, packaging, and CI.
