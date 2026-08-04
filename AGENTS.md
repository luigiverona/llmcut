# Repository guidance

- Keep canonical models provider-neutral; adapters translate only, core optimization has no HTTP,
  storage makes no policy decisions, and CLI/proxy share the same engine.
- Preserve the fail-open parity invariants. Never add lossy rewriting, model/reasoning/tool/validation
  reduction, unsupported provider claims, silent field dropping, or unlabeled estimates.
- Proxy changes must retain native round-trip and full-size checks. Index parser changes must bump
  `PARSER_VERSION`; transparent mode must never inject managed retrieval tools.
- Never persist or log credentials. Keep transport, persistence redaction, and diagnostics separate.
- Release measurements must be digest-bound to exact provider payloads or verified captures;
  untrusted fixture usage is never release evidence. MCP retrieval enforces its own repository
  boundary because it executes outside an agent's shell sandbox.
- Use `uv run ruff format .`, `uv run ruff check .`, `uv run mypy src tests`, and `uv run pytest`.
  Add invariant and round-trip tests for behavior changes; mock all network traffic.
- Commit focused changes using Conventional Commits. Do not commit `.llmcut/`, caches, databases,
  credentials, build output, disabled tests, placeholders, or unresolved work markers.
