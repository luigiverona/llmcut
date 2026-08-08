# llmcut

`llmcut` is a provider-neutral context optimization and observability layer for LLM applications,
coding agents, and API clients. It keeps the smallest confidently sufficient working set in model
context, stores original evidence locally, and expands context when exclusion could affect quality.

The unreleased v0.6 work adds adaptive Codex context routing and a structured `codex exec --json`
evaluation backend. The SDK remains the general automation default; hook-savings evaluation uses
direct exec because the SDK/App Server surface did not activate lifecycle hooks. It remains
conservative: **extreme is the default but is not lossy**. The
optimizer does not change models, reasoning settings, tools, validation requirements, or source
code. Low-confidence material stays in context; recoverable omission requires explicit,
high-confidence evidence. The policy is enforced during optimization, but outcome parity is not
independently established until the evaluation harness validates the workload.

## Install and initialize

Python 3.12 or newer is required.

```bash
uv tool install llmcut
llmcut init
llmcut doctor
```

For development:

```bash
uv sync --all-extras --frozen
uv run llmcut --help
```

Generated project configuration and private state live under `.llmcut/` with restrictive
permissions and are Git-ignored. Configuration precedence is CLI arguments, `LLMCUT_*`
environment variables, project `.llmcut/config.toml`, user config, then safe defaults.

## Use

Run or inspect a managed request:

```bash
llmcut run --request request.json --mode extreme --dry-run
llmcut run --request request.json --mode extreme
```

The Python SDK exposes the same protocol without provider SDK coupling:

```python
from llmcut import Client, Context, ManagedRequest

client = Client.from_config()
result = client.run(
    ManagedRequest(
        provider="anthropic",
        model="unchanged-model",
        task="Fix the callback timeout",
        context=[Context.source("src/auth/callback.py", source)],
    ),
    mode="extreme",
)
```

`AsyncClient` provides the corresponding awaitable API. Persistence, timeout, and cancellation are
caller-controlled. The local server exposes synchronous `POST /managed/run`, dry-run
`POST /managed/plan`, and `GET /managed/runs/{id}`. It is loopback-only by default; setting the
environment variable named by `proxy.managed_bearer_token_env` enables local bearer authentication.

Build a deterministic repository pack:

```bash
llmcut pack --repo . --task "Fix the OAuth callback timeout and add regression tests" \
  --mode extreme --format markdown
```

Optimize a provider-neutral request:

```bash
llmcut optimize --input request.json --mode parity > optimized.json
cat request.json | llmcut optimize --dry-run
```

Inspect evidence and metrics:

```bash
llmcut evidence list
llmcut evidence get sha256:...
llmcut stats
```

Configure an allowlisted upstream in `.llmcut/config.toml`:

```toml
[provider.openai]
kind = "openai"
base_url = "https://api.openai.com/v1"
credential_env = "OPENAI_API_KEY"
```

Then run `llmcut proxy` and send requests to
`http://127.0.0.1:8765/openai/chat/completions`. OpenRouter, Ollama, and vLLM are supported only
through their OpenAI-compatible HTTP interfaces and configured base URLs. External binding emits
an actionable warning and must be explicitly configured.

Supported proxy routes perform native → canonical → optimization → native reconstruction and
semantic safety validation before contacting the upstream. The reconstructed body is forwarded only
when the complete native request is safe and smaller; otherwise the original bytes are replayed.
Diagnostic response headers report status, mode, counts, and quality without prompt data. Set
`proxy.diagnostic_headers = false` to disable them.

## Architecture

The canonical dataclass model preserves ordered role-bearing blocks, tool-call metadata, unknown
provider fields, model/reasoning settings, token-count provenance, and evidence references. The
core optimizer is independent of HTTP. Provider adapters only translate formats. SQLite storage
contains evidence and measurements but no policy decisions. CLI and ASGI proxy share configuration,
storage, and security behavior. See [architecture](docs/architecture.md) and
[security](docs/security.md).

Context is byte-stably partitioned into policy, tools, repository structure, task, and dynamic
content. Cacheable input is reported separately from logical context reduction.

Canonical state, provider transport, evidence manifests, and diagnostic reports have separate
serializations. Only model-bound block content and tool definitions enter logical token counts;
digests, local paths, selection decisions, recovery references, and stored sizes never do.

## Modes

Optimization and integration are orthogonal: `strict + transparent`, `extreme + transparent`,
`parity + managed`, and `extreme + managed` are valid combinations.

| Mode | v0.3.0 behavior |
|---|---|
| `strict` | Exact duplicate removal and stable/cache planning only |
| `parity` | Strict plus proven redundancy, superseded checkpoints, and verified command output |
| `extreme` | Parity plus symbol ranges, dependency-aware packing, scoped tools, and disclosure APIs |
| `economy` | Configuration is reserved; selecting it returns a clear not-implemented error |

No mode in v0.3.0 enables lossy context, model downgrade, reasoning reduction, tool reduction that
cannot be reversed, or validation reduction.

## Provider support

| Provider/API | Canonical conversion | Proxy | Streaming |
|---|---:|---:|---:|
| OpenAI Chat Completions | Yes | Yes | Transparent passthrough |
| OpenAI Responses | Yes | Yes | Transparent passthrough |
| OpenAI-compatible / OpenRouter / Ollama / vLLM | Compatible fields preserved | Yes | Transparent passthrough |
| Anthropic Messages | System, messages, tools, cache usage | Yes | Transparent passthrough |
| Gemini generateContent | Contents, parts, functions, cache/usage | Yes | Transparent passthrough |

Streaming optimization occurs before connection; chunks are bounded by upstream
backpressure and never accumulated as an unbounded response. Safely available usage metadata can
be recorded by integrations; the transparent route does not invent unavailable usage.

## Transparent and managed recovery

`transparent` is the proxy default, preserves primary messages byte-for-byte, and never injects
tools. Missing context cannot be added during an already-running response unless the client
participates in another turn. Managed schema version `1` classifies system/developer/user/assistant
messages, tool calls/results, source, repository maps, configuration, tests, command output,
checkpoints, documents, and current tasks. Retention is `required`, `stable`, `recoverable`,
`superseded`, `redundant`, or `ephemeral`; critical instructions and the current task cannot be made
removable.

The managed planner retains policy, task, active tool continuity, named files/symbols, dependencies,
tests/configuration, checkpoints, and unresolved failures in that order. Deferred evidence is stored
exactly and exposed only through the task-scoped operations `evidence.get`, `source.range`,
`symbol.get`, `dependency.get`, `log.search`, `log.range`, `context.expand`, and `repository.map`.
Tool catalogs may additionally expose `tool.discover`. Results are digest-verified, bounded,
provenanced, cached, and added monotonically. Execution has turn, timeout, retrieval-volume, token,
and cancellation bounds and never runs shell commands.

> Transparent proxying can only remove transformations proven equivalent from the native request
> alone. Meaningful context omission requires managed integration or a participating client.

## Security and privacy

There is no telemetry. Credentials come only from explicit provider environment variables, are
added at transport time, and are neither logged nor persisted. Authorization and common secret
patterns are redacted in persisted evidence without changing the actual provider request.
Repository traversal excludes ignored content, common credential files, external symlinks,
vendored trees, and `.llmcut/`. The proxy bounds request bodies, filters hop-by-hop headers,
allowlists configured upstream origins, rejects request-controlled URLs, uses timeouts, and emits
no prompts from health or metrics endpoints.

Prompt persistence is on by default so exact restoration works. Set `persist_prompt_content = false`
for metadata-only deployments; exact restoration is then unavailable by design and clearly marked.
Retention and referenced-evidence garbage collection are local and explicit. Back up `.llmcut/state.db`
with SQLite's backup mechanism while no write transaction is active; never delete it to solve a
migration or corruption error.

## Savings and evaluation

The v0.3 recorded corpus contained editable provider-response usage values. Those records remain
useful for adapter parsing and continuation replay, but v0.4 labels them `untrusted_fixture` and
excludes them from release statistics. Release counts are derived from the exact native payloads,
bound to SHA-256 request digests, and validated with executable outcomes in paired clean worktrees:

```bash
llmcut eval --corpus tests/fixtures/benchmarks/suite.toml
llmcut tokens count --provider openai --model offline-model --input request.json
```

Payload estimates, provider-reported usage, agent-reported usage, and subscription usage are
distinct layers. llmcut does not infer subscription units from token counts. See
[evaluation](docs/evaluation.md) and [captures](docs/captures.md).

Expose allowlisted repository retrieval over MCP stdio with `llmcut mcp serve --repo .`. Generate a
Codex configuration snippet with `llmcut agent codex config --repo .`; mutation occurs only through
the explicit, backed-up, atomic `init` command. The Codex integration is experimental and no Codex
token or subscription reduction is claimed without a live supported harness measurement. See
[MCP](docs/mcp.md) and [Codex integration](docs/integrations/codex.md).

The first v0.6 pilot preserved quality but did not satisfy the release gate. Orientation produced a
+2.23% representative median reduction, guided MCP produced -0.26%, and no live MCP retrieval calls
occurred. MCP retrieval remains available for diagnostics, but it is not the default Codex
optimization path because authenticated pilots produced zero MCP calls.

The hook candidate uses the documented `PostToolUse` `decision:"block"` response to replace
supported large Bash results
with exact bounded projections and digest-based Bash recovery. Hook compaction acts only on
supported local tool results. Unknown, small, mutating, interactive, or unsafe-to-transform outputs
pass through unchanged. See [hook output compaction](docs/hooks.md).

The latest isolated user-hook bridge probe remained activation-blocked: Codex completed with
authoritative usage, but no protected lease-bound hook event was observed. The bridge was restored
cleanly and no pilot or release suite was run. This is not token-savings evidence.

Executable A/B evaluation is available with
`llmcut agent eval --agent codex --backend sdk --suite suite.toml`. It uses isolated paired
worktrees, repeated and recorded ordering, official SDK events, argv-safe deterministic validation,
changed-file restrictions, and optional verified captures. The standard baseline sends the same
ordinary task prompt as the optimized run and does not concatenate repository files. The only
intervention is llmcut MCP and managed retrieval in the optimized run. Use `--dry-run` to validate
the suite and execution plan. The
TypeScript benchmark includes a lockfile-controlled compiler/type-check plus runtime test rather
than structural source inspection.

Token values say whether they are exact, provider-reported, tokenizer-derived, or estimated. The
built-in fallback is a conservative UTF-8 byte estimate and is never labeled exact. Logical input
reduction, cached input, billed usage, recovery overhead, retries, output tokens, and reasoning
tokens remain separate.

The counter registry prefers a configured provider count call, then an official model tokenizer, a
documented compatible tokenizer, and finally the conservative estimate. Count calls are optional,
timeout-bounded, and digest-cached; they are not generation calls or included in generation usage.
The implementation follows Anthropic's `POST /v1/messages/count_tokens` semantics (a model-specific
estimate that may differ slightly from creation usage) and Gemini's
`models/{model}:countTokens` semantics (the full `generateContentRequest` is needed to include system
instructions and tools). OpenAI generation responses are the authority for reported input, cached,
output, and reasoning usage; no undocumented OpenAI preflight endpoint is claimed. See the official
[Anthropic token-counting guide](https://platform.claude.com/docs/en/build-with-claude/token-counting),
[Gemini countTokens reference](https://ai.google.dev/api/tokens), and
[OpenAI Responses usage reference](https://platform.openai.com/docs/api-reference/responses).

The JSONL evaluator executes baseline and optimized paths through the same deterministic recorded or
fake provider, checks identical settings and responses, runs optional argv-safe evaluators, reports
attempted versus effective tokens, and exits nonzero on regressions. Bundled cases cover provider
shapes, history, repository selection, pytest output, and honest no-savings fallback. They do not
establish real-provider parity. No fixed reduction is guaranteed.

Managed evaluation counts initial and every continuation provider input. Retrieval request/result
sizes, outputs, reasoning, and cache reports remain separate diagnostics. A managed case saves only
when total provider input across all turns is lower than the full-context baseline and deterministic
quality checks pass. `tests/fixtures/eval/managed.jsonl` contains long-history, repository, test-log,
60-tool catalog, documentation, no-savings, and retrieval-heavy controls. These are recorded mock
measurements, not real-provider quality claims.

### Bundled offline benchmark (Python 3.14.6)

Counts below are conservative estimates. All ten deterministic response comparisons passed; cached
tokens were zero and remain separate. “Not smaller” means the original request was selected.

| Case | Original | Attempted | Effective | Reduction | Fallback |
|---|---:|---:|---:|---:|---|
| repeated-system | 406 | 439 | 406 | 0% | not smaller |
| duplicated-tools | 372 | 419 | 372 | 0% | not smaller |
| checkpoint-history | 364 | 405 | 364 | 0% | not smaller |
| repository-symbol-range | 305 | 267 | 267 | 12.459% | none |
| pytest-failure-output | 349 | 264 | 264 | 24.3553% | none |
| OpenAI Chat | 228 | 393 | 228 | 0% | not smaller |
| OpenAI Responses | 220 | 385 | 220 | 0% | not smaller |
| Anthropic Messages | 235 | 402 | 235 | 0% | not smaller |
| Gemini content | 229 | 394 | 229 | 0% | not smaller |
| no-safe-savings | 153 | 251 | 153 | 0% | not smaller |
Quality parity must be measured per workload.

## Limitations

- Closed clients with no proxy, API, or plugin integration point cannot be optimized.
- Authenticated v0.5 Codex evaluation measured no overall agent-input reduction in its bounded live
  suite. No Codex subscription reduction is claimed because subscription accounting was unavailable.
- `v0.5.0` measured no normal Codex token reduction. The passive MCP integration added schemas but
  was not invoked by Codex.
- `v0.6.0` uses task-aware orientation, compact retrieval, and adaptive opt-out. It is released only
  if authenticated agent-reported usage demonstrates improvement without quality regression.
- Provider tokenization varies by model; local estimates do not replace provider-reported usage.
- Caching can lower billed input without lowering logical context.
- `llmcut` does not bypass provider quotas or accounting.
- JavaScript and TypeScript use maintained Tree-sitter grammars; malformed syntax falls back safely.
- Transparent mode cannot recover context during an already-running provider response.
- Streaming response bodies are passed through; full streaming usage observation depends on what the
  provider reports and what the caller integration records.
- Economy routing is not implemented.

## Development and validation

```bash
uv sync --all-extras --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src tests
uv run pytest
uv run pytest --cov=llmcut --cov-report=term-missing
uv build
```

`llmcut benchmark` reports local fixture indexing time only. It is not a production-scale claim.
