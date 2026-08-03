# llmcut

`llmcut` is a provider-neutral context optimization and observability layer for LLM applications,
coding agents, and API clients. It keeps the smallest confidently sufficient working set in model
context, stores original evidence locally, and expands context when exclusion could affect quality.

Version 0.1.0 is intentionally conservative: **extreme is the default but is not lossy**. The
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

## Architecture

The canonical dataclass model preserves ordered role-bearing blocks, tool-call metadata, unknown
provider fields, model/reasoning settings, token-count provenance, and evidence references. The
core optimizer is independent of HTTP. Provider adapters only translate formats. SQLite storage
contains evidence and measurements but no policy decisions. CLI and ASGI proxy share configuration,
storage, and security behavior. See [architecture](docs/architecture.md) and
[security](docs/security.md).

Context is byte-stably partitioned into policy, tools, repository structure, task, and dynamic
content. Cacheable input is reported separately from logical context reduction.

## Modes

| Mode | v0.1.0 behavior |
|---|---|
| `strict` | Exact duplicate removal and stable/cache planning only |
| `parity` | Strict plus recoverable selection, evidence, checkpoints, and repository packing |
| `extreme` | Default; tighter recoverable selection and cache partitioning, same parity floor |
| `economy` | Configuration is reserved; selecting it returns a clear not-implemented error |

No mode in v0.1.0 enables lossy context, model downgrade, reasoning reduction, tool reduction that
cannot be reversed, or validation reduction.

## Provider support

| Provider/API | Canonical conversion | Proxy | Streaming |
|---|---:|---:|---:|
| OpenAI Chat Completions | Yes | Yes | Transparent passthrough |
| OpenAI Responses | Yes | Yes | Transparent passthrough |
| OpenAI-compatible / OpenRouter / Ollama / vLLM | Compatible fields preserved | Yes | Transparent passthrough |
| Anthropic Messages | System, messages, tools, cache usage | Yes | Transparent passthrough |
| Gemini generateContent | Contents, parts, functions, cache/usage | Yes | Transparent passthrough |

Streaming optimization is intentionally transparent in 0.1.0: chunks are bounded by upstream
backpressure and never accumulated as an unbounded response. Safely available usage metadata can
be recorded by integrations; the transparent route does not invent unavailable usage.

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

Token values say whether they are exact, provider-reported, tokenizer-derived, or estimated. The
built-in fallback is a conservative UTF-8 byte estimate and is never labeled exact. Logical input
reduction, cached input, billed usage, recovery overhead, retries, output tokens, and reasoning
tokens remain separate.

The JSONL evaluation harness executes baseline and optimized requests through the same executor and
checks identical provider/model/reasoning settings plus deterministic invariants. CI uses fake
providers and local fixtures; real-provider execution is optional. No fixed reduction is guaranteed.
Quality parity must be measured per workload.

## Limitations

- Closed clients with no proxy, API, or plugin integration point cannot be optimized.
- Provider tokenization varies by model; local estimates do not replace provider-reported usage.
- Caching can lower billed input without lowering logical context.
- `llmcut` does not bypass provider quotas or accounting.
- JavaScript/TypeScript symbol extraction is deliberately conservative lexical extraction in 0.1.0,
  not a claim of complete parsing.
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
