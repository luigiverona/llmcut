# Experimental Codex integration

## Adaptive context routing (v0.6 candidate)

`off` is ordinary Codex and the standard baseline. `orientation` supplies deterministic,
metadata-only working-set guidance through Codex developer instructions without MCP. `guided` adds
the single compact `llmcut_context` tool. `adaptive` chooses among those three from repository size,
task specificity, candidate confidence, large evidence, and estimated cost. `legacy-passive`
preserves the v0.5 eight-tool surface for deprecated diagnostic comparison.

Orientation contains only relative paths, indexed symbols and relationships, sizes, digests, and
planner reasons. It defaults to a hard 200 estimated-token budget and is discarded if it cannot fit.
It never contains source text, absolute paths, credentials, or validation policy. Inspection via
`llmcut context plan` and `llmcut mcp inspect` emits no source contents. Evaluation supports
`--context-strategy adaptive` and release-ineligible `--pilot` runs.

`v0.5.0` measured no normal Codex token reduction. The passive MCP integration added schemas but was
not invoked by Codex. `v0.6.0` uses task-aware orientation, compact retrieval, and adaptive opt-out.
It is released only if authenticated agent-reported usage demonstrates improvement without quality
regression.

The default live backend is the official `openai-codex` Python SDK with its pinned Codex runtime.
MCP configuration supplies llmcut retrieval tools. The direct `codex app-server` JSON-RPC client is
an explicitly selected compatibility backend; the fake runtime exercises the same evaluator in CI.
Core models do not import the SDK, and no Codex internals, credential contents, or private APIs are
intercepted.

`llmcut agent codex doctor` detects the CLI and SDK versions, pinned runtime, App Server, MCP,
authentication category, and live readiness. `llmcut agent codex auth` reports only authentication
availability, method category, credential-store category, Codex-home category, and automation
readiness. It never reads or prints the credential store. The Codex process receives only the
environment needed to discover its existing session; validation receives its explicit suite
allowlist and safe runtime defaults; MCP receives no Codex credentials.

The doctor also detects the configuration
location, and token-event availability without printing credentials. `config` prints a minimal TOML
snippet. `init` is the only mutating command; it preserves unrelated TOML, validates before writing,
backs up the original, atomically replaces the file with mode 0600, supports `--dry-run`, and can
remove only llmcut's table.

Execution sends explicit model, reasoning effort, working directory, sandbox, and approval policy
through the SDK and invalidates a run if Codex reports model rerouting. Operational events exclude private
reasoning. Token usage is `agent_reported` only when `thread/tokenUsage/updated` is emitted;
otherwise it is unavailable. Subscription usage is always `subscription_unavailable` unless a
future supported interface exposes it. No measured Codex or subscription savings are claimed by the
offline release suite.

## Executable A/B evaluation

`llmcut agent eval --agent codex --backend sdk --suite suite.toml` performs real, non-dry-run SDK execution.
For every task and repetition it materializes separate baseline and optimized Git worktrees at the
same commit, verifies tracked-file and execution-setting parity, starts a fresh App Server process,
and runs validation directly as argv arrays. Under the default `standard-baseline` design both modes
receive the same ordinary task prompt and repository access; baseline has no llmcut optimization,
while optimized enables llmcut MCP, planning, and retrieval. Core execution settings remain equal,
and the report identifies the MCP integration as the intervention. `tool-parity-baseline` loads the
same MCP schemas in both modes for a separate experiment. `synthetic-full-context` retains the v0.4
planner benchmark and is never mixed into standard Codex statistics.

A version 1 suite has this shape:

```toml
schema_version = "1"
agent = "codex"
repetitions = 3
order = "random" # baseline-first, optimized-first, alternating, or random
seed = 1729
timeout_seconds = 900

[execution]
model = "MODEL"
reasoning_effort = "high"
sandbox = "workspace-write"
approval_policy = "never"
backend = "sdk"
auth_mode = "existing-session"
comparison_design = "standard-baseline"
environment_allowlist = []

[[tasks]]
id = "python-timeout"
repository = "repositories/python-timeout"
starting_ref = "HEAD"
prompt = "Fix the timeout defect and run validation."
validation = [["python", "tests/validate_callback.py"]]
allowed_changes = ["app/callback.py"]
forbidden_changes = ["README.md"]
required_files = ["app/callback.py"]
max_turns = 2
```

All paths and bounds are validated before worktrees are created; validation never invokes a shell.
CLI overrides take precedence. `--dry-run` validates and prints the deterministic plan without
starting Codex. `--keep-worktrees` preserves the registered evaluation tree; otherwise successful
and failed runs are cleaned safely. `--format json --output result.json` writes a versioned,
mode-0600 report atomically. `--capture DIR` writes bounded, redacted event and digest metadata that
can be checked with `llmcut capture verify DIR`.

The client performs `initialize`/`initialized`, thread creation, bounded turn creation and event
consumption, interruption, graceful shutdown, then forced termination after a bounded grace period.
Malformed protocol data, unsupported versions, process exits, rerouting, timeout, and cancellation
fail safely. Unknown noncritical events are retained as bounded opaque metadata. Reasoning content,
prompts, credentials, and environment values are not placed in routine reports or captures.

A correction turn is a subsequent turn started only after deterministic validation failed or the
previous turn failed to complete. First-attempt success therefore requires a completed first turn
and passing validation. Final quality also enforces required files, allowed/forbidden changed paths,
repository integrity, and validation exit codes; response prose is not an acceptance criterion.

Exit status 0 means all eligible task outcomes passed, 1 means execution completed with a quality
failure, 2 is CLI/schema misuse, and 3 is an unsupported runtime or internal execution failure.

CI uses a subprocess fake App Server over the identical JSON-RPC path. For a local smoke test, run a
small disposable suite only after `llmcut agent codex doctor` confirms an installed, authenticated,
compatible App Server. A single smoke comparison is operational evidence, not statistical evidence.

> `llmcut agent eval` can measure payload reduction and deterministic task outcomes. Agent token
> reduction is reported only when Codex exposes supported usage events. Subscription usage is
> unavailable unless explicitly exposed by the subscription system.

The release-gating v0.5 authenticated suite completed six tasks with three repetitions per mode and
no deterministic quality regressions. Supported agent usage events were available, but the median
paired agent-input result showed no reduction. Accordingly llmcut makes no Codex token-reduction or
subscription-reduction claim from this release.
## Hook-based strategies

Supported intervention names are `off`, `orientation`, `compact-output`, `hybrid`, `guided-mcp`,
deprecated `legacy-passive`, and `adaptive`. Compatibility name `guided` retains the compact MCP
diagnostic. Adaptive does not select MCP merely because a repository is large. Hook evaluation
requires the explicit one-off trust-bypass option and a disposable fixture. See
[hooks](../hooks.md).
