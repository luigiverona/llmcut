# Experimental Codex integration

The integration uses supported Codex surfaces: MCP configuration supplies llmcut retrieval tools,
and `codex app-server` JSON-RPC starts and observes explicit threads/turns. Core models do not import
Codex and no Codex internals, ChatGPT credentials, or private backend APIs are intercepted.

`llmcut agent codex doctor` detects the executable, version, App Server, MCP support, configuration
location, and token-event availability without printing credentials. `config` prints a minimal TOML
snippet. `init` is the only mutating command; it preserves unrelated TOML, validates before writing,
backs up the original, atomically replaces the file with mode 0600, supports `--dry-run`, and can
remove only llmcut's table.

`run` sends explicit model, reasoning effort, working directory, sandbox, and approval policy to App
Server and invalidates a run if Codex reports model rerouting. Operational events exclude private
reasoning. Token usage is `agent_reported` only when `thread/tokenUsage/updated` is emitted;
otherwise it is unavailable. Subscription usage is always `subscription_unavailable` unless a
future supported interface exposes it. No measured Codex or subscription savings are claimed by the
offline release suite.

## Executable A/B evaluation

`llmcut agent eval --agent codex --suite suite.toml` performs real, non-dry-run App Server execution.
For every task and repetition it materializes separate baseline and optimized Git worktrees at the
same commit, verifies tracked-file and execution-setting parity, starts a fresh App Server process,
and runs validation directly as argv arrays. The baseline supplies full repository context; the
optimized run supplies the task plus the same llmcut MCP server with managed retrieval enabled. Both
runs retain the same tool semantics, model, reasoning effort, sandbox, approvals, limits, task, and
validation.

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
