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

Current Codex documentation states that project configuration is loaded only for trusted projects;
an untrusted project skips its `.codex/` configuration, including `config.toml`, `hooks.json`, and
rules. The supported durable control is `projects.<absolute-path>.trust_level = "trusted"`.
Project hooks may be declared in `.codex/hooks.json` or `.codex/config.toml`. Matching hook groups
from multiple sources are cumulative, so an evaluation must not declare the same llmcut hook both
as a project hook and as an inline CLI hook. See the official [configuration
reference](https://developers.openai.com/codex/config-reference), [advanced configuration
guide](https://developers.openai.com/codex/config-advanced), [hooks
guide](https://developers.openai.com/codex/hooks), and [`codex exec` CLI
reference](https://developers.openai.com/codex/cli/reference#codex-exec).

`--dangerously-bypass-hook-trust` runs already-enabled hooks without persisted definition trust; it
does not promise to activate an otherwise skipped project layer. `--ignore-user-config` disables
the normal `$CODEX_HOME/config.toml` layer while authentication discovery still uses `CODEX_HOME`.
On `codex-cli 0.146.1`, an invocation-only `projects."<worktree>".trust_level="trusted"` override
was accepted but did not make a disposable worktree's project hooks observable. CLI-only,
duplicated CLI+project, and temporary-profile variants also produced command events but no hook
events under user-config isolation. The same runtime still produced exclusive replacement from the
already trusted repository with normal configuration loading. Metadata is retained in
`docs/evidence/v06-hook-source-activation.json`; this is an activation blocker, not savings evidence.

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

## Conformance result

The first hook probe tested a combined `continue:false`, `stopReason`, and `additionalContext`
response. It did not establish that every supported PostToolUse replacement form was ineffective.
The follow-up matrix tested each documented form separately with unpredictable canaries.
`additionalContext` was additive; the tested `continue:false` forms were not exclusive. The smallest
exclusive response was `decision:"block"` with `reason`, repeated twice on `codex-cli 0.146.0` and
also reproduced on the SDK-pinned `codex-cli 0.144.4`. `llmcut` now emits only that shape.

Capability evidence is bound to the exact runtime version and recorded, without canary values or
raw output, in `docs/evidence/v06-posttool-conformance.json`. PreToolUse command rewriting was not
probed or implemented because canonical PostToolUse replacement passed the selection rule. Hook
pilots remain separate from release statistics, and conformance input counts are not savings
evidence.

Direct `codex exec` conformance and the production parser probe passed, but the first bounded SDK
pilot observed zero hook events through the SDK-pinned App Server launch. All eight paired outcomes
passed quality, yet no apparent token difference is attributed to llmcut without activation. The
pilot stopped before hybrid and is retained as negative metadata in
`docs/evidence/v06-posttool-sdk-pilot.json`.

## Exec evaluation surface

Direct `codex exec` conformance proved exclusive PostToolUse replacement. SDK/App Server evaluation
did not activate hooks and remains ineligible for hook-savings measurements. Hook-capable evaluation
therefore uses `--backend exec`, a protected disposable hook definition, the runtime-proven
`decision:"block"` response, and the explicit one-off trust bypass.

Baseline exec runs disable hooks and MCP. Optimized runs reconcile completed Codex command digests
with metadata-only hook-event digests. Missing, duplicate, or mismatched activation invalidates a
comparison. Agent-input reductions are credited to hook compaction only when activation is observed
and execution, settings, repository, validation, and quality parity all pass.

The first isolated exec-backend probe on `codex-cli 0.146.0` completed with JSONL usage and command
events but recorded zero hook events across three bounded configuration attempts. The output pilot
was therefore not run. This is an activation/configuration blocker, not evidence against the earlier
exclusive-replacement conformance result and not a token-savings result.

The follow-up source matrix on `codex-cli 0.146.1` confirmed that project trust is part of the
activation boundary, but it did not validate the proposed invocation-only trust override as a
working isolated activation mechanism. Production evaluation therefore continues to fail closed
when an eligible command has no matching hook event. No output pilot or release suite was run.

## User-level bridge investigation

The next isolated candidate uses the documented user hook source at `$CODEX_HOME/hooks.json`.
Official Codex documentation says user hooks are independent of project trust, while
`--ignore-user-config` disables `$CODEX_HOME/config.toml` and leaves authentication discovery in
`CODEX_HOME`. The documentation does not explicitly guarantee whether the sibling `hooks.json`
source remains active under that flag, so llmcut treats this as a runtime conformance question.

The candidate definition is static and contains no repository, task, run, session, or evidence
identifier. `llmcut hook bridge` is inert without a protected lease. A lease uses a random ID plus a
separate secret token, binds mode, repository, cwd, revision, run, state, metrics, definition digest,
and expiry, and is stored under `0700`/`0600` state outside the repository. Observe mode records only
bounded metadata and emits no replacement; compact mode delegates to the existing exact-evidence
handler. User hook mutation is locked, journaled, atomic, preserves unrelated definitions, restores
exact original bytes when unchanged, and surgically removes only llmcut's definition after a safe
concurrent edit.

The bounded production bridge probe on `codex-cli 0.146.1` completed and emitted authoritative
usage, but recorded zero lease-bound hook events and no compaction. Cleanup restored the previously
absent user hooks file. Because the bridge is deliberately silent when its lease cannot be resolved,
the result does not distinguish hook-source non-execution from a launched bridge that did not
receive its activation variables. It is recorded in
`docs/evidence/v06-user-hook-production-probe.json`. OTel, resume, pilot, and release-suite work were
withheld at this stop condition. No token reduction is claimed.
