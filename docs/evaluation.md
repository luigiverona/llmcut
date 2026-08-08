# Trustworthy evaluation

Version 0.6 separates quota-conscious pilots from release statistics. Pilots compare `off`,
`orientation`, `guided`, and `legacy-passive` and are always labeled release-ineligible. A release
suite requires at least four non-control tasks, three repetitions per mode, median paired
agent-input reduction of 5%, 60% positive pairs, no task median below -5%, safe no-benefit behavior,
quality parity, a 70% guided-schema reduction, and observed intervention. Estimated component costs
are diagnostics and are never subtracted from agent-reported totals.

Version 0.3's recorded responses were deterministic runtime fixtures, not independent measurement:
their editable `usage` fields could affect reported savings and their principal quality assertions
were short response matches. Version 0.4 retains those fixtures only as `untrusted_fixture` parser
and replay evidence.

The release suite materializes independent repositories, creates baseline and optimized worktrees
from the same local Git revision, verifies identical starting files, applies the same accepted
behavioral change, executes the same validation command, and rejects unrelated file changes. Every
input count comes from the exact deterministic provider-native serialization and carries provider,
model, request digest, quality, source, counter version, timestamp, trust, and measurement layer.
Retrieval and continuation payloads are separately serialized and counted.

`tests/fixtures/benchmarks/suite.toml` covers Python, TypeScript, coordinated configuration, a large
test failure, long coding history, zero-savings, and retrieval-heavy tasks. Statistics include all
eligible cases, exclusions, positive, zero, and negative cases, p25/p75, all-case median, and
positive-case median. No LLM judge is required. The local conservative byte counter is explicitly an
estimate; authenticated provider and live coding-agent evaluation remain optional.

## Codex measurement scope

The v0.4 deterministic benchmark compared full model-bound repository context with managed
selection. It measured planner compression, not ordinary Codex usage. Beginning with v0.5, the
standard live comparison gives baseline and optimized Codex the same ordinary task prompt and the
same repository access. Only the optimized run enables llmcut MCP and managed retrieval.

The report separates the exact user-task payload, llmcut MCP traffic, any observable agent payload,
agent-reported usage, provider-reported usage, and subscription usage. Task text plus MCP bytes is
not described as complete Codex input. Agent usage is bound to run/thread/turn identifiers and is
used only when the supported SDK emits it. Subscription usage remains unavailable unless Codex
explicitly exposes such a metric.

Live suites use at least three repetitions per mode and seeded ordering. Reports retain every
paired difference, min/max, quartiles, quality outcome, duration, correction turn, MCP call, and
retrieval call. Three repetitions do not establish statistical significance; results describe only
the observed bounded suite.

Measurement layers are never conflated: `payload` describes model-bound requests llmcut generated;
`provider` requires bound API usage or an official count endpoint; `agent` requires supported harness
events; `subscription` is reported only if the subscription system provides a reliable metric.
## First v0.6 pilot and hook measurement

The first v0.6 adaptive-routing pilot is retained as metadata-only negative evidence in
`docs/evidence/v06-first-pilot.json` and is excluded from later release statistics. All 24 outcomes
passed; orientation measured +2.23%, guided MCP -0.26%, and MCP adoption was zero. Hook pilots and
release A/B runs remain separate. Component compaction estimates include recovery overhead, while
agent-reported total input remains the release authority.

The PostToolUse conformance matrix is metadata-only protocol evidence, not a task pilot.
Original-only head, middle, and tail canaries were absent and the compact-only canary was present
for `decision:"block"`; additive and `continue:false` variants did not meet that rule. The selected
shape was repeated twice before adoption.

The later SDK pilot stopped at the activation gate: 8/8 outcomes passed, but the App Server-backed
SDK runs recorded no hook events. Its apparent +7.15% representative median is explicitly not a
compaction result and is excluded from release statistics. No hybrid or full-suite run followed.

Hook evaluation now uses `codex exec --json`. A valid run requires a zero process status, exactly one
completed terminal event, valid `turn.completed.usage`, no fatal top-level error, and reconciled hook
metrics. Unknown additive events retain bounded key metadata only; reasoning, agent messages,
command output, prompts, and environment values are not persisted.

The installed JSONL contract does not expose a resolved model. A suite requiring resolved-model
observation is rejected before quota use until a safe, version-bound observation path is available.

The initial authenticated exec-backend probe observed terminal usage and command events but zero
hook events across three bounded attempts. It failed the activation gate, so no output-compaction
pilot or release suite followed and no reduction is claimed.

A later user-source probe tested the static `$CODEX_HOME/hooks.json` bridge with user config ignored,
hooks enabled, the explicit trust bypass, and a protected compact lease. The Codex turn completed
with authoritative usage, but no lease-bound event was recorded. Cleanup restored the absent hooks
file. This probe is metadata-only conformance evidence, is excluded from release statistics, and
stopped OTel, continuation, pilot, and release-suite execution.
