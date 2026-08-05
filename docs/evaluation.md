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
