# Trustworthy evaluation

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

Measurement layers are never conflated: `payload` describes model-bound requests llmcut generated;
`provider` requires bound API usage or an official count endpoint; `agent` requires supported harness
events; `subscription` is reported only if the subscription system provides a reliable metric.
