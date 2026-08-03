# Architecture

## Canonical model and pipeline

`CanonicalRequest` owns ordered `ContextBlock` instances and separate tool definitions. Blocks carry
kind/role, byte-exact content, source, SHA-256 digest, priority, dependencies, arbitrary metadata,
recoverable evidence reference, and labeled token count. Provider-specific unknown fields live in a
round-tripped passthrough map; adapters do not influence selection policy.

The pipeline is: validate policy, store the original request, remove only byte-identical blocks of
the same semantic kind, store each unique block, make deterministic selection decisions, fail open
below the confidence threshold, partition stable/dynamic content, label counts, and emit an
optimization report plus evidence manifest. Restoration loads and verifies the original digest.
Expansion is monotonic: evidence, source ranges, matches, and dependencies are added from verified
references. An explicit checkpoint may supersede prior in-context prose, but never its evidence.

## Evidence lifecycle and checkpoints

SQLite transactions atomically store content and metadata. SHA-256 identity is verified on every
read and collision checks prevent unrelated overwrite. Checkpoint references are foreign-keyed and
protected from garbage collection. Checkpoint load verifies all evidence and can compare the exact
Git revision, rejecting stale state. Retention deletes only expired, unreferenced evidence.

Command output keeps complete bounded-at-ingestion raw data and exposes failures, warnings, skipped
tests, locations, summary, and retrieval reference. Range, exact, regex, and surrounding-context
retrieval do not execute commands.

## Repository selection

Git tracked files define default scope. Explicit untracked indexing uses Git's ignore rules. Secret
names and state directories are excluded; symlinks are not followed. Python uses `ast` for exact
symbol ranges and imports. JavaScript/TypeScript use maintained Tree-sitter grammars with generic
full-file fallback. Clean tracked records are reused across commits by Git blob identity and parser
version; working-tree changes, deletions, and renames invalidate transactionally. Scores use task
terms, paths, changes, imports, symbols, tests, instructions, and configuration.

## Adapters, caching, and request flow

OpenAI Chat/Responses, Anthropic Messages, and Gemini generateContent adapters translate canonical
requests and usage. Provider-specific tool IDs and native content are retained in block metadata.
Every supported proxy request proves an unchanged native round trip, then optimizes and validates
protected fields and ordered content again. Full reconstructed native size—not omitted-block count—
selects the candidate. Invalid, unsafe, or larger candidates replay the original request bytes.
Stable serialization partitions policy, tools, repository, task, and dynamic content. Potential
cacheability is not reported as actual cache use or logical reduction.

Proxy flow is: resolve named configured provider, bound body, reject origin override, filter headers,
inject credentials from the configured environment variable, apply timeout, and stream or return the
upstream response. Streaming is backpressure-driven and closed on completion/cancellation.
Transparent mode adds no tools. Managed integrations can explicitly obtain retrieval schemas, but
must dispatch them through the recovery API themselves.

## Evaluation trade-offs

Baseline and optimized execution share an executor and immutable provider/model/settings. Results
track completeness, deterministic invariants, tokens, caching, recovery, retries, latency, and
regression. A text heuristic can efficiently identify candidates, but never proves irrelevance;
therefore the engine includes uncertain content and preserves every exclusion by reference.
