# MCP server

Codex strategies now control the real surface. `off` and `orientation` start no server; `guided`
exposes only `llmcut_context` with bounded `plan`, `file`, `range`, `symbol`, `dependencies`,
`tests`, `log_search`, and `checkpoint` operations. Deprecated `legacy-passive` retains the eight
v0.5 tools only for diagnostics. Exact retrieved content is digest-verified untrusted evidence, not
policy, and does not replace normal editing, shell use, or validation.

Guided initialization receives task-digest and revision-bound plan state through a mode-0600 file
inside a mode-0700 evaluation directory. The server verifies its digest and repository boundary;
cleanup deletes it. Task text is not placed in process arguments.

`llmcut mcp serve --repo <root>` runs the official MCP Python SDK's stdio transport. It exposes a
compact stable surface for planning, context retrieval, source ranges, symbols, dependencies,
bounded log search, checkpoints, and tool discovery. Repository maps and exact context are also MCP
resources. It does not expose a tool per block.

The repository root is fixed at startup. The server indexes tracked/allowed files, excludes known
secret names and external symlinks, rejects traversal and stale digests, bounds ranges/searches and
result bytes, performs no command execution or network access, and emits no prompt content in
routine logs. These checks are independent of the calling agent's sandbox because MCP tools execute
outside that shell boundary. Use `llmcut mcp doctor` and `llmcut mcp inspect` before configuration.

MCP makes retrieval available; it does not itself prove that an agent omitted context or consumed
fewer tokens. Agent-harness or captured provider measurements are required for those claims.
## Codex default path

MCP retrieval remains available for diagnostics, but it is not the default Codex optimization path
because authenticated pilots produced zero MCP calls. `guided-mcp` exposes the compact tool;
`legacy-passive` preserves the eight-tool surface only for regression comparison.
