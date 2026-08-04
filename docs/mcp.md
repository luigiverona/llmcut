# MCP server

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
