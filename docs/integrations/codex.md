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
