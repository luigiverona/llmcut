# User-level Codex hook bridge hypothesis

Status: unverified design note; not release evidence.

The next bounded conformance question is whether `$CODEX_HOME/hooks.json` remains an active
user-level hook source when `codex exec` uses `--ignore-user-config` in an untrusted disposable
repository. The proposed production path is conditional on that result:

```text
$CODEX_HOME/hooks.json
    -> user-level hook source independent of project trust
    -> static llmcut bridge definition
    -> protected run-scoped lease lookup
    -> no lease: inert success with no response, metrics, or evidence
    -> observe lease: bounded metadata only and no model-visible replacement
    -> compact lease: existing exact recoverable decision:block projection
```

The persistent definition must contain no repository, worktree, task, session, run, or evidence
identifier. Those bindings belong only in an expiring protected lease. Evaluation may use this
design only when the authenticated user-source matrix proves activation under user-config
isolation. If `$CODEX_HOME/hooks.json` is also suppressed by `--ignore-user-config`, implementation
stops rather than loading arbitrary user configuration or inventing an unsupported loader.
