import json

from llmcut.model import BlockKind, CanonicalRequest, digest_bytes


def stable_partition(request: CanonicalRequest) -> tuple[list[str], str]:
    order = ["policy", "tools", "repository", "task", "dynamic"]
    groups: dict[str, list[dict[str, object]]] = {key: [] for key in order}
    for block in [*request.blocks, *request.tools]:
        if block.kind in {BlockKind.SYSTEM, BlockKind.DEVELOPER}:
            group = "policy"
        elif block.kind is BlockKind.TOOL_DEFINITION:
            group = "tools"
        elif block.kind is BlockKind.REPOSITORY:
            group = "repository"
        elif block.kind is BlockKind.USER:
            group = "task"
        else:
            group = "dynamic"
        groups[group].append({"id": block.id, "kind": block.kind.value, "content": block.content})
    serialized = [json.dumps(groups[key], sort_keys=True, separators=(",", ":")) for key in order]
    stable = "".join(serialized[:3]).encode()
    return serialized, digest_bytes(stable)
