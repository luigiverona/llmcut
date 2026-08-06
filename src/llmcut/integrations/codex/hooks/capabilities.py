from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class HookCapabilities:
    runtime_version: str
    post_replacement: str
    pre_rewrite: str
    probe_timestamp: str | None
    probe_digest: str | None

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


_POST_REPLACEMENT = {
    "codex-cli 0.146.0": (
        "2026-08-06T01:56:27Z",
        "sha256:bb58914951eef6c3773d4baccb651473cfb678e8776b637daeae03799a7609c9",
    ),
    "codex-cli 0.144.4": (
        "2026-08-06T01:56:27Z",
        "sha256:bb58914951eef6c3773d4baccb651473cfb678e8776b637daeae03799a7609c9",
    ),
}


def capabilities_for(runtime_version: str) -> HookCapabilities:
    evidence = _POST_REPLACEMENT.get(runtime_version)
    return HookCapabilities(
        runtime_version=runtime_version,
        post_replacement="supported" if evidence else "unverified",
        pre_rewrite="unverified",
        probe_timestamp=evidence[0] if evidence else None,
        probe_digest=evidence[1] if evidence else None,
    )
