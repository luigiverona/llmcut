from __future__ import annotations


def canonical_principal(value: str) -> str:
    return value.strip().lower()


def may_access(requested: str, allowed: set[str]) -> bool:
    return canonical_principal(requested) in {canonical_principal(item) for item in allowed}
