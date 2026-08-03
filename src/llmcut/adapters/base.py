from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from llmcut.model import CanonicalRequest


class ProviderAdapter(ABC):
    @abstractmethod
    def from_native(self, payload: dict[str, Any]) -> CanonicalRequest: ...

    @abstractmethod
    def to_native(self, request: CanonicalRequest) -> dict[str, Any]: ...

    @abstractmethod
    def usage(self, response: dict[str, Any]) -> dict[str, int]: ...

    def validate_native(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload.get("model"), str) or not payload["model"]:
            raise ValueError("request model must be a non-empty string")

    def semantically_equal(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_copy, right_copy = dict(left), dict(right)
        for key in ("tools",):
            if key not in left_copy and right_copy.get(key) == []:
                right_copy.pop(key)
            if key not in right_copy and left_copy.get(key) == []:
                left_copy.pop(key)
        return left_copy == right_copy
