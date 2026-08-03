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
