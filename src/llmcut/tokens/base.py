from typing import Protocol

from llmcut.model import TokenCount


class TokenCounter(Protocol):
    def count(self, text: str, *, model: str | None = None) -> TokenCount: ...
