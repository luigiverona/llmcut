from __future__ import annotations

from dataclasses import dataclass

from llmcut.errors import RetrievalError
from llmcut.managed.protocol import ToolDefinition


@dataclass(slots=True)
class ToolRegistry:
    """Immutable-name canonical registry used before adapter-native generation."""

    tools: tuple[ToolDefinition, ...]

    def __post_init__(self) -> None:
        names = [item.name for item in self.tools]
        if len(names) != len(set(names)):
            raise ValueError("canonical tool registry names must be unique")

    def categories(self) -> tuple[str, ...]:
        return tuple(sorted({item.category for item in self.tools}))

    def discover(self, query: str, *, limit: int = 20) -> tuple[ToolDefinition, ...]:
        query = query.casefold()
        return tuple(
            item
            for item in self.tools
            if query in item.name.casefold()
            or query in item.category.casefold()
            or query in item.description.casefold()
        )[: min(max(limit, 1), 100)]

    def load(self, name: str) -> ToolDefinition:
        for item in self.tools:
            if item.name == name:
                return item
        raise RetrievalError(f"tool is unavailable: {name}")
