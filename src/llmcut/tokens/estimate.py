import math

from llmcut.model import CountQuality, TokenCount


class ConservativeEstimator:
    """UTF-8 byte heuristic; conservative for ordinary Latin text, explicitly estimated."""

    def count(self, text: str, *, model: str | None = None) -> TokenCount:
        del model
        return TokenCount(
            max(1, math.ceil(len(text.encode("utf-8")) / 3)), CountQuality.ESTIMATED, "utf8-bytes/3"
        )
