from dataclasses import dataclass
from enum import StrEnum

from llmcut.errors import UnsupportedModeError


class OptimizationMode(StrEnum):
    STRICT = "strict"
    PARITY = "parity"
    EXTREME = "extreme"
    ECONOMY = "economy"


@dataclass(frozen=True, slots=True)
class Policy:
    mode: OptimizationMode = OptimizationMode.EXTREME
    quality_floor: str = "baseline"
    allow_lossy_context: bool = False
    allow_model_change: bool = False
    allow_reasoning_change: bool = False
    allow_validation_reduction: bool = False
    retain_original_context: bool = True
    fallback: str = "full_context"
    confidence_threshold: float = 0.8

    def validate(self) -> None:
        if self.mode is OptimizationMode.ECONOMY:
            raise UnsupportedModeError("economy routing is defined but not implemented in v0.1.0")
        if any(
            (
                self.allow_lossy_context,
                self.allow_model_change,
                self.allow_reasoning_change,
                self.allow_validation_reduction,
            )
        ):
            raise ValueError(
                "v0.1.0 quality invariants prohibit lossy or capability-reducing policy"
            )
