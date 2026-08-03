class LlmcutError(Exception):
    """Base class for safe, actionable llmcut errors."""


class IntegrityError(LlmcutError):
    """Stored evidence or checkpoint failed integrity validation."""


class ConfigurationError(LlmcutError):
    """Configuration is absent, invalid, or unsafe."""


class UnsupportedModeError(ConfigurationError):
    """A declared but unavailable mode was selected."""
