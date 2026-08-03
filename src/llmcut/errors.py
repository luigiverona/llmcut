class LlmcutError(Exception):
    """Base class for safe, actionable llmcut errors."""


class IntegrityError(LlmcutError):
    """Stored evidence or checkpoint failed integrity validation."""


class ConfigurationError(LlmcutError):
    """Configuration is absent, invalid, or unsafe."""


class UnsupportedModeError(ConfigurationError):
    """A declared but unavailable mode was selected."""


class ProtocolError(LlmcutError, ValueError):
    """A managed request violates the versioned protocol or policy."""


class RetrievalError(LlmcutError):
    """A managed retrieval operation was invalid or unsafe."""


class ExecutionError(LlmcutError):
    """A bounded managed execution could not complete safely."""
