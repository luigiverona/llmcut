import sys

sys.path.insert(0, ".")

from app.schema import DEFAULT_POLICY, RetryPolicy  # noqa: E402

assert isinstance(DEFAULT_POLICY, RetryPolicy)
