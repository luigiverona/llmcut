import sys

sys.path.insert(0, ".")

from app.worker import attempts  # noqa: E402

assert attempts() == 3
