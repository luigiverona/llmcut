import sys

sys.path.insert(0, ".")

from app.callback import callback_timeout  # noqa: E402

assert callback_timeout() == 30
