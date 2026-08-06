import sys

sys.path.insert(0, ".")

from app.orders import discounted_total  # noqa: E402

assert discounted_total(2000, 2, 25) == 3000
