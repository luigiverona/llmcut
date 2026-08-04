import sys

sys.path.insert(0, ".")

from app.parser import parse_status  # noqa: E402

assert parse_status(" READY ") == "ready"
