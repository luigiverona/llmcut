import sys

sys.path.insert(0, ".")

from app.permissions import may_access  # noqa: E402

assert may_access("Straße@example.test", {"STRASSE@example.test"})
