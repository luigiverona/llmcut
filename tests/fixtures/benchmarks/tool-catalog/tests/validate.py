import sys

sys.path.insert(0, ".")

from app.formatter import issue_key  # noqa: E402

assert issue_key("CUT", 42) == "CUT-42"
