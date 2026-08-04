import json
from pathlib import Path

assert "ca-central-1" in Path("app/settings.py").read_text()
assert json.loads(Path("config.json").read_text())["region"] == "ca-central-1"
assert "ca-central-1" in Path("docs/deployment.md").read_text()
