from pathlib import Path

source = Path("src/timeout.ts").read_text()
assert "timeoutSeconds * 1000" in source
