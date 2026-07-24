from __future__ import annotations

import re
from pathlib import Path


def test_just_python_entrypoints_exist() -> None:
    root = Path(__file__).resolve().parent.parent
    justfile = (root / "justfile").read_text()
    targets = re.findall(r"uv run ((?:scripts|training)/[^\s{]+\.py)", justfile)
    assert targets
    missing = [target for target in targets if not (root / target).is_file()]
    assert missing == []
