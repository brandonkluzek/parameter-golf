from __future__ import annotations

import json
from pathlib import Path

from . import RUNS_DIR


def rebuild_index(runs_dir: Path = RUNS_DIR) -> Path:
    runs_dir.mkdir(parents=True, exist_ok=True)
    index_path = runs_dir / "index.jsonl"
    seen: set[str] = set()
    lines: list[str] = []
    for summary_path in sorted(runs_dir.glob("*/summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        run_id = str(summary.get("run_id", summary_path.parent.name))
        if run_id in seen:
            continue
        seen.add(run_id)
        lines.append(json.dumps(summary, sort_keys=True))
    index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return index_path
