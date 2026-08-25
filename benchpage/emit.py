"""Write summary.json / documents.json and maintain results/index.json."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import index_entry, validate_summary


def write_run(results_dir: str | Path, summary: dict, documents: list[dict],
              label: str | None = None) -> Path:
    problems = validate_summary(summary)
    if problems:
        raise ValueError("summary failed validation:\n  " + "\n  ".join(problems))

    results = Path(results_dir)
    run_id = summary["run"]["id"]
    run_dir = results / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    _dump(run_dir / "summary.json", summary)
    _dump(run_dir / "documents.json", documents)

    index_path = results / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else []
    index = [e for e in index if e.get("run_id") != run_id]
    index.append(index_entry(run_id, summary["run"]["started"], label))
    index.sort(key=lambda e: (e.get("started") or "", e["run_id"]), reverse=True)
    _dump(index_path, index)

    return run_dir


def _dump(path: Path, doc) -> None:
    path.write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
