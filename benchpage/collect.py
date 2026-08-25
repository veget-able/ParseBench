"""Read ParseBench run artifacts into page-schema aggregates.

A ParseBench run writes, per pipeline, under ``<output_dir>/<pipeline>/``:

    _summary.json             totals, success rate, overall latency
    _metadata.json            pipeline spec + run config (max_concurrent, ...)
    _evaluation_results.csv   one row per document:
        test_id, tags, success, latency_ms, latency_ms_per_page,
        grits_con, grits_trm_composite (= GTRM), table_record_match, ...

This module only reads those files; it never touches ParseBench internals,
so upstream refactors surface as loud load errors rather than silent drift.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import median, percentile

GTRM_COLUMN = "grits_trm_composite"  # ParseBench's field name for GTRM


def load_run(pipeline_dir: str | Path) -> dict:
    """Load one pipeline's artifacts from one ParseBench run directory."""
    d = Path(pipeline_dir)
    meta = json.loads((d / "_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "_summary.json").read_text(encoding="utf-8"))

    docs = []
    with (d / "_evaluation_results.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            docs.append(
                {
                    "doc": row["test_id"],
                    "tags": row.get("tags", ""),
                    "success": row.get("success", "").strip().lower() == "true",
                    "latency_ms": _float(row.get("latency_ms")),
                    "latency_ms_per_page": _float(row.get("latency_ms_per_page")),
                    "gtrm": _scale100(_float(row.get(GTRM_COLUMN))),
                    "grits_con": _scale100(_float(row.get("grits_con"))),
                    "table_record_match": _scale100(_float(row.get("table_record_match"))),
                }
            )
    return {"meta": meta, "summary": summary, "docs": docs}


def merge_reps(runs: list[dict]) -> dict:
    """Merge repeated runs of the same pipeline.

    Latency is taken as the per-document median across repetitions. Quality
    is expected to be identical in every repetition (the stack is
    deterministic); ``deterministic`` records whether that actually held.
    """
    if not runs:
        raise ValueError("merge_reps needs at least one run")

    by_doc: dict[str, list[dict]] = {}
    for run in runs:
        for doc in run["docs"]:
            by_doc.setdefault(doc["doc"], []).append(doc)

    deterministic = True
    merged_docs = []
    for doc_id, rows in sorted(by_doc.items()):
        gtrms = {r["gtrm"] for r in rows}
        if len(gtrms) > 1:
            deterministic = False
        base = dict(rows[0])
        base["latency_ms"] = median([r["latency_ms"] for r in rows if r["latency_ms"] is not None])
        base["latency_ms_per_page"] = median(
            [r["latency_ms_per_page"] for r in rows if r["latency_ms_per_page"] is not None]
        )
        merged_docs.append(base)

    return {
        "meta": runs[0]["meta"],
        "summary": runs[0]["summary"],
        "docs": merged_docs,
        "repetitions": len(runs),
        "deterministic": deterministic,
    }


def quality_block(merged: dict, source: str) -> dict:
    docs = [d for d in merged["docs"] if d["success"]]
    return {
        "source": source,
        "gtrm": _round2(_mean([d["gtrm"] for d in docs])),
        "grits_con": _round2(_mean([d["grits_con"] for d in docs])),
        "table_record_match": _round2(_mean([d["table_record_match"] for d in docs])),
        "docs_scored": len(docs),
        "deterministic": merged.get("deterministic"),
    }


def performance_block(merged: dict, source: str, peak_rss_mb: float | None = None,
                      cold_start: dict | None = None) -> dict:
    spp = [
        d["latency_ms_per_page"] / 1000.0
        for d in merged["docs"]
        if d["success"] and d["latency_ms_per_page"] is not None
    ]
    med = median(spp)
    return {
        "source": source,
        "s_per_page": {
            "median": _round4(med),
            "p95": _round4(percentile(spp, 95)),
            "mean": _round4(_mean(spp)),
        },
        "pages_per_min": _round2(60.0 / med) if med else None,
        "cold_start_s": cold_start,
        "peak_rss_mb": _round2(peak_rss_mb) if peak_rss_mb is not None else None,
    }


def document_rows(per_pipeline: dict[str, dict]) -> list[dict]:
    """Join per-pipeline merged runs into documents.json rows."""
    all_docs: dict[str, dict] = {}
    for pid, merged in per_pipeline.items():
        for d in merged["docs"]:
            row = all_docs.setdefault(d["doc"], {"doc": d["doc"], "tags": d["tags"], "pipelines": {}})
            row["pipelines"][pid] = {
                "gtrm": _round2(d["gtrm"]),
                "s_per_page": _round4(
                    d["latency_ms_per_page"] / 1000.0
                    if d["latency_ms_per_page"] is not None
                    else None
                ),
                "success": d["success"],
            }
    return [all_docs[k] for k in sorted(all_docs)]


def _float(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _scale100(v: float | None) -> float | None:
    return v * 100.0 if v is not None else None


def _mean(values: list) -> float | None:
    xs = [v for v in values if v is not None]
    return sum(xs) / len(xs) if xs else None


def _round2(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


def _round4(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None
