"""Read ParseBench run artifacts into page-schema blocks.

Fairness rule: every aggregate is lifted verbatim from ParseBench's own
``_evaluation_report.json`` (``aggregate_metrics`` for quality,
``aggregate_stats.latency_ms_per_page`` for latency); this module performs
no re-aggregation of its own. The only cross-run arithmetic is the median
across repetitions, and per-document rows are the unmodified rows of the
median repetition.

A ParseBench run writes, per pipeline, under ``<output_dir>/<pipeline>/``:

    _summary.json             totals, success rate, overall latency
    _metadata.json            pipeline spec + run config (max_concurrent, ...)
    _evaluation_report.json   official aggregates + per-example results
    _evaluation_results.csv   one row per document (latency, GTRM, ...)
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .schema import median

GTRM_COLUMN = "grits_trm_composite"  # ParseBench's field name for GTRM


def load_run(pipeline_dir: str | Path) -> dict:
    """Load one pipeline's artifacts from one ParseBench run directory."""
    d = Path(pipeline_dir)
    meta = json.loads((d / "_metadata.json").read_text(encoding="utf-8"))
    summary = json.loads((d / "_summary.json").read_text(encoding="utf-8"))
    report = json.loads((d / "_evaluation_report.json").read_text(encoding="utf-8"))

    docs = []
    with (d / "_evaluation_results.csv").open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            docs.append(
                {
                    "doc": row["test_id"],
                    "tags": row.get("tags", ""),
                    "success": row.get("success", "").strip().lower() == "true",
                    "latency_ms_per_page": _float(
                        row.get("latency_ms_per_page") or row.get("latency_ms")
                    ),
                    "gtrm": _scale100(_float(row.get(GTRM_COLUMN))),
                }
            )
    return {"meta": meta, "summary": summary, "report": report, "docs": docs}


def combine_reps(runs: list[dict]) -> dict:
    """Combine repeated runs of the same pipeline.

    The reference repetition is the one whose official latency p50 is the
    median across repetitions; per-document rows come from it unchanged.
    Quality aggregates are expected to be identical in every repetition
    (the stack is deterministic); ``deterministic`` records whether that
    actually held.
    """
    if not runs:
        raise ValueError("combine_reps needs at least one run")

    p50s = [_latency_stats(r)["p50"] for r in runs]
    target = median([p for p in p50s if p is not None])
    reference = min(
        runs,
        key=lambda r: abs((_latency_stats(r)["p50"] or 0) - (target or 0)),
    )
    gtrms = {r["report"]["aggregate_metrics"].get("avg_grits_trm_composite")
             for r in runs}
    return {
        "reference": reference,
        "runs": runs,
        "repetitions": len(runs),
        "deterministic": len(gtrms) == 1,
    }


def quality_block(combined: dict, source: str) -> dict:
    """Official quality aggregates of the reference repetition, rescaled 0-100."""
    report = combined["reference"]["report"]
    metrics = report["aggregate_metrics"]
    return {
        "source": source,
        "gtrm": _round2(_scale100(metrics.get("avg_grits_trm_composite"))),
        "grits_con": _round2(_scale100(metrics.get("avg_grits_con"))),
        "table_record_match": _round2(_scale100(metrics.get("avg_table_record_match"))),
        "docs_scored": report.get("successful"),
        "deterministic": combined["deterministic"],
    }


def performance_block(combined: dict, source: str, peak_rss_mb: float | None = None,
                      cold_start: dict | None = None) -> dict:
    """Official latency stats; across repetitions, the median of each stat."""

    def across_reps(stat: str) -> float | None:
        vals = [_latency_stats(r).get(stat) for r in combined["runs"]]
        ms = median([v for v in vals if v is not None])
        return _round4(ms / 1000.0) if ms is not None else None

    med = across_reps("p50")
    return {
        "source": source,
        "s_per_page": {
            "median": med,
            "p95": across_reps("p95"),
            "mean": across_reps("avg"),
        },
        "pages_per_min": _round2(60.0 / med) if med else None,
        "cold_start_s": cold_start,
        "peak_rss_mb": _round2(peak_rss_mb) if peak_rss_mb is not None else None,
    }


def document_rows(per_pipeline: dict[str, dict]) -> list[dict]:
    """Join reference-repetition rows into documents.json rows."""
    all_docs: dict[str, dict] = {}
    for pid, combined in per_pipeline.items():
        for d in combined["reference"]["docs"]:
            row = all_docs.setdefault(
                d["doc"], {"doc": d["doc"], "tags": d["tags"], "pipelines": {}}
            )
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


def _latency_stats(run: dict) -> dict:
    # Some providers (docling_serve among them) report no page counts, so the
    # per-page stat is absent; latency_ms is equivalent for the single-page
    # documents of the table group.
    stats = run["report"].get("aggregate_stats", {})
    return stats.get("latency_ms_per_page") or stats.get("latency_ms") or {}


def _float(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _scale100(v: float | None) -> float | None:
    return v * 100.0 if v is not None else None


def _round2(v: float | None) -> float | None:
    return round(v, 2) if v is not None else None


def _round4(v: float | None) -> float | None:
    return round(v, 4) if v is not None else None
