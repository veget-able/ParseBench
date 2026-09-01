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

GROUPS = ("table", "chart", "layout", "text_content", "text_formatting")

# ParseBench's own per-category headline metric (see upstream
# analysis/aggregation_report.py::_DEFAULT_METRICS); categories without an
# entry there fall back to rule_pass_rate, exactly as upstream does.
HEADLINE_METRICS = {
    "table": "grits_trm_composite",
    "chart": "rule_pass_rate",
    "layout": "layout_element_rule_pass_rate",
    "text_content": "content_faithfulness",
    "text_formatting": "semantic_formatting",
}


def _read_json(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def load_run(pipeline_dir: str | Path) -> dict:
    """Load one official evaluation directory (single group, or one category
    subdirectory of a full run; the latter has no metadata/summary files)."""
    d = Path(pipeline_dir)
    meta = _read_json(d / "_metadata.json")
    summary = _read_json(d / "_summary.json")
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

    # A repetition whose evaluation covered fewer documents (a truncated
    # evaluation pass) is excluded from aggregation: its aggregates describe
    # a different subset. rep_doc_counts records what each repetition covered.
    counts = [len(r["docs"]) for r in runs]
    full = [r for r in runs if len(r["docs"]) == max(counts)]

    p50s = [_latency_stats(r).get("p50") for r in full]
    target = median([p for p in p50s if p is not None])
    reference = min(
        full,
        key=lambda r: abs((_latency_stats(r).get("p50") or 0) - (target or 0)),
    )
    gtrms = {r["report"]["aggregate_metrics"].get("avg_grits_trm_composite")
             for r in full}
    return {
        "reference": reference,
        "runs": full,
        "repetitions": len(runs),
        "rep_doc_counts": counts,
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
        "rep_doc_counts": combined["rep_doc_counts"],
    }


def load_full_run(pipeline_dir: str | Path) -> dict:
    """Load one full-dataset run (no group filter).

    ParseBench evaluates each category separately in that mode, writing one
    official report per category subdirectory.
    """
    d = Path(pipeline_dir)
    categories: dict = {}
    docs: list = []
    for g in GROUPS:
        if (d / g / "_evaluation_report.json").exists():
            r = load_run(d / g)
            categories[g] = r
            docs.extend(r["docs"])
    if not categories:
        raise FileNotFoundError(f"no category evaluations under {d}")
    return {"categories": categories, "docs": docs}


def combine_full_reps(runs: list[dict]) -> dict:
    """Combine repeated full-dataset runs; reference is the median-latency
    repetition among those with full document coverage."""
    if not runs:
        raise ValueError("combine_full_reps needs at least one run")
    counts = [len(r["docs"]) for r in runs]
    full = [r for r in runs if len(r["docs"]) == max(counts)]
    p50s = {id(r): median(_doc_latencies(r)) for r in full}
    target = median([p for p in p50s.values() if p is not None])
    reference = min(full, key=lambda r: abs((p50s[id(r)] or 0) - (target or 0)))
    return {
        "reference": reference,
        "runs": full,
        "repetitions": len(runs),
        "rep_doc_counts": counts,
    }


def _headline(metrics: dict, group: str) -> float | None:
    for key in (HEADLINE_METRICS.get(group), "rule_pass_rate"):
        if key and f"avg_{key}" in metrics:
            return metrics[f"avg_{key}"]
    return None


def group_quality_blocks(combined: dict, source: str) -> dict:
    """Per-category blocks plus Overall, from a full-dataset run.

    Category scores are ParseBench's own aggregates from each category's
    evaluation report, using upstream's headline metric per category;
    Overall follows the leaderboard's definition, the plain average across
    the five categories.
    """
    ref_cats = combined["reference"]["categories"]
    blocks: dict = {}
    for g in GROUPS:
        if g not in ref_cats:
            continue
        metrics = ref_cats[g]["report"].get("aggregate_metrics", {})
        score = _headline(metrics, g)
        if score is None:
            continue
        deterministic = len({
            _headline(r["categories"][g]["report"].get("aggregate_metrics", {}), g)
            for r in combined["runs"] if g in r["categories"]
        }) == 1
        block = {
            "source": source,
            "score": _round2(_scale100(score)),
            "metric": HEADLINE_METRICS[g],
            "docs_scored": ref_cats[g]["report"].get("successful"),
            "deterministic": deterministic,
            "rep_doc_counts": combined["rep_doc_counts"],
        }
        if g == "table":
            block["gtrm"] = block["score"]
            block["grits_con"] = _round2(_scale100(metrics.get("avg_grits_con")))
            block["table_record_match"] = _round2(
                _scale100(metrics.get("avg_table_record_match"))
            )
        blocks[g] = block

    if all(g in blocks for g in GROUPS):
        cats = {g: blocks[g]["score"] for g in GROUPS}
        blocks["overall"] = {
            "source": source,
            "score": _round2(sum(cats.values()) / len(cats)),
            "metric": "average_across_categories",
            "docs_scored": len(combined["reference"]["docs"]),
            "deterministic": all(blocks[g]["deterministic"] for g in GROUPS),
            "categories": cats,
        }
    return blocks


def _doc_latencies(run: dict) -> list[float]:
    return [
        d["latency_ms_per_page"] for d in run["docs"]
        if d["latency_ms_per_page"] is not None
    ]


def full_performance_block(combined: dict, source: str,
                           peak_rss_mb: float | None = None,
                           cold_start: dict | None = None) -> dict:
    """Latency pooled over all documents of a full-dataset run.

    Per-document values are ParseBench's own; pooling them into p50/p95/mean
    is ours, and across repetitions the median of each stat is taken.
    """
    def across(stat: str) -> float | None:
        vals = []
        for r in combined["runs"]:
            xs = _doc_latencies(r)
            if not xs:
                continue
            if stat == "p50":
                vals.append(median(xs))
            elif stat == "p95":
                vals.append(percentile(xs, 95))
            else:
                vals.append(sum(xs) / len(xs))
        ms = median(vals) if vals else None
        return _round4(ms / 1000.0) if ms is not None else None

    med = across("p50")
    return {
        "source": source,
        "s_per_page": {"median": med, "p95": across("p95"), "mean": across("mean")},
        "pages_per_min": _round2(60.0 / med) if med else None,
        "cold_start_s": cold_start,
        "peak_rss_mb": _round2(peak_rss_mb) if peak_rss_mb is not None else None,
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
