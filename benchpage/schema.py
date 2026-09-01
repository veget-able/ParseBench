"""Result schema for the public PyMuPDF benchmark page.

Three layers, all JSON, all static-hostable:

    results/index.json                   -> run manifest (history)
    results/runs/<run_id>/summary.json   -> everything the page needs on first load
    results/runs/<run_id>/documents.json -> per-document detail, loaded lazily

Design rules:

* The frontend formats numbers, it never computes them. Medians, percentiles
  and ratios are precomputed here.
* Quality scores are stored on the 0-100 scale used by the ParseBench
  leaderboard (the raw CSVs are 0-1; ``collect`` converts).
* Every metric block carries ``"source": "measured" | "placeholder"`` so
  mockups and live data share one schema.
* Objects are keyed by stable pipeline ids so an embed can slice out a
  subset (e.g. PyMuPDF vs Docling) without re-aggregating.
"""

from __future__ import annotations

import math

SCHEMA_VERSION = 1

SOURCE_MEASURED = "measured"
SOURCE_PLACEHOLDER = "placeholder"

CATEGORY_LOCAL = "local"
CATEGORY_CLOUD = "cloud"


def summary_skeleton(run_id: str) -> dict:
    """Empty summary document; the runner fills the sections in."""
    return {
        "schema_version": SCHEMA_VERSION,
        "run": {
            "id": run_id,
            "started": None,          # ISO 8601 UTC
            "parsebench_commit": None,
            "dataset": {"group": None, "docs": None},
            "llm_normalization": None,
            "repetitions": None,
        },
        "env": {
            "instance": None,         # e.g. "c7i.xlarge"; null on a dev box
            "cpu": None,
            "ram_gb": None,
            "os": None,
            "python": None,
        },
        # pipeline_id -> {label, category, versions, fee_per_1k_pages, lock_sha256}
        "pipelines": {},
        # group -> pipeline_id -> {source, gtrm, grits_con, table_record_match,
        #                          docs_scored, deterministic}
        "quality": {},
        # pipeline_id -> {source, s_per_page: {median, p95, mean}, pages_per_min,
        #                 cold_start_s: {import, first_doc}, peak_rss_mb,
        #                 ratio_vs_baseline}
        "performance": {},
        # pipeline_id -> {source, install_mb, download_mb, dep_count, install_s}
        "footprint": {},
    }


def index_entry(run_id: str, started: str, label: str | None = None) -> dict:
    return {
        "run_id": run_id,
        "started": started,
        "label": label,
        "files": {
            "summary": f"runs/{run_id}/summary.json",
            "documents": f"runs/{run_id}/documents.json",
        },
    }


def percentile(values: list[float], p: float) -> float | None:
    """Nearest-rank-with-interpolation percentile; None on empty input."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    k = (len(xs) - 1) * (p / 100.0)
    lo = math.floor(k)
    hi = math.ceil(k)
    if lo == hi:
        return xs[lo]
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def median(values: list[float]) -> float | None:
    return percentile(values, 50)


def validate_summary(doc: dict) -> list[str]:
    """Return a list of problems; empty list means the document is valid."""
    problems: list[str] = []
    if doc.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version must be {SCHEMA_VERSION}")
    for key in ("run", "env", "pipelines", "quality", "performance", "footprint"):
        if key not in doc:
            problems.append(f"missing top-level key: {key}")
    if problems:
        return problems

    known = set(doc["pipelines"])
    for pid, spec in doc["pipelines"].items():
        for k in ("label", "category"):
            if not spec.get(k):
                problems.append(f"pipelines.{pid}: missing {k}")
        if spec.get("category") not in (CATEGORY_LOCAL, CATEGORY_CLOUD):
            problems.append(f"pipelines.{pid}: bad category {spec.get('category')!r}")

    for group, per_pipeline in doc["quality"].items():
        for pid, block in per_pipeline.items():
            if pid not in known:
                problems.append(f"quality.{group}.{pid}: unknown pipeline id")
                continue
            if block.get("source") not in (SOURCE_MEASURED, SOURCE_PLACEHOLDER):
                problems.append(f"quality.{group}.{pid}: bad source")
            for field in ("gtrm", "score"):
                v = block.get(field)
                if v is not None and not (0.0 <= v <= 100.0):
                    problems.append(f"quality.{group}.{pid}: {field} {v} outside 0-100")

    for section in ("performance", "footprint"):
        for pid, block in doc[section].items():
            if pid not in known:
                problems.append(f"{section}.{pid}: unknown pipeline id")
            elif block.get("source") not in (SOURCE_MEASURED, SOURCE_PLACEHOLDER):
                problems.append(f"{section}.{pid}: bad source")

    return problems
