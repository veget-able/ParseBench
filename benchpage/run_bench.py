"""Benchmark orchestrator for the public PyMuPDF benchmark page.

Wraps the stock ``parse-bench`` CLI (never patches it) and produces the
page's result JSON. One invocation is one "run" in results/index.json.

Measurement rules baked in:

* ``--max_concurrent 1`` always; the sequential path is the only one whose
  per-document latency is meaningful.
* Repetitions are interleaved A/B/A/B across pipelines inside the same
  session, so machine noise cancels in the ratio even off the pinned runner.
* Aggregates are ParseBench's own (_evaluation_report.json); across
  repetitions the median of each official stat is taken, per-document rows
  come from the median repetition unchanged, and quality must be identical
  across repetitions (the summary records whether it was).

Usage (from the repo root, config listing the pipelines and their venvs):

    python -m benchpage.run_bench --config benchpage/pipelines.example.json \
        --group table --reps 3 --results-dir results --label weekly
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import threading
import time
from pathlib import Path

from . import collect, sysinfo
from .emit import write_run
from .schema import SOURCE_MEASURED, summary_skeleton

MAX_CONCURRENT = 1  # hard rule; see module docstring


def run(config_path: str, group: str, reps: int, results_dir: str,
        label: str | None, work_dir: str, run_id: str | None,
        with_coldstart: bool, with_footprint: bool) -> Path:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    pipelines = config["pipelines"]
    baseline_id = config.get("baseline")

    started = sysinfo.utc_now_iso()
    rid = run_id or time.strftime("%Y%m%d-%H%M%S")
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)

    peak_rss: dict[str, float] = {}
    for rep in range(reps):
        for spec in pipelines:  # interleaved on purpose; see module docstring
            out_dir = work / f"rep{rep}" / spec["id"]
            rss = _run_parse_bench(spec, group, out_dir)
            if rss is not None:
                peak_rss[spec["id"]] = max(peak_rss.get(spec["id"], 0.0), rss)

    summary = summary_skeleton(rid)
    summary["run"].update(
        started=started,
        parsebench_commit=sysinfo.git_commit("."),
        dataset={"group": group, "docs": None},
        llm_normalization=config.get("llm_normalization", False),
        repetitions=reps,
    )
    summary["env"] = sysinfo.collect()

    merged_by_pipeline: dict[str, dict] = {}
    for spec in pipelines:
        pid = spec["id"]
        runs = [
            collect.load_run(work / f"rep{rep}" / pid / spec["parse_bench_pipeline"])
            for rep in range(reps)
        ]
        merged = collect.combine_reps(runs)
        merged_by_pipeline[pid] = merged
        summary["run"]["dataset"]["docs"] = len(merged["reference"]["docs"])

        summary["pipelines"][pid] = {
            "label": spec["label"],
            "category": spec["category"],
            "versions": spec.get("versions", {}),
            "fee_per_1k_pages": spec.get("fee_per_1k_pages"),
            "lock_sha256": sysinfo.venv_lock_sha256(spec["python"]),
        }
        summary["quality"].setdefault(group, {})[pid] = collect.quality_block(
            merged, SOURCE_MEASURED
        )

        cold = None
        if with_coldstart and "coldstart" in spec:
            from . import coldstart

            cs = spec["coldstart"]
            cold = coldstart.measure(spec["python"], cs["import_stmt"],
                                     cs["first_doc_expr"], cs["doc"])
        summary["performance"][pid] = collect.performance_block(
            merged, SOURCE_MEASURED, peak_rss.get(pid), cold
        )

        if with_footprint and "footprint_packages" in spec:
            from . import footprint

            fp = footprint.measure(spec["footprint_packages"],
                                   extra_index_url=spec.get("footprint_extra_index_url"))
            summary["footprint"][pid] = {"source": SOURCE_MEASURED, **fp}

    _add_ratios(summary["performance"], baseline_id)
    documents = collect.document_rows(merged_by_pipeline)
    return write_run(results_dir, summary, documents, label)


def _run_parse_bench(spec: dict, group: str, out_dir: Path) -> float | None:
    """Run one pipeline once; return peak RSS in MB (None without psutil)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = _parse_bench_cmd(spec) + [
        "run", spec["parse_bench_pipeline"],
        "--group", group,
        "--max_concurrent", str(MAX_CONCURRENT),
        "--open_report", "False",
        "--output_dir", str(out_dir),
    ]
    proc = subprocess.Popen(cmd, env={**os.environ, **spec.get("env", {})})
    peak = _watch_rss(proc)
    if proc.wait() != 0:
        raise RuntimeError(f"parse-bench failed for {spec['id']} (exit {proc.returncode})")
    return peak


def _parse_bench_cmd(spec: dict) -> list[str]:
    if "parse_bench_cmd" in spec:
        return list(spec["parse_bench_cmd"])
    py = Path(spec["python"])
    for name in ("parse-bench.exe", "parse-bench"):
        candidate = py.parent / name
        if candidate.exists():
            return [str(candidate)]
    return ["parse-bench"]


def _watch_rss(proc: subprocess.Popen, interval_s: float = 0.5) -> float | None:
    try:
        import psutil
    except ImportError:
        return None

    peak = 0.0

    def sample() -> None:
        nonlocal peak
        try:
            root = psutil.Process(proc.pid)
            while proc.poll() is None:
                procs = [root] + root.children(recursive=True)
                rss = sum(p.memory_info().rss for p in procs
                          if _alive(p)) / 1024**2
                peak = max(peak, rss)
                time.sleep(interval_s)
        except psutil.NoSuchProcess:
            pass

    t = threading.Thread(target=sample, daemon=True)
    t.start()
    proc.wait()
    t.join(timeout=2)
    return round(peak, 1) if peak else None


def _alive(p) -> bool:
    try:
        return p.is_running()
    except Exception:
        return False


def _add_ratios(performance: dict, baseline_id: str | None) -> None:
    if not baseline_id or baseline_id not in performance:
        return
    base = performance[baseline_id]["s_per_page"]["median"]
    if not base:
        return
    for pid, block in performance.items():
        med = block["s_per_page"]["median"]
        block["ratio_vs_baseline"] = round(med / base, 3) if med else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--group", default="table")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--label")
    ap.add_argument("--work-dir", default="benchpage-out")
    ap.add_argument("--run-id")
    ap.add_argument("--with-coldstart", action="store_true")
    ap.add_argument("--with-footprint", action="store_true")
    args = ap.parse_args(argv)
    run_dir = run(args.config, args.group, args.reps, args.results_dir,
                  args.label, args.work_dir, args.run_id,
                  args.with_coldstart, args.with_footprint)
    print(f"wrote {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
