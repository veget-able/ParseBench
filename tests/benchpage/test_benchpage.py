"""Round-trip smoke tests for the benchpage harness (no ParseBench run needed)."""

import json
from pathlib import Path

from benchpage import collect
from benchpage.emit import write_run
from benchpage.schema import (
    SOURCE_MEASURED,
    summary_skeleton,
    validate_summary,
)

CSV_HEADER = (
    "test_id,example_id,pipeline_name,product_type,success,error,tags,"
    "latency_ms,latency_ms_per_page,grits_con,grits_trm_composite,"
    "structural_consistency,table_record_match\n"
)


def _fake_run_dir(tmp_path: Path, name: str, latencies: list[float]) -> Path:
    d = tmp_path / name / "pymupdf4llm_markdown"
    d.mkdir(parents=True)
    (d / "_metadata.json").write_text(json.dumps({
        "pipeline": {"pipeline_name": "pymupdf4llm_markdown"},
        "run_config": {"max_concurrent": 1},
    }), encoding="utf-8")
    (d / "_summary.json").write_text(json.dumps({
        "total": len(latencies), "successful": len(latencies), "failed": 0,
    }), encoding="utf-8")
    xs = sorted(latencies)
    (d / "_evaluation_report.json").write_text(json.dumps({
        "total_examples": len(latencies),
        "successful": len(latencies),
        "failed": 0,
        "aggregate_metrics": {
            "avg_grits_trm_composite": 0.72,
            "avg_grits_con": 0.9,
            "avg_table_record_match": 0.8,
        },
        "aggregate_stats": {
            "latency_ms_per_page": {
                "avg": sum(xs) / len(xs),
                "p50": xs[len(xs) // 2],
                "p95": xs[-1],
                "count": len(xs),
                "unit": "ms/page",
            }
        },
    }), encoding="utf-8")
    rows = [
        f'table/doc{i},table/doc{i},pymupdf4llm_markdown,parse,True,,"table,easy",'
        f"{ms},{ms},0.9,0.72,1.0,0.8\n"
        for i, ms in enumerate(latencies)
    ]
    (d / "_evaluation_results.csv").write_text(CSV_HEADER + "".join(rows),
                                               encoding="utf-8")
    return d


def test_collect_uses_official_aggregates(tmp_path):
    r1 = collect.load_run(_fake_run_dir(tmp_path, "rep0", [100.0, 200.0]))
    r2 = collect.load_run(_fake_run_dir(tmp_path, "rep1", [300.0, 400.0]))
    combined = collect.combine_reps([r1, r2])

    assert combined["repetitions"] == 2
    assert combined["deterministic"] is True

    q = collect.quality_block(combined, SOURCE_MEASURED)
    assert q["gtrm"] == 72.0          # official avg_grits_trm_composite, 0-100 scale
    assert q["docs_scored"] == 2

    p = collect.performance_block(combined, SOURCE_MEASURED, peak_rss_mb=123.4)
    # official p50s are 200 and 400 ms; median across reps -> 0.3 s/page
    assert p["s_per_page"]["median"] == 0.3
    assert p["s_per_page"]["p95"] == 0.3   # median of 200 and 400 ms p95s
    assert p["pages_per_min"] == 200.0
    assert p["peak_rss_mb"] == 123.4

    rows = collect.document_rows({"pymupdf4llm": combined})
    assert len(rows) == 2
    assert rows[0]["pipelines"]["pymupdf4llm"]["success"] is True


def test_emit_round_trip(tmp_path):
    summary = summary_skeleton("20990101-test")
    summary["run"].update(started="2099-01-01T00:00:00Z",
                          dataset={"group": "table", "docs": 2},
                          llm_normalization=False, repetitions=1)
    summary["pipelines"]["pymupdf4llm"] = {
        "label": "PyMuPDF", "category": "local", "versions": {},
        "fee_per_1k_pages": None, "lock_sha256": None,
    }
    summary["quality"]["table"] = {
        "pymupdf4llm": {"source": "measured", "gtrm": 72.0, "grits_con": 90.0,
                        "table_record_match": 80.0, "docs_scored": 2,
                        "deterministic": True},
    }
    summary["performance"]["pymupdf4llm"] = {
        "source": "measured",
        "s_per_page": {"median": 0.25, "p95": 0.4, "mean": 0.25},
        "pages_per_min": 240.0, "cold_start_s": None, "peak_rss_mb": None,
    }
    assert validate_summary(summary) == []

    run_dir = write_run(tmp_path / "results", summary,
                        [{"doc": "table/doc0", "tags": "table", "pipelines": {}}])
    assert (run_dir / "summary.json").exists()
    index = json.loads((tmp_path / "results" / "index.json").read_text())
    assert index[0]["run_id"] == "20990101-test"


def _cat(metrics, successful=3, docs=None):
    return {"report": {"aggregate_metrics": metrics, "successful": successful},
            "docs": docs or []}


def test_group_quality_blocks_overall():
    run = {
        "categories": {
            "table": _cat({"avg_grits_trm_composite": 0.72, "avg_grits_con": 0.81,
                           "avg_table_record_match": 0.60}),
            "chart": _cat({"avg_rule_pass_rate": 0.10}),
            "layout": _cat({"avg_layout_element_rule_pass_rate": 0.60}),
            "text_content": _cat({"avg_content_faithfulness": 0.80}),
            "text_formatting": _cat({"avg_semantic_formatting": 0.50}),
        },
        "docs": [{"tags": "table,easy", "latency_ms_per_page": 100.0},
                 {"tags": "chart", "latency_ms_per_page": 300.0}],
    }
    combined = collect.combine_full_reps([run])

    blocks = collect.group_quality_blocks(combined, "measured")
    assert blocks["table"]["score"] == 72.0
    assert blocks["table"]["gtrm"] == 72.0
    assert blocks["chart"]["metric"] == "rule_pass_rate"
    # Overall follows the leaderboard definition: average across categories
    assert blocks["overall"]["score"] == 54.4
    assert blocks["overall"]["metric"] == "average_across_categories"
    assert blocks["overall"]["categories"]["layout"] == 60.0

    perf = collect.full_performance_block(combined, "measured")
    assert perf["s_per_page"]["median"] == 0.2  # pooled median of 100 and 300 ms


def test_validate_rejects_unknown_pipeline():
    summary = summary_skeleton("x")
    summary["run"].update(started="2099-01-01T00:00:00Z")
    summary["performance"]["ghost"] = {"source": "measured"}
    assert any("unknown pipeline" in p for p in validate_summary(summary))
