# benchpage — measurement harness for the public PyMuPDF benchmark page

This directory produces the JSON consumed by the live benchmark section on
pymupdf.io (and by embeds of it, e.g. the Speed section of
`/compare/docling`). It wraps the stock `parse-bench` CLI as a subprocess
and never modifies ParseBench itself, so the quality numbers are exactly
what an unmodified checkout produces.

## What gets measured

| Section | Metrics | How |
|---|---|---|
| quality | GTRM (`avg_grits_trm_composite`), GriTS content, TableRecordMatch, determinism | lifted verbatim from ParseBench's `_evaluation_report.json` aggregates |
| performance | seconds/page (p50, p95, mean), pages/min, cold start (import + first document), peak RSS, ratio vs baseline | ParseBench's own `aggregate_stats.latency_ms_per_page`; RSS sampled via psutil; cold start in fresh subprocesses |
| footprint | installed MB, download MB, transitive dependency count, install seconds | throwaway venv per pipeline |

The harness performs no re-aggregation of ParseBench data: quality and
latency aggregates are ParseBench's own numbers, the only cross-run
arithmetic is the median across repetitions, and per-document rows are
the unmodified rows of the median repetition.

## Result layout

```
results/index.json                   run manifest (history, newest first)
results/runs/<run_id>/summary.json   everything the page needs on first load
results/runs/<run_id>/documents.json per-document detail, loaded lazily
```

`summary.json` is keyed by stable pipeline ids throughout, so an embed can
slice out a two-parser subset without re-aggregating. Every metric block
carries `"source": "measured" | "placeholder"`. Schema reference and
validation live in `schema.py`; `emit.write_run` refuses to write an
invalid summary. The checked-in run `20260820-sample-placeholder` is an
illustrative schema demo only and will be replaced by the first run from
the pinned runner.

## Running

Each pipeline runs from its own venv (clean installs, no cross
contamination). Describe them in a config, then:

```
python -m benchpage.run_bench --config benchpage/pipelines.example.json \
    --group table --reps 3 --results-dir results --label weekly
```

Cold start and footprint are slower and opt-in via `--with-coldstart` and
`--with-footprint`; both also work standalone (`python -m
benchpage.coldstart --help`, `python -m benchpage.footprint --help`).

## Measurement rules

* `--max_concurrent 1` always (hardcoded). The sequential path is the only
  one whose per-document latency is meaningful.
* Repetitions interleave pipelines A/B/A/B in the same session, so machine
  noise cancels in the ratio; each official latency stat is reported as
  its median across repetitions.
* Quality must be identical across repetitions; `deterministic` in the
  summary records whether it was.
* Cold start requires model files already on disk (run each tool once
  first) so the network is never inside the timed path.
* Absolute times are comparable across runs only from the pinned runner
  (fixed instance type, set `BENCH_INSTANCE_TYPE`; never a burstable
  t-family instance). Ratios are meaningful anywhere.
* Provenance travels with every run: ParseBench commit, per-venv
  `pip freeze` hash, instance/CPU/OS/Python, and the LLM-normalization
  setting (off by default since upstream #107).

## Docling

Docling is measured through upstream's stock `docling_serve` pipeline,
which calls the official docling-serve HTTP API
(`DOCLING_SERVE_ENDPOINT_URL`, endpoint `/v1/convert/source`), so both
sides of the pair are unmodified upstream code. The runner starts
docling-serve on the same machine before the run; Docling's latency
therefore includes a localhost HTTP hop, recorded here as a caveat, while
cold start and footprint measure the docling library directly in its own
venv. Because the docling venv only hosts the server, the docling entry
sets `parse_bench_cmd` to the shared parse-bench venv's CLI; that client
venv is installed as `parse-bench[<engine extras>,runners]`, since the
`runners` extra carries the provider-side dependencies (docling-core for
the docling_serve provider, among others). Note the published
leaderboard's "Docling-models" row was produced via the `docling_parse`
pipeline against a hosted inference endpoint; the scoring code downstream
of the provider is the same.

## Follow-ups

* Start-run-stop workflow for the pinned AWS runner (schedule/dispatch
  triggered only; no PR-triggered execution on the self-hosted runner),
  including bringing docling-serve up and down around the run.
