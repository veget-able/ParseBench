"""Cold-start measurement: import time and first-document time, separately.

Model-loading stacks pay their cost at import or on the first document; a
pages/min figure measured on a warm process hides it. Each sample runs in a
fresh subprocess of the pipeline's own venv so nothing is cached between
samples. Model files must already be on disk (run each tool once before
measuring) so the network never sits inside the timed path.

Usage:

    python -m benchpage.coldstart --python .venvs/pymupdf4llm/Scripts/python.exe \
        --import-stmt "import pymupdf4llm" \
        --first-doc-expr "pymupdf4llm.to_markdown(DOC)" \
        --doc sample.pdf --runs 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from .schema import median

_SNIPPET = """
import json, sys, time
DOC = sys.argv[1]
t0 = time.perf_counter()
{import_stmt}
t1 = time.perf_counter()
_result = {first_doc_expr}
t2 = time.perf_counter()
print(json.dumps({{"import_s": t1 - t0, "first_doc_s": t2 - t1}}))
"""


def measure(python_exe: str, import_stmt: str, first_doc_expr: str,
            doc_path: str, runs: int = 3, timeout_s: int = 600) -> dict:
    samples = []
    code = _SNIPPET.format(import_stmt=import_stmt, first_doc_expr=first_doc_expr)
    for _ in range(runs):
        out = subprocess.run(
            [python_exe, "-c", code, doc_path],
            capture_output=True, text=True, timeout=timeout_s,
        )
        if out.returncode != 0:
            raise RuntimeError(f"cold-start subprocess failed:\n{out.stderr[-2000:]}")
        samples.append(json.loads(out.stdout.strip().splitlines()[-1]))
    return {
        "import": round(median([s["import_s"] for s in samples]), 3),
        "first_doc": round(median([s["first_doc_s"] for s in samples]), 3),
        "runs": runs,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", required=True)
    ap.add_argument("--import-stmt", required=True)
    ap.add_argument("--first-doc-expr", required=True)
    ap.add_argument("--doc", required=True)
    ap.add_argument("--runs", type=int, default=3)
    args = ap.parse_args(argv)
    result = measure(args.python, args.import_stmt, args.first_doc_expr,
                     args.doc, args.runs)
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
