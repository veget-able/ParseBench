"""Install-footprint measurement in a throwaway venv.

Reports, per pipeline: wheel download size, installed size on disk,
transitive dependency count, and wall-clock install time. Comparable across
tools only when measured on the same machine and Python; record both (the
runner stores them in the summary's env block).

Usage:

    python -m benchpage.footprint --packages "pymupdf4llm==1.28.2"
    python -m benchpage.footprint --packages docling \
        --extra-index-url https://download.pytorch.org/whl/cpu
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def measure(packages: list[str], python_exe: str = sys.executable,
            index_url: str | None = None, extra_index_url: str | None = None,
            keep: bool = False) -> dict:
    work = Path(tempfile.mkdtemp(prefix="benchpage-footprint-"))
    try:
        venv_dir = work / "venv"
        subprocess.run([python_exe, "-m", "venv", str(venv_dir)], check=True,
                       capture_output=True)
        vpy = _venv_python(venv_dir)
        _pip(vpy, ["install", "--quiet", "--upgrade", "pip"])
        baseline_deps = _dep_count(vpy)

        extra = []
        if index_url:
            extra += ["--index-url", index_url]
        if extra_index_url:
            extra += ["--extra-index-url", extra_index_url]

        download_dir = work / "wheels"
        _pip(vpy, ["download", "--quiet", "-d", str(download_dir), *extra, *packages])
        download_mb = _dir_mb(download_dir)

        t0 = time.perf_counter()
        _pip(vpy, ["install", "--quiet", "--no-index",
                   "--find-links", str(download_dir), *packages])
        install_s = time.perf_counter() - t0

        return {
            "packages": packages,
            "install_mb": _dir_mb(_site_packages(venv_dir)),
            "download_mb": download_mb,
            "dep_count": _dep_count(vpy) - baseline_deps,
            "install_s": round(install_s, 1),
        }
    finally:
        if not keep:
            shutil.rmtree(work, ignore_errors=True)


def _venv_python(venv_dir: Path) -> str:
    win = venv_dir / "Scripts" / "python.exe"
    return str(win if win.exists() else venv_dir / "bin" / "python")


def _site_packages(venv_dir: Path) -> Path:
    win = venv_dir / "Lib" / "site-packages"
    if win.exists():
        return win
    candidates = sorted(venv_dir.glob("lib/python*/site-packages"))
    return candidates[0] if candidates else venv_dir


def _pip(python_exe: str, args: list[str]) -> None:
    out = subprocess.run([python_exe, "-m", "pip", *args],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"pip {args[0]} failed:\n{out.stderr[-2000:]}")


def _dep_count(python_exe: str) -> int:
    out = subprocess.run([python_exe, "-m", "pip", "list", "--format", "json"],
                         capture_output=True, text=True, check=True)
    return len(json.loads(out.stdout))


def _dir_mb(path: Path) -> float:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1024**2, 1)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--packages", required=True,
                    help="space-separated pip requirement specs")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--index-url")
    ap.add_argument("--extra-index-url")
    args = ap.parse_args(argv)
    result = measure(args.packages.split(), args.python,
                     args.index_url, args.extra_index_url)
    json.dump(result, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
