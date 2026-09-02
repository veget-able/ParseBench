"""Environment fingerprint for a benchmark run.

Every number on the page is only as good as the record of where it was
measured, so the summary carries instance type, CPU, RAM, OS, Python and a
lock hash per pipeline venv. On the pinned AWS runner, set
``BENCH_INSTANCE_TYPE`` (start-run-stop scripts know their own type); as a
fallback, ``BENCH_PROBE_IMDS=1`` asks the EC2 metadata service.
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
from datetime import datetime, timezone


def collect() -> dict:
    return {
        "instance": _instance_type(),
        "cpu": _cpu_name(),
        "ram_gb": _ram_gb(),
        "os": f"{platform.system()} {platform.release()}",
        "python": platform.python_version(),
    }


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git_commit(repo_root: str = ".") -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", repo_root, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def venv_lock_sha256(python_exe: str) -> str | None:
    """Hash of ``pip freeze`` output; ties a run to exact dependency versions."""
    try:
        out = subprocess.run(
            [python_exe, "-m", "pip", "freeze"],
            capture_output=True, text=True, timeout=120, check=True,
        )
        frozen = "\n".join(sorted(out.stdout.splitlines()))
        return hashlib.sha256(frozen.encode("utf-8")).hexdigest()
    except Exception:
        return None


def _instance_type() -> str | None:
    env = os.environ.get("BENCH_INSTANCE_TYPE")
    if env:
        return env
    if os.environ.get("BENCH_PROBE_IMDS") == "1":
        return _imds_instance_type()
    return None


def _imds_instance_type() -> str | None:
    import urllib.request

    try:
        token_req = urllib.request.Request(
            "http://169.254.169.254/latest/api/token",
            method="PUT",
            headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"},
        )
        token = urllib.request.urlopen(token_req, timeout=1).read().decode()
        req = urllib.request.Request(
            "http://169.254.169.254/latest/meta-data/instance-type",
            headers={"X-aws-ec2-metadata-token": token},
        )
        return urllib.request.urlopen(req, timeout=1).read().decode()
    except Exception:
        return None


def _cpu_name() -> str | None:
    if platform.system() == "Linux":
        # platform.processor() is just "x86_64" here; prefer the model name
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    for candidate in (os.environ.get("PROCESSOR_IDENTIFIER"), platform.processor()):
        if candidate:
            return candidate
    return None


def _ram_gb() -> float | None:
    try:
        import psutil

        return round(psutil.virtual_memory().total / 1024**3, 1)
    except Exception:
        return None
