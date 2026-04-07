"""Crash-resistant watchdog for the synthetic validation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--rpm", type=int, default=35)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stall-seconds", type=int, default=1200)
    parser.add_argument("--restart-delay-seconds", type=int, default=15)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, default=Path("data/synthetic"))
    return parser.parse_args()


def _latest_progress_mtime(base_dir: Path) -> float:
    latest = 0.0
    tracked_prefixes = (
        "target_scores_",
        "responses_",
        "scores_primary_",
        "scores_secondary_",
        "resolved_scores_",
    )
    for path in base_dir.rglob("*"):
        if not path.is_file():
            continue
        if path.parent.name in {"launch_logs", "logs"}:
            continue
        if not path.name.startswith(tracked_prefixes):
            continue
        latest = max(latest, path.stat().st_mtime)
    return latest


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _status_snapshot(output_dir: Path) -> dict[str, Any]:
    files = [
        "responses_power_law.jsonl",
        "scores_primary_power_law.jsonl",
        "scores_secondary_power_law.jsonl",
        "resolved_scores_power_law.jsonl",
        "responses_exponential.jsonl",
        "scores_primary_exponential.jsonl",
        "scores_secondary_exponential.jsonl",
        "resolved_scores_exponential.jsonl",
        "responses_uniform.jsonl",
        "scores_primary_uniform.jsonl",
        "scores_secondary_uniform.jsonl",
        "resolved_scores_uniform.jsonl",
    ]
    snapshot: dict[str, Any] = {}
    for name in files:
        path = output_dir / name
        snapshot[name] = {
            "exists": path.exists(),
            "lines": _line_count(path),
            "last_modified": path.stat().st_mtime if path.exists() else None,
        }
    return snapshot


def _write_status(status_path: Path, payload: dict[str, Any]) -> None:
    status_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _terminate_child(process: subprocess.Popen[str], grace_seconds: int = 20) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=grace_seconds)
        return
    except subprocess.TimeoutExpired:
        pass
    process.kill()
    process.wait(timeout=10)


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    output_dir = (repo / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    launch_dir = output_dir / "launch_logs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    status_path = launch_dir / "phase2a_watchdog_status.json"
    stdout_path = launch_dir / "phase2a_stdout.log"
    stderr_path = launch_dir / "phase2a_stderr.log"
    watchdog_log_path = launch_dir / "phase2a_watchdog.log"

    attempt = 0
    while True:
        attempt += 1
        with watchdog_log_path.open("a", encoding="utf-8") as watchdog_log:
            watchdog_log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} START attempt={attempt}\n")
            watchdog_log.flush()
            with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
                "a", encoding="utf-8"
            ) as stderr_handle:
                started_at = time.time()
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "scripts/run_synthetic.py",
                        "--n",
                        str(args.n),
                        "--rpm",
                        str(args.rpm),
                        "--resume",
                    ],
                    cwd=repo,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    env=os.environ.copy(),
                )
                last_progress = _latest_progress_mtime(output_dir)
                while process.poll() is None:
                    time.sleep(args.poll_seconds)
                    current_progress = _latest_progress_mtime(output_dir)
                    if current_progress > last_progress:
                        last_progress = current_progress
                    payload = {
                        "attempt": attempt,
                        "pid": process.pid,
                        "started_at_epoch": started_at,
                        "last_progress_epoch": last_progress,
                        "stall_seconds": args.stall_seconds,
                        "snapshot": _status_snapshot(output_dir),
                    }
                    _write_status(status_path, payload)
                    if time.time() - last_progress > args.stall_seconds:
                        watchdog_log.write(
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} STALL pid={process.pid}\n"
                        )
                        watchdog_log.flush()
                        _terminate_child(process)
                        break

                return_code = process.poll()
                payload = {
                    "attempt": attempt,
                    "pid": process.pid,
                    "return_code": return_code,
                    "finished_at_epoch": time.time(),
                    "snapshot": _status_snapshot(output_dir),
                }
                _write_status(status_path, payload)
                watchdog_log.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} EXIT attempt={attempt} code={return_code}\n"
                )
                watchdog_log.flush()
        if return_code == 0:
            return 0
        time.sleep(args.restart_delay_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
