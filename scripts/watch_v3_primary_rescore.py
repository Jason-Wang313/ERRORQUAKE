"""Watchdog for DeepSeek-V3.2 primary synthetic rescoring."""

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
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--stall-seconds", type=int, default=1200)
    parser.add_argument("--restart-delay-seconds", type=int, default=20)
    return parser.parse_args()


def _line_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def _tracked_snapshot(base_dir: Path) -> dict[str, Any]:
    names = [
        "scores_primary_v3judge_power_law.jsonl",
        "scores_primary_v3judge_exponential.jsonl",
        "scores_primary_v3judge_uniform.jsonl",
        "resolved_scores_v3judge_power_law.jsonl",
        "resolved_scores_v3judge_exponential.jsonl",
        "resolved_scores_v3judge_uniform.jsonl",
        str(Path("results/analysis/experiment_0_report_v3judge.json")),
    ]
    snapshot: dict[str, Any] = {}
    for name in names:
        path = (base_dir / name).resolve()
        snapshot[name] = {
            "exists": path.exists(),
            "lines": _line_count(path) if path.suffix == ".jsonl" else None,
            "last_modified": path.stat().st_mtime if path.exists() else None,
        }
    return snapshot


def _latest_progress_mtime(repo: Path) -> float:
    snapshot = _tracked_snapshot(repo)
    latest = 0.0
    for item in snapshot.values():
        modified = item.get("last_modified")
        if modified is not None:
            latest = max(latest, float(modified))
    return latest


def _write_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _terminate(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
        return
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def main() -> int:
    args = _parse_args()
    repo = args.repo.resolve()
    launch_dir = repo / "data" / "synthetic" / "launch_logs"
    launch_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = launch_dir / "v3judge_watch_stdout.log"
    stderr_path = launch_dir / "v3judge_watch_stderr.log"
    log_path = launch_dir / "v3judge_watchdog.log"
    status_path = launch_dir / "v3judge_watchdog_status.json"

    attempt = 0
    while True:
        attempt += 1
        with log_path.open("a", encoding="utf-8") as watch_log:
            watch_log.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} START attempt={attempt}\n")
            watch_log.flush()
            with stdout_path.open("a", encoding="utf-8") as stdout_handle, stderr_path.open(
                "a", encoding="utf-8"
            ) as stderr_handle:
                started_at = time.time()
                process = subprocess.Popen(
                    [sys.executable, "-u", "scripts/run_v3_primary_rescore.py", "--resume"],
                    cwd=repo,
                    stdout=stdout_handle,
                    stderr=stderr_handle,
                    text=True,
                    env=os.environ.copy(),
                )
                last_progress = _latest_progress_mtime(repo)
                while process.poll() is None:
                    time.sleep(args.poll_seconds)
                    current_progress = _latest_progress_mtime(repo)
                    if current_progress > last_progress:
                        last_progress = current_progress
                    _write_status(
                        status_path,
                        {
                            "attempt": attempt,
                            "pid": process.pid,
                            "started_at_epoch": started_at,
                            "last_progress_epoch": last_progress,
                            "stall_seconds": args.stall_seconds,
                            "snapshot": _tracked_snapshot(repo),
                        },
                    )
                    if time.time() - last_progress > args.stall_seconds:
                        watch_log.write(
                            f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} STALL pid={process.pid}\n"
                        )
                        watch_log.flush()
                        _terminate(process)
                        break
                return_code = process.poll()
                _write_status(
                    status_path,
                    {
                        "attempt": attempt,
                        "pid": process.pid,
                        "return_code": return_code,
                        "finished_at_epoch": time.time(),
                        "snapshot": _tracked_snapshot(repo),
                    },
                )
                watch_log.write(
                    f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} EXIT attempt={attempt} code={return_code}\n"
                )
                watch_log.flush()
        if return_code == 0:
            return 0
        time.sleep(args.restart_delay_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
