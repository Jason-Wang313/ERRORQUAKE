"""End-to-end v6 → v7 orchestrator.

Runs Phase A merge → Phase B baseline → Phase B new (B1-B5) →
Phase C paper update → compile → ready for git push v7.

Each step is idempotent and resumable. Safe to re-run after a crash.
Re-running with completed inputs is a near-instant no-op.
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable

STEPS = [
    ("A5 merge", "scripts/run_v6_merge.py"),
    ("Phase B baseline", "scripts/run_v6_phase_b.py"),
    ("Phase B new (B1-B5)", "scripts/run_v6_phase_b_new.py"),
    ("Phase C paper update", "scripts/run_v6_paper_update.py"),
]


def run_step(name: str, script: str) -> bool:
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {name}\n{'='*70}")
    start = time.time()
    result = subprocess.run(
        [PYTHON, str(ROOT / script)],
        cwd=str(ROOT),
        capture_output=False,
    )
    dur = (time.time() - start) / 60
    if result.returncode == 0:
        print(f"\n  [{name}] DONE in {dur:.1f} min")
        return True
    print(f"\n  [{name}] FAILED with exit code {result.returncode}")
    return False


def compile_paper() -> bool:
    print(f"\n{'='*70}\nCompiling errorquakev7.pdf\n{'='*70}")
    paper = ROOT / "paper"
    cmds = [
        ["pdflatex", "-interaction=nonstopmode", "-jobname=errorquakev7", "main.tex"],
        ["bibtex", "errorquakev7"],
        ["pdflatex", "-interaction=nonstopmode", "-jobname=errorquakev7", "main.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-jobname=errorquakev7", "main.tex"],
    ]
    for cmd in cmds:
        result = subprocess.run(cmd, cwd=str(paper), capture_output=True)
        # Don't bail on bibtex non-zero — it warns about missing entries
        if result.returncode != 0 and "bibtex" not in cmd[0]:
            print(f"  {cmd[0]} returned {result.returncode}")
            tail = result.stdout.decode("utf-8", errors="replace").splitlines()[-30:]
            print("\n".join(tail))
    pdf = paper / "errorquakev7.pdf"
    if pdf.exists():
        size_kb = pdf.stat().st_size // 1024
        print(f"  PDF written: {pdf} ({size_kb} KB)")
        return True
    return False


def main() -> None:
    print(f"v6 → v7 orchestrator | {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Working dir: {ROOT}")

    for name, script in STEPS:
        if not run_step(name, script):
            print(f"\nABORTED at: {name}")
            return

    if not compile_paper():
        print("\nABORTED: paper compile failed")
        return

    print(f"\n[{time.strftime('%H:%M:%S')}] v7 ready. Run git commit/tag/push.")


if __name__ == "__main__":
    main()

