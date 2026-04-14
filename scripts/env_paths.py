"""Helpers for locating repo-relative paths and optional env files."""

from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def get_env_path() -> Path:
    override = os.environ.get("ERRORQUAKE_ENV_PATH", "").strip()
    if override:
        return Path(override)

    candidates = [
        ROOT / ".env",
        ROOT / "MIRROR" / ".env",
        Path.cwd() / ".env",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return ROOT / ".env"


def load_keys_from_env_file(prefix: str) -> list[str]:
    env_path = get_env_path()
    if not env_path.exists():
        return []
    return [
        line.split("=", 1)[1].strip()
        for line in env_path.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix) and "=" in line and line.split("=", 1)[1].strip()
    ]
