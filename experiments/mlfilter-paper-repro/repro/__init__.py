"""Self-contained reproduction package for the multilayer optical-filter XY-FMQAOA study."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

STUDY_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = STUDY_DIR.parents[1]
IN_PROJECT_TREE = (
    STUDY_DIR.parent.name == "experiments"
    and (PROJECT_ROOT / "data/inputs/mlfilter/v1-public-code").exists()
)


def default_output_dirs(mode: str = "full") -> tuple[Path, Path]:
    """Return `(run_dir, fig_dir)` adapted to project-tree vs standalone server execution."""
    if IN_PROJECT_TREE:
        return (
            PROJECT_ROOT / "data/runs/mlfilter-paper-repro" / mode,
            PROJECT_ROOT / "analysis/mlfilter-paper-repro" / mode,
        )
    return (
        STUDY_DIR / "outputs" / mode / "runs",
        STUDY_DIR / "outputs" / mode / "figures",
    )


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def atomic_write_json(path: Path | str, data: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    tmp.replace(target)
