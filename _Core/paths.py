from __future__ import annotations
from pathlib import Path


def get_processed_root(reference_folder: str | Path) -> Path:
    """Return <reference_folder>/_Processed as a Path (no modality appended)."""
    ref = Path(reference_folder).expanduser().resolve()
    return ref / "_Processed"


def get_modality_out_dir(reference_folder: str | Path, modality: str) -> Path:
    """Return <reference_folder>/_Processed/<modality> with a guard against double nesting."""
    root = get_processed_root(reference_folder)
    m = str(modality).strip().strip("/").strip("\\")
    out = root / m
    parts = [p.lower() for p in out.parts]
    if len(parts) >= 2 and parts[-1] == parts[-2]:
        raise RuntimeError(f"Nested modality folder detected: {out}")
    return out
