from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from _Core.paths import get_processed_root


def _params_dir(reference_folder: str | Path) -> Path:
    ref = Path(reference_folder).expanduser().resolve()
    root = get_processed_root(ref)
    params_dir = root / "_Params"
    params_dir.mkdir(parents=True, exist_ok=True)
    return params_dir


def save_params(reference_folder: str | Path, modality: str, tool: str, sample: str, payload: dict) -> Path:
    """
    Store per-sample parameters under:
      <reference>/_Processed/_Params/_params_<modality>_<sample>_<tool>.json
    """
    params_dir = _params_dir(reference_folder)
    sample_tag = (sample or "default").strip().replace(" ", "_")
    fname = f"_params_{modality.lower()}_{sample_tag}_{tool.lower()}.json"
    path = params_dir / fname
    data = {
        "modality": modality,
        "tool": tool,
        "sample": sample,
        "reference": str(Path(reference_folder).expanduser().resolve()),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "payload": payload or {},
    }
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_params(reference_folder: str | Path, modality: str, tool: str, sample: str) -> dict | None:
    params_dir = _params_dir(reference_folder)
    sample_tag = (sample or "default").strip().replace(" ", "_")
    fname = f"_params_{modality.lower()}_{sample_tag}_{tool.lower()}.json"
    path = params_dir / fname
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
