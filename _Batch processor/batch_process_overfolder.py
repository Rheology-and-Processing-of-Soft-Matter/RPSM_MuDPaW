#!/usr/bin/env python3
# batch_process_overfolder.py
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path
from datetime import datetime

# --- ensure MuDPaW _Helpers subfolder is importable ---
base_dir = os.path.dirname(os.path.abspath(__file__))
helpers_dir = os.path.join(base_dir, "_Helpers")
if helpers_dir not in sys.path:
    sys.path.insert(0, helpers_dir)

try:
    import SAXS_helper
    print("[MuDPaW] SAXS_helper:", getattr(SAXS_helper, "__file__", "?"))
except Exception as e:
    print("[MuDPaW] WARNING SAXS_helper:", e)
try:
    import Rheology_helper
    print("[MuDPaW] Rheology_helper:", getattr(Rheology_helper, "__file__", "?"))
except Exception as e:
    print("[MuDPaW] WARNING Rheology_helper:", e)

def list_files_recursive(root: Path, exts=None):
    exts = tuple(e.lower() for e in (exts or []))
    for p in root.rglob("*"):
        if p.is_file() and (not exts or p.suffix.lower() in exts):
            yield p

def is_up_to_date(input_paths, output_paths):
    if not output_paths:
        return False
    inputs_mtime = max((os.path.getmtime(p) for p in input_paths if os.path.exists(p)), default=0)
    outputs_mtime = min((os.path.getmtime(p) for p in output_paths if os.path.exists(p)), default=0)
    return outputs_mtime and outputs_mtime >= inputs_mtime

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    return d

def _drop_legacy(kwargs: dict) -> dict:
    if not isinstance(kwargs, dict):
        return {}
    new_kwargs = dict(kwargs)
    new_kwargs.pop("rate_threshold", None)
    return new_kwargs

def load_callable(module_name: str, candidates: list[str]):
    import importlib
    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        raise RuntimeError(f"Cannot import {module_name}: {e}")
    for name in candidates:
        if hasattr(mod, name):
            return getattr(mod, name)
    raise RuntimeError(f"{module_name} has none of the expected entrypoints: {', '.join(candidates)}")

def run_saxs_for_folder(ref_dir: Path, cfg: dict, skip_if_up_to_date=True, dry_run=False):
    raw_dir = ref_dir / "SAXS"
    out_dir = ensure_dir(ref_dir / cfg.get("output_dirname", "_Processed/SAXS"))
    inputs = list(list_files_recursive(raw_dir))
    outputs = [out_dir / "output_SAXS.json", out_dir / "output_SAXS.csv"]

    if skip_if_up_to_date and is_up_to_date([str(p) for p in inputs], [str(p) for p in outputs]):
        return {"status":"skipped_up_to_date","out":str(out_dir)}
    if dry_run:
        return {"status":"dry_run","inputs":len(inputs),"out":str(out_dir)}

    fn = load_callable("SAXS_helper", ["batch_process_saxs_reference"])

    smoothing = float(cfg.get("smoothing_window", 0.04))
    sigma     = float(cfg.get("gaussian_sigma_deg", 0.5))
    tmin      = float(cfg.get("theta_min_deg", 2.0))
    tmax      = float(cfg.get("theta_max_deg", 180.0))
    no_plots  = bool(cfg.get("no_plots", True))

    fn(str(ref_dir), smoothing, sigma, tmin, tmax, no_plots=no_plots)
    return {"status":"called_helper","out":str(out_dir)}

def run_rheology_for_folder(ref_dir: Path, cfg: dict, skip_if_up_to_date=True, dry_run=False):
    raw_dir = ref_dir / "Rheology"
    out_dir = ensure_dir(ref_dir / cfg.get("output_dirname", "_Processed/Rheology"))
    inputs = list(list_files_recursive(raw_dir, exts=[".csv",".txt",".tsv"]))
    outputs = [out_dir / "output_viscosity.json", out_dir / "output_viscosity.csv"]

    if skip_if_up_to_date and is_up_to_date([str(p) for p in inputs], [str(p) for p in outputs]):
        return {"status":"skipped_up_to_date","out":str(out_dir)}
    if dry_run:
        return {"status":"dry_run","inputs":len(inputs),"out":str(out_dir)}

    fn = load_callable("Rheology_helper", ["process_all_rheology_files"])

    try:
        from Rheology_helper import list_rheology_samples
        sample_folders = list_rheology_samples(str(raw_dir))
    except Exception:
        sample_folders = []

    mode   = (cfg.get("modes") or ["triggered"])[0]
    steady = float(cfg.get("steady_window_s", 10.0))
    no_plots = bool(cfg.get("no_plots", True))

    kwargs = _drop_legacy({
        "mode": mode,
        "steady_window_s": steady,
        "no_plots": no_plots,
    })
    try:
        fn(str(raw_dir), sample_folders, **kwargs)
        status = "called_helper"
    except TypeError:
        fn(str(raw_dir), sample_folders)
        status = "called_helper_legacy"

    return {"status":status, "out":str(out_dir)}

def main():
    ap = argparse.ArgumentParser(description="Batch process SAXS and Rheology over an overfolder via YAML config.")
    ap.add_argument("--config", "-c", required=True, help="Path to batch_config.yaml produced by the GUI.")
    args = ap.parse_args()

    # Load YAML without requiring pyyaml (support quick json as well)
    cfg_path = Path(args.config).expanduser().resolve()
    if not cfg_path.exists():
        print(f"[orchestrator] Config missing: {cfg_path}", file=sys.stderr)
        sys.exit(2)

    try:
        text = cfg_path.read_text(encoding="utf-8")
        if text.strip().startswith("{"):
            data = json.loads(text)
        else:
            try:
                import yaml  # type: ignore
            except Exception:
                print("[orchestrator] pyyaml not installed and config is YAML; please install pyyaml.", file=sys.stderr)
                sys.exit(2)
            data = yaml.safe_load(text) or {}
    except Exception as e:
        print(f"[orchestrator] Failed to read config: {e}", file=sys.stderr)
        sys.exit(2)

    overfolder = Path(data.get("overfolder","")).expanduser().resolve()
    if not overfolder.exists():
        print(f"[orchestrator] Overfolder does not exist: {overfolder}", file=sys.stderr)
        sys.exit(2)

    defaults = data.get("defaults", {})
    modal = data.get("modalities", {"saxs": True, "rheology": True})
    dry_run = bool(defaults.get("dry_run", False))
    skip_if_uptodate = bool(defaults.get("skip_if_up_to_date", True))

    # Discover reference folders (children with at least SAXS/ or Rheology/)
    ref_folders = []
    for c in sorted([p for p in overfolder.iterdir() if p.is_dir()]):
        if (c / "SAXS").exists() or (c / "Rheology").exists():
            ref_folders.append(c)

    results = []
    for ref in ref_folders:
        rec = {
            "reference": ref.name,
            "path": str(ref),
            "saxs": None, "rheology": None,
            "error": None,
            "started": datetime.now().isoformat(),
            "finished": None
        }
        try:
            if modal.get("saxs") and (ref / "SAXS").exists():
                rec["saxs"] = run_saxs_for_folder(ref, defaults.get("saxs", {}), skip_if_uptodate, dry_run)
            if modal.get("rheology") and (ref / "Rheology").exists():
                rec["rheology"] = run_rheology_for_folder(ref, defaults.get("rheology", {}), skip_if_uptodate, dry_run)
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
        rec["finished"] = datetime.now().isoformat()
        results.append(rec)

    # Write summary in overfolder
    summary_json = overfolder / "_batch_summary.json"
    try:
        summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"[orchestrator] Failed to write summary: {e}", file=sys.stderr)

    # Optional CSV
    try:
        import csv
        summary_csv = overfolder / (defaults.get("summary_filename") or "_batch_summary.csv")
        with open(summary_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["reference","saxs_status","rheology_status","error","started","finished"])
            for r in results:
                w.writerow([
                    r["reference"],
                    (r["saxs"] or {}).get("status") if r["saxs"] else "",
                    (r["rheology"] or {}).get("status") if r["rheology"] else "",
                    "OK" if not r["error"] else "ERROR",
                    r["started"], r["finished"]
                ])
    except Exception as e:
        print(f"[orchestrator] Failed to write CSV summary: {e}", file=sys.stderr)

    print(f"Wrote summary JSON → {summary_json}")
    had_error = any(r.get("error") for r in results)
    if had_error:
        print("[orchestrator] One or more references failed. See summary JSON/CSV.", file=sys.stderr)
        sys.exit(1)
    print("Done.")

if __name__ == "__main__":
    main()
