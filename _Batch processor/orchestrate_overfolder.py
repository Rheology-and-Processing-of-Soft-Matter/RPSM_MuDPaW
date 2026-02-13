#!/usr/bin/env python3
# orchestrate_overfolder.py
# Fresh, fail-fast batch runner that shells out to your standalone processors:
#   - SAXS_data_processor_v4.py
#   - Read_viscosity_v3.py
#
# Optional: call DataGraph writer per SAXS dataset once results exist.

from __future__ import annotations
import argparse, csv, json, os, re, sys, subprocess, traceback, shlex
from pathlib import Path
from datetime import datetime
from typing import Iterable, Tuple, List, Dict, Any

# --------------------------- CLI ---------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Fresh batch runner for MuDPaW over an overfolder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("overfolder", help="Path to folder containing reference folders")
    ap.add_argument("--modalities", choices=["both","saxs","rheology","none"], default="both",
                    help="Which modalities to run (use 'none' for DataGraph-only runs)")
    ap.add_argument("--skip-fresh", action="store_true", default=True,
                    help="Skip a reference if outputs are up-to-date")
    ap.add_argument("--no-skip-fresh", action="store_false", dest="skip_fresh",
                    help="Do not skip up-to-date references")
    ap.add_argument("--dry-run", action="store_true", help="Discovery only, no processing")
    ap.add_argument("--workers", type=int, default=1, help="Parallel processes (subprocess fanout)")
    ap.add_argument("--python", dest="py", default=sys.executable,
                    help="Python interpreter to call the standalone processors")

    # SAXS knobs (used only if modalities includes saxs; passed to standalone if supported)
    ap.add_argument("--saxs-no-plots", action="store_true", help="Disable plotting in SAXS")
    ap.add_argument("--saxs-smoothing", type=float, default=0.04)
    ap.add_argument("--saxs-sigma", type=float, default=0.5)
    ap.add_argument("--saxs-theta-min", type=float, default=2.0)
    ap.add_argument("--saxs-theta-max", type=float, default=180.0)
    ap.add_argument("--saxs-method", choices=["fitting", "direct"], default="fitting",
                    help="HoP computation method to use in SAXS processing.")
    ap.add_argument("--saxs-mirror", action="store_true",
                    help="Mirror azimuthal data prior to processing.")

    # Rheology knobs
    ap.add_argument("--rheo-no-plots", action="store_true", help="Disable plotting in Rheology")
    ap.add_argument("--rheo-mode", choices=["triggered","nontriggered","other"], default="triggered")
    ap.add_argument("--rheo-steady-window", type=float, default=10.0)

    # DataGraph writer (optional)
    ap.add_argument("--write-dg", action="store_true",
                    help="Write DataGraph exports for each SAXS dataset after processing")
    ap.add_argument("--dg-template", help="Path to .dgraph template to export per (SAXS,Rheo) pair")
    ap.add_argument("--dg-cli", default="/Applications/DataGraph.app/Contents/Library/dgraph",
                    help="DataGraph CLI path")
    ap.add_argument("--dg-restart-frequency", type=int, default=12,
                    help="Quit and relaunch the DataGraph CLI after this many exports (helps long runs stay fast). "
                         "Use 1 to restart after every export, or 0 to keep the previous 'quit only at the end' behavior.")
    ap.add_argument("--dg-single-file", action="store_true",
                    help="Combine processed outputs into one DataGraph export instead of per-sample exports.")
    ap.add_argument("--dg-single-output", help="Override the output path for the combined DataGraph file.")
    return ap.parse_args()

# ------------------------ Utilities ------------------------------------

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def list_files_recursive(root: Path, exts: Iterable[str] | None = None) -> List[Path]:
    exts = tuple(e.lower() for e in (exts or []))
    out = []
    for q in root.rglob("*"):
        if q.is_file() and (not exts or q.suffix.lower() in exts):
            out.append(q)
    return out

def is_up_to_date(input_paths: Iterable[Path], output_paths: Iterable[Path]) -> bool:
    inputs_m = [p.stat().st_mtime for p in input_paths if p.exists()]
    outputs_m = [p.stat().st_mtime for p in output_paths if p.exists()]
    if not inputs_m or not outputs_m:
        return False
    return min(outputs_m) >= max(inputs_m)

def discover_references(overfolder: Path) -> List[Path]:
    refs = []
    for c in sorted([p for p in overfolder.iterdir() if p.is_dir()]):
        if (c / "SAXS").exists() or (c / "Rheology").exists():
            refs.append(c)
    return refs

# ------------------------ Standalone calls -----------------------------

def run_saxs_standalone(py: str, base_dir: Path, ref_dir: Path, args: argparse.Namespace) -> None:
    """Invoke SAXS_data_processor_v4.py for the reference's SAXS folder."""
    script = base_dir.parent / "_Routines" / "SAXS" / "SAXS_data_processor_v4.py"
    if not script.exists():
        raise FileNotFoundError(f"SAXS processor not found: {script}")
    saxs_root = ref_dir / "SAXS"
    if not saxs_root.exists():
        raise FileNotFoundError(f"SAXS folder missing in {ref_dir}")

    cmd = [
        py, str(script),
        str(saxs_root),
        ref_dir.name,
        str(args.saxs_smoothing),
        str(args.saxs_sigma),
        str(args.saxs_theta_min),
        str(args.saxs_theta_max),
    ]
    cmd += ["--method", args.saxs_method]
    if args.saxs_mirror:
        cmd.append("--mirror")
    env = os.environ.copy()
    repo_root = str(base_dir.parent)
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")])
    if args.saxs_no_plots:
        env["MUDRAW_SAXS_NO_PLOTS"] = "1"
    subprocess.run(cmd, check=True, env=env, cwd=repo_root)

def run_rheology_standalone(py: str, base_dir: Path, ref_dir: Path, args: argparse.Namespace) -> None:
    """Invoke Read_viscosity_v3.py for the reference's Rheology folder."""
    script = base_dir.parent / "_Routines" / "Rheo" / "Read_viscosity_v3.py"
    if not script.exists():
        raise FileNotFoundError(f"Rheology processor not found: {script}")
    rheo_root = ref_dir / "Rheology"
    if not rheo_root.exists():
        raise FileNotFoundError(f"Rheology folder missing in {ref_dir}")

    cmd = [py, str(script), str(rheo_root),
           "--mode", args.rheo_mode,
           "--steady-sec", str(args.rheo_steady_window)]
    if args.rheo_no_plots:
        cmd += ["--no-preview"]

    env = os.environ.copy()
    repo_root = str(base_dir.parent)
    env["PYTHONPATH"] = os.pathsep.join([repo_root, env.get("PYTHONPATH", "")])
    subprocess.run(cmd, check=True, env=env, cwd=repo_root)

# ------------------------ DataGraph writer -----------------------------

def _read_json_dict(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    except Exception:
        pass
    return {}

def _extract_sample_name(data: dict, path: Path) -> str:
    for key in ("sample_name", "name", "sample", "dataset", "label"):
        val = data.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    stem = path.stem
    if stem.startswith("_output_"):
        stem = stem[len("_output_"):]
    return stem

def _tokenize(name: str) -> List[str]:
    return [tok for tok in re.split(r"[^a-z0-9]+", name.lower()) if tok]

def _resolve_csvs(data: dict, processed_root: Path) -> List[str]:
    candidates: List[str] = []
    csv_lists = data.get("csv_outputs")
    if isinstance(csv_lists, list):
        candidates.extend(csv_lists)
    single = data.get("csv_output") or data.get("csv_output_rel")
    if isinstance(single, str):
        candidates.append(single)
    for key in ("radial_matrix_csv", "radial_flat_csv", "radial_csv"):
        val = data.get(key)
        if isinstance(val, str):
            candidates.append(val)
    resolved: List[str] = []
    seen: set[str] = set()
    for item in candidates:
        if not item:
            continue
        p = Path(item)
        if not p.is_absolute():
            p = processed_root / item
        norm = str(p)
        if norm not in seen:
            seen.add(norm)
            resolved.append(norm)
    return resolved

def _select_primary_csvs(csvs: List[str]) -> List[str]:
    """
    Return the key SAXS CSVs to feed DataGraph:
      - SAXS_1_*_export.csv (P2/P4/P6/OAS summary)
      - SAXS_radial_flat_*.csv (flattened radial for color plots)
      - SAXS_radial_matrix_*.csv (full radial matrix)
    If none are found, fall back to the first available CSVs.
    """
    def pick(pattern: str) -> str | None:
        for f in csvs:
            if pattern in Path(f).name:
                return f
        return None
    picks = []
    s_export = pick("SAXS_1_")                # azimuthal summary (P2/P4/P6/OAS)
    s_raw_flat = pick("SAXS_2_")              # azimuthal flattened (for color plots)
    s_rad_flat = pick("SAXS_radial_flat_")    # radial flattened
    s_rad_matrix = pick("SAXS_radial_matrix_")# full radial matrix
    for item in (s_export, s_raw_flat, s_rad_flat, s_rad_matrix):
        if item:
            picks.append(item)
    if not picks and csvs:
        picks.append(csvs[0])
    return picks

def _categorize_modal(path: Path) -> str | None:
    name = path.name.lower()
    if "saxs" in name:
        return "SAXS"
    if "rheo" in name or "visco" in name:
        return "Rheology"
    if "pli" in name:
        return "PLI"
    return None

def _load_modal_entries(ref_dir: Path) -> Dict[str, List[dict]]:
    processed = ref_dir / "_Processed"
    outputs = {"SAXS": [], "Rheology": [], "PLI": []}
    if not processed.exists():
        return outputs
    for path in sorted(processed.glob("_output_*.json")):
        kind = _categorize_modal(path)
        if not kind:
            continue
        data = _read_json_dict(path)
        entry = {
            "path": path,
            "data": data,
            "sample": _extract_sample_name(data, path),
            "csvs": _resolve_csvs(data, processed),
            "tokens": _tokenize(_extract_sample_name(data, path)),
            "processed_root": processed,
        }
        outputs[kind].append(entry)
    return outputs

def _pick_best_match(target_tokens: List[str], candidates: List[dict]) -> dict | None:
    if not candidates:
        return None
    if not target_tokens:
        return candidates[0]
    best = candidates[0]
    best_score = -1
    target_set = set(target_tokens)
    for cand in candidates:
        score = len(target_set & set(cand.get("tokens", [])))
        if score > best_score:
            best = cand
            best_score = score
    return best

def _build_writer_jobs(ref_dir: Path) -> List[dict]:
    outputs = _load_modal_entries(ref_dir)
    saxs_entries = outputs["SAXS"]
    rheo_entries = outputs["Rheology"]
    pli_entries = outputs["PLI"]
    jobs: List[dict] = []
    if not (saxs_entries or rheo_entries or pli_entries):
        return jobs

    if saxs_entries:
        anchor_kind = "SAXS"
        anchors = saxs_entries
    elif rheo_entries:
        anchor_kind = "Rheology"
        anchors = rheo_entries
    else:
        anchor_kind = "PLI"
        anchors = pli_entries

    for anchor in anchors:
        tokens = anchor.get("tokens", [])
        saxs = anchor if anchor_kind == "SAXS" else _pick_best_match(tokens, saxs_entries)
        rheo = anchor if anchor_kind == "Rheology" else _pick_best_match(tokens, rheo_entries)
        pli = anchor if anchor_kind == "PLI" else _pick_best_match(tokens, pli_entries)
        files = []
        for group in (
            (saxs or {}).get("csvs", []),
            (rheo or {}).get("csvs", []),
            (pli or {}).get("csvs", []),
        ):
            for item in group:
                if item and item not in files:
                    files.append(item)
        # Restrict SAXS to key CSVs (export + raw flat) to avoid unexpected multiple q-ranges while keeping needed files
        if saxs:
            sax_csvs = saxs.get("csvs", [])
            primary = _select_primary_csvs(sax_csvs)
            # Replace any SAXS entries in files with just the primary picks
            files = primary + [f for f in files if f not in sax_csvs]
        if not files:
            continue
        processed_root = None
        for entry in (saxs, rheo, pli):
            if entry and entry.get("processed_root"):
                processed_root = entry["processed_root"]
                break
        jobs.append({
            "saxs": saxs,
            "rheo": rheo,
            "pli": pli,
            "files": files,
            "processed_root": processed_root,
        })
    return jobs

def run_datagraph_writer_for_ref(ref_dir: Path, tpl: Path, dg_cli: Path,
                                 restart_frequency: int | None = None) -> List[str]:
    jobs = _build_writer_jobs(ref_dir)
    exports: List[str] = []
    if not jobs:
        return exports

    tpl = tpl.expanduser().resolve()
    dg_cli = dg_cli.expanduser().resolve()
    if not tpl.exists():
        raise FileNotFoundError(f".dgraph template not found: {tpl}")
    if not dg_cli.exists():
        raise FileNotFoundError(f"DataGraph CLI not found: {dg_cli}")

    total_jobs = len(jobs)
    chunk = restart_frequency if restart_frequency and restart_frequency > 0 else None
    for idx, job in enumerate(jobs):
        sample_src = job.get("saxs") or job.get("rheo") or job.get("pli")
        sample = job.get("sample_override")
        if sample is None and sample_src:
            sample = sample_src.get("sample") or Path(sample_src["path"]).stem
        sample = sample or ref_dir.name
        safe_sample = sample.replace(os.sep, "_").strip() or "sample"
        files: List[str] = []
        processed_root = job.get("processed_root") or (ref_dir / "_Processed")
        for raw in job["files"]:
            candidate = Path(raw).expanduser()
            if not candidate.is_absolute():
                candidate = (processed_root / candidate).resolve()
            try:
                if not candidate.resolve().is_relative_to(ref_dir.resolve()):
                    # Ignore inputs that are not inside this reference; prevents cross-ref duplication.
                    continue
            except Exception:
                pass
            if candidate.exists():
                files.append(str(candidate))
            else:
                print(f"[writer] Missing input file for {sample}: {candidate}")
        if not files:
            ref_hint = (sample_src or {}).get("path", ref_dir)
            print(f"[writer] Skipping {ref_hint} (no resolved CSV inputs)")
            continue
        output_name = f"{ref_dir.name}_{safe_sample}.dgraph"
        output_path = ref_dir / output_name
        parts = [shlex.quote(str(dg_cli))]
        parts.extend(shlex.quote(f) for f in files)
        parts.extend(["-script", shlex.quote(str(tpl)),
                      "-v", shlex.quote(f"Sample_1={ref_dir.name}"),
                      "-output", shlex.quote(str(output_path))])
        should_quit = False
        if chunk is None:
            should_quit = (idx == total_jobs - 1)
        else:
            should_quit = ((idx + 1) % chunk == 0) or (idx == total_jobs - 1)
        if should_quit:
            parts.append("-quitAfterScript")
        cmd_str = " ".join(parts)
        print(f"[writer] Running DataGraph export: {cmd_str}")
        subprocess.run(cmd_str, check=True, cwd=str(dg_cli.parent), shell=True)
        exports.append(str(output_path))
    return exports

# ------------------------ Single-file DataGraph helpers ----------------

def _select_column(fieldnames: List[str], candidates: List[str]) -> str | None:
    """Pick the first header that matches a candidate (case/space/punct insensitive, tolerant of unit suffixes)."""
    canon = []
    for name in fieldnames or []:
        key = re.sub(r"[^a-z0-9]", "", name.lower())
        canon.append((key, name))
    for cand in candidates:
        key = re.sub(r"[^a-z0-9]", "", cand.lower())
        for ck, orig in canon:
            if ck == key or ck.startswith(key) or key in ck:
                return orig
    return None

def _collect_saxs_rows(ref_dir: Path, entries: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for entry in entries:
        processed_root = entry.get("processed_root") or (ref_dir / "_Processed")
        target = None
        for raw in entry.get("csvs", []):
            p = Path(raw)
            if not p.is_absolute():
                p = processed_root / p
            if p.name.startswith("SAXS_1_") and p.name.endswith("_export.csv") and p.exists():
                target = p
                break
        if not target:
            continue
        try:
            with open(target, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                c_p2 = _select_column(reader.fieldnames, ["p2"])
                c_p4 = _select_column(reader.fieldnames, ["p4"])
                c_p6 = _select_column(reader.fieldnames, ["p6"])
                c_oas = _select_column(reader.fieldnames, ["oas"])
                if not all([c_p2, c_p4, c_p6, c_oas]):
                    continue
                for row in reader:
                    rows.append({
                        "Folder": ref_dir.name,
                        "P2": row.get(c_p2, ""),
                        "P4": row.get(c_p4, ""),
                        "P6": row.get(c_p6, ""),
                        "OAS": row.get(c_oas, ""),
                    })
        except Exception as exc:
            print(f"[single-dg] Skipping SAXS file {target}: {exc}")
    return rows

def _collect_rheo_rows(ref_dir: Path, entries: List[dict]) -> List[dict]:
    rows: List[dict] = []
    for entry in entries:
        processed_root = entry.get("processed_root") or (ref_dir / "_Processed")
        target = None
        for raw in entry.get("csvs", []):
            p = Path(raw)
            if not p.is_absolute():
                p = processed_root / p
            if p.suffix.lower() == ".csv" and p.exists():
                target = p
                break
        if not target:
            continue
        try:
            with open(target, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                c_rate = _select_column(reader.fieldnames, ["shear_rate", "shearrate", "rate"])
                c_visc = _select_column(reader.fieldnames, ["viscosity", "eta"])
                c_stress = _select_column(reader.fieldnames, ["shear_stress", "shearstress", "stress"])
                if not all([c_rate, c_visc, c_stress]):
                    continue
                for row in reader:
                    rows.append({
                        "Folder": ref_dir.name,
                        "ShearRate": row.get(c_rate, ""),
                        "Viscosity": row.get(c_visc, ""),
                        "ShearStress": row.get(c_stress, ""),
                    })
        except Exception as exc:
            print(f"[single-dg] Skipping Rheology file {target}: {exc}")
    return rows

def _write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    ensure_dir(path.parent)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def run_single_datagraph(overfolder: Path, refs: List[Path], tpl: Path, dg_cli: Path,
                         include_saxs: bool, include_rheo: bool,
                         output_override: Path | None = None) -> str | None:
    tpl = tpl.expanduser().resolve()
    dg_cli = dg_cli.expanduser().resolve()
    if not tpl.exists():
        raise FileNotFoundError(f".dgraph template not found: {tpl}")
    if not dg_cli.exists():
        raise FileNotFoundError(f"DataGraph CLI not found: {dg_cli}")

    combined_root = ensure_dir(overfolder / "_Processed")
    saxs_rows: list[dict] = []
    rheo_rows: list[dict] = []
    for ref in refs:
        outputs = _load_modal_entries(ref)
        if include_saxs:
            saxs_rows.extend(_collect_saxs_rows(ref, outputs["SAXS"]))
        if include_rheo:
            rheo_rows.extend(_collect_rheo_rows(ref, outputs["Rheology"]))

    inputs: list[Path] = []
    if include_saxs and saxs_rows:
        saxs_csv = combined_root / "_combined_SAXS.csv"
        _write_csv(saxs_csv, ["Folder", "P2", "P4", "P6", "OAS"], saxs_rows)
        inputs.append(saxs_csv)
    if include_rheo and rheo_rows:
        rheo_csv = combined_root / "_combined_Rheology.csv"
        _write_csv(rheo_csv, ["Folder", "ShearRate", "Viscosity", "ShearStress"], rheo_rows)
        inputs.append(rheo_csv)

    if not inputs:
        print("[single-dg] No combined inputs to export.")
        return None

    output_path = output_override or (combined_root / f"{overfolder.name}_combined.dgraph")
    parts = [shlex.quote(str(dg_cli))]
    parts.extend(shlex.quote(str(p)) for p in inputs)
    parts.extend(["-script", shlex.quote(str(tpl)),
                  "-v", shlex.quote(f"Sample_1={overfolder.name}"),
                  "-output", shlex.quote(str(output_path)),
                  "-quitAfterScript"])
    cmd_str = " ".join(parts)
    print(f"[single-dg] Running combined export: {cmd_str}")
    subprocess.run(cmd_str, check=True, cwd=str(dg_cli.parent), shell=True)
    return str(output_path)

# ------------------------ Main run loop --------------------------------

def main() -> None:
    args = parse_args()
    overfolder = Path(args.overfolder).expanduser().resolve()
    if not overfolder.exists():
        print(f"[orchestrate] Overfolder not found: {overfolder}", file=sys.stderr)
        sys.exit(2)

    tpl_path = Path(args.dg_template).expanduser().resolve() if args.dg_template else None
    dg_cli_path = Path(args.dg_cli).expanduser().resolve()
    if (args.write_dg or args.dg_single_file) and not tpl_path:
        print("[orchestrate] DataGraph export requested but --dg-template is missing.", file=sys.stderr)
        sys.exit(2)

    base_dir = Path(__file__).resolve().parent  # location of scripts
    refs = discover_references(overfolder)
    if args.dry_run:
        print(f"[orchestrate] DRY-RUN: {len(refs)} references under {overfolder}")
        for r in refs: print(" -", r.name)
        sys.exit(0)

    results: List[Dict[str, Any]] = []
    combined_export: str | None = None
    combined_error: str | None = None
    for ref in refs:
        rec: Dict[str, Any] = {
            "reference": ref.name,
            "path": str(ref),
            "saxs": None, "rheology": None,
            "writer": None,
            "error": None,
            "started": datetime.now().isoformat(),
            "finished": None
        }
        try:
            # --- SAXS ---
            if args.modalities in ("both","saxs") and (ref / "SAXS").exists():
                sax_in = list_files_recursive(ref / "SAXS")
                sax_out_dir = ref / "_Processed" / "SAXS"
                sax_outs = [sax_out_dir / "output_SAXS.json", sax_out_dir / "output_SAXS.csv"]
                if args.skip_fresh and is_up_to_date(sax_in, sax_outs):
                    rec["saxs"] = {"status": "skipped_up_to_date", "out": str(sax_out_dir)}
                else:
                    run_saxs_standalone(args.py, base_dir, ref, args)
                    rec["saxs"] = {"status": "called_standalone", "out": str(sax_out_dir)}

            # --- Rheology ---
            if args.modalities in ("both","rheology") and (ref / "Rheology").exists():
                rh_in = list_files_recursive(ref / "Rheology", exts=[".csv",".txt",".tsv"])
                rh_out_dir = ref / "_Processed" / "Rheology"
                rh_outs = [rh_out_dir / "output_viscosity.json", rh_out_dir / "output_viscosity.csv"]
                if args.skip_fresh and is_up_to_date(rh_in, rh_outs):
                    rec["rheology"] = {"status": "skipped_up_to_date", "out": str(rh_out_dir)}
                else:
                    run_rheology_standalone(args.py, base_dir, ref, args)
                    rec["rheology"] = {"status": "called_standalone", "out": str(rh_out_dir)}

            # --- DataGraph writer (optional) ---
            if args.write_dg:
                exports = run_datagraph_writer_for_ref(
                    ref, tpl_path, dg_cli_path, args.dg_restart_frequency)
                if exports:
                    rec["writer"] = {"status": "exported", "outputs": exports}
                else:
                    rec["writer"] = {"status": "skipped_no_pairs"}

        except subprocess.CalledProcessError as e:
            rec["error"] = f"SubprocessError: {e}"
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"

        rec["finished"] = datetime.now().isoformat()
        results.append(rec)

    # --- single combined DataGraph export (optional) ---
    if args.dg_single_file:
        try:
            combined_export = run_single_datagraph(
                overfolder,
                refs,
                tpl_path,
                dg_cli_path,
                include_saxs=args.modalities in ("both", "saxs"),
                include_rheo=args.modalities in ("both", "rheology"),
                output_override=Path(args.dg_single_output).expanduser().resolve() if args.dg_single_output else None,
            )
        except Exception as exc:
            combined_error = f"{type(exc).__name__}: {exc}"

    # --- write summaries ---
    summary_json = overfolder / "_batch_summary.json"
    summary_csv  = overfolder / "_batch_summary.csv"
    summary_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["reference","saxs_status","rheology_status","writer_status","error","started","finished"])
        for r in results:
            w.writerow([
                r["reference"],
                (r["saxs"] or {}).get("status") if r.get("saxs") else "",
                (r["rheology"] or {}).get("status") if r.get("rheology") else "",
                (r["writer"] or {}).get("status") if r.get("writer") else "",
                "OK" if not r.get("error") else "ERROR",
                r["started"], r["finished"]
            ])

    if args.dg_single_file:
        combined_meta = {"output": combined_export, "error": combined_error}
        single_json = overfolder / "_batch_single_datagraph.json"
        single_json.write_text(json.dumps(combined_meta, indent=2), encoding="utf-8")
        if combined_export:
            print(f"[orchestrate] Single DataGraph metadata → {single_json}")

    # --- exit status ---
    had_error = any(r.get("error") for r in results) or (args.dg_single_file and combined_error)
    print(f"[orchestrate] Summary JSON → {summary_json}")
    print(f"[orchestrate] Summary CSV  → {summary_csv}")
    if args.dg_single_file:
        if combined_export:
            print(f"[orchestrate] Combined DataGraph → {combined_export}")
        else:
            print("[orchestrate] Combined DataGraph failed or had no inputs.")
    if had_error:
        print("[orchestrate] One or more references failed. See summary.", file=sys.stderr)
        sys.exit(1)
    print("[orchestrate] Done.")

if __name__ == "__main__":
    main()
