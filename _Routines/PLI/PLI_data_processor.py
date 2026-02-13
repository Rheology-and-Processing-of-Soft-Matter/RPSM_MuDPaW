import os
import sys
import numpy as np
import pandas as pd

import imageio.v3 as iio
import cv2 as cv2

import json
import argparse     
import matplotlib.pyplot as plt
from typing import List, Tuple
from scipy.optimize import curve_fit
from Time_stamper_PLI_v1 import compute_pair_widths_from_times

try:
    import LabHWHM as DataAnalyzer
except Exception:
    DataAnalyzer = None
try:
    from LabHWHM import HWHM
except Exception:
    HWHM = None


import re

# --- Analyzer window guards (prevent duplicate openings) ---
_ANALYZER_ACTIVE = False
_ANALYZER_WIN = None

# --- Path hygiene ---

def _sanitize_misjoined_user_path(p: str) -> str:
    """If `p` contains a second absolute '/Users/...' segment (e.g.,
    '/repo/.../RPSM_MuDPaW_v5/Users/kroland/...'), trim to the second one.
    If nothing to fix, return `p` unchanged."""
    try:
        tok = os.sep + "Users" + os.sep
        first = p.find(tok)
        if first == -1:
            return p
        second = p.find(tok, first + 1)
        if second == -1:
            return p
        fixed = p[second:]
        if not fixed.startswith(os.sep):
            fixed = os.sep + fixed
        if fixed != p:
            print(f"[PLI] sanitize(processor): misjoined path →\n  in : {p}\n  out: {fixed}")
        return fixed
    except Exception:
        return p

# Natural sort key for filenames like img1.tif, img2.tif, img10.tif
def _natural_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

# Pick a row from an image robustly
def _pick_row(img: np.ndarray, center_line: bool, line_index: int | None) -> np.ndarray:
    if line_index is not None:
        idx = max(0, min(img.shape[0]-1, int(line_index)))
        return img[idx, :]
    # line_index is None → default to center line regardless of center_line flag
    return img[img.shape[0] // 2, :]

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts

def get_reference_folder_from_path(path):
    """
    Resolve the *reference* folder robustly.
    Walks up the path to find known modality markers ("PLI", "PI", "SAXS", "Rheology").
    If the marker's parent is _Processed, returns one level above _Processed (the true reference).
    Ensures absolute path with leading slash.
    Fallback: two levels up from the provided path (legacy behavior).
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    abspath = _sanitize_misjoined_user_path(os.path.abspath(path))
    parts = _split_parts(abspath)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            # If parent is _Processed, skip it to get the true reference root
            if i - 1 >= 0 and parts[i - 1] == "_Processed":
                reference = os.sep.join(parts[: i - 1])
            else:
                reference = os.sep.join(parts[:i])
            reference = _sanitize_misjoined_user_path(reference)
            reference = os.path.abspath(reference)
            if not reference.startswith(os.sep):
                reference = os.sep + reference
            return reference if reference else os.sep

    # Fallback for legacy layouts: two levels up
    ref = os.path.dirname(os.path.dirname(abspath))
    ref = _sanitize_misjoined_user_path(ref)
    ref = os.path.abspath(ref)
    if not ref.startswith(os.sep):
        ref = os.sep + ref
    return ref

def get_unified_processed_folder(path):
    """Return the unified `_Processed/PLI` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed", "PLI")
    processed_root = _sanitize_misjoined_user_path(processed_root)
    processed_root = os.path.abspath(processed_root)
    if not processed_root.startswith(os.sep):
        processed_root = os.sep + processed_root
    os.makedirs(processed_root, exist_ok=True)
    return processed_root


# --- Frame & interval utilities ---

def list_tifs_sorted(folder_path: str) -> List[str]:
    files = [f for f in os.listdir(folder_path) if f.lower().endswith(".tif") and not f.startswith(".")]
    files.sort(key=_natural_key)
    return [os.path.join(folder_path, f) for f in files]



def select_frames_by_widths(n_frames: int, widths: List[int], end_aligned: bool = True) -> Tuple[List[int], List[Tuple[int,int]]]:
    """
    Given total frames and a list of widths [w_last_T, w_last_S, ..., w_first_T, w_first_S] (end-aligned order),
    return the list of frame indices to keep (sorted ascending) and a list of segment (start,end) indices per interval
    pair in chronological order (first pair → last pair). Each (start,end) is inclusive of start and exclusive of end
    in 0-based frame indices.
    """
    w = [max(0, int(x)) for x in widths]
    total = sum(w)
    if total == 0 or n_frames == 0:
        return [], []

    if end_aligned:
        # consume from the tail
        start_idx = max(0, n_frames - total)
        # Build segments from the end backwards
        cursor = n_frames
        segments_rev: List[Tuple[int,int]] = []
        # group by pairs as [T_last, S_last], [T_prev, S_prev], ...
        for i in range(0, len(w), 2):
            wt = w[i]
            ws = w[i+1] if i+1 < len(w) else 0
            # steady segment (even) is last of the pair
            s_end = cursor
            s_start = max(start_idx, s_end - ws)
            # transient before steady
            t_end = s_start
            t_start = max(start_idx, t_end - wt)
            # store as (T_start,T_end),(S_start,S_end) for this pair
            segments_rev.append((t_start, t_end))
            segments_rev.append((s_start, s_end))
            cursor = t_start
        # reverse to chronological order: first pair at the front
        segments = list(reversed(segments_rev))
        # collapse to frame indices to keep
        keep = []
        for a, b in segments:
            keep.extend(range(a, b))
        return sorted(set(keep)), segments
    else:
        # consume from the head, order provided from first→last pair
        cursor = 0
        segments: List[Tuple[int,int]] = []
        for i in range(0, len(w), 2):
            wt = w[i]
            ws = w[i+1] if i+1 < len(w) else 0
            t_start = cursor
            t_end = min(n_frames, t_start + wt)
            s_start = t_end
            s_end = min(n_frames, s_start + ws)
            segments.append((t_start, t_end))
            segments.append((s_start, s_end))
            cursor = s_end
        keep = []
        for a, b in segments:
            keep.extend(range(a, b))
        return sorted(set(keep)), segments


def _segments_from_intervals(height: int, cfg: dict, reverse_from_end: bool = True):
    """Build (start,end) segments from widths or segments for a stitched image of given height."""
    if not isinstance(cfg, dict):
        return [(0, height)]
    segs = cfg.get("segments")
    if isinstance(segs, list) and segs and isinstance(segs[0], (list, tuple)):
        return [(max(0, int(a)), min(height, int(b))) for a, b in segs if int(b) > int(a)]
    widths = cfg.get("widths") or []
    if isinstance(widths, list) and widths:
        _, segs2 = select_frames_by_widths(height, [int(x) for x in widths], end_aligned=reverse_from_end)
        return segs2 or [(0, height)]
    return [(0, height)]


def extract_space_time_with_intervals(folder_path: str, widths: List[int], end_aligned: bool = True, center_line: bool = True, line_index: int | None = None) -> Tuple[np.ndarray, dict]:
    """
    Extract a space–time diagram restricted to interval pairs defined by `widths`.
    - widths: list of ints (pixels/frames) in end-aligned order [T_last, S_last, ..., T_first, S_first] if end_aligned
    - end_aligned: if True, count widths from the *last* frame backwards
    - center_line: if True, use central horizontal line; else if line_index provided, use that row
    Returns (space_time_array, provenance_dict)
    """
    tif_paths = list_tifs_sorted(folder_path)
    if not tif_paths:
        raise RuntimeError("No .tif frames found in folder")

    keep_idx, segments = select_frames_by_widths(len(tif_paths), widths, end_aligned=end_aligned)
    total_requested = sum(int(x) for x in widths)
    if total_requested > len(tif_paths):
        print(f"Warning: requested {total_requested} frames by widths but only {len(tif_paths)} frames exist. Result will be capped from the start (end-aligned).")

    if not keep_idx:
        raise RuntimeError("No frames selected by provided interval widths")

    frames = []
    for idx in keep_idx:
        try:
            img = iio.imread(tif_paths[idx])
            row = _pick_row(img, center_line=center_line, line_index=line_index)
            frames.append(row)
        except Exception as e:
            print(f"Error reading frame {idx} ({os.path.basename(tif_paths[idx])}): {e}")
    space_time = np.array(frames)

    prov = {
        "n_input_frames": len(tif_paths),
        "n_kept_frames": len(keep_idx),
        "kept_indices": keep_idx,
        "segments": segments,
        "end_aligned": end_aligned,
        "widths": widths,
    }
    return space_time, prov



# --- Gaussian + baseline model and fitting helper (for stitched profiles) ---
def _gauss_baseline(x: np.ndarray, A: float, x0: float, sigma: float, B: float) -> np.ndarray:
    return A * np.exp(-((x - x0) ** 2) / (2.0 * (sigma ** 2 + 1e-12))) + B


def _fit_profile_gaussian(x: np.ndarray, y: np.ndarray) -> tuple[dict, np.ndarray | None]:
    """Fit y(x) with Gaussian + baseline. Returns (params_dict, y_fit) where params has keys A,x0,sigma,B,success,rmse."""
    # Initial guesses
    y_min = float(np.min(y))
    y_max = float(np.max(y))
    A0 = max(1e-9, y_max - y_min)
    x0_0 = float(x[int(np.argmax(y))])
    sigma0 = 0.1  # in normalized units
    B0 = y_min
    p0 = [A0, x0_0, sigma0, B0]
    bounds = ([0.0, 0.0, 1e-4, -np.inf], [np.inf, 1.0, 0.5, np.inf])
    try:
        popt, pcov = curve_fit(_gauss_baseline, x, y, p0=p0, bounds=bounds, maxfev=20000)
        A, x0, sigma, B = [float(v) for v in popt]
        y_fit = _gauss_baseline(x, A, x0, sigma, B)
        rmse = float(np.sqrt(np.mean((y_fit - y) ** 2)))
        return {"A": A, "x0": x0, "sigma": sigma, "B": B, "success": True, "rmse": rmse}, y_fit
    except Exception as e:
        return {"A": np.nan, "x0": np.nan, "sigma": np.nan, "B": np.nan, "success": False, "rmse": np.nan, "error": str(e)}, None


# --- Helper: run stitched-PNG processing/export (profiles + fits + JSON) ---
def run_stitched_processing(path: str, reverse_from_end: bool = True, intervals_cfg: dict | None = None) -> tuple[str, str]:
    """Process a stitched PNG and export profiles, fits and JSON.
    Returns (profiles_csv, json_path)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    sample_name = os.path.splitext(os.path.basename(path))[0]
    processed_folder = get_unified_processed_folder(path)
    processed_pli = os.path.join(processed_folder, "PLI")
    os.makedirs(processed_pli, exist_ok=True)

    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Cannot read stitched PNG: {path}")

    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = L.astype(np.float32)
    a = a.astype(np.float32) - 128.0
    b = b.astype(np.float32) - 128.0
    M = np.sqrt(a * a + b * b)
    H, W = M.shape

    # Build x-intervals
    x_segments = []
    interval_width = 300
    if isinstance(intervals_cfg, dict):
        # Preferred: explicit widths or n_intervals
        widths = intervals_cfg.get("interval_widths_px") if isinstance(intervals_cfg.get("interval_widths_px"), list) else None
        if widths:
            cur = 0
            for w in widths:
                w_int = int(max(1, w))
                x1 = min(W, cur + w_int)
                x_segments.append((cur, x1))
                cur = x1
            if not x_segments:
                widths = None  # fallback below
        if not widths:
            n_int = int(intervals_cfg.get("n_intervals", 0))
            if n_int > 0:
                interval_width = int(math.ceil(W / float(n_int)))
        # Fallback to interval_width hint
        interval_width = int(max(1, intervals_cfg.get('interval_width', interval_width)))
    else:
        interval_width = 300
    if not x_segments:
        x0 = 0
        while x0 < W:
            x1 = min(W, x0 + interval_width)
            x_segments.append((x0, x1))
            x0 = x1

    profiles = []              # (k, x0, x1, prof_raw_vs_y)
    profiles_corr = []         # (k, x0, x1, prof_corr_vs_y)
    fit_params = []
    fit_curves = []            # y_fit on y_norm
    fit_curves_corr = []       # y_fit - B
    y_norm = np.linspace(0.0, 1.0, H, dtype=np.float32)
    for k, (x0, x1) in enumerate(x_segments):
        prof = np.mean(M[:, x0:x1], axis=1)  # average across columns → profile vs y
        params, y_fit = _fit_profile_gaussian(y_norm, prof)
        B = float(params.get("B", 0.0)) if params else 0.0
        prof_corr = prof - B
        y_fit_corr = (y_fit - B) if y_fit is not None else None
        profiles.append((k, x0, x1, prof))
        profiles_corr.append((k, x0, x1, prof_corr))
        fit_params.append({"interval": int(k), "x0": int(x0), "x1": int(x1), **params})
        fit_curves.append(y_fit)
        fit_curves_corr.append(y_fit_corr)

    if not profiles:
        raise RuntimeError('No valid intervals; no profiles created.')

    angles_deg = np.linspace(0.0, 360.0, H, endpoint=False, dtype=np.float32)
    df = pd.DataFrame({
        "angle_deg": angles_deg,
        "y_norm": y_norm
    })
    for idx, (k, x0, x1, prof_raw) in enumerate(profiles):
        df[f'interval_{k}_raw_x{x0}_{x1}'] = prof_raw
        if fit_curves[idx] is not None:
            df[f'interval_{k}_raw_fit'] = fit_curves[idx]
    for idx, (k, x0, x1, prof_corr) in enumerate(profiles_corr):
        df[f'interval_{k}_corr_x{x0}_{x1}'] = prof_corr
        if fit_curves_corr[idx] is not None:
            df[f'interval_{k}_corr_fit'] = fit_curves_corr[idx]

    out_csv = os.path.join(processed_pli, f'_pli_{sample_name}_stitched_profiles.csv')
    df.to_csv(out_csv, index=False)
    fits_csv = os.path.join(processed_pli, f'_pli_{sample_name}_stitched_fit_params.csv')
    pd.DataFrame(fit_params).to_csv(fits_csv, index=False)

    json_path = os.path.join(processed_pli, f'_output_PLI_{sample_name}.json')
    # Derive relative suffixes after '/_Processed/' when possible
    _rel_profiles = out_csv.split('/_Processed/', 1)[1] if '/_Processed/' in out_csv else os.path.basename(out_csv)
    _rel_fits = fits_csv.split('/_Processed/', 1)[1] if '/_Processed/' in fits_csv else os.path.basename(fits_csv)
    meta = {
        'mode': 'stitched_png',
        'sample_name': sample_name,
        'processed_folder': processed_folder,
        'stitched_png': os.path.abspath(path),
        'profiles_csv': out_csv,
        'profiles_csv_rel': _rel_profiles,
        'intervals': {'x_segments': x_segments, 'interval_width': interval_width},
        'parameter': 'sqrt(a^2+b^2)',
        'fits_csv': fits_csv,
        'fits_csv_rel': _rel_fits,
        'fit_model': 'gaussian+baseline',
        'x_axis': 'angle_deg',
        'baseline_correction': 'subtract fitted B per interval',
    }
    with open(json_path, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"Saved stitched profiles (with fits): {out_csv}")
    print(f"PLI output JSON written: {json_path}")
    return out_csv, json_path


def _find_unscaled_stitched_in_temp(any_path: str) -> str | None:
    """Given any path under a dataset, return the newest unscaled stitched PNG from <reference>/PLI/_Temp.
    Returns None if not found."""
    try:
        ref = get_reference_folder_from_path(any_path)
        temp_dir = os.path.join(ref, "PLI", "_Temp")
        if not os.path.isdir(temp_dir):
            return None
        cand = [
            os.path.join(temp_dir, f) for f in os.listdir(temp_dir)
            if f.lower().endswith('.png') and '_steady_unscaled_stitched_' in f.lower() and not f.startswith('.')
        ]
        if not cand:
            return None
        # pick newest by mtime
        cand.sort(key=lambda p: os.path.getmtime(p))
        return cand[-1]
    except Exception:
        return None


def _prefer_unscaled_stitched_path(path: str) -> tuple[str, bool]:
    """If `path` is a folder (or not a stitched png), try to locate the latest
    unscaled stitched PNG in <reference>/PLI/_Temp and return it.
    Returns (resolved_path, is_png)."""
    if os.path.isfile(path) and path.lower().endswith('.png'):
        return path, True
    # If a folder or non-png file: try to find the temp unscaled stitched panel
    best = _find_unscaled_stitched_in_temp(path)
    if best:
        print(f"[PLI] Using unscaled stitched panel from Temp: {best}")
        return best, True
    return path, (os.path.isfile(path) and path.lower().endswith('.png'))


def main():
    parser = argparse.ArgumentParser(description="PLI data processor: build space–time diagrams or analyze stitched PNGs.")
    parser.add_argument("path", help="Folder with .tif frames OR a stitched .png file")
    parser.add_argument("sample_name", nargs="?", help="Sample name (optional for stitched .png; defaults to file stem)")
    parser.add_argument("--intervals", help="Comma-separated widths (end-aligned: T_last,S_last,...,T_first,S_first)")
    parser.add_argument("--intervals_json", help="Path to JSON containing {'widths': [...], 'end_aligned': true}")
    # deprecated --end_aligned argument removed
    parser.add_argument("--line_index", type=int, default=None, help="Row index to extract instead of center line")
    parser.add_argument("--no_center", action="store_true", help="Disable center-line selection (requires --line_index)")
    parser.add_argument("--time_source", help="Path to SAXS timestamps or Anton Paar export to derive interval pairs")
    parser.add_argument("--fps", type=float, default=None, help="Frame rate (frames/s) to convert time to frames when --time_source is used")
    parser.add_argument("--reverse_from_end", dest="reverse_from_end", action="store_true", help="Count interval pairs from the end of the image (default)")
    parser.add_argument("--from_start", dest="reverse_from_end", action="store_false", help="Count interval pairs from the start (head-aligned)")
    parser.set_defaults(reverse_from_end=True)
    parser.add_argument("--enforce_constant_even", action="store_true", help="If set, enforce constant width across steady (even) intervals; will fail if not constant")

    args = parser.parse_args()
    path = args.path
    sample_name = args.sample_name

    # Prefer the unscaled stitched panel in <reference>/PLI/_Temp when present
    path, is_png = _prefer_unscaled_stitched_path(path)
    is_file = os.path.isfile(path)

    if is_png and not sample_name:
        sample_name = os.path.splitext(os.path.basename(path))[0]

    if is_png:
        # Open setup/analyzer BEFORE processing; ENGAGE inside the window can call run_stitched_processing()
        global _ANALYZER_ACTIVE, _ANALYZER_WIN
        try:
            if _ANALYZER_ACTIVE and _ANALYZER_WIN is not None:
                # If a window already exists, just lift/focus it
                try:
                    if _ANALYZER_WIN.winfo_exists():
                        print("[PLI] Analyzer already open; focusing existing window…")
                        _ANALYZER_WIN.lift(); _ANALYZER_WIN.focus_force()
                        return
                except Exception:
                    pass
            print("[PLI] Opening analyzer/setup before processing…")
            open_maltese_cross_analyzer(path, interval_height=300)
        finally:
            return


    processed_folder = get_unified_processed_folder(path)
    processed_pli = os.path.join(processed_folder, "PLI")
    os.makedirs(processed_pli, exist_ok=True)
    print(f"Unified _Processed folder for PLI outputs: {processed_folder}")

    # Parse widths if provided
    widths = None

    if args.intervals:
        try:
            widths = [int(x.strip()) for x in args.intervals.split(",") if x.strip()]
        except Exception as e:
            print(f"Invalid --intervals: {e}")
            sys.exit(1)
    elif args.intervals_json:
        try:
            with open(args.intervals_json, "r") as f:
                cfg = json.load(f)
            widths = [int(x) for x in cfg.get("widths", [])]
            # end_aligned = bool(cfg.get("end_aligned", True))  # no longer used
        except Exception as e:
            print(f"Invalid --intervals_json: {e}")
            sys.exit(1)

    # If a time source is provided, compute widths directly here
    if args.time_source:
        if not args.fps:
            print("Error: --fps is required when --time_source is provided (to convert seconds → frames).")
            sys.exit(1)
        try:
            pair_info = compute_pair_widths_from_times(args.time_source, args.fps, reverse_from_end=args.reverse_from_end)
            widths = pair_info.get("widths", [])
            # Optionally enforce constant even-interval widths
            if args.enforce_constant_even:
                even_w = pair_info.get("even_widths", [])
                if not even_w or not all(w == even_w[0] for w in even_w):
                    print("Error: even (steady) interval widths are not constant; remove --enforce_constant_even or ensure constant export.")
                    sys.exit(1)
        except Exception as e:
            print(f"Failed to compute widths from time source: {e}")
            sys.exit(1)

    try:
        if widths:
            st, prov = extract_space_time_with_intervals(
                path,
                widths=widths,
                end_aligned=args.reverse_from_end,
                center_line=(not args.no_center),
                line_index=args.line_index,
            )
        else:
            st = extract_space_time_diagram(path, center_line=(not args.no_center), line_index=args.line_index)
            prov = {"n_input_frames": st.shape[0], "n_kept_frames": st.shape[0], "segments": [], "end_aligned": None, "widths": None}

        # Save outputs
        output_csv = os.path.join(processed_pli, f"_pli_{sample_name}_output.csv")
        pd.DataFrame(st).to_csv(output_csv, index=False)
        print(f"Saved space-time diagram: {output_csv}")

        json_path = os.path.join(processed_pli, f"_output_PLI_{sample_name}.json")
        _rel_csv = output_csv.split('/_Processed/', 1)[1] if '/_Processed/' in output_csv else os.path.basename(output_csv)
        meta = {
            "sample_name": sample_name,
            "processed_folder": processed_folder,
            "csv_output": output_csv,
            "csv_output_rel": _rel_csv,
            "intervals": prov,
        }
        if args.time_source:
            meta["time_source"] = os.path.abspath(args.time_source)
            meta["fps_used"] = float(args.fps)
            meta["reverse_from_end"] = bool(args.reverse_from_end)
        with open(json_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"PLI output JSON written: {json_path}")
    except Exception as e:
        print(f"Failed to process space-time diagram: {e}")
        sys.exit(2)

# (moved) main() is invoked at end of file after all GUI helpers are defined


import tkinter as _tk
from tkinter import ttk as _ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as _np
import cv2 as _cv2


def _load_lab_arrays(_png_path: str):
    img_bgr = _cv2.imread(_png_path, _cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {_png_path}")
    lab = _cv2.cvtColor(img_bgr, _cv2.COLOR_BGR2LAB)
    L, a, b = _cv2.split(lab)
    L = L.astype(_np.float32)
    a = a.astype(_np.float32) - 128.0
    b = b.astype(_np.float32) - 128.0
    return L, a, b  # shapes: (H, W)


def _compute_param(L: _np.ndarray, a: _np.ndarray, b: _np.ndarray, mode: str) -> _np.ndarray:
    mode = (mode or "L").lower()
    if mode == "l":
        return L
    if mode == "a":
        return a
    if mode == "b":
        return b
    if mode in ("sqrt(a^2+b^2)", "chroma", "ab_mag"):
        return _np.sqrt(a * a + b * b)
    if mode in ("sqrt(l^2+a^2+b^2)", "lab_mag"):
        return _np.sqrt(L * L + a * a + b * b)
    # default
    return L


def open_pli_distribution_view(stitched_path: str, interval_height: int = 300):
    """Open an interactive window to plot parameter distributions from a stitched PLI space–time image.

    - `stitched_path`: path to a color `_stitched_*.png` file (gray variants are not supported here)
    - `interval_height`: vertical height (in pixels) used to segment the image into intervals (default 300)
    The plot shows the column-wise mean of the selected parameter over the chosen vertical interval.
    """
    if not os.path.isfile(stitched_path):
        raise FileNotFoundError(stitched_path)
    if "_stitched_" not in os.path.basename(stitched_path).lower():
        print("Warning: selected image does not look like a stitched output; proceeding anyway.")

    L, a, b = _load_lab_arrays(stitched_path)
    H, W = L.shape
    n_intervals = max(1, int(_np.ceil(H / float(max(1, int(interval_height))))))

    win = _tk.Toplevel()
    win.title("PLI Distribution Analysis")
    win.geometry("820x620")

    top_frm = _ttk.Frame(win)
    top_frm.pack(fill=_tk.X, padx=8, pady=6)

    _ttk.Label(top_frm, text=os.path.basename(stitched_path)).pack(side=_tk.LEFT)

    # Parameter chooser
    _ttk.Label(top_frm, text="  Parameter:").pack(side=_tk.LEFT)
    param_cmb = _ttk.Combobox(top_frm, values=[
        "L", "a", "b", "sqrt(a^2+b^2)", "sqrt(L^2+a^2+b^2)"
    ], state="readonly", width=22)
    param_cmb.set("L")
    param_cmb.pack(side=_tk.LEFT, padx=6)

    # Baseline toggle
    base_var = _tk.BooleanVar(value=True)
    _ttk.Checkbutton(top_frm, text="Baseline subtract (fit B)", variable=base_var).pack(side=_tk.LEFT, padx=8)

    # Interval slider
    mid_frm = _ttk.Frame(win)
    mid_frm.pack(fill=_tk.X, padx=8, pady=(0, 6))
    _ttk.Label(mid_frm, text="Interval index (height≈{} px):".format(int(interval_height))).pack(side=_tk.LEFT)
    interval_var = _tk.IntVar(value=0)
    interval_scl = _tk.Scale(mid_frm, from_=0, to=max(0, n_intervals - 1), orient=_tk.HORIZONTAL, variable=interval_var, length=400)
    interval_scl.pack(side=_tk.LEFT, padx=8)

    # Figure
    fig = Figure(figsize=(7.6, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasTkAgg(fig, master=win)
    canvas.get_tk_widget().pack(fill=_tk.BOTH, expand=True, padx=8, pady=4)

    # Bottom buttons
    btn_frm = _ttk.Frame(win)
    btn_frm.pack(fill=_tk.X, padx=8, pady=6)

    # compute + plot
    def _update_plot(*_):
        mode = param_cmb.get()
        M = _compute_param(L, a, b, mode)
        k = max(0, min(n_intervals - 1, int(interval_var.get())))
        y0 = int(k * interval_height)
        y1 = int(min(H, y0 + interval_height))
        if y1 <= y0:
            return
        prof = _np.mean(M[y0:y1, :], axis=0)
        x_norm = _np.linspace(0.0, 1.0, prof.size)
        if base_var.get():
            params, y_fit = _fit_profile_gaussian(x_norm, prof)
            B = float(params.get("B", 0.0)) if params else 0.0
            prof_plot = prof - B
        else:
            prof_plot = prof
        angles = _np.linspace(0.0, 360.0, prof.size, endpoint=False)
        ax.clear()
        ax.plot(angles, prof_plot)
        ax.set_xlabel("Azimuthal angle [°]")
        ax.set_ylabel(f"{mode} [a.u.]{' (B-subtracted)' if base_var.get() else ''}")
        ax.grid(True, which="both", alpha=0.3)
        canvas.draw_idle()

    def _save_csv():
        mode = param_cmb.get()
        k = max(0, min(n_intervals - 1, int(interval_var.get())))
        y0 = int(k * interval_height)
        y1 = int(min(H, y0 + interval_height))
        M = _compute_param(L, a, b, mode)
        prof = _np.mean(M[y0:y1, :], axis=0)
        x_norm = _np.linspace(0.0, 1.0, prof.size)
        params, y_fit = _fit_profile_gaussian(x_norm, prof)
        B = float(params.get("B", 0.0)) if params else 0.0
        prof_corr = prof - B
        angles = _np.linspace(0.0, 360.0, prof.size, endpoint=False)
        out_root = get_unified_processed_folder(stitched_path)  # already .../_Processed/PLI
        out_dir = out_root
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(stitched_path))[0]
        out_csv = os.path.join(out_dir, f"_pli_{base}_interval{k}_{mode.replace('/', '').replace('^', '').replace('*', '')}.csv")
        pd.DataFrame({
            "angle_deg": angles,
            "x_norm": x_norm,
            "value_raw": prof,
            "value_corr": prof_corr,
            "B": B
        }).to_csv(out_csv, index=False)
        print(f"Saved distribution CSV: {out_csv}")

    _ttk.Button(btn_frm, text="Update", command=_update_plot).pack(side=_tk.LEFT)
    _ttk.Button(btn_frm, text="Save CSV", command=_save_csv).pack(side=_tk.LEFT, padx=6)

    # reactive updates
    param_cmb.bind("<<ComboboxSelected>>", _update_plot)
    interval_scl.configure(command=lambda *_: _update_plot())
    base_var.trace_add("write", lambda *_: _update_plot())

    _update_plot()
    win.lift()
    win.focus_force()   


# --- Maltese-cross analyzer ---

import math as _math

# --- Helper: fit Gaussian+baseline with x in degrees (for Maltese cross analysis) ---
def _fit_profile_gaussian_deg(x_deg: _np.ndarray, y: _np.ndarray):
    """Fit Gaussian+baseline with x in degrees. Internally normalizes to x_norm=x_deg/360.
    Returns (params_dict, y_fit) where params has A, x0 (deg), sigma (deg), B, success, rmse.
    """
    x_deg = _np.asarray(x_deg, dtype=_np.float64)
    y = _np.asarray(y, dtype=_np.float64)
    # Normalize degrees to [0,1] scale expected by _fit_profile_gaussian
    x_norm = (x_deg % 360.0) / 360.0
    # Guard against degenerate ranges
    if not _np.isfinite(x_norm).all() or x_norm.size < 3:
        return {"A": _np.nan, "x0": _np.nan, "sigma": _np.nan, "B": _np.nan, "success": False, "rmse": _np.nan, "error": "insufficient data"}, None
    pars, y_fit = _fit_profile_gaussian(x_norm, y)
    # Convert center and sigma back to degrees
    if pars and pars.get("success", False):
        pars = dict(pars)
        pars["x0"] = float(pars["x0"]) * 360.0
        pars["sigma"] = float(pars["sigma"]) * 360.0
    return pars, y_fit


def _find_shear_rate_column(_df: pd.DataFrame):
    keys = ["shear", "gamma", "γ", "sr", "shearrate", "shear_rate", "shear-rate"]
    for c in _df.columns:
        cl = str(c).lower()
        if any(k in cl for k in keys):
            return c
    return None

# --- External Maltese-cross analysis hook ---

def _call_external_maltese_analysis(
    stitched_path: str,
    n_intervals: int,
    trimmed_payload: dict | None,
    *,
    preview: bool,
):
    """Unified hook to call into the user's HWHM module.
    - If preview=True: returns Data_for_preview_window from HWHM.ExPLIDat(stitched_path, n_intervals).
    - If preview=False: expects `trimmed_payload` (dict) and returns Final_PLI_data from HWHM.LabHWHM(trimmed_payload).
    On any failure or if HWHM is unavailable, returns None.
    """
    if HWHM is None:
        print("[PLI] HWHM module not available (LabHWHM import failed). Skipping external analysis.")
        return None
    try:
        if preview:
            [trim_L, trim_a, trim_b] = HWHM.ExPLIDat(stitched_path, n_intervals)
        else:
            if trim_L is None:
                print("[PLI] No trimmed payload provided for final HWHM analysis.")
                return None
            Final_data = HWHM.LabHWHM(trimmed_payload)
    except Exception as e:
        what = "ExPLIDat" if preview else "LabHWHM"
        print(f"[PLI] HWHM.{what} failed: {e}")
        return None


def open_maltese_cross_analyzer(stitched_path: str, interval_height: int = 300):
    """Interactive tool to detect/quantify Maltese-cross peak across intervals.

    - Baseline correction: fit B on a chosen baseline interval and subtract it from *all* intervals.
    - Constrain analysis to an azimuthal window (one peak) using two angle sliders.
    - Metric per interval: HWHM (deg) from Gaussian fit (and store x0, sigma, A, rmse).
    """
    if not os.path.isfile(stitched_path):
        raise FileNotFoundError(stitched_path)

    global _ANALYZER_ACTIVE, _ANALYZER_WIN
    # Reuse existing window if it's still alive
    if _ANALYZER_WIN is not None:
        try:
            if _ANALYZER_WIN.winfo_exists():
                _ANALYZER_WIN.lift(); _ANALYZER_WIN.focus_force()
                return
        except Exception:
            pass
    _ANALYZER_ACTIVE = True

    L, a, b = _load_lab_arrays(stitched_path)
    H, W = L.shape
    # Attempt to load sidecar metadata to honor custom interval counts/widths
    x_segments = []
    interval_width = int(max(1, interval_height))  # reuse arg as width (300 by default)
    meta_path = stitched_path + ".json"
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, "r") as f:
                meta = json.load(f)
            widths = meta.get("interval_widths_px") if isinstance(meta.get("interval_widths_px"), list) else None
            n_int = int(meta.get("n_intervals", 0)) if meta else 0
            if widths:
                cur = 0
                for w in widths:
                    w_int = int(max(1, w))
                    x1 = min(W, cur + w_int)
                    x_segments.append((cur, x1))
                    cur = x1
            elif n_int > 0:
                interval_width = int(math.ceil(W / float(n_int)))
        except Exception as _e:
            print(f"[PLI] analyzer meta load failed ({meta_path}): {_e}")

    if not x_segments:
        x0 = 0
        while x0 < W:
            x1 = min(W, x0 + interval_width)
            x_segments.append((x0, x1))
            x0 = x1
    n_intervals = len(x_segments)
    print(f"[PLI] Analyzer using {n_intervals} x-intervals of ~{interval_width}px")

    # --- Insert: External LabHWHM precompute for per-interval Lab arrays ---
    try:
        L_mat, a_mat, b_mat = DataAnalyzer.LabHWHM(stitched_path, n_intervals)
    except Exception as e:
        print(f"[PLI] LabHWHM failed; falling back to local LAB computation: {e}")
        # fallback: recompute locally with ±128 a,b centering
        lab_local = _cv2.cvtColor(_cv2.imread(stitched_path, _cv2.IMREAD_COLOR), _cv2.COLOR_BGR2LAB)
        _L, _a, _b = _cv2.split(lab_local)
        L_mat = _np.tile(_L.astype(_np.float32), (1, n_intervals))
        a_mat = _np.tile((_a.astype(_np.float32) - 128.0), (1, n_intervals))
        b_mat = _np.tile((_b.astype(_np.float32) - 128.0), (1, n_intervals))

    def _param_matrix(mode: str) -> _np.ndarray:
        """Return HxN matrix for the selected parameter."""
        mode = (mode or "sqrt(a^2+b^2)").lower()
        if mode in ("l",):
            return _np.asarray(L_mat, dtype=_np.float32)
        if mode in ("a",):
            return _np.asarray(a_mat, dtype=_np.float32)
        if mode in ("b",):
            return _np.asarray(b_mat, dtype=_np.float32)
        if mode in ("sqrt(a^2+b^2)", "ab", "chroma", "ab_mag"):
            return _np.sqrt(a_mat * a_mat + b_mat * b_mat, dtype=_np.float32)
        if mode in ("sqrt(l^2+a^2+b^2)", "lab", "lab_mag"):
            return _np.sqrt(L_mat * L_mat + a_mat * a_mat + b_mat * b_mat, dtype=_np.float32)
        # default to chroma
        return _np.sqrt(a_mat * a_mat + b_mat * b_mat, dtype=_np.float32)

    win = _tk.Toplevel()
    win.title("Maltese Cross Analyzer")
    win.geometry("920x720")
    _ANALYZER_WIN = win
    # State for ENGAGE results
    _ax2_bot = {"axis": None}
    _fit_store = {"x": None, "fits": None, "r2": None, "cols": None}
    def _on_close():
        global _ANALYZER_ACTIVE, _ANALYZER_WIN
        try:
            _ANALYZER_ACTIVE = False
            _ANALYZER_WIN = None
        finally:
            win.destroy()
    win.protocol("WM_DELETE_WINDOW", _on_close)

    # Top controls (two rows): row1=file name, row2=parameter/baseline controls
    top = _ttk.Frame(win); top.pack(fill=_tk.X, padx=8, pady=6)
    file_fr = _ttk.Frame(top); file_fr.pack(fill=_tk.X)
    _ttk.Label(file_fr, text=os.path.basename(stitched_path), anchor="w").pack(side=_tk.LEFT)

    ctrl_fr = _ttk.Frame(top); ctrl_fr.pack(fill=_tk.X, pady=(4, 0))
    # Row: parameter
    param_fr = _ttk.Frame(ctrl_fr); param_fr.pack(fill=_tk.X)
    _ttk.Label(param_fr, text="Parameter:").pack(side=_tk.LEFT)
    param_cmb = _ttk.Combobox(param_fr, values=["Lab","ab","L","a","b"], state="readonly", width=22)
    param_cmb.set("Lab")
    param_cmb.pack(side=_tk.LEFT, padx=6)
    param_cmb.bind("<<ComboboxSelected>>", lambda *_: _analyze(preview_only=True))

    # Row: fit model
    model_fr = _ttk.Frame(ctrl_fr); model_fr.pack(fill=_tk.X, pady=(4, 0))
    _ttk.Label(model_fr, text="Fit model:").pack(side=_tk.LEFT)
    fit_model_var = _tk.StringVar(value="No model (exp HWHM)")
    fit_model_cmb = _ttk.Combobox(model_fr, values=["No model (exp HWHM)", "Gaussian", "Asym q-Gaussian"], state="readonly", width=20, textvariable=fit_model_var)
    fit_model_cmb.pack(side=_tk.LEFT, padx=(6, 0))
    fit_model_cmb.bind("<<ComboboxSelected>>", lambda *_: _analyze(preview_only=True))
    use_model_baseline_var = _tk.BooleanVar(value=True)
    _ttk.Checkbutton(model_fr, text="Fit baseline", variable=use_model_baseline_var,
                     command=lambda: _analyze(preview_only=True)).pack(side=_tk.LEFT, padx=8)

    # Row: baseline settings
    base_fr = _ttk.Frame(ctrl_fr); base_fr.pack(fill=_tk.X, pady=(4, 0))
    base_var = _tk.BooleanVar(value=False)
    base_chk = _ttk.Checkbutton(base_fr, text="Baseline correction", variable=base_var)
    base_chk.pack(side=_tk.LEFT, padx=8)
    _ttk.Label(base_fr, text="from interval:").pack(side=_tk.LEFT)
    base_idx = _tk.IntVar(value=0)
    base_idx_spn = _ttk.Spinbox(base_fr, from_=0, to=max(0,n_intervals-1), textvariable=base_idx, width=5, command=lambda: _analyze(preview_only=True))
    base_idx_spn.pack(side=_tk.LEFT, padx=(2,4))
    _ttk.Label(base_fr, text="method:").pack(side=_tk.LEFT)
    base_method_cmb = _ttk.Combobox(base_fr, values=["Subtract interval", "Subtract fit"], state="readonly", width=18)
    base_method_cmb.set("Subtract interval")
    base_method_cmb.pack(side=_tk.LEFT, padx=(2,8))
    base_method_cmb.bind("<<ComboboxSelected>>", lambda *_: _analyze(preview_only=True))
    base_var.trace_add("write", lambda *_: _analyze(preview_only=True))

    # Angle window (now normalized units)
    ang_frame = _ttk.Frame(win); ang_frame.pack(fill=_tk.X, padx=8, pady=(6,0))
    _ttk.Label(ang_frame, text="Angle window [normalized]:").pack(side=_tk.LEFT)
    ang0 = _tk.DoubleVar(value=0.0)
    ang1 = _tk.DoubleVar(value=1.0)
    _ttk.Scale(ang_frame, from_=0, to=1.0, orient=_tk.HORIZONTAL, length=260, variable=ang0, command=lambda *_: _analyze(preview_only=True)).pack(side=_tk.LEFT, padx=6)
    _ttk.Scale(ang_frame, from_=0, to=1.0, orient=_tk.HORIZONTAL, length=260, variable=ang1, command=lambda *_: _analyze(preview_only=True)).pack(side=_tk.LEFT, padx=6)

    # Interval range (use all by default)
    range_frame = _ttk.Frame(win); range_frame.pack(fill=_tk.X, padx=8, pady=(6,0))
    _ttk.Label(range_frame, text=f"Intervals (0..{n_intervals-1}) to analyze:").pack(side=_tk.LEFT)
    i0_var = _tk.IntVar(value=0)
    i1_var = _tk.IntVar(value=max(0,n_intervals-1))
    i0_spn = _ttk.Spinbox(range_frame, from_=0, to=max(0,n_intervals-1), textvariable=i0_var, width=6, command=lambda: _analyze(preview_only=True))
    i0_spn.pack(side=_tk.LEFT, padx=4)
    i1_spn = _ttk.Spinbox(range_frame, from_=0, to=max(0,n_intervals-1), textvariable=i1_var, width=6, command=lambda: _analyze(preview_only=True))
    i1_spn.pack(side=_tk.LEFT, padx=4)

    # Action buttons placed directly under the interval selector
    btns = _ttk.Frame(win); btns.pack(fill=_tk.X, padx=8, pady=(6, 6))
    _ttk.Button(btns, text="Preview (update)", command=lambda: _analyze(preview_only=True)).pack(side=_tk.LEFT)
    _ttk.Button(btns, text="Accept: fit + save", command=lambda: _analyze(preview_only=False)).pack(side=_tk.LEFT, padx=8)

    # Extra actions: show fits, show R², reset view
    extras = _ttk.Frame(win); extras.pack(fill=_tk.X, padx=8, pady=(0,6))
    show_fits_var = _tk.BooleanVar(value=False)
    _ttk.Checkbutton(extras, text="Show fit curves (after accept)", variable=show_fits_var,
                     command=lambda: _analyze(preview_only=True)).pack(side=_tk.LEFT)
    _ttk.Button(extras, text="Show R² table", command=lambda: _print_r2()).pack(side=_tk.LEFT, padx=8)
    _ttk.Button(extras, text="Reset view", command=lambda: _reset_view()).pack(side=_tk.LEFT)

    def _print_r2():
        r2 = _fit_store.get("r2")
        cols = _fit_store.get("cols")
        if not r2:
            print("[PLI] No R² available. Run 'Accept: fit + save' first.")
            return
        print("[PLI] R² per interval:")
        for j, val in zip(cols, r2):
            print(f"  interval {j}: R² = {val:.4f}" if _np.isfinite(val) else f"  interval {j}: R² = NaN")

    def _reset_view():
        try:
            base_var.set(False)
        except Exception:
            pass
        # clear bottom axis and any secondary axis
        ax_bot.clear()
        if _ax2_bot["axis"] is not None:
            try:
                _ax2_bot["axis"].remove()
            except Exception:
                pass
            _ax2_bot["axis"] = None
        canvas.draw_idle()
        # force a fresh preview redraw without baseline
        _analyze(preview_only=True)

    # Figure with two stacked axes: top=angle preview, bottom=HWHM summary
    fig = Figure(figsize=(7.8, 6.2), dpi=100, constrained_layout=True)
    ax_top = fig.add_subplot(211)
    ax_bot = fig.add_subplot(212)
    try:
        # extra breathing room between subplots if constrained_layout isn't enough
        fig.subplots_adjust(hspace=0.35)
    except Exception:
        pass
    canvas = FigureCanvasTkAgg(fig, master=win)

    # Cache for latest HWHM results from external analysis
    _last_hwhm = {"arr": None, "labels": ["L_HWHM", "a_HWHM", "b_HWHM"], "angle_window": None}
    canvas.get_tk_widget().pack(fill=_tk.BOTH, expand=True, padx=8, pady=6)


    def _compute_param_map():
        mode = (param_cmb.get() or "ab")
        return _param_matrix(mode)

    def _analyze(preview_only: bool = True):
        a0 = ang0.get()
        a1 = ang1.get()
        angY = _np.linspace(0.0, 1.0, H, endpoint=False)
        a_lo, a_hi = (float(ang0.get()), float(ang1.get()))
        if a_hi < a_lo:
            a_lo, a_hi = a_hi, a_lo
        maskY = (angY >= a_lo) & (angY <= a_hi)
        i0 = max(0, min(n_intervals-1, int(i0_var.get())))
        i1 = max(0, min(n_intervals-1, int(i1_var.get())))
        if i1 < i0: i0, i1 = i1, i0

        # --- Preview: plot all intervals for selected parameter (with optional baseline correction) ---
        ax_top.clear()
        M2D = _compute_param_map()
        x_norm_prev = angY[maskY].astype(_np.float64)

        # Optional baseline for preview
        y_base_prev_raw = None
        y_fit_base = None
        if base_var.get():
            kB_prev = max(0, min(n_intervals-1, int(base_idx.get())))
            y_base_prev_raw = M2D[maskY, kB_prev].astype(_np.float64)
            method_prev = base_method_cmb.get().strip().lower()
            if method_prev == "subtract fit":
                # Fit Gaussian (no baseline) on baseline interval within the preview window
                def _g(x, A, x0, sigma):
                    return A * _np.exp(-((x - x0) ** 2) / (2.0 * (sigma ** 2) + 1e-18))
                try:
                    j_max = int(_np.nanargmax(y_base_prev_raw))
                    A0 = float(y_base_prev_raw[j_max]) if _np.isfinite(y_base_prev_raw[j_max]) else float(_np.nanmax(y_base_prev_raw))
                    x0_0 = float(x_norm_prev[j_max]) if _np.isfinite(x_norm_prev[j_max]) else float(_np.nanmedian(x_norm_prev))
                    span = float(_np.nanmax(x_norm_prev) - _np.nanmin(x_norm_prev))
                    sigma0 = max(span / 6.0, 1e-6)
                    p0 = (A0, x0_0, sigma0)
                    bounds = ((0.0, float(_np.nanmin(x_norm_prev)), 0.0),
                              (_np.inf, float(_np.nanmax(x_norm_prev)), _np.inf))
                    popt, _ = curve_fit(_g, x_norm_prev, y_base_prev_raw, p0=p0, bounds=bounds, method="trf", maxfev=20000)
                    y_fit_base = _g(x_norm_prev, *popt)
                except Exception as _e:
                    print(f"[PLI] Preview baseline Gaussian fit failed ({_e}); using raw baseline interval instead.")
                    y_fit_base = None

        for k in range(i0, i1+1):
            y_seg = M2D[maskY, k].astype(_np.float64)
            if base_var.get():
                if y_fit_base is not None and y_fit_base.size == y_seg.size:
                    y_seg = y_seg - y_fit_base
                elif y_base_prev_raw is not None and y_base_prev_raw.size == y_seg.size:
                    y_seg = y_seg - y_base_prev_raw
            ax_top.plot(x_norm_prev, y_seg, linestyle='None', marker='o', markersize=2.0, alpha=0.6)
        # draw baseline fit overlay for reference
        try:
            if y_base_prev_raw is not None and y_base_prev_raw.size:
                ax_top.plot(x_norm_prev, y_base_prev_raw, linewidth=1.0, alpha=0.45)
            if y_fit_base is not None and (y_base_prev_raw is None or y_fit_base.size == y_base_prev_raw.size):
                ax_top.plot(x_norm_prev, y_fit_base, linestyle='--', linewidth=1.2, alpha=0.9)
        except Exception:
            pass

        ax_top.set_xlabel("")
        try:
            ax_top.set_title("Position (normalized) — window", fontsize=10, pad=6)
        except Exception:
            pass
        ylab = f"{param_cmb.get()} (a.u.)" + (" — baseline corrected" if base_var.get() else "")
        ax_top.set_ylabel(ylab)
        ax_top.grid(True, alpha=0.3)
        # If we have stored fits and user requested to show them, overlay on preview
        try:
            if show_fits_var.get() and _fit_store["fits"] is not None and _fit_store["x"] is not None:
                x_fit = _fit_store["x"]
                # Only overlay fits within current window mask
                wmask = (x_fit >= a_lo) & (x_fit <= a_hi)
                if _np.any(wmask):
                    xf = x_fit[wmask]
                    for j, yhat in enumerate(_fit_store["fits"] or []):
                        if yhat is None: continue
                        yh = _np.asarray(yhat)
                        if yh.size == x_fit.size:
                            ax_top.plot(xf, yh[wmask], linestyle='--', linewidth=0.8, alpha=0.8)
        except Exception:
            pass
        canvas.draw_idle()

        if not preview_only:
            # First, run the stitched processing/export so profiles/fits/JSON are guaranteed present
            try:
                run_stitched_processing(stitched_path, reverse_from_end=True, intervals_cfg={})
            except Exception as e:
                print(f"[PLI] stitched processing during ENGAGE failed: {e}")

            # Build trimmed matrix for selected parameter and selected angle window
            mode = (param_cmb.get() or "ab")
            M2D = _compute_param_map()  # H x N
            angles_full = _np.linspace(0.0, 1.0, H, endpoint=False)
            a_lo, a_hi = (float(ang0.get()), float(ang1.get()))
            if a_hi < a_lo:
                a_lo, a_hi = a_hi, a_lo
            mask = (angles_full >= a_lo) & (angles_full <= a_hi)
            if not _np.any(mask):
                print("[PLI] No samples in selected angle window; aborting ENGAGE.")
                return
            # subset intervals (ENGAGE passes ALL intervals downstream regardless of preview selection)
            cols = list(range(n_intervals))
            trim_ = M2D[mask][:, cols]  # m x n

            # --- Optional baseline correction using a chosen interval ---
            if base_var.get():
                kB = max(0, min(n_intervals-1, int(base_idx.get())))
                x_norm_window = angles_full[mask].astype(_np.float64)
                y_base = M2D[mask, kB].astype(_np.float64)
                method = base_method_cmb.get().strip().lower()
                if method == "subtract interval":
                    # subtract the baseline interval profile directly from all intervals
                    trim_ = trim_ - y_base[:, None]
                else:
                    # Fit a Gaussian (no baseline) to the baseline interval, then subtract the fitted curve
                    def _g(x, A, x0, sigma):
                        return A * _np.exp(-((x - x0) ** 2) / (2.0 * (sigma ** 2) + 1e-18))
                    # Initial guesses
                    j_max = int(_np.nanargmax(y_base))
                    A0 = float(y_base[j_max]) if _np.isfinite(y_base[j_max]) else float(_np.nanmax(y_base))
                    x0_0 = float(x_norm_window[j_max]) if _np.isfinite(x_norm_window[j_max]) else float(_np.nanmedian(x_norm_window))
                    span = float(_np.nanmax(x_norm_window) - _np.nanmin(x_norm_window))
                    sigma0 = max(span / 6.0, 1e-6)
                    p0 = (A0, x0_0, sigma0)
                    try:
                        bounds = ((0.0, float(_np.nanmin(x_norm_window)), 0.0),
                                  (_np.inf, float(_np.nanmax(x_norm_window)), _np.inf))
                        popt, _ = curve_fit(_g, x_norm_window, y_base, p0=p0, bounds=bounds, method="trf", maxfev=20000)
                        y_fit_base = _g(x_norm_window, *popt)
                    except Exception as _e:
                        print(f"[PLI] Baseline fit failed ({_e}); falling back to subtracting raw baseline interval.")
                        y_fit_base = y_base
                    trim_ = trim_ - y_fit_base[:, None]

            # Numerical area under experimental (baseline-corrected) data over the window
            x_norm_window = angles_full[mask].astype(_np.float64)
            AUC_exp_vec = _np.trapz(trim_, x=x_norm_window, axis=0)

            # Tell external analyzer what x-axis to use (normalized over selected window)
            try:
                DataAnalyzer.X_centered = angles_full[mask]
            except Exception:
                pass

            # Helpers for model-free half-maximum widths (signed peaks)
            def _model_free_widths(phi: _np.ndarray, Y: _np.ndarray, apply_baseline: bool, span_norm: float):
                ncols = Y.shape[1]
                hwhm = _np.full(ncols, _np.nan, dtype=_np.float64)
                fwhm = _np.full(ncols, _np.nan, dtype=_np.float64)
                mu_list = []
                peak_list = []
                half_list = []
                phiL_list = []
                phiR_list = []
                hL_list = []
                hR_list = []

                dx = _np.nanmedian(_np.diff(phi)) if phi.size > 1 else 1.0
                if not _np.isfinite(dx) or dx <= 0:
                    dx = 1.0

                def _huh_from_fwhm_norm_local(fwhm_norm: float, span: float) -> float:
                    if not _np.isfinite(fwhm_norm) or fwhm_norm <= 0 or span <= 0:
                        return _np.nan
                    hwhm_norm = 0.5 * fwhm_norm
                    hwhm_rad = (hwhm_norm / span) * (2.0 * _np.pi)
                    return float(_np.exp(-(hwhm_rad ** 2) / _np.log(2.0)))

                def _crossing(ph_arr, y_arr, target, start_idx, direction: int):
                    nloc = y_arr.size
                    if direction < 0:
                        rng = range(start_idx - 1, -1, -1)
                    else:
                        rng = range(start_idx, nloc - 1)
                    for j in rng:
                        y1, y2 = y_arr[j], y_arr[j + 1]
                        if not (_np.isfinite(y1) and _np.isfinite(y2)):
                            continue
                        if (y1 - target) * (y2 - target) <= 0:
                            denom = (y2 - y1)
                            if abs(denom) < 1e-18:
                                t = 0.0
                            else:
                                t = float((target - y1) / denom)
                            return float(ph_arr[j] + t * (ph_arr[j + 1] - ph_arr[j]))
                    return _np.nan

                def _attempt(phi_arr, y_arr):
                    # returns phiL, phiR, mu, peak, half
                    if y_arr.size == 0 or phi_arr.size != y_arr.size:
                        return _np.nan, _np.nan, _np.nan, _np.nan, _np.nan
                    y0 = y_arr
                    i0 = int(_np.nanargmax(_np.abs(y0)))
                    peak = float(y0[i0])
                    if not _np.isfinite(peak):
                        return _np.nan, _np.nan, _np.nan, _np.nan, _np.nan
                    mu = float(phi_arr[i0])
                    half = 0.5 * peak
                    phiL = _crossing(phi_arr, y0, half, i0, -1)
                    phiR = _crossing(phi_arr, y0, half, i0, +1)
                    return phiL, phiR, mu, peak, half

                span_full = float(phi[-1] - phi[0]) if phi.size else 1.0
                for j in range(ncols):
                    y = _np.asarray(Y[:, j], dtype=_np.float64)
                    if apply_baseline:
                        B = float(_np.nanpercentile(y, 5.0))
                        y0 = y - B
                    else:
                        y0 = y

                    phiL, phiR, mu, peak, half = _attempt(phi, y0)
                    # If crossings fail and data covers ~full revolution, try rolling to center the peak
                    if (_np.isnan(phiL) or _np.isnan(phiR)) and span_full >= 0.9 and phi.size > 2:
                        i0 = int(_np.nanargmax(_np.abs(y0)))
                        shift = (phi.size // 2) - i0
                        y_roll = _np.roll(y0, shift)
                        center_idx = phi.size // 2
                        phi_center = dx * (_np.arange(phi.size) - center_idx) + mu
                        phiL2, phiR2, mu2, peak2, half2 = _attempt(phi_center, y_roll)
                        # accept if improved
                        if _np.isfinite(phiL2) or _np.isfinite(phiR2):
                            phiL, phiR, mu, peak, half = phiL2, phiR2, mu2, peak2, half2

                    phiL_list.append(phiL)
                    phiR_list.append(phiR)
                    mu_list.append(mu)
                    peak_list.append(peak)
                    half_list.append(half)

                    hL = mu - phiL if _np.isfinite(phiL) else _np.nan
                    hR = phiR - mu if _np.isfinite(phiR) else _np.nan
                    hL_list.append(hL)
                    hR_list.append(hR)

                    if _np.isfinite(phiL) and _np.isfinite(phiR):
                        fwhm[j] = float(phiR - phiL)
                        hwhm[j] = 0.5 * fwhm[j]
                    elif _np.isfinite(hL) and _np.isfinite(hR):
                        hwhm[j] = 0.5 * (hL + hR)
                        fwhm[j] = hL + hR
                    elif _np.isfinite(hL):
                        hwhm[j] = hL
                        fwhm[j] = 2.0 * hL
                    elif _np.isfinite(hR):
                        hwhm[j] = hR
                        fwhm[j] = 2.0 * hR

                # HUH from HWHM using normalized span
                span_norm = span_norm if span_norm > 0 else 1.0
                HUH_local = _np.array([_huh_from_fwhm_norm_local(2.0 * h if _np.isfinite(h) else _np.nan, span_norm) for h in hwhm], dtype=_np.float64)
                details_local = {
                    "model": "none",
                    "mu": mu_list,
                    "peak": peak_list,
                    "half": half_list,
                    "phi_L": phiL_list,
                    "phi_R": phiR_list,
                    "hwhm_left": hL_list,
                    "hwhm_right": hR_list,
                    "FWHM": fwhm,
                    "HWHM": hwhm,
                    "fits": None,
                    "r2": None,
                    "x": phi,
                }
                return hwhm, HUH_local, fwhm, details_local

            # Compute HWHM (index units) and HUH (Hermans-like S in [0,1])
            details = None
            try:
                model_sel = fit_model_var.get().strip().lower()
                if model_sel.startswith("no model"):
                    HWHM_vec, HUH_vec, FWHM_vec, details = _model_free_widths(
                        angles_full[mask],
                        trim_,
                        apply_baseline=bool(use_model_baseline_var.get()),
                        span_norm=float(a_hi - a_lo) if float(a_hi - a_lo) > 0 else 1.0,
                    )
                else:
                    model_key = "asym_qgauss" if model_sel.startswith("asym") else "gaussian"
                    ret = DataAnalyzer.ExPLIDat(
                        trim_,
                        x_axis=angles_full[mask],
                        model=model_key,
                        use_baseline=bool(use_model_baseline_var.get()),
                    )
                    if isinstance(ret, (list, tuple)):
                        if len(ret) >= 2:
                            HWHM_vec, HUH_vec = ret[0], ret[1]
                            if len(ret) >= 3 and isinstance(ret[2], dict):
                                details = ret[2]
                        else:
                            HWHM_vec = _np.asarray(ret)
                            HUH_vec = _np.full(len(cols), _np.nan)
                    else:
                        raise TypeError("unexpected return type from ExPLIDat")
            except Exception as e:
                print(f"[PLI] ExPLIDat failed: {e}")
                HWHM_vec = _np.full(len(cols), _np.nan)
                HUH_vec = _np.full(len(cols), _np.nan)
                FWHM_vec = _np.full(len(cols), _np.nan)
                details = None

            # Prefer FWHM returned in details, otherwise compute from HWHM; also collect AUC if present
            if isinstance(details, dict) and details.get("FWHM") is not None:
                FWHM_vec = _np.asarray(details.get("FWHM"))
            else:
                FWHM_vec = 2.0 * _np.asarray(HWHM_vec)
            # Use experimental trapezoidal area for AUC display/export
            AUC_vec = _np.asarray(AUC_exp_vec)
            # Optionally keep fitted AUC for export if available
            AUC_fit_vec = _np.asarray(details.get("AUC")) if isinstance(details, dict) and details.get("AUC") is not None else None

            # --- Fallback: experimental FWHM (no fit) from half-maximum crossings ---
            def _exp_fwhm(xarr: _np.ndarray, yarr: _np.ndarray) -> float:
                y = _np.asarray(yarr, dtype=_np.float64)
                if y.size < 3 or not _np.isfinite(y).any():
                    return _np.nan
                ymax = float(_np.nanmax(y))
                if not _np.isfinite(ymax) or ymax <= 0:
                    return _np.nan
                half = 0.5 * ymax
                jmax = int(_np.nanargmax(y))
                # search left crossing
                jl = jmax
                while jl > 0 and not (_np.isfinite(y[jl-1]) and y[jl-1] <= half < y[jl]):
                    jl -= 1
                # interpolate left
                try:
                    xl = x_norm_window[jl-1] + (x_norm_window[jl]-x_norm_window[jl-1]) * (half - y[jl-1]) / (y[jl]-y[jl-1])
                except Exception:
                    xl = _np.nan
                # search right crossing
                jr = jmax
                nloc = y.size
                while jr < nloc-1 and not (_np.isfinite(y[jr+1]) and y[jr+1] <= half < y[jr]):
                    jr += 1
                # interpolate right
                try:
                    xr = x_norm_window[jr+1] + (x_norm_window[jr]-x_norm_window[jr+1]) * (half - y[jr+1]) / (y[jr]-y[jr+1])
                except Exception:
                    xr = _np.nan
                if _np.isfinite(xl) and _np.isfinite(xr):
                    return abs(xr - xl)
                return _np.nan

            FWHM_exp_vec = _np.array([_exp_fwhm(x_norm_window, trim_[:, j]) for j in range(trim_.shape[1])], dtype=_np.float64)

            # Replace missing/invalid FWHM with experimental FWHM
            FWHM_vec = _np.where(_np.isfinite(FWHM_vec) & (FWHM_vec > 0), FWHM_vec, FWHM_exp_vec)

            # If HUH is missing or ~0, synthesize from HWHM derived from (possibly experimental) FWHM
            def _huh_from_fwhm_norm(fwhm_norm: float, span: float) -> float:
                if not _np.isfinite(fwhm_norm) or fwhm_norm <= 0 or span <= 0:
                    return _np.nan
                hwhm_norm = 0.5 * fwhm_norm
                # map normalized width to radians (full revolution = 1.0 → 2π)
                hwhm_rad = (hwhm_norm / span) * (2.0 * _np.pi)
                return float(_np.exp(-(hwhm_rad ** 2) / _np.log(2.0)))

            span_norm = float(a_hi - a_lo) if float(a_hi - a_lo) > 0 else 1.0
            HUH_fallback = _np.array([_huh_from_fwhm_norm(FWHM_vec[k], span_norm) for k in range(len(FWHM_vec))], dtype=_np.float64)
            HUH_vec = _np.where(_np.isfinite(HUH_vec) & (HUH_vec > 0), HUH_vec, HUH_fallback)

            # Prefer using details to populate the fit store; fallback to local computation if needed
            if isinstance(details, dict) and all(k in details for k in ("fits", "r2", "x")) and details.get("fits") is not None:
                _fit_store["x"] = _np.asarray(details.get("x"))
                try:
                    _fit_store["fits"] = [ _np.asarray(f) if f is not None else None for f in details.get("fits") ]
                except TypeError:
                    _fit_store["fits"] = None
                try:
                    _fit_store["r2"] = list(details.get("r2")) if details.get("r2") is not None else None
                except Exception:
                    _fit_store["r2"] = None
                _fit_store["cols"] = cols
            else:
                # Fallback: compute per-interval Gaussian fits and R² on the corrected data
                def _gauss(x, A, x0, s):
                    return A * _np.exp(-((x - x0) ** 2) / (2.0 * (s ** 2) + 1e-18))
                x_fit = angles_full[mask].astype(_np.float64)
                fits = []
                r2_list = []
                for j in range(trim_.shape[1]):
                    y = trim_[:, j].astype(_np.float64)
                    jmax = int(_np.nanargmax(y)) if y.size else 0
                    A0 = float(y[jmax]) if _np.isfinite(y[jmax]) else float(_np.nanmax(y))
                    x0_ = float(x_fit[jmax]) if _np.isfinite(x_fit[jmax]) else float(_np.nanmedian(x_fit))
                    s0 = max(float(_np.nanmax(x_fit) - _np.nanmin(x_fit)) / 6.0, 1e-6)
                    p0 = (A0, x0_, s0)
                    try:
                        bnds = ((0.0, float(_np.nanmin(x_fit)), 0.0), ( _np.inf, float(_np.nanmax(x_fit)), _np.inf))
                        popt, _ = curve_fit(_gauss, x_fit, y, p0=p0, bounds=bnds, method='trf', maxfev=20000)
                        yhat = _gauss(x_fit, *popt)
                        ss_res = float(_np.nansum((y - yhat)**2))
                        ss_tot = float(_np.nansum((y - _np.nanmean(y))**2))
                        r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else _np.nan
                    except Exception:
                        yhat = _np.full_like(x_fit, _np.nan)
                        r2 = _np.nan
                    fits.append(yhat)
                    r2_list.append(r2)
                _fit_store["x"] = x_fit
                _fit_store["fits"] = fits
                _fit_store["r2"] = r2_list
                _fit_store["cols"] = cols

            # Plot results on bottom axis: primary FWHM, secondary AUC
            # remove previous secondary axis if present
            if _ax2_bot["axis"] is not None:
                try:
                    _ax2_bot["axis"].remove()
                except Exception:
                    pass
                _ax2_bot["axis"] = None
            ax_bot.clear()
            x_idx = _np.arange(len(cols))

            # Primary y-axis: FWHM
            fwhm_plot = ax_bot.plot(
                x_idx, FWHM_vec,
                linestyle='None', marker='o', mfc='none', mec='C0', mew=1.25,
                label='FWHM (index units)'
            )[0]
            ax_bot.set_xlabel("Interval index (relative)")
            ax_bot.set_ylabel("FWHM (index units)")
            ax_bot.grid(True, alpha=0.3)

            # Secondary y-axis: AUC (green)
            auc_plot = None
            if AUC_vec is not None and AUC_vec.size == len(cols):
                ax2 = ax_bot.twinx()
                _ax2_bot["axis"] = ax2
                auc_plot = ax2.plot(
                    x_idx, AUC_vec,
                    linestyle='None', marker='o', mfc='none', mec='green', mew=1.25,
                    label='AUC (a.u.)'
                )[0]
                ax2.set_ylabel("AUC (a.u.)", color='green')
                ax2.tick_params(axis='y', colors='green')

            # Combined legend using available handles
            try:
                lines = [fwhm_plot]
                if auc_plot is not None:
                    lines.append(auc_plot)
                labels = [l.get_label() for l in lines]
                ax_bot.legend(lines, labels, loc='best', fontsize='small')
            except Exception:
                pass
            canvas.draw_idle()

            # Save results to CSV with parameter and limits in filename
            out_root = get_unified_processed_folder(stitched_path)
            out_root = get_unified_processed_folder(stitched_path)
            out_dir = out_root
            os.makedirs(out_dir, exist_ok=True)
            base = os.path.splitext(os.path.basename(stitched_path))[0]
            lim_tag = f"x{a_lo:.3f}-{a_hi:.3f}"
            mode_tag = mode.replace('/', '').replace('^','').replace('*','')
            # Overwrite the same summary per stitched image (no per-mode suffix to avoid clutter)
            csv_out = os.path.join(out_dir, f"_pli_{base}_maltese_summary.csv")
            df_dict = {
                "interval_global": cols,
                "interval_rel": x_idx.astype(int),
                "FWHM_norm": FWHM_vec,
                "HWHM_norm": HWHM_vec,
                "HUH": HUH_vec,
                "AUC_exp_norm": AUC_vec,
            }
            if 'AUC_fit_vec' in locals() and AUC_fit_vec is not None and AUC_fit_vec.size == len(cols):
                df_dict["AUC_fit_norm"] = AUC_fit_vec
            df_out = pd.DataFrame(df_dict)
            df_out.to_csv(csv_out, index=False)
            print(f"Saved Maltese-cross summary (param={mode}; limits={lim_tag}): {csv_out}")

            # --- DataGraph-ready export (degrees, compact) ---
            # Convert normalized widths (fraction of full 360°) to degrees
            FWHM_deg = _np.asarray(FWHM_vec, dtype=_np.float64) * 360.0
            HWHM_deg = _np.asarray(HWHM_vec, dtype=_np.float64) * 360.0
            dg_df = pd.DataFrame({
                "interval": x_idx.astype(int),
                "FWHM_deg": FWHM_deg,
                "HWHM_deg": HWHM_deg,
                "HUH": _np.asarray(HUH_vec, dtype=_np.float64),
                "AUC_exp": _np.asarray(AUC_vec, dtype=_np.float64)
            })
            if 'AUC_fit_vec' in locals() and AUC_fit_vec is not None and AUC_fit_vec.size == len(cols):
                dg_df["AUC_fit"] = _np.asarray(AUC_fit_vec, dtype=_np.float64)

            # Overwrite the same DG export per stitched image
            dg_csv = os.path.join(out_dir, f"_dg_pli_{base}_maltese.csv")
            dg_df.to_csv(dg_csv, index=False)
            print(f"Saved DataGraph-ready CSV: {dg_csv}")

            # Update single master PLI JSON so the writer detects everything by image name
            master_json = os.path.join(out_dir, f"_output_PLI_{base}.json")
            # If stitched processing hasn't created a master file yet, create a minimal one
            master_obj = {
                "mode": "pli",
                "source_png": os.path.abspath(stitched_path),
                "processed_folder": get_unified_processed_folder(stitched_path),
                "datasets": {}
            }
            if os.path.isfile(master_json):
                try:
                    with open(master_json, "r") as _f:
                        master_obj = json.load(_f) or master_obj
                except Exception as _e:
                    print(f"[PLI] Warning: could not read master JSON; recreating: {_e}")
            # Ensure datasets dict exists
            if not isinstance(master_obj.get("datasets"), dict):
                master_obj["datasets"] = {}

            # Key by image base name so the writer can find it via drop-down
            key = base
            entry = master_obj["datasets"].get(key, {})
            # Ensure nested maltese_cross dict
            if not isinstance(entry.get("maltese_cross"), dict):
                entry["maltese_cross"] = {}

            # Build a unique tag for this analysis (parameter + window)
            analysis_tag = f"{mode_tag}|{lim_tag}"
            # Columns list for JSON
            columns_list = [
                "interval_global", "interval_rel",
                "FWHM_norm", "HWHM_norm", "HUH",
                "AUC_exp_norm"
            ]
            auc_fit_included = False
            if 'AUC_fit_vec' in locals() and AUC_fit_vec is not None and AUC_fit_vec.size == len(cols):
                columns_list.append("AUC_fit_norm")
                auc_fit_included = True
            _rel_summary = csv_out.split('/_Processed/', 1)[1] if '/_Processed/' in csv_out else os.path.basename(csv_out)
            _rel_dg = dg_csv.split('/_Processed/', 1)[1] if '/_Processed/' in dg_csv else os.path.basename(dg_csv)
            entry["maltese_cross"][analysis_tag] = {
                "parameter": mode,
                "axis_limits": [a_lo, a_hi],
                "x_axis_units": "unit",
                "summary_csv": os.path.abspath(csv_out),
                "summary_csv_rel": _rel_summary,
                "columns": columns_list,
                "auc_source": "experimental_trapz",
                "auc_fit_included": auc_fit_included,
                "n_intervals_total": int(n_intervals),
                "intervals_analyzed": {"start": int(i0), "end": int(i1)},
                "hu_definition": "HUH = exp(-(HWHM_rad^2)/ln 2) assuming wrapped-Gaussian",
                "datagraph_ready": True,
                "datagraph_csv": os.path.abspath(dg_csv),
                "datagraph_csv_rel": _rel_dg
            }

            # Maintain a flat list of DG exports for quick discovery by the writer UI
            if not isinstance(entry.get("dg_exports"), list):
                entry["dg_exports"] = []
            if os.path.abspath(dg_csv) not in entry["dg_exports"]:
                entry["dg_exports"].append(os.path.abspath(dg_csv))
            # Maintain relative list for portability
            rel_list = [
                (p.split('/_Processed/', 1)[1] if isinstance(p, str) and '/_Processed/' in p else os.path.basename(p) if isinstance(p, str) else p)
                for p in entry.get("dg_exports", [])
            ]
            entry["dg_exports_rel"] = rel_list

            # Write back to master JSON
            master_obj["datasets"][key] = entry
            try:
                with open(master_json, "w") as _f:
                    json.dump(master_obj, _f, indent=2)
                print(f"Updated master PLI JSON: {master_json}")
            except Exception as _e:
                print(f"[PLI] Failed to update master PLI JSON: {_e}")

    win.update_idletasks()

    # Initial preview
    _analyze(preview_only=True)
    win.lift(); win.focus_force()


# --- Entry point: main() is invoked here after all GUI helpers are defined
if __name__ == "__main__":
    main()
