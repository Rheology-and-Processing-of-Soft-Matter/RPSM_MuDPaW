import sys
import os
import re
import shutil
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from _Routines.SAXS import P2468OAS_v4 as hop
from _Routines.SAXS import P2468OAS_v4_2pk as hop2
from scipy.special import legendre
import json

# --- Sample naming helper ---

def _clean_sample_label(raw: str) -> str:
    """Return a readable sample label from a filename, stripping prefix/extension noise."""
    stem = Path(raw).stem if raw else ""
    cleaned = re.sub(r"^azi[_-]?saxs[_-]?", "", stem, flags=re.IGNORECASE)
    cleaned = cleaned.lstrip("_.- ")
    return cleaned or stem or "sample"

# --- Radial normalization helpers ---
from typing import Tuple, Optional

# --- ASCII azimuth loader (angle + f1..fN) ---
def _load_ascii_azimuth(path: str):
    """
    Load an azimuthal matrix stored as plain text (.dat/.txt) with the
    first column = angle and remaining columns = frame intensities.
    Handles comma or whitespace delimiters and skips leading comments/blank
    lines so files exported by the Connector (v5.8.1 style) work unchanged.
    Returns (angles, cake, header_line) where cake has shape (frames, angles).
    """
    header_line, skiprows = _scan_header_info(path)
    # Try automatic delimiter detection first (python engine supports sep=None)
    try:
        df = pd.read_csv(
            path,
            sep=None,
            engine="python",
            comment="#",
            skiprows=skiprows,
        )
    except Exception:
        # Fallback to whitespace
        df = pd.read_csv(
            path,
            sep=r"\\s+",
            engine="python",
            comment="#",
            skiprows=skiprows,
        )

    # Drop completely empty columns/rows
    df = df.dropna(axis=1, how="all").dropna(axis=0, how="all")
    if df.shape[1] < 2:
        raise ValueError(f"Azimuth file {path} has <2 columns after cleaning.")

    angles = pd.to_numeric(df.iloc[:, 0], errors="coerce").to_numpy()
    # pd.to_numeric does not accept DataFrames; convert column-wise then back to ndarray
    vals = (
        df.iloc[:, 1:]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=float)
    )

    # Keep rows where angle parsed successfully
    ok = ~np.isnan(angles)
    angles = angles[ok]
    vals = vals[ok, :]

    cake = vals.T  # (frames, angles)
    return angles, cake, header_line

def _looks_like_header(line: str) -> bool:
    """Return True if the first line contains non-numeric tokens (e.g., 'ang,f1,...')."""
    stripped = (line or "").strip()
    if not stripped:
        return True
    tokens = stripped.replace(",", " ").split()
    if not tokens:
        return True
    try:
        float(tokens[0])
        return False
    except ValueError:
        return True

def _scan_header_info(path: str, max_lines: int = 20) -> tuple[str, int]:
    """
    Inspect the file and return (header_line, skiprows) where skiprows counts any
    blank/comment/header rows preceding numeric data.
    """
    header_line = ""
    skiprows = 0
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                raw = line.rstrip("\r\n")
                stripped = raw.strip()
                if not stripped:
                    skiprows += 1
                    continue
                if stripped.startswith("#"):
                    header_line = stripped
                    skiprows += 1
                    continue
                if _looks_like_header(stripped):
                    header_line = stripped
                    skiprows += 1
                    continue
                # first non-empty, non-header, non-comment line reached
                break 
    except Exception:
        header_line = ""
        skiprows = 0
    return header_line, skiprows

def _load_dat_matrix(path: str) -> tuple[np.ndarray, str]:
    """Robustly load a .dat file that may or may not have a header row."""
    header_line, skiprows = _scan_header_info(path)
    try:
        data = np.loadtxt(path, skiprows=skiprows)
    except ValueError:
        data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
    return data, header_line

def _header_has_angle(header_line: str) -> bool:
    cleaned = (header_line or "").split("#", 1)[-1].strip().lower()
    if not cleaned:
        return False
    tokens = [tok for tok in re.split(r"[,\s]+", cleaned) if tok]
    if not tokens:
        return False
    first = tokens[0]
    angle_keys = {
        "angle", "ang",
        "azi", "azimuth", "azimuthal",
        "theta", "2theta",
        "deg", "degrees",
        "phi",
    }
    return first in angle_keys

def _derive_radial_headers(header_line: str, column_count: int) -> list[str]:
    """Return a robust set of column labels for the radial matrix."""
    cleaned = header_line.split("#", 1)[-1].strip() if header_line else ""
    tokens = [tok.strip() for tok in cleaned.replace(",", " ").split() if tok.strip()]
    headers: list[str] = []
    if tokens:
        headers = tokens[:column_count]
    if len(headers) < column_count:
        headers.extend([f"frame_{i+1}" for i in range(len(headers), column_count)])
    if headers:
        headers[0] = "q"
    else:
        headers = ["q"] + [f"frame_{i+1}" for i in range(column_count - 1)]
    return headers

def _build_header_line(first_label: str, data_cols: int, prefix: str) -> str:
    labels = [first_label] + [f"{prefix}{i+1}" for i in range(data_cols)]
    return " ".join(labels)

def _env_disables_plots() -> bool:
    val = os.environ.get("MUDRAW_SAXS_NO_PLOTS", "")
    return val.strip().lower() in {"1", "true", "yes", "on"}

def _plots_enabled(requested: bool) -> bool:
    return bool(requested) and not _env_disables_plots()

def _write_matrix_with_header(path: str, data: np.ndarray, header_line: Optional[str]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        if header_line:
            fh.write(header_line.strip() + "\n")
        for row in data:
            fh.write("\t".join(f"{val:.10e}" for val in row) + "\n")

def _average_columns_if_needed(
    matrix: np.ndarray,
    first_label: str,
    header_prefix: str,
    group_size: int = 4,
    trigger_total_cols: Optional[int] = 117,
    write_back_path: Optional[str] = None,
    force: bool = False,
) -> tuple[np.ndarray, Optional[str], bool]:
    """Average every `group_size` columns (excluding the first).
    If `force` is False, only act when total columns match trigger_total_cols.
    """
    total_cols = matrix.shape[1]
    if not force and trigger_total_cols is not None and total_cols != trigger_total_cols:
        return matrix, None, False
    data_block = matrix[:, 1:]
    rem = data_block.shape[1] % group_size
    if rem != 0:
        if not force:
            return matrix, None, False
        # Trim trailing columns so averaging can proceed
        usable = data_block.shape[1] - rem
        if usable <= 0:
            return matrix, None, False
        data_block = data_block[:, :usable]
    n_groups = data_block.shape[1] // group_size
    averaged = data_block.reshape(matrix.shape[0], n_groups, group_size).mean(axis=2)
    combined = np.hstack([matrix[:, [0]], averaged])
    header_line = _build_header_line(first_label, n_groups, header_prefix)
    if write_back_path:
        try:
            _write_matrix_with_header(write_back_path, combined, header_line)
            print(f"[SAXS] Averaged {data_block.shape[1]} columns into {n_groups} for {write_back_path}")
        except Exception as exc:
            print(f"[SAXS] Warning: failed to rewrite averaged file {write_back_path}: {exc}")
    return combined, header_line, True

def _export_radial_csvs(radial_path: str, sample_tag: str, results_path: str, *, average_columns: bool = False, average_group: int = 4) -> Tuple[Optional[str], Optional[str], list[str], Optional[str]]:
    """
    Read the radial integration matrix and emit:
      1) A matrix CSV (all frames preserved)
      2) A flattened CSV (Intensity, Frames, q)
    Returns (matrix_csv, flat_csv, headers, error_message)
    """
    header_line, skiprows = _scan_header_info(radial_path)
    try:
        data = np.loadtxt(radial_path, dtype=float, comments="#", skiprows=skiprows)
    except Exception:
        try:
            data = np.loadtxt(radial_path, dtype=float, comments="#", delimiter=",", skiprows=skiprows)
        except Exception as exc:
            return None, None, [], f"Failed to parse radial data: {exc}"

    if data.ndim == 1:
        data = data.reshape(-1, 1)
    if data.size == 0:
        return None, None, [], "Radial file appears to be empty."

    data, averaged_header, averaged = _average_columns_if_needed(
        data,
        first_label="q",
        header_prefix="frame_",
        group_size=max(1, int(average_group or 4)),
        write_back_path=radial_path,
        force=average_columns,
        trigger_total_cols=None if average_columns else 117,
    )
    if averaged and averaged_header:
        header_line = averaged_header

    column_count = data.shape[1]
    headers = _derive_radial_headers(header_line, column_count)

    matrix_df = pd.DataFrame(data, columns=headers)
    matrix_df = matrix_df.apply(pd.to_numeric, errors="coerce")
    if "q" in matrix_df.columns:
        matrix_df = matrix_df.dropna(subset=["q"])
    matrix_df = matrix_df.reset_index(drop=True)
    matrix_csv = os.path.join(results_path, f"SAXS_radial_matrix_{sample_tag}.csv")
    matrix_df.to_csv(matrix_csv, index=False)

    flat_csv: Optional[str] = None
    if column_count >= 2:
        numeric_data = matrix_df.to_numpy()
        q_vals = numeric_data[:, 0]
        frame_block = numeric_data[:, 1:]
        n_q = q_vals.shape[0]
        n_frames = frame_block.shape[1] if frame_block.ndim > 1 else 1
        if n_frames == 1 and frame_block.ndim == 1:
            frame_block = frame_block.reshape(-1, 1)
        intensities = frame_block.reshape(-1, order="F")
        q_repeat = np.tile(q_vals, n_frames)
        frame_ids = np.repeat(np.arange(1, n_frames + 1), n_q)
        flat_df = pd.DataFrame(
            {
                "Q": q_repeat,
                "Frame_rad": frame_ids,
                "I_rad": intensities,
            }
        )
        flat_csv = os.path.join(results_path, f"SAXS_radial_flat_{sample_tag}.csv")
        flat_df.to_csv(flat_csv, index=False)

    return matrix_csv, flat_csv, headers, None

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    """Resolve the *reference* folder robustly.

    Walks up the path to find known modality markers ("PLI", "PI", "SAXS", "Rheology").
    The *reference* folder is the parent directory of the modality folder.
    Fallback: two levels up from the provided path (legacy behavior).
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    abspath = os.path.abspath(path)
    parts = _split_parts(abspath)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            # Reconstruct with a leading slash to preserve absolute path
            reference = os.path.join(os.sep, *parts[:i])
            return reference

    # Fallback: two levels up from the provided absolute path
    return os.path.dirname(os.path.dirname(abspath))


def _extract_scan_range(name: str):
    for pat in (r"scan_(\d+)-(\d+)", r"(\d{3,})-(\d{3,})"):
        m = re.search(pat, name)
        if m:
            try:
                return int(m.group(1)), int(m.group(2))
            except Exception:
                pass
    return (None, None)


def _find_radial_for_azimuthal(azi_path: str) -> str | None:
    p = Path(azi_path)
    folder = p.parent
    if not folder.exists():
        return None
    lo, hi = _extract_scan_range(p.name)
    cands = sorted(q for q in folder.glob("rad_saxs*") if q.is_file())
    if not cands:
        cands = sorted(q for q in folder.glob("*rad*.*") if q.is_file())
    if not cands:
        return None
    if lo is not None and hi is not None:
        scored = []
        for q in cands:
            qlo, qhi = _extract_scan_range(q.name)
            score = int(qlo == lo) + int(qhi == hi)
            scored.append((score, q))
        scored.sort(key=lambda t: (-t[0], t[1].name))
        return str(scored[0][1].resolve())
    return str(cands[0].resolve())


def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    processed_root = os.path.abspath(processed_root)
    os.makedirs(processed_root, exist_ok=True)
    print(f"[SAXS] Using processed folder: {processed_root}")
    return processed_root

from _Routines.common import update_json_file

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


def SAXS_data_processor(path_name, sample_name, smoo_, sigma_zero, lower_limit, upper_limit, plots=True, method="fitting", mirror=False, two_peak=False, secondary_limits=None, flatten_tail_fit=False, flatten_tail_extent=0.9, average_columns=False, average_group=4, centering_mode="auto", weak_metric="A2", weak_threshold=None, side_mode="auto", side_mode2=None):
    print(f"Processing SAXS data with parameters: {path_name}, {sample_name}, {smoo_}, {sigma_zero}, {lower_limit}, {upper_limit}, method={method}, mirror={mirror}")
    last_fig = None

    # Convert limits to degrees
    limit_deg = [lower_limit, upper_limit]

    source = Path(path_name)
    path_entries: list[tuple[str, list[str] | None, bool]] = []
    if source.is_file():
        # Direct file input: process only this .dat
        path_entries.append((str(source.parent), [source.name], True))
    else:
        # Get the list of folders in the root directory
        folder_list = [
            f for f in os.listdir(path_name)
            if os.path.isdir(os.path.join(path_name, f))
            and not f.startswith('.')
            and not f.startswith('_Processed')
            and not f.startswith('_output')
        ]

        if folder_list:
            for folder in folder_list:
                path_entries.append((os.path.join(path_name, folder), None, False))
        else:
            print("No subfolders found. Searching for .dat files directly in the provided path.")
            dats = [
                f for f in os.listdir(path_name)
                if f.lower().endswith(".dat") and not f.startswith('_')
            ]
            if not dats:
                print("No .dat files found in the provided path.")
                return
            path_entries.append((path_name, dats, True))

    # Define the results path (resolved once for non-file inputs)
    processed_folder = get_unified_processed_folder(path_name)
    results_path = processed_folder
    print(f"Unified _Processed folder for SAXS outputs: {processed_folder}")

    print(f"Found folders: {[p for p, _, _ in path_entries if _ is None]}")
    plot_enabled = _plots_enabled(plots)
    # Avoid blocking GUI windows; PyQt will embed figures instead.
    show_backup = None
    if plot_enabled and plt is not None:
        try:
            show_backup = plt.show
            plt.show = lambda *args, **kwargs: None
        except Exception:
            show_backup = None
    
    for path, explicit_files, flat_files in path_entries:
        file_names = explicit_files or [f for f in os.listdir(path) if f.lower().endswith((".dat", ".txt"))]
        print(f"Processing folder: {path}, found files: {file_names}")
        
        for file_name in file_names:
            load_location = os.path.join(path, file_name)
            ext_lower = load_location.lower()
            if not ext_lower.endswith((".dat", ".txt")):
                print(f"Skipping unsupported SAXS file (only .dat/.txt handled): {load_location}")
                continue
            sample_base = _clean_sample_label(file_name) if flat_files else Path(file_name).stem
            sample_name2 = sample_base if flat_files else f"{sample_name}_{sample_base}"
            print(f"Loading file: {load_location}")
            radial_file = _find_radial_for_azimuthal(load_location)
            if radial_file:
                print(f"Found radial integration file: {radial_file}")
            else:
                print("No radial integration file located for this azimuthal dataset.")
            try:
                if load_location.lower().endswith((".dat", ".txt")):
                    angles, cake, header_line = _load_ascii_azimuth(load_location)  # cake: frames x angles
                    # Arrange as (angles x frames) with angle in col0
                    working_raw = np.column_stack([angles, cake.T])
                    has_angle_header = True
                    averaged = False
                    avg_write_path = None
                    prev_cols = working_raw.shape[1]
                else:
                    file_raw, header_line = _load_dat_matrix(load_location)
                    avg_group = max(1, int(average_group or 4))
                    prev_cols = file_raw.shape[1] if file_raw.ndim == 2 else 0
                    has_angle_header = _header_has_angle(header_line)
                    if has_angle_header:
                        working_raw = file_raw
                        avg_write_path = load_location
                    else:
                        synth_angles = np.linspace(0.0, 360.0, file_raw.shape[0], endpoint=False)
                        working_raw = np.column_stack([synth_angles, file_raw])
                        avg_write_path = None
                    working_raw, _, averaged = _average_columns_if_needed(
                        working_raw,
                        first_label="ang",
                        header_prefix="f",
                        group_size=avg_group,
                        write_back_path=avg_write_path,
                        force=average_columns,
                        trigger_total_cols=None if average_columns else 117,
                    )
                    if averaged:
                        if has_angle_header:
                            print(f"[SAXS] Averaged azimuthal frames {prev_cols - 1} → {working_raw.shape[1] - 1} for {load_location}")
                        else:
                            print(f"[SAXS] Averaged azimuthal frames {prev_cols} → {working_raw.shape[1] - 1} for {load_location}")
            except Exception as e:
                print(f"Failed to load file: {load_location}. Error: {e}")
                continue

            n_file, m_file = working_raw.shape
            print(f"File shape: {n_file} x {m_file}")
            
            n_frames = m_file - 1
            angles = working_raw[:, 0]
            file_part = working_raw[:, 1:]

            # Normalize angle axis to degrees for downstream selection & plotting.
            # Legacy files sometimes store radians; when users specify limits in degrees
            # (e.g., 70–170°) the previous code would collapse the window to a single
            # column because the radians range is ~0–3.14. Converting here keeps the
            # UI limits aligned with the data and preserves all frames.
            if np.isfinite(angles).any() and np.nanmax(np.abs(angles)) <= (2 * np.pi + 0.1):
                angles_deg = np.rad2deg(angles)
            else:
                angles_deg = angles

            def _process_one_peak(tag_suffix: str, limit_pair, side_choice: str):
                radial_copy_path = None
                radial_csv_abs = None
                radial_csv_rel = None
                radial_matrix_csv_abs = None
                radial_matrix_csv_rel = None
                radial_flat_csv_abs = None
                radial_flat_csv_rel = None
                radial_headers: list[str] = []

                flattened_raw = file_part.flatten(order='F')
                frames = np.repeat(np.arange(n_frames), n_file).reshape(-1, 1)
                angle_flat = np.tile(angles, n_frames).reshape(-1, 1)
                flat_data = np.hstack((flattened_raw.reshape(-1, 1), frames, angle_flat))

                # Map requested angle limits (deg) to indices using actual angles array
                lim_array = np.array(limit_pair, dtype=float)
                ang_min, ang_max = float(np.nanmin(angles_deg)), float(np.nanmax(angles_deg))
                if ang_max == ang_min:
                    limits = np.array([0, n_file - 1])
                else:
                    idx_float = np.interp(lim_array, (ang_min, ang_max), (0, n_file - 1))
                    limits = np.round(idx_float).astype(int)
                # Clamp limits to valid bounds and ensure at least one column
                limits = np.clip(limits, 0, n_file - 1)
                if limits[1] <= limits[0]:
                    limits[1] = min(n_file, limits[0] + 1)
                print(f"Processing data with limits ({tag_suffix}): {limits}")
                try:
                    processor = hop2.P2468OAS_v4 if two_peak else hop.P2468OAS_v4
                    kwargs = dict(
                        method=method,
                        mirror=mirror,
                        Angles=angles_deg,
                        File_part_=file_part,
                        lims=limits,
                        smoo_=smoo_,
                        sigma_zero=sigma_zero,
                        plot=plot_enabled,
                        window_title=f"SAXS — {sample_name2}{tag_suffix}" if plot_enabled else None,
                        centering_mode=centering_mode,
                        weak_metric=weak_metric,
                        weak_threshold=weak_threshold,
                        side_mode=side_choice,
                    )
                    if processor is hop2.P2468OAS_v4:
                        kwargs["flatten_tail_fit"] = flatten_tail_fit
                        kwargs["flatten_tail_extent"] = flatten_tail_extent
                    P2468OAS = processor(**kwargs)
                    if plot_enabled:
                        # Use the pyplot instance from the processor module (correct backend)
                        try:
                            if processor is hop2.P2468OAS_v4:
                                last_fig = hop2.plt.gcf()
                            else:
                                last_fig = hop.plt.gcf()
                        except Exception:
                            if plt is not None:
                                last_fig = plt.gcf()
                except Exception as e:
                    print(f"Processing failed for {sample_name2}{tag_suffix}: {e}")
                    return

                m_out, n_out = P2468OAS.shape
                print(f"Processed data shape: {m_out} x {n_out} ({tag_suffix})")

                # Save the processed data
                saxs_files = []
                if radial_file:
                    try:
                        suffix = Path(radial_file).suffix or ".dat"
                        radial_copy_path = os.path.join(results_path, f"SAXS_radial_{sample_name2}{tag_suffix}{suffix}")
                        shutil.copy2(radial_file, radial_copy_path)
                        print(f"Copied radial integration to: {radial_copy_path}")
                    except Exception as e:
                        print(f"Failed to copy radial integration file: {e}")
                        radial_copy_path = None

                    if radial_copy_path:
                        try:
                            matrix_csv, flat_csv, headers, err = _export_radial_csvs(
                                radial_copy_path,
                                f"{sample_name2}{tag_suffix}",
                                results_path,
                                average_columns=average_columns,
                                average_group=average_group,
                            )
                            if err:
                                print(f"Radial export warning: {err}")
                            if matrix_csv and os.path.exists(matrix_csv):
                                radial_matrix_csv_abs = matrix_csv
                                radial_matrix_csv_rel = os.path.relpath(matrix_csv, processed_folder)
                                radial_csv_abs = matrix_csv  # legacy alias for UI
                                radial_csv_rel = radial_matrix_csv_rel
                                print(f"Wrote radial matrix CSV: {matrix_csv}")
                            if flat_csv and os.path.exists(flat_csv):
                                radial_flat_csv_abs = flat_csv
                                radial_flat_csv_rel = os.path.relpath(flat_csv, processed_folder)
                                print(f"Wrote radial flat CSV: {flat_csv}")
                            if headers:
                                radial_headers = headers
                        except Exception as e:
                            print(f"Failed to normalize radial file to CSV: {e}")

                export_path = os.path.join(results_path, f"SAXS_1_{sample_name2}{tag_suffix}_export.csv")
                column_names = ["P2", "P4", "P6", "OAS"]
                pd.DataFrame(P2468OAS, columns=column_names).to_csv(export_path, index=False, header=True)
                saxs_files.append(export_path)
                print(f"Saved processed data to: {export_path}")

                raw_flat_export_path = os.path.join(results_path, f"SAXS_2_{sample_name2}{tag_suffix}_raw_flat.csv")
                column_names = ["Intensity", "Frames", "Angles"]
                pd.DataFrame(flat_data, columns=column_names).to_csv(raw_flat_export_path, index=False, header=True)
                saxs_files.append(raw_flat_export_path)
                print(f"Saved raw flattened data to: {raw_flat_export_path}")

                if radial_matrix_csv_abs and os.path.exists(radial_matrix_csv_abs):
                    saxs_files.append(radial_matrix_csv_abs)
                if radial_flat_csv_abs and os.path.exists(radial_flat_csv_abs):
                    saxs_files.append(radial_flat_csv_abs)

                print(f"SAXS results saved to unified folder: {results_path}")

                json_path = os.path.join(processed_folder, f"_output_SAXS_{sample_name2}{tag_suffix}.json")
                rel_suffixes = [os.path.relpath(p, processed_folder) for p in saxs_files]
                azimuthal_file = load_location
                azimuthal_file_rel = (
                    os.path.relpath(azimuthal_file, processed_folder)
                    if os.path.exists(azimuthal_file) else None
                )
                radial_file_rel = (
                    os.path.relpath(radial_file, processed_folder)
                    if radial_file and os.path.exists(radial_file) else None
                )
                radial_copy_rel = (
                    os.path.relpath(radial_copy_path, processed_folder)
                    if radial_copy_path and os.path.exists(radial_copy_path) else None
                )
                radial_matrix_csv_rel = (
                    os.path.relpath(radial_matrix_csv_abs, processed_folder)
                    if radial_matrix_csv_abs and os.path.exists(radial_matrix_csv_abs) else None
                )
                radial_flat_csv_rel = (
                    os.path.relpath(radial_flat_csv_abs, processed_folder)
                    if radial_flat_csv_abs and os.path.exists(radial_flat_csv_abs) else None
                )

                payload = {
                    "sample_name": f"{sample_name2}{tag_suffix}",
                    "csv_outputs": saxs_files,
                    "csv_outputs_rel": rel_suffixes,
                    "azimuthal_file": azimuthal_file,
                    "azimuthal_file_rel": azimuthal_file_rel,
                    "radial_file": radial_file,
                    "radial_file_rel": radial_file_rel,
                    "radial_copy": radial_copy_path,
                    "radial_copy_rel": radial_copy_rel,
                    "radial_csv": radial_csv_abs,
                    "radial_csv_rel": radial_csv_rel,
                    "radial_matrix_csv": radial_matrix_csv_abs,
                    "radial_matrix_csv_rel": radial_matrix_csv_rel,
                    "radial_flat_csv": radial_flat_csv_abs,
                    "radial_flat_csv_rel": radial_flat_csv_rel,
                    "radial_headers": radial_headers,
                }
                try:
                    with open(json_path, "w") as jf:
                        json.dump(payload, jf, indent=2)
                    print(f"Wrote unified JSON: {json_path}")
                    try:
                        print("[SAXS] Output summary:")
                        print(f"  processed root : {processed_folder}")
                        for p in saxs_files:
                            print(f"  csv            : {p}")
                        for label, p_abs, p_rel in (
                            ("azimuthal", azimuthal_file, azimuthal_file_rel),
                            ("radial_src", radial_file, radial_file_rel),
                            ("radial_copy", radial_copy_path, radial_copy_rel),
                            ("radial_matrix_csv", radial_matrix_csv_abs, radial_matrix_csv_rel),
                            ("radial_flat_csv", radial_flat_csv_abs, radial_flat_csv_rel),
                        ):
                            if p_abs:
                                print(f"  {label:14s}: {p_abs} (rel {p_rel})")
                        print(f"  json           : {json_path}")
                    except Exception:
                        pass
                except Exception as e:
                    print(f"Failed to write JSON {json_path}: {e}")

            if two_peak and secondary_limits:
                _process_one_peak("_p1", (lower_limit, upper_limit), side_mode)
                _process_one_peak("_p2", secondary_limits, side_mode2 or side_mode)
            else:
                _process_one_peak("", (lower_limit, upper_limit), side_mode)

    if show_backup is not None and plt is not None:
        try:
            plt.show = show_backup
        except Exception:
            pass
    return last_fig

def _parse_cli_args():
    parser = argparse.ArgumentParser(
        description="Process SAXS azimuthal/radial datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("path_name", help="Folder containing the sample subfolders or .dat files.")
    parser.add_argument("sample_name", help="Sample prefix for outputs.")
    parser.add_argument("smoo_", type=float, help="Smoothing value.")
    parser.add_argument("sigma_zero", type=float, help="Sigma zero.")
    parser.add_argument("lower_limit", type=float, help="Lower angular limit (deg).")
    parser.add_argument("upper_limit", type=float, help="Upper angular limit (deg).")
    parser.add_argument("--fast", action="store_true", help="Skip plotting windows.")
    parser.add_argument("--no-plots", action="store_true", dest="no_plots", help="Alias for --fast.")
    parser.add_argument("--two-peak", action="store_true", help="Enable two-peak processing (beta).")
    parser.add_argument("--lower2", type=float, help="Lower angular limit for peak 2 (deg).")
    parser.add_argument("--upper2", type=float, help="Upper angular limit for peak 2 (deg).")
    parser.add_argument("--flatten-tail-fit", action="store_true", help="Extrapolate tail with LG fit when flattening minima (two-peak mode).")
    parser.add_argument("--flatten-tail-extent", type=float, default=0.9, help="Fraction of pre-minimum segment to use when fitting LG tail (0-1).")
    parser.add_argument(
        "--method",
        choices=["fitting", "direct"],
        default="fitting",
        help="HoP computation method.",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="Mirror azimuthal data prior to processing.",
    )
    parser.add_argument(
        "--average-columns",
        type=int,
        metavar="GROUP",
        help="Average azimuthal frame columns in groups of N (e.g., 4).",
    )
    parser.add_argument(
        "--centering-mode",
        choices=["auto", "argmax", "harmonic2", "xcorr"],
        default="auto",
        help="Centering mode for azimuthal windowing.",
    )
    parser.add_argument(
        "--weak-metric",
        choices=["A2", "S2", "a2", "s2"],
        default="A2",
        help="Metric for weak-alignment detection.",
    )
    parser.add_argument(
        "--weak-threshold",
        type=float,
        help="Weak alignment threshold (leave empty for auto).",
    )
    parser.add_argument(
        "--side-mode",
        choices=["auto", "left", "right"],
        default="auto",
        help="Select which side of the peak to integrate.",
    )
    parser.add_argument(
        "--side-mode2",
        choices=["auto", "left", "right"],
        default=None,
        help="Side selection for peak 2 (two-peak mode).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_cli_args()
    plots = not (args.fast or args.no_plots)
    SAXS_data_processor(
        args.path_name,
        args.sample_name,
        args.smoo_,
        args.sigma_zero,
        args.lower_limit,
        args.upper_limit,
        plots=plots,
        method=args.method,
        mirror=args.mirror,
        two_peak=args.two_peak,
        secondary_limits=(args.lower2, args.upper2) if args.lower2 is not None and args.upper2 is not None else None,
        flatten_tail_fit=args.flatten_tail_fit,
        flatten_tail_extent=args.flatten_tail_extent,
        average_columns=args.average_columns is not None,
        average_group=args.average_columns or 4,
        centering_mode=args.centering_mode,
        weak_metric=args.weak_metric,
        weak_threshold=args.weak_threshold,
        side_mode=args.side_mode,
        side_mode2=args.side_mode2,
    )
