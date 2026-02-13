import os
import sys
import pandas as pd
import numpy as np


import json

from datetime import datetime
from typing import List, Optional, Tuple

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    """Resolve the *reference* folder robustly.

    Walk up the path to find known modality markers ("PLI", "PI", "SAXS", "Rheology").
    The *reference* folder is the parent directory of the modality folder.
    Fallback: two levels up from the provided path (legacy behavior).
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    abspath = os.path.abspath(path)
    parts = _split_parts(abspath)

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            reference = os.sep.join(parts[:i])
            if reference == "":
                break
            return reference

    return os.path.dirname(os.path.dirname(abspath))


def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    os.makedirs(processed_root, exist_ok=True)
    return processed_root


def get_sample_root(path: str) -> str:
    abspath = os.path.abspath(path)
    cur = abspath
    while True:
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            return abspath
        if os.path.basename(parent).lower() == "pi":
            return cur
        cur = parent


def get_temp_processed_folder(path: str) -> str:
    sample_root = get_sample_root(path)
    temp_root = os.path.join(sample_root, "_Temp_processed")
    os.makedirs(temp_root, exist_ok=True)
    return temp_root


# --- Timestamp readers and format detection ---

# Utility: split columns by tab or aligned multi-spaces (Anton Paar)
import re

def _split_cols(line: str) -> list[str]:
    """
    Split a table line into columns. Prefer tabs; if there are no tabs,
    split on runs of >=2 spaces (Anton Paar 'aligned spaces' export).
    """
    if "\t" in line:
        return [t.strip() for t in line.split("\t")]
    # Collapse multiple spaces; split on 2+ spaces to avoid splitting inside values like "1.2"
    return [t.strip() for t in re.split(r"\s{2,}", line.strip()) if t.strip()]

def _read_text_best_effort(file_path: str) -> str:
    """Try UTF-16 (Anton Paar) then UTF-8, then default."""
    for enc in ("utf-16", "utf-8"):
        try:
            with open(file_path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            pass
    with open(file_path, "r") as f:
        return f.read()

def _looks_like_saxs_timefile(file_path: str, max_lines: int = 50) -> bool:
    """
    Heuristic: a SAXS-style time file is a single-column list of ISO timestamps
    like 'YYYY-MM-DD HH:MM:SS[.ffffff]' with no headers.
    """
    try:
        lines = []
        with open(file_path, "r", encoding="utf-8") as f:
            for i, ln in enumerate(f):
                if i >= max_lines:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                lines.append(ln)
    except UnicodeDecodeError:
        # try utf-16 (but SAXS files are usually utf-8/plain)
        with open(file_path, "r", encoding="utf-16") as f:
            for i, ln in enumerate(f):
                if i >= max_lines:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                lines.append(ln)
    except Exception:
        return False

    if not lines:
        return False

    fmts = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
    ok = 0
    for ln in lines:
        parsed = False
        for fmt in fmts:
            try:
                datetime.strptime(ln, fmt)
                parsed = True
                break
            except Exception:
                continue
        if parsed:
            ok += 1
        else:
            return False
    return ok > 0

# Detect plain Δt seconds (_Triggers style) file
def _looks_like_delta_seconds(file_path: str, max_lines: int = 50) -> bool:
    """
    Heuristic: a Δt-seconds file (_Triggers style) is a single-column list of floats.
    """
    try:
        lines = []
        with open(file_path, "r") as f:
            for i, ln in enumerate(f):
                if i >= max_lines:
                    break
                ln = ln.strip()
                if not ln:
                    continue
                lines.append(ln)
        if not lines:
            return False
        cnt = 0
        for ln in lines:
            try:
                float(ln)
                cnt += 1
            except Exception:
                return False
        return cnt > 0
    except Exception:
        return False

def _parse_saxs_timefile(file_path: str) -> List[datetime]:
    """Parse single-column ISO timestamps to a list of datetimes."""
    dts = []
    # allow both utf-8 and default
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            rows = [ln.strip() for ln in f if ln.strip()]
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="utf-16") as f:
            rows = [ln.strip() for ln in f if ln.strip()]
    fmts = ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S")
    for r in rows:
        parsed = None
        for fmt in fmts:
            try:
                parsed = datetime.strptime(r, fmt)
                break
            except Exception:
                continue
        if parsed is None:
            # also try pandas just in case
            try:
                parsed = pd.to_datetime(r, errors="raise").to_pydatetime()
            except Exception:
                continue
        dts.append(parsed if isinstance(parsed, datetime) else parsed)
    return dts

def _parse_anton_paar(file_path: str) -> Optional[List[datetime]]:
    """
    Parse Anton Paar ASCII 'Interval data:' tables. Prefer 'Test Start Date and Time' + 'Time of Day'.
    Fallback to 'Time [min]' if Time of Day missing. Robust even if headers are missing.
    """
    text = _read_text_best_effort(file_path)
    lines = text.splitlines()

    # Find 'Interval data:'
    start_idx = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Interval data:"):
            start_idx = i
            break
    if start_idx is None:
        return None

    # Helper to find next non-empty line index
    def next_nonempty(idx):
        idx += 1
        while idx < len(lines) and not lines[idx].strip():
            idx += 1
        return idx

    hdr_idx = next_nonempty(start_idx)
    units_idx = next_nonempty(hdr_idx)
    data_start = next_nonempty(units_idx)

    # Use header width as canonical; Anton Paar may add leading blanks in units/data
    header_tokens = _split_cols(lines[hdr_idx]) if hdr_idx < len(lines) else []
    unit_tokens   = _split_cols(lines[units_idx]) if units_idx < len(lines) else []
    ncols = len(header_tokens)

    # Collect rows
    rows = []
    for ln in lines[data_start:]:
        if not ln.strip():
            continue
        parts = _split_cols(ln)
        # If parts are wider than header due to leading blanks, drop them
        if len(parts) > ncols:
            extra = len(parts) - ncols
            # drop only if the extra tokens are empty (common in Anton Paar)
            if all(t == "" for t in parts[:extra]):
                parts = parts[extra:]
        if ncols and len(parts) < ncols:
            parts += [""] * (ncols - len(parts))
        rows.append(parts[:ncols] if ncols else parts)
    if not rows:
        return None

    # Heuristics to identify columns by value patterns
    def looks_like_datetime_str(s: str) -> bool:
        for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                datetime.strptime(s, fmt)
                return True
            except Exception:
                pass
        return False

    def looks_like_time_of_day(s: str) -> bool:
        for fmt in ("%H:%M:%S",):
            try:
                datetime.strptime(s, fmt)
                return True
            except Exception:
                pass
        return False

    def looks_like_minutes(s: str) -> bool:
        try:
            float(s)
            return True
        except Exception:
            return False

    dt_idx = tod_idx = min_idx = None
    probe = rows[:30]
    for ci in range(len(probe[0])):
        vals = [r[ci] for r in probe if r[ci] != ""]
        if not vals:
            continue
        if dt_idx is None and any(looks_like_datetime_str(v) for v in vals):
            dt_idx = ci
        if tod_idx is None and any(looks_like_time_of_day(v) for v in vals):
            tod_idx = ci
        if min_idx is None and all(looks_like_minutes(v) for v in vals[:5]):
            min_idx = ci

    if dt_idx is None:
        return None  # cannot build absolute wall-clock without base date-time

    # Fetch base date-time (constant) from first non-empty dt cell
    base_dt = None
    for r in rows:
        if r[dt_idx]:
            for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                try:
                    base_dt = datetime.strptime(r[dt_idx], fmt)
                    break
                except Exception:
                    continue
            if base_dt:
                break
    if base_dt is None:
        return None

    abs_times: List[datetime] = []
    if tod_idx is None:
        # Strict mode: require both Test Start Date and Time + Time of Day
        return None
    # Preferred path only: date from base_dt + each Time of Day
    for r in rows:
        t = r[tod_idx]
        if not t:
            continue
        try:
            tod = datetime.strptime(t, "%H:%M:%S").time()
            abs_times.append(datetime.combine(base_dt.date(), tod))
        except Exception:
            continue
    if not abs_times:
        return None

    return abs_times if abs_times else None


# --- Extract all intervals from Anton Paar (multi-box) export ---
def _extract_anton_paar_intervals(file_path: str) -> List[Tuple[int, List[datetime]]]:
    """
    Parse an Anton Paar export that may contain multiple 'measurement boxes' and multiple
    'Interval and data points:' + 'Interval data:' sections. Return a list of
    (interval_index, [absolute datetimes]) tuples. Strict: requires both 'Test Start Date and Time'
    and 'Time of Day' columns within each Interval data block; otherwise the interval is skipped.
    """
    text = _read_text_best_effort(file_path)
    lines = text.splitlines()

    results: List[Tuple[int, List[datetime]]] = []

    i = 0
    n = len(lines)

    def next_nonempty(idx: int) -> int:
        idx += 1
        while idx < n and not lines[idx].strip():
            idx += 1
        return idx

    while i < n:
        line = lines[i].strip()

        # Look for "Interval and data points:"
        if line.startswith("Interval and data points:"):
            # Example: "Interval and data points:\t1\t100"
            parts = _split_cols(lines[i])
            # Extract interval index if possible
            interval_idx = None
            if len(parts) >= 2:
                try:
                    interval_idx = int(parts[1])
                except Exception:
                    interval_idx = None

            # Now we expect an "Interval data:" header later (skip any extra text)
            j = i + 1
            while j < n and not lines[j].strip().startswith("Interval data:"):
                j += 1
            if j >= n:
                # No data block for this interval; move on
                i += 1
                continue

            # Parse the "Interval data:" table
            header_line = lines[j].strip()  # starts with "Interval data:"
            # Columns start after the literal label; split on tabs then drop the first token
            header_tokens = _split_cols(lines[j])
            # The literal "Interval data:" is part of the first token; drop it
            if header_tokens and header_tokens[0].startswith("Interval data:"):
                header_tokens[0] = header_tokens[0].replace("Interval data:", "").strip()
            # If after replacement the first token is empty, drop it
            if header_tokens and header_tokens[0] == "":
                header_tokens = header_tokens[1:]
            # If the first token still contains multiple names glued by spaces, split it again
            if len(header_tokens) == 1 and " " in header_tokens[0]:
                header_tokens = _split_cols(header_tokens[0])

            # Units line = next non-empty
            units_idx = next_nonempty(j)
            units_tokens = _split_cols(lines[units_idx]) if units_idx < n else []

            # DEBUG: show how we parsed the header/units for this interval
            try:
                print("[AntonPaar] Parsed header tokens: ", header_tokens)
                print("[AntonPaar] Parsed units tokens:  ", units_tokens)
            except Exception:
                pass

            # Data starts after units (skip blank lines)
            data_start = next_nonempty(units_idx)

            # Use header width as canonical; units/data may have leading blanks
            ncols = len(header_tokens)
            try:
                print(f"[AntonPaar] Using header width ncols={ncols} with header: {header_tokens}")
            except Exception:
                pass

            # Identify important columns by name (case-insensitive exact/contains)
            def find_col(name_options):
                lower_map = {c.lower(): k for k, c in enumerate(header_tokens)}
                # exact first
                for opt in name_options:
                    k = lower_map.get(opt.lower())
                    if k is not None:
                        return k
                # contains fallback
                for opt in name_options:
                    for k, c in enumerate(header_tokens):
                        if opt.lower() in c.lower():
                            return k
                return None

            col_dt = find_col(["Test Start Date and Time", "Start Date and Time", "Test Start"])
            col_tod = find_col(["Time of Day", "Time-Of-Day", "Time Of Day"])

            if col_dt is None or col_tod is None:
                try:
                    print("[AntonPaar] Could not locate required columns in this interval header.")
                    print("[AntonPaar] Raw header line:", lines[j].rstrip())
                    print("[AntonPaar] Tokens:", header_tokens)
                except Exception:
                    pass
                # Strict: skip this interval if required columns are not present
                i = j + 1
                continue

            # Gather rows until a blank line or next "Interval and data points:" appears
            k = data_start
            abs_times: List[datetime] = []

            # Parse base date-time from the first non-empty row's dt column
            base_dt = None

            while k < n:
                raw = lines[k]
                s = raw.strip()
                if not s:
                    # Blank line ends this interval
                    break
                if s.startswith("Interval and data points:"):
                    # New interval starts; stop current
                    break

                parts = _split_cols(raw)
                # If parts wider than header due to leading blanks, drop them
                if len(parts) > ncols:
                    extra = len(parts) - ncols
                    if all(t == "" for t in parts[:extra]):
                        parts = parts[extra:]
                # Pad to expected length
                if ncols and len(parts) < ncols:
                    parts += [""] * (ncols - len(parts))

                dt_cell = parts[col_dt] if col_dt < len(parts) else ""
                tod_cell = parts[col_tod] if col_tod < len(parts) else ""

                # Initialize base date-time once
                if base_dt is None and dt_cell:
                    for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                        try:
                            base_dt = datetime.strptime(dt_cell, fmt)
                            break
                        except Exception:
                            continue

                # Build absolute datetime per row from Time of Day
                if base_dt is not None and tod_cell:
                    try:
                        tod = datetime.strptime(tod_cell, "%H:%M:%S").time()
                        abs_times.append(datetime.combine(base_dt.date(), tod))
                    except Exception:
                        pass

                k += 1

            # Store only if we got timestamps
            try:
                if abs_times:
                    print(f"[AntonPaar] Interval {interval_idx}: captured {len(abs_times)} rows; first={abs_times[0]}, last={abs_times[-1]}")
            except Exception:
                pass
            if abs_times and interval_idx is not None:
                results.append((interval_idx, abs_times))

            # Continue scanning from where we left off
            i = k
            continue

        # No interval header on this line; advance
        i += 1

    return results

def detect_file_kind(file_path: str) -> str:
    """
    Return:
      'saxs'  for SAXS time list files (absolute timestamps),
      'rheo'  for Anton Paar interval exports,
      'delta' for plain Δt seconds lists (_Triggers style),
      'unknown' otherwise.
    """
    if _looks_like_delta_seconds(file_path):
        return "delta"
    if _looks_like_saxs_timefile(file_path):
        return "saxs"
    text = _read_text_best_effort(file_path)
    if "Interval data:" in text or "Interval  data:" in text:
        return "rheo"
    return "unknown"

def _select_file_dialog(title: str) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception as exc:
        print(f"Tkinter not available for file dialog: {exc}")
        return None
    root = tk.Tk()
    root.withdraw()
    return filedialog.askopenfilename(title=title, filetypes=[("All files", "*.*")])

def process_time_file(sample_dir, file_path):
    sample_dir = os.path.abspath(sample_dir)
    sample_dir_root = sample_dir
    sample_dir = get_temp_processed_folder(sample_dir_root)
    if not file_path:
        file_path = _select_file_dialog("Select Reference Time File")
        if not file_path:
            print("No file selected. Exiting...")
            sys.exit(1)

    # If caller passed the target _Triggers path itself, force the user to pick a source file
    if os.path.basename(file_path) == "_Triggers":
        print("Note: destination '_Triggers' path was provided; opening file dialog to select a source timestamp file...")
        picked = _select_file_dialog("Select Source Timestamp File (SAXS or Anton Paar)")
        if not picked:
            print("No source file selected. Exiting...")
            sys.exit(1)
        file_path = picked

    print(f"Processing file: {file_path}")  # Debugging print

    kind = detect_file_kind(file_path)
    if kind == "delta":
        # Source already contains Δt seconds (e.g., an existing _Triggers file). Normalize and pass through.
        try:
            vals = np.loadtxt(file_path, dtype=float)
            if vals.ndim == 0:
                vals = np.array([float(vals)])
            time_diffs_seconds = np.asarray(vals, dtype=float)
        except Exception as e:
            print(f"Could not read Δt-seconds file: {e}")
            sys.exit(1)
        print(f"Detected Δt-seconds file; {len(time_diffs_seconds)} values will be written to destination _Triggers.")
        # Minimal preview
        prev = time_diffs_seconds[:5].tolist()
        print(f"Preview Δt (first 5 s): {prev}")
    elif kind == "saxs":
        # SAXS-style: parse an absolute timestamp list and compute relative seconds
        dts = _parse_saxs_timefile(file_path)
        if not dts:
            print("Could not parse SAXS-style timestamp file. Exiting...")
            sys.exit(1)
        d0 = dts[0]
        time_diffs_seconds = np.array([(dt - d0).total_seconds() for dt in dts], dtype=float)
        print(f"Detected SAXS timestamp file; computed {len(time_diffs_seconds)} Δt values.")
        # Debug preview (absolute + Δt)
        try:
            preview_abs = dts[:5]
            preview_dt = time_diffs_seconds[:5].tolist()
            print("Preview (first 5):")
            for i, t in enumerate(preview_abs):
                print(f"  {i}: {t}  Δt={preview_dt[i] if i < len(preview_dt) else 'NA'} s")
        except Exception:
            pass
    elif kind == "rheo":
        print("Detected Anton Paar file; scanning all measurement boxes and intervals...")
        # Extract all intervals strictly requiring Test Start Date and Time + Time of Day
        interval_series = _extract_anton_paar_intervals(file_path)
        if not interval_series:
            print("Error: Anton Paar export must include 'Interval data:' blocks with both 'Test Start Date and Time' and 'Time of Day'. Please re-export and try again.")
            sys.exit(1)

        # Select even-numbered intervals as SAXS-aligned recording windows
        even_intervals = [(idx, dts) for (idx, dts) in interval_series if idx % 2 == 0]
        if not even_intervals:
            print("Error: No even-numbered intervals detected. Expected steady-recording intervals at 2,4,6,...")
            sys.exit(1)

        # Write one _Triggers file per even interval, and a primary _Triggers for the first even interval
        generated = []
        for idx, dts in even_intervals:
            d0 = dts[0]
            dt_seconds = np.array([(dt - d0).total_seconds() for dt in dts], dtype=float)
            out_name = f"_Triggers_interval_{idx:02d}"
            out_path = os.path.join(sample_dir, out_name)
            np.savetxt(out_path, dt_seconds, comments='')
            generated.append((idx, out_path, len(dt_seconds)))

        # Primary _Triggers points to the first even interval by default
        first_idx, first_path, first_n = generated[0]
        primary_out = os.path.join(sample_dir, "_Triggers")
        # Copy contents of the first even interval into _Triggers
        try:
            with open(first_path, "r") as src, open(primary_out, "w") as dst:
                dst.write(src.read())
        except Exception as e:
            print(f"Warning: could not write primary _Triggers: {e}")

        print(f"Detected Anton Paar rheology export; wrote {len(generated)} even-interval trigger files.")
        for idx, pth, nn in generated[:5]:
            print(f"  Interval {idx:02d}: {nn} points → {pth}")
        if len(generated) > 5:
            print(f"  ...and {len(generated)-5} more intervals.")

        # For downstream, set time_diffs_seconds to the primary (first even) one for compatibility
        with open(primary_out, "r") as f:
            time_diffs_seconds = np.array([float(x) for x in f.read().strip().splitlines()], dtype=float)

        # Debug preview (absolute + Δt) for the first even interval
        try:
            preview_abs = even_intervals[0][1][:5]
            preview_dt = time_diffs_seconds[:5].tolist()
            print("Preview (first 5):")
            for i, t in enumerate(preview_abs):
                print(f"  {i}: {t}  Δt={preview_dt[i] if i < len(preview_dt) else 'NA'} s")
        except Exception:
            pass
    else:
        # Fallback: treat as legacy 1-column datetime CSV (skip header issue removed)
        try:
            df = pd.read_csv(file_path, header=None, names=["timestamp"])
            ts = pd.to_datetime(df["timestamp"], errors="coerce").dropna()
            d0 = ts.iloc[0].to_pydatetime()
            dts = [t.to_pydatetime() for t in ts]
            time_diffs_seconds = np.array([(dt - d0).total_seconds() for dt in dts], dtype=float)
            print(f"Unknown format → parsed as one-column datetimes; computed {len(time_diffs_seconds)} Δt values.")
        except Exception as e:
            print(f"Unsupported timestamp file format: {e}") 
            sys.exit(1)
        # Debug preview (absolute + Δt) using the first 5 parsed datetimes
        try:
            preview_abs = [t for t in dts[:5]]
            preview_dt = time_diffs_seconds[:5].tolist()
            print("Preview (first 5):")
            for i, t in enumerate(preview_abs):
                print(f"  {i}: {t}  Δt={preview_dt[i] if i < len(preview_dt) else 'NA'} s")
        except Exception:
            pass

    if len(time_diffs_seconds) == 0:
        print("No valid timestamps found. Exiting...")
        sys.exit(1)

    # Define the output file path
    output_file_path = os.path.join(sample_dir, '_Triggers')

    print(f"Saving file: {output_file_path}")  # Debugging print

    # Save the time differences to a file without a header
    try:
        np.savetxt(output_file_path, time_diffs_seconds, comments='')
    except Exception as e:
        print(f"Error saving file: {e}. Exiting...")
        sys.exit(1)

    print(f"Processing completed for {file_path}")

    processed_folder = get_unified_processed_folder(sample_dir_root)
    print(f"Unified _Processed folder for Triggers outputs: {processed_folder}")

    json_path = os.path.join(processed_folder, f"_output_PI_triggers.json")

    def _rel(p: str) -> str:
        try:
            return p.split('/_Processed/', 1)[1]
        except Exception:
            return os.path.basename(p)

    payload = {
        "sample_dir": sample_dir_root,
        "processed_folder": processed_folder,
        # absolute primary
        "output_triggers": output_file_path,
        # relative primary (portable)
        "output_triggers_rel": _rel(output_file_path),
    }

    if kind == "rheo":
        # absolute list with intervals
        payload["output_triggers_all"] = [
            {"interval": int(idx), "path": os.path.join(sample_dir, f"_Triggers_interval_{int(idx):02d}")}
            for idx, _ in even_intervals
        ]
        # relative-only convenience list
        payload["output_triggers_all_rel"] = [
            {"interval": int(idx), "path_rel": _rel(os.path.join(sample_dir, f"_Triggers_interval_{int(idx):02d}"))}
            for idx, _ in even_intervals
        ]
        # absolute + relative primary alias
        payload["primary_output_triggers"] = output_file_path
        payload["primary_output_triggers_rel"] = _rel(output_file_path)

    if kind == "delta":
        payload["source_kind"] = "delta_seconds"
        payload["source_file"] = file_path
        payload["source_file_rel"] = _rel(file_path)

    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Trigger processing output JSON written: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) >= 2:
        # Allow: script <sample_dir> <file_path>  OR  script <file_path>
        if len(sys.argv) >= 3:
            sample_dir = sys.argv[1]
            file_path = sys.argv[2]
        else:
            file_path = sys.argv[1]
            sample_dir = os.path.dirname(file_path)
        if not os.path.exists(sample_dir):
            print(f"Sample directory does not exist: {sample_dir}")
            sys.exit(1)
        process_time_file(sample_dir, file_path)
    else:
        # GUI fallback
        selected_file = _select_file_dialog("Select Reference Time File")
        if not selected_file:
            print("No file selected. Exiting...")
            sys.exit(1)
        sample_dir = os.path.dirname(selected_file)
        process_time_file(sample_dir, selected_file)
