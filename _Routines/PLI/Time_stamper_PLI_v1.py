from tkinter import messagebox
import os
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from _UI_modules.PLI_helper import pick_and_compute_pair_widths, verify_constant_even, get_reference_folder_from_path

# --- Added for compute_pair_widths_from_times ---
from datetime import datetime
import os
import re

STEADY_DELAY_SEC = 2.0

def on_load_from_file(self):
    try:
        fps = float(self.fps_var.get())
    except Exception:
        messagebox.showerror("FPS missing", "Please enter a valid frame rate (fps) before loading.")
        return

    try:
        # Use the resolved Reference folder (parent of PI/PLI/SAXS/Rheology marker) as default dialog location
        ref_folder = get_reference_folder_from_path(self.current_sample_dir)
        if not os.path.isdir(ref_folder):
            ref_folder = self.current_sample_dir
        result = pick_and_compute_pair_widths(
            self.current_sample_dir,
            fps,
            reverse_from_end=True,
            initialdir=ref_folder,
            filetypes=[
                ("Anton Paar CSV", "*.csv"),
                ("SAXS time files", "*.txt *.dat *.time"),
                ("All files", "*.*"),
            ],
        )
    except Exception as e:
        messagebox.showerror("Load error", str(e))
        return

    widths = result["widths"]              # [transient_last, steady_last, ..., transient_first, steady_first]
    even_widths = result["even_widths"]    # [steady_first, steady_second, ...] (in chronological order)
    num_pairs = result["num_pairs"]

    # 1) Set number of intervals = 2 * num_pairs
    total_intervals = 2 * num_pairs
    self.num_intervals_var.set(total_intervals)

    # 2) Ensure enough entry widgets exist (expand UI if needed)
    if len(self.interval_entries) < total_intervals:
        self._ensure_interval_entries(total_intervals)  # implement to add extra rows if missing

    # 3) Populate entries, bottom-up end alignment:
    #    'widths' already ordered from last pair to first, so write top→bottom reversed
    for i, entry in enumerate(reversed(self.interval_entries[:total_intervals])):
        w = widths[i]
        entry.delete(0, "end")
        entry.insert(0, str(w))

    # 4) Handle "constant width (even intervals)" checkbox
    is_const, const_val = verify_constant_even(even_widths)
    if self.constant_even_var.get():
        if is_const:
            # Show the global width field (if you have one) and enforce const_val
            if hasattr(self, "interval_width_global_entry"):
                self.interval_width_global_entry.delete(0, "end")
                self.interval_width_global_entry.insert(0, str(const_val))
                self.interval_width_global_entry.config(state="readonly")
            # Overwrite every even interval width in the GUI with const_val
            # Remember: GUI rows are end-aligned; even intervals are the 2nd of each pair in 'widths'.
            # Recompute indexes accordingly:
            #   widths = [T_last, S_last, T_prev, S_prev, ..., T_first, S_first]
            #   entries were filled from bottom; enforce S_k where index 1,3,5... in 'widths' → translate to rows
            idx = 1
            for pair in range(num_pairs):
                # row_from_bottom = idx (since we filled reversed), find the corresponding Entry
                row_from_bottom = idx
                gui_index = total_intervals - 1 - row_from_bottom
                e = self.interval_entries[gui_index]
                e.delete(0, "end")
                e.insert(0, str(const_val))
                idx += 2
        else:
            messagebox.showwarning("Not constant",
                                   "Even-interval widths are not equal; unchecking the 'constant width' option.")
            self.constant_even_var.set(False)
            if hasattr(self, "interval_width_global_entry"):
                self.interval_width_global_entry.config(state="normal")
                self.interval_width_global_entry.delete(0, "end")

    # Optionally, auto-set End position(px) to current image width or keep user-controlled
    # self.end_pos_var.set(self._default_end_px())

    # After this, when the user clicks PREVIEW:
    # - read back the per-interval widths (bottom-up end alignment),
    # - pass them to your PLI_data_processor exactly as you do today,
    # - no files need to be written by the stamper.


# --- Utilities for reading Anton Paar CSV and SAXS timestamp lists ---

def _split_cols(line: str) -> list[str]:
    if "\t" in line:
        return [t.strip() for t in line.split("\t")]
    return [t.strip() for t in re.split(r"\s{2,}", line.strip()) if t.strip()]


def _parse_iso_dt(s: str) -> datetime | None:
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


# --- Core: compute pair widths (frames) from a time source ---

def _steady_delay_frames(fps: float) -> int:
    try:
        return max(0, int(round(STEADY_DELAY_SEC * float(fps))))
    except Exception:
        return 0


def _apply_delay_to_pair(wt: int, ws: int, fps: float) -> tuple[int, int]:
    trim = min(_steady_delay_frames(fps), max(0, ws - 1))
    if trim <= 0:
        return wt, ws
    return wt + trim, max(1, ws - trim)


def compute_pair_widths_from_times(file_path: str, fps: float, reverse_from_end: bool = True) -> dict:
    """
    Given a time-source file (Anton Paar CSV or SAXS absolute timestamp list),
    return interval-pair widths in frames for the PLI rescale UI.
    Output dict keys: widths, even_widths, even_constant, even_value, num_pairs.
    """
    fps = float(fps)
    ext = os.path.splitext(file_path)[1].lower()

    # Case A: SAXS absolute timestamps (.txt/.dat/.time) → single steady window
    if ext in (".txt", ".dat", ".time"):
        # Read lines (utf-8, fallback utf-16)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                rows = [ln.strip() for ln in f if ln.strip()]
        except UnicodeDecodeError:
            with open(file_path, "r", encoding="utf-16") as f:
                rows = [ln.strip() for ln in f if ln.strip()]
        dts = [dt for dt in (_parse_iso_dt(r) for r in rows) if dt is not None]
        if len(dts) < 2:
            raise ValueError("Not enough SAXS timestamps to compute a duration.")
        dt_s = (dts[-1] - dts[0]).total_seconds()
        w_even = max(1, int(round(dt_s * fps)))
        wt_adj, ws_adj = _apply_delay_to_pair(0, w_even, fps)
        widths = [wt_adj, ws_adj]  # transient, steady
        return {
            "widths": list(reversed(widths)) if reverse_from_end else widths,
            "even_widths": [ws_adj],
            "even_constant": True,
            "even_value": ws_adj,
            "num_pairs": 1,
        }

    # Case B: Anton Paar CSV (UTF-16/UTF-8) with Interval data blocks
    if ext == ".csv":
        # Read as UTF-16 then UTF-8
        text = None
        for enc in ("utf-16", "utf-8"):
            try:
                with open(file_path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except Exception:
                continue
        if text is None:
            raise ValueError("Could not read CSV (UTF-16/UTF-8).")
        lines = text.splitlines()
        n = len(lines)

        # Collect absolute datetimes per interval index, splitting into boxes when index resets or headers appear
        boxes_pairs: list[tuple[int,int]] = []  # accumulated (wt, ws) pairs across boxes (chrono order)
        box_intervals: dict[int, list[datetime]] = {}  # current box local intervals by index

        def _flush_box():
            nonlocal boxes_pairs, box_intervals
            if not box_intervals:
                return
            # Build (odd, even) pairs for this box in ascending local index
            for odd in sorted(ix for ix in box_intervals if ix % 2 == 1):
                even = odd + 1
                if even in box_intervals:
                    seg_odd = box_intervals[odd]
                    seg_even = box_intervals[even]
                    wt = max(1, int(round((seg_odd[-1] - seg_odd[0]).total_seconds() * fps)))
                    ws = max(1, int(round((seg_even[-1] - seg_even[0]).total_seconds() * fps)))
                    wt, ws = _apply_delay_to_pair(wt, ws, fps)
                    boxes_pairs.append((wt, ws))
            box_intervals = {}

        i = 0
        def next_nonempty(idx: int) -> int:
            idx += 1
            while idx < n and not lines[idx].strip():
                idx += 1
            return idx

        while i < n:
            s = lines[i].strip()
            # Header lines indicating start of a new measurement box
            if s.startswith("Project:") or s.startswith("Test:") or s.startswith("Result:"):
                _flush_box()
                i += 1
                continue

            if s.startswith("Interval and data points:"):
                parts = _split_cols(lines[i])
                idx_val = None
                if len(parts) >= 2:
                    try:
                        idx_val = int(parts[1])
                    except Exception:
                        idx_val = None
                # If local index resets to 1 and we already collected intervals → new box
                if idx_val == 1 and box_intervals:
                    _flush_box()

                # Find the "Interval data:" header
                j = i + 1
                while j < n and not lines[j].strip().startswith("Interval data:"):
                    j += 1
                if j >= n:
                    i += 1
                    continue

                header_tokens = _split_cols(lines[j])
                if header_tokens and header_tokens[0].startswith("Interval data:"):
                    header_tokens[0] = header_tokens[0].replace("Interval data:", "").strip()
                    if header_tokens[0] == "":
                        header_tokens = header_tokens[1:]
                # Units → data start
                units_idx = next_nonempty(j)
                data_start = next_nonempty(units_idx)
                ncols = len(header_tokens)

                # Find the date/time columns by fuzzy match
                def find_col(options):
                    for k, name in enumerate(header_tokens):
                        low = name.lower()
                        for opt in options:
                            if opt in low:
                                return k
                    return None
                c_dt = find_col(["test start date", "start date"])  # Test Start Date and Time
                c_tod = find_col(["time of day", "time-of-day", "time ofday"])  # Time of Day
                if c_dt is None or c_tod is None or idx_val is None:
                    i = data_start
                    continue

                # Parse rows for this interval until blank, new interval, or a new box header
                base_dt: datetime | None = None
                k = data_start
                seg: list[datetime] = []
                while k < n:
                    raw = lines[k]
                    st = raw.strip()
                    if (not st or st.startswith("Interval and data points:") or
                        st.startswith("Project:") or st.startswith("Test:") or st.startswith("Result:")):
                        break
                    parts = _split_cols(raw)
                    if len(parts) > ncols and all(t == "" for t in parts[:len(parts)-ncols]):
                        parts = parts[len(parts)-ncols:]
                    if len(parts) < ncols:
                        parts += [""] * (ncols - len(parts))
                    dt_cell = parts[c_dt]
                    tod_cell = parts[c_tod]
                    if base_dt is None and dt_cell:
                        for fmt in ("%d/%m/%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
                            try:
                                base_dt = datetime.strptime(dt_cell, fmt)
                                break
                            except Exception:
                                continue
                    if base_dt is not None and tod_cell:
                        try:
                            tod = datetime.strptime(tod_cell, "%H:%M:%S").time()
                            seg.append(datetime.combine(base_dt.date(), tod))
                        except Exception:
                            pass
                    k += 1
                if seg:
                    box_intervals[idx_val] = seg
                i = k
                continue

            i += 1

        # Flush the last open box
        _flush_box()

        if not boxes_pairs:
            raise ValueError("No intervals parsed from CSV (across boxes).")

        # Build widths from accumulated pairs across boxes (chrono), reverse if end-aligned
        pairs = boxes_pairs[:]  # chrono order by scan
        if reverse_from_end:
            pairs = list(reversed(pairs))
        widths: list[int] = []
        for wt, ws in pairs:
            widths.extend([wt, ws])
        even_widths = [ws for (_, ws) in boxes_pairs]
        even_constant = all(w == even_widths[0] for w in even_widths)
        return {
            "widths": widths,
            "even_widths": even_widths,
            "even_constant": even_constant,
            "even_value": (even_widths[0] if even_constant else None),
            "num_pairs": len(even_widths),
        }

    # Unknown extension
    raise ValueError("Unsupported time source format. Pick a CSV (Anton Paar) or a SAXS time file.")
