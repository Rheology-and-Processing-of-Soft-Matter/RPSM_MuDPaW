
import os
import sys
import json
from tkinter import messagebox
import numpy as np
import pandas as pd
import cv2
from _Routines.common import normalize_modal_json

# --- Path hygiene helper (fixes accidentally concatenated absolute paths) ---

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
        print(f"[PLI] sanitize: detected misjoined path →\n  in : {p}\n  out: {fixed}")
        return fixed
    except Exception:
        return p

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    """Resolve a *reference* folder robustly and predictably.

    Rules:
    - Normalize `path` with realpath/abspath.
    - If inside an existing "_Processed" folder, return its parent (prevents nesting `_Processed/_Processed`).
    - Else, if a modality folder ("PLI", "PI", "SAXS", "Rheology") is present, return the parent of that folder.
      If that modality sits directly under `_Processed`, step one higher (parent of `_Processed`).
    - Otherwise, if `path` points to a file, use its directory; if it is a directory, use it as-is.
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    # Normalize
    path = os.path.realpath(path)
    abspath = _sanitize_misjoined_user_path(os.path.abspath(path))
    if os.path.isfile(abspath):
        abspath = os.path.dirname(abspath)

    parts = _split_parts(abspath)

    # Anchor to the parent of an existing _Processed, if present
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "_Processed":
            reference = os.sep.join(parts[:i])
            reference = os.path.abspath(reference)
            if not reference.startswith(os.sep):
                reference = os.sep + reference
            return reference if reference else os.sep

    # Next, search for a modality folder
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            if i - 1 >= 0 and parts[i - 1] == "_Processed":
                reference = os.sep.join(parts[: i - 1])
            else:
                reference = os.sep.join(parts[:i])
            reference = os.path.abspath(reference)
            if not reference.startswith(os.sep):
                reference = os.sep + reference
            return reference if reference else os.sep

    # Fallback: the directory itself
    abspath = os.path.abspath(abspath)
    if not abspath.startswith(os.sep):
        abspath = os.sep + abspath
    return abspath


def get_unified_processed_folder(path):
    """Return `<reference>/_Processed/PLI` for PLI outputs, creating it if needed."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed", "PLI")
    processed_root = _sanitize_misjoined_user_path(processed_root)
    if not processed_root.startswith(os.sep):
        processed_root = os.sep + processed_root
    processed_root = os.path.abspath(processed_root)
    print(f"[PLI] unified processed root → reference: {reference_folder}\n                                   processed: {processed_root}")
    os.makedirs(processed_root, exist_ok=True)
    return processed_root

# --- Unscaled Stitched Panel Discovery in _Temp ---

def get_temp_unscaled_folder(path: str) -> str:
    """Return `<reference>/PLI/_Temp` for unscaled stitched panels, creating it if needed."""
    reference_folder = get_reference_folder_from_path(path)
    temp_root = os.path.join(reference_folder, "PLI", "_Temp")
    temp_root = _sanitize_misjoined_user_path(temp_root)
    if not temp_root.startswith(os.sep):
        temp_root = os.sep + temp_root
    temp_root = os.path.abspath(temp_root)
    os.makedirs(temp_root, exist_ok=True)
    return temp_root


def list_unscaled_temp_stitched_outputs(path: str) -> list[str]:
    """Return absolute paths to unscaled stitched space–time PLI images under `<reference>/PLI/_Temp`.
    Matches files like `*_steady_unscaled_stitched_*x*.png`.
    """
    results: list[str] = []
    try:
        temp_root = get_temp_unscaled_folder(path)
        print(f"[PLI] list_unscaled_temp_stitched_outputs: scanning only: {temp_root}")
        if not os.path.isdir(temp_root):
            return results
        sample = sorted(os.listdir(temp_root))[:10]
        print(f"[PLI] dir sample(PLI/_Temp): {sample}")
        for fname in sorted(os.listdir(temp_root)):
            fl = fname.lower()
            if fl.endswith('.png') and '_steady_unscaled_stitched_' in fl and not fl.startswith('.'):
                full = os.path.join(temp_root, fname)
                results.append(full)
        print(f"[PLI] list_unscaled_temp_stitched_outputs: total found={len(results)}")
        for ex in list(sorted(results))[:5]:
            print("   example:", ex)
    except Exception as e:
        try:
            print(f"[PLI] list_unscaled_temp_stitched_outputs: error: {e}")
        except Exception:
            pass
    return sorted(results)


# --- UI List Helpers for PLI Space–Time Artifacts ---

def list_raw_st_images(path: str) -> list[str]:
    """Return raw space–time PLI images in `path`, excluding stitched/scaled/gray variants.
    A raw ST image is any PNG that contains `_st_` in the name, but NOT `_stitched_`, NOT `_scaled_`, and NOT the `_gray` variant.
    """
    items: list[str] = []
    try:
        for f in sorted(os.listdir(path)):
            fl = f.lower()
            if (
                fl.endswith(".png")
                and "_st_" in fl
                and "_stitched_" not in fl
                and "_scaled_" not in fl
                and not fl.endswith("_gray.png")
            ):
                items.append(f)
    except Exception:
        pass
    return items



def list_stitched_outputs(path: str) -> list[str]:
    results: list[str] = []
    try:
        processed_root = get_unified_processed_folder(path)
    except Exception:
        return results

    try:
        print(f"[PLI] list_stitched_outputs: scanning only: {processed_root}")
        if not os.path.isdir(processed_root):
            print(f"[PLI] list_stitched_outputs: not a directory → {processed_root}")
            return results

        # Ignore accidental stray 'Users' folder inside processed_root (from older cwd side-effects)
        if os.path.isdir(os.path.join(processed_root, 'Users')):
            print("[PLI] warning: ignoring stray 'Users' subfolder under _Processed/PLI (cleanup from older runs)")

        # Directory sample for debugging
        try:
            sample = sorted(os.listdir(processed_root))[:10]
            print(f"[PLI] dir sample(_Processed/PLI): {sample}")
        except Exception as _e_ls:
            print(f"[PLI] warning: cannot list directory {processed_root}: {_e_ls}")

        for fname in sorted(os.listdir(processed_root)):
            fl = fname.lower()
            # Only stitched, color variants; exclude gray and raw `_st_` outputs
            if (
                fl.endswith('.png')
                and '_stitched_' in fl
                and '_gray' not in fl
            ):
                full = os.path.join(processed_root, fname)
                results.append(full)
        print(f"[PLI] list_stitched_outputs: total found={len(results)}")
        for ex in list(sorted(results))[:5]:
            print("   example:", ex)
    except Exception as e:
        try:
            print(f"[PLI] list_stitched_outputs: error: {e}")
        except Exception:
            pass
    return sorted(results)

# --- DG-ready CSV discovery (from master JSON written by PLI_data_processor) ---

def list_datagraph_ready_csvs(path: str) -> list[str]:
    """Return absolute paths to DG-ready CSVs under <reference>/_Processed/PLI.
    Looks for JSON entries with either `datagraph_ready: true` and a `datagraph_csv` path,
    or a top-level `dg_exports` list under each dataset entry.
    """
    out: list[str] = []
    try:
        processed_root = get_unified_processed_folder(path)
    except Exception:
        return out

    # Scan JSON files for the expected markers
    try:
        for fname in os.listdir(processed_root):
            if not fname.lower().endswith('.json'):
                continue
            fpath = os.path.join(processed_root, fname)
            try:
                with open(fpath, 'r') as fh:
                    data = json.load(fh)
            except Exception:
                continue

            # Normalize paths (absolute + relative) for PLI-specific keys
            try:
                reference_folder = get_reference_folder_from_path(path)
                data = normalize_modal_json(data, reference_folder, keys=('datagraph_csv', 'dg_exports'))
            except Exception:
                pass

            # Check direct fields (simple case): datagraph_ready + datagraph_csv (str or list)
            if isinstance(data, dict) and data.get('datagraph_ready') and data.get('datagraph_csv'):
                p = data.get('datagraph_csv')
                if isinstance(p, str) and os.path.isfile(p):
                    out.append(os.path.abspath(p))
                    continue
                if isinstance(p, list):
                    for q in p:
                        if isinstance(q, str) and os.path.isfile(q):
                            out.append(os.path.abspath(q))
                    continue

            # Check flat exports list if present (already normalized to absolutes)
            if isinstance(data, dict):
                exp = data.get('dg_exports')
                if isinstance(exp, list):
                    for p in exp:
                        if isinstance(p, str) and os.path.isfile(p):
                            out.append(os.path.abspath(p))
    except Exception:
        pass

    # De-duplicate and sort
    seen = set()
    uniq = []
    for p in out:
        q = os.path.abspath(p)
        if q not in seen:
            seen.add(q)
            uniq.append(q)
    return sorted(uniq)

# --- PLI Interactive extraction launcher ---

def extract_space_time_pli(path, files):
    """Launch the interactive space–time extractor for each selected video file."""
    import subprocess
    processed_folder = get_unified_processed_folder(path)
    print(f"Unified _Processed folder for PLI outputs: {processed_folder}")
    filepaths = []
    for file in files:
        filepath = file if os.path.isabs(file) else os.path.join(path, file)
        if not filepath.lower().endswith((".mp4", ".avi", ".mov")):
            raise ValueError(f"Only video files are supported for interactive extraction: {file}")
        filepaths.append(filepath)
    helper_dir = os.path.dirname(__file__)
    script_path = os.path.join(os.path.dirname(helper_dir), "_Routines", "PLI", "PLI_extract_st_diag_qt.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Extractor script not found at {script_path}")
    print(f"Launching extractor for: {filepaths}")
    subprocess.Popen([sys.executable, script_path] + filepaths, cwd=path)

# --- PLI Rescale window launcher ---

def rescale_space_time_pli(path, files):
    """Launch the interactive intervals & rescaling window for selected space–time images."""
    import subprocess
    helper_dir = os.path.dirname(__file__)
    script_path = os.path.join(os.path.dirname(helper_dir), "_Routines", "PLI", "PLI_rescale_space_time_diagram.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Rescaling GUI script not found at {script_path}")
    # Build full paths for each selected image
    image_paths = [os.path.join(path, f) for f in files] if files else [path]
    print(f"Launching PLI rescaling interface for: {image_paths}")
    subprocess.Popen([sys.executable, script_path] + image_paths, cwd=path)

# --- Steady-State Extraction for Space–Time PLI Images ---

def steady_state_extraction_pli(selected_image):
    """Run steady-state extraction routine for a selected space–time diagram image."""
    import subprocess
    helper_dir = os.path.dirname(__file__)
    script_path = os.path.join(helper_dir, "PLI_space_time_steady_state_extractor.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Extractor script not found at {script_path}")
    print(f"Running steady-state extraction for: {selected_image}")
    subprocess.Popen([sys.executable, script_path, selected_image])



# --- Scaled-data Processor Launcher for Space–Time PLI Images ---

def process_scaled_outputs_pli(stitched_path: str, intervals_json: str | None = None, fps: float | None = None):
    """Launch the scaled-data processor for a stitched color space–time image.

    Parameters
    ----------
    stitched_path : str
        Full path to a `_stitched_*.png` (color) file.
    intervals_json : str | None
        Optional path to a unified `_output_PLI_*.json` containing an `intervals` block. If not provided,
        the function will try to auto-discover one under `<reference>/_Processed/PLI`.
    fps : float | None
        Optional FPS override used by the processor when widths must be recomputed from a time source.
    """
    import subprocess

    helper_dir = os.path.dirname(__file__)
    # Use the unified processor script.
    script_path = os.path.join(os.path.dirname(helper_dir), "_Routines", "PLI", "PLI_data_processor.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"PLI_data_processor not found at {script_path}")

    # If intervals JSON not provided, try to auto-discover under unified _Processed/PLI
    if intervals_json is None:
        try:
            ref = get_reference_folder_from_path(stitched_path)
            processed = os.path.join(ref, "_Processed", "PLI")
            if os.path.isdir(processed):
                for fname in sorted(os.listdir(processed)):
                    if fname.startswith("_output_PLI_") and fname.endswith(".json"):
                        # quick sanity check: file contains an 'intervals' block
                        cand = os.path.join(processed, fname)
                        try:
                            with open(cand, "r") as fh:
                                data = json.load(fh)
                            if isinstance(data, dict) and "intervals" in data:
                                intervals_json = cand
                                break
                        except Exception:
                            continue
        except Exception:
            pass

    # Derive sample_name from the stitched filename (without extension)
    sample_name = os.path.splitext(os.path.basename(stitched_path))[0]

    args = [sys.executable, script_path, stitched_path, sample_name]
    if intervals_json:
        args += ["--intervals-json", intervals_json]
    if fps is not None:
        args += ["--fps", str(float(fps))]

    print(
        "[PLI] Running PLI_data_processor on stitched image\n  image: {}\n  intervals: {}\n".format(stitched_path, intervals_json)
    )
    subprocess.Popen(args, cwd=os.path.dirname(stitched_path) or ".")

def load_intervals_pli(path):
    """
    Load previously saved PLI intervals from the unified _Processed folder.
    Looks for a JSON with an 'intervals' block (e.g., _output_PLI_<sample>.json) and reports what it found.
    """
    try:
        processed = get_unified_processed_folder(path)
        candidates = []
        if os.path.isdir(processed):
            for fname in os.listdir(processed):
                if fname.lower().endswith('.json'):
                    fpath = os.path.join(processed, fname)
                    try:
                        with open(fpath, 'r') as fh:
                            data = json.load(fh)
                        if isinstance(data, dict) and 'intervals' in data and isinstance(data['intervals'], dict):
                            candidates.append((fname, fpath, data['intervals']))
                    except Exception:
                        continue
        if not candidates:
            messagebox.showinfo("Load intervals", f"No saved PLI interval JSON found in:\n{processed}\n\nRun the rescale workflow to generate intervals first.")
            return
        # Prefer files with '_output_PLI_' in the name
        candidates.sort(key=lambda t: ("_output_PLI_" not in t[0], t[0].lower()))
        fname, fpath, intervals = candidates[0]
        widths = intervals.get('widths') if isinstance(intervals, dict) else None
        n_pairs = len(widths)//2 if isinstance(widths, list) else None
        msg = [
            f"Loaded: {fname}",
            f"Location: {fpath}",
        ]
        if n_pairs:
            msg.append(f"Pairs: {n_pairs}  (total widths: {len(widths)})")
        messagebox.showinfo("Load intervals", "\n".join(msg))
    except Exception as e:
        messagebox.showerror("Load intervals", f"Failed to load intervals from _Processed:\n{e}")


# --- Time-source auto-discovery & interval helpers ---

def _auto_find_time_source_in_reference(reference_root: str) -> str | None:
    """Search the reference folder for a suitable time source.
    Priority: Rheology/*.csv with 'Interval data:' → SAXS/time*.txt|*.dat|*.time.
    Returns the first match or None.
    """
    try:
        # 1) Rheology CSVs (prefer Anton Paar exports)
        rheo_dir = os.path.join(reference_root, "Rheology")
        if os.path.isdir(rheo_dir):
            csvs = sorted([f for f in os.listdir(rheo_dir) if f.lower().endswith('.csv')])
            for f in csvs:
                p = os.path.join(rheo_dir, f)
                # quick header peek: try UTF‑16 then UTF‑8
                for enc in ("utf-16", "utf-8"):
                    try:
                        with open(p, "r", encoding=enc) as fh:
                            head = fh.read(4096)
                        if "Interval data:" in head:
                            return p
                    except Exception:
                        continue
        # 2) SAXS time files (time*.txt/.dat/.time)
        saxs_dir = os.path.join(reference_root, "SAXS")
        if os.path.isdir(saxs_dir):
            for f in sorted(os.listdir(saxs_dir)):
                fl = f.lower()
                if fl.startswith("time") and (fl.endswith('.txt') or fl.endswith('.dat') or fl.endswith('.time')):
                    return os.path.join(saxs_dir, f)
    except Exception:
        return None
    return None


def verify_constant_even(widths_even: list) -> tuple[bool, int | None]:
    """Return (is_constant, value_if_constant) for steady (even) interval widths."""
    if not widths_even:
        return False, None
    ok = all(w == widths_even[0] for w in widths_even)
    return ok, (widths_even[0] if ok else None)


def pick_and_compute_pair_widths(sample_dir: str, fps: float, reverse_from_end: bool = True, initialdir: str | None = None) -> dict:
    """Auto-discover a time source under the reference folder and compute interval pair widths (no dialogs).
    Returns a dict with keys: widths, even_widths, even_constant, even_value, num_pairs.
    """
    # Resolve initial directory to the reference root
    if initialdir is None:
        try:
            initialdir = get_reference_folder_from_path(sample_dir)
        except Exception:
            initialdir = sample_dir

    # Auto-discover the time source
    src = _auto_find_time_source_in_reference(initialdir)
    if not src:
        raise RuntimeError(f"No time source found under reference folder: {initialdir}")

    # Compute widths using the PLI stamper core (import locally to avoid cycles)
    try:
        from Time_stamper_PLI_v1 import compute_pair_widths_from_times
        print(f"[PLI helper] Using time source: {src}")
        return compute_pair_widths_from_times(src, fps, reverse_from_end=reverse_from_end)
    except Exception as e:
        raise RuntimeError(f"Failed to compute widths from time source: {e}")
#
# --- Distribution Analysis Viewer for stitched PLI outputs ---
import tkinter as _tk
from tkinter import ttk as _ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


def _load_lab_arrays(_png_path: str):
    img_bgr = cv2.imread(_png_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {_png_path}")
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(lab)
    L = L.astype(np.float32)
    a = a.astype(np.float32) - 128.0
    b = b.astype(np.float32) - 128.0
    return L, a, b


def _compute_param(L: np.ndarray, a: np.ndarray, b: np.ndarray, mode: str) -> np.ndarray:
    mode = (mode or "L").lower()
    if mode == "l":
        return L
    if mode == "a":
        return a
    if mode == "b":
        return b
    if mode in ("sqrt(a^2+b^2)", "chroma", "ab_mag"):
        return np.sqrt(a * a + b * b)
    if mode in ("sqrt(l^2+a^2+b^2)", "lab_mag"):
        return np.sqrt(L * L + a * a + b * b)
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
    n_intervals = max(1, int(np.ceil(H / float(max(1, int(interval_height))))))

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
        # column-wise mean across rows y0:y1
        prof = np.mean(M[y0:y1, :], axis=0)
        x = np.linspace(0.0, 1.0, prof.size)
        ax.clear()
        ax.plot(x, prof)
        ax.set_xlabel("L1,norm [−]")
        ax.set_ylabel("{} [a.u.]".format(mode))
        ax.grid(True, which="both", alpha=0.3)
        canvas.draw_idle()

    def _save_csv():
        mode = param_cmb.get()
        k = max(0, min(n_intervals - 1, int(interval_var.get())))
        y0 = int(k * interval_height)
        y1 = int(min(H, y0 + interval_height))
        M = _compute_param(L, a, b, mode)
        prof = np.mean(M[y0:y1, :], axis=0)
        out_dir = get_unified_processed_folder(stitched_path)
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(stitched_path))[0]
        out_csv = os.path.join(out_dir, f"_pli_dist_{base}_interval{k}_{mode.replace('/', '').replace('^', '').replace('*', '')}.csv")
        pd.DataFrame({"x_norm": np.linspace(0.0, 1.0, prof.size), "value": prof}).to_csv(out_csv, index=False)
        print(f"Saved distribution CSV: {out_csv}")

    _ttk.Button(btn_frm, text="Update", command=_update_plot).pack(side=_tk.LEFT)
    _ttk.Button(btn_frm, text="Save CSV", command=_save_csv).pack(side=_tk.LEFT, padx=6)

    # reactive updates
    param_cmb.bind("<<ComboboxSelected>>", _update_plot)
    interval_scl.configure(command=lambda *_: _update_plot())

    _update_plot()
    win.lift()
    win.focus_force()

# --- Process PLI folder GUI (snippet for scaled outputs widgets) ---
# (elsewhere in the code, in the window for "Process PLI folder", find and update this section:)
#    scaled_label = ttk.Label(root, text="Scaled space–time diagram outputs:")
#    scaled_label.pack()
#    scaled_listbox = tk.Listbox(root, height=5)
#    scaled_listbox.pack(fill=tk.X, padx=20)
#    refresh_scaled_btn = ttk.Button(root, text="Refresh scaled outputs", command=refresh_scaled_outputs)
#    refresh_scaled_btn.pack(pady=5)
#
# Replace with:
#    refresh_scaled_btn = ttk.Button(root, text="Refresh scaled outputs", command=refresh_scaled_outputs)
#    refresh_scaled_btn.pack(pady=(10, 3))
#
#    scaled_label = ttk.Label(root, text="Scaled space–time diagram outputs:")
#    scaled_label.pack()
#    scaled_listbox = tk.Listbox(root, height=5)
#    scaled_listbox.pack(fill=tk.X, padx=20)
#
# New: prefer unscaled stitched panels from PLI/_Temp for processing
# def refresh_scaled_outputs():
#     lst = list_unscaled_temp_stitched_outputs(current_folder)
#     scaled_listbox.delete(0, tk.END)
#     for p in lst:
#         scaled_listbox.insert(tk.END, p)
#     print(f"[PLI] refreshed unscaled stitched list: {len(lst)} items")
