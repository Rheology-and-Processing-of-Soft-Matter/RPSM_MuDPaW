import os
import json
import shutil
import subprocess
import tkinter as tk
from tkinter import messagebox, ttk
from common import update_json_file

# --- Path helpers ---

def _split_parts(path):
    norm = os.path.normpath(path)
    return [p for p in norm.split(os.sep) if p not in ("", ".")]


def get_reference_folder_from_path(path):
    """Parent of the nearest modality marker (PLI/PI/SAXS/Rheology). Fallback: two levels up."""
    markers = {"PLI", "PI", "SAXS", "Rheology"}
    abspath = os.path.abspath(path)
    parts = _split_parts(abspath)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            reference = os.sep.join(parts[:i])
            if reference:
                return reference
            break
    return os.path.dirname(os.path.dirname(abspath))


def get_unified_processed_folder(path):
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    os.makedirs(processed_root, exist_ok=True)
    return processed_root

# --- Prefixed file materializer ---

def _materialize_prefixed_files(files, category, out_dir):
    """Create prefixed copies (SAXS_#, PI_#, PLI_#, Rheo_#) in out_dir and return their paths.
    Does not modify the originals; ensures writer sees correctly named, existing files.
    """
    prefix_map = {"PLI": "PLI_", "PI": "PI_", "SAXS": "SAXS_", "Rheology": "Rheo_"}
    pref = prefix_map.get(category, "")
    realized = []
    counter = 1
    os.makedirs(out_dir, exist_ok=True)
    for src in files or []:
        try:
            ext = os.path.splitext(src)[1]
            dst = os.path.join(out_dir, f"{pref}{counter}{ext}") if pref else os.path.join(out_dir, os.path.basename(src))
            if os.path.abspath(src) != os.path.abspath(dst):
                shutil.copy2(src, dst)
            realized.append(dst)
            counter += 1
        except Exception as e:
            print(f"Warning: could not materialize prefixed copy for {src}: {e}")
            realized.append(src)
    return realized

# --- UI flows ---

base_path = os.path.dirname(os.path.abspath(__file__))
_SAXS_DEFAULTS_FILE = os.path.abspath(os.path.join(base_path, "..", "_saxs_defaults.json"))
_DEFAULT_METHOD = "fitting"
_VALID_METHODS = {"fitting", "direct"}
# smoothing, sigma, lower1, upper1, method, mirror, lower2, upper2, tail_fit, two_peak
_SAXS_DEFAULT_VALUES = (0.04, 0.05, 70, 170, _DEFAULT_METHOD, False, 70, 170, False, False, 0.9, False, 4, "auto", "auto")


def _load_default_file():
    try:
        with open(_SAXS_DEFAULTS_FILE, "r", encoding="utf-8") as fh:
            data = json.load(fh)
            return (
                float(data.get("smoothing", _SAXS_DEFAULT_VALUES[0])),
                float(data.get("sigma", _SAXS_DEFAULT_VALUES[1])),
                float(data.get("theta_min", _SAXS_DEFAULT_VALUES[2])),
                float(data.get("theta_max", _SAXS_DEFAULT_VALUES[3])),
                str(data.get("method", _DEFAULT_METHOD)).strip().lower() or _DEFAULT_METHOD,
                bool(data.get("mirror", _SAXS_DEFAULT_VALUES[5])),
                float(data.get("lower2", _SAXS_DEFAULT_VALUES[6])),
                float(data.get("upper2", _SAXS_DEFAULT_VALUES[7])),
                bool(data.get("tail_fit", _SAXS_DEFAULT_VALUES[8])),
                bool(data.get("two_peak", _SAXS_DEFAULT_VALUES[9])),
                float(data.get("fit_extent", _SAXS_DEFAULT_VALUES[10])),
                bool(data.get("average", _SAXS_DEFAULT_VALUES[11])),
                int(data.get("average_group", _SAXS_DEFAULT_VALUES[12])),
                str(data.get("side_mode", _SAXS_DEFAULT_VALUES[13]) or "auto"),
                str(data.get("side_mode2", _SAXS_DEFAULT_VALUES[14]) or "auto"),
            )
    except Exception:
        return _SAXS_DEFAULT_VALUES


def save_default_saxs_parameters(
    smoothing,
    sigma,
    theta_min,
    theta_max,
    method,
    mirror,
    lower2=None,
    upper2=None,
    tail_fit=None,
    two_peak=None,
    fit_extent=None,
    average=False,
    average_group=4,
    side_mode="auto",
    side_mode2="auto",
):
    payload = {
        "smoothing": float(smoothing),
        "sigma": float(sigma),
        "theta_min": float(theta_min),
        "theta_max": float(theta_max),
        "method": (str(method).strip().lower() or _DEFAULT_METHOD),
        "mirror": bool(mirror),
        "lower2": float(lower2) if lower2 is not None else float(theta_min),
        "upper2": float(upper2) if upper2 is not None else float(theta_max),
        "tail_fit": bool(tail_fit) if tail_fit is not None else False,
        "two_peak": bool(two_peak) if two_peak is not None else False,
        "fit_extent": float(fit_extent) if fit_extent is not None else float(_SAXS_DEFAULT_VALUES[10]),
        "average": bool(average),
        "average_group": int(average_group) if average_group is not None else int(_SAXS_DEFAULT_VALUES[12]),
        "side_mode": str(side_mode or "auto"),
        "side_mode2": str(side_mode2 or "auto"),
    }
    try:
        with open(_SAXS_DEFAULTS_FILE, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
    except Exception as exc:
        print(f"[SAXS] Failed to persist default parameters: {exc}")


def engage_saxs(
    path,
    sample_name,
    window,
    smoo_entry,
    sigma_entry,
    lower_limit_entry,
    upper_limit_entry,
    add_action,
    update_actions_list,
    fast_var=None,
    method_choice=_DEFAULT_METHOD,
    mirror_choice=False,
    two_peak_mode=False,
    secondary_limits=None,
    flatten_tail_fit=False,
    flatten_tail_extent=0.9,
    average_columns=False,
    average_group=4,
    centering_mode="auto",
    weak_metric="A2",
    weak_threshold=None,
    side_mode="auto",
    side_mode2=None,
):
    # Accept either live Tk widgets/BooleanVar or plain values
    def _to_float(v):
        try:
            if hasattr(v, 'get'):
                return float(v.get())
            return float(v)
        except Exception as e:
            raise ValueError(f"Expected a numeric value, got {v!r}: {e}")

    def _to_bool(b):
        try:
            if hasattr(b, 'get'):
                return bool(b.get())
            return bool(b)
        except Exception:
            return False

    def _to_method(val):
        raw = val.get() if hasattr(val, "get") else val
        raw = str(raw).strip().lower()
        if raw not in _VALID_METHODS:
            return _DEFAULT_METHOD
        return raw

    smoo_ = _to_float(smoo_entry)
    sigma_zero = _to_float(sigma_entry)
    lower_limit = _to_float(lower_limit_entry)
    upper_limit = _to_float(upper_limit_entry)
    lower_limit2 = None
    upper_limit2 = None
    if secondary_limits and len(secondary_limits) == 2:
        try:
            lower_limit2 = _to_float(secondary_limits[0])
            upper_limit2 = _to_float(secondary_limits[1])
        except Exception:
            lower_limit2 = None
            upper_limit2 = None
    fast = _to_bool(fast_var)
    method = _to_method(method_choice)
    mirror_flag = _to_bool(mirror_choice)
    tail_fit_flag = _to_bool(flatten_tail_fit)
    average_flag = _to_bool(average_columns)
    try:
        average_group_val = int(average_group.get() if hasattr(average_group, "get") else average_group)
        if average_group_val <= 0:
            average_group_val = 4
    except Exception:
        average_group_val = 4
    try:
        tail_extent_val = float(flatten_tail_extent)
    except Exception:
        tail_extent_val = 0.9

    processed_folder = get_unified_processed_folder(path)
    print(f"Unified _Processed folder for SAXS outputs: {processed_folder}")

    export_file = os.path.join(processed_folder, f"{sample_name}_export.csv")
    raw_flat_export_file = os.path.join(processed_folder, f"{sample_name}_raw_flat_export.csv")

    if os.path.exists(export_file):
        if not messagebox.askyesno("Overwrite", f"The file {sample_name}_export.csv already exists. Overwrite?"):
            window.after(0, add_action, sample_name, "SAXS processing skipped", "red")
            window.after(0, update_actions_list, window, sample_name)
            return

    def run_saxs():
        try:
            script_path = os.path.join(base_path, '../SAXS_data_processor_v4.py')
            print(f"Running {script_path} with parameters: {path, sample_name, smoo_, sigma_zero, lower_limit, upper_limit}")
            cmd = [
                'python3',
                script_path,
                path,
                sample_name,
                str(smoo_),
                str(sigma_zero),
                str(lower_limit),
                str(upper_limit),
                "--method",
                method,
            ]
            if two_peak_mode and lower_limit2 is not None and upper_limit2 is not None:
                cmd += ["--two-peak", "--lower2", str(lower_limit2), "--upper2", str(upper_limit2)]
                if tail_fit_flag:
                    cmd.append("--flatten-tail-fit")
                    cmd += ["--flatten-tail-extent", str(tail_extent_val)]
            if mirror_flag:
                cmd.append("--mirror")
            if fast:
                cmd.append("--fast")
            if average_flag:
                cmd += ["--average-columns", str(average_group_val)]
            if centering_mode:
                cmd += ["--centering-mode", str(centering_mode.get() if hasattr(centering_mode, "get") else centering_mode)]
            if weak_metric:
                cmd += ["--weak-metric", str(weak_metric.get() if hasattr(weak_metric, "get") else weak_metric)]
            if weak_threshold not in (None, ""):
                try:
                    cmd += ["--weak-threshold", str(float(weak_threshold.get() if hasattr(weak_threshold, "get") else weak_threshold))]
                except Exception:
                    pass
            if side_mode:
                cmd += ["--side-mode", str(side_mode.get() if hasattr(side_mode, "get") else side_mode)]
            if two_peak_mode and side_mode2:
                cmd += ["--side-mode2", str(side_mode2.get() if hasattr(side_mode2, "get") else side_mode2)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
            if result.returncode != 0:
                window.after(0, add_action, sample_name, "SAXS processing failed", "red")
                window.after(0, update_actions_list, window, sample_name)
                return

            if os.path.exists(export_file):
                window.after(0, add_action, sample_name, "SAXS data available", "green")
                window.after(0, update_actions_list, window, sample_name)
                new_files_list = [export_file, raw_flat_export_file]
                prefixed_files = _materialize_prefixed_files(new_files_list, "SAXS", processed_folder)
                json_path = os.path.join(processed_folder, f"_output_SAXS_{sample_name}.json")
                print(f"Writing SAXS output to unified JSON: {json_path}")
                update_json_file(json_path, sample_name, prefixed_files, overwrite=False)
            else:
                window.after(0, add_action, sample_name, "SAXS processing failed", "red")
                window.after(0, update_actions_list, window, sample_name)
        except Exception as e:
            print(f"Exception during SAXS for {sample_name} at {path}: {e}")
            window.after(0, add_action, sample_name, "SAXS processing failed", "red")
            window.after(0, update_actions_list, window, sample_name)

    run_saxs()


def process_all_saxs_files(path, sample_folders, window, add_action, update_actions_list, params=None, fast=False):
    def run_all_saxs(
        smoo_,
        sigma_zero,
        lower_limit,
        upper_limit,
        method,
        mirror_flag,
        two_peak_flag=False,
        lower2=None,
        upper2=None,
        tail_fit=False,
        fit_extent=0.9,
        fast_flag=fast,
        average_flag=False,
        average_group_val=4,
        side_mode="auto",
        side_mode2=None,
    ):
        for sample in sample_folders:
            sample_path = os.path.join(path, sample)
            sample_name = sample
            processed_folder = get_unified_processed_folder(sample_path)
            print(f"Unified _Processed folder for SAXS outputs: {processed_folder}")
            export_file = os.path.join(processed_folder, f"{sample_name}_export.csv")
            raw_flat_export_file = os.path.join(processed_folder, f"{sample_name}_raw_flat_export.csv")
            if os.path.exists(export_file):
                if not messagebox.askyesno("Overwrite", f"The file {sample_name}_export.csv already exists. Overwrite?"):
                    window.after(0, add_action, sample_name, "SAXS processing skipped", "red")
                    window.after(0, update_actions_list, window, sample_name)
                    continue
            try:
                script_path = os.path.join(base_path, '../SAXS_data_processor_v4.py')
                print(f"Running {script_path} for {sample_name} with parameters: {sample_path, sample_name, smoo_, sigma_zero, lower_limit, upper_limit}")
                cmd = [
                    'python3',
                    script_path,
                    sample_path,
                    sample_name,
                    str(smoo_),
                    str(sigma_zero),
                    str(lower_limit),
                    str(upper_limit),
                    "--method",
                    method,
                ]
                if two_peak_flag and lower2 is not None and upper2 is not None:
                    cmd += ["--two-peak", "--lower2", str(lower2), "--upper2", str(upper2)]
                    if tail_fit:
                        cmd.append("--flatten-tail-fit")
                        cmd += ["--flatten-tail-extent", str(fit_extent)]
                if mirror_flag:
                    cmd.append("--mirror")
                if fast_flag:
                    cmd.append("--fast")
                if average_flag:
                    cmd += ["--average-columns", str(int(average_group_val)) if average_group_val is not None else "4"]
                if side_mode:
                    cmd += ["--side-mode", str(side_mode)]
                if two_peak_flag and side_mode2:
                    cmd += ["--side-mode2", str(side_mode2)]
                result = subprocess.run(cmd, capture_output=True, text=True)
                print(result.stdout)
                if result.stderr:
                    # Log warnings/errors from the processor but do not signal failure in UI
                    print(result.stderr)
                if result.returncode != 0:
                    # Non-zero exit: log and continue without UI failure banner
                    print(f"SAXS processor exited with code {result.returncode} for {sample_name}")
                    continue
                if os.path.exists(export_file):
                    window.after(0, add_action, sample_name, "SAXS data available", "green")
                    window.after(0, update_actions_list, window, sample_name)
                    new_files_list = [export_file, raw_flat_export_file]
                    prefixed_files = _materialize_prefixed_files(new_files_list, "SAXS", processed_folder)
                    json_path = os.path.join(processed_folder, f"_output_SAXS_{sample_name}.json")
                    print(f"Writing SAXS output to unified JSON: {json_path}")
                    update_json_file(json_path, sample_name, prefixed_files, overwrite=False)
                else:
                    # No export produced; do not flag red in UI. Log and continue.
                    print(f"No export file produced for {sample_name} (expected {export_file})")
            except Exception as e:
                # Log exceptions but avoid red UI signaling in batch mode
                print(f"Exception during SAXS for {sample_name} at {sample_path}: {e}")
                continue

    if params:
        smoo_, sigma_zero, lower_limit, upper_limit, method, mirror_flag = params
        run_all_saxs(smoo_, sigma_zero, lower_limit, upper_limit, method, mirror_flag, fast_flag=fast)
        return

    defaults = get_last_used_saxs_parameters()
    smoo_def, sigma_def, lower_def, upper_def, method_def, mirror_def, lower2_def, upper2_def, tail_fit_def, two_peak_def, fit_extent_def, avg_def, avg_grp_def, side_mode_def, side_mode2_def = defaults

    param_window = tk.Toplevel(window)
    param_window.title("Set Parameters for All SAXS Files")

    ttk.Label(param_window, text="Smoothing value (default: 0.01):").pack()
    smoo_entry = ttk.Entry(param_window)
    smoo_entry.insert(0, str(smoo_def))
    smoo_entry.pack()

    ttk.Label(param_window, text="Sigma zero (default: 0.05):").pack()
    sigma_entry = ttk.Entry(param_window)
    sigma_entry.insert(0, str(sigma_def))
    sigma_entry.pack()

    ttk.Label(param_window, text="Lower limit (default: 1):").pack()
    lower_limit_entry = ttk.Entry(param_window)
    lower_limit_entry.insert(0, str(lower_def))
    lower_limit_entry.pack()

    ttk.Label(param_window, text="Upper limit (default: 180):").pack()
    upper_limit_entry = ttk.Entry(param_window)
    upper_limit_entry.insert(0, str(upper_def))
    upper_limit_entry.pack()

    ttk.Label(param_window, text="Peak 2 lower (deg):").pack()
    lower2_entry = ttk.Entry(param_window)
    lower2_entry.insert(0, str(lower2_def))
    lower2_entry.pack()

    ttk.Label(param_window, text="Peak 2 upper (deg):").pack()
    upper2_entry = ttk.Entry(param_window)
    upper2_entry.insert(0, str(upper2_def))
    upper2_entry.pack()

    fast_var = tk.BooleanVar(value=fast)
    ttk.Checkbutton(param_window, text="Fast (no plots)", variable=fast_var).pack(pady=(4, 6))

    ttk.Label(param_window, text="Method:").pack()
    method_var = tk.StringVar(value=method_def)
    method_frame = ttk.Frame(param_window)
    method_frame.pack(pady=(0, 6))
    ttk.Radiobutton(method_frame, text="Fitting", value="fitting", variable=method_var).pack(side=tk.LEFT, padx=4)
    ttk.Radiobutton(method_frame, text="Direct", value="direct", variable=method_var).pack(side=tk.LEFT, padx=4)
    mirror_var = tk.BooleanVar(value=mirror_def)
    ttk.Checkbutton(param_window, text="Mirror azimuthal data", variable=mirror_var).pack(pady=(0, 6))
    two_peak_var = tk.BooleanVar(value=two_peak_def)
    ttk.Checkbutton(param_window, text="Enable two-peak mode", variable=two_peak_var).pack(pady=(0, 4))
    tail_fit_var = tk.BooleanVar(value=tail_fit_def)
    ttk.Checkbutton(param_window, text="Extrapolate tail with LG fit", variable=tail_fit_var).pack(pady=(0, 4))
    ttk.Label(param_window, text="Fit extent (0-1, fraction of pre-min segment)").pack()
    fit_extent_entry = ttk.Entry(param_window)
    fit_extent_entry.insert(0, str(fit_extent_def))
    fit_extent_entry.pack()

    def on_confirm():
        smoo_ = float(smoo_entry.get())
        sigma_zero = float(sigma_entry.get())
        lower_limit = float(lower_limit_entry.get())
        upper_limit = float(upper_limit_entry.get())
        lower2_val = float(lower2_entry.get())
        upper2_val = float(upper2_entry.get())
        fast_flag = bool(fast_var.get())
        method = str(method_var.get()).strip().lower()
        mirror_flag = bool(mirror_var.get())
        two_peak_flag = bool(two_peak_var.get())
        tail_fit_flag = bool(tail_fit_var.get())
        fit_extent_val = float(fit_extent_entry.get())
        param_window.destroy()
        run_all_saxs(
            smoo_,
            sigma_zero,
            lower_limit,
            upper_limit,
            method,
            mirror_flag,
            two_peak_flag=two_peak_flag,
            lower2=lower2_val,
            upper2=upper2_val,
            tail_fit=tail_fit_flag,
            fit_extent=fit_extent_val,
            fast_flag=fast_flag,
        )

    ttk.Button(param_window, text="Run All", command=on_confirm).pack(pady=(0, 6))


def get_last_used_saxs_parameters():
    return _load_default_file()


def process_sample_SAXS(window, path, subfolder_name, sample_name, add_action, update_actions_list, load_last_folder=None):
    """
    UI entry point expected by common.process_sample(...). Builds a small params dialog
    and then calls engage_saxs(...) with the collected values.
    Signature keeps compatibility with other helpers even if some args are unused here.
    """
    # Defaults (can later be replaced by persisted values if desired)
    (
        smoo_default,
        sigma_default,
        lower_default,
        upper_default,
        method_default,
        mirror_default,
        lower2_default,
        upper2_default,
        tail_fit_default,
        two_peak_default,
        _fit_extent_default,
        average_default,
        average_group_default,
        side_mode_default,
        side_mode2_default,
    ) = get_last_used_saxs_parameters()

    param_window = tk.Toplevel(window)
    param_window.title(f"SAXS parameters — {sample_name}")

    frm = ttk.Frame(param_window, padding=10)
    frm.pack(fill=tk.BOTH, expand=True)

    # Smoothing
    ttk.Label(frm, text="Smoothing value (e.g., 0.01)").grid(row=0, column=0, sticky="w", padx=4, pady=4)
    smoo_entry = ttk.Entry(frm, width=12)
    smoo_entry.insert(0, str(smoo_default))
    smoo_entry.grid(row=0, column=1, sticky="w", padx=4, pady=4)

    # Sigma zero
    ttk.Label(frm, text="Sigma zero (e.g., 0.05)").grid(row=1, column=0, sticky="w", padx=4, pady=4)
    sigma_entry = ttk.Entry(frm, width=12)
    sigma_entry.insert(0, str(sigma_default))
    sigma_entry.grid(row=1, column=1, sticky="w", padx=4, pady=4)

    # Lower / Upper limits
    ttk.Label(frm, text="Lower limit (deg)").grid(row=2, column=0, sticky="w", padx=4, pady=4)
    lower_limit_entry = ttk.Entry(frm, width=12)
    lower_limit_entry.insert(0, str(lower_default))
    lower_limit_entry.grid(row=2, column=1, sticky="w", padx=4, pady=4)

    ttk.Label(frm, text="Upper limit (deg)").grid(row=3, column=0, sticky="w", padx=4, pady=4)
    upper_limit_entry = ttk.Entry(frm, width=12)
    upper_limit_entry.insert(0, str(upper_default))
    upper_limit_entry.grid(row=3, column=1, sticky="w", padx=4, pady=4)

    # Method
    ttk.Label(frm, text="Method").grid(row=4, column=0, sticky="w", padx=4, pady=(4, 2))
    method_var = tk.StringVar(value=method_default)
    method_opts = ttk.Frame(frm)
    method_opts.grid(row=4, column=1, sticky="w", padx=4, pady=(4, 2))
    ttk.Radiobutton(method_opts, text="Fitting", value="fitting", variable=method_var).pack(side=tk.LEFT)
    ttk.Radiobutton(method_opts, text="Direct", value="direct", variable=method_var).pack(side=tk.LEFT, padx=(6, 0))
    mirror_var = tk.BooleanVar(value=mirror_default)
    ttk.Checkbutton(frm, text="Mirror azimuthal data", variable=mirror_var).grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 4))

    # Fast mode checkbox (optional; wired to engage_saxs)
    fast_var = tk.BooleanVar(value=False)
    fast_chk = ttk.Checkbutton(frm, text="Fast (no plots)", variable=fast_var)
    fast_chk.grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=(0, 8))

    # Buttons
    btns = ttk.Frame(frm)
    btns.grid(row=7, column=0, columnspan=2, sticky="ew")
    btns.columnconfigure(0, weight=1)
    btns.columnconfigure(1, weight=1)

    def on_engage():
        try:
            # Read values BEFORE destroying the dialog so widgets can be safely GC’d
            smoo_val = float(smoo_entry.get())
            sigma_val = float(sigma_entry.get())
            lower_val = float(lower_limit_entry.get())
            upper_val = float(upper_limit_entry.get())
            fast_flag = bool(fast_var.get())
            method_choice = method_var.get().strip().lower()
            mirror_choice = bool(mirror_var.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter numeric values for all parameters.")
            return

        # Close the dialog now that values are captured
        param_window.destroy()

        # Delegate to the runner using plain values (works with both old & new engage_saxs)
        engage_saxs(
            path,
            sample_name,
            window,
            smoo_val,
            sigma_val,
            lower_val,
            upper_val,
            add_action,
            update_actions_list,
            fast_var=fast_flag,
            method_choice=method_choice,
            mirror_choice=mirror_choice,
        )

    def on_cancel():
        param_window.destroy()
        window.after(0, add_action, sample_name, "SAXS canceled", "red")
        window.after(0, update_actions_list, window, sample_name)

    ttk.Button(btns, text="Engage", command=on_engage).grid(row=0, column=0, sticky="ew", padx=(0, 4))
    ttk.Button(btns, text="Cancel", command=on_cancel).grid(row=0, column=1, sticky="ew", padx=(4, 0))

    # Make the dialog modal-ish
    param_window.transient(window)
    try:
        param_window.grab_set()
    except Exception:
        pass
    param_window.focus_set()
