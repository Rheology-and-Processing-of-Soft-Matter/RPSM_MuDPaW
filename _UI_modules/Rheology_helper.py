import os
import subprocess
#import threading
from tkinter import messagebox
import tkinter as tk
from tkinter import ttk
import json
import sys

base_path = os.path.dirname(os.path.abspath(__file__))

# --- Unified _Processed Folder Helpers ---

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    """Resolve the *reference* folder robustly.

    Walks up the path to find known modality markers ("PLI", "PI", "SAXS", "Rheology").
    If the path is inside a `_Processed/<modality>` subtree, we step out of `_Processed` first.
    Handles accidental duplicated prefixes like `.../Users/.../Users/...` by trimming to the last occurrence.
    Fallback: return the parent directory of the provided path.
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    # Normalize and collapse any accidental double "/Users/.../Users/..." patterns
    abspath = os.path.realpath(path)
    abspath = os.path.abspath(abspath)

    # If a file was provided, use its directory
    if os.path.isfile(abspath):
        abspath = os.path.dirname(abspath)

    # If we have a duplicated "/Users/.../Users/..." path, keep the tail from the last "Users"
    try:
        tok = os.sep + "Users" + os.sep
        first = abspath.find(tok)
        if first != -1:
            second = abspath.find(tok, first + 1)
            if second != -1:
                abspath = abspath[second:]
                if not abspath.startswith(os.sep):
                    abspath = os.sep + abspath
                print(f"[Rheology] sanitize: detected misjoined path →\n  normalized: {abspath}")
    except Exception:
        pass

    parts = _split_parts(abspath)

    # Prefer an explicit `_Processed` anchor if present: reference is everything above it
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "_Processed":
            reference = os.path.join(os.sep, *parts[:i])
            print(f"[Rheology] resolved reference via _Processed: {reference}")
            return reference

    # Otherwise, look for a known modality folder and return its parent
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            reference = os.path.join(os.sep, *parts[:i])
            print(f"[Rheology] resolved reference via marker '{parts[i]}': {reference}")
            return reference

    # Fallback: the parent directory of the given path
    fallback = os.path.dirname(abspath)
    print(f"[Rheology] fallback reference: {fallback}")
    return fallback


def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder.

    Ensures the directory exists and returns its absolute path.
    """
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    os.makedirs(processed_root, exist_ok=True)
    print(f"[Rheology] unified processed root → reference: {reference_folder}\n                                   processed: {processed_root}")
    return processed_root


# --- Session save/load for options ---

def _rheo_session_path(reference_folder: str) -> str:
    processed_root = os.path.join(reference_folder, "_Processed", "_Params")
    os.makedirs(processed_root, exist_ok=True)
    return os.path.join(processed_root, "_rheo_reader_session.json")


def _rheo_save_session(reference_folder: str, data: dict) -> None:
    try:
        with open(_rheo_session_path(reference_folder), "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Rheology] session saved → {_rheo_session_path(reference_folder)}")
    except Exception as e:
        print("[Rheology] session save failed:", e)


def _rheo_load_session(reference_folder: str) -> dict:
    try:
        with open(_rheo_session_path(reference_folder), "r") as f:
            return json.load(f)
    except Exception:
        return {}


# --- Launcher ---

def _launch_viscosity_reader(csv_path: str, mode: str, steady_sec: float) -> None:
    try:
        # Script sits one level up from _Helpers
        script_path = os.path.abspath(os.path.join(base_path, '../Read_viscosity_v3.py'))
        if not os.path.isfile(script_path):
            messagebox.showerror("Rheology", f"Viscosity reader not found:\n{script_path}")
            return
        args = [sys.executable or 'python3', script_path, csv_path,
                "--mode", mode,
                "--steady-sec", str(steady_sec)]
        print("[Rheology] Launching viscosity reader:", args)
        subprocess.Popen(args, cwd=os.path.dirname(csv_path))
    except Exception as e:
        messagebox.showerror("Rheology", f"Failed to launch reader:\n{e}")


# --- New: Find Rheology folder helper ---

def get_rheology_folder_from_path(path: str) -> str | None:
    """Given any path inside a reference tree, return the absolute path to the
    `Rheology` modality folder if it exists.

    Rules:
    - If `path` is already the Rheology folder, return it.
    - If `path` is inside a `_Processed/*` subtree, step out to the reference root, then append `Rheology`.
    - Otherwise, compute the reference root and append `Rheology`.
    - If none found on disk, return None.
    """
    abspath = os.path.abspath(os.path.realpath(path))
    # If it's a file, use its directory
    if os.path.isfile(abspath):
        abspath = os.path.dirname(abspath)

    parts = _split_parts(abspath)
    # If we're already in .../Rheology
    if parts and parts[-1] == 'Rheology' and os.path.isdir(abspath):
        return abspath

    # If inside _Processed subtree, step out of it to the reference root
    if '_Processed' in parts:
        idx = len(parts) - 1 - parts[::-1].index('_Processed')
        ref_root = os.path.join(os.sep, *parts[:idx])
        cand = os.path.join(ref_root, 'Rheology')
        if os.path.isdir(cand):
            return cand

    # Else derive reference root via markers and append Rheology
    ref_root = get_reference_folder_from_path(abspath)
    cand = os.path.join(ref_root, 'Rheology')
    if os.path.isdir(cand):
        return cand

    print(f"[Rheology] No 'Rheology' folder found near: {abspath}")
    return None

# --- Listing helpers for Rheology CSVs ---
import re

def _natural_key(name: str):
    base = os.path.basename(name)
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", base)]


def list_rheology_files(path: str) -> list[str]:
    """Return absolute CSV paths from the dataset's Rheology folder. Ignores hidden/system files.
    Uses natural sort by basename.
    """
    rh_dir = get_rheology_folder_from_path(path)
    results: list[str] = []
    if not rh_dir or not os.path.isdir(rh_dir):
        print(f"[Rheology] list_rheology_files: no Rheology dir for {path}")
        return results
    try:
        for fname in os.listdir(rh_dir):
            if fname.startswith('.') or fname == '.DS_Store':
                continue
            if fname.lower().endswith('.csv'):
                results.append(os.path.join(rh_dir, fname))
        results.sort(key=_natural_key)
        print(f"[Rheology] list_rheology_files: total={len(results)} dir={rh_dir}")
        for ex in results[:5]:
            print("   example:", ex)
    except Exception as e:
        print("[Rheology] list_rheology_files error:", e)
    return results


def list_rheology_samples(path: str) -> list[str]:
    """Return display sample names (CSV basenames without extension)."""
    files = list_rheology_files(path)
    samples: list[str] = []
    for p in files:
        base = os.path.basename(p)
        name = re.sub(r"\.csv$", "", base, flags=re.IGNORECASE)
        samples.append(name)
    return samples


# --- UI: Process Rheology folder ---

def open_rheology_folder_window(parent, start_path: str):
    # Resolve reference + Rheology folder
    reference_folder = get_reference_folder_from_path(start_path)
    rheo_folder = get_rheology_folder_from_path(start_path)
    if not rheo_folder:
        messagebox.showwarning("Rheology", "No Rheology folder found near this path.")
        return

    # Load previous options
    sess = _rheo_load_session(reference_folder)
    mode_default = sess.get("mode", "triggered")
    steady_default = float(sess.get("steady_sec", 10.0))

    # Build window
    win = tk.Toplevel(parent)
    win.title("Process Rheology folder")

    # Options row
    opt = ttk.Frame(win)
    opt.pack(fill=tk.X, padx=12, pady=(10, 6))
    ttk.Label(opt, text="Mode:").grid(row=0, column=0, sticky="w")
    mode_var = tk.StringVar(value=mode_default)
    mode_box = ttk.Combobox(opt, textvariable=mode_var, values=["triggered", "nontriggered", "other"], state="readonly", width=14)
    mode_box.grid(row=0, column=1, sticky="w", padx=(4, 12))

    ttk.Label(opt, text="Steady (s):").grid(row=0, column=2, sticky="w")
    steady_var = tk.StringVar(value=str(steady_default))
    steady_ent = ttk.Entry(opt, textvariable=steady_var, width=8)
    steady_ent.grid(row=0, column=3, sticky="w", padx=(4, 12))

    for i in range(4):
        opt.grid_columnconfigure(i, weight=0)

    # List frame
    body = ttk.Frame(win)
    body.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 8))

    cols = ("name",)
    tree = ttk.Treeview(body, columns=cols, show="headings", height=12)
    tree.heading("name", text="Rheology CSVs")
    tree.column("name", width=480, anchor="w")

    yscroll = ttk.Scrollbar(body, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=yscroll.set)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    yscroll.pack(side=tk.RIGHT, fill=tk.Y)

    def _refresh():
        tree.delete(*tree.get_children())
        files = list_rheology_files(start_path)
        for p in files:
            tree.insert('', 'end', values=(os.path.basename(p),), tags=(p,))
        print(f"[Rheology] UI list refreshed: {len(files)} file(s)")

    def _selected_csv_path() -> str | None:
        sel = tree.selection()
        if not sel:
            return None
        item = sel[0]
        tags = tree.item(item, 'tags')
        for t in tags:
            if t and t.endswith('.csv'):
                # Recover absolute path from tag by joining with folder if tag is only basename
                if os.path.isabs(t):
                    return t
        # If tags didn't store abs path, reconstruct from folder + displayed name
        name = tree.item(item, 'values')[0]
        return os.path.join(rheo_folder, name)

    def _run_selected():
        p = _selected_csv_path()
        if not p or not os.path.isfile(p):
            messagebox.showinfo("Rheology", "Please select a CSV to run.")
            return
        try:
            steady = float(steady_var.get())
        except Exception:
            messagebox.showerror("Rheology", "Invalid numeric value for Steady.")
            return
        # Save session
        _rheo_save_session(reference_folder, {
            "mode": mode_var.get(),
            "steady_sec": steady,
        })
        _launch_viscosity_reader(p, mode_var.get(), steady)

    def _process_all():
        try:
            steady = float(steady_var.get())
        except Exception:
            messagebox.showerror("Rheology", "Invalid numeric value for Steady.")
            return
        _rheo_save_session(reference_folder, {
            "mode": mode_var.get(),
            "steady_sec": steady,
        })
        files = list_rheology_files(start_path)
        if not files:
            messagebox.showinfo("Rheology", "No CSV files found to process.")
            return
        for p in files:
            _launch_viscosity_reader(p, mode_var.get(), steady)

    # Buttons
    btns = ttk.Frame(win)
    btns.pack(fill=tk.X, padx=12, pady=(0, 10))
    ttk.Button(btns, text="Refresh Rheology list", command=_refresh).pack(side=tk.LEFT)
    ttk.Button(btns, text="Run selected", command=_run_selected).pack(side=tk.LEFT, padx=(8,0))
    ttk.Button(btns, text="Process all Rheology data", command=_process_all).pack(side=tk.LEFT, padx=(8,0))
    ttk.Button(btns, text="Close", command=win.destroy).pack(side=tk.RIGHT)

    # Double-click to run selected
    def _on_dclick(event):
        _run_selected()
    tree.bind('<Double-1>', _on_dclick)

    _refresh()

def process_sample_rheology(self, path, subfolder_name, sample_name, add_action, update_actions_list, load_last_folder):
    new_window = tk.Toplevel(self)
    new_window.title("Sample Details")

    ttk.Label(new_window, text=f"Reference folder: {load_last_folder()}").pack()
    ttk.Label(new_window, text=f"Process folder: {subfolder_name}").pack()
    ttk.Label(new_window, text=f"Sample name: {sample_name}").pack()

    ttk.Label(new_window, text="Enter 1st point to average:").pack()
    preset_start_entry = ttk.Entry(new_window)
    preset_start_entry.insert(0, "50")  # Set default value
    preset_start_entry.pack()

    ttk.Label(new_window, text="Enter how many points to average:").pack()
    preset_end_entry = ttk.Entry(new_window)
    preset_end_entry.insert(0, "95")  # Set default value
    preset_end_entry.pack()

    ttk.Button(new_window, text="Engage", command=lambda: engage_viscosity(path, sample_name, new_window, preset_start_entry, preset_end_entry, add_action, update_actions_list)).pack()

    update_actions_list(new_window, sample_name)

def engage_viscosity(path, sample_name, window, preset_start_entry, preset_end_entry, add_action, update_actions_list):
    preset_start = int(float(preset_start_entry.get()))  # Ensure integer conversion
    preset_end = int(float(preset_end_entry.get()))  # Ensure integer conversion
    processed_folder = get_unified_processed_folder(path)
    print(f"Unified _Processed folder for Rheology outputs: {processed_folder}")
    output_file = os.path.join(processed_folder, f"_viscosity_{sample_name}.csv")
    print(f"Output CSV will be written to: {output_file}")
    print(f"Unified Rheology JSON path: {json_path}")
    #output_file = os.path.join(path, f"{sample_name}_viscosity_output.csv")

    if os.path.exists(output_file):
        if not messagebox.askyesno("Overwrite", f"The file {os.path.basename(output_file)} already exists. Overwrite?"):
            window.after(0, add_action, sample_name, "Rheology processing skipped", "red")
            window.after(0, update_actions_list, window, sample_name)
            return

    def run_viscosity():
        try:
            script_path = os.path.join(base_path, '../Read_viscosity_v3.py')
            print(f"Running {script_path} with parameters: {path, sample_name, preset_start, preset_end}")
            
            result = subprocess.run(['python3', script_path, path, sample_name, str(preset_start), str(preset_end)], capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)

            import json
            try:
                with open(json_path, "w") as f:
                    json.dump({
                        "sample_name": sample_name,
                        "processed_folder": processed_folder,
                        "csv_output": output_file
                    }, f, indent=2)
                print(f"Rheology output JSON written: {json_path}")
            except Exception as e:
                print(f"Failed to write Rheology JSON: {e}")

        except Exception as e:
            print(f"Exception occurred while processing Rheology for sample {sample_name} in path {path}: {e}")
            window.after(0, add_action, sample_name, "Rheology processing failed", "red")
            window.after(0, update_actions_list, window, sample_name)
    run_viscosity()
    #threading.Thread(target=run_viscosity).start()

def process_all_rheology_files(path, sample_folders):
    preset_start = 40  # Default preset start value
    preset_end = 95  # Default preset end value

    processed_folder = get_unified_processed_folder(path)
    print(f"Unified _Processed folder for batch Rheology outputs: {processed_folder}")

    def run_all_viscosity():
        for sample in sample_folders:
            sample_path = os.path.join(path, sample)
            sample_name = os.path.splitext(sample)[0]  # Remove the .csv extension for the sample name
            script_path = os.path.join(base_path, '../Read_viscosity_v3.py')
            print(f"Running {script_path} for {sample_name} with parameters: {sample_path, sample_name, preset_start, preset_end}")
            result = subprocess.run(['python3', script_path, sample_path, sample_name, str(preset_start), str(preset_end)], capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)

            import json
            output_file = os.path.join(processed_folder, f"_viscosity_{sample_name}.csv")
            with open(json_path, "w") as f:
                json.dump({
                    "sample_name": sample_name,
                    "processed_folder": processed_folder,
                    "csv_output": output_file
                }, f, indent=2)
            print(f"Batch Rheology output JSON written: {json_path}")

    #threading.Thread(target=run_all_viscosity).start()
    run_all_viscosity()
