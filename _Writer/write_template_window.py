from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import os
import json
import subprocess
import time
import numpy as np
import pandas as pd
# Import normalize_modal_json for path normalization
from common import normalize_modal_json
#from data_writer import 

from _Writer.dg_runner import resolve_datagraph_exec, run_datagraph_write

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
            reference = os.sep.join(parts[:i])
            if reference == "":
                break
            return reference

    return os.path.dirname(os.path.dirname(abspath))


def _auto_find_radial_for_azi(azi_path: str | Path) -> str | None:
    """Locate the first radial integration file living next to the provided azimuthal file."""
    try:
        p = Path(azi_path).expanduser()
    except Exception:
        return None
    folder = p if p.is_dir() else p.parent
    if not folder.exists():
        return None
    patterns = ("rad_saxs*", "*rad*.*")
    for pattern in patterns:
        candidates = sorted(q for q in folder.glob(pattern) if q.is_file())
        if candidates:
            return str(candidates[0])
    return None


def _radial_from_saxs_json(json_path: str | Path) -> str | None:
    """Inspect a SAXS JSON payload to determine the likely radial counterpart."""
    try:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    except Exception:
        payload = None

    def _scan_entry(entry):
        if not isinstance(entry, dict):
            return None
        for key in ("file", "path", "source", "input"):
            val = entry.get(key)
            if isinstance(val, str):
                radial = _auto_find_radial_for_azi(val)
                if radial:
                    return radial
        return None

    if isinstance(payload, dict):
        radial = _scan_entry(payload)
        if radial:
            return radial
    elif isinstance(payload, list):
        for entry in payload:
            radial = _scan_entry(entry)
            if radial:
                return radial

    return _auto_find_radial_for_azi(Path(json_path).parent)


def _auto_find_radial_for_azi_input(azi_input: str | Path) -> str | None:
    """Find 'rad_saxs*' next to an azimuthal file (or folder)."""
    try:
        p = Path(azi_input).expanduser()
    except Exception:
        return None
    folder = p if p.is_dir() else p.parent
    if not folder.exists():
        return None
    cands = sorted(q for q in folder.glob("rad_saxs*") if q.is_file())
    if not cands:
        cands = sorted(q for q in folder.glob("*rad*.*") if q.is_file())
    return str(cands[0].resolve()) if cands else None


def _prefer_radial_csv(path: str | Path) -> str:
    """If a CSV companion exists for the radial file, return it."""
    try:
        p = Path(path).expanduser()
    except Exception:
        return str(path)
    if p.suffix.lower() == ".csv":
        return str(p)
    if p.name.startswith("SAXS_radial_"):
        sample_tag = p.stem.replace("SAXS_radial_", "", 1)
        matrix_candidate = p.with_name(f"SAXS_radial_matrix_{sample_tag}.csv")
        if matrix_candidate.exists():
            return str(matrix_candidate)
        flat_candidate = p.with_name(f"SAXS_radial_flat_{sample_tag}.csv")
        if flat_candidate.exists():
            return str(flat_candidate)
    csv_candidate = p.with_suffix(".csv")
    if csv_candidate.exists():
        return str(csv_candidate)
    return str(p)


def _strip_rate_threshold_key(d):
    if isinstance(d, dict) and "rate_threshold" in d:
        d.pop("rate_threshold", None)
    return d

# --- Headless backend entrypoint for batch pairing ---
def write_pairs_to_template(reference_folder: str | Path, pairs: list[tuple[str | Path, str | Path]]):
    """
    reference_folder: path to the reference folder (parent of _Processed)
    pairs: list of (saxs_json_path, rheo_json_path)
    """
    from pathlib import Path
    ref = Path(reference_folder)
    processed = ref / "_Processed"

    # Whatever your writer window does internally, call the same _core writer here.
    # Example (adapt these names to your implementation):
    #
    # 1) Load both JSONs, merge/select what the DG template needs
    # 2) Resolve the template path (e.g., in the DG subfolder)
    # 3) Write the rows/files as needed

    for saxs_json, rheo_json in pairs:
        # Replace with your existing internal writer call:
        _write_one_pair_to_dg_template(processed, Path(saxs_json), Path(rheo_json))
        # If you already have a function like write_to_template(saxs_json, rheo_json), just call it here.

def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    os.makedirs(processed_root, exist_ok=True)
    return processed_root

class WriteTemplateWindow(tk.Toplevel):
    def __init__(self, parent, base_path, processed_root=None, available_outputs=None):
        super().__init__(parent)
        self.title("Write Data to Template")
        self.geometry("1000x800")
        self.minsize(900, 700)

        # Store base_path
        self.base_path = base_path

        # Prefer explicitly provided args; otherwise fall back to parent attributes
        if processed_root is not None:
            self.processed_root = processed_root
        elif hasattr(parent, "processed_root"):
            self.processed_root = parent.processed_root
        else:
            self.processed_root = None
        if self.processed_root:
            print(f"Unified _Processed folder detected: {self.processed_root}")

        if available_outputs is not None:
            self.available_outputs = available_outputs
        elif hasattr(parent, "available_outputs"):
            self.available_outputs = parent.available_outputs
            print("Available datasets loaded from main window.")
        else:
            self.available_outputs = {"PI": [], "PLI": [], "SAXS": [], "Rheology": [], "Other": []}

        self.datagraph_path_var = tk.StringVar(value="/Applications/DataGraph.app/Contents/Library")
        self.template_path_var = tk.StringVar(value=self.load_last_template())
        self.base_name_var = tk.StringVar(value=os.path.basename(self.load_last_folder()))
        self.additional_files = []
        self._radial_by_saxs: dict[str, str] = {}

        # Initialize default_values as an empty dictionary
        self.default_values = {}

        self.create_widgets()

    def create_widgets(self):
        print("[WriterUI] create_widgets: start")
        container = ttk.Frame(self, padding=10)
        container.pack(fill=tk.BOTH, expand=True)
        main = ttk.Frame(container)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side = ttk.Frame(container)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0), pady=10)
        ttk.Button(side, text="Main menu", command=self.destroy).pack(fill=tk.X)
        # Title
        ttk.Label(main, text="Write Data to Template", font=("Helvetica", 16)).pack(pady=10)

        # Info Label
        ttk.Label(main, text="This module writes available data to selected templates").pack(pady=5)

        # Note Label
        ttk.Label(main, text="Note that the template needs to be prepared in advance, with only the data to be written in the correct order visible as columns.").pack(pady=5)

        # DataGraph Path
        datagraph_frame = ttk.Frame(main)
        datagraph_frame.pack(pady=5, fill=tk.X)
        datagraph_frame.columnconfigure(1, weight=1)
        ttk.Label(datagraph_frame, text="Default DG location:").grid(row=0, column=0, padx=5)
        dg_entry = ttk.Entry(datagraph_frame, textvariable=self.datagraph_path_var)
        dg_entry.grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(datagraph_frame, text="Browse", command=self.browse_datagraph_path).grid(row=0, column=2, padx=5)
        ttk.Button(datagraph_frame, text="Test DG", command=self.test_datagraph_path).grid(row=0, column=3, padx=5)
        ttk.Button(datagraph_frame, text="Load / Refresh datasets", command=self.populate_data_columns).grid(row=0, column=4, padx=5)

        # Template Path
        template_frame = ttk.Frame(main)
        template_frame.pack(pady=5, fill=tk.X)
        template_frame.columnconfigure(1, weight=1)
        ttk.Label(template_frame, text="Select DG template:").grid(row=0, column=0, padx=5)
        ttk.Entry(template_frame, textvariable=self.template_path_var).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Button(template_frame, text="Browse", command=self.browse_template_path).grid(row=0, column=2, padx=5)

        # Scrollable data columns frame
        scroll_container = ttk.Frame(main)
        scroll_container.pack(pady=5, fill=tk.BOTH, expand=True)

        scroll_canvas = tk.Canvas(scroll_container)
        scroll_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(scroll_container, orient="vertical", command=scroll_canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        scroll_canvas.configure(yscrollcommand=scrollbar.set)
        scroll_canvas.bind('<Configure>', lambda e: scroll_canvas.configure(scrollregion=scroll_canvas.bbox("all")))

        self.data_columns_frame = ttk.Frame(scroll_canvas)
        scroll_canvas.create_window((0, 0), window=self.data_columns_frame, anchor="nw")

        # --- Controls directly under dataset grid ---
        bottom_bar = ttk.Frame(main)
        bottom_bar.pack(pady=8, fill=tk.X)

        # Left: file controls and base name
        controls_frame = ttk.Frame(bottom_bar)
        controls_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)

        add_files_row = ttk.Frame(controls_frame)
        add_files_row.pack(fill=tk.X)
        ttk.Button(add_files_row, text="Load additional files", command=self.load_additional_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(add_files_row, text="Clear files", command=self.clear_files).pack(side=tk.LEFT, padx=5)
        ttk.Button(add_files_row, text="Show list", command=self.show_files).pack(side=tk.LEFT, padx=5)

        base_row = ttk.Frame(controls_frame)
        base_row.pack(fill=tk.X, pady=(6, 0))
        ttk.Label(base_row, text="Base name:").pack(side=tk.LEFT, padx=5)
        ttk.Entry(base_row, textvariable=self.base_name_var).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=5)

        progress_row = ttk.Frame(controls_frame)
        progress_row.pack(fill=tk.X, pady=(6, 0))
        self.progress_label_var = tk.StringVar(value="Idle")
        ttk.Label(progress_row, textvariable=self.progress_label_var).pack(side=tk.LEFT, padx=5)
        self.progress_bar = ttk.Progressbar(progress_row, mode="determinate", maximum=1, value=0)
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        # Engage button
        self.engage_btn = ttk.Button(controls_frame, text="Engage", command=self.engage)
        self.engage_btn.pack(pady=8, fill=tk.X)

        # Right: Available datasets panel
        self.available_datasets_frame = ttk.Frame(bottom_bar)
        self.available_datasets_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10)
        ttk.Label(self.available_datasets_frame, text="Available Datasets").pack(pady=5)

        print("[WriterUI] create_widgets: end")
        self.update_idletasks()

    def _is_datagraph_running(self) -> bool:
        try:
            script = 'tell application "System Events" to (name of processes) contains "DataGraph"'
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            return res.returncode == 0 and res.stdout.strip().lower() == "true"
        except Exception:
            return False

    def _datagraph_modified_count(self) -> int:
        try:
            script = (
                'tell application "DataGraph"\n'
                '    if not (exists documents) then return "0"\n'
                '    set modifiedDocs to (count of documents whose modified is true)\n'
                '    return modifiedDocs as string\n'
                'end tell'
            )
            res = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
            if res.returncode != 0:
                return -1
            return int(res.stdout.strip() or "0")
        except Exception:
            return -1

    def _prompt_quit_datagraph(self):
        if not self._is_datagraph_running():
            return
        modified = self._datagraph_modified_count()
        if modified is None:
            modified = -1
        if modified > 0:
            ok = messagebox.askyesno(
                "Quit DataGraph",
                f"DataGraph has {modified} unsaved document(s). Quit anyway?",
            )
            if not ok:
                return
        elif modified < 0:
            ok = messagebox.askyesno(
                "Quit DataGraph",
                "Could not determine if DataGraph has unsaved changes. Quit anyway?",
            )
            if not ok:
                return
        try:
            subprocess.run(["osascript", "-e", 'tell application "DataGraph" to quit'])
        except Exception:
            pass

    def _background_datagraph(self):
        if not self._is_datagraph_running():
            return
        try:
            script = (
                'tell application "System Events"\n'
                '    if (name of processes) contains "DataGraph" then\n'
                '        set frontmost of process "DataGraph" to false\n'
                '    end if\n'
                'end tell'
            )
            subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
        except Exception:
            pass

    def test_datagraph_path(self):
        """Validate the DataGraph path provided by the user and show a quick diagnostic."""
        path = self.datagraph_path_var.get().strip()
        if not path:
            messagebox.showerror("DataGraph path", "Path is empty.")
            return
        if os.path.isfile(path):
            ok = os.access(path, os.X_OK)
            print(f"[DG Test] File exists: {path}; executable={ok}")
            if ok:
                messagebox.showinfo("DataGraph path", f"Executable found and is runnable:\n{path}")
            else:
                messagebox.showerror("DataGraph path", f"File exists but is not executable:\n{path}")
            return
        if os.path.isdir(path):
            # Try to detect dgraph/DataGraph inside this directory
            cand = None
            for name in ("dgraph", "DataGraph", "datagraph"):
                p = os.path.join(path, name)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    cand = p
                    break
            print(f"[DG Test] Dir exists: {path}; found exec={cand}")
            if cand:
                messagebox.showinfo("DataGraph path", f"Executable found in directory:\n{cand}")
            else:
                messagebox.showerror("DataGraph path", f"No executable named dgraph/DataGraph in:\n{path}")
            return
        messagebox.showerror("DataGraph path", f"Path does not exist:\n{path}")

    def browse_datagraph_path(self):
        initdir = self.base_path if os.path.isdir(self.base_path) else os.path.expanduser("~")
        directory = filedialog.askdirectory(title="Select DataGraph Path", initialdir=initdir)
        if directory:
            self.datagraph_path_var.set(directory)

    def browse_template_path(self):
        initdir = self.base_path if os.path.isdir(self.base_path) else os.path.expanduser("~")
        file_path = filedialog.askopenfilename(title="Select Template File", initialdir=initdir)
        if file_path:
            self.template_path_var.set(file_path)
            self.save_last_template(file_path)

    def load_additional_files(self):
        initdir = self.base_path if os.path.isdir(self.base_path) else os.path.expanduser("~")
        file_paths = filedialog.askopenfilenames(title="Select Additional Files", initialdir=initdir)
        if file_paths:
            self.additional_files.extend(file_paths)
            self.populate_data_columns()  # Refresh the data columns to include additional files

    def clear_files(self):
        """Clears the contents of the specified .json files after user confirmation."""
        # List of JSON files to clear
        json_files = [
            "output_SAXS.json",
            "output_PI.json",
            "output_viscosity.json",
            "output_PLI.json"
        ]

        # Prompt the user for confirmation
        confirm = messagebox.askyesno(
            "Clear All Files",
            "This will delete all the contents of the .json files. Are you sure?"
        )

        if confirm:
            for json_file in json_files:
                if os.path.exists(json_file):
                    # Clear the contents of the file
                    with open(json_file, 'w') as file:
                        file.write("[]")  # Write an empty JSON array
                    print(f"Cleared contents of {json_file}")
                else:
                    print(f"{json_file} does not exist, skipping.")
            
            messagebox.showinfo("Clear All Files", "All specified .json files have been cleared.")
        else:
            print("Clear operation canceled by the user.")

        def show_additional_files(self):
            messagebox.showinfo("Additional Files", "\n".join(self.additional_files))
        self.populate_data_columns()  # Refresh the data columns to remove additional files

    def show_files(self):
        file_contents = []
        json_files = ["output_SAXS.json", "output_PI.json", "output_viscosity.json", "output_PLI.json"]

        for json_file in json_files:
            if os.path.exists(json_file):
                with open(json_file, 'r') as file:
                    try:
                        data = json.load(file)
                        file_contents.append(f"{json_file}:\n{json.dumps(data, indent=4)}")
                    except json.JSONDecodeError:
                        file_contents.append(f"{json_file}:\n<Invalid JSON format>")
            else:
                file_contents.append(f"{json_file}:\n<File does not exist>")

        messagebox.showinfo("File Contents", "\n\n".join(file_contents))

    def load_last_folder(self):
        last_folder_file = "last_folder.txt"
        if os.path.exists(last_folder_file):
            with open(last_folder_file, 'r') as file:
                return file.read().strip()
        return ""

    def load_last_template(self):
        last_template_file = "last_template.txt"
        if os.path.exists(last_template_file):
            with open(last_template_file, 'r') as file:
                return file.read().strip()
        return ""

    def save_last_template(self, template_path):
        last_template_file = "last_template.txt"
        with open(last_template_file, 'w') as file:
            file.write(template_path)

    def populate_data_columns(self):
        if not hasattr(self, "data_columns_frame"):
            print("[Writer] UI not ready yet — click 'Load / Refresh datasets' again in a moment.")
            return
        for widget in self.data_columns_frame.winfo_children():
            widget.destroy()

        # Load unified _output_*.json files from the unified _Processed folder if available
        if self.processed_root and os.path.isdir(self.processed_root):
            print(f"Loading unified outputs from: {self.processed_root}")
            output_files_SAXS = self.load_unified_outputs("SAXS")
            output_files_PI = self.load_unified_outputs("PI")
            output_files_PLI = self.load_unified_outputs("PLI")
            output_files_viscosity = self.load_unified_outputs("Rheology")
            # Cache unified outputs for later lookups in engage()
            self.unified_outputs = {
                "SAXS": output_files_SAXS,
                "PI": output_files_PI,
                "PLI": output_files_PLI,
                "Rheology": output_files_viscosity,
            }
        else:
            # Fallback to legacy files in base_path
            output_files_SAXS = self.load_output_data("output_SAXS.json")
            output_files_PI = self.load_output_data("output_PI.json")
            output_files_PLI = self.load_output_data("output_PLI.json")
            output_files_viscosity = self.load_output_data("output_viscosity.json")
            # Cache legacy outputs for later lookups in engage()
            self.unified_outputs = {
                "SAXS": output_files_SAXS,
                "PI": output_files_PI,
                "PLI": output_files_PLI,
                "Rheology": output_files_viscosity,
            }
        sample_names_SAXS = [sample['name'] for sample in output_files_SAXS]
        sample_names_PI = [sample['name'] for sample in output_files_PI]
        sample_names_PLI = [sample['name'] for sample in output_files_PLI]
        sample_names_viscosity = [sample['name'] for sample in output_files_viscosity]
        self._saxs_json_for_name = {}
        try:
            for entry in output_files_SAXS:
                nm = entry.get("name")
                jp = entry.get("json_path")
                if isinstance(nm, str) and isinstance(jp, str):
                    self._saxs_json_for_name[nm] = jp
        except Exception:
            pass

        # Determine total rows by max across categories (not just SAXS)
        row_count = max(
            len(sample_names_SAXS) if sample_names_SAXS else 0,
            len(sample_names_PI) if sample_names_PI else 0,
            len(sample_names_PLI) if sample_names_PLI else 0,
            len(sample_names_viscosity) if sample_names_viscosity else 0,
            len(self.additional_files) if hasattr(self, 'additional_files') and self.additional_files else 0,
        )

        if not any([sample_names_SAXS, sample_names_PI, sample_names_PLI, sample_names_viscosity]):
            ttk.Label(self.data_columns_frame, text="No datasets found. Check that _output_*.json files exist in _Processed or legacy output_*.json are present.").grid(row=1, column=0, columnspan=5, sticky="w", padx=4, pady=4)
            return

        columns = ["SAXS", "PI", "PLI", "Viscosity", "Radial Integration"]
        self._columns = columns[:]
        self._radial_col = columns.index("Radial Integration")
        for j, col in enumerate(columns):
            ttk.Label(self.data_columns_frame, text=col).grid(row=0, column=j)

        options_SAXS = ["None"] + sample_names_SAXS
        options_PI = ["None"] + sample_names_PI
        options_PLI = ["None"] + sample_names_PLI
        options_viscosity = ["None"] + sample_names_viscosity
        options_radial_integration = ["None"] + self.additional_files

        column_options = {
            "SAXS": options_SAXS,
            "PI": options_PI,
            "PLI": options_PLI,
            "Viscosity": options_viscosity,
            "Radial Integration": options_radial_integration,
        }

        self.vars = {}
        self.default_values = {}

        for j, col in enumerate(columns):
            options = column_options.get(col, ["None"])  # options for this column
            for i in range(row_count):
                var = tk.StringVar()
                self.vars[(i, j)] = var
                default_value = "None"
                self.default_values[(i, j)] = default_value
                var.set(default_value)
                if col == "SAXS":
                    option_menu = ttk.OptionMenu(
                        self.data_columns_frame,
                        var, default_value, *options,
                        command=lambda value, row=i: self._on_saxs_select(row, value)
                    )
                else:
                    option_menu = ttk.OptionMenu(self.data_columns_frame, var, default_value, *options)
                option_menu.config(width=14)
                option_menu.grid(row=i+1, column=j, padx=2, pady=1)

    def _on_saxs_select(self, row_idx: int, selected_name: str):
        """When a SAXS dataset is selected, auto-fill the radial column."""
        try:
            if selected_name == "None":
                rv = self.vars.get((row_idx, self._radial_col))
                if rv:
                    rv.set("None")
                return
            json_path = self._saxs_json_for_name.get(selected_name)
            radial_path = None
            if json_path:
                try:
                    self._ensure_radial_for_json(Path(json_path))
                except Exception:
                    pass
                key = str(Path(json_path).resolve())
                radial_path = self._radial_by_saxs.get(key)
                if not radial_path:
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        for key in ("radial_matrix_csv", "radial_flat_csv", "radial_csv", "radial_copy", "radial_file"):
                            candidate = data.get(key)
                            if isinstance(candidate, str) and candidate:
                                radial_path = candidate
                                break
                        if not radial_path:
                            src = (
                                data.get("azimuthal_file")
                                or data.get("file")
                                or data.get("path")
                                or data.get("source")
                            )
                            if src:
                                radial_path = _auto_find_radial_for_azi_input(src)
                    except Exception:
                        radial_path = None
                    if not radial_path:
                        radial_path = _auto_find_radial_for_azi_input(Path(json_path).parent)
                    if radial_path:
                        self._radial_by_saxs[key] = radial_path
            if radial_path:
                try:
                    radial_path = str(Path(radial_path).expanduser())
                except Exception:
                    pass
                rv = self.vars.get((row_idx, self._radial_col))
                if rv:
                    rv.set(radial_path)
                print(f"[Writer] Auto-attached radial for row {row_idx+1}: {radial_path}")
        except Exception as e:
            print(f"[Writer] Radial auto-select failed: {e}")
    def _filter_files_by_prefix(self, files, category):
        """Return only files whose base name starts with the required prefix for the category, case-insensitive.
        Also includes new unified filename patterns so master-summary CSVs aren’t filtered out.
        For Rheology, prefer steady-state CSVs if available.
        """
        prefix_map = {
            "PLI": ("PLI_", "_pli_"),
            "PI": ("PI_", "_pi_"),
            "SAXS": ("SAXS_", "_saxs_"),
            "Rheology": ("Rheo_", "_rheo_", "_visco_"),
        }
        prefixes = tuple(p.lower() for p in prefix_map.get(category, ()))
        out = []
        steady = []
        for p in files:
            base = os.path.basename(p)
            bl = base.lower()
            if not prefixes or any(bl.startswith(pref) or pref in bl for pref in prefixes):
                if category == "Rheology" and bl.startswith("rheo_steady_"):
                    steady.append(p)
                else:
                    out.append(p)
        return steady or out

    def get_files_for(self, category, sample_name):
        """Return data files for a given category and sample name, using unified outputs if available.
        Applies file-name prefix filtering according to the agreed nomenclature.
        Now with verbose diagnostics and fallback if prefix filtering removes all files.
        Enhanced: For Rheology, fallback to scan _Processed for matching Rheo_*.csv if no files found, preferring steady-state files.
        """
        # Prefer unified outputs cache if present
        if hasattr(self, "unified_outputs") and isinstance(self.unified_outputs, dict):
            samples = self.unified_outputs.get(category, [])
            for s in samples:
                if s.get("name") == sample_name:
                    files = s.get("files", [])
                    print(f"[Writer:get_files_for] {category}:{sample_name} unified files -> {len(files)}")
                    filtered = self._filter_files_by_prefix(files, category)
                    if not filtered and files:
                        print(f"[Writer:get_files_for] Prefix filter removed all files for {category}:{sample_name}; using unfiltered list.")
                        filtered = files
                    return filtered
        # Fallback to legacy JSONs in base_path
        legacy_map = {
            "SAXS": "output_SAXS.json",
            "PI": "output_PI.json",
            "PLI": "output_PLI.json",
            "Rheology": "output_viscosity.json",
        }
        json_filename = legacy_map.get(category)
        if json_filename:
            files = self.get_files_from_json(json_filename, sample_name) or []
            filtered = self._filter_files_by_prefix(files, category)
            if not filtered and files:
                print(f"[Writer:get_files_for] Prefix filter removed all files for {category}:{sample_name} (legacy); using unfiltered list.")
                filtered = files
            return filtered
        # Fallback for Rheology: scan the _Processed folders for Rheo_*.csv matching the sample name, prefer steady
        if category == "Rheology" and self.processed_root:
            try:
                name = sample_name or ""
                search_dirs = [self.processed_root, os.path.join(self.processed_root, "Rheology")]
                hits_all = []
                hits_steady = []
                for d in search_dirs:
                    if not os.path.isdir(d):
                        continue
                    for fname in os.listdir(d):
                        fl = fname.lower()
                        if fl.endswith('.csv') and fl.startswith('rheo') and name.lower() in fl:
                            full = os.path.join(d, fname)
                            if fl.startswith('rheo_steady_'):
                                hits_steady.append(full)
                            else:
                                hits_all.append(full)
                hits = hits_steady or hits_all
                if hits:
                    print(f"[Writer:get_files_for] Rheology fallback found {len(hits)} file(s) for '{name}'")
                    return hits
            except Exception as e:
                print("[Writer:get_files_for] Rheology fallback scan error:", e)
        return []

    def _remember_radial_hint(self, json_path=None, files=None):
        """Record the detected radial integration path for later use."""
        candidates = []
        if json_path:
            candidates.append(json_path)
        for f in files or []:
            candidates.append(f)
        for candidate in candidates:
            if not candidate:
                continue
            try:
                resolved = Path(candidate).expanduser()
                key = str(resolved.resolve())
            except Exception:
                key = str(candidate)
                resolved = Path(candidate)
            if key in self._radial_by_saxs:
                continue
            radial = None
            if resolved.suffix.lower() == ".json":
                radial = _radial_from_saxs_json(resolved)
                self._ensure_radial_for_json(resolved)
            if not radial:
                radial = _auto_find_radial_for_azi(resolved)
            if radial:
                radial = _prefer_radial_csv(radial)
                self._radial_by_saxs[key] = radial
                print(f"[Writer] Auto-attached radial for {resolved}: {radial}")

    def _ensure_radial_for_json(self, saxs_json_path: Path):
        """Ensure radial path is cached for a given SAXS JSON."""
        if not hasattr(self, "_radial_by_saxs"):
            self._radial_by_saxs = {}
        try:
            key = str(Path(saxs_json_path).resolve())
        except Exception:
            key = str(saxs_json_path)
        if key in self._radial_by_saxs:
            return
        radial_path = None
        try:
            with open(saxs_json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = None
        if isinstance(data, dict):
            for key in ("radial_matrix_csv", "radial_flat_csv", "radial_csv", "radial_copy", "radial_file"):
                candidate = data.get(key)
                if isinstance(candidate, str) and candidate:
                    radial_path = candidate
                    break
            if not radial_path:
                src = (data.get("azimuthal_file") or data.get("file") or
                       data.get("path") or data.get("source") or data.get("input"))
                if src:
                    radial_path = _auto_find_radial_for_azi_input(src)
        if not radial_path:
            radial_path = _auto_find_radial_for_azi_input(saxs_json_path)
        if radial_path:
            try:
                radial_path = str(Path(radial_path).expanduser())
            except Exception:
                radial_path = str(radial_path)
            radial_path = _prefer_radial_csv(radial_path)
            self._radial_by_saxs[key] = radial_path
            print(f"[Writer] Auto-attached radial: {radial_path}")
    def load_unified_outputs(self, category):
        """Filename-only discovery of outputs under _Processed (skips _Other)."""
        outputs = []
        if not self.processed_root:
            return outputs
        root = Path(self.processed_root)
        prefixes = {
            "PLI": ("_dg_pli_", "_pli_"),
            "PI": ("_dg_pi_", "_pi_"),
            "SAXS": ("saxs_1_", "saxs_2_", "saxs_radial_"),
            "Rheology": ("_rheo_", "_visco_"),
        }
        allowed_exts = {".csv", ".dat", ".txt"}
        dirs = [root] + [p for p in root.iterdir() if p.is_dir() and p.name != "_Other"]
        names_to_files = {}
        for folder in dirs:
            try:
                items = list(folder.iterdir())
            except Exception:
                continue
            for path in items:
                if not path.is_file():
                    continue
                if path.suffix.lower() not in allowed_exts:
                    continue
                low = path.name.lower()
                pref_hit = None
                for pref in prefixes.get(category, ()):  # order matters
                    if low.startswith(pref):
                        pref_hit = pref
                        break
                if not pref_hit:
                    continue
                name = path.stem
                if name.lower().startswith(pref_hit):
                    name = name[len(pref_hit):]
                names_to_files.setdefault(name, []).append(str(path))
        for name, files in sorted(names_to_files.items()):
            outputs.append({"name": name, "files": files})
        return outputs

