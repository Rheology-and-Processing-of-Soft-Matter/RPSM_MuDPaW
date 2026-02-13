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
                    if category == "SAXS":
                        self._remember_radial_hint(s.get("json_path"), filtered)
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
            if category == "SAXS":
                json_path = os.path.join(self.base_path, json_filename)
                self._remember_radial_hint(json_path, filtered)
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
        """Loads per-sample _output_<category>_*.json files from the unified _Processed folder,
        searching both the root _Processed and the category subfolder (e.g., _Processed/PLI).
        Applies opt-in filters for template writing.
        Enhanced for unified master JSON with a top-level "datasets" map.
        Filters implemented:
          - If the JSON root has `exclude_from_template = true`, the file is skipped.
          - If the JSON root has `template_allow = [ ... ]`, the file is only included when
            the current base_name is in that list (exact match). If missing/empty, included by default.
        """
        outputs = []
        # Compute the current reference folder from processed_root
        reference_folder = os.path.dirname(self.processed_root) if self.processed_root else None
        if not self.processed_root:
            return outputs

        # Search both the root _Processed and the category subfolder (e.g., _Processed/PLI)
        search_dirs = [self.processed_root]
        subdir = os.path.join(self.processed_root, category)
        if os.path.isdir(subdir):
            search_dirs.append(subdir)

        current_base = self.base_name_var.get().strip()

        for d in search_dirs:
            try:
                files = os.listdir(d)
            except Exception:
                continue
            for fname in files:
                if fname.lower().startswith(f"_output_{category.lower()}") and fname.lower().endswith(".json"):
                    fpath = os.path.join(d, fname)
                    try:
                        with open(fpath, "r") as f:
                            data = json.load(f)

                        # Normalize absolute/relative paths to current reference
                        try:
                            if reference_folder:
                                # Try to normalize the whole root first (covers simple schemas)
                                data = normalize_modal_json(
                                    data, reference_folder,
                                    keys=(
                                        'csv_output','csv_outputs','files','paths','outputs',
                                        'profiles_csv','fits_csv','datagraph_csv','summary_csv','dg_exports'
                                    )
                                )
                        except Exception as _e:
                            print(f"[Writer] normalize warning for {fpath}: {_e}")
                        if isinstance(data, dict):
                            _strip_rate_threshold_key(data)

                        # ---- Filtering rules ----
                        if isinstance(data, dict) and data.get("exclude_from_template", False):
                            continue

                        allow_list = []
                        if isinstance(data, dict):
                            allow_list = data.get("template_allow", [])
                        if isinstance(allow_list, list) and len(allow_list) > 0:
                            if current_base not in allow_list:
                                continue
                        # ---- End filtering rules ----

                        # Enhanced: Handle unified master JSON with top-level "datasets" map
                        if isinstance(data, dict) and "datasets" in data and isinstance(data["datasets"], dict):
                            # Normalize each dataset entry (and nested maltese_cross sections)
                            for _k, _entry in list(data["datasets"].items()):
                                try:
                                    data["datasets"][_k] = normalize_modal_json(
                                        _entry, reference_folder,
                                        keys=(
                                            'csv_output','csv_outputs','files','paths','outputs',
                                            'profiles_csv','fits_csv','datagraph_csv','summary_csv','dg_exports'
                                        )
                                    )
                                    for _mc in data["datasets"][_k].get("maltese_cross", {}).values():
                                        normalize_modal_json(
                                            _mc, reference_folder,
                                            keys=('datagraph_csv','summary_csv','dg_exports')
                                        )
                                except Exception as _e:
                                    print(f"[Writer] normalize datasets warning for {fpath}::{_k}: {_e}")
                                if isinstance(data["datasets"].get(_k), dict):
                                    _strip_rate_threshold_key(data["datasets"][_k])
                            for dkey, entry in data.get("datasets", {}).items():
                                _strip_rate_threshold_key(entry)
                                files = []
                                if category == "SAXS":
                                    files = entry.get("csv_outputs", [])
                                elif category == "PLI":
                                    files = []
                                    for mc in entry.get("maltese_cross", {}).values():
                                        f = mc.get("summary_csv")
                                        if f:
                                            files.append(f)
                                else:
                                    # fallback: try csv_outputs or files
                                    files = entry.get("csv_outputs") or entry.get("files") or []
                                # Normalize to flat list of strings
                                if isinstance(files, dict):
                                    files = list(files.values())
                                elif not isinstance(files, list):
                                    files = [str(files)] if files else []
                                if not files:
                                    continue
                                sample_name = entry.get("sample_name") or dkey
                                # Derive a cleaner display name for Rheology
                                if category == "Rheology":
                                    sample_name = self._derive_rheology_display_name(sample_name, files, fname)
                                outputs.append({"name": sample_name, "files": files, "json_path": fpath})
                            continue  # skip legacy parsing for this file

                        # Legacy/unified per-sample schema
                        if isinstance(data, dict):
                            try:
                                if reference_folder:
                                    data = normalize_modal_json(
                                        data, reference_folder,
                                        keys=(
                                            'csv_output','csv_outputs','files','paths','outputs',
                                            'profiles_csv','fits_csv','datagraph_csv','summary_csv','dg_exports'
                                        )
                                    )
                            except Exception as _e:
                                print(f"[Writer] normalize per-sample warning for {fpath}: {_e}")
                            _strip_rate_threshold_key(data)
                            sample_name = data.get("sample_name") or data.get("name") or os.path.splitext(fname)[0]
                            files = data.get("csv_outputs") or data.get("files") or []
                            # Normalize to list of strings
                            if isinstance(files, dict):
                                files = list(files.values())
                            elif not isinstance(files, list):
                                files = [str(files)] if files else []
                            # Derive a cleaner display name for Rheology
                            if category == "Rheology":
                                sample_name = self._derive_rheology_display_name(sample_name, files, fname)
                            outputs.append({"name": sample_name, "files": files, "json_path": fpath})
                        elif isinstance(data, list):
                            for entry in data:
                                if not isinstance(entry, dict):
                                    continue
                                try:
                                    if reference_folder:
                                        entry = normalize_modal_json(
                                            entry, reference_folder,
                                            keys=(
                                                'csv_output','csv_outputs','files','paths','outputs',
                                                'profiles_csv','fits_csv','datagraph_csv','summary_csv','dg_exports'
                                            )
                                        )
                                except Exception as _e:
                                    print(f"[Writer] normalize list-entry warning for {fpath}: {_e}")
                                _strip_rate_threshold_key(entry)
                                sample_name = entry.get("sample_name") or entry.get("name") or os.path.splitext(fname)[0]
                                files = entry.get("csv_outputs") or entry.get("files") or []
                                if isinstance(files, dict):
                                    files = list(files.values())
                                elif not isinstance(files, list):
                                    files = [str(files)] if files else []
                                if category == "Rheology":
                                    sample_name = self._derive_rheology_display_name(sample_name, files, fname)
                                outputs.append({"name": sample_name, "files": files, "json_path": fpath})
                        else:
                            pass
                    except Exception as e:
                        print(f"Warning: Could not load {fpath}: {e}")
        print(f"[Writer] {category}: found {len(outputs)} dataset(s)")
        return outputs

    def _derive_rheology_display_name(self, sample_name: str, files: list, json_fname: str) -> str:
        """Return a nicer display name for Rheology entries.
        Prefer a name derived from associated Rheo_*.csv if available.
        Fallback: strip the `_output_Rheology_` prefix from the JSON base name.
        """
        # If the sample_name is already a clean label, keep it
        if sample_name and not sample_name.lower().startswith("_output_rheology"):
            return sample_name
        # Try to derive from CSV files
        for f in files or []:
            b = os.path.basename(f)
            if b.lower().endswith('.csv') and 'rheo' in b.lower():
                stem = os.path.splitext(b)[0]
                for pref in ("Rheo_steady_", "rheo_steady_", "Rheo_", "rheo_"):
                    if stem.startswith(pref):
                        return stem[len(pref):]
                return stem
        # Fallback: strip JSON prefix
        base = os.path.splitext(json_fname)[0]
        if base.lower().startswith("_output_rheology_"):
            return base[len("_output_rheology_"):]
        return sample_name or base




    def load_output_data(self, filename):
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                try:
                    data = json.load(file)
                    if isinstance(data, dict):
                        _strip_rate_threshold_key(data)
                    if isinstance(data, list):
                        abs_path = os.path.abspath(filename)
                        for entry in data:
                            if isinstance(entry, dict) and "json_path" not in entry:
                                entry["json_path"] = abs_path
                                _strip_rate_threshold_key(entry)
                    return data
                except json.JSONDecodeError:
                    return []
        return []

    def engage(self):
        print("=== USING WriteTemplateWindow.engage() ===")
        datagraph_path = self.datagraph_path_var.get()
        template_path = self.template_path_var.get()
        base_name = self.base_name_var.get()
        reference_folder = self.load_last_folder()
        legacy_hint = os.path.join(self.base_path, "last_folder.txt")
        inferred_reference = get_reference_folder_from_path(self.base_path)

        print("Checking for reference folder in:", self.base_path)
        print("Legacy last_folder.txt path:", legacy_hint)
        print("Legacy reference folder loaded:", reference_folder)
        print("Inferred reference folder:", inferred_reference)

        # Choose reference folder: prefer last_folder.txt when valid, else inferred
        if not reference_folder or not os.path.isdir(reference_folder):
            reference_folder = inferred_reference
            print("[Writer] Using inferred reference folder:", reference_folder)

        if not reference_folder or not os.path.isdir(reference_folder):
            messagebox.showerror("Error", f"Reference folder not found.\nTried:\n- {legacy_hint}\n- {inferred_reference}")
            return

        selected_files_per_row = []  # Store file lists for each row
        output_file_names = []  # Store output names to avoid overwriting

        # Determine row count from the max row index present in self.vars
        columns_list = ["SAXS", "PI", "PLI", "Viscosity", "Radial Integration"]
        if not getattr(self, 'vars', None):
            messagebox.showerror("Error", "No selections grid found. Populate data columns first.")
            return
        max_row = -1
        for (i, j) in self.vars.keys():
            if i > max_row:
                max_row = i
        row_count = max_row + 1 if max_row >= 0 else 0
        if row_count == 0:
            messagebox.showerror("Error", "No rows available for writing. Make at least one selection.")
            return
        # Debug: print counts per category and first few options
        if hasattr(self, 'unified_outputs') and isinstance(self.unified_outputs, dict):
            for cat in ("SAXS","PI","PLI","Rheology"):
                lst = self.unified_outputs.get(cat, [])
                print(f"[Writer] Unified {cat}: {len(lst)} entries")
        # Debug: print selected values per row/column
        columns_list = ["SAXS", "PI", "PLI", "Viscosity", "Radial Integration"]
        for i in range(row_count):
            vals = []
            for j, col in enumerate(columns_list):
                vals.append(self.vars.get((i,j), tk.StringVar(value="None")).get())
            print(f"[Writer] Row {i+1} selections: {vals}")

        category_map = {"SAXS": "SAXS", "PI": "PI", "PLI": "PLI", "Viscosity": "Rheology"}

        for i in range(row_count):
            row_files = []
            name_token = None
            selected_saxs_name = None  # track which SAXS sample is chosen in this row

            # Iterate over columns and collect files
            for j, col in enumerate(columns_list):
                selected_value = self.vars.get((i, j), tk.StringVar(value="None")).get()
                if selected_value != "None":
                    if name_token is None:
                        name_token = selected_value
                    if col == "SAXS":
                        selected_saxs_name = selected_value
                    cat_key = category_map.get(col)
                    if cat_key:
                        files = self.get_files_for(cat_key, selected_value)
                        row_files.extend(files)

            # Try to attach a radial integration file for this row
            radial_path = None
            # 1) If there is a value in the explicit "Radial Integration" column, use it
            try:
                ridx = self._radial_col if hasattr(self, "_radial_col") else None
                if ridx is not None:
                    rv = self.vars.get((i, ridx))
                    if rv:
                        cand = (rv.get() or "").strip()
                        if cand and cand != "None" and os.path.exists(cand):
                            radial_path = cand
            except Exception:
                pass
            # 2) If not set, try cached radial from the selected SAXS JSON
            if not radial_path and selected_saxs_name:
                jp = None
                try:
                    if hasattr(self, "_saxs_json_for_name"):
                        jp = self._saxs_json_for_name.get(selected_saxs_name)
                except Exception:
                    jp = None
                if jp:
                    try:
                        self._ensure_remote = getattr(self, "_ensure_radial_for_json", None)
                        if callable(self._ensure_remote):
                            self._ensure_remote(Path(jp))
                    except Exception:
                        pass
                    key = str(Path(jp).resolve())
                    if hasattr(self, "_radial_by_saxs") and isinstance(self._radial_by_saxs, dict):
                        radial_path = self._radial_by_saxs.get(key)

            # 3) If still nothing, last resort: infer from the folder next to the first SAXS file
            if not radial_path and row_files:
                try:
                    for f in row_files:
                        base = os.path.basename(f).lower()
                        if base.startswith("saxs_") and f.endswith((".csv", ".dat")):
                            guess = _auto_find_radial_for_azi_input(Path(f).parent)
                            if guess and os.path.exists(guess):
                                radial_path = guess
                                break
                except Exception:
                    pass

            if radial_path:
                radial_path = _prefer_radial_csv(radial_path)
            if radial_path and radial_path not in row_files:
                row_files.append(radial_path)
                print(f"[Writer] Row {i+1}: appended radial file → {radial_path}")

            # --- Insert warning if row has non-None selection but no files ---
            if not row_files and any(self.vars.get((i, j), tk.StringVar(value="None")).get() != "None" for j in range(len(columns_list))):
                messagebox.showwarning("No files found", f"No files could be resolved for row {i+1}. Check that the selected dataset has summary CSVs in your _Processed folders.")

            if row_files:
                selected_files_per_row.append(row_files)
                safe_token = name_token.replace(" ", "_") if name_token else f"Row_{i+1}"
                output_file_name = f"{base_name}_{safe_token}.dgraph"
                output_file_names.append(output_file_name)
                print(f"[Writer] Row {i+1}: {len(row_files)} files -> naming token='{name_token}'")

        total_rows = len(output_file_names)
        self.progress_bar.configure(maximum=max(1, total_rows), value=0)
        self.progress_label_var.set(f"Writing 0/{total_rows}")
        if hasattr(self, "engage_btn"):
            self.engage_btn.configure(state="disabled")
        self.update_idletasks()

        completed = 0
        try:
            # Generate the OS command for each row
            for idx, (file_list, output_file_name) in enumerate(zip(selected_files_per_row, output_file_names)):
                self.progress_label_var.set(f"Writing {idx+1}/{total_rows}: {output_file_name}")
                self.update_idletasks()
                if not file_list:
                    print(f"[Writer] Row {idx+1}: no files selected, skipping.")
                    completed += 1
                    self.progress_bar["value"] = completed
                    continue
                # Ensure files exist before launching DataGraph
                missing = [p for p in file_list if not os.path.isfile(p)]
                if missing:
                    print("[Writer] Warning: some files do not exist on disk:", missing)
                existing_files = [p for p in file_list if os.path.isfile(p)]
                if not existing_files:
                    messagebox.showwarning("No input files", f"Row {idx+1} resolved 0 existing files. Check your _Processed folder.")
                    completed += 1
                    self.progress_bar["value"] = completed
                    continue
                # Print the files that will be passed to DataGraph for this row
                print(f"[Writer] Row {idx+1} files:\n  " + "\n  ".join(existing_files))
                write_in_data = " ".join([f'"{file}"' for file in existing_files])
                script_in = f' -script "{template_path}"'
                # Insert sample_label extraction here
                sample_label = os.path.splitext(output_file_name)[0]
                env_var_input = f' -v Sample_1="{base_name}"'
                print(f'Debug: env_var_input string = {env_var_input}')  # Debug check for env_var_input
                output_path_full = os.path.join(reference_folder, output_file_name)
                Path(reference_folder).mkdir(parents=True, exist_ok=True)
                output_in = f' -output "{output_path_full}"'
                # ---- Insert debug print block here ----
                print("---- Command Components ----")
                print(f"write_in_data: {write_in_data}")
                print(f"script_in: {script_in}")
                print(f"env_var_input: {env_var_input}")
                print(f"output_in: {output_in}")
                print("----------------------------")
                # ---- End debug print block ----
                if idx == len(output_file_names) - 1:
                    os_write_command = f'./dgraph {write_in_data}{script_in}{env_var_input}{output_in} -quitAfterScript'
                else:
                    os_write_command = f'./dgraph {write_in_data}{script_in}{env_var_input}{output_in}'

                # Insert requested debug lines before execute_command
                print("ENV VAR INPUT:", repr(env_var_input))
                print("FULL CMD:", repr(os_write_command))
                self._background_datagraph()
                self.execute_command(os_write_command, datagraph_path, output_full_path=output_path_full)
                self._background_datagraph()
                completed += 1
                self.progress_bar["value"] = completed
                self.update_idletasks()
        finally:
            self.progress_label_var.set(f"Done ({completed}/{total_rows})")
            if hasattr(self, "engage_btn"):
                self.engage_btn.configure(state="normal")
            self.update_idletasks()
            self._prompt_quit_datagraph()

    def execute_command(self, command, working_directory, output_full_path=None):
        """ Runs the OS command for DataGraph and ensures directory is restored. Detects executable and working directory automatically. """
        # Allow working_directory to be either a directory or a full path to the executable
        if os.path.isfile(working_directory) and os.access(working_directory, os.X_OK):
            exec_dir = os.path.dirname(working_directory)
            exec_name = os.path.basename(working_directory)
            # Normalize: change to its directory and use discovered exec_name
            prev_cwd = os.getcwd()
            try:
                os.chdir(exec_dir)
                command = command.replace("./dgraph", f"./{exec_name}")
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error: Data writing failed: {e}")
            finally:
                os.chdir(prev_cwd)
            return
        # Detect executable in the working directory; if not found, try common alternate path
        def _find_exec_dir_and_name(dir_path):
            candidates = ["dgraph", "DataGraph", "datagraph"]
            for name in candidates:
                p = os.path.join(dir_path, name)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return dir_path, name
            return None, None

        exec_dir, exec_name = _find_exec_dir_and_name(working_directory)
        if exec_name is None and working_directory.endswith("/Contents/Library"):
            alt_dir = working_directory.replace("/Contents/Library", "/Contents/MacOS")
            exec_dir, exec_name = _find_exec_dir_and_name(alt_dir)
            if exec_name:
                print(f"[Writer] Using alternate DataGraph dir: {alt_dir}")
                working_directory = alt_dir
        if exec_name is None:
            messagebox.showerror("DataGraph not found", f"No 'dgraph' executable in {working_directory}. Please set the correct DataGraph path.")
            return
        # Normalize the command to use the discovered executable name
        command = command.replace("./dgraph", f"./{exec_name}")

        prev_cwd = os.getcwd()
        try:
            os.chdir(working_directory)
            subprocess.run(command, shell=True, check=True)
            # Determine output file path
            if output_full_path is None:
                # Best-effort extraction
                out_idx = command.rfind("-output ")
                if out_idx != -1:
                    part = command[out_idx + len("-output "):].strip()
                    if part.startswith('"'):
                        endq = part.find('"', 1)
                        output_full_path = part[1:endq] if endq != -1 else part.strip('"')
                    else:
                        output_full_path = part.split()[0]
            # Wait for the output file to be created and stabilized
            if output_full_path:
                print(f"Waiting for {os.path.basename(output_full_path)} to finish writing...")
                prev_size = -1
                for _ in range(20):  # wait up to ~5 seconds
                    if os.path.exists(output_full_path):
                        size = os.path.getsize(output_full_path)
                        if size == prev_size and size > 100:
                            print(f"{os.path.basename(output_full_path)} is stable at {size} bytes.")
                            break
                        prev_size = size
                    time.sleep(0.25)
                else:
                    print(f"Warning: {os.path.basename(output_full_path)} may not have stabilized.")
            print(f"Successfully executed: {command}")
        except subprocess.CalledProcessError as e:
            print(f"Error: Data writing failed: {e}")
        finally:
            os.chdir(prev_cwd)  # Restore previous working dir no matter what

    def get_saxs_files(self, sample_name):
        output_files_SAXS = self.load_output_data("output_SAXS.json")
        for sample in output_files_SAXS:
            if sample['name'] == sample_name:
                return sample['files']
        return []

    def get_pi_files(self, sample_name):
        output_files_PI = self.load_output_data("output_PI.json")
        for sample in output_files_PI:
            if sample['name'] == sample_name:
                return sample['files']
        return []

    def get_pli_files(self, sample_name):  # please do not change this function unless specified
        output_files_PLI = self.load_output_data("output_PLI.json")
        for sample in output_files_PLI:
            if sample['name'] == sample_name:
                return sample['files']
        return []

    def get_viscosity_files(self, sample_name):
        output_files_viscosity = self.load_output_data("output_viscosity.json")
        for sample in output_files_viscosity:
            if sample['name'] == sample_name:
                return sample['files']
        return []

    def get_files_from_json(self, json_filename, sample_name):
        """Fetch associated file paths from a JSON file based on the sample name."""
        json_path = os.path.join(self.base_path, json_filename)  # Assuming JSON files are in the base directory
        if os.path.exists(json_path):
            with open(json_path, 'r') as file:
                try:
                    data = json.load(file)
                    for sample in data:
                        if sample["name"] == sample_name:
                            return sample["files"]
                except json.JSONDecodeError:
                    print(f"Warning: {json_filename} is corrupted or empty.")
                    return []
        return []

def save_output_data_PI(sample_name, files, json_filename="output_PI.json"):
    """Save PI output data to a JSON file."""
    if not os.path.exists(json_filename):
        # Initialize the file with an empty list if it doesn't exist
        data = []
    else:
        # Load existing data, handling empty or invalid JSON
        with open(json_filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                print(f"Warning: {json_filename} is empty or invalid. Initializing with an empty list.")
                data = []

    # Check if the sample already exists in the JSON file
    sample_exists = False
    for sample in data:
        if sample['name'] == sample_name:
            sample['files'] = files
            sample_exists = True
            break

    # If the sample doesn't exist, add it
    if not sample_exists:
        data.append({'name': sample_name, 'files': files})

    # Save the updated data back to the JSON file
    with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Saved PI output data for sample '{sample_name}' to {json_filename}.")   
