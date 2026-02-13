import os
import json
import glob
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import importlib

# --- Path hygiene helpers ---

def _abs_path(p: str) -> str:
    """Expanduser + abspath and enforce a leading slash on macOS."""
    if not isinstance(p, str):
        return p
    p2 = os.path.abspath(os.path.expanduser(p))
    if p2 and not p2.startswith(os.sep):
        p2 = os.sep + p2
    return p2



def _clean_user_path(p: str) -> str:
    """If a repo prefix was concatenated before '/Users/...', trim to the real user path.
    Keeps leading slash; otherwise returns the original.
    """
    if not isinstance(p, str):
        return p
    p2 = _abs_path(p)
    needle = "/Users/"
    j = p2.find(needle)
    if j <= 0:
        return p2
    k = p2.find(needle, j + 1)
    if k != -1:
        fixed = p2[k:]
        if not fixed.startswith(os.sep):
            fixed = os.sep + fixed
        print(f"[COMMON] sanitize: trimmed misjoined path\n  in : {p2}\n  out: {fixed}")
        return fixed
    return p2

# --- Rebase helper for paths stored with foreign absolute prefixes ---

def _rebase_to_processed(maybe_path, reference_folder):
    """Rebase an absolute path that points to another machine so it resolves
    inside the current reference folder's `_Processed` tree. If the file
    already exists, return it unchanged. Strategy:
      A) If the path contains '/_Processed/', take the suffix after it and
         join with the local reference_folder/_Processed/.
      B) Otherwise, or if (A) fails, search by basename under the reference.
    Returns a valid local absolute path if found, else None.
    """
    try:
        if not isinstance(maybe_path, str):
            return None
        # Normalize incoming string and early‑out if already valid
        p = _abs_path(maybe_path)
        if os.path.isfile(p):
            return p

        # Strategy A: rebase using suffix after '_Processed/'
        parts = p.split('/_Processed/', 1)
        if len(parts) == 2:
            suffix = parts[1]
            cand = os.path.join(reference_folder, '_Processed', suffix)
            cand = _abs_path(cand)
            if os.path.isfile(cand):
                print(f"[COMMON][PATH] Rebased → {cand}")
                return cand

        # Strategy B: fallback — search by basename under current reference
        base = os.path.basename(p)
        hits = glob.glob(os.path.join(reference_folder, '**', base), recursive=True)
        for h in hits:
            h2 = _abs_path(h)
            if os.path.isfile(h2):
                print(f"[COMMON][PATH] Resolved by basename → {h2}")
                return h2

        print(f"[COMMON][PATH] Could not rebase: {maybe_path}")
        return None
    except Exception as e:
        print(f"[COMMON][PATH] Rebase error for {maybe_path}: {e}")
        return None


def normalize_paths_list(paths, reference_folder):
    """Given a list of file paths, return a list where each element has been
    validated or rebased to the current reference folder if necessary.
    Supports absolute and relative entries.
    """
    out = []
    for p in paths or []:
        rp = _rebase_one(p, reference_folder)
        if rp:
            out.append(rp)
    return out


def normalize_saxs_json(obj, reference_folder):
    """Normalize `_output_SAXS_*.json` structure in‑memory by fixing any
    foreign absolute paths under the 'csv_outputs' key.
    Usage:
        with open(fp) as f:
            obj = json.load(f)
        obj = normalize_saxs_json(obj, reference_folder)
    """
    if not isinstance(obj, dict):
        return obj
    # Reuse generic merger for both absolute and relative fields
    return _normalize_paths_in_dict(obj, ['csv_outputs'], reference_folder)


# --- Generic path normalization for modality JSONs ---

def _rebase_one(p, reference_folder):
    """Rebase a single path string; returns valid absolute path or None.
    Also resolves relative paths by trying reference_folder/_Processed/<p>.
    """
    if not isinstance(p, str):
        return None
    # If p looks relative, try joining under current reference/_Processed first
    if not os.path.isabs(p):
        cand = _abs_path(os.path.join(reference_folder, '_Processed', p))
        if os.path.isfile(cand):
            return cand
    p_abs = _abs_path(p)
    if os.path.isfile(p_abs):
        return p_abs
    return _rebase_to_processed(p_abs, reference_folder)


def _normalize_paths_in_dict(d: dict, keys, reference_folder):
    """For each key in `keys`, normalize value(s) if they are str / list / dict.
    Also merges corresponding `*_rel` keys into the primary key when present.
    Mutates and returns the dict.
    """
    if not isinstance(d, dict):
        return d

    def _norm_list(val_list):
        return normalize_paths_list(val_list, reference_folder)

    def _norm_one(val):
        r = _rebase_one(val, reference_folder)
        return [r] if r else []

    for k in keys:
        if k not in d and (k + '_rel') not in d:
            continue

        v = d.get(k, None)
        v_rel = d.get(k + '_rel', None)

        # Handle dict values (name->path) as before
        if isinstance(v, dict):
            out = {}
            for name, path in v.items():
                if isinstance(path, str):
                    rp = _rebase_one(path, reference_folder)
                    out[name] = rp if rp else path
                elif isinstance(path, list):
                    out[name] = _norm_list(path)
                else:
                    out[name] = path
            d[k] = out
            continue

        # Normalize primary
        acc = []
        if isinstance(v, list):
            acc.extend(_norm_list(v))
        elif isinstance(v, str):
            acc.extend(_norm_one(v))

        # Merge from *_rel
        if isinstance(v_rel, list):
            acc.extend(_norm_list(v_rel))
        elif isinstance(v_rel, str):
            acc.extend(_norm_one(v_rel))

        # Dedupe preserving order
        seen = set()
        combined = []
        for item in acc:
            if item and item not in seen:
                seen.add(item)
                combined.append(item)

        # Choose scalar vs list form based on pluralization heuristic
        is_plural = k.endswith('s')
        if not is_plural:
            d[k] = combined[0] if combined else d.get(k)
        else:
            d[k] = combined
    return d


def normalize_modal_json(obj, reference_folder, keys=(
    'csv_outputs', 'image_outputs', 'files', 'paths', 'outputs'
)):
    """Normalize common path-carrying keys in an arbitrary modality JSON.
    - Works for str, list, or dict values.
    - Leaves unknown structures untouched.
    """
    if isinstance(obj, dict):
        return _normalize_paths_in_dict(obj, keys, reference_folder)
    # lists of entries like [{name, files:[...]}, ...]
    if isinstance(obj, list):
        for i, item in enumerate(obj):
            if isinstance(item, dict):
                obj[i] = _normalize_paths_in_dict(item, keys, reference_folder)
        return obj
    return obj


# --- Modality-specific wrappers (call these where you load each JSON) ---

def normalize_pi_json(obj, reference_folder):
    return normalize_modal_json(obj, reference_folder, keys=('files', 'image_outputs', 'paths'))


def normalize_pli_json(obj, reference_folder):
    return normalize_modal_json(obj, reference_folder, keys=('files', 'image_outputs', 'paths'))


def normalize_rheology_json(obj, reference_folder):
    return normalize_modal_json(obj, reference_folder, keys=('csv_output', 'csv_outputs', 'files', 'paths'))

# Define global variables
performed_actions = {}  # Ensure this variable is defined
global_log_widget = None
last_folder_file = "last_folder.txt"
renamed_samples_file = "renamed_samples.json"
performed_actions_file = "performed_actions.json"  # Define this variable
output_files_SAXS_file = "output_SAXS.json"
output_files_PI_file = "output_PI.json"
output_files_viscosity_file = "output_viscosity.json"
output_files_PLI_file = "output_PLI.json"
last_selections_file = "last_selections.json"
last_preset_values_file = "last_preset_values.json"

# Define standard subfolders
standard_subfolders = {
    'PI': 'PI',
    'SAXS': 'SAXS',
    'Viscosity': 'Viscosity',
    'PLI': 'PLI'
}

def rename_sample(entry, sample, current_folder):
    """Rename the sample folder and update the sample name."""
    new_name = entry.get().strip()  # Get the new name from the entry widget
    if not new_name:
        print("Error: New name cannot be empty.")
        return

    old_path = os.path.join(current_folder, sample)  # Path to the old folder
    new_path = os.path.join(current_folder, new_name)  # Path to the new folder

    if os.path.exists(new_path):
        print(f"Error: A folder with the name '{new_name}' already exists.")
        return

    try:
        os.rename(old_path, new_path)  # Rename the folder
        print(f"Renamed '{sample}' to '{new_name}'.")

        # Update the renamed_samples.json file
        renamed_samples = load_renamed_samples()
        renamed_samples[sample] = new_name
        save_renamed_samples(renamed_samples)

    except Exception as e:
        print(f"Error renaming folder: {e}")

def confirm_clear_all(self):
    if messagebox.askyesno("Clear all", "This will erase all recorded progress. Are you sure?"):
        clear_all_data()
        self.folder_label.config(text="No folder chosen")
        for widget in self.sample_frame.winfo_children():
            widget.destroy()

def clear_all_data():
    global performed_actions
    performed_actions = {}
    stored_geometry_centering = {}
    stored_triggers = {}

    # Remove cached state files if present
    for path in [
        last_folder_file,
        renamed_samples_file,
        performed_actions_file,
        last_selections_file,
        last_preset_values_file,
        output_files_SAXS_file,
        output_files_PI_file,
        output_files_viscosity_file,
        output_files_PLI_file,
    ]:
        if os.path.exists(path):
            os.remove(path)

def load_last_folder(self):
    last_folder = load_last_folder_internal()
    if last_folder:
        last_folder = _clean_user_path(last_folder)
        self.folder_label.config(text=f"Reference folder:\n{last_folder}")
        self.scan_subfolders(last_folder)

def load_last_folder_internal():
    try:
        with open(last_folder_file, 'r') as file:
            raw = file.read().strip()
            return _clean_user_path(raw)
    except FileNotFoundError:
        return ''

def set_global_log_widget(widget):
    """Register the Tk Text widget that should act as the shared terminal/log."""
    global global_log_widget
    global_log_widget = widget


def update_actions_list(window, sample_name):
    text_widget = getattr(window, "log_text", None)
    if not isinstance(text_widget, tk.Text):
        text_widget = global_log_widget
    if isinstance(text_widget, tk.Text):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        entries = performed_actions.get(sample_name, [])
        if not entries:
            text_widget.insert(tk.END, f"{sample_name}: No recorded actions yet.\n")
        else:
            for action, _color in entries:
                text_widget.insert(tk.END, f"{sample_name}: {action}\n")
        text_widget.config(state=tk.DISABLED)
        return

    actions_list = getattr(window, "sample_frame", None)
    if actions_list is None:
        actions_list = ttk.Frame(window)
        actions_list.pack(pady=10, fill=tk.BOTH, expand=True)
        window.sample_frame = actions_list

    for widget in actions_list.winfo_children():
        widget.destroy()
    if sample_name in performed_actions:
        for action, color in performed_actions[sample_name]:
            label = ttk.Label(actions_list, text=action, foreground=color)
            label.pack()

def add_action_ui(self, sample_name, action, color):
    global performed_actions  # Ensure this variable is used correctly
    if sample_name not in performed_actions:
        performed_actions[sample_name] = []
    performed_actions[sample_name].append((action, color))
    self.after(0, update_actions_list, self, sample_name)

# New free function for non-UI contexts
def add_action(sample_name, action, color):
    """Record an action without needing a Tkinter `self`.
    UI refresh should be triggered separately via `update_actions_list`.
    """
    global performed_actions
    if sample_name not in performed_actions:
        performed_actions[sample_name] = []
    performed_actions[sample_name].append((action, color))

def create_sample_button_in_window(self, window, sample, sample_name, path, subfolder_name, add_action, update_actions_list, current_folder):
    frame = ttk.Frame(window)
    frame.pack(fill=tk.X)
    button = ttk.Button(frame, text=sample_name, command=lambda: process_sample(self, path, subfolder_name, sample_name, subfolder_name, add_action, update_actions_list, load_last_folder))
    button.pack(side=tk.LEFT)
    entry = ttk.Entry(frame)
    entry.insert(0, sample_name)
    entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    rename_button = ttk.Button(frame, text="Rename", command=lambda: rename_sample(entry, sample, current_folder))
    rename_button.pack(side=tk.LEFT)

def process_sample(self, path, subfolder_name, sample_name, subfolder_type, add_action, update_actions_list, load_last_folder):
    # Lazy-import helper module and call the best-available processing function
    module_name = None
    candidates = []
    if subfolder_type == 'SAXS':
        module_name = 'SAXS_helper'
        candidates = ['process_sample_SAXS', 'process_sample_saxs', 'process_saxs', 'process_sample']
    elif subfolder_type == 'PI':
        module_name = 'PI_helper'
        candidates = ['process_sample_PI', 'process_sample_pi', 'process_pi', 'process_sample']
    elif subfolder_type == 'Viscosity':
        module_name = 'Viscosity_helper'
        candidates = ['process_sample_viscosity', 'process_viscosity', 'process_sample']
    elif subfolder_type == 'PLI':
        module_name = 'PLI_helper'
        candidates = ['process_sample_PLI', 'process_sample_pli', 'process_pli', 'process_sample']
    else:
        messagebox.showerror("Unknown type", f"Unknown subfolder type: {subfolder_type}")
        return

    # Ensure path passed to helpers is a clean user path
    path = _clean_user_path(path)

    try:
        mod = importlib.import_module(module_name)
    except Exception as e:
        messagebox.showerror("Import error", f"Failed to import {module_name}: {e}")
        return

    func = None
    for name in candidates:
        if hasattr(mod, name):
            func = getattr(mod, name)
            break

    if func is None:
        messagebox.showerror(
            "Function not found",
            f"'{module_name}' does not define any of: {', '.join(candidates)}"
        )
        return

    # Call the located function
    try:
        func(self, path, subfolder_name, sample_name, add_action, update_actions_list, load_last_folder)
    except TypeError:
        # Some helpers might not expect all parameters; try a reduced signature.
        try:
            func(self, path, subfolder_name, sample_name, add_action, update_actions_list)
        except TypeError:
            try:
                func(self, path, subfolder_name, sample_name)
            except Exception as e:
                messagebox.showerror("Run error", f"Error while running {module_name}.{func.__name__}: {e}")
                return
    except Exception as e:
        messagebox.showerror("Run error", f"Error while running {module_name}.{func.__name__}: {e}")
        return

def save_last_selections(selections):
    with open(last_selections_file, "w") as file:
        json.dump(selections, file)

def load_last_selections():
    if os.path.exists(last_selections_file):
        with open(last_selections_file, "r") as file:
            return json.load(file)
    return {}

def save_last_folder(folder):
    folder = _clean_user_path(folder)
    with open(last_folder_file, "w") as file:
        file.write(folder)

def save_renamed_samples(renamed_samples):
    with open(renamed_samples_file, "w") as file:
        json.dump(renamed_samples, file)

def load_renamed_samples():
    try:
        with open(renamed_samples_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        return {}

def save_last_preset_values(preset_start, preset_end):
    with open(last_preset_values_file, "w") as file:
        json.dump({"preset_start": preset_start, "preset_end": preset_end}, file)

def load_last_preset_values():
    if os.path.exists(last_preset_values_file):
        with open(last_preset_values_file, "r") as file:
            values = json.load(file)
            return values.get("preset_start", 50), values.get("preset_end", 95)
    return 50, 95

def update_json_file(json_filename, sample_name, new_files, overwrite=False, metadata=None):
    """
    Updates the JSON file to ensure no duplicate entries exist.

    Args:
        json_filename (str): Path to the JSON file.
        sample_name (str): Sample name to update.
        new_files (list): List of file paths to add.
        overwrite (bool): If True, replaces existing entry. If False, appends unique paths.
    """
    # Load existing data if file exists
    if os.path.exists(json_filename):
        with open(json_filename, 'r') as file:
            try:
                data = json.load(file)
            except json.JSONDecodeError:
                data = []
    else:
        data = []

    safe_meta = {}
    if metadata:
        for k, v in metadata.items():
            if v is None:
                continue
            safe_meta[k] = str(v) if isinstance(v, Path) else v

    for entry in data:
        if entry["name"] == sample_name:
            if overwrite:
                entry["files"] = new_files  # Overwrite existing files
            else:
                # Append only new files (prevent duplicates)
                entry["files"] = list(set(entry["files"] + new_files))
            entry.update(safe_meta)
            break
    else:
        # If sample does not exist, add new entry
        new_entry = {"name": sample_name, "files": new_files}
        new_entry.update(safe_meta)
        data.append(new_entry)

    # Write updated data back to JSON
    with open(json_filename, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Updated {json_filename} successfully!")
