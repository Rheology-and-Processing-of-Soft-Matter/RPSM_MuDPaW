last_selected_tab = "Steady State"

# Track open "Sample Details" windows by sample name to prevent duplicates
_pi_windows_by_sample = {}

import os
import sys
import subprocess
import threading
import shutil
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

base_path = os.path.dirname(os.path.abspath(__file__))
ROUTINES_DIR = os.path.abspath(os.path.join(base_path, "..", "_Routines"))
PI_ROUTINES_DIR = os.path.join(ROUTINES_DIR, "PI")


def select_and_confirm_file(default_dir=None, prompt="Select a File", parent=None):
    """ Opens a file dialog and asks for overwrite confirmation if needed.
    
    NOTE: Do NOT create a new Tk() root here—this runs inside an existing Tk app.
    """
    # Use existing Tk context; filedialog can be parented to the current window
    file_path = None
    while not file_path:  # Keep asking until a file is selected
        if parent is not None:
            file_path = filedialog.askopenfilename(title=prompt, initialdir=default_dir, parent=parent)
        else:
            file_path = filedialog.askopenfilename(title=prompt, initialdir=default_dir)
        if not file_path:
            print("No file selected. Please select a valid file.")
    
    # Check if file exists and confirm overwrite
    while os.path.exists(file_path):
        print(f"The file '{file_path}' already exists.")
        sys.stdout.flush()  # Ensure the prompt is visible before waiting for input
        overwrite = input("Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Please select a new file.")
            file_path = None  # Reset file path and reopen browser window
            while not file_path:
                if parent is not None:
                    file_path = filedialog.askopenfilename(title=prompt, initialdir=default_dir, parent=parent)
                else:
                    file_path = filedialog.askopenfilename(title=prompt, initialdir=default_dir)
    
    return file_path


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
            # Preserve absolute prefix if original path was absolute
            prefix = os.sep if abspath.startswith(os.sep) else ""
            reference = prefix + os.sep.join(parts[:i])
            if reference in ("", os.sep):
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
    """Return the PI sample folder (child of the PI modality)."""
    try:
        abspath = os.path.abspath(path)
        cur = abspath
        while True:
            parent = os.path.dirname(cur)
            if not parent or parent == cur:
                return abspath
            if os.path.basename(parent).lower() == "pi":
                return cur
            cur = parent
    except Exception:
        return os.path.abspath(path)


def get_temp_processed_folder(path: str) -> str:
    """Return the sample-local `_Temp_processed` folder."""
    sample_root = get_sample_root(path)
    temp_root = os.path.join(sample_root, "_Temp_processed")
    os.makedirs(temp_root, exist_ok=True)
    return temp_root


def mirror_temp_file_to_processed(path_hint: str, filename: str):
    """Copy a file from the sample `_Temp_processed` into the reference `_Processed` folder."""
    try:
        temp_dir = get_temp_processed_folder(path_hint)
        src = os.path.join(temp_dir, filename)
        if not os.path.isfile(src):
            return
        dest_dir = get_unified_processed_folder(path_hint)
        os.makedirs(dest_dir, exist_ok=True)
        dest = os.path.join(dest_dir, filename)
        shutil.copy2(src, dest)
        print(f"[PI] Mirrored {filename} to {dest}")
    except Exception as exc:
        print(f"[PI] Failed to mirror {filename}: {exc}")

# Dictionary to store the last entered gap values for each sample
last_gap_values = {}


def _render_pi_controls(container, host, path, subfolder_name, sample_name,
                        add_action, update_actions_list, load_last_folder):
    """Build the PI control notebook inside `container`."""
    global last_selected_tab
    for child in container.winfo_children():
        child.destroy()

    if not hasattr(container, "rowconfigure"):
        # Some ttk widgets (like LabelFrame) expose rowconfigure via master
        pass
    else:
        container.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)

    header = ttk.Frame(container)
    header.pack(fill=tk.X, padx=12, pady=(0, 8))
    ttk.Label(header, text=f"Sample: {sample_name}", font=("TkDefaultFont", 11, "bold")).pack(anchor="w")
    ttk.Label(header, text=f"Process folder: {subfolder_name}").pack(anchor="w")

    notebook = ttk.Notebook(container)
    notebook.pack(expand=True, fill='both', padx=8, pady=(0, 8))

    def on_tab_change(event):
        global last_selected_tab
        tab = event.widget.tab(event.widget.index("current"))["text"]
        last_selected_tab = tab

    notebook.bind("<<NotebookTabChanged>>", on_tab_change)

    def _resolve_reference_folder():
        try:
            # Support either callable signature load_last_folder() or load_last_folder(self)
            try:
                reference = load_last_folder()
            except TypeError:
                reference = load_last_folder(host)  # type: ignore
        except Exception:
            reference = get_reference_folder_from_path(path)
        return reference

    def _populate_ss_header(label_widget):
        ref = _resolve_reference_folder()
        try:
            label_widget.config(text=f"Reference folder: {ref}")
        except Exception:
            pass

    last_gap = last_gap_values.get(sample_name, "1")

    # Steady State Tab
    ss_frame = ttk.Frame(notebook)
    notebook.add(ss_frame, text="Steady State")

    ref_label = ttk.Label(ss_frame, text="Reference folder: …")
    ref_label.pack()
    container.after(0, _populate_ss_header, ref_label)

    samp_label = ttk.Label(ss_frame, text=f"Sample name: {sample_name}")
    samp_label.pack()

    ttk.Label(ss_frame, text="Enter gap value:").pack()
    gap_entry = ttk.Entry(ss_frame)
    gap_entry.insert(0, last_gap)
    gap_entry.pack()

    ttk.Button(ss_frame, text="Center Geometry",
               command=lambda: center_geometry(path, sample_name, container, add_action, update_actions_list)).pack()
    ttk.Button(ss_frame, text="Process Triggers",
               command=lambda: trigger_times(path, sample_name, container, add_action, update_actions_list)).pack()
    ttk.Button(ss_frame, text="Extract Data",
               command=lambda: engage_pi(path, sample_name, container, gap_entry, add_action, update_actions_list, mode='steady_state')).pack()
    ttk.Button(ss_frame, text="Adjust Data",
               command=lambda: adjust_data(path, sample_name, container, gap_entry, add_action, update_actions_list, mode='steady_state')).pack()

    # Flow Reversal Tab
    fr_frame = ttk.Frame(notebook)
    notebook.add(fr_frame, text="Flow Reversal")

    ttk.Label(fr_frame, text="Enter gap value:").pack()
    fr_gap_entry = ttk.Entry(fr_frame)
    fr_gap_entry.insert(0, last_gap)
    fr_gap_entry.pack()

    ttk.Button(fr_frame, text="Center Geometry",
               command=lambda: center_geometry(path, sample_name, container, add_action, update_actions_list)).pack()
    ttk.Button(fr_frame, text="Extract Data",
               command=lambda: engage_pi(path, sample_name, container, fr_gap_entry, add_action, update_actions_list, mode='flow_reversal')).pack()
    ttk.Button(fr_frame, text="Adjust Data",
               command=lambda: adjust_data(path, sample_name, container, fr_gap_entry, add_action, update_actions_list, mode='flow_reversal')).pack()

    # Restore last selected tab
    for index in range(notebook.index("end")):
        if notebook.tab(index, "text") == last_selected_tab:
            notebook.select(index)
            break

    # Action log / status area
    actions_holder = ttk.LabelFrame(container, text="Recent actions")
    actions_holder.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
    container.sample_frame = actions_holder
    update_actions_list(container, sample_name)


def process_sample_PI(self, path, subfolder_name, sample_name, add_action=None,
                      update_actions_list=None, load_last_folder=None, target_container=None):
    # --- Compatibility fallbacks for optional callbacks ---
    if add_action is None:
        add_action = lambda *a, **k: None
    if update_actions_list is None:
        update_actions_list = lambda *a, **k: None
    if load_last_folder is None:
        # Infer reference folder from the provided path
        def load_last_folder():
            try:
                return get_reference_folder_from_path(path)
            except Exception:
                return os.path.dirname(os.path.dirname(os.path.abspath(path)))

    if target_container is not None:
        _render_pi_controls(target_container, self, path, subfolder_name, sample_name,
                            add_action, update_actions_list, load_last_folder)
        return

    # Standalone window flow (legacy behaviour)
    existing = _pi_windows_by_sample.get(sample_name)
    try:
        if existing and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
    except Exception:
        pass

    new_window = tk.Toplevel(self)
    _pi_windows_by_sample[sample_name] = new_window
    new_window.protocol("WM_DELETE_WINDOW",
                        lambda s=sample_name, w=new_window: (_pi_windows_by_sample.pop(s, None), w.destroy()))
    new_window.title("Sample Details")
    new_window.minsize(700, 480)
    new_window.geometry("900x600")
    new_window.rowconfigure(0, weight=1)
    new_window.columnconfigure(0, weight=1)

    _render_pi_controls(new_window, self, path, subfolder_name, sample_name,
                        add_action, update_actions_list, load_last_folder)

def _find_analysis_csv_dir(start_dir):
    """
    Return a directory that contains CSVs for analysis (filenames including 'retard' or 'axis').
    Prefer the provided start_dir; otherwise search one level down.
    """
    try:
        candidates = []
        # Check the start directory first
        for name in os.listdir(start_dir):
            if name.lower().endswith('.csv') and ('retard' in name.lower() or 'axis' in name.lower()):
                return start_dir
        # Shallow search subfolders
        for entry in os.scandir(start_dir):
            if entry.is_dir():
                try:
                    for name in os.listdir(entry.path):
                        if name.lower().endswith('.csv') and ('retard' in name.lower() or 'axis' in name.lower()):
                            candidates.append(entry.path)
                            break
                except Exception:
                    continue
        return candidates[0] if candidates else start_dir
    except Exception:
        return start_dir

def engage_pi(path, sample_name, window, gap_entry, add_action, update_actions_list, mode='steady_state'):
    try:
        gap = float(gap_entry.get())
        # Save the entered gap value for this sample
        last_gap_values[sample_name] = str(gap)

        csv_folder = path
        parent_folder = os.path.dirname(csv_folder)
        temp_processed = get_temp_processed_folder(path)

        geometry_file = os.path.join(temp_processed, '_Geometry_positioning.txt')
        if not os.path.exists(geometry_file):
            # Fallback: look for legacy geometry file in unified _Processed folder
            legacy_processed = get_unified_processed_folder(path)
            legacy_geo = os.path.join(legacy_processed, '_Geometry_positioning.txt')
            if os.path.exists(legacy_geo):
                geometry_file = legacy_geo

        if not os.path.exists(geometry_file):
            messagebox.showerror("Error", "Geometry positioning file not found (checked both Temp and legacy _Processed folders).")
            window.after(0, add_action, sample_name, "PI processing failed", "red")
            window.after(0, update_actions_list, window, sample_name)
            return

        triggers_file = os.path.join(temp_processed, '_Triggers') if mode != 'flow_reversal' else ''
        if mode != 'flow_reversal' and not os.path.exists(triggers_file):
            messagebox.showerror("Error", "Triggers file not found.")
            window.after(0, add_action, sample_name, "PI processing failed", "red")
            window.after(0, update_actions_list, window, sample_name)
            return

        try:
            with open(geometry_file, 'r') as file:
                geometry_data = file.readlines()

            geometry_params = {}
            for line in geometry_data:
                key, value = line.strip().split(': ')
                geometry_params[key] = int(value)

            # Check for existing output files in flow reversal mode
            if mode == 'flow_reversal':
                save_dir = temp_processed
                output_files = [
                    'retardation_circle.csv',
                    'retardation_line.csv',
                    'orientation_circle.csv',
                    'orientation_line.csv'
                ]
                existing_outputs = [f for f in output_files if os.path.exists(os.path.join(save_dir, f))]
                if existing_outputs:
                    response = input(f"The following output files already exist:\n{', '.join(existing_outputs)}\nOverwrite them? (y/n): ").strip().lower()
                    if response != 'y':
                        print("Extraction aborted by user.")
                        return

            output_files_list = [
                'a_mean_sem_circle.csv', 'mean_sem_circle.csv', 'a_mean_sem_line.csv', 'mean_sem_line.csv',
                'a_extracted_space_time_circle.csv', 'extracted_space_time_circle.csv',
                'a_extracted_space_time_line.csv', 'extracted_space_time_line.csv'
            ]
            processed_folder = temp_processed
            print(f"Temp processed folder for PI outputs: {processed_folder}")
            existing_files = [f for f in output_files_list if os.path.exists(os.path.join(processed_folder, f))]

            if existing_files:
                if not messagebox.askyesno("Overwrite", f"The following files already exist:\n{', '.join(existing_files)}\nOverwrite?"):
                    window.after(0, add_action, sample_name, "PI processing skipped", "red")
                    window.after(0, update_actions_list, window, sample_name)
                    return

            def run_extraction():
                try:
                    # Select script based on mode
                    if mode == 'flow_reversal':
                        script_path1 = os.path.join(PI_ROUTINES_DIR, "Data_extracter_flow_reversal.py")
                    else:
                        script_path1 = os.path.join(PI_ROUTINES_DIR, "Data_extracter_v2_1.py")
                    if not os.path.isfile(script_path1):
                        print(f"[PI] Processor not found: {script_path1}")
                        window.after(0, add_action, sample_name, "PI processor missing", "red")
                        window.after(0, update_actions_list, window, sample_name)
                        return
                    py = sys.executable or 'python3'
                    # Decide where results are expected for post-checks
                    expected_dir = temp_processed
                    print(
                        f"Running {script_path1} with parameters:\n"
                        f"  csv_folder: {csv_folder}\n"
                        f"  gap: {gap}\n"
                        f"  offset_x: {geometry_params['offset_x']}\n"
                        f"  offset_y: {geometry_params['offset_y']}\n"
                        f"  Inner_initial: {geometry_params['Inner_initial']}\n"
                        f"  Outer_initial: {geometry_params['Outer_initial']}\n"
                        f"  triggers_file: {triggers_file}"
                    )
                    args = [py, script_path1, csv_folder, str(gap), str(geometry_params['offset_x']), str(geometry_params['offset_y']), str(geometry_params['Inner_initial']), str(geometry_params['Outer_initial'])]
                    if triggers_file:
                        args.append(triggers_file)
                    result1 = subprocess.run(args, capture_output=True, text=True)
                    print(result1.stdout)
                    print(result1.stderr)

                    if result1.returncode != 0:
                        print(f"Error: {result1.stderr}")
                        window.after(0, add_action, sample_name, "PI processing failed", "red")
                        window.after(0, update_actions_list, window, sample_name)
                        return
                    # Post-check: make sure we have 'retard' and 'axis' CSVs somewhere sensible
                    analysis_dir = _find_analysis_csv_dir(expected_dir)
                    csvs = [f for f in os.listdir(analysis_dir) if f.lower().endswith('.csv')]
                    has_retard = any('retard' in f.lower() for f in csvs)
                    has_axis = any('axis' in f.lower() for f in csvs)
                    if mode == 'flow_reversal' and not has_axis:
                        # Try running the axis extractor too (without triggers)
                        script_path2 = os.path.join(PI_ROUTINES_DIR, "Data_extracter_v2_1_axis.py")
                        if os.path.isfile(script_path2):
                            print("[PI] Axis CSVs not found after flow-reversal extract; attempting axis extractor.")
                            args2 = [py, script_path2, csv_folder, str(gap), str(geometry_params['offset_x']), str(geometry_params['offset_y']), str(geometry_params['Inner_initial']), str(geometry_params['Outer_initial'])]
                            result2 = subprocess.run(args2, capture_output=True, text=True)
                            print(result2.stdout)
                            print(result2.stderr)
                            # Refresh check
                            try:
                                csvs = [f for f in os.listdir(analysis_dir) if f.lower().endswith('.csv')]
                                has_axis = any('axis' in f.lower() for f in csvs)
                            except Exception:
                                has_axis = False
                            else:
                                mirror_temp_file_to_processed(path, "_output_PI_axis.json")
                        else:
                            print(f"[PI] Axis processor not found: {script_path2}")
                    # Only run axis script for steady_state mode
                    if mode != 'flow_reversal':
                        script_path2 = os.path.join(PI_ROUTINES_DIR, "Data_extracter_v2_1_axis.py")
                        if not os.path.isfile(script_path2):
                            print(f"[PI] Axis processor not found: {script_path2}")
                            window.after(0, add_action, sample_name, "PI axis processor missing", "red")
                            window.after(0, update_actions_list, window, sample_name)
                            return
                        print(f"Running {script_path2} with parameters: {csv_folder, gap, geometry_params['offset_x'], geometry_params['offset_y'], geometry_params['Inner_initial'], geometry_params['Outer_initial'], triggers_file}")
                        result2 = subprocess.run([py, script_path2, csv_folder, str(gap), str(geometry_params['offset_x']), str(geometry_params['offset_y']), str(geometry_params['Inner_initial']), str(geometry_params['Outer_initial']), triggers_file], capture_output=True, text=True)
                        print(result2.stdout)
                        print(result2.stderr)

                        if result2.returncode != 0:
                            print(f"Error: {result2.stderr}")
                            window.after(0, add_action, sample_name, "PI processing failed", "red")
                            window.after(0, update_actions_list, window, sample_name)
                            return
                        mirror_temp_file_to_processed(path, "_output_PI_axis.json")

                    window.after(0, add_action, sample_name, "PI processing completed", "green")
                    window.after(0, update_actions_list, window, sample_name)
                except Exception as e:
                    print(f"Exception occurred: {e}")
                    window.after(0, add_action, sample_name, "PI processing failed", "red")
                    window.after(0, update_actions_list, window, sample_name)

            threading.Thread(target=run_extraction).start()
        except Exception as e:
            print(f"Exception occurred while processing PI for sample {sample_name} in path {path}: {e}")
            window.after(0, add_action, sample_name, "PI processing failed", "red")
            window.after(0, update_actions_list, window, sample_name)
    except ValueError:
        messagebox.showerror("Error", "Invalid gap value. Please enter a numeric value.")
        return

def center_geometry(path, sample_name, window, add_action, update_actions_list):
    print(f"Centering geometry for sample: {sample_name} in path: {path}")  # Debugging statement
    reference_folder = get_reference_folder_from_path(path)
    print(f"Reference folder resolved for centering: {reference_folder}")
    temp_processed = get_temp_processed_folder(path)
    print(f"Temp processed folder for PI: {temp_processed}")
    def run_centering():
        try:
            # Use the sample root as input for CSV discovery (not _Temp_processed)
            csv_folder = get_sample_root(path)
            script_path = os.path.join(PI_ROUTINES_DIR, "_PI_centering_image_v1.py")
            print(f"Running {script_path} with CSV folder: {csv_folder}")
            py = sys.executable or 'python3'
            process = subprocess.Popen([py, script_path, csv_folder, temp_processed], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(stdout.decode())
            print(stderr.decode())
        except Exception as e:
            print(f"Exception occurred while centering geometry for sample {sample_name} in path {path}: {e}")
            window.after(0, add_action, sample_name, "Centering failed", "red")
            window.after(0, update_actions_list, window, sample_name)

    run_centering()
    #threading.Thread(target=run_centering).start()

    # If the centering script displays a matplotlib plot, ensure non-blocking display
    try:
        import matplotlib.pyplot as plt
        plt.show(block=False)
        plt.pause(0.1)
    except ImportError:
        pass
    except Exception:
        pass

def trigger_times(path, sample_name, window, add_action, update_actions_list):
    def run_trigger_script():
        try:
            temp_processed = get_temp_processed_folder(path)
            triggers_file = os.path.join(temp_processed, '_Triggers')
            script_path = os.path.join(ROUTINES_DIR, "Time_stamper_v1.py")
            reference_folder = get_reference_folder_from_path(path)
            print(f"Reference folder resolved for trigger_times: {reference_folder}")
            print(f"Temp processed folder for triggers: {temp_processed}")
            if os.path.exists(triggers_file):
                if not messagebox.askyesno("Overwrite", "The triggers have already been processed. Overwrite?"):
                    window.after(0, add_action, sample_name, "Trigger processing skipped", "red")
                    window.after(0, update_actions_list, window, sample_name)
                    return

            py = sys.executable or 'python3'
            process = subprocess.Popen([py, script_path, temp_processed, triggers_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
            print(stdout.decode())
            print(stderr.decode())

            if os.path.exists(triggers_file):
                window.after(0, add_action, sample_name, "Trigger processing completed", "green")
            else:
                window.after(0, add_action, sample_name, "Trigger processing failed", "red")

            window.after(0, update_actions_list, window, sample_name)
        except Exception as e:
            print(f"Exception occurred while processing triggers for sample {sample_name} in path {path}: {e}")
            window.after(0, add_action, sample_name, "Trigger processing failed", "red")
            window.after(0, update_actions_list, window, sample_name)

    run_trigger_script()

def adjust_data(path, sample_name, window, gap_entry, add_action, update_actions_list, mode):
    gap = float(gap_entry.get())
    print(f"Adjusting data for sample: {sample_name} in path: {path} with gap: {gap} (mode: {mode})")

    temp_processed = get_temp_processed_folder(path)
    processed_folder = temp_processed
    json_path = os.path.join(processed_folder, f"_output_PI_{sample_name}.json")
    final_json = os.path.join(get_unified_processed_folder(path), f"_output_PI_{sample_name}.json")
    print(f"PI data directory for adjuster: {processed_folder}")
    print(f"Writing PI output JSON to (temp): {json_path}")
    print(f"Final PI output JSON will mirror to: {final_json}")

    def run_adjuster():
        try:
            # Select script based on mode
            if mode == 'flow_reversal':
                script_path = os.path.join(PI_ROUTINES_DIR, "Data_adjuster_flow_reversal.py")
            else:
                script_path = os.path.join(PI_ROUTINES_DIR, "Data_adjuster_v4.py")
            # Use the selected processed_folder for adjustment
            print(f"Running {script_path} with parameters: {processed_folder, sample_name, gap}")
            result = subprocess.run(['python3', script_path, processed_folder, sample_name, str(gap)], capture_output=True, text=True)
            print(result.stdout)
            print(result.stderr)

            if result.returncode != 0:
                print(f"Error: {result.stderr}")
                window.after(0, add_action, sample_name, "Data adjustment failed", "red")
                window.after(0, update_actions_list, window, sample_name)
                return

            mirror_temp_file_to_processed(path, f"_output_PI_{sample_name}.json")
            window.after(0, add_action, sample_name, "Data adjustment completed", "green")
            window.after(0, update_actions_list, window, sample_name)
        except Exception as e:
            print(f"Exception occurred while adjusting data for sample {sample_name} in path {path}: {e}")
            window.after(0, add_action, sample_name, "Data adjustment failed", "red")
            window.after(0, update_actions_list, window, sample_name)

    run_adjuster()
    #threading.Thread(target=run_adjuster).start()  
