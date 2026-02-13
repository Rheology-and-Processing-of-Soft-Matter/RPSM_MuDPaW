import os, sys
print(">>> DEBUG: running Data_adjuster_v4.py from:", os.path.abspath(__file__))
print(">>> DEBUG: sys.argv:", sys.argv)
def apply_one_step_circle(circle_data, angle_data, x_path1):
    """
    Applies 1-step correction to circle_data using externally supplied x_path1.
    Returns corrected_circle, corrected_angle.
    """
    corrected_circle = circle_data.copy()
    corrected_angle = angle_data.copy()

    height, width = circle_data.shape

    print(">>> DEBUG: using clipped version of apply_one_step_circle (width check active)")

    for y in range(height):
        x1 = int(np.clip(x_path1[y], 0, width - 1))
        ref1 = corrected_circle[y, x1]
        for x in range(x1, width):
            corrected_circle[y, x] = 2 * ref1 - corrected_circle[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180 - 1

    return corrected_circle, corrected_angle

# --- New function for line 1-step correction ---
def apply_one_step_line(line_data, angle_data, x_path1):
    """
    Applies 1-step correction to line_data using externally supplied x_path1.
    Returns corrected_line, corrected_angle.
    """
    corrected_line = line_data.copy()
    corrected_angle = angle_data.copy()

    height, width = line_data.shape

    for y in range(height):
        x1 = int(np.clip(x_path1[y], 0, width - 1))
        ref1 = corrected_line[y, x1]
        for x in range(x1, width):
            corrected_line[y, x] = 2 * ref1 - corrected_line[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180 - 1

    return corrected_line, corrected_angle

def apply_two_step_circle(circle_data, angle_data, x_path1, x_path2):
    """
    Applies 2-step correction to circle_data using externally supplied x_path1 and x_path2.
    Returns corrected_circle, corrected_angle.
    """
    corrected_circle = circle_data.copy()
    corrected_angle = angle_data.copy()

    height, width = circle_data.shape

    for y in range(height):
        x1 = int(np.clip(x_path1[y], 0, width - 1))
        x2 = int(np.clip(x_path2[y], 0, width - 1))
        ref1 = corrected_circle[y, x1]
        for x in range(x1, min(x2 + 1, width)):
            corrected_circle[y, x] = 2 * ref1 - corrected_circle[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180 - 1

    for y in range(height):
        x1 = int(np.clip(x_path1[y], 0, width - 1))
        x2 = int(np.clip(x_path2[y], 0, width - 1))
        ref2 = corrected_circle[y, x2]
        for x in range(x2, width):
            corrected_circle[y, x] = ref2 + corrected_circle[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180 - 1

    return corrected_circle, corrected_angle


# --- New function for line 2-step correction ---
def apply_two_step_line(line_data, angle_data, x_path1, x_path2):
    """
    Applies 2-step correction to line_data using externally supplied x_path1 and x_path2.
    Returns corrected_line, corrected_angle.
    """
    corrected_line = line_data.copy()
    corrected_angle = angle_data.copy()

    height, width = line_data.shape

    for y in range(height):
        x1 = int(np.clip(x_path1[y], 0, width - 1))
        x2 = int(np.clip(x_path2[y], 0, width - 1))
        ref1 = corrected_line[y, x1]
        for x in range(x1, min(x2 + 1, width)):
            corrected_line[y, x] = 2 * ref1 - corrected_line[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180 - 1

    for y in range(height):
        x1 = int(np.clip(x_path1[y], 0, width - 1))
        x2 = int(np.clip(x_path2[y], 0, width - 1))
        ref2 = corrected_line[y, x2]
        for x in range(x2, width):
            corrected_line[y, x] = ref2 + corrected_line[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180 - 1

    return corrected_line, corrected_angle
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import json
import shutil
from scipy.ndimage import label
from scipy.interpolate import interp1d
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

def get_unified_processed_folder(path):
    """Return the unified `_Processed` folder inside the resolved *reference* folder."""
    reference_folder = get_reference_folder_from_path(path)
    processed_root = os.path.join(reference_folder, "_Processed")
    os.makedirs(processed_root, exist_ok=True)
    return processed_root


def get_sample_root(path):
    abspath = os.path.abspath(path)
    cur = abspath
    while True:
        parent = os.path.dirname(cur)
        if not parent or parent == cur:
            return abspath
        if os.path.basename(parent).lower() == "pi":
            return cur
        cur = parent


def get_temp_processed_folder(path):
    sample_root = get_sample_root(path)
    temp_root = os.path.join(sample_root, "_Temp_processed")
    os.makedirs(temp_root, exist_ok=True)
    return temp_root

def compute_mean_path(binary_image, label='circle'):
    """
    Computes the mean (or rightmost, for 'line') x-position of '1' pixels for each row in a binarized image.
    For 'circle', applies smoothing; for 'line', uses rightmost edge and no smoothing.
    Returns: np.ndarray of shape (height, 2), columns are (x, y)
    """
    height, width = binary_image.shape
    mean_binarization_path = []
    if 'line' in label.lower():
        last_valid_x = 0  # left edge for 'line'
        use_limit = '2-step' in label.lower()
        width_limit = int(width * 0.7) if use_limit else width
        for y in range(height):
            x_positions = np.where(binary_image[y, :width_limit] == 1)[0]
            if len(x_positions) > 0:
                x_val = np.max(x_positions)
                last_valid_x = x_val
            else:
                x_val = last_valid_x
            mean_binarization_path.append((x_val, y))
        # No smoothing for line
        return np.array(mean_binarization_path)
    else:
        last_valid_x = width // 2  # center for 'circle'
        for y in range(height):
            x_positions = np.where(binary_image[y, :] == 1)[0]
            if len(x_positions) > 0:
                x_val = np.mean(x_positions)
                last_valid_x = x_val
            else:
                x_val = last_valid_x
            mean_binarization_path.append((x_val, y))
        # Apply smoothing for 'circle'
        x_vals = np.array([p[0] for p in mean_binarization_path])
        y_vals = np.array([p[1] for p in mean_binarization_path])
        window_size = 10
        half_window = window_size // 2
        smoothed_x_vals = x_vals.copy()
        for i in range(len(x_vals)):
            start = max(0, i - half_window)
            end = min(len(x_vals), i + half_window + 1)
            window = x_vals[start:end]
            sem = np.std(window) / np.sqrt(len(window))
            if sem > 0.05 * np.mean(window):
                smoothed_x_vals[i] = np.mean(window)
        mean_binarization_path = list(zip(smoothed_x_vals, y_vals))
        return np.array(mean_binarization_path)

# --- Threshold GUI function ---
def launch_threshold_gui(image_circle, image_line, image_a_circle, image_a_line, title="Threshold Adjustment", initial_thresholds=None, initial_alphas=None, output_dir=None):
    # Additional controls for 2nd Step Threshold Adjustment
    cutoff_column = None
    threshold_use_binarization = None
    import os
    import json
    # --- Load previous threshold settings if available ---
    # This block must be after threshold_use_binarization is initialized
    root = tk.Tk()
    root.withdraw()
    root.title(title)
    root.geometry("1280x800")
    # --- Add a separate control window ---
    control_window = tk.Toplevel(root)
    control_window.title("Threshold Controls")
    # Set the position and size of the control window
    control_window.geometry("260x800+100+100")

    # Correction mode string var
    correction_mode = tk.StringVar(value="1-step")
    # If initial values are given, use them, else defaults
    def get_init(key, default, dct):
        return dct[key] if dct and key in dct else default
    thresholds = {
        "circle": tk.DoubleVar(value=get_init("circle", 260, initial_thresholds)),
        "line": tk.DoubleVar(value=get_init("line", 260, initial_thresholds)),
        "a_circle": tk.IntVar(value=get_init("a_circle", image_a_circle.shape[1] // 2, initial_thresholds)),  # pixel position for column
        "a_line": tk.DoubleVar(value=get_init("a_line", 175, initial_thresholds))
    }
    # Alpha transparency values
    alphas = {
        "circle": tk.DoubleVar(value=get_init("circle", 0.5, initial_alphas)),
        "line": tk.DoubleVar(value=get_init("line", 0.5, initial_alphas)),
        "a_circle": tk.DoubleVar(value=get_init("a_circle", 0.5, initial_alphas)),
        "a_line": tk.DoubleVar(value=get_init("a_line", 0.5, initial_alphas))
    }
    # Smoothing window values
    smoothing = {
        "circle": tk.IntVar(value=5),
        "line": tk.IntVar(value=5),
        "a_circle": tk.IntVar(value=5),
        "a_line": tk.IntVar(value=5),
    }
    # Toggles for mean path overlays (kept for compatibility, not user-facing)
    mean_path_toggles = {
        "circle": tk.BooleanVar(value=True),
        "line": tk.BooleanVar(value=True),
        "a_circle": tk.BooleanVar(value=False),
        "a_line": tk.BooleanVar(value=False)
    }

    # Helper to compute mean path for a binary mask (circle)
    #def compute_mean_path_simple(mask):
    #    y = np.arange(mask.shape[0])
    #    x_mean = []
    #    last = mask.shape[1] // 2
    #    for i in range(mask.shape[0]):
    #        idxs = np.where(mask[i, :] == 1)[0]
    #        if len(idxs) > 0:
    #            m = np.mean(idxs)
    #            last = m
    #        else:
    #            m = last
    #        x_mean.append(m)
    #    return np.array(x_mean), y

    #def compute_mean_path2(mask):
    #    y = np.arange(mask.shape[0])
    #    x_contour = []
    #    last = mask.shape[1] // 2
    #    for i in range(mask.shape[0]):
    #        idxs = np.where(mask[i, :] == 1)[0]
    #        if len(idxs) > 0:
    #            m = np.max(idxs)
    #            last = m
    #        else:
    #            m = last
    #        x_contour.append(m)
    #    return np.array(x_contour), y

    # --- Create figure and canvas at the top ---
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    canvas_window = tk.Toplevel(root)
    canvas_window.title("Threshold Preview")
    # Set the position and size of the canvas window
    canvas_window.geometry("1024x768+380+100")
    canvas = FigureCanvasTkAgg(fig, master=canvas_window)
    canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # --- Helper for smoothing arrays for GUI preview overlays ---
    def smooth_array(arr, window):
        if window <= 1:
            return np.array(arr)
        kernel = np.ones(window) / window
        return np.convolve(arr, kernel, mode='same')

    # --- External holder for xmean_circle for result use ---
    global_xmean_circle = [None]  # mutable holder for external access
    global_xmean_line = [None]    # mutable holder for xmean_line

    # --- Plot preview ---
    def plot_preview():
        # Clear all axes before plotting
        for ax in axes.flat:
            ax.clear()
        # Circle Retardation
        axes[0, 0].imshow(image_circle, cmap='viridis', aspect='auto')
        # Add pixel scale ticks for axes[0, 0]
        axes[0, 0].set_xticks(np.linspace(0, image_circle.shape[1], 5))
        axes[0, 0].set_xticklabels([f"{int(tick)}" for tick in np.linspace(0, image_circle.shape[1], 5)])
        # Set ylim and aspect after imshow for axes[0,0]
        axes[0, 0].set_ylim(0, image_circle.shape[0])
        axes[0, 0].set_aspect('auto')
        # --- Updated logic for 2nd step: use angle data for path if appropriate ---
        if isinstance(title, str) and title.lower().startswith("2nd step"):
            mask_circle = (image_a_circle > thresholds["circle"].get()).astype(int)
            xypath_circle = compute_mean_path(mask_circle, label='circle')
            xmean_circle, y = xypath_circle[:,0], xypath_circle[:,1]
            # Apply smoothing to xmean_circle for overlay
            xmean_circle = smooth_array(xmean_circle, smoothing["circle"].get())
        else:
            mask_circle = (image_circle > thresholds["circle"].get()).astype(int)
            xypath_circle = compute_mean_path(mask_circle, label='circle')
            xmean_circle, y = xypath_circle[:,0], xypath_circle[:,1]
            # Apply smoothing to xmean_circle for overlay
            xmean_circle = smooth_array(xmean_circle, smoothing["circle"].get())
        # Store xmean_circle for external access
        global_xmean_circle[0] = xmean_circle
        if not (title != "2nd Step Threshold Adjustment" and manual_override_enabled["circle"].get()):
            axes[0, 0].imshow(mask_circle, cmap='gray', alpha=alphas["circle"].get(), aspect='auto')
        axes[0, 0].set_ylim(0, image_circle.shape[0])
        axes[0, 0].set_aspect('auto')
        axes[0, 0].set_title("Circle Retardation", fontsize=10)
        # Restore red mean binarization path for circle preview in Step 1 only
        if (not (isinstance(title, str) and title.lower().startswith("2nd step"))) and not (manual_override_enabled["circle"].get()):
            interp_circle = interp1d(np.linspace(0, 1, len(xmean_circle)), xmean_circle)
            y_interp = np.arange(image_circle.shape[0])
            x_interp = interp_circle(np.linspace(0, 1, image_circle.shape[0]))
            axes[0, 0].plot(x_interp, y_interp, color='red', lw=1)
            axes[1, 0].plot(x_interp, y_interp, color='red', lw=1)

        # --- Draw manual override vertical lines for 1-step only ---
        if title != "2nd Step Threshold Adjustment":
            if manual_override_enabled["circle"].get():
                x_manual = manual_override_value["circle"].get()
                x_manual_clamped = np.clip(x_manual, 0, image_circle.shape[1] - 1)
                axes[0, 0].axvline(x_manual_clamped, color='red', linestyle='--', lw=1)
                axes[1, 0].axvline(x_manual_clamped, color='red', linestyle='--', lw=1)
            if manual_override_enabled["line"].get():
                x_manual = manual_override_value["line"].get()
                x_manual_clamped = np.clip(x_manual, 0, image_line.shape[1] - 1)
                axes[0, 1].axvline(x_manual_clamped, color='red', linestyle='--', lw=1)
                axes[1, 1].axvline(x_manual_clamped, color='red', linestyle='--', lw=1)

        # Line Retardation
        axes[0, 1].imshow(image_line, cmap='viridis', aspect='auto')
        # Add pixel scale ticks for axes[0, 1]
        axes[0, 1].set_xticks(np.linspace(0, image_line.shape[1], 5))
        axes[0, 1].set_xticklabels([f"{int(tick)}" for tick in np.linspace(0, image_line.shape[1], 5)])
        axes[0, 1].set_ylim(0, image_line.shape[0])
        axes[0, 1].set_aspect('auto')
        # For second correction, use minimum criterion for line thresholding mask
        try:
            line_thresh = float(thresholds["line"].get())
        except tk.TclError:
            line_thresh = 0
        if isinstance(title, str) and title.lower().startswith("2nd step"):
            # For 2nd step: binarize using angle data for line
            mask_line = (image_a_line < line_thresh).astype(int)
        else:
            mask_line = (image_line > line_thresh).astype(int)
        # Compute mean path for line: last quarter only for 2nd step, else full width
        if isinstance(title, str) and title.lower().startswith("2nd step") and threshold_use_binarization is not None and threshold_use_binarization["line"].get():
            last_quarter_start = image_line.shape[1] * 3 // 4
            mask_line_last_quarter = mask_line[:, last_quarter_start:]
            print("Line mask (angle data) nonzero in last quarter:", np.sum(mask_line_last_quarter))
            try:
                # Use leftmost edge of thresholded region per row (min x where mask==1, or 0 if none)
                xmean_line = np.array([
                    np.min(np.where(mask_line_last_quarter[y] == 1)[0]) if np.any(mask_line_last_quarter[y]) else 0
                    for y in range(mask_line_last_quarter.shape[0])
                ]) + last_quarter_start
                # Clamp xmean_line to last_quarter_start .. image_line.shape[1] - 1
                xmean_line = np.clip(xmean_line, last_quarter_start, image_line.shape[1] - 1)
                # Apply smoothing to xmean_line for overlay
                xmean_line = smooth_array(xmean_line, smoothing["line"].get())
            except Exception as e:
                print("Could not compute xmean_line:", e)
                xmean_line = np.full(mask_line.shape[0], image_line.shape[1] - 1)
            print("xmean_line range:", np.min(xmean_line), "to", np.max(xmean_line))
            global_xmean_line[0] = xmean_line
        elif not (isinstance(title, str) and title.lower().startswith("2nd step")):
            xypath_line = compute_mean_path(mask_line, label='line')
            xmean_line, _ = xypath_line[:,0], xypath_line[:,1]
            # Apply smoothing to xmean_line for overlay
            xmean_line = smooth_array(xmean_line, smoothing["line"].get())
            print("xmean_line range:", np.min(xmean_line), "to", np.max(xmean_line))
            global_xmean_line[0] = xmean_line
        else:
            xmean_line = None
            global_xmean_line[0] = xmean_line
        # Overlay mask_line (binarization) -- restrict to last quarter for 2nd step if binarization enabled
        if not (title != "2nd Step Threshold Adjustment" and manual_override_enabled["line"].get()):
            if title.lower().startswith("2nd step") and threshold_use_binarization is not None and threshold_use_binarization["line"].get():
                last_quarter_start = image_line.shape[1] * 3 // 4
                masked_overlay = np.zeros_like(mask_line)
                masked_overlay[:, last_quarter_start:] = mask_line[:, last_quarter_start:]
                axes[0, 1].imshow(masked_overlay, cmap='gray', alpha=alphas["line"].get(), aspect='auto')
            else:
                axes[0, 1].imshow(mask_line, cmap='gray', alpha=alphas["line"].get(), aspect='auto')
        axes[0, 1].set_ylim(0, image_line.shape[0])
        axes[0, 1].set_aspect('auto')
        axes[0, 1].set_title("Line Retardation", fontsize=10)
        # Only plot mean path for line (blue/orange) on axes[0,1] if thresholding is enabled for line
        if ((isinstance(title, str) and title.lower().startswith("2nd step") and threshold_use_binarization is not None and threshold_use_binarization["line"].get() and xmean_line is not None) or (not (isinstance(title, str) and title.lower().startswith("2nd step")) and xmean_line is not None)) and not (title != "2nd Step Threshold Adjustment" and manual_override_enabled["line"].get()):
            interp_line = interp1d(np.linspace(0, 1, len(xmean_line)), xmean_line)
            y_interp_line = np.arange(image_line.shape[0])
            x_interp_line = interp_line(np.linspace(0, 1, image_line.shape[0]))
            line_color = 'orange' if title.lower().startswith("2nd step") else 'red'
            axes[0, 1].plot(x_interp_line, y_interp_line, color=line_color, lw=1)

        # Circle Angle: use a vertical line at the selected column
        axes[1, 0].imshow(image_a_circle, cmap='viridis', aspect='auto')
        axes[1, 0].set_ylim(0, image_a_circle.shape[0])
        axes[1, 0].set_aspect('auto')
        col = thresholds["a_circle"].get()
        # (Red mean binarization path for circle angle preview is handled above for Step 1)
        axes[1, 0].set_title("Circle Angle", fontsize=10)

        # Line Angle
        axes[1, 1].imshow(image_a_line, cmap='viridis', aspect='auto')
        axes[1, 1].set_ylim(0, image_a_line.shape[0])
        axes[1, 1].set_aspect('auto')
        # Only plot mean path for line (blue/orange) on axes[1,1] if thresholding is enabled for line
        if ((isinstance(title, str) and title.lower().startswith("2nd step") and threshold_use_binarization is not None and threshold_use_binarization["line"].get() and xmean_line is not None) or (not (isinstance(title, str) and title.lower().startswith("2nd step")) and xmean_line is not None)) and not (title != "2nd Step Threshold Adjustment" and manual_override_enabled["line"].get()):
            interp_line = interp1d(np.linspace(0, 1, len(xmean_line)), xmean_line)
            y_interp_line = np.arange(image_line.shape[0])
            x_interp_line = interp_line(np.linspace(0, 1, image_line.shape[0]))
            line_color = 'orange' if title.lower().startswith("2nd step") else 'blue'
            axes[1, 1].plot(x_interp_line, y_interp_line, color=line_color, lw=1)
        # Use the same mask_line as above for overlay in 2nd step, else default
        if isinstance(title, str) and title.lower().startswith("2nd step"):
            try:
                line_thresh = float(thresholds["line"].get())
            except tk.TclError:
                line_thresh = 0
            mask_a_line = (image_a_line < line_thresh).astype(int)
        else:
            mask_a_line = (image_a_line < thresholds["a_line"].get()).astype(int)
        # For 2nd step, overlay mask_line (from angle data) in the last quarter if binarization enabled
        if title.lower().startswith("2nd step") and threshold_use_binarization is not None and threshold_use_binarization["line"].get():
            last_quarter_start = image_line.shape[1] * 3 // 4
            masked_overlay = np.zeros_like(mask_a_line)
            masked_overlay[:, last_quarter_start:] = mask_a_line[:, last_quarter_start:]
            axes[1, 1].imshow(masked_overlay, cmap='gray', alpha=alphas["a_line"].get(), aspect='auto')
        else:
            axes[1, 1].imshow(mask_a_line, cmap='gray', alpha=alphas["a_line"].get(), aspect='auto')
        axes[1, 1].set_ylim(0, image_a_line.shape[0])
        axes[1, 1].set_aspect('auto')
        axes[1, 1].set_title("Line Angle", fontsize=10)

        # --- Draw vertical orange line at cutoff position for 2nd Step Threshold Adjustment ---
        if title.lower().startswith("2nd step"):
            # Defensive: get cutoff_column safely
            try:
                cutoff_val = cutoff_column.get()
            except tk.TclError:
                cutoff_val = image_circle.shape[1] // 2  # fallback if empty
            # Circle x_path2 display: only shown when not using threshold
            if not threshold_use_binarization["circle"].get():
                axes[0, 0].axvline(cutoff_val, color='orange', linestyle='--', lw=1)
                axes[1, 0].axvline(cutoff_val, color='orange', linestyle='--', lw=1)

            # Line x_path2 display: if manual override active, show override value; else fallback to cutoff
            if not threshold_use_binarization["line"].get():
                if manual_override_2_enabled["line"].get():
                    x_override = manual_override_2_value["line"].get()
                    axes[0, 1].axvline(x_override, color='orange', linestyle='--', lw=1)
                    axes[1, 1].axvline(x_override, color='orange', linestyle='--', lw=1)
                else:
                    axes[0, 1].axvline(cutoff_val, color='orange', linestyle='--', lw=1)
                    axes[1, 1].axvline(cutoff_val, color='orange', linestyle='--', lw=1)
        # --- Highlight thresholding region with transparent orange rectangle for 2nd Step, line binarization ---
        if title.lower().startswith("2nd step") and threshold_use_binarization is not None and threshold_use_binarization["line"].get():
            col_start = image_line.shape[1] * 3 // 4
            width = image_line.shape[1] - col_start
            for ax in [axes[0, 1], axes[1, 1]]:
                ax.axvspan(col_start, col_start + width, color='orange', alpha=0.1)

        for ax in axes.flat:
            ax.axis('off')
        # fig.tight_layout()  # Removed as constrained_layout is used
        return fig

    def update_plot(*args):
        plot_preview()
        canvas.draw()

    frame = ttk.Frame(control_window)
    frame.config(width=220)
    frame.pack_propagate(False)
    frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    # --- Add custom controls for 2nd Step Threshold Adjustment ---
    if title == "2nd Step Threshold Adjustment":
        cutoff_column = tk.IntVar(value=int(image_circle.shape[1] * 2 // 3))
        cutoff_column.trace("w", lambda *args: update_plot())
        threshold_use_binarization = {
            "circle": tk.BooleanVar(value=False),
            "line": tk.BooleanVar(value=True)
        }
        # Manual override controls for 2nd step x_path2 (line only)
        manual_override_2_enabled = {
            "line": tk.BooleanVar(value=False)
        }
        manual_override_2_value = {
            "line": tk.IntVar(value=image_line.shape[1] * 2 // 3)
        }
        # After initialization, load previous threshold settings if available
        if output_dir is not None:
            config_path = os.path.join(output_dir, f"last_thresholds_{title.replace(' ', '_').lower()}.json")
            if os.path.exists(config_path):
                with open(config_path) as f:
                    saved = json.load(f)
                    if "cutoff_column" in saved and cutoff_column is not None:
                        cutoff_column.set(saved["cutoff_column"])
                    if "use_binarization" in saved and threshold_use_binarization is not None:
                        ub = saved["use_binarization"]
                        threshold_use_binarization["circle"].set(ub.get("circle", False))
                        threshold_use_binarization["line"].set(ub.get("line", True))
        # GUI
        cutoff_frame = ttk.LabelFrame(frame, text="Second Correction Options")
        cutoff_frame.pack(fill="x", pady=10)
        ttk.Label(cutoff_frame, text="Cutoff Column (last third)").pack(anchor="w")
        cutoff_buttons = ttk.Frame(cutoff_frame)
        cutoff_buttons.pack(fill="x", pady=2)
        ttk.Button(cutoff_buttons, text="←", width=2, command=lambda: cutoff_column.set(cutoff_column.get() - 1)).pack(side="left", padx=1)
        ttk.Button(cutoff_buttons, text="→", width=2, command=lambda: cutoff_column.set(cutoff_column.get() + 1)).pack(side="left", padx=1)
        # --- Add manual entry field for cutoff_column ---
        ttk.Entry(cutoff_buttons, width=5, textvariable=cutoff_column).pack(side="left", padx=8)
        ttk.Checkbutton(cutoff_frame, text="Use thresholding for circle",
                        variable=threshold_use_binarization["circle"]).pack(anchor="w")
        ttk.Checkbutton(cutoff_frame, text="Use thresholding for line",
                        variable=threshold_use_binarization["line"]).pack(anchor="w")
        # --- Manual override controls for 2nd step x_path2 (line only) ---
        override_frame = ttk.LabelFrame(frame, text="Manual x_path2 Override")
        override_frame.pack(fill="x", pady=5)
        for key in ["line"]:
            row = ttk.Frame(override_frame)
            row.pack(fill="x", pady=1)
            ttk.Checkbutton(row, text=f"Override x_path2 {key}", variable=manual_override_2_enabled[key], command=update_plot).pack(side="left")
            ttk.Entry(row, width=6, textvariable=manual_override_2_value[key]).pack(side="left", padx=(5, 2))

    # Only render correction mode and threshold toggles if not in "2nd Step Threshold Adjustment"
    if title != "2nd Step Threshold Adjustment":
        # Correction mode radio group
        ttk.Label(frame, text="Correction Mode:").pack(anchor="w")
        for mode in ["No correction", "1-step", "2-step"]:
            ttk.Radiobutton(frame, text=mode, variable=correction_mode, value=mode.lower()).pack(anchor="w")

        # Toggle for showing/hiding threshold frame
        show_thresholds = tk.BooleanVar(value=correction_mode.get() != "no correction")
        def update_threshold_visibility(*args):
            show = correction_mode.get() != "no correction"
            show_thresholds.set(show)
            if show:
                threshold_frame.pack(fill="x", pady=5)
            else:
                threshold_frame.pack_forget()
            update_plot()
        correction_mode.trace("w", update_threshold_visibility)

        # Checkbutton to show/hide threshold controls
        checkbtn = ttk.Checkbutton(frame, text="Show Threshold/Alpha Controls", variable=show_thresholds,
                                   command=lambda: threshold_frame.pack(fill="x", pady=5) if show_thresholds.get() else threshold_frame.pack_forget())
        checkbtn.pack(anchor="w", pady=(5,0))
    else:
        # For 2nd Step, always show threshold frame
        show_thresholds = tk.BooleanVar(value=True)
        def update_threshold_visibility(*args):
            threshold_frame.pack(fill="x", pady=5)
            update_plot()
        # No correction_mode trace needed

    # Threshold and alpha controls inside a frame
    threshold_frame = ttk.LabelFrame(frame, text="Thresholds (use arrows) and Overlay Alpha")
    # Always pack the threshold_frame immediately after creation to ensure it is rendered in both main and 2nd step modes
    threshold_frame.pack(fill="x", pady=5)

    # --- Manual override controls for 1-step correction (circle and line) ---
    if title != "2nd Step Threshold Adjustment":
        manual_override_enabled = {
            "circle": tk.BooleanVar(value=False),
            "line": tk.BooleanVar(value=False)
        }
        manual_override_value = {
            "circle": tk.IntVar(value=image_circle.shape[1] // 2),
            "line": tk.IntVar(value=image_line.shape[1] // 2)
        }
        # --- Helper to rescale manual override values on focus out ---
        def bind_rescale_on_focus_out(entry_widget, var, image_shape, widget):
            def on_focus_out(event):
                try:
                    display_value = int(var.get())
                    display_width = widget.winfo_width() or image_shape[1]
                    if display_width > image_shape[1] and display_value < display_width:
                        scaled_value = int(display_value / display_width * image_shape[1])
                        var.set(max(0, min(scaled_value, image_shape[1] - 1)))
                except Exception:
                    pass
            entry_widget.bind("<FocusOut>", on_focus_out)

    # Helper for left/right button controls
    def make_lr_buttons(parent, var, step=1, minval=0, maxval=360, update=None, is_int=False):
        def left():
            v = var.get()
            v = int(v) if is_int else float(v)
            if v-step >= minval:
                var.set(v-step)
                if update: update()
        def right():
            v = var.get()
            v = int(v) if is_int else float(v)
            if v+step <= maxval:
                var.set(v+step)
                if update: update()
        b1 = ttk.Button(parent, text="←", width=2, command=left)
        b2 = ttk.Button(parent, text="→", width=2, command=right)
        b1.pack(side="left", padx=1)
        b2.pack(side="left", padx=1)
    # Controls for each threshold + alpha (+ smoothing)
    for key in ["circle", "line"]:
        row = ttk.Frame(threshold_frame)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=f"{key.replace('_', ' ').title()}").pack(side="left")
        # For a_circle, use column position, others are numeric thresholds
        if key == "a_circle":
            # Use pixel position slider, but with left/right buttons
            make_lr_buttons(row, thresholds[key], step=1, minval=0, maxval=image_a_circle.shape[1]-1, update=update_plot, is_int=True)
            col_entry = ttk.Entry(row, width=5, textvariable=thresholds[key])
            col_entry.pack(side="left", padx=2)
            ttk.Label(row, text="(Col)").pack(side="left", padx=2)
        else:
            make_lr_buttons(row, thresholds[key], step=1, minval=0, maxval=360, update=update_plot, is_int=False)
            entry = ttk.Entry(row, width=5, textvariable=thresholds[key])
            entry.pack(side="left", padx=2)
            ttk.Label(row, text="Thr").pack(side="left", padx=2)
        ttk.Label(row, text="Alpha").pack(side="left", padx=(8,0))
        alpha_entry = ttk.Entry(row, width=5, textvariable=alphas[key])
        alpha_entry.pack(side="left", padx=2)
        # Clamp alpha between 0 and 1
        def clamp_alpha(var=alphas[key]):
            try:
                v = float(var.get())
                if v < 0: var.set(0)
                if v > 1: var.set(1)
            except Exception: pass
            update_plot()
        alphas[key].trace("w", lambda *a, v=alphas[key]: clamp_alpha(v))
        # Place smoothing controls in a new row below threshold group
        smooth_row = ttk.Frame(threshold_frame)
        smooth_row.pack(fill="x", pady=1)
        ttk.Label(smooth_row, text=f"Smoothing {key.title()}", width=14).pack(side="left", padx=(8, 2))
        ttk.Entry(smooth_row, width=5, textvariable=smoothing[key]).pack(side="left", padx=2)
        # --- Manual override controls for 1-step correction (circle/line) ---
        if title != "2nd Step Threshold Adjustment":
            override_row = ttk.Frame(threshold_frame)
            override_row.pack(fill="x", pady=1)
            ttk.Checkbutton(override_row, text=f"Manual {key} override", variable=manual_override_enabled[key], command=update_plot).pack(side="left")
            entry_widget = ttk.Entry(override_row, width=5, textvariable=manual_override_value[key])
            entry_widget.pack(side="left", padx=(5, 2))
            # Bind rescale on focus out
            bind_rescale_on_focus_out(
                entry_widget,
                manual_override_value[key],
                image_circle.shape if key == "circle" else image_line.shape,
                canvas.get_tk_widget()
            )

    button_frame = ttk.Frame(frame)
    button_frame.pack(fill="x", pady=10)

    def on_confirm():
        # Save the current figure as a PNG before quitting
        fig = plot_preview()
        if output_dir is not None:
            fig.savefig(os.path.join(output_dir, f"{title.replace(' ', '_')}_threshold_preview.png"))
            # Save current threshold and alpha settings for next time
            config_path = os.path.join(output_dir, f"last_thresholds_{title.replace(' ', '_').lower()}.json")
            config = {
                "thresholds": {k: v.get() for k, v in thresholds.items()},
                "alphas": {k: v.get() for k, v in alphas.items()}
            }
            if title == "2nd Step Threshold Adjustment":
                config["cutoff_column"] = cutoff_column.get()
                config["use_binarization"] = {
                    "circle": threshold_use_binarization["circle"].get(),
                    "line": threshold_use_binarization["line"].get()
                }
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
        root.quit()
        root.destroy()

    ttk.Button(button_frame, text="Apply", command=on_confirm).pack()

    # Initial plot already created above; no need to recreate fig/canvas here

    # Initial visibility
    update_threshold_visibility()

    # Trace all threshold and alpha variables to update plot
    for v in list(thresholds.values()) + list(alphas.values()):
        v.trace("w", lambda *a: update_plot())
    root.mainloop()

    results = {
        "mode": correction_mode.get(),
        "thresholds": {k: v.get() for k, v in thresholds.items()},
        "alphas": {k: v.get() for k, v in alphas.items()},
        "mean_path": {k: v.get() for k, v in mean_path_toggles.items()},
        "smoothing": {k: v.get() for k, v in smoothing.items()}
    }
    # Add x_path arrays for use in correction functions
    results["x_path"] = {
        "circle": global_xmean_circle[0],
        "a_circle": global_xmean_circle[0],
        "line": global_xmean_line[0],
        "a_line": global_xmean_line[0]
    }
    # Manual override results for 1-step
    if title != "2nd Step Threshold Adjustment":
        results["manual_override"] = {k: v.get() for k, v in manual_override_enabled.items()}
        results["manual_value"] = {k: v.get() for k, v in manual_override_value.items()}
    # Manual override results for 2nd step (x_path2)
    if title == "2nd Step Threshold Adjustment":
        results["manual_override_2"] = {k: v.get() for k, v in manual_override_2_enabled.items()}
        results["manual_value_2"] = {k: v.get() for k, v in manual_override_2_value.items()}
    # Add cutoff_column and use_binarization if present
    if title == "2nd Step Threshold Adjustment" and cutoff_column is not None and threshold_use_binarization is not None:
        results["cutoff_column"] = cutoff_column.get()
        results["use_binarization"] = {
            "circle": threshold_use_binarization["circle"].get(),
            "line": threshold_use_binarization["line"].get()
        }
    return results
def find_csv_a_file(folder, pattern="CSV"):
    for f in sorted(os.listdir(folder)):
        if ('axis' in f) and f.endswith(".csv"):
            return os.path.join(folder, f)
    return None

def adjust_data(input_dir, sample_name, gap):
    # Load the extracted space-time diagrams
    extracted_circle_path = os.path.join(input_dir, 'extracted_space_time_circle.csv')
    extracted_line_path = os.path.join(input_dir, 'extracted_space_time_line.csv')

    base_dir = os.path.dirname(input_dir)  # Sample directory
    Base_name = sample_name

    frequency = 15  # Hz
    Gap = gap

    if not os.path.exists(extracted_circle_path) or not os.path.exists(extracted_line_path):
        print(f"Error: Extracted space-time diagrams not found in {input_dir}")
        return

    # Load previous correction parameters if available
    correction_json = os.path.join(input_dir, "_correction_state.json")
    if os.path.exists(correction_json):
        with open(correction_json) as f:
            previous_correction = json.load(f)
        print(f"Loaded previous correction parameters from {correction_json}")
    else:
        previous_correction = None

    extracted_space_time_circle_path = os.path.join(input_dir, 'extracted_space_time_circle.csv')
    extracted_space_time_line_path = os.path.join(input_dir, 'extracted_space_time_line.csv')
    a_extracted_space_time_circle_path = os.path.join(input_dir, 'a_extracted_space_time_circle.csv')
    a_extracted_space_time_line_path = os.path.join(input_dir, 'a_extracted_space_time_line.csv')

    intervals_file_path = os.path.join(input_dir, '_actual_intervals.txt')

    try:
        with open(intervals_file_path, 'r') as file:
            counter = int(float(file.read().strip() or "0"))  # Convert to float first, then to int
        print(f"Counter value from '_actual_intervals': {counter}")
    except FileNotFoundError:
        counter = 1
        print(f"Warning: missing '{intervals_file_path}'. Defaulting interval counter to {counter}.")
    except ValueError as exc:
        counter = 1
        print(f"Warning: could not parse interval counter from '{intervals_file_path}' ({exc}); defaulting to {counter}.")

    # Load the CSV files
    extracted_space_time_circle = np.genfromtxt(extracted_space_time_circle_path, delimiter=',', skip_header=1)
    extracted_space_time_line = np.genfromtxt(extracted_space_time_line_path, delimiter=',', skip_header=1)
    a_extracted_space_time_circle = np.genfromtxt(a_extracted_space_time_circle_path, delimiter=',', skip_header=1)
    a_extracted_space_time_line = np.genfromtxt(a_extracted_space_time_line_path, delimiter=',', skip_header=1)

    # Transpose the data
    extracted_space_time_circle = extracted_space_time_circle.T
    extracted_space_time_line = extracted_space_time_line.T
    a_extracted_space_time_circle = a_extracted_space_time_circle.T
    a_extracted_space_time_line = a_extracted_space_time_line.T

    # --- Data shape check after loading and transpose ---
    print("Data shape check after loading and transpose:")
    print("extracted_space_time_circle:", extracted_space_time_circle.shape)
    print("extracted_space_time_line:", extracted_space_time_line.shape)
    print("a_extracted_space_time_circle:", a_extracted_space_time_circle.shape)
    print("a_extracted_space_time_line:", a_extracted_space_time_line.shape)

    # --- LAUNCH GUI FOR THRESHOLD ADJUSTMENT ---
    gui_results = launch_threshold_gui(
        extracted_space_time_circle, extracted_space_time_line,
        a_extracted_space_time_circle, a_extracted_space_time_line,
        title="Threshold Adjustment",
        initial_thresholds=previous_correction.get("thresholds_step1") if previous_correction else None,
        initial_alphas=previous_correction.get("alphas_step1") if previous_correction else None,
        output_dir=input_dir
    )
    correction_mode = gui_results["mode"]
    thresholds_1 = gui_results["thresholds"]
    alphas_1 = gui_results["alphas"]
    # --- Smoothing fallback and logic for both steps ---
    smoothing_1 = gui_results.get("smoothing", {"circle": 1, "line": 1, "a_circle": 1, "a_line": 1})
    # If 2-step, launch a second threshold window for the new values
    if correction_mode == "2-step":
        gui_results2 = launch_threshold_gui(
            extracted_space_time_circle, extracted_space_time_line,
            a_extracted_space_time_circle, a_extracted_space_time_line,
            title="2nd Step Threshold Adjustment",
            initial_thresholds=previous_correction.get("thresholds_step2") if previous_correction else thresholds_1,
            initial_alphas=previous_correction.get("alphas_step2") if previous_correction else alphas_1,
            output_dir=input_dir
        )

    if correction_mode == "no correction":
        # CASE 1: No correction — use raw data directly
        corrected_circle = extracted_space_time_circle.copy()
        corrected_line = extracted_space_time_line.copy()
        corrected_a_circle = a_extracted_space_time_circle.copy()
        corrected_a_line = a_extracted_space_time_line.copy()

    elif correction_mode == "1-step":
        # Support manual override for circle and line
        if gui_results.get("manual_override", {}).get("circle", False):
            x_val_circle = min(gui_results["manual_value"]["circle"], extracted_space_time_circle.shape[1] - 1)
            x_path1_circle = np.full(extracted_space_time_circle.shape[0], x_val_circle)
        else:
            x_path1_circle = gui_results["x_path"]["circle"]

        if gui_results.get("manual_override", {}).get("line", False):
            x_val_line = min(gui_results["manual_value"]["line"], extracted_space_time_line.shape[1] - 1)
            x_path1_line = np.full(extracted_space_time_line.shape[0], x_val_line)
        else:
            x_path1_line = gui_results["x_path"]["line"]

        corrected_circle, corrected_a_circle = apply_one_step_circle(
            extracted_space_time_circle,
            a_extracted_space_time_circle,
            x_path1_circle
        )
        corrected_line, corrected_a_line = apply_one_step_line(
            extracted_space_time_line,
            a_extracted_space_time_line,
            x_path1_line
        )

    elif correction_mode == "2-step":
        # Prepare x_path1 for circle
        if gui_results.get("manual_override", {}).get("circle", False):
            x_val_circle = min(gui_results["manual_value"]["circle"], extracted_space_time_circle.shape[1] - 1)
            x_path1_circle = np.full(extracted_space_time_circle.shape[0], x_val_circle)
        else:
            x_path1_circle = gui_results["x_path"]["circle"]
        # Prepare x_path1 for line
        if gui_results.get("manual_override", {}).get("line", False):
            x_val_line = min(gui_results["manual_value"]["line"], extracted_space_time_line.shape[1] - 1)
            x_path1_line = np.full(extracted_space_time_line.shape[0], x_val_line)
        else:
            x_path1_line = gui_results["x_path"]["line"]
        # For circle, manual override for x_path2 is not available; always use cutoff_column
        x_path2_val = gui_results2["cutoff_column"]
        print(f"x_path2_circle value: {x_path2_val}")
        corrected_circle, corrected_a_circle = apply_two_step_circle(
            extracted_space_time_circle,
            a_extracted_space_time_circle,
            x_path1_circle,
            np.full(a_extracted_space_time_circle.shape[0], x_path2_val)
        )
        # Manual override for x_path2 (line)
        if gui_results2.get("manual_override_2", {}).get("line", False):
            x_val_line2 = min(gui_results2["manual_value_2"]["line"], a_extracted_space_time_line.shape[1] - 1)
            x_path2_line = np.full(a_extracted_space_time_line.shape[0], x_val_line2)
        else:
            x_path2_line = gui_results2["x_path"]["a_line"]
        corrected_line, corrected_a_line = apply_two_step_line(
            extracted_space_time_line,
            a_extracted_space_time_line,
            x_path1_line,
            x_path2_line
        )
    # Regardless of correction, proceed to interval reduction and stats

    num_intervals = counter
    down_to = 5
    scaled_down_length = down_to * num_intervals

    interim_circle_data = np.zeros((corrected_circle.shape[0], scaled_down_length))
    interim_a_circle_data = np.zeros((corrected_a_circle.shape[0], scaled_down_length))
    interim_line_data = np.zeros((corrected_line.shape[0], scaled_down_length))
    interim_a_line_data = np.zeros((corrected_a_line.shape[0], scaled_down_length))

    # Define interval widths before using them in the loop
    interval_width_circle = max(1, corrected_circle.shape[1] // num_intervals)
    interval_width_line = max(1, corrected_line.shape[1] // num_intervals)

    counter = 0
    interval_width_circle = max(1, corrected_circle.shape[1] // num_intervals-1)
    for i in range(0, corrected_circle.shape[1] - interval_width_circle, interval_width_circle):
        for j in range(0, interval_width_circle, 29):
            k = i + j
            # Bounds check for circle arrays
            if k + 30 <= corrected_circle.shape[1]:
                interim_circle_data[:, counter * down_to + j // 30] = np.mean(corrected_circle[:, k:k + 30], axis=1)
                interim_a_circle_data[:, counter * down_to + j // 30] = np.mean(corrected_a_circle[:, k:k + 30], axis=1)
            # Bounds check for line arrays
            if k + 30 <= corrected_line.shape[1]:
                interim_line_data[:, counter * down_to + j // 30] = np.mean(corrected_line[:, k:k + 30], axis=1)
                interim_a_line_data[:, counter * down_to + j // 30] = np.mean(corrected_a_line[:, k:k + 30], axis=1)
        counter = counter + 1

    print('To double-check the reduced dimensions:', interim_circle_data.shape)

    # %% Compute mean and SEM values based on reduced data

    reduced_circle_intervals = [interim_circle_data[:, i*down_to:(i+1)*down_to] for i in range(num_intervals)]
    reduced_line_intervals = [interim_line_data[:, i*down_to:(i+1)*down_to] for i in range(num_intervals)]
    reduced_a_circle_intervals = [interim_a_circle_data[:, i*down_to:(i+1)*down_to] for i in range(num_intervals)]
    reduced_a_line_intervals = [interim_a_line_data[:, i*down_to:(i+1)*down_to] for i in range(num_intervals)]

    mean_values_circle = [np.mean(interval) / (Gap * 10**6) for interval in reduced_circle_intervals]
    sem_values_circle = [np.std(interval) / np.sqrt(interval.size) / (Gap * 10**6) for interval in reduced_circle_intervals]
    a_mean_values_circle = [np.mean(interval) for interval in reduced_a_circle_intervals]

    mean_values_line = [np.mean(interval) / (Gap * 10**6) for interval in reduced_line_intervals]
    sem_values_line = [np.std(interval) / np.sqrt(interval.size) / (Gap * 10**6) for interval in reduced_line_intervals]
    a_mean_values_line = [np.mean(interval) for interval in reduced_a_line_intervals]

    print("Updated mean/SEM values computed from reduced data.")

    interval_width_circle = interval_width_circle // 30 + 1
    interval_width_line = interval_width_line // 30 + 1


    # %% CONSTRUCT VECTORFIELD

    # Define separate steps for the x and y axes
    step_x_circle = max(1, int(a_extracted_space_time_circle.shape[1] / counter)) if counter > 0 else 1
    step_y_circle = max(1, 10)
    step_x_line = max(1, int(a_extracted_space_time_line.shape[1] / counter)) if counter > 0 else 1
    step_y_line = max(1, 1)

    # Define arrow sizes for the vector fields
    arrow_size_circle = 80  # Adjust this value as needed
    arrow_size_line = 80  # Adjust this value as needed

    # Function to create symmetric lines for quiver plot
    def symmetric_quiver(ax, X, Y, U, V, **kwargs):
        ax.quiver(X, Y, U, V, **kwargs)
        ax.quiver(X, Y, -U, -V, **kwargs)

    # For angle data, subtract row-based offset for visualization (as in original code)
    for i in range(1, a_extracted_space_time_circle.shape[1]):
        for j in range(1, a_extracted_space_time_circle.shape[0]):
            a_extracted_space_time_circle[j, i] = a_extracted_space_time_circle[j, i] - j * (360 / a_extracted_space_time_circle.shape[0])

    for i in range(1, a_extracted_space_time_line.shape[1]):
        for j in range(1, a_extracted_space_time_line.shape[0]):
            a_extracted_space_time_line[j, i] = a_extracted_space_time_line[j, i] - j * (360 / a_extracted_space_time_line.shape[0])

    # --- 2x2 Plot Layout ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Top-left: Circle Retardation (corrected)
    axes[0, 0].imshow(corrected_circle, aspect='auto', cmap='viridis')
    axes[0, 0].set_title('Corrected Circle Retardation')

    # Bottom-left: Line Retardation (corrected)
    axes[1, 0].imshow(corrected_line, aspect='auto', cmap='viridis')
    axes[1, 0].set_title('Corrected Line Retardation')

    # Top-right: Circle Angle with Vector Field
    im_a_circle = axes[0, 1].imshow(corrected_a_circle, aspect='auto', cmap='viridis')
    axes[0, 1].set_title('Corrected Circle Angle with Vector Field')

    # Bottom-right: Line Angle with Vector Field
    im_a_line = axes[1, 1].imshow(corrected_a_line, aspect='auto', cmap='viridis')
    axes[1, 1].set_title('Corrected Line Angle with Vector Field')

    # Prepare vector fields for angle plots
    # Circle
    Y_circle, X_circle = np.mgrid[0:corrected_a_circle.shape[0], 0:corrected_a_circle.shape[1]]
    corrected_a_circle_rad = np.deg2rad(corrected_a_circle)
    U_circle = np.cos(corrected_a_circle_rad)
    V_circle = np.sin(corrected_a_circle_rad)
    symmetric_quiver(
        axes[0, 1],
        X_circle[::step_y_circle, ::step_x_circle],
        Y_circle[::step_y_circle, ::step_x_circle],
        U_circle[::step_y_circle, ::step_x_circle],
        V_circle[::step_y_circle, ::step_x_circle],
        color='w', scale=arrow_size_circle, headlength=0, headwidth=0
    )

    # Line
    Y_line, X_line = np.mgrid[0:corrected_a_line.shape[0], 0:corrected_a_line.shape[1]]
    corrected_a_line_rad = np.deg2rad(corrected_a_line)
    U_line = np.cos(corrected_a_line_rad)
    V_line = np.sin(corrected_a_line_rad)
    symmetric_quiver(
        axes[1, 1],
        X_line[::step_y_line, ::step_x_line],
        Y_line[::step_y_line, ::step_x_line],
        U_line[::step_y_line, ::step_x_line],
        V_line[::step_y_line, ::step_x_line],
        color='w', scale=arrow_size_line, headlength=0, headwidth=0
    )

    # Add colorbars for angle plots (right column)
    fig.colorbar(im_a_circle, ax=axes[0, 1], orientation='vertical')
    fig.colorbar(im_a_line, ax=axes[1, 1], orientation='vertical')

    plt.tight_layout()
    plt.show()  # Removed display for saving-only figure

    # Plot the mean values and SEMs as scatter plots with error bars
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot mean values and SEM for circle
    axes[0].errorbar(range(num_intervals), mean_values_circle, yerr=sem_values_circle, fmt='o', mfc='none', mec='b', ecolor='b', capsize=5)
    axes[0].set_title('Mean Values and SEM for Extracted Space-Time Circle')
    axes[0].set_xlabel('Interval Number')
    axes[0].set_ylabel('Mean Value')
    axes[0].grid(True)

    # Plot mean values and SEM for line
    axes[1].errorbar(range(num_intervals), mean_values_line, yerr=sem_values_line, fmt='o', mfc='none', mec='r', ecolor='r', capsize=5)
    axes[1].set_title('Mean Values and SEM for Extracted Space-Time Line')
    axes[1].set_xlabel('Interval Number')
    axes[1].set_ylabel('Mean Value')
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()  # Removed display for saving-only figure

    # Create the '_Final results' folder if it does not exist
    final_results_dir = os.path.join(base_dir)
    os.makedirs(final_results_dir, exist_ok=True)

    # Create the sample-local _Temp_processed folder for PI outputs
    sample_results_dir = get_temp_processed_folder(input_dir)
    print(f"_Temp_processed folder for PI outputs: {sample_results_dir}")

    print(extracted_space_time_circle.shape)

    # Create row and column indices for the flattened data
    rows, cols = np.indices(extracted_space_time_circle.shape)

    # Flatten the indices
    rows_flat = rows.flatten()
    cols_flat = cols.flatten()

    # Save the data to CSV files with row and column indices
    # Determine the number of columns to add to reach 29 intervals
    num_intervals_max = 29
    #interval_width_circle_full = (extracted_space_time_circle.shape[1] + num_intervals_max) // num_intervals_max
    #interval_width_line_full = (extracted_space_time_line.shape[1] + num_intervals_max) // num_intervals_max


    #print(interval_width_circle)
    #print(interim_circle_data.shape[1])

    # Create new arrays with NaNs to fill the remaining intervals
    new_circle_data = np.full((interim_circle_data.shape[0], interval_width_circle * num_intervals_max), np.nan)
    new_line_data = np.full((interim_line_data.shape[0], interval_width_line * num_intervals_max), np.nan)
    new_a_circle_data = np.full((interim_a_circle_data.shape[0], interval_width_circle * num_intervals_max), np.nan)
    new_a_line_data = np.full((interim_a_line_data.shape[0], interval_width_line * num_intervals_max), np.nan)

    print(new_circle_data.shape[1])

    # Copy the original data into the new arrays
    new_circle_data[:, :interim_circle_data.shape[1]] = interim_circle_data
    new_line_data[:, :interim_line_data.shape[1]] = interim_line_data
    new_a_circle_data[:, :interim_a_circle_data.shape[1]] = interim_a_circle_data
    new_a_line_data[:, :interim_a_line_data.shape[1]] = interim_a_line_data

    # Flatten the new data with indices
    rows, cols = np.indices(new_circle_data.shape)
    rows_flat = rows.flatten()[:new_circle_data.size]
    cols_flat = cols.flatten()[:new_circle_data.size]

    #test = new_circle_data
    np.savetxt(os.path.join(sample_results_dir, 'test.csv'), new_circle_data, delimiter=',')

    circle_data_with_indices = np.column_stack((rows_flat, cols_flat, new_circle_data.flatten()))
    line_data_with_indices = np.column_stack((rows_flat[:new_line_data.size], cols_flat[:new_line_data.size], new_line_data.flatten()))
    a_circle_data_with_indices = np.column_stack((rows_flat[:new_a_circle_data.size], cols_flat[:new_a_circle_data.size], new_a_circle_data.flatten()))
    a_line_data_with_indices = np.column_stack((rows_flat[:new_a_line_data.size], cols_flat[:new_a_line_data.size], new_a_line_data.flatten()))

    a_circle_data_with_indices = np.column_stack((rows_flat[:new_a_circle_data.size], cols_flat[:new_a_circle_data.size], new_a_circle_data.flatten()))
    a_line_data_with_indices = np.column_stack((rows_flat[:new_a_line_data.size], cols_flat[:new_a_line_data.size], new_a_line_data.flatten()))
 
   

    # Save the data to CSV files with row and column indices
    mean_sem_circle = np.column_stack((mean_values_circle, sem_values_circle, a_mean_values_circle))
    header = "mean_circle,SEM_circle,angle_mean_circle"
    np.savetxt(os.path.join(sample_results_dir, 'PI_1_mean_sem_circle.csv'), mean_sem_circle, delimiter=',', header=header, comments='')
    a_circle_data_with_indices = np.column_stack((rows_flat[:new_a_circle_data.size], cols_flat[:new_a_circle_data.size], new_a_circle_data.flatten()))
    header = "x_c,y_c,flat_circle"
    np.savetxt(os.path.join(sample_results_dir, 'PI_2_extracted_space_time_circle_flattened.csv'), circle_data_with_indices, delimiter=',', header=header, comments='')
    header = "x,y,flat_circle_angle"
    np.savetxt(os.path.join(sample_results_dir, 'PI_3_a_extracted_space_time_circle_flattened.csv'), a_circle_data_with_indices, delimiter=',', header=header, comments='')

    mean_sem_line = np.column_stack((mean_values_line, sem_values_line, a_mean_values_line))
    header = "mean_line,SEM_line,angle_mean_line"
    np.savetxt(os.path.join(sample_results_dir, '4_mean_sem_line.csv'), mean_sem_line, delimiter=',', header=header, comments='')
    a_line_data_with_indices = np.column_stack((rows_flat[:new_a_line_data.size], cols_flat[:new_a_line_data.size], new_a_line_data.flatten()))
    header = "x_l,y_l,flat_line"
    np.savetxt(os.path.join(sample_results_dir, 'PI_5_extracted_space_time_line_flattened.csv'), line_data_with_indices, delimiter=',', header=header, comments='')
    header = "x,y,flat_line_angle"
    np.savetxt(os.path.join(sample_results_dir, 'PI_6_a_extracted_space_time_line_flattened.csv'), a_line_data_with_indices, delimiter=',', header=header, comments='')

    # Save the vector field data to CSV files
    # vector_field_data = []
    # for i in range(0, a_extracted_space_time_line.shape[0], step_y_line):
    #     for j in range(0, a_extracted_space_time_line.shape[1], step_x_line):
    #         x_center = j
    #         y_center = i
    #         u = U[i, j]
    #         v = V[i, j]
    #         x_left = x_center - u
    #         y_left = y_center - v
    #         x_right = x_center + u
    #         y_right = y_center + v
    #         vector_field_data.append([x_center, y_center, x_left, y_left, x_right, y_right])

    # vector_field_data = np.array(vector_field_data)
    # np.savetxt(os.path.join(sample_results_dir, 'vector_field_a_extracted_space_time_line.csv'), vector_field_data, delimiter=',', header='x_center,y_center,x_left,y_left,x_right,y_right', comments='')

    # # # Load the data
    # extracted_space_time_circle = np.loadtxt(extracted_space_time_circle_path, delimiter=',')
    # extracted_space_time_line = np.loadtxt(extracted_space_time_line_path, delimiter=',')
    # a_extracted_space_time_circle = np.loadtxt(a_extracted_space_time_circle_path, delimiter=',')
    # a_extracted_space_time_line = np.loadtxt(a_extracted_space_time_line_path, delimiter=',')

    # # Ensure the indices are within bounds
    # for i in range(a_extracted_space_time_circle.shape[0]):
    #     for j in range(a_extracted_space_time_circle.shape[1]):
    #         if i < a_extracted_space_time_circle.shape[0] and j < a_extracted_space_time_circle.shape[1]:
    #             a_extracted_space_time_circle[i, j] = (a_extracted_space_time_circle[i, j] + 90) % 180 + 1

    # # Save the adjusted data
    # np.savetxt(a_extracted_space_time_circle_path, a_extracted_space_time_circle, delimiter=',')
    # np.savetxt(a_extracted_space_time_line_path, a_extracted_space_time_line, delimiter=',')

    # Save full correction parameters for reproducibility
    correction_state = {
        "mode": correction_mode,
        "thresholds_step1": thresholds_1,
        "alphas_step1": alphas_1,
        "smoothing_step1": smoothing_1,
        "manual_override": gui_results.get("manual_override"),
        "manual_value": gui_results.get("manual_value")
    }

    if correction_mode == "2-step":
        correction_state.update({
            "thresholds_step2": gui_results2.get("thresholds"),
            "alphas_step2": gui_results2.get("alphas"),
            "cutoff_column": gui_results2.get("cutoff_column"),
            "use_binarization": gui_results2.get("use_binarization"),
            "manual_override_2": gui_results2.get("manual_override_2"),
            "manual_value_2": gui_results2.get("manual_value_2"),
        })

    correction_json = os.path.join(input_dir, "_correction_state.json")
    with open(correction_json, "w") as f:
        json.dump(correction_state, f, indent=2)
    print(f"Correction parameters saved to: {correction_json}")

    # Write output JSON file listing the CSV outputs for this sample
    import json
    files = [
        os.path.join(sample_results_dir, 'PI_1_mean_sem_circle.csv'),
        os.path.join(sample_results_dir, 'PI_2_extracted_space_time_circle_flattened.csv'),
        os.path.join(sample_results_dir, 'PI_3_a_extracted_space_time_circle_flattened.csv'),
        os.path.join(sample_results_dir, '4_mean_sem_line.csv'),
        os.path.join(sample_results_dir, 'PI_5_extracted_space_time_line_flattened.csv'),
        os.path.join(sample_results_dir, 'PI_6_a_extracted_space_time_line_flattened.csv')
    ]
    json_path = os.path.join(sample_results_dir, f"_output_PI_{sample_name}.json")
    with open(json_path, "w") as f:
        json.dump({
            "sample_name": sample_name,
            "processed_folder": sample_results_dir,
            "csv_outputs": files
        }, f, indent=2)
    print(f"PI output JSON written: {json_path}")
    try:
        final_dir = get_unified_processed_folder(input_dir)
        final_dest = os.path.join(final_dir, os.path.basename(json_path))
        shutil.copy2(json_path, final_dest)
        print(f"PI output JSON mirrored to: {final_dest}")
    except Exception as mirror_exc:
        print(f"Warning: failed to mirror PI JSON to unified folder: {mirror_exc}")
    print(f"Data adjusted for sample: {sample_name}")

# def save_output_data_PI(sample_name, files):
#     output_file = "output_PI.json"
#     if os.path.exists(output_file):
#         with open(output_file, 'r') as file:
#             data = json.load(file)
#     else:
#         data = []
#
#     # Check if the sample name already exists
#     sample_exists = False
#     for sample in data:
#         if sample['name'] == sample_name:
#             sample['files'] = files  # Update the existing entry
#             sample_exists = True
#             break
#
#     # If the sample doesn't exist, add a new entry
#     if not sample_exists:
#         data.append({
#             'name': sample_name,
#             'files': files
#         })
#
#     # Write the updated data back to the JSON file
#     with open(output_file, 'w') as file:
#         json.dump(data, file, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python Data_adjuster_v3_1.py <input_dir> <sample_name> <gap>")
        sys.exit(1)
    input_dir = sys.argv[1]
    sample_name = sys.argv[2]
    gap = float(sys.argv[3])
    adjust_data(input_dir, sample_name, gap)
