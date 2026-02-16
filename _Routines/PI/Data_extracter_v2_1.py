import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from scipy.ndimage import zoom
import pickle
from concurrent.futures import ProcessPoolExecutor

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


def find_csv_a_file(folder, pattern="CSV"):
    for f in sorted(os.listdir(folder)):
        if f.startswith(pattern) and f.endswith(".csv"):
            return os.path.join(folder, f)
    return None


def process_csv_file(csv_path, offset_x, offset_y, Inner, Outer):
    import numpy as np
    data = np.loadtxt(csv_path, delimiter=',', dtype=np.float32)
    Image_centre = [data.shape[0] // 2 + offset_y, data.shape[1] // 2 + offset_x]
    circle_values = []
    radius = Inner + 4
    for angle in range(360):
        x = int(Image_centre[0] + radius * np.cos(np.deg2rad(angle)))
        y = int(Image_centre[1] + radius * np.sin(np.deg2rad(angle)))
        if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
            circle_values.append(data[x, y])
    line_data = data[Image_centre[0], Image_centre[1] - Outer:Image_centre[1] - Inner]
    line_data = np.concatenate((line_data[:Outer - Inner], line_data[Outer + Inner:]))
    return np.array(circle_values), np.array(line_data)


def _find_input_dir_with_csvs(input_dir):
    csv_files = sorted([f for f in os.listdir(input_dir) if ('retard' in f) and f.endswith('.csv')])
    if csv_files:
        return input_dir, csv_files
    found = False
    for root, dirs, files in os.walk(input_dir):
        csv_files = sorted([f for f in files if ('retard' in f) and f.endswith('.csv')])
        if csv_files:
            input_dir = root
            found = True
            break
    if not found:
        return None, []
    return input_dir, csv_files


def extract_space_time(
    input_dir,
    gap,
    offset_x,
    offset_y,
    inner_initial,
    outer_initial,
    time_data_path,
    plot=False,
    use_multiprocessing=True,
):
    print("[extract_space_time] start", flush=True)
    print(f"  input_dir={input_dir}\n  gap={gap}\n  offsets=({offset_x},{offset_y})\n  radii=({inner_initial},{outer_initial})", flush=True)
    gap = float(gap)
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    Inner = int(inner_initial)
    Outer = int(outer_initial)

    fixed_interval = 150
    frequency = 15

    save_dir = get_temp_processed_folder(input_dir)

    with open(time_data_path, 'r') as file:
        triggers = [float(line.strip()) for line in file]
    print(f"  triggers count={len(triggers)} from {time_data_path}", flush=True)

    input_dir, csv_files = _find_input_dir_with_csvs(input_dir)
    if not input_dir or not csv_files:
        print("No suitable CSV files found in directory tree. Exiting...", flush=True)
        return None
    print(f"  csv root={input_dir} files={len(csv_files)}", flush=True)

    # Load some data to check the position of the circles
    try:
        csv_path = os.path.join(input_dir, csv_files[1])
        data = pd.read_csv(csv_path, header=None).values
        height, width = data.shape
        c_x = width // 2 + offset_x
        c_y = height // 2 + offset_y
    except Exception as e:
        print(f"Error loading CSV file: {e}", flush=True)
        return None

    # Define the time vector
    time_vector = np.arange(len(csv_files)) * (1/frequency)

    if use_multiprocessing:
        with ProcessPoolExecutor() as executor:
            futures = []
            for i in range(len(triggers)):
                start_time = triggers[i]
                end_time = triggers[i] + 10
                start_index = np.searchsorted(time_vector, start_time)
                end_index = np.searchsorted(time_vector, end_time)
                for index in range(start_index, end_index):
                    if index >= len(csv_files):
                        continue
                    csv_path = os.path.join(input_dir, csv_files[index])
                    futures.append(executor.submit(process_csv_file, csv_path, offset_x, offset_y, Inner, Outer))
            results = [f.result() for f in futures]
    else:
        results = []
        total_batches = 0
        for i in range(len(triggers)):
            start_time = triggers[i]
            end_time = triggers[i] + 10
            start_index = np.searchsorted(time_vector, start_time)
            end_index = np.searchsorted(time_vector, end_time)
            batch_count = 0
            for index in range(start_index, end_index):
                if index >= len(csv_files):
                    continue
                csv_path = os.path.join(input_dir, csv_files[index])
                results.append(process_csv_file(csv_path, offset_x, offset_y, Inner, Outer))
                batch_count += 1
                total_batches += 1
                if total_batches % 50 == 0:
                    print(f"  processed {total_batches} CSVs (trigger {i+1}/{len(triggers)})", flush=True)
            print(f"  trigger {i+1}/{len(triggers)} processed {batch_count} frames", flush=True)

    print(f"  total results={len(results)}", flush=True)
    space_time_circle = np.array([r[0] for r in results], dtype=np.float32)
    space_time_line = np.array([r[1] for r in results], dtype=np.float32)

    no_of_pi_intervals = len(space_time_circle) / frequency / 10

    # Save the extracted space-time diagrams to CSV files
    extracted_space_time_circle_path = os.path.join(save_dir, 'extracted_space_time_circle.csv')
    extracted_space_time_line_path = os.path.join(save_dir, 'extracted_space_time_line.csv')
    intervals_path = os.path.join(save_dir, '_actual_intervals.txt')
    np.savetxt(extracted_space_time_circle_path, space_time_circle, delimiter=',')
    np.savetxt(extracted_space_time_line_path, space_time_line, delimiter=',')
    with open(intervals_path, 'w') as f:
        f.write(str(no_of_pi_intervals))

    print(f"  saved outputs to {save_dir}", flush=True)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        im1 = ax1.imshow(space_time_circle.T, aspect='auto', cmap='viridis', vmin=0, vmax=270)
        ax1.set_title('Extracted Space-Time Diagram (Circle)')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Radius')
        fig.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(space_time_line.T, aspect='auto', cmap='viridis', vmin=0, vmax=270)
        ax2.set_title('Extracted Space-Time Diagram (Line)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Position')
        fig.colorbar(im2, ax=ax2)

        interval = fixed_interval
        for i in range(int(no_of_pi_intervals)):
            ax1.axvline(x=int(i * interval), color='white', linestyle='--')
            ax2.axvline(x=int(i * interval), color='white', linestyle='--')

        plt.ioff()
        plt.show(block=True)

    return {
        "circle": space_time_circle,
        "line": space_time_line,
        "interval": fixed_interval,
        "n_intervals": no_of_pi_intervals,
        "input_dir": input_dir,
        "save_dir": save_dir,
    }


def main(input_dir, gap, offset_x, offset_y, inner_initial, outer_initial, time_data_path):
    result = extract_space_time(
        input_dir,
        gap,
        offset_x,
        offset_y,
        inner_initial,
        outer_initial,
        time_data_path,
        plot=True,
    )
    if result is None:
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python Data_extracter_v2_1_axis.py <input_dir> <gap> <offset_x> <offset_y> <inner_initial> <outer_initial> <time_data_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
