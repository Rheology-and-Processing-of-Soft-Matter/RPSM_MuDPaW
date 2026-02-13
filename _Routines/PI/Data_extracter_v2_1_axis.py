import sys
import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import json
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

def main(input_dir, gap, offset_x, offset_y, inner_initial, outer_initial, time_data_path):
    gap = float(gap)
    offset_x = int(offset_x)
    offset_y = int(offset_y)
    Inner = int(inner_initial)
    Outer = int(outer_initial)
    print(f"gap: {gap}")
    print(f"offset_x: {offset_x}")
    print(f"offset_y: {offset_y}")
    print(f"Inner: {Inner}")
    print(f"Outer: {Outer}")

    frequency = 15
    fixed_interval = 150  # Fixed interval for processing

    # Use sample-local _Temp_processed folder for saving results
    save_dir = get_temp_processed_folder(input_dir)
    print(f"_Temp_processed folder for PI axis outputs: {save_dir}")

    # Load the additional file and convert timestamps to seconds
    print("Loading triggers...")
    with open(time_data_path, 'r') as file:
        triggers = [float(line.strip()) for line in file]

    print("Loaded triggers:", triggers)

    # Load all .csv files that contain 'axis' or 'retard' and sort them alphabetically
    print("Loading CSV files...")
    try:
        csv_files = sorted([f for f in os.listdir(input_dir) if ('axis' in f) and f.endswith('.csv')])
    except FileNotFoundError:
        print(f"Directory '{input_dir}' not found. Searching parent directory for subfolders...")
        parent_dir = os.path.dirname(input_dir)
        found = False
        for root, dirs, files in os.walk(parent_dir):
            csv_files = sorted([f for f in files if ('axis' in f) and f.endswith('.csv')])
            if csv_files:
                input_dir = root
                print(f"Found suitable CSV files in: {input_dir}")
                found = True
                break
        if not found:
            print("No suitable CSV files found in directory tree. Exiting...")
            sys.exit(1)
    print('Co of CSV files loaded =', len(csv_files))
    if not csv_files:
        print("No CSV files found in the specified directory. Searching subdirectories...")
        found = False
        for root, dirs, files in os.walk(input_dir):
            csv_files = sorted([f for f in files if ('axis' in f) and f.endswith('.csv')])
            if csv_files:
                input_dir = root
                print(f"Found suitable CSV files in: {input_dir}")
                found = True
                break
        if not found:
            print("No suitable CSV files found in directory tree. Exiting...")
            sys.exit(1)

    # Define the time vector
    time_vector = np.arange(len(csv_files)) * (1/frequency)
    print("Time vector:", time_vector)

    # Parallel processing of CSV files
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
    space_time_circle = np.array([r[0] for r in results], dtype=np.float32)
    space_time_line = np.array([r[1] for r in results], dtype=np.float32)

    no_of_pi_intervals = len(space_time_circle) / frequency / 10   

    # Print the size of the space-time diagrams
    print(f"Size of space_time_circle: {space_time_circle.shape[1]} x {space_time_circle.shape[0]}")
    print(f"Size of space_time_line: {space_time_line.shape[1]} x {space_time_line.shape[0]}")

    # Plot the final space-time diagrams
    print("Plotting final space-time diagrams...")
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

    # Define the interval for superimposing vertical lines
    interval = fixed_interval
    print('Interval...', interval)

    # Superimpose interval bounds as white vertical lines
    for i in range(int(no_of_pi_intervals)):
        ax1.axvline(x=int(i*interval), color='white', linestyle='--')
        ax2.axvline(x=int(i*interval), color='white', linestyle='--')

    plt.ioff()
    plt.show()

    print('Writing data to files...')

    # Save the extracted space-time diagrams to CSV files
    extracted_space_time_circle_path = os.path.join(save_dir, 'a_extracted_space_time_circle.csv')
    extracted_space_time_line_path = os.path.join(save_dir, 'a_extracted_space_time_line.csv')
    
    np.savetxt(extracted_space_time_circle_path, space_time_circle, delimiter=',')
    np.savetxt(extracted_space_time_line_path, space_time_line, delimiter=',')

    # Remove the intermediate results file after processing all files
    intermediate_file = os.path.join(save_dir, 'a_intermediate_results.pkl')
    if os.path.exists(intermediate_file):
        os.remove(intermediate_file)

    print(f"Extracted space-time diagrams saved to: {extracted_space_time_circle_path} and {extracted_space_time_line_path}")

    json_path = os.path.join(save_dir, "_output_PI_axis.json")
    with open(json_path, "w") as f:
        json.dump({
            "input_dir": input_dir,
            "processed_folder": save_dir,
            "outputs": {
                "circle": extracted_space_time_circle_path,
                "line": extracted_space_time_line_path
            }
        }, f, indent=2)
    print(f"PI axis output JSON written: {json_path}")

if __name__ == "__main__":
    if len(sys.argv) != 8:
        print("Usage: python Data_extracter_v2_1_axis.py <input_dir> <gap> <offset_x> <offset_y> <inner_initial> <outer_initial> <time_data_path>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7])
