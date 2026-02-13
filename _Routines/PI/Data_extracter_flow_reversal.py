import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
plt.switch_backend("TkAgg")
from matplotlib.patches import Circle
from scipy.ndimage import zoom
import pickle


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

def main(input_dir, gap, offset_x, offset_y, inner_initial, outer_initial):
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

    # Load all .csv files that contain '_retardation' and exclude 'line_nominal_sr', then sort them alphabetically
    # Exclude any line_nominal_sr files
    print("Loading CSV files...")
    try:
        csv_files = sorted([f for f in os.listdir(input_dir) if ('_retardation' in f) and f.endswith('.csv') and 'line_nominal_sr' not in f])
    except FileNotFoundError:
        print(f"Directory '{input_dir}' not found. Searching subfolders for valid CSV sets...")
        csv_files = []
    if not csv_files:
        # Search subfolders for at least one file with '_retardation' and one with '_axis'
        found = False
        for root, dirs, files in os.walk(input_dir):
            retard_files = [f for f in files if '_retardation' in f and f.endswith('.csv')]
            axis_files = [f for f in files if '_axis' in f and f.endswith('.csv')]
            if retard_files and axis_files:
                input_dir = root
                csv_files = sorted([f for f in retard_files if 'line_nominal_sr' not in f])
                print(f"Using valid CSV subfolder: {input_dir}")
                found = True
                break
        if not found:
            print("No suitable CSV files found in directory tree. Exiting...")
            sys.exit(1)
    print(f"Processing all {len(csv_files)} retardation files")
    print('Co of CSV files loaded =', len(csv_files))
    if not csv_files:
        print("No CSV files found in the specified directory. Searching subdirectories...")
        found = False
        for root, dirs, files in os.walk(input_dir):
            csv_files = sorted([f for f in files if ('_retardation' in f) and f.endswith('.csv') and 'line_nominal_sr' not in f])
            if csv_files:
                input_dir = root
                csv_files = sorted([f for f in os.listdir(input_dir) if ('_retardation' in f) and f.endswith('.csv') and 'line_nominal_sr' not in f])
                print(f"Found suitable CSV files in: {input_dir}")
                found = True
                break
        if not found:
            print("No suitable CSV files found in directory tree. Exiting...")
            sys.exit(1)

    # Always use the first filtered CSV file for overlay visualization
    first_csv_path = os.path.join(input_dir, csv_files[0])
    print(f"[PREVIEW] Overlay is based on file: {first_csv_path}")
    data = pd.read_csv(first_csv_path, header=None).values

    # Create a color image from grayscale
    img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_colored = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Compute center
    center_x = img.shape[1] // 2 + offset_x
    center_y = img.shape[0] // 2 + offset_y

    # Draw inner and outer circles
    cv2.circle(img_colored, (center_x, center_y), Inner, (0, 255, 0), 2)
    cv2.circle(img_colored, (center_x, center_y), Outer, (0, 0, 255), 2)
    # Draw the nominal shear rate line (blue) at +2/3 Outer radius (chord line exactly at circle boundary)
    offset = int((2/3) * Outer)
    x_extent = int(np.sqrt(Outer**2 - offset**2))
    x_start = center_x - x_extent
    x_end = center_x + x_extent
    cv2.line(img_colored, (x_start, center_y + offset), (x_end, center_y + offset), (255, 0, 0), 2)

    # Show image using matplotlib
    plt.figure(figsize=(6, 6))
    plt.imshow(img_colored[..., ::-1])  # Convert BGR to RGB
    plt.title('Overlay of Inner (green), Outer (red) Circles and Nominal SR Line (blue, chord at +2/3 Outer)')
    plt.axis('off')
    plt.show()

    space_time_circle = []
    space_time_line = []
    space_time_line_nominal_sr = []

    for i, filename in enumerate(csv_files):
        if len(csv_files) > 1:
            progress = int((i + 1) / len(csv_files) * 100)
            if (i + 1) % max(1, len(csv_files) // 100) == 0 or i == len(csv_files) - 1:
                print(f"Processing CSV files: {progress}% complete")

        csv_path = os.path.join(input_dir, filename)
        print(f"Processing retardation file: {csv_path}")
        data = pd.read_csv(csv_path, header=None).values

        Image_centre = [data.shape[0] // 2 + offset_y, data.shape[1] // 2 + offset_x]

        # Extract the values along the circle with radius Inner + 4 without geometric correction
        circle_values = []
        radius = Inner + 4
        for angle in range(360):
            x = int(Image_centre[0] + radius * np.cos(np.deg2rad(angle)))
            y = int(Image_centre[1] + radius * np.sin(np.deg2rad(angle)))
            if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
                circle_values.append(data[x, y])
        space_time_circle.append(circle_values)

        # Extract the horizontal line excluding Inner radius
        line_data = data[Image_centre[0], Image_centre[1] - Outer:Image_centre[1] - Inner]
        line_data = np.concatenate((line_data[:Outer - Inner], line_data[Outer + Inner:]))
        space_time_line.append(line_data)

        # Extract the nominal shear rate line (full chord across Outer radius) at +2/3 Outer below center without correction
        offset = int((2/3) * Outer)
        row = Image_centre[0] + offset
        line_nominal_sr_data = []
        for col in range(Image_centre[1] - Outer, Image_centre[1] + Outer):
            dx = col - Image_centre[1]
            dy = offset
            if dx**2 + dy**2 <= Outer**2:  # keep only inside the outer circle
                line_nominal_sr_data.append(data[row, col])
        space_time_line_nominal_sr.append(line_nominal_sr_data)

    space_time_circle = np.array(space_time_circle)
    space_time_line = np.array(space_time_line)
    space_time_line_nominal_sr = np.array(space_time_line_nominal_sr)

    print(f"Extracted {space_time_circle.shape[0]} frames of circle data and {space_time_line.shape[0]} frames of line data.")

    frequency = 15
    fixed_interval = 150  # Fixed interval for processing

    # Define the save directory inside the sample-local _Temp_processed folder
    save_dir = get_temp_processed_folder(input_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Files will be saved to: {save_dir}")

    # Load the additional file and convert timestamps to seconds
    # Removed trigger loading as per instructions

    # Load all .csv files that contain 'axis'
    # Exclude any line_nominal_sr files
    space_time_circle_angle = []
    space_time_line_angle = []
    space_time_line_nominal_sr_angle = []

    csv_files_angle = sorted([f for f in os.listdir(input_dir) if ('axis' in f) and f.endswith('.csv') and 'line_nominal_sr' not in f])
    print(f"Processing all {len(csv_files_angle)} orientation files")
    for i, filename in enumerate(csv_files_angle):
        if len(csv_files_angle) > 1:
            progress = int((i + 1) / len(csv_files_angle) * 100)
            if (i + 1) % max(1, len(csv_files_angle) // 100) == 0 or i == len(csv_files_angle) - 1:
                print(f"Processing orientation CSV files: {progress}% complete")

        csv_path = os.path.join(input_dir, filename)
        print(f"Processing orientation file: {csv_path}")
        data = pd.read_csv(csv_path, header=None).values

        Image_centre = [data.shape[0] // 2 + offset_y, data.shape[1] // 2 + offset_x]

        circle_values = []
        radius = Inner + 4
        for angle in range(360):
            x = int(Image_centre[0] + radius * np.cos(np.deg2rad(angle)))
            y = int(Image_centre[1] + radius * np.sin(np.deg2rad(angle)))
            if 0 <= x < data.shape[0] and 0 <= y < data.shape[1]:
                theta = np.arctan2(y - Image_centre[1], x - Image_centre[0])
                phi_flow = np.rad2deg(theta) + 90.0
                corrected_value = (data[x, y] - phi_flow + 90.0) % 180.0 - 90.0
                circle_values.append(corrected_value)
        space_time_circle_angle.append(circle_values)

        line_data = data[Image_centre[0], Image_centre[1] - Outer:Image_centre[1] - Inner]
        line_data = np.concatenate((line_data[:Outer - Inner], line_data[Outer + Inner:]))
        space_time_line_angle.append(line_data)

        offset = int((2/3) * Outer)
        row = Image_centre[0] + offset
        line_nominal_sr_angle_data = []
        for col in range(Image_centre[1] - Outer, Image_centre[1] + Outer):
            dx = col - Image_centre[1]
            dy = offset
            if dx**2 + dy**2 <= Outer**2:
                theta = np.arctan2(dx, dy)
                phi_flow = np.rad2deg(theta) + 90.0
                corrected_value = (data[row, col] - phi_flow + 90.0) % 180.0 - 90.0
                line_nominal_sr_angle_data.append(corrected_value)
        space_time_line_nominal_sr_angle.append(line_nominal_sr_angle_data)

    space_time_circle_angle = np.array(space_time_circle_angle)
    space_time_line_angle = np.array(space_time_line_angle)
    space_time_line_nominal_sr_angle = np.array(space_time_line_nominal_sr_angle)

    # Plot the six space-time diagrams with updated subplot arrangement (2 rows, 3 columns)
    fig, axs = plt.subplots(2, 3, figsize=(18, 10))

    im0 = axs[0, 0].imshow(space_time_line.T, aspect='auto', cmap='viridis')
    axs[0, 0].set_title("Retardation Line")
    axs[0, 0].set_xlabel("Frame")
    axs[0, 0].set_ylabel("Angle or position")
    fig.colorbar(im0, ax=axs[0, 0])

    im1 = axs[0, 1].imshow(space_time_circle.T, aspect='auto', cmap='viridis')
    axs[0, 1].set_title("Retardation Circle")
    axs[0, 1].set_xlabel("Frame")
    axs[0, 1].set_ylabel("Angle or position")
    fig.colorbar(im1, ax=axs[0, 1])

    im4 = axs[0, 2].imshow(space_time_line_nominal_sr.T, aspect='auto', cmap='viridis')
    axs[0, 2].set_title("Retardation Line Nominal Shear Rate")
    axs[0, 2].set_xlabel("Frame")
    axs[0, 2].set_ylabel("Position")
    fig.colorbar(im4, ax=axs[0, 2])

    im3 = axs[1, 0].imshow(space_time_line_angle.T, aspect='auto', cmap='viridis')
    axs[1, 0].set_title("Orientation Line")
    axs[1, 0].set_xlabel("Frame")
    axs[1, 0].set_ylabel("Angle or position")
    fig.colorbar(im3, ax=axs[1, 0])

    im2 = axs[1, 1].imshow(space_time_circle_angle.T, aspect='auto', cmap='viridis')
    axs[1, 1].set_title("Orientation Circle")
    axs[1, 1].set_xlabel("Frame")
    axs[1, 1].set_ylabel("Angle or position")
    fig.colorbar(im2, ax=axs[1, 1])

    im5 = axs[1, 2].imshow(space_time_line_nominal_sr_angle.T, aspect='auto', cmap='viridis')
    axs[1, 2].set_title("Orientation Line Nominal Shear Rate")
    axs[1, 2].set_xlabel("Frame")
    axs[1, 2].set_ylabel("Position")
    fig.colorbar(im5, ax=axs[1, 2])

    plt.tight_layout()
    print(">>> Finished processing, displaying plots now...")
    plt.show()

    np.savetxt(os.path.join(save_dir, 'retardation_circle.csv'), space_time_circle, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'retardation_line.csv'), space_time_line, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'retardation_line_nominal_sr.csv'), space_time_line_nominal_sr, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'orientation_circle.csv'), space_time_circle_angle, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'orientation_line.csv'), space_time_line_angle, delimiter=",")
    np.savetxt(os.path.join(save_dir, 'orientation_line_nominal_sr.csv'), space_time_line_nominal_sr_angle, delimiter=",")

if __name__ == "__main__":
    if len(sys.argv) != 7:
        print("Usage: python Data_extracter_flow_reversal.py <input_dir> <gap> <offset_x> <offset_y> <inner_initial> <outer_initial>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6])
