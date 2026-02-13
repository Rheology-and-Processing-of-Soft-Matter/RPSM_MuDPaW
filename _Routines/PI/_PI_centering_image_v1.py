def find_csv_folder_with_files(base_path, pattern="CSV"):
    for root, dirs, files in os.walk(base_path):
        # Do not search inside _Temp_processed (results/output folder)
        if "_Temp_processed" in dirs:
            dirs.remove("_Temp_processed")
        if any(('axis' in f or 'retard' in f) and f.endswith('.csv') for f in files):
            return root
    return None

def find_csv_a_file(folder, pattern="CSV"):
    for f in sorted(os.listdir(folder)):
        if ('axis' in f or 'retard' in f) and f.endswith(".csv"):
            return os.path.join(folder, f)
    return None
import matplotlib
try:
    matplotlib.use("MacOSX")
except Exception as e:
    print(f"[PI Centering] MacOSX backend unavailable, using default. Reason: {e}")
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.widgets import Button
import sys

def _load_existing_geometry(output_dir):
    """Return (offset_x, offset_y, Inner_initial, Outer_initial) if saved geometry exists."""
    json_file = os.path.join(output_dir, '_Geometry_positioning.json')
    txt_file = os.path.join(output_dir, '_Geometry_positioning.txt')
    if os.path.isfile(json_file):
        try:
            import json as _json
            with open(json_file, 'r') as jf:
                data = _json.load(jf)
            return (
                int(data.get("offset_x", 0)),
                int(data.get("offset_y", 0)),
                int(data.get("Inner_initial", 86)),
                int(data.get("Outer_initial", 126)),
            )
        except Exception as e:
            print(f"[PI Centering] Failed to read JSON geometry file: {e}")
    if os.path.isfile(txt_file):
        try:
            vals = {}
            with open(txt_file, 'r') as tf:
                for line in tf:
                    if ":" in line:
                        k, v = line.split(":", 1)
                        vals[k.strip()] = v.strip()
            return (
                int(vals.get("offset_x", 0)),
                int(vals.get("offset_y", 0)),
                int(vals.get("Inner_initial", 86)),
                int(vals.get("Outer_initial", 126)),
            )
        except Exception as e:
            print(f"[PI Centering] Failed to read TXT geometry file: {e}")
    return None


def center_geometry(input_dir, output_dir, offset_x=0, offset_y=0, Inner_initial=86, Outer_initial=126):
    """Centers geometry based on CSV data and saves the results."""

    # Ensure output directory exists
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        print(f"Failed to create output directory '{output_dir}': {e}")
        sys.exit(1)

    # Load existing geometry from output folder if present
    loaded = _load_existing_geometry(output_dir)
    if loaded:
        offset_x, offset_y, Inner_initial, Outer_initial = loaded
        print(f"[PI Centering] Loaded existing geometry from {output_dir}: "
              f"offset_x={offset_x}, offset_y={offset_y}, "
              f"Inner_initial={Inner_initial}, Outer_initial={Outer_initial}")

    input_dir = find_csv_folder_with_files(input_dir)
    if not input_dir:
        print("No suitable CSV files found in directory tree. Exiting...")
        sys.exit(1)
    print("Starting center_geometry with resolved input_dir:", input_dir)

    try:
        csv_files = sorted([f for f in os.listdir(input_dir) if ('axis' in f or 'retard' in f) and f.endswith('.csv')])
    except FileNotFoundError:
        print(f"Directory '{input_dir}' not found. Exiting...")
        sys.exit(1)
    if not csv_files:
        print("No suitable CSV files found in directory tree. Exiting...")
        sys.exit(1)
    print('Co of CSV files loaded =', len(csv_files))

    first_file = find_csv_a_file(input_dir)
    if not first_file:
        print("No valid CSV file found. Exiting...")
        sys.exit(1)
    print("Attempting to load first file:", first_file)

    # Load the data

    try:
        data = np.loadtxt(first_file, delimiter=',')
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        sys.exit(1)
    print("Loaded data shape:", data.shape)
    print("Data min/max:", np.nanmin(data), np.nanmax(data))

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    fig.subplots_adjust(bottom=0.25)

    circle_inner = Circle((0, 0), Inner_initial, color='r', fill=False)
    circle_outer = Circle((0, 0), Outer_initial, color='b', fill=False)
    ax.add_patch(circle_inner)
    ax.add_patch(circle_outer)

    height, width = data.shape
    center_x = width // 2 + offset_x
    center_y = height // 2 + offset_y

    ax.imshow(data, cmap='gray')
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)  # keep image orientation

    # Sliders (commented out to disable)
    # axcolor = 'lightgoldenrodyellow'
    # ax_slider_x = plt.axes([0.25, 0.01, 0.65, 0.02], facecolor=axcolor)
    # ax_slider_y = plt.axes([0.25, 0.04, 0.65, 0.02], facecolor=axcolor)
    # ax_slider_inner = plt.axes([0.25, 0.07, 0.65, 0.02], facecolor=axcolor)
    # ax_slider_outer = plt.axes([0.25, 0.10, 0.65, 0.02], facecolor=axcolor)

    # slider_x = Slider(ax_slider_x, 'X Center', 0, width, valinit=center_x, valstep=1)
    # slider_y = Slider(ax_slider_y, 'Y Center', 0, height, valinit=center_y, valstep=1)
    # slider_inner = Slider(ax_slider_inner, 'Inner Radius', 1, min(height, width)//2, valinit=Inner_initial)
    # slider_outer = Slider(ax_slider_outer, 'Outer Radius', 1, min(height, width)//2, valinit=Outer_initial)

    # def update_circles(val=None):
    #     center_x = slider_x.val
    #     center_y = slider_y.val
    #     inner_radius = slider_inner.val
    #     outer_radius = slider_outer.val
    #     circle_inner.center = (center_x, center_y)
    #     circle_outer.center = (center_x, center_y)
    #     circle_inner.set_radius(inner_radius)
    #     circle_outer.set_radius(outer_radius)
    #     fig.canvas.draw_idle()

    # slider_x.on_changed(update_circles)
    # slider_y.on_changed(update_circles)
    # slider_inner.on_changed(update_circles)
    # slider_outer.on_changed(update_circles)

    # Instead, implement button controls
    center = [center_x, center_y]
    inner_radius = Inner_initial
    outer_radius = Outer_initial

    circle_inner.center = (center[0], center[1])
    circle_outer.center = (center[0], center[1])
    circle_inner.set_radius(inner_radius)
    circle_outer.set_radius(outer_radius)

    def update_circles():
        print("Updating circles at center:", center, "with radii:", inner_radius, outer_radius)
        circle_inner.center = (center[0], center[1])
        circle_outer.center = (center[0], center[1])
        circle_inner.set_radius(inner_radius)
        circle_outer.set_radius(outer_radius)
        # Remove old line if it exists
        for line in ax.lines[:]:
            line.remove()
        # Draw horizontal chord at 2/3 of outer_radius below center
        y_offset = (2/3) * outer_radius
        y_line = center[1] + y_offset
        if outer_radius > abs(y_offset):
            half_span = np.sqrt(outer_radius**2 - y_offset**2)
            x_min = center[0] - half_span
            x_max = center[0] + half_span
            ax.plot([x_min, x_max], [y_line, y_line], 'r--', linewidth=1)
        fig.canvas.draw_idle()

    # Button axes
    axcolor = 'lightgoldenrodyellow'

    # Movement buttons
    ax_left = plt.axes([0.2, 0.08, 0.08, 0.05])
    ax_right = plt.axes([0.32, 0.08, 0.08, 0.05])
    ax_up = plt.axes([0.26, 0.14, 0.08, 0.05])
    ax_down = plt.axes([0.26, 0.02, 0.08, 0.05])

    btn_left = Button(ax_left, '←')
    btn_up = Button(ax_up, '↑')
    btn_right = Button(ax_right, '→')
    btn_down = Button(ax_down, '↓')

    # Radius buttons
    ax_inner_plus = plt.axes([0.55, 0.12, 0.1, 0.05])
    ax_inner_minus = plt.axes([0.66, 0.12, 0.1, 0.05])
    ax_outer_plus = plt.axes([0.55, 0.05, 0.1, 0.05])
    ax_outer_minus = plt.axes([0.66, 0.05, 0.1, 0.05])

    btn_inner_plus = Button(ax_inner_plus, 'Inner +')
    btn_inner_minus = Button(ax_inner_minus, 'Inner -')
    btn_outer_plus = Button(ax_outer_plus, 'Outer +')
    btn_outer_minus = Button(ax_outer_minus, 'Outer -')

    # Save button
    ax_save = plt.axes([0.8, 0.02, 0.15, 0.05])
    btn_save = Button(ax_save, 'Save & Close')

    def save_and_exit(event):
        plt.close(fig)

    btn_save.on_clicked(save_and_exit)

    def move_left(event):
        center[0] = max(0, center[0] - 1)
        update_circles()

    def move_right(event):
        center[0] = min(width, center[0] + 1)
        update_circles()

    def move_up(event):
        center[1] = max(0, center[1] - 1)
        update_circles()

    def move_down(event):
        center[1] = min(height, center[1] + 1)
        update_circles()

    def inner_increase(event):
        nonlocal inner_radius
        max_radius = min(height, width) // 2
        if inner_radius < max_radius and inner_radius < outer_radius:
            inner_radius += 1
            update_circles()

    def inner_decrease(event):
        nonlocal inner_radius
        if inner_radius > 1:
            inner_radius -= 1
            update_circles()

    def outer_increase(event):
        nonlocal outer_radius
        max_radius = min(height, width) // 2
        if outer_radius < max_radius:
            outer_radius += 1
            update_circles()

    def outer_decrease(event):
        nonlocal outer_radius
        if outer_radius > inner_radius:
            outer_radius -= 1
            update_circles()

    btn_left.on_clicked(move_left)
    btn_right.on_clicked(move_right)
    btn_up.on_clicked(move_up)
    btn_down.on_clicked(move_down)

    btn_inner_plus.on_clicked(inner_increase)
    btn_inner_minus.on_clicked(inner_decrease)
    btn_outer_plus.on_clicked(outer_increase)
    btn_outer_minus.on_clicked(outer_decrease)

    update_circles()

    plt.show(block=True)

    plt.ioff()

    offset_x = int(center[0] - width // 2)
    offset_y = int(center[1] - height // 2)
    Inner_initial = int(inner_radius)
    Outer_initial = int(outer_radius)

    # Save the final circle positions (TXT for human-readability + JSON for machine-readability)
    output_file = os.path.join(output_dir, '_Geometry_positioning.txt')
    json_file = os.path.join(output_dir, '_Geometry_positioning.json')
    print("Saving geometry to:", output_file)

    try:
        with open(output_file, 'w') as file:
            file.write(f"offset_x: {offset_x}\n")
            file.write(f"offset_y: {offset_y}\n")
            file.write(f"Inner_initial: {Inner_initial}\n")
            file.write(f"Outer_initial: {Outer_initial}\n")
        print(f"Geometry positioning saved to {output_file}")
    except Exception as e:
        print(f"Failed to write TXT geometry file: {e}")

    try:
        import json as _json
        with open(json_file, 'w') as jf:
            _json.dump({
                "offset_x": int(offset_x),
                "offset_y": int(offset_y),
                "Inner_initial": int(Inner_initial),
                "Outer_initial": int(Outer_initial)
            }, jf, indent=2)
        print(f"Geometry positioning JSON saved to {json_file}")
    except Exception as e:
        print(f"Failed to write JSON geometry file: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python _PI_centering_image_v1.py <input_dir> <output_dir> [offset_x] [offset_y] [Inner_initial] [Outer_initial]")
        sys.exit(1)

    print("sys.argv received:", sys.argv)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    # Read optional parameters from command line if provided
    offset_x = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    offset_y = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    Inner_initial = int(sys.argv[5]) if len(sys.argv) > 5 else 86
    Outer_initial = int(sys.argv[6]) if len(sys.argv) > 6 else 126

    center_geometry(input_dir, output_dir, offset_x, offset_y, Inner_initial, Outer_initial)
