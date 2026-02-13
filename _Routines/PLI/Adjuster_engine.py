import numpy as np

__all__ = [
    "apply_one_step_circle",
    "apply_one_step_line",
    "apply_two_step_circle",
    "apply_two_step_line",
    "compute_mean_path",
]

# --- Invariants & path preparation ---

def _verify_dims(tag, data, angle, x1, x2=None):
    rh, rw = data.shape
    ah, aw = angle.shape
    if (rh, rw) != (ah, aw):
        raise ValueError(f"[{tag}] shape mismatch: data={data.shape} vs angle={angle.shape}")
    def _check(name, arr):
        if arr is None:
            raise ValueError(f"[{tag}] {name} is None")
        if len(arr) != rh:
            raise ValueError(f"[{tag}] {name} length {len(arr)} != height {rh}")
        mn = np.nanmin(arr); mx = np.nanmax(arr)
        if mn < 0 or mx > rw - 1:
            raise ValueError(f"[{tag}] {name} out of bounds: min={mn}, max={mx}, width={rw}")
    _check("x_path1", x1)
    if x2 is not None:
        _check("x_path2", x2)


def _prep_path(x, width):
    x = np.asarray(x, dtype=float)
    # Forward‑fill non‑finite values to avoid jumps to 0
    if not np.all(np.isfinite(x)):
        for i in range(len(x)):
            if not np.isfinite(x[i]):
                x[i] = x[i-1] if i > 0 and np.isfinite(x[i-1]) else 0
    # Clip to valid image bounds and cast to int
    x = np.clip(x, 0, max(0, width - 1)).astype(np.int64)
    return x

def apply_one_step_circle(circle_data, angle_data, x_path1):
    """1-step correction for circle retardation/angle."""
    corrected_circle = circle_data.copy()
    corrected_angle = angle_data.copy()
    height, width = circle_data.shape
    _verify_dims("1-step circle", circle_data, angle_data, x_path1)
    x_path1 = _prep_path(x_path1, width)
    for y in range(height):
        x1 = x_path1[y]
        ref1 = corrected_circle[y, x1]
        for x in range(x1, width):
            corrected_circle[y, x] = 2 * ref1 - corrected_circle[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180
    return corrected_circle, corrected_angle

def apply_one_step_line(line_data, angle_data, x_path1):
    """1-step correction for line retardation/angle."""
    corrected_line = line_data.copy()
    corrected_angle = angle_data.copy()
    height, width = line_data.shape
    _verify_dims("1-step line", line_data, angle_data, x_path1)
    x_path1 = _prep_path(x_path1, width)
    for y in range(height):
        x1 = x_path1[y]
        ref1 = corrected_line[y, x1]
        for x in range(x1, width):
            corrected_line[y, x] = 2 * ref1 - corrected_line[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180
    return corrected_line, corrected_angle

def apply_two_step_circle(circle_data, angle_data, x_path1, x_path2):
    """2-step correction for circle."""
    corrected_circle = circle_data.copy()
    corrected_angle = angle_data.copy()
    height, width = circle_data.shape
    _verify_dims("2-step circle", circle_data, angle_data, x_path1, x_path2)
    x_path1 = _prep_path(x_path1, width)
    x_path2 = _prep_path(x_path2, width)
    # Step 1: reflect between x1..x2
    for y in range(height):
        x1 = x_path1[y]
        x2 = x_path2[y]
        ref1 = corrected_circle[y, x1]
        for x in range(x1, min(x2 + 1, width)):
            corrected_circle[y, x] = 2 * ref1 - corrected_circle[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180
    # Step 2: translate x2..end
    for y in range(height):
        x1 = x_path1[y]
        x2 = x_path2[y]
        ref2 = corrected_circle[y, x2]
        for x in range(x2, width):
            corrected_circle[y, x] = ref2 + corrected_circle[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180
    return corrected_circle, corrected_angle

def apply_two_step_line(line_data, angle_data, x_path1, x_path2):
    """2-step correction for line."""
    corrected_line = line_data.copy()
    corrected_angle = angle_data.copy()
    height, width = line_data.shape
    _verify_dims("2-step line", line_data, angle_data, x_path1, x_path2)
    x_path1 = _prep_path(x_path1, width)
    x_path2 = _prep_path(x_path2, width)
    # Step 1: reflect
    for y in range(height):
        x1 = x_path1[y]
        x2 = x_path2[y]
        ref1 = corrected_line[y, x1]
        for x in range(x1, min(x2 + 1, width)):
            corrected_line[y, x] = 2 * ref1 - corrected_line[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180
    # Step 2: translate
    for y in range(height):
        x1 = x_path1[y]
        x2 = x_path2[y]
        ref2 = corrected_line[y, x2]
        for x in range(x2, width):
            corrected_line[y, x] = ref2 + corrected_line[y, x]
            corrected_angle[y, x] = (corrected_angle[y, x] + 90) % 180
    return corrected_line, corrected_angle

def compute_mean_path(binary_image, label='circle'):
    """Compute mean/rightmost x-position per row from a binary mask.
    Returns array (H,2): (x,y). For x only, take [:,0]."""
    height, width = binary_image.shape
    path = []
    if 'line' in label.lower():
        last_valid_x = 0
        use_limit = '2-step' in label.lower()
        width_limit = int(width * 0.7) if use_limit else width
        for y in range(height):
            x_positions = np.where(binary_image[y, :width_limit] == 1)[0]
            if len(x_positions) > 0:
                x_val = np.max(x_positions)
                last_valid_x = x_val
            else:
                x_val = last_valid_x
            path.append((x_val, y))
        return np.array(path)
    else:
        last_valid_x = width // 2
        for y in range(height):
            x_positions = np.where(binary_image[y, :] == 1)[0]
            if len(x_positions) > 0:
                x_val = np.mean(x_positions)
                last_valid_x = x_val
            else:
                x_val = last_valid_x
            path.append((x_val, y))
        x_vals = np.array([p[0] for p in path], dtype=float)
        y_vals = np.array([p[1] for p in path])
        window_size = 11  # odd length for centered moving average
        if window_size > len(x_vals):
            window_size = max(1, (len(x_vals) // 2) * 2 + 1)
        if window_size <= 1:
            smoothed = x_vals
        else:
            kernel = np.ones(window_size, dtype=float) / window_size
            # pad at edges to avoid edge shrinkage
            pad = window_size // 2
            x_pad = np.pad(x_vals, (pad, pad), mode='edge')
            smoothed = np.convolve(x_pad, kernel, mode='valid')
        return np.column_stack((smoothed, y_vals))