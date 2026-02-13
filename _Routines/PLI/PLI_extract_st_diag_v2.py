import os
import sys
import shutil
import gc
import psutil
import numpy as np
from PIL import Image
# --- Parallelism imports ---
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED

# --- Interactive Extraction GUI Dependencies ---
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk

import cv2
import json
import threading

# --- Unified _Processed Folder Helpers ---
def _sanitize_misjoined_user_path(p: str) -> str:
    """If `p` contains a second absolute '/Users/...' segment (e.g.,
    '/repo/.../RPSM_MuDPaW_v5/Users/kroland/...'), trim to the second one.
    If nothing to fix, return `p` unchanged."""
    try:
        tok = os.sep + "Users" + os.sep
        first = p.find(tok)
        if first == -1:
            return p
        second = p.find(tok, first + 1)
        if second == -1:
            return p
        fixed = p[second:]
        if not fixed.startswith(os.sep):
            fixed = os.sep + fixed
        if fixed != p:
            print(f"[PLI] sanitize(helper): misjoined path →\n  in : {p}\n  out: {fixed}")
        return fixed
    except Exception:
        return p

def _split_parts(path):
    """Return normalized path parts without empty tokens."""
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts

def get_reference_folder_from_path(path):
    """Resolve the *reference* folder robustly.

    We walk up the path and look for known modality markers ("PLI", "PI", "SAXS", "Rheology").
    The *reference* folder is defined as the parent directory of the modality folder.

    Examples expected to resolve to the same reference folder:
        <reference>/PLI/<sample>
        <reference>/PLI/<sample>/Frames
        <reference>/Rheology/<sample>
        <reference>/SAXS/<sample>/something/deeper

    Fallback: if no marker is found, go two levels up from the provided path (legacy behavior).
    """
    markers = {"PLI", "PI", "SAXS", "Rheology"}

    # Work with absolute path to avoid surprises
    abspath = _sanitize_misjoined_user_path(os.path.abspath(path))
    parts = _split_parts(abspath)

    # Walk from the end towards the root and find the nearest modality folder
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            # Parent of the modality folder is the reference
            reference = os.sep.join(parts[:i])
            if reference == "":
                # Degenerate, but keep safe fallback
                break
            reference = _sanitize_misjoined_user_path(reference)
            reference = os.path.abspath(reference)
            if not reference.startswith(os.sep):
                reference = os.sep + reference
            return reference if reference else os.sep

    # Fallback for legacy layouts
    abspath = _sanitize_misjoined_user_path(abspath)
    abspath = os.path.abspath(abspath)
    if not abspath.startswith(os.sep):
        abspath = os.sep + abspath
    return abspath


def get_unified_processed_folder(path):
    """Return the unified `_Processed/PLI` folder located inside the resolved *reference* folder.

    This avoids accidentally placing `_Processed` one level above the reference when the video
    path is deeper (e.g., `<reference>/PLI/<sample>/Frames`) and keeps PLI outputs scoped.
    """
    reference_folder = get_reference_folder_from_path(path)
    reference_folder = _sanitize_misjoined_user_path(reference_folder)
    reference_folder = os.path.abspath(reference_folder)
    if not reference_folder.startswith(os.sep):
        reference_folder = os.sep + reference_folder
    processed_root = os.path.join(reference_folder, "_Processed", "PLI")
    processed_root = _sanitize_misjoined_user_path(processed_root)
    processed_root = os.path.abspath(processed_root)
    if not processed_root.startswith(os.sep):
        processed_root = os.sep + processed_root
    os.makedirs(processed_root, exist_ok=True)
    return processed_root

# --- Global/module-level config for circular extraction ---
_last_n_angles = 360


def _sample_line_from_coords(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinearly sample `arr` along xs/ys coordinates. Returns NaN for OOB samples."""
    h, w = arr.shape
    n = len(xs)
    vals = np.empty(n, dtype=np.float32)
    for i in range(n):
        x = float(xs[i])
        y = float(ys[i])
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        if x0 < 0 or y0 < 0 or x1 >= w or y1 >= h:
            vals[i] = np.nan
            continue
        dx = x - x0
        dy = y - y0
        v00 = arr[y0, x0]
        v10 = arr[y0, x1]
        v01 = arr[y1, x0]
        v11 = arr[y1, x1]
        vals[i] = (
            (1 - dx) * (1 - dy) * v00 +
            dx * (1 - dy) * v10 +
            (1 - dx) * dy * v01 +
            dx * dy * v11
        )
    return vals


def _format_rotation_suffix(angle_deg: float) -> str:
    """Return a filename-safe suffix describing the rotation angle."""
    if abs(angle_deg) < 1e-6:
        return ""
    mag = abs(angle_deg)
    if abs(mag - round(mag)) < 1e-6:
        angle_txt = str(int(round(mag)))
    else:
        angle_txt = f"{mag:.1f}".rstrip("0").rstrip(".")
    sign = "+" if angle_deg >= 0 else "-"
    return f"_rot{sign}{angle_txt}"


# --- Helper: Load config for a given video (per-video or fallback to legacy) ---
def load_config_for_video(video_path):
    """
    Returns a dict config for a given video:
    1) Prefer per-video config: <base>_st_config.json
    2) Fallback to folder-level preview_config.json (legacy)
    3) If neither exists, return None
    """
    base, _ = os.path.splitext(video_path)
    per_video = base + "_st_config.json"
    if os.path.isfile(per_video):
        try:
            with open(per_video, "r") as f:
                return json.load(f)
        except Exception:
            pass
    folder = os.path.dirname(video_path)
    legacy = os.path.join(folder, "preview_config.json")
    if os.path.isfile(legacy):
        try:
            with open(legacy, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return None
# --- Extraction Function ---
def extract_streamlined(img_or_arr, mode, params, n_angles=1000):
    """
    Extracts grayscale values per mode.
    Accepts either a NumPy 2D array (uint8/float) or a PIL Image.
    Returns a 1D np.float32 vector. For circular, samples bilinearly and
    returns np.nan for samples outside image bounds (masking).
    """
    # Accept both PIL.Image and ndarray
    if hasattr(img_or_arr, "convert"):  # PIL Image-like
        arr = np.array(img_or_arr.convert("L"))
    else:
        arr = np.asarray(img_or_arr)
        if arr.ndim != 2:
            raise ValueError("extract_streamlined expects a single-channel 2D array")
        if arr.dtype != np.uint8:
            # ensure numeric type we can interpolate with
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
    h, w = arr.shape

    if mode == "circular":
        cx, cy, r = params["cx"], params["cy"], params["r"]
        thetas = np.linspace(0, 2*np.pi, n_angles, endpoint=False).astype(np.float32)
        xs = cx + r * np.cos(thetas)
        ys = cy + r * np.sin(thetas)
        vals = np.empty(n_angles, dtype=np.float32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = x0 + 1
            y1 = y0 + 1
            # out-of-bounds → NaN (masked)
            if (x0 < 0 or y0 < 0 or x1 >= w or y1 >= h):
                vals[i] = np.nan
                continue
            dx = x - x0
            dy = y - y0
            # bilinear
            v00 = arr[y0, x0]
            v10 = arr[y0, x1]
            v01 = arr[y1, x0]
            v11 = arr[y1, x1]
            vals[i] = (
                (1 - dx) * (1 - dy) * v00 +
                dx * (1 - dy) * v10 +
                (1 - dx) * dy * v01 +
                dx * dy * v11
            )
        return vals

    elif mode == "horizontal":
        y = float(params["y"])
        if y < 0 or y >= h:
            raise ValueError(f"Row index y={y} out of bounds for image height {h}")
        angle_deg = float(params.get("angle_deg", 0.0))
        if abs(angle_deg) < 1e-6:
            return arr[int(round(y)), :].astype(np.float32)
        slope = np.tan(np.deg2rad(angle_deg))
        xs = np.arange(w, dtype=np.float32)
        center = (w - 1) / 2.0
        ys = y + (xs - center) * slope
        return _sample_line_from_coords(arr, xs, ys)

    elif mode == "vertical":
        x = float(params["x"])
        if x < 0 or x >= w:
            raise ValueError(f"Column index x={x} out of bounds for image width {w}")
        angle_deg = float(params.get("angle_deg", 0.0))
        if abs(angle_deg) < 1e-6:
            return arr[:, int(round(x))].astype(np.float32)
        slope = np.tan(np.deg2rad(angle_deg))
        ys = np.arange(h, dtype=np.float32)
        center = (h - 1) / 2.0
        xs = x + (ys - center) * slope
        return _sample_line_from_coords(arr, xs, ys)

    raise NotImplementedError(f"Mode {mode} not implemented in extract_streamlined")

# --- Interactive Extraction GUI ---
def run_extraction(img, mode, params, n_angles_entry=None):
    global _last_n_angles
    if mode == "circular":
        # Read N_angles from entry widget, fallback to default
        try:
            n_angles = int(n_angles_entry.get())
        except Exception:
            n_angles = 360
        _last_n_angles = n_angles
        result = extract_streamlined(img, mode, params, n_angles=n_angles)
        return result
    elif mode in ("horizontal", "vertical"):
        result = extract_streamlined(img, mode, params)
        return result
    # ... handle other modes ...
    raise NotImplementedError(f"Mode {mode} not implemented in run_extraction")

def _process_frame_worker(frame, mode, params, n_angles=None):
    b_chan, g_chan, r_chan = cv2.split(frame)
    if mode == "circular":
        na = int(n_angles or 360)
        vr = extract_streamlined(r_chan, mode, params, n_angles=na)
        vg = extract_streamlined(g_chan, mode, params, n_angles=na)
        vb = extract_streamlined(b_chan, mode, params, n_angles=na)
    else:
        vr = extract_streamlined(r_chan, mode, params)
        vg = extract_streamlined(g_chan, mode, params)
        vb = extract_streamlined(b_chan, mode, params)
    return vr, vg, vb

def _longest_true_run_wrap(mask: np.ndarray):
    """Return (start_index, length) of the longest contiguous True run in a
    circular boolean mask (wrap-around allowed). If no True values, returns (0, 0)."""
    if mask is None or mask.size == 0:
        return 0, 0
    n = mask.size
    if not np.any(mask):
        return 0, 0
    # Duplicate mask to handle wrap-around
    m2 = np.concatenate([mask, mask]).astype(np.uint8)
    best_len = 0
    best_start = 0
    cur_len = 0
    cur_start = 0
    for i, v in enumerate(m2):
        if v:
            if cur_len == 0:
                cur_start = i
            cur_len += 1
            if cur_len > best_len and cur_len <= n:
                best_len = cur_len
                best_start = cur_start
        else:
            cur_len = 0
    # Cap by n and wrap start
    best_len = min(best_len, n)
    best_start = best_start % n
    return best_start, best_len

def _build_angle_index_map_from_valid(valid_mask: np.ndarray) -> np.ndarray:
    """Given a boolean mask over angles (True = inside image), build an index map
    that starts at the first angle of the **longest** valid arc and spans only
    that arc (excludes out-of-bounds/black region)."""
    start, length = _longest_true_run_wrap(valid_mask)
    if length <= 0:
        return np.array([], dtype=np.int64)
    return (start + np.arange(length, dtype=np.int64)) % valid_mask.size

def process_video_with_config(video_path, config, progress_cb=None):
    """
    Processes a video with fixed memory:
    - Preallocates per-channel (space, frames) float32 arrays
    - Writes each frame's vector in place (no Python list growth)
    - Uses NaN-aware normalization
    - Saves **only** a color PNG: RGBA (alpha masks OOB circular arcs)
    """
    mode = config.get("mode")
    params = config.get("params", {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        # Some codecs don't report; we will accumulate then trim — still preallocate conservatively
        print(f"[{os.path.basename(video_path)}] Warning: unknown total frame count; processing until EOF.")

    # Prime first frame to determine space length and seed column 0
    ret, frame = cap.read()
    if not ret:
        cap.release()
        print(f"No frames in {video_path}")
        return

    b_chan, g_chan, r_chan = cv2.split(frame)  # BGR from OpenCV
    if mode == "circular":
        n_angles = config.get("n_angles", 360)
        v_r = extract_streamlined(r_chan, mode, params, n_angles=n_angles)
        v_g = extract_streamlined(g_chan, mode, params, n_angles=n_angles)
        v_b = extract_streamlined(b_chan, mode, params, n_angles=n_angles)
    else:
        v_r = extract_streamlined(r_chan, mode, params)
        v_g = extract_streamlined(g_chan, mode, params)
        v_b = extract_streamlined(b_chan, mode, params)

    # --- Determine valid angular arc and reorder to avoid black (OOB) region ---
    if mode == "circular":
        valid0 = ~(np.isnan(v_r) | np.isnan(v_g) | np.isnan(v_b))
        angle_index_map = _build_angle_index_map_from_valid(valid0)
        if angle_index_map.size == 0:
            print("[cosmetic] Circular extraction: radius/center fall completely outside image — aborting.")
            return
        # Reindex initial vectors to start at first valid angle and drop OOB segment
        v_r = v_r[angle_index_map]
        v_g = v_g[angle_index_map]
        v_b = v_b[angle_index_map]
    else:
        angle_index_map = None

    space_len = int(v_r.shape[0])

    # Preallocate (space, frames) as float32; we'll trim to actual frames read
    if total_frames > 0:
        max_frames = total_frames
    else:
        # Fallback if unknown: start with a chunk and grow in big blocks to minimize reallocations
        max_frames = 4096
    R = np.empty((space_len, max_frames), dtype=np.float32)
    G = np.empty((space_len, max_frames), dtype=np.float32)
    B = np.empty((space_len, max_frames), dtype=np.float32)

    R[:, 0] = v_r
    G[:, 0] = v_g
    B[:, 0] = v_b

    frame_idx = 1  # we've filled column 0 already

    processed_frames = 1  # we seeded column 0

    def _print_progress(force=False):
        if total_frames > 0:
            pct = 100.0 * processed_frames / total_frames
            msg = f"[{os.path.basename(video_path)}] processed {processed_frames}/{total_frames} frames ({pct:5.1f}%)"
        else:
            msg = f"[{os.path.basename(video_path)}] processed {processed_frames} frames…"
        if not callable(progress_cb):
            print(msg)
        if callable(progress_cb):
            try:
                progress_cb(processed_frames, total_frames, msg)
            except Exception:
                pass

    # Parallel per-frame extraction (bounded inflight)
    max_workers = max(1, min((os.cpu_count() or 4) - 1, 4))
    # Circular mode is Python-loop heavy; threads still help hide I/O, but we keep inflight modest
    max_inflight = max_workers * 4
    pending = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        # Submit frames starting from index 1 (we already processed column 0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx = frame_idx
            fut = ex.submit(_process_frame_worker, frame, mode, params, config.get("n_angles", 360))
            pending[idx] = fut
            frame_idx += 1
            # Backpressure: wait for some to finish if too many in flight
            while len(pending) >= max_inflight:
                done, _ = wait(list(pending.values()), return_when=FIRST_COMPLETED)
                # Write back any completed futures
                for k, f in list(pending.items()):
                    if f.done():
                        try:
                            vr, vg, vb = f.result()
                        except Exception as e:
                            print(f"[frame {k}] extraction error: {e}")
                            vr = vg = vb = None
                        # Reorder and trim angular dimension to skip OOB region (cosmetic fix)
                        if vr is not None and angle_index_map is not None:
                            try:
                                vr = vr[angle_index_map]
                                vg = vg[angle_index_map]
                                vb = vb[angle_index_map]
                            except Exception as _remap_e:
                                print("[frame remap]", _remap_e)
                        # Grow capacity if needed
                        if k >= R.shape[1]:
                            new_cols = max(k + 1, R.shape[1] * 2)
                            R = np.pad(R, ((0, 0), (0, new_cols - R.shape[1])), mode='constant', constant_values=np.nan)
                            G = np.pad(G, ((0, 0), (0, new_cols - G.shape[1])), mode='constant', constant_values=np.nan)
                            B = np.pad(B, ((0, 0), (0, new_cols - B.shape[1])), mode='constant', constant_values=np.nan)
                        if vr is not None:
                            R[:, k] = vr
                            G[:, k] = vg
                            B[:, k] = vb
                            processed_frames = max(processed_frames, k+1)
                            if processed_frames % 500 == 0:
                                _print_progress()
                        del pending[k]
        # Drain remaining
        for k, f in sorted(pending.items()):
            try:
                vr, vg, vb = f.result()
            except Exception as e:
                print(f"[frame {k}] extraction error: {e}")
                vr = vg = vb = None
            # Reorder and trim to valid arc
            if vr is not None and angle_index_map is not None:
                try:
                    vr = vr[angle_index_map]
                    vg = vg[angle_index_map]
                    vb = vb[angle_index_map]
                except Exception as _remap_e:
                    print("[frame remap]", _remap_e)
            if k >= R.shape[1]:
                new_cols = max(k + 1, R.shape[1] * 2)
                R = np.pad(R, ((0, 0), (0, new_cols - R.shape[1])), mode='constant', constant_values=np.nan)
                G = np.pad(G, ((0, 0), (0, new_cols - G.shape[1])), mode='constant', constant_values=np.nan)
                B = np.pad(B, ((0, 0), (0, new_cols - B.shape[1])), mode='constant', constant_values=np.nan)
            if vr is not None:
                R[:, k] = vr
                G[:, k] = vg
                B[:, k] = vb
                processed_frames = max(processed_frames, k+1)
                if processed_frames % 500 == 0:
                    _print_progress()
        cap.release()
    _print_progress(force=True)
    used_frames = frame_idx
    if total_frames > 0 and used_frames != total_frames:
        # Trim to actual frames read (robustness)
        R = R[:, :used_frames]
        G = G[:, :used_frames]
        B = B[:, :used_frames]
    elif total_frames <= 0:
        # Unknown total: trim to used frames
        R = R[:, :used_frames]
        G = G[:, :used_frames]
        B = B[:, :used_frames]

    tf = f"/{total_frames}" if total_frames > 0 else ""
    print(f"[{os.path.basename(video_path)}] finished reading {used_frames}{tf} frames.")
    if callable(progress_cb):
        try:
            progress_cb(used_frames, total_frames, f"Finished reading {used_frames}{tf} frames.")
        except Exception:
            pass

    if used_frames == 0:
        print(f"No frames processed for video {video_path}")
        return

    # NaN-aware normalization per channel; keep NaNs (mask later)
    def _norm_uint8_nan(m):
        m = m.astype(np.float32, copy=False)
        if np.all(np.isnan(m)):
            return np.zeros_like(m, dtype=np.uint8)
        mn = np.nanmin(m)
        mx = np.nanmax(m)
        if not np.isfinite(mn) or not np.isfinite(mx) or mx <= mn:
            return np.zeros_like(m, dtype=np.uint8)
        scaled = (m - mn) / (mx - mn) * 255.0
        return np.nan_to_num(scaled, nan=0.0).astype(np.uint8)

    norm_r = _norm_uint8_nan(R)
    norm_g = _norm_uint8_nan(G)
    norm_b = _norm_uint8_nan(B)

    # Alpha: fully opaque (we trimmed OOB angular region away)
    alpha = np.ones_like(R, dtype=np.uint8) * 255

    # --- Vertical crop to valid data only (remove all-NaN rows) ---
    valid_mask = ~np.all(np.isnan(R), axis=1)
    if np.any(valid_mask):
        top = np.argmax(valid_mask)
        bottom = len(valid_mask) - np.argmax(valid_mask[::-1])
        R, G, B = R[top:bottom, :], G[top:bottom, :], B[top:bottom, :]
        norm_r, norm_g, norm_b = norm_r[top:bottom, :], norm_g[top:bottom, :], norm_b[top:bottom, :]
        alpha = alpha[top:bottom, :]
        print(f"Cropped vertically to region {top}:{bottom} (kept {bottom-top} pixels)")

    # Compose RGBA (space, time, 4)
    rgba = np.dstack([norm_r, norm_g, norm_b, alpha])

    # --- Unified _Processed folder output (ensure first) ---
    processed_folder = get_unified_processed_folder(os.path.dirname(video_path))  # already .../_Processed/PLI
    print(f"Unified _Processed folder for PLI extraction outputs: {processed_folder}")
    pli_proc_dir = processed_folder
    os.makedirs(pli_proc_dir, exist_ok=True)

    base, _ = os.path.splitext(video_path)
    rotation_suffix = ""
    if mode in ("horizontal", "vertical"):
        rotation_suffix = _format_rotation_suffix(float(params.get("angle_deg", 0.0)))
    mode_suffix = f"_{mode}" if mode else ""
    mode_suffix += rotation_suffix
    # Legacy, next-to-video save (kept for backward compatibility)
    legacy_png_path = f"{base}_st{mode_suffix}.png"
    # Canonical location in unified _Processed/PLI
    processed_png_name = os.path.basename(legacy_png_path)
    processed_png_path = os.path.join(pli_proc_dir, processed_png_name)

    # Save RGBA to both locations (legacy + canonical)
    Image.fromarray(rgba, mode="RGBA").save(legacy_png_path)
    try:
        # Save again directly into unified folder to avoid copy of large arrays in memory
        Image.fromarray(rgba, mode="RGBA").save(processed_png_path)
    except Exception:
        # Fallback: copy if second save fails for any reason
        try:
            shutil.copy2(legacy_png_path, processed_png_path)
        except Exception as _copy_e:
            print("Warning: could not create unified copy:", _copy_e)
    print(f"Processed video {video_path}: saved {legacy_png_path} and {processed_png_path}")

    # Output JSON in unified folder; include relative suffixes for portability
    json_path = os.path.join(processed_folder, f"_output_PLI_{os.path.splitext(os.path.basename(video_path))[0]}.json")
    def _rel(p):
        try:
            return p.split('/_Processed/', 1)[1]
        except Exception:
            return os.path.basename(p)
    outputs_abs = {
        "color_png": processed_png_path,
        "color_png_legacy": legacy_png_path,
    }
    outputs_rel = {
        "color_png_rel": _rel(processed_png_path),
        "color_png_legacy_rel": _rel(legacy_png_path),
    }
    payload = {
        "video": video_path,
        "processed_folder": processed_folder,
        "outputs": outputs_abs,
        "outputs_rel": outputs_rel,
        "exclude_from_template": True
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"PLI extraction output JSON written: {json_path}")

def load_preview_config(path):
    """
    Loads extraction mode and parameters from a JSON file.
    """
    with open(path, "r") as f:
        config = json.load(f)
    return config

def process_all_videos_in_folder(folder):
    """
    Detects all video files in folder and processes them if a config is found.
    Prefers per-video config, falls back to legacy preview_config.json.
    """
    video_extensions = {".mp4", ".avi", ".mov"}
    files = os.listdir(folder)
    video_files = [f for f in files if os.path.splitext(f)[1].lower() in video_extensions]
    tasks = []
    with ProcessPoolExecutor(max_workers=max(1, min((os.cpu_count() or 4) - 1, 4))) as ex:
        for video_file in video_files:
            video_path = os.path.join(folder, video_file)
            cfg = load_config_for_video(video_path)
            if cfg is not None:
                print(f"Queueing {video_file} with its saved config")
                tasks.append(ex.submit(process_video_with_config, video_path, cfg))
            else:
                print(f"No config found for {video_file}, skipping. (Expecting {os.path.splitext(video_path)[0]}_st_config.json or preview_config.json)")
        # Optionally wait for completion
        for fut in as_completed(tasks):
            try:
                fut.result()
            except Exception as e:
                print("Video task error:", e)

# --- Example GUI Setup (simplified for patch) ---
def setup_gui():
    global _last_n_angles
    root = tk.Tk()
    # ... other widgets ...
    mode_var = tk.StringVar(value="circular")
    # Frame for circular mode controls
    circular_frame = ttk.Frame(root)
    circular_frame.pack()
    ttk.Label(circular_frame, text="N_angles:").grid(row=0, column=0)
    n_angles_var = tk.StringVar(value=str(_last_n_angles))
    n_angles_entry = ttk.Entry(circular_frame, textvariable=n_angles_var, width=6)
    n_angles_entry.grid(row=0, column=1)
    # Extraction button
    def on_extract():
        img = Image.new("L", (100, 100))  # placeholder
        params = {'cx':50, 'cy':50, 'r':30}
        mode = mode_var.get()
        result = run_extraction(img, mode, params, n_angles_entry=n_angles_entry)
        print("Extraction result:", result)
    ttk.Button(root, text="Extract", command=on_extract).pack()
    root.mainloop()

# Uncomment to run GUI for manual testing
# setup_gui()

def extract_space_time_interactive(video_path, out_prefix):
    """
    Opens an interactive Tkinter GUI for selecting extraction parameters on a preview frame of one video
    (the last one alphabetically when multiple are provided). The chosen parameters are then applied to
    all selected videos.
    """
    if isinstance(video_path, (list, tuple, set)):
        raw_paths = list(video_path)
    else:
        raw_paths = [video_path]

    video_candidates = []
    seen = set()
    for entry in raw_paths:
        if not entry:
            continue
        abs_path = os.path.abspath(entry)
        if not os.path.isfile(abs_path):
            print(f"[PLI] Skipping non-file selection: {entry}")
            continue
        if abs_path in seen:
            continue
        seen.add(abs_path)
        video_candidates.append(abs_path)

    if not video_candidates:
        raise FileNotFoundError("No valid video files provided for extraction.")

    target_videos = list(video_candidates)
    preview_video = max(target_videos, key=lambda p: os.path.basename(p).lower())
    video_path = preview_video  # Keep legacy variable name for the interactive UI

    cap = cv2.VideoCapture(preview_video)
    if not cap.isOpened():
        print(f"Failed to open video: {preview_video}")
        return
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    def _fetch_frame_gray(frame_idx: int):
        """Read a single frame from disk and return it as grayscale; None on failure."""
        idx = max(0, int(frame_idx))
        capture = cv2.VideoCapture(preview_video)
        if not capture.isOpened():
            return None
        if idx:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw = capture.read()
        capture.release()
        if not ret:
            return None
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    preview_idx = 9
    if total_frames > 0:
        preview_idx = min(preview_idx, max(total_frames - 1, 0))
    frame_gray = _fetch_frame_gray(preview_idx)
    if frame_gray is None and preview_idx != 0:
        frame_gray = _fetch_frame_gray(0)
        preview_idx = 0
    if frame_gray is None:
        print(f"Could not read a preview frame from: {preview_video}")
        return
    pil_img = Image.fromarray(frame_gray)
    h, w = pil_img.size[1], pil_img.size[0]

    # Load previously saved config (per video or legacy)
    loaded_config = load_config_for_video(preview_video) or {}
    initial_mode = loaded_config.get("mode", "horizontal")
    initial_params = loaded_config.get("params", {})
    initial_n_angles = int(loaded_config.get("n_angles", _last_n_angles))
    remember_defaults = loaded_config.get("remember", {})  # { 'horizontal':{'y':..}, 'vertical':{'x':..}, 'circular':{'cx':..,'cy':..,'r':..,'n_angles':..} }
    rotation_memory = {
        "horizontal": float(remember_defaults.get("horizontal", {}).get("angle_deg", initial_params.get("angle_deg", 0.0))),
        "vertical": float(remember_defaults.get("vertical", {}).get("angle_deg", initial_params.get("angle_deg", 0.0))),
    }

    # --- Preview image and overlay canvas ---
    preview_scale = 0.5
    preview_w, preview_h = int(w * preview_scale), int(h * preview_scale)
    preview_img = pil_img.resize((preview_w, preview_h))
    tk_img = ImageTk.PhotoImage(preview_img)

    root = tk.Toplevel()
    root.title("Space-Time Interactive Extraction")

    if len(target_videos) > 1:
        info = ttk.Label(root, text=(
            f"Previewing: {os.path.basename(preview_video)} (alphabetically last)\n"
            f"{len(target_videos)} videos will be processed with these settings."
        ), foreground="#666666", justify="left")
        info.pack(fill="x", padx=10, pady=(6, 0))

    mode_var = tk.StringVar(value=initial_mode)
    current_frame_idx = preview_idx

    def _frame_info_text(idx: int) -> str:
        if total_frames > 0:
            return f"Preview frame: {idx + 1} / {total_frames}"
        return f"Preview frame: {idx + 1}"

    def _format_angle_display(val: float) -> str:
        txt = f"{val:.1f}"
        if txt.endswith(".0"):
            txt = txt[:-2]
        return txt or "0"

    # Frame for mode selection
    mode_frame = ttk.LabelFrame(root, text="Mode Selection")
    mode_frame.pack(fill="x", padx=10, pady=5)

    # Preview frame selector placed under mode controls
    frame_select = ttk.Frame(root)
    frame_select.pack(fill="x", padx=10, pady=(0, 5))
    if total_frames > 0:
        frame_range_label = f"Frame (1-{total_frames}):"
    else:
        frame_range_label = "Frame (>=1):"
    ttk.Label(frame_select, text=frame_range_label).grid(row=0, column=0, sticky="w")
    frame_entry_var = tk.StringVar(value=str(current_frame_idx + 1))
    frame_entry = ttk.Entry(frame_select, textvariable=frame_entry_var, width=8)
    frame_entry.grid(row=0, column=1, sticky="w")
    frame_load_btn = ttk.Button(frame_select, text="Refresh preview")
    frame_load_btn.grid(row=0, column=2, padx=(6, 0))
    frame_info_var = tk.StringVar(value=_frame_info_text(current_frame_idx))
    ttk.Label(frame_select, textvariable=frame_info_var).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 0))
    frame_select.columnconfigure(1, weight=1)

    # --- Top toolbar with ENGAGE button (always visible) ---
    topbar = ttk.Frame(root)
    topbar.pack(fill="x", padx=10, pady=(0, 5))
    engage_btn = ttk.Button(topbar, text="ENGAGE", command=lambda: on_extract())
    engage_btn.pack(side="right")

    # Progress UI (label + bar)
    progress_frame = ttk.Frame(root)
    progress_frame.pack(fill="x", padx=10, pady=(0, 6))
    progress_var = tk.StringVar(value="Idle")
    progress_label = ttk.Label(progress_frame, textvariable=progress_var)
    progress_label.pack(anchor="w")
    progress_bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
    progress_bar.pack(fill="x")

    # Frame for controls depending on mode
    controls_frame = ttk.Frame(root)
    controls_frame.pack(fill="x", padx=10, pady=5)

    # Display the 10th frame image for reference (with overlay)
    img_frame = ttk.Frame(root)
    img_frame.pack(padx=10, pady=5)
    canvas = tk.Canvas(img_frame, width=preview_w, height=preview_h)
    canvas.pack()
    canvas_img_id = canvas.create_image(0, 0, anchor="nw", image=tk_img)
    canvas._preview_img = tk_img  # Keep reference to prevent GC
    canvas._overlay_id = None
    canvas._overlay_shape = None

    rotation_frame = ttk.Frame(root)
    rotation_frame.pack(fill="x", padx=10, pady=(0, 6))
    ttk.Label(rotation_frame, text="Rotation (deg):").grid(row=0, column=0, sticky="w")
    rotation_var = tk.DoubleVar(value=rotation_memory.get(initial_mode, 0.0))
    rotation_entry_var = tk.StringVar(value=_format_angle_display(rotation_var.get()))
    def _on_rotation_slider_change(event=None):
        val = rotation_var.get()
        rotation_entry_var.set(_format_angle_display(val))
        current_mode = mode_var.get()
        if current_mode in rotation_memory:
            rotation_memory[current_mode] = val
        draw_overlay()
    def _on_rotation_entry_change(*args):
        try:
            current_text = rotation_entry_var.get()
            val = float(current_text)
        except Exception:
            return
        val = max(-60.0, min(60.0, val))
        if abs(rotation_var.get() - val) > 1e-3:
            rotation_var.set(val)
        formatted = _format_angle_display(val)
        if current_text != formatted:
            rotation_entry_var.set(formatted)
        current_mode = mode_var.get()
        if current_mode in rotation_memory:
            rotation_memory[current_mode] = val
        draw_overlay()
    rotation_slider = ttk.Scale(
        rotation_frame,
        from_=-60.0,
        to=60.0,
        orient="horizontal",
        variable=rotation_var,
        command=lambda e: _on_rotation_slider_change(),
    )
    rotation_slider.grid(row=0, column=1, sticky="ew", padx=(8, 6))
    rotation_entry = ttk.Entry(rotation_frame, textvariable=rotation_entry_var, width=6)
    rotation_entry.grid(row=0, column=2, sticky="e")
    rotation_entry_var.trace_add("write", _on_rotation_entry_change)
    rotation_frame.columnconfigure(1, weight=1)

    def _set_rotation_enabled(enabled: bool):
        if enabled:
            rotation_slider.state(["!disabled"])
            rotation_entry.state(["!disabled"])
        else:
            rotation_slider.state(["disabled"])
            rotation_entry.state(["disabled"])

    def draw_overlay():
        """Draw overlay (line/circle) on the preview canvas according to current mode and parameters."""
        # Remove previous overlay
        if hasattr(canvas, '_overlay_id') and canvas._overlay_id is not None:
            try:
                canvas.delete(canvas._overlay_id)
            except Exception:
                pass
            canvas._overlay_id = None
        if hasattr(canvas, '_overlay_shape') and canvas._overlay_shape is not None:
            try:
                if isinstance(canvas._overlay_shape, (list, tuple)):
                    for _id in canvas._overlay_shape:
                        canvas.delete(_id)
                else:
                    canvas.delete(canvas._overlay_shape)
            except Exception:
                pass
            canvas._overlay_shape = None
        mode = mode_var.get()
        # Draw overlay
        if mode == "horizontal":
            try:
                y = float(getattr(controls_frame, "y_var", tk.StringVar(value="0")).get())
            except Exception:
                y = h // 2
            angle_val = float(rotation_var.get())
            slope = np.tan(np.deg2rad(angle_val))
            center = (w - 1) / 2.0
            x0 = 0.0
            x1 = w - 1
            y0 = y + (x0 - center) * slope
            y1 = y + (x1 - center) * slope
            overlay_id = canvas.create_line(
                x0 * preview_scale,
                y0 * preview_scale,
                x1 * preview_scale,
                y1 * preview_scale,
                fill="red",
                width=2,
            )
            canvas._overlay_id = overlay_id
        elif mode == "vertical":
            try:
                x = float(getattr(controls_frame, "x_var", tk.StringVar(value="0")).get())
            except Exception:
                x = w // 2
            angle_val = float(rotation_var.get())
            slope = np.tan(np.deg2rad(angle_val))
            center = (h - 1) / 2.0
            y0 = 0.0
            y1 = h - 1
            x0 = x + (y0 - center) * slope
            x1 = x + (y1 - center) * slope
            overlay_id = canvas.create_line(
                x0 * preview_scale,
                y0 * preview_scale,
                x1 * preview_scale,
                y1 * preview_scale,
                fill="red",
                width=2,
            )
            canvas._overlay_id = overlay_id
        elif mode == "circular":
            try:
                cx = int(getattr(controls_frame, "cx_var", tk.StringVar(value=str(w//2))).get())
                cy = int(getattr(controls_frame, "cy_var", tk.StringVar(value=str(h//2))).get())
                r = int(getattr(controls_frame, "r_var", tk.StringVar(value=str(min(w,h)//4))).get())
            except Exception:
                cx, cy, r = w//2, h//2, min(w,h)//4
            free_var = getattr(controls_frame, "free_circle_var", None)
            free_mode = True if free_var is None else bool(free_var.get())
            cx_p, cy_p = int(cx * preview_scale), int(cy * preview_scale)
            if free_mode:
                r_p = int(r * preview_scale)
                overlay_shape = canvas.create_oval(
                    cx_p - r_p, cy_p - r_p, cx_p + r_p, cy_p + r_p,
                    outline="red", width=2
                )
                canvas._overlay_shape = overlay_shape
            else:
                outer_r = r
                inner_r = int(round(outer_r * (2.0 / 3.0)))
                outer_r_p = int(outer_r * preview_scale)
                inner_r_p = int(inner_r * preview_scale)
                outer_shape = canvas.create_oval(
                    cx_p - outer_r_p, cy_p - outer_r_p, cx_p + outer_r_p, cy_p + outer_r_p,
                    outline="red", width=2
                )
                inner_shape = canvas.create_oval(
                    cx_p - inner_r_p, cy_p - inner_r_p, cx_p + inner_r_p, cy_p + inner_r_p,
                    outline="red", width=2, dash=(4, 3)
                )
                canvas._overlay_shape = (outer_shape, inner_shape)

    def update_frame_info_label():
        frame_info_var.set(_frame_info_text(current_frame_idx))

    def load_preview_frame(frame_idx: int):
        """Load a requested frame into the preview canvas."""
        nonlocal pil_img, preview_img, tk_img, current_frame_idx
        if total_frames > 0:
            idx = max(0, min(int(frame_idx), total_frames - 1))
        else:
            idx = max(0, int(frame_idx))
        new_gray = _fetch_frame_gray(idx)
        if new_gray is None:
            tk.messagebox.showerror("Preview", f"Could not load frame {idx + 1}.")
            return False
        pil_img = Image.fromarray(new_gray)
        preview_img = pil_img.resize((preview_w, preview_h))
        tk_img = ImageTk.PhotoImage(preview_img)
        canvas.itemconfig(canvas_img_id, image=tk_img)
        canvas._preview_img = tk_img
        current_frame_idx = idx
        frame_entry_var.set(str(current_frame_idx + 1))
        update_frame_info_label()
        draw_overlay()
        return True

    def refresh_preview_from_entry(event=None):
        try:
            requested = int(frame_entry_var.get())
        except ValueError:
            tk.messagebox.showerror("Preview", "Please enter a valid frame number.")
            return
        if requested <= 0:
            tk.messagebox.showerror("Preview", "Frame numbers start at 1.")
            return
        load_preview_frame(requested - 1)

    frame_load_btn.config(command=refresh_preview_from_entry)
    frame_entry.bind("<Return>", refresh_preview_from_entry)

    def on_mode_change():
        mode = mode_var.get()
        # Show/hide controls frame accordingly
        for widget in controls_frame.winfo_children():
            widget.destroy()
        # Remove overlay
        draw_overlay()
        if mode == "horizontal":
            # Row slider
            y_default = int(remember_defaults.get("horizontal", {}).get("y", initial_params.get("y", h//2)))
            y_var = tk.StringVar(value=str(y_default))
            def on_row_slider_change(event=None):
                val = int(row_slider.get())
                y_var.set(str(val))
                draw_overlay()
            def on_y_entry_change(*args):
                try:
                    val = int(y_var.get())
                    if 0 <= val <= h - 1:
                        row_slider.set(val)
                        draw_overlay()
                except Exception:
                    pass
            ttk.Label(controls_frame, text="Row (y):").grid(row=0, column=0, sticky="w")
            row_slider = ttk.Scale(controls_frame, from_=0, to=h-1, orient="horizontal", command=lambda e: on_row_slider_change())
            row_slider.set(y_default)
            row_slider.grid(row=0, column=1, sticky="ew")
            y_entry = ttk.Entry(controls_frame, textvariable=y_var, width=6)
            y_entry.grid(row=0, column=2, sticky="w")
            y_var.trace_add('write', on_y_entry_change)
            controls_frame.row_slider = row_slider
            controls_frame.y_var = y_var
            controls_frame.y_entry = y_entry
            angle_default = rotation_memory.get("horizontal", rotation_var.get())
            rotation_var.set(angle_default)
            rotation_entry_var.set(_format_angle_display(angle_default))
            _set_rotation_enabled(True)
            # Initial draw
            draw_overlay()
        elif mode == "vertical":
            # Column slider
            x_default = int(remember_defaults.get("vertical", {}).get("x", initial_params.get("x", w//2)))
            x_var = tk.StringVar(value=str(x_default))
            def on_col_slider_change(event=None):
                val = int(col_slider.get())
                x_var.set(str(val))
                draw_overlay()
            def on_x_entry_change(*args):
                try:
                    val = int(x_var.get())
                    if 0 <= val <= w - 1:
                        col_slider.set(val)
                        draw_overlay()
                except Exception:
                    pass
            ttk.Label(controls_frame, text="Column (x):").grid(row=0, column=0, sticky="w")
            col_slider = ttk.Scale(controls_frame, from_=0, to=w-1, orient="horizontal", command=lambda e: on_col_slider_change())
            col_slider.set(x_default)
            col_slider.grid(row=0, column=1, sticky="ew")
            x_entry = ttk.Entry(controls_frame, textvariable=x_var, width=6)
            x_entry.grid(row=0, column=2, sticky="w")
            x_var.trace_add('write', on_x_entry_change)
            controls_frame.col_slider = col_slider
            controls_frame.x_var = x_var
            controls_frame.x_entry = x_entry
            angle_default = rotation_memory.get("vertical", rotation_var.get())
            rotation_var.set(angle_default)
            rotation_entry_var.set(_format_angle_display(angle_default))
            _set_rotation_enabled(True)
            draw_overlay()
        elif mode == "circular":
            # Center X, Y sliders and entries, radius slider and entry, N_angles entry
            circ_defaults = remember_defaults.get("circular", {})
            cx_default = int(circ_defaults.get("cx", initial_params.get("cx", w//2)))
            cy_default = int(circ_defaults.get("cy", initial_params.get("cy", h//2)))
            free_default = circ_defaults.get("free", None)
            if free_default is None:
                if "outer_r" in circ_defaults or "outer_r" in initial_params:
                    free_default = False
                elif "r" in circ_defaults or "r" in initial_params:
                    free_default = True
                else:
                    free_default = False
            free_default = bool(free_default)
            if free_default:
                r_default = int(circ_defaults.get("r", initial_params.get("r", min(w, h)//4)))
            else:
                r_default = int(circ_defaults.get("outer_r", initial_params.get("outer_r", initial_params.get("r", min(w, h)//4))))
            cx_var = tk.StringVar(value=str(cx_default))
            cy_var = tk.StringVar(value=str(cy_default))
            r_var  = tk.StringVar(value=str(r_default))
            def on_cx_slider_change(event=None):
                val = int(cx_slider.get())
                cx_var.set(str(val))
                draw_overlay()
            def on_cy_slider_change(event=None):
                val = int(cy_slider.get())
                cy_var.set(str(val))
                draw_overlay()
            def on_r_slider_change(event=None):
                val = int(r_slider.get())
                r_var.set(str(val))
                draw_overlay()
            def on_cx_entry_change(*args):
                try:
                    val = int(cx_var.get())
                    # Allow -2*w to 2*w for circular overlay
                    if -2*w <= val <= 2*w:
                        cx_slider.set(val)
                        draw_overlay()
                except Exception:
                    pass
            def on_cy_entry_change(*args):
                try:
                    val = int(cy_var.get())
                    if -2*h <= val <= 2*h:
                        cy_slider.set(val)
                        draw_overlay()
                except Exception:
                    pass
            def on_r_entry_change(*args):
                try:
                    val = int(r_var.get())
                    if 0 <= val <= 2*min(w, h):
                        r_slider.set(val)
                        draw_overlay()
                except Exception:
                    pass
            def on_free_toggle():
                label_text = "Radius:" if free_circle_var.get() else "Outer radius:"
                radius_label.config(text=label_text)
                draw_overlay()
            ttk.Label(controls_frame, text="Center X:").grid(row=0, column=0, sticky="w")
            cx_slider = ttk.Scale(controls_frame, from_=-2*w, to=2*w, orient="horizontal", command=lambda e: on_cx_slider_change())
            cx_slider.set(cx_default)
            cx_slider.grid(row=0, column=1, sticky="ew")
            cx_entry = ttk.Entry(controls_frame, textvariable=cx_var, width=6)
            cx_entry.grid(row=0, column=2, sticky="w")
            cx_var.trace_add('write', on_cx_entry_change)

            ttk.Label(controls_frame, text="Center Y:").grid(row=1, column=0, sticky="w")
            cy_slider = ttk.Scale(controls_frame, from_=-2*h, to=2*h, orient="horizontal", command=lambda e: on_cy_slider_change())
            cy_slider.set(cy_default)
            cy_slider.grid(row=1, column=1, sticky="ew")
            cy_entry = ttk.Entry(controls_frame, textvariable=cy_var, width=6)
            cy_entry.grid(row=1, column=2, sticky="w")
            cy_var.trace_add('write', on_cy_entry_change)

            free_circle_var = tk.BooleanVar(value=free_default)
            free_cb = ttk.Checkbutton(
                controls_frame,
                text="Free circle definition (default 2/3R)",
                variable=free_circle_var,
                command=on_free_toggle
            )
            free_cb.grid(row=2, column=0, columnspan=3, sticky="w", pady=(2, 2))

            radius_label = ttk.Label(controls_frame, text="Radius:" if free_default else "Outer radius:")
            radius_label.grid(row=3, column=0, sticky="w")
            r_slider = ttk.Scale(controls_frame, from_=0, to=2*min(w, h), orient="horizontal", command=lambda e: on_r_slider_change())
            r_slider.set(r_default)
            r_slider.grid(row=3, column=1, sticky="ew")
            r_entry = ttk.Entry(controls_frame, textvariable=r_var, width=6)
            r_entry.grid(row=3, column=2, sticky="w")
            r_var.trace_add('write', on_r_entry_change)

            ttk.Label(controls_frame, text="N_angles:").grid(row=4, column=0, sticky="w")
            n_angles_var = tk.StringVar(value=str(int(circ_defaults.get("n_angles", initial_n_angles))))
            n_angles_entry = ttk.Entry(controls_frame, textvariable=n_angles_var, width=6)
            n_angles_entry.grid(row=4, column=1, sticky="w")
            # Store references for extraction
            controls_frame.cx_var = cx_var
            controls_frame.cy_var = cy_var
            controls_frame.r_var = r_var
            controls_frame.n_angles_var = n_angles_var
            controls_frame.cx_entry = cx_entry
            controls_frame.cy_entry = cy_entry
            controls_frame.r_entry = r_entry
            controls_frame.n_angles_entry = n_angles_entry
            controls_frame.cx_slider = cx_slider
            controls_frame.cy_slider = cy_slider
            controls_frame.r_slider = r_slider
            controls_frame.free_circle_var = free_circle_var
            cx_slider.set(cx_default)
            cy_slider.set(cy_default)
            r_slider.set(r_default)
            _set_rotation_enabled(False)
            draw_overlay()
        controls_frame.columnconfigure(1, weight=1)

    # Mode radio buttons
    for i, m in enumerate(["horizontal", "vertical", "circular"]):
        rb = ttk.Radiobutton(mode_frame, text=m.capitalize(), variable=mode_var, value=m, command=on_mode_change)
        rb.grid(row=0, column=i, padx=5, pady=5)

    on_mode_change()

    def on_extract():
        print("Extract button clicked")
        mode = mode_var.get()
        params = {}
        # For circular mode, persist n_angles across session
        global _last_n_angles
        if mode == "horizontal":
            y = float(controls_frame.y_var.get())
            angle_val = float(rotation_var.get())
            params = {'y': y, 'angle_deg': angle_val}
        elif mode == "vertical":
            x = float(controls_frame.x_var.get())
            angle_val = float(rotation_var.get())
            params = {'x': x, 'angle_deg': angle_val}
        elif mode == "circular":
            try:
                cx = int(controls_frame.cx_var.get())
                cy = int(controls_frame.cy_var.get())
                r = int(controls_frame.r_var.get())
                free_mode = bool(controls_frame.free_circle_var.get())
                n_angles = int(controls_frame.n_angles_var.get())
                if free_mode:
                    r_extract = int(r)
                    outer_r = None
                else:
                    outer_r = int(r)
                    r_extract = int(round(outer_r * (2.0 / 3.0)))
                params = {'cx': cx, 'cy': cy, 'r': r_extract}
                _last_n_angles = n_angles
            except Exception as e:
                tk.messagebox.showerror("Input error", f"Invalid circular parameters: {e}")
                return
        else:
            tk.messagebox.showerror("Mode error", f"Unsupported mode: {mode}")
            return
        # Update per-mode remembered defaults
        # Make a shallow copy so we don't mutate loaded_config directly
        nonlocal remember_defaults
        rd = dict(remember_defaults)
        if mode == "horizontal":
            rd["horizontal"] = {"y": float(params["y"]), "angle_deg": float(params.get("angle_deg", 0.0))}
        elif mode == "vertical":
            rd["vertical"] = {"x": float(params["x"]), "angle_deg": float(params.get("angle_deg", 0.0))}
        elif mode == "circular":
            circ_prev = dict(remember_defaults.get("circular", {}))
            circ_prev["cx"] = int(params["cx"])
            circ_prev["cy"] = int(params["cy"])
            circ_prev["n_angles"] = int(n_angles)
            circ_prev["free"] = bool(free_mode)
            if free_mode:
                circ_prev["r"] = int(r_extract)
            else:
                circ_prev["outer_r"] = int(outer_r)
                circ_prev["r"] = int(r_extract)
            rd["circular"] = circ_prev
        remember_defaults = rd
        config = {
            "mode": mode,
            "params": params,
            "remember": remember_defaults
        }
        if mode == "circular":
            config["n_angles"] = n_angles  # legacy field retained
        def _persist_config_for(path):
            base_no_ext, _ = os.path.splitext(path)
            per_video_config_path = base_no_ext + "_st_config.json"
            folder = os.path.dirname(path)
            legacy_config_path = os.path.join(folder, "preview_config.json")
            try:
                with open(per_video_config_path, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"Saved per-video config to {per_video_config_path}")
            except Exception as e:
                tk.messagebox.showerror("File error", f"Could not save {per_video_config_path}: {e}")
                raise
            try:
                with open(legacy_config_path, "w") as f:
                    json.dump(config, f, indent=4)
                print(f"Updated legacy config at {legacy_config_path}")
            except Exception as e:
                print(f"Warning: could not update legacy preview_config.json: {e}")

        try:
            for vid in target_videos:
                _persist_config_for(vid)
        except Exception:
            return
        print("Starting video processing")
        engage_btn.config(state="disabled")
        progress_var.set("Starting…")
        progress_bar['value'] = 0

        def gui_progress_cb(done, total, msg):
            # Marshal updates onto Tk thread
            def _upd():
                progress_var.set(msg)
                if total and total > 0:
                    pct = max(0.0, min(100.0, 100.0 * float(done) / float(total)))
                    progress_bar['value'] = pct
                else:
                    # Indeterminate when total unknown
                    progress_bar['mode'] = 'indeterminate'
                    progress_bar.start(50)
            root.after(0, _upd)

        def worker():
            try:
                total_targets = len(target_videos)
                for idx, vid in enumerate(target_videos, start=1):
                    def _cb(done, total, msg, idx=idx):
                        prefix = f"[{idx}/{total_targets}] " if total_targets > 1 else ""
                        gui_progress_cb(done, total, prefix + msg)
                    process_video_with_config(vid, config, progress_cb=_cb)
            except Exception as e:
                def _err():
                    progress_var.set(f"Error: {e}")
                root.after(0, _err)
            finally:
                def _done():
                    try:
                        progress_bar.stop()
                        progress_bar['mode'] = 'determinate'
                        progress_bar['value'] = 100
                        progress_var.set("Done.")
                        engage_btn.config(state="normal")
                        tk.messagebox.showinfo("Done", "Extraction completed and saved.")
                    except Exception:
                        pass
                root.after(0, _done)

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    # (ENGAGE button is now at the top; remove bottom Extract button)



# Allow running as a script with a video path argument
if __name__ == "__main__":
    import sys
    try:
        if len(sys.argv) < 2:
            print("Usage: python PLI_extract_st_diag_v2.py /path/to/video [extra_videos…]")
        else:
            extract_space_time_interactive(sys.argv[1:], "output")
    except Exception as e:
        import traceback
        print("❌ ERROR in PLI_extract_st_diag_v2:")
        traceback.print_exc()
