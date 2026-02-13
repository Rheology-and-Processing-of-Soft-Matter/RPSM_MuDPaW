import os
import sys
import json
import shutil
import numpy as np
import cv2
from PIL import Image

from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QProgressBar,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
    QMessageBox,
    QSizePolicy,
)

# --- Unified _Processed Folder Helpers (copied from v2, no Tk deps) ---

def _sanitize_misjoined_user_path(p: str) -> str:
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
    norm = os.path.normpath(path)
    parts = [p for p in norm.split(os.sep) if p not in ("", ".")]
    return parts


def get_reference_folder_from_path(path):
    markers = {"PLI", "PI", "SAXS", "Rheology"}
    abspath = _sanitize_misjoined_user_path(os.path.abspath(path))
    parts = _split_parts(abspath)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            reference = os.sep.join(parts[:i])
            if reference == "":
                break
            reference = _sanitize_misjoined_user_path(reference)
            reference = os.path.abspath(reference)
            if not reference.startswith(os.sep):
                reference = os.sep + reference
            return reference if reference else os.sep
    abspath = _sanitize_misjoined_user_path(abspath)
    abspath = os.path.abspath(abspath)
    if not abspath.startswith(os.sep):
        abspath = os.sep + abspath
    return abspath


def get_unified_processed_folder(path):
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


_last_n_angles = 360


def _sample_line_from_coords(arr: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
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
    if abs(angle_deg) < 1e-6:
        return ""
    mag = abs(angle_deg)
    if abs(mag - round(mag)) < 1e-6:
        angle_txt = str(int(round(mag)))
    else:
        angle_txt = f"{mag:.1f}".rstrip("0").rstrip(".")
    sign = "+" if angle_deg >= 0 else "-"
    return f"_rot{sign}{angle_txt}"


def load_config_for_video(video_path):
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


# --- Extraction core (copied from v2) ---

def extract_streamlined(img_or_arr, mode, params, n_angles=1000):
    if hasattr(img_or_arr, "convert"):
        arr = np.array(img_or_arr.convert("L"))
    else:
        arr = np.asarray(img_or_arr)
        if arr.ndim != 2:
            raise ValueError("extract_streamlined expects a single-channel 2D array")
        if arr.dtype != np.uint8:
            arr = arr.astype(np.float32)
        else:
            arr = arr.astype(np.float32)
    h, w = arr.shape

    if mode == "circular":
        cx, cy, r = params["cx"], params["cy"], params["r"]
        thetas = np.linspace(0, 2 * np.pi, n_angles, endpoint=False).astype(np.float32)
        xs = cx + r * np.cos(thetas)
        ys = cy + r * np.sin(thetas)
        vals = np.empty(n_angles, dtype=np.float32)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x0 = int(np.floor(x))
            y0 = int(np.floor(y))
            x1 = x0 + 1
            y1 = y0 + 1
            if (x0 < 0 or y0 < 0 or x1 >= w or y1 >= h):
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

    if mode == "horizontal":
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

    if mode == "vertical":
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
    if mask is None or mask.size == 0:
        return 0, 0
    n = mask.size
    if not np.any(mask):
        return 0, 0
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
    best_len = min(best_len, n)
    best_start = best_start % n
    return best_start, best_len


def _build_angle_index_map_from_valid(valid_mask: np.ndarray) -> np.ndarray:
    start, length = _longest_true_run_wrap(valid_mask)
    if length <= 0:
        return np.array([], dtype=np.int64)
    return (start + np.arange(length, dtype=np.int64)) % valid_mask.size


def process_video_with_config(video_path, config, progress_cb=None):
    mode = config.get("mode")
    params = config.get("params", {})

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total_frames <= 0:
        print(f"[{os.path.basename(video_path)}] Warning: unknown total frame count; processing until EOF.")

    ret, frame = cap.read()
    if not ret:
        cap.release()
        print(f"No frames in {video_path}")
        return

    b_chan, g_chan, r_chan = cv2.split(frame)
    if mode == "circular":
        n_angles = config.get("n_angles", 360)
        v_r = extract_streamlined(r_chan, mode, params, n_angles=n_angles)
        v_g = extract_streamlined(g_chan, mode, params, n_angles=n_angles)
        v_b = extract_streamlined(b_chan, mode, params, n_angles=n_angles)
    else:
        v_r = extract_streamlined(r_chan, mode, params)
        v_g = extract_streamlined(g_chan, mode, params)
        v_b = extract_streamlined(b_chan, mode, params)

    if mode == "circular":
        valid0 = ~(np.isnan(v_r) | np.isnan(v_g) | np.isnan(v_b))
        angle_index_map = _build_angle_index_map_from_valid(valid0)
        if angle_index_map.size == 0:
            print("[cosmetic] Circular extraction: radius/center fall completely outside image — aborting.")
            return
        v_r = v_r[angle_index_map]
        v_g = v_g[angle_index_map]
        v_b = v_b[angle_index_map]
    else:
        angle_index_map = None

    space_len = int(v_r.shape[0])
    if total_frames > 0:
        max_frames = total_frames
    else:
        max_frames = 4096
    R = np.empty((space_len, max_frames), dtype=np.float32)
    G = np.empty((space_len, max_frames), dtype=np.float32)
    B = np.empty((space_len, max_frames), dtype=np.float32)

    R[:, 0] = v_r
    G[:, 0] = v_g
    B[:, 0] = v_b

    frame_idx = 1
    processed_frames = 1

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

    max_workers = max(1, min((os.cpu_count() or 4) - 1, 4))
    max_inflight = max_workers * 4
    pending = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            idx = frame_idx
            fut = ex.submit(_process_frame_worker, frame, mode, params, config.get("n_angles", 360))
            pending[idx] = fut
            frame_idx += 1
            while len(pending) >= max_inflight:
                wait(list(pending.values()), return_when=FIRST_COMPLETED)
                for k, f in list(pending.items()):
                    if f.done():
                        try:
                            vr, vg, vb = f.result()
                        except Exception as e:
                            print(f"[frame {k}] extraction error: {e}")
                            vr = vg = vb = None
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
                            processed_frames = max(processed_frames, k + 1)
                            if processed_frames % 500 == 0:
                                _print_progress()
                        del pending[k]
        for k, f in sorted(pending.items()):
            try:
                vr, vg, vb = f.result()
            except Exception as e:
                print(f"[frame {k}] extraction error: {e}")
                vr = vg = vb = None
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
                processed_frames = max(processed_frames, k + 1)
                if processed_frames % 500 == 0:
                    _print_progress()
        cap.release()
    _print_progress(force=True)
    used_frames = frame_idx
    if total_frames > 0 and used_frames != total_frames:
        R = R[:, :used_frames]
        G = G[:, :used_frames]
        B = B[:, :used_frames]
    elif total_frames <= 0:
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

    alpha = np.ones_like(R, dtype=np.uint8) * 255

    valid_mask = ~np.all(np.isnan(R), axis=1)
    if np.any(valid_mask):
        top = np.argmax(valid_mask)
        bottom = len(valid_mask) - np.argmax(valid_mask[::-1])
        R, G, B = R[top:bottom, :], G[top:bottom, :], B[top:bottom, :]
        norm_r, norm_g, norm_b = norm_r[top:bottom, :], norm_g[top:bottom, :], norm_b[top:bottom, :]
        alpha = alpha[top:bottom, :]
        print(f"Cropped vertically to region {top}:{bottom} (kept {bottom - top} pixels)")

    rgba = np.dstack([norm_r, norm_g, norm_b, alpha])

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
    legacy_png_path = f"{base}_st{mode_suffix}.png"
    processed_png_name = os.path.basename(legacy_png_path)
    processed_png_path = os.path.join(pli_proc_dir, processed_png_name)

    Image.fromarray(rgba, mode="RGBA").save(legacy_png_path)
    try:
        Image.fromarray(rgba, mode="RGBA").save(processed_png_path)
    except Exception:
        try:
            shutil.copy2(legacy_png_path, processed_png_path)
        except Exception as _copy_e:
            print("Warning: could not create unified copy:", _copy_e)
    print(f"Processed video {video_path}: saved {legacy_png_path} and {processed_png_path}")

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
        "exclude_from_template": True,
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"PLI extraction output JSON written: {json_path}")


class ExtractionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, videos, config):
        super().__init__()
        self.videos = videos
        self.config = config

    def run(self):
        try:
            total_targets = len(self.videos)
            for idx, vid in enumerate(self.videos, start=1):
                def _cb(done, total, msg, idx=idx):
                    prefix = f"[{idx}/{total_targets}] " if total_targets > 1 else ""
                    self.progress.emit(done, total, prefix + msg)
                process_video_with_config(vid, self.config, progress_cb=_cb)
        except Exception as e:
            self.failed.emit(str(e))
            return
        self.finished.emit()


class ExtractorWindow(QWidget):
    def __init__(self, video_paths):
        super().__init__()
        self.setWindowTitle("Space-Time Interactive Extraction (Qt)")
        self.target_videos = self._normalize_videos(video_paths)
        if not self.target_videos:
            raise FileNotFoundError("No valid video files provided for extraction.")
        self.preview_video = max(self.target_videos, key=lambda p: os.path.basename(p).lower())

        self.preview_scale = 0.5
        self.total_frames = self._get_total_frames(self.preview_video)
        self.current_frame_idx = 0
        self.frame_gray = None

        loaded_config = load_config_for_video(self.preview_video) or {}
        self.remember_defaults = loaded_config.get("remember", {})
        self.rotation_memory = {
            "horizontal": float(self.remember_defaults.get("horizontal", {}).get("angle_deg", loaded_config.get("params", {}).get("angle_deg", 0.0))),
            "vertical": float(self.remember_defaults.get("vertical", {}).get("angle_deg", loaded_config.get("params", {}).get("angle_deg", 0.0))),
        }
        self.initial_mode = loaded_config.get("mode", "horizontal")
        self.initial_params = loaded_config.get("params", {})
        self.initial_n_angles = int(loaded_config.get("n_angles", _last_n_angles))

        self._build_ui()
        self._load_frame(9 if self.total_frames > 0 else 0)
        self._apply_initial_params()
        self._update_overlay()

    # --- UI helpers for paired sliders/spinboxes ---
    def _wrap_spin_slider(self, spin: QWidget, slider: QSlider) -> QWidget:
        box = QWidget()
        h = QHBoxLayout(box)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(6)
        h.addWidget(spin)
        slider.setMinimumWidth(140)
        slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        h.addWidget(slider, 1)
        return box

    def _bind_slider(self, slider: QSlider, spin, *, scale: float = 1.0) -> None:
        """Keep slider (int) and spin (int/float) in sync."""
        def spin_to_slider(val):
            try:
                slider.blockSignals(True)
                slider.setValue(int(round(float(val) * scale)))
            finally:
                slider.blockSignals(False)

        def slider_to_spin(val):
            try:
                spin.blockSignals(True)
                spin.setValue(val / scale)
            finally:
                spin.blockSignals(False)
            try:
                spin.valueChanged.emit(spin.value())
            except Exception:
                pass

        spin.valueChanged.connect(spin_to_slider)
        slider.valueChanged.connect(slider_to_spin)

    def _set_slider_range(self, slider: QSlider, lo: float, hi: float, *, scale: float = 1.0) -> None:
        try:
            slider.blockSignals(True)
            slider.setRange(int(round(lo * scale)), int(round(hi * scale)))
        finally:
            slider.blockSignals(False)

    def _normalize_videos(self, raw_paths):
        if isinstance(raw_paths, (list, tuple, set)):
            raw = list(raw_paths)
        else:
            raw = [raw_paths]
        video_candidates = []
        seen = set()
        for entry in raw:
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
        return video_candidates

    def _get_total_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()
        return total

    def _fetch_frame_gray(self, frame_idx: int):
        idx = max(0, int(frame_idx))
        capture = cv2.VideoCapture(self.preview_video)
        if not capture.isOpened():
            return None
        if idx:
            capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, raw = capture.read()
        capture.release()
        if not ret:
            return None
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    def _build_ui(self):
        root = QVBoxLayout(self)

        if len(self.target_videos) > 1:
            info = QLabel(
                f"Previewing: {os.path.basename(self.preview_video)} (alphabetically last)\n"
                f"{len(self.target_videos)} videos will be processed with these settings."
            )
            info.setStyleSheet("color: #666666;")
            root.addWidget(info)

        topbar = QHBoxLayout()
        self.btnEngage = QPushButton("ENGAGE")
        self.btnEngage.clicked.connect(self._on_engage)
        topbar.addWidget(self.btnEngage)
        topbar.addStretch(1)
        root.addLayout(topbar)

        self.progressLabel = QLabel("")
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        root.addWidget(self.progressLabel)
        root.addWidget(self.progressBar)

        # Mode selection
        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.modeCombo = QComboBox()
        self.modeCombo.addItems(["horizontal", "vertical", "circular"])
        self.modeCombo.currentTextChanged.connect(self._on_mode_changed)
        mode_row.addWidget(self.modeCombo)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        # Frame selection
        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Preview frame:"))
        self.frameSpin = QSpinBox()
        self.frameSpin.setMinimum(1)
        max_frame = max(1, self.total_frames)
        self.frameSpin.setMaximum(max_frame)
        self.frameSpin.setValue(min(10, max_frame))
        frame_row.addWidget(self.frameSpin)
        self.btnRefreshFrame = QPushButton("Refresh preview")
        self.btnRefreshFrame.clicked.connect(self._on_refresh_frame)
        frame_row.addWidget(self.btnRefreshFrame)
        self.frameInfo = QLabel("")
        frame_row.addWidget(self.frameInfo)
        frame_row.addStretch(1)
        root.addLayout(frame_row)

        # Rotation controls (spin + slider stacked)
        rotation_grid = QGridLayout()
        rotation_grid.setContentsMargins(0, 0, 0, 0)
        rotation_grid.setHorizontalSpacing(8)
        rotation_grid.setVerticalSpacing(4)
        rotation_grid.addWidget(QLabel("Rotation (deg):"), 0, 0)
        self.rotationSpin = QDoubleSpinBox()
        self.rotationSpin.setRange(-60.0, 60.0)
        self.rotationSpin.setDecimals(1)
        self.rotationSpin.setSingleStep(0.5)
        self.rotationSpin.valueChanged.connect(self._on_rotation_changed)
        self.rotationSlider = QSlider(Qt.Orientation.Horizontal)
        self.rotationSlider.setSingleStep(1)
        self.rotationSlider.setPageStep(5)
        self.rotationSlider.setMinimumWidth(200)
        self._bind_slider(self.rotationSlider, self.rotationSpin, scale=10.0)
        self._set_slider_range(self.rotationSlider, -60.0, 60.0, scale=10.0)
        rotation_grid.addWidget(self.rotationSpin, 0, 1)
        rotation_grid.addWidget(self.rotationSlider, 1, 0, 1, 3)
        rotation_grid.setColumnStretch(2, 1)
        root.addLayout(rotation_grid)

        # Controls per mode
        self.controlsStack = QStackedWidget()
        self._build_horizontal_controls()
        self._build_vertical_controls()
        self._build_circular_controls()
        root.addWidget(self.controlsStack)

        # Preview view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        root.addWidget(self.view, 1)

        self.overlay_items = []
        self.modeCombo.setCurrentText(self.initial_mode)

    def _build_horizontal_controls(self):
        box = QGroupBox("Horizontal")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        self.h_y = QSpinBox()
        self.h_y.setMinimum(0)
        self.h_y.valueChanged.connect(self._update_overlay)
        self.h_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_y_slider.setSingleStep(1)
        self.h_y_slider.setPageStep(5)
        self.h_y_slider.setMinimumWidth(220)
        self._bind_slider(self.h_y_slider, self.h_y)
        layout.addWidget(QLabel("Row (y):"), 0, 0)
        layout.addWidget(self.h_y, 0, 1)
        layout.addWidget(self.h_y_slider, 1, 0, 1, 3)
        layout.setColumnStretch(2, 1)
        layout.setColumnMinimumWidth(2, 220)
        self.controlsStack.addWidget(box)

    def _build_vertical_controls(self):
        box = QGroupBox("Vertical")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        self.v_x = QSpinBox()
        self.v_x.setMinimum(0)
        self.v_x.valueChanged.connect(self._update_overlay)
        self.v_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_x_slider.setSingleStep(1)
        self.v_x_slider.setPageStep(5)
        self.v_x_slider.setMinimumWidth(220)
        self._bind_slider(self.v_x_slider, self.v_x)
        layout.addWidget(QLabel("Column (x):"), 0, 0)
        layout.addWidget(self.v_x, 0, 1)
        layout.addWidget(self.v_x_slider, 1, 0, 1, 3)
        layout.setColumnStretch(2, 1)
        layout.setColumnMinimumWidth(2, 220)
        self.controlsStack.addWidget(box)

    def _build_circular_controls(self):
        box = QGroupBox("Circular")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(6)
        self.c_cx = QSpinBox()
        self.c_cy = QSpinBox()
        self.c_r = QSpinBox()
        self.c_n = QSpinBox()
        self.c_free = QCheckBox("Free radius (no outer)")
        for w in (self.c_cx, self.c_cy, self.c_r, self.c_n):
            w.valueChanged.connect(self._update_overlay)
        self.c_free.stateChanged.connect(self._update_overlay)
        self.c_n.setMinimum(10)
        self.c_n.setMaximum(5000)
        self.c_cx_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_cy_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_r_slider = QSlider(Qt.Orientation.Horizontal)
        for s in (self.c_cx_slider, self.c_cy_slider, self.c_r_slider):
            s.setSingleStep(1)
            s.setPageStep(10)
            s.setMinimumWidth(220)
        self._bind_slider(self.c_cx_slider, self.c_cx)
        self._bind_slider(self.c_cy_slider, self.c_cy)
        self._bind_slider(self.c_r_slider, self.c_r)

        layout.addWidget(QLabel("Center X:"), 0, 0)
        layout.addWidget(self.c_cx, 0, 1)
        layout.addWidget(self.c_cx_slider, 1, 0, 1, 3)
        layout.addWidget(QLabel("Center Y:"), 2, 0)
        layout.addWidget(self.c_cy, 2, 1)
        layout.addWidget(self.c_cy_slider, 3, 0, 1, 3)
        layout.addWidget(QLabel("Radius:"), 4, 0)
        layout.addWidget(self.c_r, 4, 1)
        layout.addWidget(self.c_r_slider, 5, 0, 1, 3)
        layout.addWidget(QLabel("N_angles:"), 6, 0)
        layout.addWidget(self.c_n, 6, 1)
        layout.addWidget(self.c_free, 7, 0, 1, 3)
        layout.setColumnStretch(2, 1)
        layout.setColumnMinimumWidth(2, 220)
        self.controlsStack.addWidget(box)

    def _apply_initial_params(self):
        if self.frame_gray is None:
            return
        h, w = self.frame_gray.shape
        self.h_y.setMaximum(max(0, h - 1))
        self._set_slider_range(self.h_y_slider, 0, max(0, h - 1))
        self.v_x.setMaximum(max(0, w - 1))
        self._set_slider_range(self.v_x_slider, 0, max(0, w - 1))
        self.c_cx.setRange(-2 * w, 2 * w)
        self.c_cy.setRange(-2 * h, 2 * h)
        self.c_r.setRange(0, 2 * min(w, h))
        self._set_slider_range(self.c_cx_slider, -2 * w, 2 * w)
        self._set_slider_range(self.c_cy_slider, -2 * h, 2 * h)
        self._set_slider_range(self.c_r_slider, 0, 2 * min(w, h))

        params = dict(self.initial_params)
        mode = self.initial_mode
        if mode == "horizontal":
            y = int(params.get("y", h // 2))
            self.h_y.setValue(max(0, min(h - 1, y)))
        elif mode == "vertical":
            x = int(params.get("x", w // 2))
            self.v_x.setValue(max(0, min(w - 1, x)))
        elif mode == "circular":
            circ = self.remember_defaults.get("circular", {})
            cx = int(circ.get("cx", params.get("cx", w // 2)))
            cy = int(circ.get("cy", params.get("cy", h // 2)))
            if circ.get("free"):
                r = int(circ.get("r", params.get("r", min(w, h) // 3)))
            else:
                r = int(circ.get("outer_r", params.get("r", min(w, h) // 2)))
            self.c_cx.setValue(cx)
            self.c_cy.setValue(cy)
            self.c_r.setValue(r)
            self.c_n.setValue(int(circ.get("n_angles", self.initial_n_angles)))
            self.c_free.setChecked(bool(circ.get("free", False)))

        rot = self.rotation_memory.get(mode, float(params.get("angle_deg", 0.0)))
        self.rotationSpin.setValue(float(rot))
        self._set_slider_range(self.rotationSlider, -60.0, 60.0, scale=10.0)

    def _on_mode_changed(self, mode):
        idx = {"horizontal": 0, "vertical": 1, "circular": 2}.get(mode, 0)
        self.controlsStack.setCurrentIndex(idx)
        rot_enabled = mode in {"horizontal", "vertical"}
        self.rotationSpin.setEnabled(rot_enabled)
        self.rotationSlider.setEnabled(rot_enabled)
        self._update_overlay()

    def _on_rotation_changed(self):
        mode = self.modeCombo.currentText()
        if mode in self.rotation_memory:
            self.rotation_memory[mode] = float(self.rotationSpin.value())
        self._update_overlay()

    def _on_refresh_frame(self):
        idx = max(1, int(self.frameSpin.value())) - 1
        self._load_frame(idx)
        self._apply_initial_params()
        self._update_overlay()

    def _load_frame(self, idx):
        frame_gray = self._fetch_frame_gray(idx)
        if frame_gray is None and idx != 0:
            frame_gray = self._fetch_frame_gray(0)
            idx = 0
        if frame_gray is None:
            QMessageBox.critical(self, "Preview", f"Could not load frame {idx + 1}.")
            return
        self.frame_gray = frame_gray
        h, w = frame_gray.shape
        self.current_frame_idx = idx
        self.frameInfo.setText(f"{idx + 1}/{self.total_frames if self.total_frames > 0 else '?'}")

        self._frame_gray_ref = frame_gray
        qimg = QImage(self._frame_gray_ref.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._qimg_ref = qimg
        pix = QPixmap.fromImage(qimg)
        if self.preview_scale != 1.0:
            pix = pix.scaled(int(w * self.preview_scale), int(h * self.preview_scale), Qt.AspectRatioMode.KeepAspectRatio)
        self.pixmap_item.setPixmap(pix)
        self.scene.setSceneRect(0, 0, pix.width(), pix.height())

    def _clear_overlay(self):
        for item in self.overlay_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items = []

    def _update_overlay(self):
        if self.frame_gray is None:
            return
        self._clear_overlay()
        mode = self.modeCombo.currentText()
        h, w = self.frame_gray.shape
        scale = self.preview_scale

        if mode == "horizontal":
            y = float(self.h_y.value())
            angle = float(self.rotationSpin.value())
            if abs(angle) < 1e-6:
                y_scaled = y * scale
                line = QGraphicsLineItem(0, y_scaled, w * scale, y_scaled)
                line.setPen(QPen(Qt.GlobalColor.red))
                self.scene.addItem(line)
                self.overlay_items.append(line)
            else:
                slope = np.tan(np.deg2rad(angle))
                x0 = 0.0
                x1 = w - 1.0
                center = (w - 1) / 2.0
                y0 = y + (x0 - center) * slope
                y1 = y + (x1 - center) * slope
                line = QGraphicsLineItem(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
                line.setPen(QPen(Qt.GlobalColor.red))
                self.scene.addItem(line)
                self.overlay_items.append(line)

        elif mode == "vertical":
            x = float(self.v_x.value())
            angle = float(self.rotationSpin.value())
            if abs(angle) < 1e-6:
                x_scaled = x * scale
                line = QGraphicsLineItem(x_scaled, 0, x_scaled, h * scale)
                line.setPen(QPen(Qt.GlobalColor.red))
                self.scene.addItem(line)
                self.overlay_items.append(line)
            else:
                slope = np.tan(np.deg2rad(angle))
                y0 = 0.0
                y1 = h - 1.0
                center = (h - 1) / 2.0
                x0 = x + (y0 - center) * slope
                x1 = x + (y1 - center) * slope
                line = QGraphicsLineItem(x0 * scale, y0 * scale, x1 * scale, y1 * scale)
                line.setPen(QPen(Qt.GlobalColor.red))
                self.scene.addItem(line)
                self.overlay_items.append(line)

        elif mode == "circular":
            cx = float(self.c_cx.value())
            cy = float(self.c_cy.value())
            r = float(self.c_r.value())
            rect = (cx - r) * scale, (cy - r) * scale, 2 * r * scale, 2 * r * scale
            ellipse = QGraphicsEllipseItem(*rect)
            ellipse.setPen(QPen(Qt.GlobalColor.red))
            self.scene.addItem(ellipse)
            self.overlay_items.append(ellipse)

    def _collect_config(self):
        mode = self.modeCombo.currentText()
        params = {}
        global _last_n_angles
        if mode == "horizontal":
            params = {"y": float(self.h_y.value()), "angle_deg": float(self.rotationSpin.value())}
        elif mode == "vertical":
            params = {"x": float(self.v_x.value()), "angle_deg": float(self.rotationSpin.value())}
        elif mode == "circular":
            try:
                cx = int(self.c_cx.value())
                cy = int(self.c_cy.value())
                r = int(self.c_r.value())
                free_mode = bool(self.c_free.isChecked())
                n_angles = int(self.c_n.value())
                if free_mode:
                    r_extract = int(r)
                    outer_r = None
                else:
                    outer_r = int(r)
                    r_extract = int(round(outer_r * (2.0 / 3.0)))
                params = {"cx": cx, "cy": cy, "r": r_extract}
                _last_n_angles = n_angles
            except Exception as e:
                raise ValueError(f"Invalid circular parameters: {e}")
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        rd = dict(self.remember_defaults)
        if mode == "horizontal":
            rd["horizontal"] = {"y": float(params["y"]), "angle_deg": float(params.get("angle_deg", 0.0))}
        elif mode == "vertical":
            rd["vertical"] = {"x": float(params["x"]), "angle_deg": float(params.get("angle_deg", 0.0))}
        elif mode == "circular":
            circ_prev = dict(self.remember_defaults.get("circular", {}))
            circ_prev["cx"] = int(params["cx"])
            circ_prev["cy"] = int(params["cy"])
            circ_prev["n_angles"] = int(self.c_n.value())
            circ_prev["free"] = bool(self.c_free.isChecked())
            if self.c_free.isChecked():
                circ_prev["r"] = int(params["r"])
            else:
                circ_prev["outer_r"] = int(self.c_r.value())
                circ_prev["r"] = int(params["r"])
            rd["circular"] = circ_prev
        self.remember_defaults = rd

        config = {
            "mode": mode,
            "params": params,
            "remember": self.remember_defaults,
        }
        if mode == "circular":
            config["n_angles"] = int(self.c_n.value())
        return config

    def _persist_config_for(self, path, config):
        base_no_ext, _ = os.path.splitext(path)
        per_video_config_path = base_no_ext + "_st_config.json"
        folder = os.path.dirname(path)
        legacy_config_path = os.path.join(folder, "preview_config.json")
        with open(per_video_config_path, "w") as f:
            json.dump(config, f, indent=4)
        try:
            with open(legacy_config_path, "w") as f:
                json.dump(config, f, indent=4)
        except Exception as e:
            print(f"Warning: could not update legacy preview_config.json: {e}")

    def _on_engage(self):
        try:
            config = self._collect_config()
        except Exception as e:
            QMessageBox.critical(self, "Input error", str(e))
            return

        try:
            for vid in self.target_videos:
                self._persist_config_for(vid, config)
        except Exception as e:
            QMessageBox.critical(self, "File error", f"Could not save config: {e}")
            return

        self.btnEngage.setEnabled(False)
        self.progressLabel.setText("Starting...")
        self.progressBar.setValue(0)

        self.worker = ExtractionWorker(self.target_videos, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, done, total, msg):
        self.progressLabel.setText(msg)
        if total and total > 0:
            pct = max(0.0, min(100.0, 100.0 * float(done) / float(total)))
            self.progressBar.setValue(int(pct))
        else:
            self.progressBar.setRange(0, 0)

    def _on_finished(self):
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(100)
        self.progressLabel.setText("Done.")
        self.btnEngage.setEnabled(True)
        QMessageBox.information(self, "Done", "Extraction completed and saved.")

    def _on_failed(self, msg):
        self.progressBar.setRange(0, 100)
        self.progressLabel.setText(f"Error: {msg}")
        self.btnEngage.setEnabled(True)
        QMessageBox.critical(self, "Error", msg)


def main():
    if len(sys.argv) < 2:
        print("Usage: python PLI_extract_st_diag_qt.py /path/to/video [extra_videos…]")
        sys.exit(1)
    app = QApplication(sys.argv)
    win = ExtractorWindow(sys.argv[1:])
    win.resize(1000, 800)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
