from __future__ import annotations

import os
import json
import cv2
import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import (
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
    QFrame,
)

from _Routines.PLI.PLI_extract_st_diag_qt import process_video_with_config


_last_n_angles = 360


def load_config_for_video(video_path: str) -> dict | None:
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


def persist_config_for(path: str, config: dict) -> None:
    base_no_ext, _ = os.path.splitext(path)
    per_video_config_path = base_no_ext + "_st_config.json"
    folder = os.path.dirname(path)
    legacy_config_path = os.path.join(folder, "preview_config.json")
    with open(per_video_config_path, "w") as f:
        json.dump(config, f, indent=4)
    try:
        with open(legacy_config_path, "w") as f:
            json.dump(config, f, indent=4)
    except Exception:
        pass


class ExtractionWorker(QThread):
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self, videos: list[str], config: dict):
        super().__init__()
        self.videos = videos
        self.config = config

    def run(self) -> None:
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


class PLIExtractPreview(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.scene = QGraphicsScene(self)
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setFrameShape(QFrame.Shape.NoFrame)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        layout.addWidget(self.view)
        self.frame_gray: np.ndarray | None = None
        self._frame_gray_ref = None
        self._qimg_ref = None
        self.overlay_items: list = []

    def set_frame(self, frame_gray: np.ndarray) -> None:
        self.frame_gray = frame_gray
        h, w = frame_gray.shape
        self._frame_gray_ref = frame_gray
        qimg = QImage(self._frame_gray_ref.data, w, h, w, QImage.Format.Format_Grayscale8)
        self._qimg_ref = qimg
        pix = QPixmap.fromImage(qimg)
        self.pixmap_item.setPixmap(pix)
        self.scene.setSceneRect(self.pixmap_item.boundingRect())
        self._fit_view()
        QTimer.singleShot(0, self._fit_view)

    def ensure_fit(self) -> None:
        self._fit_view()
        QTimer.singleShot(0, self._fit_view)

    def clear_overlay(self) -> None:
        for item in self.overlay_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items = []

    def update_overlay(self, mode: str, params: dict, rotation: float) -> None:
        if self.frame_gray is None:
            return
        self.clear_overlay()
        h, w = self.frame_gray.shape

        if mode == "horizontal":
            y = float(params.get("y", h // 2))
            angle = float(rotation)
            if abs(angle) < 1e-6:
                line = QGraphicsLineItem(0, y, w, y)
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
                line = QGraphicsLineItem(x0, y0, x1, y1)
                line.setPen(QPen(Qt.GlobalColor.red))
                self.scene.addItem(line)
                self.overlay_items.append(line)

        elif mode == "vertical":
            x = float(params.get("x", w // 2))
            angle = float(rotation)
            if abs(angle) < 1e-6:
                line = QGraphicsLineItem(x, 0, x, h)
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
                line = QGraphicsLineItem(x0, y0, x1, y1)
                line.setPen(QPen(Qt.GlobalColor.red))
                self.scene.addItem(line)
                self.overlay_items.append(line)

        elif mode == "circular":
            cx = float(params.get("cx", w // 2))
            cy = float(params.get("cy", h // 2))
            r = float(params.get("r", min(w, h) // 3))
            rect = (cx - r), (cy - r), 2 * r, 2 * r
            ellipse = QGraphicsEllipseItem(*rect)
            ellipse.setPen(QPen(Qt.GlobalColor.red))
            self.scene.addItem(ellipse)
            self.overlay_items.append(ellipse)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_view()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.ensure_fit()

    def _fit_view(self) -> None:
        if self.frame_gray is None:
            return
        self.view.resetTransform()
        rect = self.pixmap_item.boundingRect()
        if rect.isNull():
            return
        viewport = self.view.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        scale_x = viewport.width() / rect.width()
        scale_y = viewport.height() / rect.height()
        scale = min(scale_x, scale_y)
        if scale <= 0:
            return
        self.view.scale(scale, scale)
        self.view.centerOn(self.pixmap_item)


class PLIExtractControls(QWidget):
    def __init__(self, total_frames: int, parent: QWidget | None = None):
        super().__init__(parent)
        self.total_frames = total_frames
        self.remember_defaults: dict = {}
        self.rotation_memory: dict = {"horizontal": 0.0, "vertical": 0.0}
        self.initial_params: dict = {}
        self.initial_mode: str = "horizontal"
        self.initial_n_angles: int = _last_n_angles

        self._build_ui()

    # --- helpers for spin/slider pairs ---
    def _bind_slider(self, slider: QSlider, spin: QSpinBox | QDoubleSpinBox, *, scale: float = 1.0) -> None:
        def spin_to_slider(val):
            try:
                slider.blockSignals(True)
                slider.setValue(int(round(float(val) * scale)))
            finally:
                slider.blockSignals(False)

        def slider_to_spin(val):
            try:
                spin.blockSignals(True)
                # QSpinBox requires int; QDoubleSpinBox accepts float.
                new_val = val / scale
                if isinstance(spin, QSpinBox):
                    spin.setValue(int(round(new_val)))
                else:
                    spin.setValue(new_val)
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

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        topbar = QHBoxLayout()
        self.btnEngage = QPushButton("ENGAGE")
        topbar.addWidget(self.btnEngage)
        topbar.addStretch(1)
        root.addLayout(topbar)

        self.progressLabel = QLabel("")
        self.progressBar = QProgressBar()
        self.progressBar.setRange(0, 100)
        self.progressBar.setValue(0)
        root.addWidget(self.progressLabel)
        root.addWidget(self.progressBar)

        mode_row = QHBoxLayout()
        mode_row.addWidget(QLabel("Mode:"))
        self.modeCombo = QComboBox()
        self.modeCombo.addItems(["horizontal", "vertical", "circular"])
        mode_row.addWidget(self.modeCombo)
        mode_row.addStretch(1)
        root.addLayout(mode_row)

        frame_row = QHBoxLayout()
        frame_row.addWidget(QLabel("Preview frame:"))
        self.frameSpin = QSpinBox()
        self.frameSpin.setMinimum(1)
        max_frame = max(1, self.total_frames)
        self.frameSpin.setMaximum(max_frame)
        self.frameSpin.setValue(min(10, max_frame))
        frame_row.addWidget(self.frameSpin)
        self.btnRefreshFrame = QPushButton("Refresh preview")
        frame_row.addWidget(self.btnRefreshFrame)
        self.frameInfo = QLabel("")
        frame_row.addWidget(self.frameInfo)
        frame_row.addStretch(1)
        root.addLayout(frame_row)

        rot_grid = QGridLayout()
        rot_grid.setHorizontalSpacing(8)
        rot_grid.setVerticalSpacing(4)
        rot_grid.addWidget(QLabel("Rotation (deg):"), 0, 0)
        self.rotationSpin = QDoubleSpinBox()
        self.rotationSpin.setRange(-60.0, 60.0)
        self.rotationSpin.setDecimals(1)
        self.rotationSpin.setSingleStep(0.5)
        rot_grid.addWidget(self.rotationSpin, 0, 1)
        self.rotationSlider = QSlider(Qt.Orientation.Horizontal)
        self.rotationSlider.setSingleStep(1)
        self.rotationSlider.setPageStep(5)
        self.rotationSlider.setMinimumWidth(200)
        self._bind_slider(self.rotationSlider, self.rotationSpin, scale=10.0)
        self._set_slider_range(self.rotationSlider, -60.0, 60.0, scale=10.0)
        rot_grid.addWidget(self.rotationSlider, 1, 0, 1, 3)
        rot_grid.setColumnStretch(2, 1)
        root.addLayout(rot_grid)

        self.controlsStack = QStackedWidget()
        self._build_horizontal_controls()
        self._build_vertical_controls()
        self._build_circular_controls()
        root.addWidget(self.controlsStack)

        self.modeCombo.currentTextChanged.connect(self._on_mode_changed)

    def _build_horizontal_controls(self) -> None:
        box = QGroupBox("Horizontal")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        self.h_y = QSpinBox()
        self.h_y.setMinimum(0)
        self.h_y_slider = QSlider(Qt.Orientation.Horizontal)
        self.h_y_slider.setSingleStep(1)
        self.h_y_slider.setPageStep(5)
        self.h_y_slider.setMinimumWidth(200)
        self._bind_slider(self.h_y_slider, self.h_y)
        layout.addWidget(QLabel("Row (y):"), 0, 0)
        layout.addWidget(self.h_y, 0, 1)
        layout.addWidget(self.h_y_slider, 1, 0, 1, 3)
        layout.setColumnStretch(2, 1)
        self.controlsStack.addWidget(box)

    def _build_vertical_controls(self) -> None:
        box = QGroupBox("Vertical")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        self.v_x = QSpinBox()
        self.v_x.setMinimum(0)
        self.v_x_slider = QSlider(Qt.Orientation.Horizontal)
        self.v_x_slider.setSingleStep(1)
        self.v_x_slider.setPageStep(5)
        self.v_x_slider.setMinimumWidth(200)
        self._bind_slider(self.v_x_slider, self.v_x)
        layout.addWidget(QLabel("Column (x):"), 0, 0)
        layout.addWidget(self.v_x, 0, 1)
        layout.addWidget(self.v_x_slider, 1, 0, 1, 3)
        layout.setColumnStretch(2, 1)
        self.controlsStack.addWidget(box)

    def _build_circular_controls(self) -> None:
        box = QGroupBox("Circular")
        layout = QGridLayout(box)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        self.c_cx = QSpinBox()
        self.c_cy = QSpinBox()
        self.c_r = QSpinBox()
        self.c_n = QSpinBox()
        self.c_free = QCheckBox("Free radius (no outer)")
        self.c_n.setMinimum(10)
        self.c_n.setMaximum(5000)
        self.c_cx_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_cy_slider = QSlider(Qt.Orientation.Horizontal)
        self.c_r_slider = QSlider(Qt.Orientation.Horizontal)
        for s in (self.c_cx_slider, self.c_cy_slider, self.c_r_slider):
            s.setSingleStep(1)
            s.setPageStep(10)
            s.setMinimumWidth(200)
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
        self.controlsStack.addWidget(box)

    def set_frame_bounds(self, h: int, w: int) -> None:
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

    def set_total_frames(self, total_frames: int) -> None:
        self.total_frames = total_frames
        max_frame = max(1, self.total_frames)
        self.frameSpin.setMaximum(max_frame)
        if self.frameSpin.value() > max_frame:
            self.frameSpin.setValue(max_frame)

    def apply_initial_config(self, config: dict | None, frame_shape: tuple[int, int]) -> None:
        if config is None:
            config = {}
        self.remember_defaults = config.get("remember", {})
        self.initial_mode = config.get("mode", "horizontal")
        self.initial_params = config.get("params", {})
        self.initial_n_angles = int(config.get("n_angles", _last_n_angles))

        h, w = frame_shape
        mode = self.initial_mode
        params = dict(self.initial_params)
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

        rot = float(self.remember_defaults.get(mode, {}).get("angle_deg", params.get("angle_deg", 0.0)))
        self.rotationSpin.setValue(rot)
        self._set_slider_range(self.rotationSlider, -60.0, 60.0, scale=10.0)
        self.modeCombo.setCurrentText(mode)

    def _on_mode_changed(self, mode: str) -> None:
        idx = {"horizontal": 0, "vertical": 1, "circular": 2}.get(mode, 0)
        self.controlsStack.setCurrentIndex(idx)
        self.rotationSpin.setEnabled(mode in {"horizontal", "vertical"})

    def current_mode(self) -> str:
        return self.modeCombo.currentText()

    def current_rotation(self) -> float:
        return float(self.rotationSpin.value())

    def current_params(self) -> dict:
        mode = self.current_mode()
        if mode == "horizontal":
            return {"y": float(self.h_y.value()), "angle_deg": self.current_rotation()}
        if mode == "vertical":
            return {"x": float(self.v_x.value()), "angle_deg": self.current_rotation()}
        if mode == "circular":
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
            return {"cx": cx, "cy": cy, "r": r_extract, "n_angles": n_angles, "free": free_mode, "outer_r": outer_r}
        return {}

    def build_config(self) -> dict:
        mode = self.current_mode()
        params = self.current_params()
        global _last_n_angles
        if mode == "circular":
            _last_n_angles = int(params.get("n_angles", _last_n_angles))

        rd = dict(self.remember_defaults)
        if mode == "horizontal":
            rd["horizontal"] = {"y": float(params["y"]), "angle_deg": float(params.get("angle_deg", 0.0))}
        elif mode == "vertical":
            rd["vertical"] = {"x": float(params["x"]), "angle_deg": float(params.get("angle_deg", 0.0))}
        elif mode == "circular":
            circ_prev = dict(self.remember_defaults.get("circular", {}))
            circ_prev["cx"] = int(params["cx"])
            circ_prev["cy"] = int(params["cy"])
            circ_prev["n_angles"] = int(params.get("n_angles", _last_n_angles))
            circ_prev["free"] = bool(params.get("free", False))
            if params.get("free"):
                circ_prev["r"] = int(params["r"])
            else:
                circ_prev["outer_r"] = int(params.get("outer_r") or params["r"])
                circ_prev["r"] = int(params["r"])
            rd["circular"] = circ_prev
        self.remember_defaults = rd

        config = {
            "mode": mode,
            "params": {k: v for k, v in params.items() if k not in {"n_angles", "free", "outer_r"}},
            "remember": self.remember_defaults,
        }
        if mode == "circular":
            config["n_angles"] = int(params.get("n_angles", _last_n_angles))
        return config


class PLIExtractController:
    def __init__(self, video_paths: list[str], preview: PLIExtractPreview, controls: PLIExtractControls):
        self.preview = preview
        self.controls = controls
        self.on_done = None
        self.target_videos = self._normalize_videos(video_paths)
        if not self.target_videos:
            raise FileNotFoundError("No valid video files provided for extraction.")
        self.preview_video = max(self.target_videos, key=lambda p: os.path.basename(p).lower())
        self.total_frames = self._get_total_frames(self.preview_video)
        self.controls.set_total_frames(self.total_frames)

        self.controls.btnRefreshFrame.clicked.connect(self._on_refresh_frame)
        self.controls.modeCombo.currentTextChanged.connect(self._update_overlay)
        self.controls.h_y.valueChanged.connect(self._update_overlay)
        self.controls.v_x.valueChanged.connect(self._update_overlay)
        self.controls.c_cx.valueChanged.connect(self._update_overlay)
        self.controls.c_cy.valueChanged.connect(self._update_overlay)
        self.controls.c_r.valueChanged.connect(self._update_overlay)
        self.controls.c_n.valueChanged.connect(self._update_overlay)
        self.controls.c_free.stateChanged.connect(self._update_overlay)
        self.controls.rotationSpin.valueChanged.connect(self._update_overlay)
        self.controls.rotationSlider.valueChanged.connect(self._update_overlay)
        self.controls.h_y_slider.valueChanged.connect(self._update_overlay)
        self.controls.v_x_slider.valueChanged.connect(self._update_overlay)
        self.controls.c_cx_slider.valueChanged.connect(self._update_overlay)
        self.controls.c_cy_slider.valueChanged.connect(self._update_overlay)
        self.controls.c_r_slider.valueChanged.connect(self._update_overlay)
        self.controls.btnEngage.clicked.connect(self._on_engage)

        self._load_frame(min(9, max(self.total_frames - 1, 0)) if self.total_frames > 0 else 0)
        if self.preview.frame_gray is not None:
            h, w = self.preview.frame_gray.shape
            self.controls.set_frame_bounds(h, w)
            loaded = load_config_for_video(self.preview_video)
            self.controls.apply_initial_config(loaded, (h, w))
        self._update_overlay()

    def _normalize_videos(self, raw_paths) -> list[str]:
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

    def _get_total_frames(self, video_path: str) -> int:
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

    def _load_frame(self, idx: int) -> None:
        frame_gray = self._fetch_frame_gray(idx)
        if frame_gray is None and idx != 0:
            frame_gray = self._fetch_frame_gray(0)
            idx = 0
        if frame_gray is None:
            QMessageBox.critical(self.controls, "Preview", f"Could not load frame {idx + 1}.")
            return
        self.preview.set_frame(frame_gray)
        self.preview.ensure_fit()
        total = self.total_frames if self.total_frames > 0 else "?"
        self.controls.frameInfo.setText(f"{idx + 1}/{total}")

    def _on_refresh_frame(self) -> None:
        idx = max(1, int(self.controls.frameSpin.value())) - 1
        self._load_frame(idx)
        if self.preview.frame_gray is not None:
            h, w = self.preview.frame_gray.shape
            self.controls.set_frame_bounds(h, w)
        self._update_overlay()

    def _update_overlay(self) -> None:
        if self.preview.frame_gray is None:
            return
        params = self.controls.current_params()
        mode = self.controls.current_mode()
        rotation = self.controls.current_rotation()
        self.preview.update_overlay(mode, params, rotation)

    def _on_engage(self) -> None:
        try:
            config = self.controls.build_config()
        except Exception as e:
            QMessageBox.critical(self.controls, "Input error", str(e))
            return

        try:
            for vid in self.target_videos:
                persist_config_for(vid, config)
        except Exception as e:
            QMessageBox.critical(self.controls, "File error", f"Could not save config: {e}")
            return

        self.controls.btnEngage.setEnabled(False)
        self.controls.progressLabel.setText("Starting...")
        self.controls.progressBar.setValue(0)

        self.worker = ExtractionWorker(self.target_videos, config)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_finished)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, done: int, total: int, msg: str) -> None:
        self.controls.progressLabel.setText(msg)
        if total and total > 0:
            pct = max(0.0, min(100.0, 100.0 * float(done) / float(total)))
            self.controls.progressBar.setValue(int(pct))
        else:
            self.controls.progressBar.setRange(0, 0)

    def _on_finished(self) -> None:
        self.controls.progressBar.setRange(0, 100)
        self.controls.progressBar.setValue(100)
        self.controls.progressLabel.setText("Done.")
        self.controls.btnEngage.setEnabled(True)
        if callable(self.on_done):
            try:
                self.on_done()
            except Exception:
                pass

    def _on_failed(self, msg: str) -> None:
        self.controls.progressBar.setRange(0, 100)
        self.controls.progressLabel.setText(f"Error: {msg}")
        self.controls.btnEngage.setEnabled(True)
        QMessageBox.critical(self.controls, "Error", msg)
        if callable(self.on_done):
            try:
                self.on_done()
            except Exception:
                pass
