from __future__ import annotations

import os
import sys
from pathlib import Path
import json
import numpy as np
import cv2
import csv
from _Core.paths import get_processed_root
from _Core.params import save_params, load_params
from _UI_modules.PLI_helper import get_reference_folder_from_path, _sanitize_misjoined_user_path

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QImage, QPixmap, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QDoubleSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QFrame,
    QDialogButtonBox,
    QSlider,
)

# Ensure project root is on path so _Routines imports resolve when running this module directly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from _Routines.PLI import PLI_engine as engine

DEFAULT_MANUAL_INTERVALS = [
    2500, 2500, 2500,
    1875, 1875, 1875, 1875, 1875, 1875, 1875, 1875,
    1250, 1250, 1250, 1250, 1250,
    1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000,
    500, 500, 500, 500, 500,
]


def _durations_to_intervals(durations, gap_s=0.0):
    pairs = []
    t = 0.0
    for d in durations or []:
        try:
            d_val = max(0.0, float(d))
        except Exception:
            d_val = 0.0
        a = t
        b = a + d_val
        if b > a:
            pairs.append((a, b))
        t = b + max(0.0, float(gap_s or 0.0))
    return pairs


class PLIRescalePreview(QWidget):
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
        self._rgb_ref = None
        self._qimg_ref = None
        self.overlay_items: list = []
        self._first_fit_done = False
        self._tail_mode = False  # flag toggled by controller when cropping tail-only previews
        self._scale = 1.0
        self._src_shape = (0, 0)
        self._end_abs_x: int | None = None  # absolute x (pixels) chosen in tail preview

    def set_rgb(self, rgb: np.ndarray) -> None:
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
        if rgb.shape[2] == 3:
            h, w = rgb.shape[:2]
            self._src_shape = (h, w)
            # Downscale for display if very large to keep slider responsive
            target_max = 2000  # px
            scale = min(1.0, float(target_max) / float(max(h, w)) if max(h, w) > 0 else 1.0)
            self._scale = scale
            if scale < 0.999:
                disp = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
            else:
                disp = rgb
            self._rgb_ref = disp
            dh, dw = disp.shape[:2]
            qimg = QImage(disp.data, dw, dh, 3 * dw, QImage.Format.Format_RGB888)
            self._qimg_ref = qimg
            pix = QPixmap.fromImage(qimg)
            if pix.isNull():
                print("[PLI][Rescale] set_rgb: pixmap is null")
            self.pixmap_item.setPixmap(pix)
            self.scene.setSceneRect(self.pixmap_item.boundingRect())
            self.ensure_fit()

    def clear_overlay(self) -> None:
        for item in self.overlay_items:
            try:
                self.scene.removeItem(item)
            except Exception:
                pass
        self.overlay_items = []

    def add_vertical_line(self, x: float, color: Qt.GlobalColor) -> None:
        rect = self.pixmap_item.boundingRect()
        sx = float(self._scale) if self._scale else 1.0
        x_disp = x * sx
        line = QGraphicsLineItem(x_disp, rect.top(), x_disp, rect.bottom())
        line.setPen(QPen(color))
        self.scene.addItem(line)
        self.overlay_items.append(line)

    def ensure_fit(self) -> None:
        self._fit_view()
        QTimer.singleShot(0, self._fit_view)

    def set_tail_mode(self, enabled: bool) -> None:
        """Toggle tail-mode visual state (no-op placeholder for compatibility)."""
        self._tail_mode = bool(enabled)
        # Optional hint: dim background when tail-cropping so users see mode.
        if self._tail_mode:
            self.view.setStyleSheet("background: #0f111a;")
        else:
            self.view.setStyleSheet("")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_view()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        self.ensure_fit()

    def _fit_view(self) -> None:
        self.view.resetTransform()
        rect = self.pixmap_item.boundingRect()
        if rect.isNull():
            return
        viewport = self.view.viewport().size()
        if viewport.width() <= 0 or viewport.height() <= 0:
            return
        # Preserve aspect ratio: fit within viewport using min scale.
        scale_x = viewport.width() / rect.width()
        scale_y = viewport.height() / rect.height()
        scale = min(scale_x, scale_y)
        if scale <= 0:
            return
        self.view.scale(scale, scale)
        self.view.centerOn(self.pixmap_item)
        self._first_fit_done = True


class PLIRescaleControls(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._manual_intervals = _durations_to_intervals(DEFAULT_MANUAL_INTERVALS, gap_s=0.0)
        self._time_col_override: dict[str, int] = {}
        self._last_time_source_path: str | None = None
        self._fps = 29.97
        self._build_ui()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # Controls row
        top = QHBoxLayout()
        self.fpsSpin = QDoubleSpinBox()
        self.fpsSpin.setRange(0.1, 240.0)
        self.fpsSpin.setDecimals(2)
        self.fpsSpin.setValue(29.97)
        top.addWidget(self.fpsSpin)

        top.addWidget(QLabel("Scaled height (px):"))
        self.scaledHeightSpin = QSpinBox()
        self.scaledHeightSpin.setRange(1, 50000)
        self.scaledHeightSpin.setValue(350)
        top.addWidget(self.scaledHeightSpin)

        top.addWidget(QLabel("Aspect ratio W/H:"))
        self.aspectSpin = QDoubleSpinBox()
        self.aspectSpin.setRange(0.1, 20.0)
        self.aspectSpin.setDecimals(3)
        self.aspectSpin.setValue(1.426)
        top.addWidget(self.aspectSpin)

        top.addWidget(QLabel("Interlude (px):"))
        self.interludeSpin = QSpinBox()
        self.interludeSpin.setRange(0, 100000)
        self.interludeSpin.setValue(0)
        top.addWidget(self.interludeSpin)

        top.addStretch(1)
        root.addLayout(top)

        # Mode selection
        mode_box = QGroupBox("Steady-state interval definition mode")
        mode_layout = QVBoxLayout(mode_box)
        self.rbTriggered = QRadioButton("Triggered steady-state intervals")
        self.rbReference = QRadioButton("Steady-state intervals not triggered — reference rheology file")
        self.rbManual = QRadioButton("Steady-state intervals not triggered — manual entry")
        self.rbTriggered.setChecked(True)
        mode_layout.addWidget(self.rbTriggered)
        mode_layout.addWidget(self.rbReference)
        mode_layout.addWidget(self.rbManual)
        root.addWidget(mode_box)

        self.modeGroup = QButtonGroup(self)
        self.modeGroup.addButton(self.rbTriggered)
        self.modeGroup.addButton(self.rbReference)
        self.modeGroup.addButton(self.rbManual)

        # File controls
        file_row = QHBoxLayout()
        self.btnLoadTime = QPushButton("Load time source…")
        self.btnRefreshIntervals = QPushButton("Refresh intervals")
        self.btnPickColumn = QPushButton("Pick time column…")
        file_row.addWidget(self.btnLoadTime)
        file_row.addWidget(self.btnRefreshIntervals)
        file_row.addWidget(self.btnPickColumn)
        file_row.addStretch(1)
        root.addLayout(file_row)

        # Reference options
        ref_row = QHBoxLayout()
        ref_row.addWidget(QLabel("Steady state width (s):"))
        self.steadySecSpin = QDoubleSpinBox()
        self.steadySecSpin.setRange(0.0, 1_000_000.0)
        self.steadySecSpin.setDecimals(2)
        self.steadySecSpin.setValue(10.0)
        ref_row.addWidget(self.steadySecSpin)
        ref_row.addStretch(1)
        self.refRow = ref_row
        root.addLayout(ref_row)

        # Manual options
        manual_row = QHBoxLayout()
        manual_row.addWidget(QLabel("Number of steady-state intervals:"))
        self.manualCountSpin = QSpinBox()
        self.manualCountSpin.setRange(1, 50)
        self.manualCountSpin.setValue(max(1, len(self._manual_intervals)))
        manual_row.addWidget(self.manualCountSpin)
        self.btnUpdateManual = QPushButton("Update rows")
        manual_row.addWidget(self.btnUpdateManual)
        manual_row.addWidget(QLabel("Gap between intervals (s):"))
        self.manualGapSpin = QDoubleSpinBox()
        self.manualGapSpin.setRange(0.0, 100000.0)
        self.manualGapSpin.setDecimals(2)
        self.manualGapSpin.setValue(1.0)
        manual_row.addWidget(self.manualGapSpin)
        manual_row.addWidget(QLabel("Steady window (s):"))
        self.manualSteadySpin = QDoubleSpinBox()
        self.manualSteadySpin.setRange(0.0, 100000.0)
        self.manualSteadySpin.setDecimals(2)
        self.manualSteadySpin.setValue(1.0)
        manual_row.addWidget(self.manualSteadySpin)
        manual_row.addStretch(1)
        self.manualRow = manual_row
        root.addLayout(manual_row)

        manual_sum = QHBoxLayout()
        manual_sum.addWidget(QLabel("Total steady width (px @ fps):"))
        self.manualTotalPx = QLineEdit("0")
        self.manualTotalPx.setReadOnly(True)
        manual_sum.addWidget(self.manualTotalPx)
        manual_sum.addWidget(QLabel("Total w/ gaps (px):"))
        self.manualTotalWithGaps = QLineEdit("0")
        self.manualTotalWithGaps.setReadOnly(True)
        manual_sum.addWidget(self.manualTotalWithGaps)
        manual_sum.addStretch(1)
        self.manualSummaryRow = manual_sum
        root.addLayout(manual_sum)

        # End-region preview controls (to aid end-of-image selection)
        # End-region controls on two rows for full-width slider/button
        end_row_top = QHBoxLayout()
        end_row_top.addWidget(QLabel("Show last (%):"))
        self.endPctSpin = QDoubleSpinBox()
        self.endPctSpin.setRange(0.1, 100.0)
        self.endPctSpin.setDecimals(1)
        self.endPctSpin.setValue(5.0)
        end_row_top.addWidget(self.endPctSpin)
        end_row_top.addWidget(QLabel("End x (px):"))
        self.endXSpin = QSpinBox()
        self.endXSpin.setRange(0, 1_000_000_000)
        self.endXSpin.setValue(0)
        end_row_top.addWidget(self.endXSpin)
        self.btnEndApply = QPushButton("Apply")
        self.btnEndApply.setToolTip("Update preview with current End x")
        end_row_top.addWidget(self.btnEndApply)
        end_row_top.addStretch(1)
        root.addLayout(end_row_top)

        end_row_bottom = QHBoxLayout()
        self.endSlider = QSlider(Qt.Orientation.Horizontal)
        self.endSlider.setRange(0, 0)
        self.endSlider.setSingleStep(1)
        self.endSlider.setPageStep(10)
        # Avoid spamming valueChanged while dragging; apply on release
        self.endSlider.setTracking(False)
        self.endSlider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        end_row_bottom.addWidget(self.endSlider, 1)
        self.btnEndPreview = QPushButton("Reference test end")
        self.btnEndPreview.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        end_row_bottom.addWidget(self.btnEndPreview)
        root.addLayout(end_row_bottom)

        # Accept/engage row placed above the intervals table for quicker access
        btn_row = QHBoxLayout()
        self.btnPreview = QPushButton("PREVIEW intervals")
        self.btnEngage = QPushButton("ACCEPT (preview + save)")
        btn_row.addWidget(self.btnPreview)
        btn_row.addWidget(self.btnEngage)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Intervals table
        self.table = QTableWidget()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        root.addWidget(self.table, 1)

        self._toggle_mode_ui()
        self.modeGroup.buttonClicked.connect(self._toggle_mode_ui)

    def _toggle_mode_ui(self) -> None:
        is_ref = self.rbReference.isChecked()
        is_manual = self.rbManual.isChecked()
        self.steadySecSpin.setEnabled(is_ref)
        for w in (
            self.manualCountSpin,
            self.btnUpdateManual,
            self.manualGapSpin,
            self.manualSteadySpin,
        ):
            w.setEnabled(is_manual)
        self.manualTotalPx.setEnabled(is_manual)
        self.manualTotalWithGaps.setEnabled(is_manual)

    def current_mode(self) -> str:
        if self.rbReference.isChecked():
            return "Reference"
        if self.rbManual.isChecked():
            return "Manual"
        return "Triggered"


class PLIRescaleController:
    def __init__(self, image_paths: list[str], preview: PLIRescalePreview, controls: PLIRescaleControls):
        self.preview = preview
        self.controls = controls
        self.image_paths = self._normalize_images(image_paths)
        self.reference_folder = self._resolve_reference_folder()
        self._active_rgb = None
        self._end_offset_px = 0  # distance from right edge (px)
        self._tail_mode = False   # switch to tail crop only after "Reference test end"
        self._has_shown_initial_fit = False

        # Button wiring with debug wrappers so we see failures in the console
        self.controls.btnLoadTime.clicked.connect(self._debug_wrap(self._on_load_time_source, "btnLoadTime"))
        self.controls.btnRefreshIntervals.clicked.connect(self._debug_wrap(self._refresh_intervals, "btnRefreshIntervals"))
        self.controls.btnPickColumn.clicked.connect(self._debug_wrap(self._pick_time_column, "btnPickColumn"))
        self.controls.btnUpdateManual.clicked.connect(self._debug_wrap(self._refresh_intervals, "btnUpdateManual"))
        self.controls.btnPreview.clicked.connect(self._debug_wrap(self._on_preview, "btnPreview"))
        self.controls.btnEngage.clicked.connect(self._debug_wrap(self._on_engage, "btnEngage"))
        self.controls.btnEndPreview.clicked.connect(self._debug_wrap(self._on_end_preview, "btnEndPreview"))
        self.controls.btnEndApply.clicked.connect(self._debug_wrap(self._on_end_preview, "btnEndApply"))
        self.controls.fpsSpin.valueChanged.connect(self._sync_fps)
        self.controls.interludeSpin.valueChanged.connect(self._on_interlude_changed)
        self.controls.endSlider.valueChanged.connect(self._debug_wrap(lambda: self._on_end_slider_changed(self.controls.endSlider.value()), "endSliderChanged"))
        self.controls.endXSpin.valueChanged.connect(self._debug_wrap(lambda: self._on_end_spin_changed(self.controls.endXSpin.value()), "endSpinChanged"))
        self.controls.endPctSpin.valueChanged.connect(self._debug_wrap(lambda: self._on_end_percent_changed(self.controls.endPctSpin.value()), "endPctChanged"))

        self._load_first_image()
        self._auto_load_single_rheology()
        self._refresh_intervals()
        # On first load, show full image shrunk to fit
        self._show_full_image_once()

    def _show_full_image_once(self):
        if self._has_shown_initial_fit:
            return
        data = self._prepare_image_data()
        if data is None:
            return
        if self.preview is not None:
            self.preview.set_rgb(data["rgb"])
            self.preview.clear_overlay()
            self.preview.ensure_fit()
        self._has_shown_initial_fit = True

    def _debug_wrap(self, fn, label):
        def _wrapped():
            print(f"[PLI][Rescale] {label} clicked")
            try:
                fn()
            except Exception as e:
                print(f"[PLI][Rescale] {label} handler error: {e}")
                try:
                    QMessageBox.critical(self.controls, "Error", f"{label} failed:\\n{e}")
                except Exception:
                    pass
        return _wrapped

    def _normalize_images(self, raw_paths) -> list[str]:
        if isinstance(raw_paths, (list, tuple, set)):
            raw = list(raw_paths)
        else:
            raw = [raw_paths]
        images = []
        seen_paths = set()
        seen_stems = set()
        for entry in raw:
            if not entry:
                continue
            p = os.path.abspath(entry)
            if not os.path.isfile(p):
                continue
            if p in seen_paths:
                continue

            # Collapse trivial duplicates such as "name" vs "name copy" so a single
            # selection cannot accidentally double up in the preview/stitch.
            stem = os.path.splitext(os.path.basename(p))[0].lower().strip()
            stem_key = stem.replace(" copy", "")
            if stem_key in seen_stems:
                print(f"[PLI][Rescale] Dropping duplicate-like selection: {p}")
                continue

            seen_paths.add(p)
            seen_stems.add(stem_key)
            images.append(p)

        # If the caller thought they passed a single image but we still have more than
        # one after normalization, trim to the first. This prevents Qt list selection
        # quirks from rendering multiple previews unintentionally.
        if len(images) > 1:
            print(f"[PLI][Rescale] Multiple images after normalization; using first: {images[0]}")
            images = images[:1]

        print(f"[PLI][Rescale] Images to load: {images}")
        return images

    def _resolve_reference_folder(self) -> str | None:
        # Reference folder is the parent of PLI folder if possible
        for p in self.image_paths:
            parts = p.split(os.sep)
            if "PLI" in parts:
                idx = len(parts) - 1 - list(reversed(parts)).index("PLI")
                return os.sep.join(parts[:idx])
        return os.path.dirname(self.image_paths[0]) if self.image_paths else None

    def _auto_load_single_rheology(self) -> None:
        """If exactly one file exists in <reference>/Rheology, auto-load it as time source."""
        try:
            if self.controls._last_time_source_path:
                return
            base = self.reference_folder or (os.path.dirname(self.image_paths[0]) if self.image_paths else None)
            if not base:
                return
            rheo_dir = os.path.join(base, "Rheology")
            if not os.path.isdir(rheo_dir):
                return
            files = [
                os.path.join(rheo_dir, f)
                for f in os.listdir(rheo_dir)
                if os.path.isfile(os.path.join(rheo_dir, f))
                and f.lower().endswith((".csv", ".tsv", ".txt"))
                and not f.startswith(".")
            ]
            if len(files) != 1:
                return
            path = files[0]
            rows, _prefer_flag, delim = engine.read_csv_rows(path)
            print(f"[PLI][Rescale] Auto-loaded rheology file ({len(rows)} rows, delim={delim!r}): {path}")
            # Auto-detect time column
            try:
                idx = engine.detect_time_column_index(path)
                if idx is not None:
                    self.controls._time_col_override[path] = int(idx)
                    print(f"[PLI][Rescale] Auto-detected time column: {idx}")
            except Exception as e:
                print(f"[PLI][Rescale] detect_time_column_index failed: {e}")
            self.controls._last_time_source_path = path
            self._refresh_intervals()
        except Exception as e:
            print(f"[PLI][Rescale] Auto-load rheology skipped: {e}")

    def _load_first_image(self) -> None:
        if not self.image_paths:
            return
        data = self._prepare_image_data()
        if data is None:
            return
        self._active_rgb = data["rgb"]
        self.preview.set_rgb(data["rgb"])
        self.preview.ensure_fit()
        self._sync_end_slider_max(data["rgb"].shape[1] - 1)

    def _preview_time_source_rows(self, path: str, n_rows: int = 3):
        """Return (header, rows) for quick preview of the time CSV."""
        try:
            rows, _prefer, delim = engine.read_csv_rows(path)
        except Exception:
            rows = []
            delim = ","
        if not rows:
            raise RuntimeError("No rows read from time source.")
        header = rows[0]
        data = rows[1 : 1 + n_rows]
        # If rows are strings (single-column), try simple CSV split as fallback
        if header and isinstance(header, str):
            # Re-parse using csv to split columns
            data_all = []
            with open(path, "r", encoding="utf-8", errors="ignore", newline="") as fh:
                sniffer = csv.Sniffer()
                sample = fh.read(2048)
                fh.seek(0)
                dialect = sniffer.sniff(sample) if sample else csv.excel
                reader = csv.reader(fh, dialect)
                data_all = list(reader)
            if data_all:
                header = data_all[0]
                data = data_all[1 : 1 + n_rows]
        # Normalize to strings
        header = [str(x) for x in header]
        data_norm = []
        for row in data:
            if isinstance(row, (list, tuple)):
                data_norm.append([str(x) for x in row])
            else:
                data_norm.append([str(row)])
        return header, data_norm

    def _on_end_preview(self) -> None:
        """Show a cropped end-region preview directly in the central canvas."""
        data = self._prepare_image_data()
        if data is None:
            return
        self._tail_mode = True
        rgb = data["rgb"]
        H, W = rgb.shape[:2]
        pct = max(0.1, min(100.0, float(self.controls.endPctSpin.value() or 5.0)))
        w0 = int(max(0, round(W * (1.0 - pct / 100.0))))
        view_width = max(1, W - w0)
        max_offset = view_width - 1
        # Slider/spin represent offset within the cropped (tail) region.
        offset = int(self.controls.endXSpin.value() or max_offset)
        offset = max(0, min(offset, max_offset))
        end_abs = w0 + offset  # absolute x in full image
        self._end_offset_px = max(0, max_offset - offset)
        self._end_abs_x = int(end_abs)
        end_x = end_abs

        view = rgb[:, w0:]
        line_x = end_x - w0
        # Downscale to speed up live preview (display only)
        disp_scale = 0.5
        if disp_scale < 0.999:
            view_disp = cv2.resize(view, (int(view.shape[1] * disp_scale), int(view.shape[0] * disp_scale)), interpolation=cv2.INTER_AREA)
            line_disp = int(round(line_x * disp_scale))
        else:
            view_disp = view
            line_disp = line_x

        print(f"[PLI][Rescale] end preview: crop width={view_disp.shape[1]}, height={view_disp.shape[0]}, end_x={end_x}, pct={pct}, scale={disp_scale}")
        self.preview.set_rgb(view_disp)
        self.preview.clear_overlay()
        if 0 <= line_disp < view_disp.shape[1]:
            self.preview.add_vertical_line(line_disp, Qt.GlobalColor.red)
        self.preview.ensure_fit()
        self._sync_end_slider_max(max_offset)

    def _sync_fps(self) -> None:
        self.controls._fps = float(self.controls.fpsSpin.value())
        if self.controls.current_mode() == "Manual":
            self._refresh_intervals()

    def _on_interlude_changed(self) -> None:
        self._load_first_image()
        self._refresh_intervals()

    def _sync_end_slider_max(self, max_x: int) -> None:
        max_x = max(0, int(max_x))
        # Slider/spin represent offset within the cropped tail width
        self.controls.endSlider.setRange(0, max_x)
        cur = int(self.controls.endXSpin.value())
        cur = min(max(cur, 0), max_x)
        self.controls.endXSpin.setValue(cur)
        self.controls.endSlider.setValue(cur)

    def _on_end_slider_changed(self, val: int) -> None:
        print(f"[PLI][Rescale] endSlider changed -> {val}")
        if self.controls.endXSpin.value() != val:
            self.controls.endXSpin.setValue(val)
        # Force tail mode and update preview so the crop responds live.
        self._tail_mode = True
        self._on_end_preview()

    def _on_end_spin_changed(self, val: int) -> None:
        print(f"[PLI][Rescale] endSpin changed -> {val}")
        if self.controls.endSlider.value() != val:
            self.controls.endSlider.setValue(val)
        self._tail_mode = True
        self._on_end_preview()

    def _on_end_percent_changed(self, val: float) -> None:
        print(f"[PLI][Rescale] endPct changed -> {val}")
        # Live refresh of the end preview when tail percentage changes
        self._tail_mode = True
        data = self._prepare_image_data()
        if data is not None:
            W = data["rgb"].shape[1]
            pct = max(0.1, min(100.0, float(val or 5.0)))
            w0 = int(max(0, round(W * (1.0 - pct / 100.0))))
            view_width = max(1, W - w0)
            self._sync_end_slider_max(view_width - 1)
        try:
            self._on_end_preview()
        except Exception:
            pass

    def _prepare_image_data(self) -> dict | None:
        if not self.image_paths:
            return None
        gap_px = int(self.controls.interludeSpin.value() or 0)
        gap_px = max(0, gap_px)

        rgbs = []
        for p in self.image_paths:
            bgr = cv2.imread(p, cv2.IMREAD_COLOR)
            if bgr is None:
                QMessageBox.critical(self.controls, "Image", f"Failed to read image: {p}")
                return None
            rgbs.append(bgr[:, :, ::-1])

        max_h = max(img.shape[0] for img in rgbs)
        padded = []
        for img in rgbs:
            if img.shape[0] == max_h:
                padded.append(img)
            else:
                canvas = np.zeros((max_h, img.shape[1], 3), dtype=img.dtype)
                canvas[:img.shape[0], :img.shape[1], :] = img[:img.shape[0], :img.shape[1], :]
                padded.append(canvas)
        tiles = []
        for idx, arr in enumerate(padded):
            tiles.append(arr)
            if gap_px > 0 and idx < len(padded) - 1:
                tiles.append(np.zeros((max_h, gap_px, 3), dtype=arr.dtype))
        composite = np.hstack(tiles) if tiles else np.zeros((max_h, 1, 3), dtype=np.uint8)
        base = "__".join(os.path.splitext(os.path.basename(p))[0] for p in self.image_paths)
        return {"rgb": composite, "base": base or "PLI_combined"}

    def _collect_manual_intervals(self):
        rows = []
        for i in range(self.controls.table.rowCount()):
            try:
                b = float(self.controls.table.item(i, 1).text())
                e = float(self.controls.table.item(i, 2).text())
            except Exception:
                continue
            if not (np.isfinite(b) and np.isfinite(e)) or e <= b:
                continue
            rows.append((b, e))
        rows.sort(key=lambda t: t[0])
        self.controls._manual_intervals = rows
        return rows

    def _compute_intervals(self):
        mode = self.controls.current_mode()
        fps = float(self.controls.fpsSpin.value())
        path = self.controls._last_time_source_path

        if mode == "Manual":
            intervals = self._collect_manual_intervals()
            if not intervals:
                raise RuntimeError("No manual intervals provided.")
            steady_window_s = float(self.controls.manualSteadySpin.value())
            b_ref = [float(a) for a, _ in intervals]
            e_ref = [float(b) for _, b in intervals]
            if steady_window_s > 0:
                T_beg, T_end, S_beg, S_end = engine.compute_reference_intervals_with_steady(b_ref, e_ref, steady_window_s)
            else:
                T_beg, T_end = b_ref, b_ref
                S_beg, S_end = b_ref, e_ref
            return T_beg, T_end, S_beg, S_end

        if not path or not os.path.isfile(path):
            raise RuntimeError("Load a time source first.")

        if mode == "Triggered":
            T_beg, T_end, S_beg, S_end = engine.extract_triggered_pairs_from_time_column(
                path, override=self.controls._time_col_override
            )
        else:
            b_ref, e_ref = engine.parse_reference_steps_from_csv(path, override=self.controls._time_col_override)
            steady_sec = float(self.controls.steadySecSpin.value())
            if steady_sec > 0:
                T_beg, T_end, S_beg, S_end = engine.compute_reference_intervals_with_steady(b_ref, e_ref, steady_sec)
            else:
                T_beg, T_end = b_ref, b_ref
                S_beg, S_end = b_ref, e_ref
        return T_beg, T_end, S_beg, S_end

    def _on_load_time_source(self) -> None:
        init_dir = self.reference_folder or os.path.expanduser("~")
        print(f"[PLI][Rescale] Load time source clicked (start dir={init_dir})")

        def _pick(parent):
            return QFileDialog.getOpenFileName(
                parent,
                "Select time source CSV file",
                init_dir,
                "CSV files (*.csv);;All files (*.*)",
                options=QFileDialog.Option.DontUseNativeDialog | QFileDialog.Option.ReadOnly,
            )

        try:
            path, _ = _pick(self.controls)
        except Exception as e:
            QMessageBox.critical(self.controls, "File dialog error", str(e))
            print(f"[PLI][Rescale] QFileDialog error: {e}")
            return

        # Fallback: try again with no parent (some WM/sandbox quirks on macOS)
        if not path:
            try:
                path, _ = _pick(None)
            except Exception as e:
                print(f"[PLI][Rescale] QFileDialog fallback error: {e}")

        # Last-resort text prompt so the user can paste a path if dialogs are blocked
        if not path:
            text, ok = QInputDialog.getText(
                self.controls,
                "Time source path",
                "Paste full path to the time-source CSV:",
            )
            if ok and text:
                path = text.strip()

        if not path:
            print("[PLI][Rescale] Load time source cancelled or no file chosen.")
            QMessageBox.information(self.controls, "Load time source", "No file selected.")
            return

        # Normalize and attempt Anton Paar auto-conversion if needed (parity with v5.8.1)
        orig_path = path
        self.controls._last_time_source_path = path
        try:
            rows, _prefer_flag, delim = engine.read_csv_rows(path)
            print(f"[PLI][Rescale] Loaded time source ({len(rows)} rows, delim={delim!r})")
            if delim == "\t":
                try:
                    clean_path = engine.auto_convert_anton_paar(path, header_skip=8)
                    if clean_path and clean_path != path:
                        print(f"[PLI][Rescale] Using converted Anton Paar file: {clean_path}")
                        # propagate any existing override
                        if self.controls._time_col_override.get(path) is not None:
                            self.controls._time_col_override[clean_path] = self.controls._time_col_override[path]
                        path = clean_path
                        self.controls._last_time_source_path = path
                except Exception as e:
                    print(f"[PLI][Rescale] Anton Paar auto-convert failed: {e}")
        except Exception as e:
            QMessageBox.critical(self.controls, "Time source", f"Failed to read CSV: {e}")
            print(f"[PLI][Rescale] read_csv_rows failed: {e}")
            return

        # Attempt auto-detect time column
        try:
            idx = engine.detect_time_column_index(path)
            if idx is not None:
                self.controls._time_col_override[path] = int(idx)
                print(f"[PLI][Rescale] Auto-detected time column: {idx}")
            else:
                QMessageBox.warning(self.controls, "Time column", "Could not detect a 'Time of Day' column automatically. Please pick it manually.")
        except Exception as e:
            print(f"[PLI][Rescale] detect_time_column_index failed: {e}")

        self._refresh_intervals()

    def _pick_time_column(self) -> None:
        path = self.controls._last_time_source_path
        if not path or not os.path.isfile(path):
            QMessageBox.warning(self.controls, "No time source", "Load a time source before selecting the Time column.")
            return
        dlg = QDialog(self.controls)
        dlg.setWindowTitle("Time column index")
        layout = QVBoxLayout(dlg)

        # Preview header + first 3 data rows to help choose the right column
        preview_table = QTableWidget()
        preview_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        preview_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        preview_table.setMinimumHeight(180)
        try:
            header, data_rows = self._preview_time_source_rows(path, n_rows=3)
            n_cols = len(header)
            preview_table.setColumnCount(n_cols)
            if header:
                preview_table.setHorizontalHeaderLabels(header)
            preview_table.setRowCount(len(data_rows))
            for r, row in enumerate(data_rows):
                for c in range(n_cols):
                    val = row[c] if c < len(row) else ""
                    preview_table.setItem(r, c, QTableWidgetItem(str(val)))
            preview_table.resizeColumnsToContents()
        except Exception as e:
            preview_table.setRowCount(1)
            preview_table.setColumnCount(1)
            preview_table.setHorizontalHeaderLabels(["Preview"])
            preview_table.setItem(0, 0, QTableWidgetItem(f"Preview unavailable: {e}"))
        layout.addWidget(QLabel("Header + first 3 rows:", dlg))
        layout.addWidget(preview_table)

        form = QFormLayout()
        col_spin = QSpinBox()
        col_spin.setRange(0, 1000)
        col_spin.setValue(self.controls._time_col_override.get(path, 0))
        form.addRow("Time column index:", col_spin)
        layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_cancel = QPushButton("Cancel")
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.controls._time_col_override[path] = int(col_spin.value())
            self._refresh_intervals()

    def _refresh_intervals(self) -> None:
        mode = self.controls.current_mode()
        fps = float(self.controls.fpsSpin.value())
        self.controls._fps = fps
        path = self.controls._last_time_source_path

        if mode != "Manual" and (not path or not os.path.isfile(path)):
            self._set_table_message("(Intervals will appear here after loading time source)")
            return

        if mode == "Triggered":
            try:
                T_beg, T_end, S_beg, S_end = engine.extract_triggered_pairs_from_time_column(
                    path, override=self.controls._time_col_override
                )
            except Exception as e:
                self._set_table_message(f"Failed to parse triggered intervals: {e}")
                return
            rows = []
            n = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
            for i in range(n):
                tb = float(T_beg[i]); te = float(T_end[i]); sb = float(S_beg[i]); se = float(S_end[i])
                T_frames = int(round(max(0.0, te - tb) * fps))
                S_frames = int(round(max(0.0, se - sb) * fps))
                rows.append([i + 1, tb, te, sb, se, T_frames, S_frames])
            headers = ["#", "T_begin [s]", "T_end [s]", "S_begin [s]", "S_end [s]", "T [frames]", "S [frames]"]
            self._populate_table(headers, rows, editable=False)
            return

        if mode == "Reference":
            try:
                b_ref, e_ref = engine.parse_reference_steps_from_csv(path, override=self.controls._time_col_override)
                steady_sec = float(self.controls.steadySecSpin.value())
                T_beg, T_end, S_beg, S_end = engine.compute_reference_intervals_with_steady(b_ref, e_ref, steady_sec)
            except Exception as e:
                self._set_table_message(f"Failed to parse reference intervals: {e}")
                return
            rows = []
            n = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
            for i in range(n):
                tb = float(T_beg[i]); te = float(T_end[i]); sb = float(S_beg[i]); se = float(S_end[i])
                T_frames = int(round(max(0.0, te - tb) * fps))
                S_frames = int(round(max(0.0, se - sb) * fps))
                rows.append([i + 1, tb, te, sb, se, T_frames, S_frames])
            headers = ["#", "T_begin [s]", "T_end [s]", "S_begin [s]", "S_end [s]", "T [frames]", "S [frames]"]
            self._populate_table(headers, rows, editable=False)
            return

        # Manual mode
        try:
            n_req = int(self.controls.manualCountSpin.value())
        except Exception:
            n_req = 1
        n_req = max(1, min(50, n_req))
        saved = self.controls._manual_intervals
        if not saved:
            saved = _durations_to_intervals(DEFAULT_MANUAL_INTERVALS, gap_s=0.0)
        if n_req < len(saved):
            n_req = len(saved)
            self.controls.manualCountSpin.setValue(n_req)
        rows = []
        total_px = 0
        gap_s = float(self.controls.manualGapSpin.value())
        steady_window_s = float(self.controls.manualSteadySpin.value())
        for i in range(n_req):
            if i < len(saved):
                b_val, e_val = saved[i]
            else:
                b_val, e_val = 0.0, 0.0
            interval_sec = max(0.0, e_val - b_val)
            steady_sec = min(interval_sec, steady_window_s if steady_window_s > 0 else interval_sec)
            px_width = int(round(steady_sec * fps)) if interval_sec > 0 else 0
            total_px += px_width
            rows.append([i + 1, b_val, e_val, px_width])
        total_with_gaps = total_px
        if n_req > 1 and gap_s > 0:
            total_with_gaps = int(round(total_px + (n_req - 1) * gap_s * fps))
        self.controls.manualTotalPx.setText(str(total_px))
        self.controls.manualTotalWithGaps.setText(str(total_with_gaps))
        headers = ["#", "Begin [s]", "End [s]", "Steady width [px @ fps]"]
        self._populate_table(headers, rows, editable=True)

    def _populate_table(self, headers: list[str], rows: list[list], editable: bool = False) -> None:
        self.controls.table.clear()
        self.controls.table.setColumnCount(len(headers))
        self.controls.table.setRowCount(len(rows))
        self.controls.table.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(rows):
            for j, val in enumerate(row):
                item = QTableWidgetItem(f"{val}")
                if not editable or j == 0 or (headers[j].startswith("Steady")):
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.controls.table.setItem(i, j, item)
        self.controls.table.resizeColumnsToContents()
        self.controls.table.resizeRowsToContents()
        if editable:
            self.controls.table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked)
        else:
            self.controls.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)

    def _set_table_message(self, msg: str) -> None:
        self.controls.table.clear()
        self.controls.table.setRowCount(1)
        self.controls.table.setColumnCount(1)
        self.controls.table.setHorizontalHeaderLabels(["Intervals"])
        self.controls.table.setItem(0, 0, QTableWidgetItem(msg))

    def _on_preview(self) -> None:
        try:
            # Always preview full view (not the tail-only crop)
            self._tail_mode = False
            data = self._prepare_image_data()
            if data is None:
                return
            rgb = data["rgb"]
            H, W = rgb.shape[:2]
            tail_pct = "full"
            w0 = 0
            rgb_view = rgb
            view_width = W
            max_offset = max(0, W - 1)
            offset_raw = max(0, int(self.controls.endXSpin.value() or 0))
            offset = min(offset_raw, max_offset)
            end_x = self._end_abs_x if self._end_abs_x is not None else (W - 1) - offset
            self._sync_end_slider_max(max_offset)
            self.preview.set_tail_mode(False)
            T_beg, T_end, S_beg, S_end = self._compute_intervals()
            if not S_end:
                QMessageBox.information(self.controls, "No intervals", "No intervals available.")
                return
            fps_val = float(self.controls.fpsSpin.value())
            t_last = float(S_end[-1])
            x_cyan = [int(round(end_x - (t_last - float(t)) * fps_val)) for t in T_beg]
            x_red = [int(round(end_x - (t_last - float(t)) * fps_val)) for t in S_end]
            print(f"[PLI][Rescale] preview image size: {rgb_view.shape[1]}x{rgb_view.shape[0]}, tail_pct={tail_pct}, intervals={len(S_end)}, end_offset={offset}")
            self.preview.set_rgb(rgb_view)
            self.preview.clear_overlay()
            for x0 in x_cyan:
                x_adj = x0 - w0
                if 0 <= x_adj < rgb_view.shape[1]:
                    self.preview.add_vertical_line(x_adj, Qt.GlobalColor.cyan)
            for xr in x_red:
                x_adj = xr - w0
                if 0 <= x_adj < rgb_view.shape[1]:
                    self.preview.add_vertical_line(x_adj, Qt.GlobalColor.red)
            self.preview.ensure_fit()
            try:
                self.preview.show()
                self.preview.update()
                self.preview.view.viewport().update()
            except Exception:
                pass
        except Exception as e:
            QMessageBox.critical(self.controls, "Preview error", str(e))

    def _on_engage(self) -> None:
        try:
            data = self._prepare_image_data()
            if data is None:
                return
            rgb = data["rgb"]
            base_img = data["base"] or "PLI_sample"
            H, W = rgb.shape[:2]
            offset = max(0, int(self.controls.endXSpin.value() or 0))
            end_x = self._end_abs_x if self._end_abs_x is not None else (W - 1) - min(offset, W - 1)
            T_beg, T_end, S_beg, S_end = self._compute_intervals()
            if not S_beg or not S_end:
                QMessageBox.warning(self.controls, "Stitch", "No intervals available for stitching.")
                return
            fps_loc = float(self.controls.fpsSpin.value())
            t_last = float(S_end[-1])
            xs0 = [int(round(end_x - (t_last - float(s)) * fps_loc)) for s in S_beg]
            xs1 = [int(round(end_x - (t_last - float(s)) * fps_loc)) for s in S_end]

            full_tiles = []
            for x0, x1 in zip(xs0, xs1):
                x0c = max(0, min(W, x0))
                x1c = max(0, min(W, x1))
                if x1c <= x0c:
                    continue
                tile = rgb[:, x0c:x1c]
                full_tiles.append(tile)
            if not full_tiles:
                # Fallback: stitch the full view so we always produce output
                print("[PLI][Rescale] No steady tiles; falling back to full image for stitching.")
                full_tiles = [rgb]

            gap_px = int(self.controls.interludeSpin.value() or 0)
            gap_px = max(0, gap_px)

            def _with_gaps(tiles, gap_width, gap_height):
                if gap_width <= 0 or len(tiles) <= 1:
                    return tiles
                spacer = np.zeros((gap_height, gap_width, tiles[0].shape[2]), dtype=tiles[0].dtype)
                pieces = []
                for idx, t in enumerate(tiles):
                    pieces.append(t)
                    if idx != len(tiles) - 1:
                        spacer_local = spacer if spacer.shape[0] == t.shape[0] else np.zeros((t.shape[0], gap_width, t.shape[2]), dtype=t.dtype)
                        pieces.append(spacer_local)
                return pieces

            unscaled_parts = _with_gaps(full_tiles, gap_px, full_tiles[0].shape[0])
            unscaled = np.hstack(unscaled_parts)

            base = _sanitize_misjoined_user_path(self.reference_folder or os.getcwd())
            ref = get_reference_folder_from_path(base)
            temp_dir = os.path.join(ref, "PLI", "_Temp")
            os.makedirs(temp_dir, exist_ok=True)
            unscaled_name = f"{base_img}_steady_unscaled_stitched_{unscaled.shape[1]}x{unscaled.shape[0]}.png"
            unscaled_path = os.path.join(temp_dir, unscaled_name)
            bgr_unscaled = unscaled[:, :, ::-1]
            if not cv2.imwrite(unscaled_path, bgr_unscaled):
                raise RuntimeError("Failed to write unscaled stitched image")
            print(f"[PLI][Rescale] Stitched (unscaled) saved to {unscaled_path}")

            meta = {
                "n_intervals": int(len(full_tiles)),
                "interval_widths_px": [int(t.shape[1]) for t in full_tiles],
                "gap_px": gap_px,
                "fps": float(self.controls.fpsSpin.value()),
            }
            try:
                with open(unscaled_path + ".json", "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass

            target_h = max(1, int(self.controls.scaledHeightSpin.value()))
            aspect = max(0.1, float(self.controls.aspectSpin.value()))
            n_tiles = len(full_tiles)
            target_total_w = int(round(aspect * target_h))
            min_total_w = max(n_tiles, 1)
            if target_total_w < min_total_w:
                target_total_w = min_total_w

            total_gap_w = gap_px * max(n_tiles - 1, 0)
            avail_w = max(target_total_w - total_gap_w, n_tiles)
            base_w = avail_w // n_tiles
            rem = avail_w - base_w * n_tiles
            tile_widths = [base_w + (1 if i < rem else 0) for i in range(n_tiles)]

            scaled_tiles = []
            for tile, w_i in zip(full_tiles, tile_widths):
                w_i = max(1, int(w_i))
                interp = cv2.INTER_AREA if (tile.shape[0] > target_h or tile.shape[1] > w_i) else cv2.INTER_LINEAR
                scaled_tiles.append(cv2.resize(tile, (w_i, target_h), interpolation=interp))

            scaled_parts = _with_gaps(scaled_tiles, gap_px, target_h)
            stitched = np.hstack(scaled_parts)
            if stitched.shape[0] != target_h:
                stitched = cv2.resize(stitched, (stitched.shape[1], target_h), interpolation=cv2.INTER_NEAREST)

            out_dir = str(get_processed_root(ref))
            os.makedirs(out_dir, exist_ok=True)
            out_name = f"{base_img}_steady_rescaled_stitched_{stitched.shape[1]}x{stitched.shape[0]}.png"
            out_path = os.path.join(out_dir, out_name)
            bgr_out = stitched[:, :, ::-1]
            if not cv2.imwrite(out_path, bgr_out):
                raise RuntimeError("Failed to write scaled stitched image")
            try:
                with open(out_path + ".json", "w") as f:
                    json.dump(meta, f, indent=2)
            except Exception:
                pass
            try:
                save_params(ref, "PLI", "rescale", base_img, {
                    "target_total_w": target_total_w,
                    "target_h": target_h,
                    "gap_px": gap_px,
                    "tile_widths": tile_widths,
                    "mode": "rescale_equal_width",
                    "source": str(unscaled_path),
                    "output": out_path,
                })
            except Exception as e:
                print(f"[PLI] Warning: could not save last rescale params: {e}")

            # Preview the rescaled (equal-width) stitched result with interval boundaries
            try:
                self.preview.set_rgb(stitched)
                self.preview.clear_overlay()
                xpos = 0
                for w_i in tile_widths:
                    # start of tile
                    self.preview.add_vertical_line(xpos, Qt.GlobalColor.red)
                    xpos += w_i
                    # gap boundary (optional)
                    if gap_px > 0:
                        self.preview.add_vertical_line(xpos, Qt.GlobalColor.lightGray)
                        xpos += gap_px
                self.preview.ensure_fit()
            except Exception:
                pass

            QMessageBox.information(self.controls, "Stitch", f"Saved:\\n{unscaled_path}\\n{out_path}")
        except Exception as e:
            QMessageBox.critical(self.controls, "Stitch error", f"Failed to stitch:\\n{e}")
