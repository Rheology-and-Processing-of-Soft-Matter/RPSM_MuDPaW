from __future__ import annotations

import os
import json
import math
import numpy as np
import cv2
import pandas as pd

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QSlider,
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

try:
    from scipy.optimize import curve_fit
except Exception:  # pragma: no cover
    curve_fit = None

try:
    import _Routines.PLI.LabHWHM as DataAnalyzer
except Exception as e:  # pragma: no cover
    DataAnalyzer = None

try:
    from _UI_modules.PLI_helper import get_unified_processed_folder
except Exception:  # pragma: no cover
    get_unified_processed_folder = None


class PLIAnalyzerPreview(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        fig = Figure(figsize=(8, 6), dpi=100, constrained_layout=False)
        # Disable any layout engine; we'll let Matplotlib pick defaults without subplots_adjust to avoid warnings.
        try:
            fig.set_layout_engine(None)
        except Exception:
            pass
        self.canvas = FigureCanvas(fig)
        self.ax_top = fig.add_subplot(211)
        self.ax_bot = fig.add_subplot(212)
        layout.addWidget(self.canvas)

    def plot_preview(self, x, series_list, ylabel: str, title: str, baseline_series=None, fit_series=None):
        self.ax_top.clear()
        if series_list:
            for y in series_list:
                self.ax_top.plot(x, y, linestyle="None", marker="o", markersize=2.0, alpha=0.6)
        if baseline_series is not None:
            self.ax_top.plot(x, baseline_series, linewidth=1.0, alpha=0.45)
        if fit_series is not None:
            self.ax_top.plot(x, fit_series, linestyle="--", linewidth=1.2, alpha=0.9)
        self.ax_top.set_title(title, fontsize=10, pad=6)
        self.ax_top.set_ylabel(ylabel)
        self.ax_top.grid(True, alpha=0.3)
        self.ax_top.set_xlabel("")

        # Clear bottom until full ENGAGE is implemented
        self.ax_bot.clear()
        self.ax_bot.set_xlabel("")
        self.ax_bot.set_ylabel("")
        self.ax_bot.grid(False)

        self.canvas.draw_idle()

    def plot_results(self, fwhm_vec, auc_vec=None):
        self.ax_bot.clear()
        x_idx = np.arange(len(fwhm_vec))
        fwhm_plot = self.ax_bot.plot(
            x_idx, fwhm_vec,
            linestyle='None', marker='o', mfc='none', mec='C0', mew=1.25,
            label='FWHM (index units)'
        )[0]
        self.ax_bot.set_xlabel("Interval index (relative)")
        self.ax_bot.set_ylabel("FWHM (index units)")
        self.ax_bot.grid(True, alpha=0.3)

        if auc_vec is not None and len(auc_vec) == len(fwhm_vec):
            ax2 = self.ax_bot.twinx()
            ax2.plot(
                x_idx, auc_vec,
                linestyle='None', marker='o', mfc='none', mec='green', mew=1.25,
                label='AUC (a.u.)'
            )
            ax2.set_ylabel("AUC (a.u.)", color='green')
            ax2.tick_params(axis='y', colors='green')
        self.canvas.draw_idle()


class PLIAnalyzerControls(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        file_row = QHBoxLayout()
        self.fileLabel = QLabel("", self)
        file_row.addWidget(self.fileLabel)
        file_row.addStretch(1)
        layout.addLayout(file_row)

        param_row = QHBoxLayout()
        param_row.addWidget(QLabel("Parameter:"))
        self.paramCombo = QComboBox()
        self.paramCombo.addItems(["Lab", "ab", "L", "a", "b"])
        self.paramCombo.setCurrentText("Lab")
        param_row.addWidget(self.paramCombo)
        param_row.addStretch(1)
        layout.addLayout(param_row)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Fit model:"))
        self.fitModelCombo = QComboBox()
        self.fitModelCombo.addItems(["No model (exp HWHM)", "Gaussian", "Asym q-Gaussian"])
        model_row.addWidget(self.fitModelCombo)
        self.fitBaselineCheck = QCheckBox("Fit baseline")
        self.fitBaselineCheck.setChecked(True)
        model_row.addWidget(self.fitBaselineCheck)
        model_row.addStretch(1)
        layout.addLayout(model_row)

        base_row = QHBoxLayout()
        self.baseCheck = QCheckBox("Baseline correction")
        base_row.addWidget(self.baseCheck)
        base_row.addWidget(QLabel("from interval:"))
        self.baseIdxSpin = QSpinBox()
        self.baseIdxSpin.setRange(0, 0)
        base_row.addWidget(self.baseIdxSpin)
        base_row.addWidget(QLabel("method:"))
        self.baseMethodCombo = QComboBox()
        self.baseMethodCombo.addItems(["Subtract interval", "Subtract fit"])
        self.baseMethodCombo.setCurrentText("Subtract interval")
        base_row.addWidget(self.baseMethodCombo)
        base_row.addStretch(1)
        layout.addLayout(base_row)

        ang_row = QHBoxLayout()
        ang_row.addWidget(QLabel("Angle window [normalized]:"))
        self.ang0 = QDoubleSpinBox()
        self.ang0.setRange(0.0, 1.0)
        self.ang0.setDecimals(3)
        self.ang0.setValue(0.0)
        ang_row.addWidget(self.ang0)
        self.ang1 = QDoubleSpinBox()
        self.ang1.setRange(0.0, 1.0)
        self.ang1.setDecimals(3)
        self.ang1.setValue(1.0)
        ang_row.addWidget(self.ang1)
        ang_row.addStretch(1)
        layout.addLayout(ang_row)

        # Dual sliders for angle window (0..1 mapped to 0..1000)
        slider_row = QHBoxLayout()
        self.angSlider0 = QSlider(Qt.Orientation.Horizontal)
        self.angSlider1 = QSlider(Qt.Orientation.Horizontal)
        for s in (self.angSlider0, self.angSlider1):
            s.setRange(0, 1000)
            s.setSingleStep(1)
            s.setPageStep(25)
            s.setTracking(True)
        self.angSlider0.setValue(0)
        self.angSlider1.setValue(1000)
        slider_row.addWidget(self.angSlider0, 1)
        slider_row.addWidget(self.angSlider1, 1)
        layout.addLayout(slider_row)

        range_row = QHBoxLayout()
        self.rangeLabel = QLabel("Intervals (0..0) to analyze:")
        range_row.addWidget(self.rangeLabel)
        self.i0Spin = QSpinBox()
        self.i0Spin.setRange(0, 0)
        range_row.addWidget(self.i0Spin)
        self.i1Spin = QSpinBox()
        self.i1Spin.setRange(0, 0)
        range_row.addWidget(self.i1Spin)
        range_row.addStretch(1)
        layout.addLayout(range_row)

        btn_row = QHBoxLayout()
        self.btnPreview = QPushButton("Preview (update)")
        self.btnAccept = QPushButton("Accept: fit + save")
        btn_row.addWidget(self.btnPreview)
        btn_row.addWidget(self.btnAccept)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        extras = QHBoxLayout()
        self.showFitsCheck = QCheckBox("Show fit curves (after accept)")
        extras.addWidget(self.showFitsCheck)
        self.btnShowR2 = QPushButton("Show R² table")
        extras.addWidget(self.btnShowR2)
        self.btnReset = QPushButton("Reset view")
        extras.addWidget(self.btnReset)
        extras.addStretch(1)
        layout.addLayout(extras)


class PLIAnalyzerController:
    def __init__(self, stitched_path: str, preview: PLIAnalyzerPreview, controls: PLIAnalyzerControls):
        self.preview = preview
        self.controls = controls
        self.stitched_path = stitched_path
        self._L = None
        self._a = None
        self._b = None
        self._n_intervals = 1
        self._interval_width = 300
        self._fit_store = {"x": None, "fits": None, "r2": None, "cols": None}

        self._load_meta_and_data()
        self._wire()
        self._preview_only()

    def _wire(self) -> None:
        # Combos
        self.controls.paramCombo.currentIndexChanged.connect(self._preview_only)
        self.controls.paramCombo.currentTextChanged.connect(self._preview_only)
        self.controls.fitModelCombo.currentIndexChanged.connect(self._preview_only)
        self.controls.baseMethodCombo.currentIndexChanged.connect(self._preview_only)
        # Checkboxes
        self.controls.fitBaselineCheck.stateChanged.connect(self._preview_only)
        self.controls.baseCheck.stateChanged.connect(self._preview_only)
        self.controls.showFitsCheck.stateChanged.connect(self._preview_only)
        # Spins
        self.controls.ang0.valueChanged.connect(self._on_ang_spin_changed)
        self.controls.ang1.valueChanged.connect(self._on_ang_spin_changed)
        self.controls.i0Spin.valueChanged.connect(self._preview_only)
        self.controls.i1Spin.valueChanged.connect(self._preview_only)
        self.controls.baseIdxSpin.valueChanged.connect(self._preview_only)
        # Sliders (sync with angle spins)
        self.controls.angSlider0.valueChanged.connect(self._on_ang_slider_changed)
        self.controls.angSlider1.valueChanged.connect(self._on_ang_slider_changed)
        # Buttons
        self.controls.btnPreview.clicked.connect(self._preview_only)
        self.controls.btnAccept.clicked.connect(self._accept_fit_save)
        self.controls.btnShowR2.clicked.connect(self._show_r2_table)
        self.controls.btnReset.clicked.connect(self._reset_view)

    def _on_ang_spin_changed(self, _val: float) -> None:
        """Keep sliders in sync when spins change and refresh preview."""
        v0 = int(round(self.controls.ang0.value() * 1000))
        v1 = int(round(self.controls.ang1.value() * 1000))
        if v1 < v0:
            v0, v1 = v1, v0
            # write back ordered values to spin boxes
            try:
                self.controls.ang0.blockSignals(True)
                self.controls.ang1.blockSignals(True)
                self.controls.ang0.setValue(v0 / 1000.0)
                self.controls.ang1.setValue(v1 / 1000.0)
            finally:
                self.controls.ang0.blockSignals(False)
                self.controls.ang1.blockSignals(False)
        try:
            if self.controls.angSlider0.value() != v0:
                self.controls.angSlider0.blockSignals(True)
                self.controls.angSlider0.setValue(v0)
                self.controls.angSlider0.blockSignals(False)
            if self.controls.angSlider1.value() != v1:
                self.controls.angSlider1.blockSignals(True)
                self.controls.angSlider1.setValue(v1)
                self.controls.angSlider1.blockSignals(False)
        finally:
            self._preview_only()

    def _on_ang_slider_changed(self, _val: int) -> None:
        """Update angle spin boxes when either slider moves, keeping order, then preview."""
        v0 = self.controls.angSlider0.value()
        v1 = self.controls.angSlider1.value()
        if v1 < v0:
            v0, v1 = v1, v0
        try:
            self.controls.ang0.blockSignals(True)
            self.controls.ang1.blockSignals(True)
            self.controls.ang0.setValue(v0 / 1000.0)
            self.controls.ang1.setValue(v1 / 1000.0)
        finally:
            self.controls.ang0.blockSignals(False)
            self.controls.ang1.blockSignals(False)
        self._preview_only()

    def _load_meta_and_data(self) -> None:
        self.controls.fileLabel.setText(os.path.basename(self.stitched_path))
        meta_path = self.stitched_path + ".json"
        meta = None
        if os.path.isfile(meta_path):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
            except Exception:
                meta = None
        if meta:
            widths = meta.get("interval_widths_px") if isinstance(meta.get("interval_widths_px"), list) else None
            n_int = int(meta.get("n_intervals", 0)) if meta else 0
            if widths:
                self._n_intervals = len(widths)
            elif n_int > 0:
                self._n_intervals = n_int
        # Fallback: derive n_intervals from image width
        img = cv2.imread(self.stitched_path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {self.stitched_path}")
        H, W = img.shape[:2]
        if not meta:
            self._interval_width = 300
            self._n_intervals = max(1, int(math.ceil(W / float(self._interval_width))))

        self.controls.baseIdxSpin.setRange(0, max(0, self._n_intervals - 1))
        self.controls.i0Spin.setRange(0, max(0, self._n_intervals - 1))
        self.controls.i1Spin.setRange(0, max(0, self._n_intervals - 1))
        self.controls.i1Spin.setValue(max(0, self._n_intervals - 1))
        self.controls.rangeLabel.setText(f"Intervals (0..{self._n_intervals-1}) to analyze:")

        if DataAnalyzer is not None:
            try:
                L_mat, a_mat, b_mat = DataAnalyzer.LabHWHM(self.stitched_path, self._n_intervals)
            except Exception:
                L_mat = a_mat = b_mat = None
        else:
            L_mat = a_mat = b_mat = None

        if L_mat is None or a_mat is None or b_mat is None:
            # Fallback: derive per-interval LAB by slicing the stitched panel using interval widths.
            lab_local = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            _L, _a, _b = cv2.split(lab_local)
            widths = meta.get("interval_widths_px") if meta and isinstance(meta.get("interval_widths_px"), list) else None
            if widths and len(widths) == self._n_intervals:
                widths_px = [int(max(1, w)) for w in widths]
            else:
                # Even split if widths unavailable
                base_w = W // self._n_intervals
                widths_px = [base_w] * self._n_intervals
                widths_px[-1] = W - base_w * (self._n_intervals - 1)
            x0 = 0
            cols_L = []
            cols_a = []
            cols_b = []
            for w_i in widths_px:
                x1 = min(W, x0 + w_i)
                slab_L = _L[:, x0:x1].astype(np.float32)
                slab_a = (_a[:, x0:x1].astype(np.float32) - 128.0)
                slab_b = (_b[:, x0:x1].astype(np.float32) - 128.0)
                cols_L.append(np.nanmean(slab_L, axis=1))
                cols_a.append(np.nanmean(slab_a, axis=1))
                cols_b.append(np.nanmean(slab_b, axis=1))
                x0 = x1
            L_mat = np.stack(cols_L, axis=1)
            a_mat = np.stack(cols_a, axis=1)
            b_mat = np.stack(cols_b, axis=1)

        self._L, self._a, self._b = L_mat, a_mat, b_mat

    def _param_matrix(self, mode: str) -> np.ndarray:
        mode = (mode or "ab").lower()
        if mode == "l":
            return np.asarray(self._L, dtype=np.float32)
        if mode == "a":
            return np.asarray(self._a, dtype=np.float32)
        if mode == "b":
            return np.asarray(self._b, dtype=np.float32)
        if mode in ("sqrt(a^2+b^2)", "ab", "chroma", "ab_mag"):
            return np.sqrt(self._a * self._a + self._b * self._b, dtype=np.float32)
        if mode in ("sqrt(l^2+a^2+b^2)", "lab", "lab_mag"):
            return np.sqrt(self._L * self._L + self._a * self._a + self._b * self._b, dtype=np.float32)
        return np.sqrt(self._a * self._a + self._b * self._b, dtype=np.float32)

    def _preview_only(self, *args, **kwargs) -> None:
        # Keep sliders in sync with spin boxes (but avoid feedback loops)
        try:
            s0 = int(round(self.controls.ang0.value() * 1000))
            s1 = int(round(self.controls.ang1.value() * 1000))
            if self.controls.angSlider0.value() != s0:
                self.controls.angSlider0.blockSignals(True)
                self.controls.angSlider0.setValue(s0)
                self.controls.angSlider0.blockSignals(False)
            if self.controls.angSlider1.value() != s1:
                self.controls.angSlider1.blockSignals(True)
                self.controls.angSlider1.setValue(s1)
                self.controls.angSlider1.blockSignals(False)
        except Exception:
            pass
        M2D = self._param_matrix(self.controls.paramCombo.currentText())
        H = M2D.shape[0]
        angY = np.linspace(0.0, 1.0, H, endpoint=False)
        a0 = float(self.controls.ang0.value())
        a1 = float(self.controls.ang1.value())
        a_lo, a_hi = (a0, a1) if a0 <= a1 else (a1, a0)
        maskY = (angY >= a_lo) & (angY <= a_hi)
        if not np.any(maskY):
            # Still show the requested window even if empty (e.g., extreme slider)
            self.preview.ax_top.clear()
            self.preview.ax_top.set_xlim(a_lo, a_hi)
            self.preview.ax_top.set_ylim(0, 1)
            self.preview.ax_top.set_title("Position (normalized) — window", fontsize=10, pad=6)
            self.preview.ax_top.grid(True, alpha=0.3)
            self.preview.canvas.draw_idle()
            print(f"[PLI][Analyzer] Empty angle mask a_lo={a_lo:.3f}, a_hi={a_hi:.3f}, H={H}")
            return

        i0 = max(0, min(self._n_intervals - 1, int(self.controls.i0Spin.value())))
        i1 = max(0, min(self._n_intervals - 1, int(self.controls.i1Spin.value())))
        if i1 < i0:
            i0, i1 = i1, i0

        x_norm = angY[maskY].astype(np.float64)
        # Debug trace to verify slider → mask mapping
        try:
            print(f"[PLI][Analyzer] Preview: angle window {a_lo:.3f}-{a_hi:.3f}, samples={x_norm.size}, intervals={i0}-{i1}")
        except Exception:
            pass
        series_list = []
        baseline = None
        fit_line = None
        if self.controls.baseCheck.isChecked():
            kB = max(0, min(self._n_intervals - 1, int(self.controls.baseIdxSpin.value())))
            y_base_raw = M2D[maskY, kB].astype(np.float64)
            if self.controls.baseMethodCombo.currentText().strip().lower() == "subtract fit" and curve_fit is not None:
                def _g(x, A, x0, sigma):
                    return A * np.exp(-((x - x0) ** 2) / (2.0 * (sigma ** 2) + 1e-18))
                try:
                    j_max = int(np.nanargmax(y_base_raw))
                    A0 = float(y_base_raw[j_max]) if np.isfinite(y_base_raw[j_max]) else float(np.nanmax(y_base_raw))
                    x0_0 = float(x_norm[j_max]) if np.isfinite(x_norm[j_max]) else float(np.nanmedian(x_norm))
                    span = float(np.nanmax(x_norm) - np.nanmin(x_norm))
                    sigma0 = max(span / 6.0, 1e-6)
                    p0 = (A0, x0_0, sigma0)
                    bounds = ((0.0, float(np.nanmin(x_norm)), 0.0), (np.inf, float(np.nanmax(x_norm)), np.inf))
                    popt, _ = curve_fit(_g, x_norm, y_base_raw, p0=p0, bounds=bounds, method="trf", maxfev=20000)
                    fit_line = _g(x_norm, *popt)
                except Exception:
                    fit_line = None
            baseline = y_base_raw

        for k in range(i0, i1 + 1):
            y_seg = M2D[maskY, k].astype(np.float64)
            if self.controls.baseCheck.isChecked():
                if fit_line is not None and fit_line.size == y_seg.size:
                    y_seg = y_seg - fit_line
                elif baseline is not None and baseline.size == y_seg.size:
                    y_seg = y_seg - baseline
            series_list.append(y_seg)

        ylabel = f"{self.controls.paramCombo.currentText()} (a.u.)"
        if self.controls.baseCheck.isChecked():
            ylabel += " — baseline corrected"
        self.preview.plot_preview(
            x_norm,
            series_list,
            ylabel=ylabel,
            title="Position (normalized) — window",
            baseline_series=baseline,
            fit_series=fit_line,
        )
        # Force the x-limits to the selected window (in case data are constant)
        try:
            self.preview.ax_top.set_xlim(a_lo, a_hi)
        except Exception:
            pass
        self.preview.canvas.draw_idle()

        # Optional fit overlay after accept
        try:
            if self.controls.showFitsCheck.isChecked():
                x_fit = self._fit_store.get("x")
                fits = self._fit_store.get("fits")
                if x_fit is not None and fits:
                    wmask = (x_fit >= a_lo) & (x_fit <= a_hi)
                    if np.any(wmask):
                        xf = x_fit[wmask]
                        for yhat in fits:
                            if yhat is None:
                                continue
                            yh = np.asarray(yhat)
                            if yh.size == x_fit.size:
                                self.preview.ax_top.plot(xf, yh[wmask], linestyle='--', linewidth=0.8, alpha=0.8)
                        self.preview.canvas.draw_idle()
        except Exception:
            pass

    def _accept_fit_save(self) -> None:
        data_analyzer_available = DataAnalyzer is not None
        M2D = self._param_matrix(self.controls.paramCombo.currentText())
        H = M2D.shape[0]
        angles_full = np.linspace(0.0, 1.0, H, endpoint=False)
        a0 = float(self.controls.ang0.value())
        a1 = float(self.controls.ang1.value())
        a_lo, a_hi = (a0, a1) if a0 <= a1 else (a1, a0)
        mask = (angles_full >= a_lo) & (angles_full <= a_hi)
        if not np.any(mask):
            QMessageBox.warning(self.controls, "Analyzer", "No samples in selected angle window.")
            return

        cols = list(range(self._n_intervals))
        trim_ = M2D[mask][:, cols]

        # Optional baseline correction using a chosen interval
        if self.controls.baseCheck.isChecked():
            kB = max(0, min(self._n_intervals - 1, int(self.controls.baseIdxSpin.value())))
            x_norm_window = angles_full[mask].astype(np.float64)
            y_base = M2D[mask, kB].astype(np.float64)
            method = self.controls.baseMethodCombo.currentText().strip().lower()
            if method == "subtract fit" and curve_fit is not None:
                def _g(x, A, x0, sigma):
                    return A * np.exp(-((x - x0) ** 2) / (2.0 * (sigma ** 2) + 1e-18))
                try:
                    j_max = int(np.nanargmax(y_base))
                    A0 = float(y_base[j_max]) if np.isfinite(y_base[j_max]) else float(np.nanmax(y_base))
                    x0_0 = float(x_norm_window[j_max]) if np.isfinite(x_norm_window[j_max]) else float(np.nanmedian(x_norm_window))
                    span = float(np.nanmax(x_norm_window) - np.nanmin(x_norm_window))
                    sigma0 = max(span / 6.0, 1e-6)
                    p0 = (A0, x0_0, sigma0)
                    bounds = ((0.0, float(np.nanmin(x_norm_window)), 0.0),
                              (np.inf, float(np.nanmax(x_norm_window)), np.inf))
                    popt, _ = curve_fit(_g, x_norm_window, y_base, p0=p0, bounds=bounds, method="trf", maxfev=20000)
                    y_fit_base = _g(x_norm_window, *popt)
                    trim_ = trim_ - y_fit_base[:, None]
                except Exception:
                    trim_ = trim_ - y_base[:, None]
            else:
                trim_ = trim_ - y_base[:, None]

        # AUC experimental
        x_norm_window = angles_full[mask].astype(np.float64)
        AUC_exp_vec = np.trapz(trim_, x=x_norm_window, axis=0)

        # Compute widths
        model_sel = self.controls.fitModelCombo.currentText().strip().lower()
        if not data_analyzer_available and not model_sel.startswith("no model"):
            # Fall back to model-free widths when LabHWHM is absent
            model_sel = "no model"
            print("[PLI] LabHWHM unavailable; using model-free widths.")
        details = None
        if model_sel.startswith("no model"):
            HWHM_vec, HUH_vec, FWHM_vec, details = self._model_free_widths(x_norm_window, trim_)
        else:
            model_key = "asym_qgauss" if model_sel.startswith("asym") else "gaussian"
            try:
                ret = DataAnalyzer.ExPLIDat(
                    trim_,
                    x_axis=x_norm_window,
                    model=model_key,
                    use_baseline=bool(self.controls.fitBaselineCheck.isChecked()),
                )
                if isinstance(ret, (list, tuple)) and len(ret) >= 2:
                    HWHM_vec, HUH_vec = ret[0], ret[1]
                    if len(ret) >= 3 and isinstance(ret[2], dict):
                        details = ret[2]
                else:
                    HWHM_vec = np.asarray(ret)
                    HUH_vec = np.full(len(cols), np.nan)
            except Exception as e:
                QMessageBox.warning(self.controls, "Analyzer", f"ExPLIDat failed: {e}")
                HWHM_vec = np.full(len(cols), np.nan)
                HUH_vec = np.full(len(cols), np.nan)
        if isinstance(details, dict) and details.get("FWHM") is not None:
            FWHM_vec = np.asarray(details.get("FWHM"))
        else:
            FWHM_vec = 2.0 * np.asarray(HWHM_vec)

        # Save fit store
        if isinstance(details, dict) and all(k in details for k in ("fits", "r2", "x")) and details.get("fits") is not None:
            self._fit_store["x"] = np.asarray(details.get("x"))
            try:
                self._fit_store["fits"] = [np.asarray(f) if f is not None else None for f in details.get("fits")]
            except Exception:
                self._fit_store["fits"] = None
            self._fit_store["r2"] = list(details.get("r2")) if details.get("r2") is not None else None
            self._fit_store["cols"] = cols
        else:
            self._fit_store = {"x": None, "fits": None, "r2": None, "cols": cols}

        # Plot bottom results
        self.preview.plot_results(FWHM_vec, auc_vec=AUC_exp_vec)

        # Export CSVs + master JSON
        out_root = get_unified_processed_folder(self.stitched_path) if callable(get_unified_processed_folder) else None
        if out_root is None:
            out_root = os.path.dirname(self.stitched_path)
        out_dir = os.path.join(out_root, "PLI")
        os.makedirs(out_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.stitched_path))[0]
        csv_out = os.path.join(out_dir, f"_pli_{base}_maltese_summary.csv")
        df_out = pd.DataFrame({
            "interval_global": cols,
            "interval_rel": np.arange(len(cols)).astype(int),
            "FWHM_norm": np.asarray(FWHM_vec),
            "HWHM_norm": np.asarray(HWHM_vec),
            "HUH": np.asarray(HUH_vec),
            "AUC_exp_norm": np.asarray(AUC_exp_vec),
        })
        df_out.to_csv(csv_out, index=False)

        dg_csv = os.path.join(out_dir, f"_dg_pli_{base}_maltese.csv")
        dg_df = pd.DataFrame({
            "interval": np.arange(len(cols)).astype(int),
            "FWHM_deg": np.asarray(FWHM_vec) * 360.0,
            "HWHM_deg": np.asarray(HWHM_vec) * 360.0,
            "HUH": np.asarray(HUH_vec),
            "AUC_exp": np.asarray(AUC_exp_vec),
        })
        dg_df.to_csv(dg_csv, index=False)

        # Update master JSON
        master_json = os.path.join(out_dir, f"_output_PLI_{base}.json")
        master_obj = {
            "mode": "pli",
            "source_png": os.path.abspath(self.stitched_path),
            "processed_folder": out_root,
            "datasets": {},
        }
        if os.path.isfile(master_json):
            try:
                with open(master_json, "r") as _f:
                    master_obj = json.load(_f) or master_obj
            except Exception:
                pass
        if not isinstance(master_obj.get("datasets"), dict):
            master_obj["datasets"] = {}
        key = base
        entry = master_obj["datasets"].get(key, {})
        if not isinstance(entry.get("maltese_cross"), dict):
            entry["maltese_cross"] = {}
        analysis_tag = f"{self.controls.paramCombo.currentText()}|x{a_lo:.3f}-{a_hi:.3f}"
        entry["maltese_cross"][analysis_tag] = {
            "parameter": self.controls.paramCombo.currentText(),
            "axis_limits": [a_lo, a_hi],
            "x_axis_units": "unit",
            "summary_csv": os.path.abspath(csv_out),
            "summary_csv_rel": os.path.basename(csv_out),
            "columns": list(df_out.columns),
            "auc_source": "experimental_trapz",
            "auc_fit_included": False,
            "n_intervals_total": int(self._n_intervals),
            "intervals_analyzed": {"start": 0, "end": int(self._n_intervals - 1)},
            "datagraph_ready": True,
            "datagraph_csv": os.path.abspath(dg_csv),
            "datagraph_csv_rel": os.path.basename(dg_csv),
        }
        if not isinstance(entry.get("dg_exports"), list):
            entry["dg_exports"] = []
        if os.path.abspath(dg_csv) not in entry["dg_exports"]:
            entry["dg_exports"].append(os.path.abspath(dg_csv))
        master_obj["datasets"][key] = entry
        try:
            with open(master_json, "w") as _f:
                json.dump(master_obj, _f, indent=2)
        except Exception:
            pass

        QMessageBox.information(self.controls, "Analyzer", f"Saved:\\n{csv_out}\\n{dg_csv}")

    def _show_r2_table(self) -> None:
        r2 = self._fit_store.get("r2")
        cols = self._fit_store.get("cols")
        if not r2:
            QMessageBox.information(self.controls, "R²", "No R² available. Run 'Accept: fit + save' first.")
            return
        lines = []
        for j, val in zip(cols or [], r2):
            if np.isfinite(val):
                lines.append(f"interval {j}: R² = {val:.4f}")
            else:
                lines.append(f"interval {j}: R² = NaN")
        QMessageBox.information(self.controls, "R²", "\n".join(lines))
    def _model_free_widths(self, x_norm_window: np.ndarray, trim_: np.ndarray):
        phi = np.asarray(x_norm_window, dtype=np.float64)
        Y = np.asarray(trim_, dtype=np.float64)
        ncols = Y.shape[1]
        hwhm = np.full(ncols, np.nan, dtype=np.float64)
        fwhm = np.full(ncols, np.nan, dtype=np.float64)
        HUH_local = np.full(ncols, np.nan, dtype=np.float64)

        def _crossing(phi_arr, y_arr, half, idx0, step):
            j = idx0
            while 0 <= j + step < len(phi_arr):
                if np.isfinite(y_arr[j + step]) and y_arr[j + step] <= half < y_arr[j]:
                    x0, y0 = phi_arr[j], y_arr[j]
                    x1, y1 = phi_arr[j + step], y_arr[j + step]
                    if y1 == y0:
                        return x0
                    return x0 + (x1 - x0) * (half - y0) / (y1 - y0)
                j += step
            return np.nan

        for j in range(ncols):
            y = np.asarray(Y[:, j], dtype=np.float64)
            if not np.isfinite(y).any():
                continue
            i0 = int(np.nanargmax(np.abs(y)))
            peak = float(y[i0])
            if not np.isfinite(peak):
                continue
            mu = float(phi[i0])
            half = 0.5 * peak
            phiL = _crossing(phi, y, half, i0, -1)
            phiR = _crossing(phi, y, half, i0, +1)
            if np.isfinite(phiL) and np.isfinite(phiR):
                fwhm[j] = float(phiR - phiL)
                hwhm[j] = 0.5 * fwhm[j]
            HUH_local[j] = np.exp(-((0.5 * fwhm[j]) ** 2) / np.log(2.0)) if np.isfinite(fwhm[j]) else np.nan
        return hwhm, HUH_local, fwhm, None

    def _reset_view(self) -> None:
        self.controls.baseCheck.setChecked(False)
        self._preview_only()
