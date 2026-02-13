from __future__ import annotations

import json
import os
import re
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)
from PyQt6.QtWidgets import QHeaderView

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


BASE_DIR = Path(__file__).resolve().parents[1]


class SAXSInspector:
    def __init__(self, main) -> None:
        self.main = main
        self.canvas: FigureCanvas | None = None
        self._saxs_defaults_path = BASE_DIR / "_Miscell" / "_saxs_defaults.json"
        self._saxs_samples: list[dict] = []
        self._saxs_tab_widgets: dict[str, dict] = {}
        self.saxsTabs: QTabWidget | None = None

    # ---- Canvas ----
    def ensure_canvas(self, page: QWidget) -> None:
        if self.canvas is not None:
            return
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        fig = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)

    def _display_figure(self, fig) -> None:
        page = self.main.canvas_pages.get("pageSAXSCanvas")
        if page is None:
            return
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        if self.canvas is not None:
            self.canvas.setParent(None)
        self.canvas = FigureCanvas(fig)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.canvas)
        try:
            self.canvas.draw()
        except Exception:
            pass

    # ---- Inspector UI ----
    def build_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(8)
        # Full inspector as tabs (single-peak / two-peak)
        self.saxsTabs = QTabWidget(page)
        layout.addWidget(self.saxsTabs, 1)

        self._saxs_tab_widgets = {}

        single_tab = QWidget(self.saxsTabs)
        self.saxsTabs.addTab(single_tab, "Single-peak")
        self._saxs_tab_widgets["single"] = self._build_tab(single_tab, two_peak=False)

        two_tab = QWidget(self.saxsTabs)
        self.saxsTabs.addTab(two_tab, "Two-peak")
        self._saxs_tab_widgets["two"] = self._build_tab(two_tab, two_peak=True)

        self._load_defaults()

    def _build_tab(self, page: QWidget, *, two_peak: bool) -> dict:
        layout = QVBoxLayout(page)
        layout.setSpacing(8)

        title = QLabel("Anisotropy parameters", page)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        grid = QGridLayout()
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        layout.addLayout(grid)

        smoothing = QDoubleSpinBox(page)
        smoothing.setDecimals(4)
        smoothing.setRange(0.0, 10.0)
        smoothing.setSingleStep(0.01)
        sigma = QDoubleSpinBox(page)
        sigma.setDecimals(4)
        sigma.setRange(0.0, 100.0)
        sigma.setSingleStep(0.01)
        theta_min = QDoubleSpinBox(page)
        theta_min.setRange(0.0, 360.0)
        theta_min.setDecimals(2)
        theta_max = QDoubleSpinBox(page)
        theta_max.setRange(0.0, 360.0)
        theta_max.setDecimals(2)

        grid.addWidget(QLabel("Smoothing"), 0, 0)
        grid.addWidget(smoothing, 0, 1)
        grid.addWidget(QLabel("Sigma"), 0, 2)
        grid.addWidget(sigma, 0, 3)
        grid.addWidget(QLabel("θ min"), 1, 0)
        grid.addWidget(theta_min, 1, 1)
        grid.addWidget(QLabel("θ max"), 1, 2)
        grid.addWidget(theta_max, 1, 3)

        opts_row = QHBoxLayout()
        fast_check = QCheckBox("Fast (no plots)", page)
        persist_defaults = QCheckBox("Set as default", page)
        opts_row.addWidget(fast_check)
        opts_row.addWidget(persist_defaults)
        opts_row.addStretch(1)
        layout.addLayout(opts_row)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:", page))
        method_fit = QCheckBox("Fitting", page)
        method_direct = QCheckBox("Direct", page)
        method_fit.setChecked(True)
        method_row.addWidget(method_fit)
        method_row.addWidget(method_direct)
        method_row.addStretch(1)
        layout.addLayout(method_row)

        side_row = QHBoxLayout()
        side_row.addWidget(QLabel("Peak side", page))
        peak_side = QComboBox(page)
        peak_side.addItems(["auto", "left", "right"])
        side_row.addWidget(peak_side)
        side_row.addStretch(1)
        layout.addLayout(side_row)

        mirror_row = QHBoxLayout()
        mirror_check = QCheckBox("Mirror azimuthal data", page)
        average_check = QCheckBox("Average frames", page)
        mirror_row.addWidget(mirror_check)
        mirror_row.addWidget(average_check)
        mirror_row.addWidget(QLabel("Group"))
        average_group = QSpinBox(page)
        average_group.setRange(1, 100)
        average_group.setValue(4)
        average_group.setEnabled(False)
        mirror_row.addWidget(average_group)
        mirror_row.addStretch(1)
        layout.addLayout(mirror_row)
        average_check.stateChanged.connect(lambda _=False: average_group.setEnabled(average_check.isChecked()))

        # Two-peak only controls
        theta2_min = None
        theta2_max = None
        peak2_side = None
        tail_fit = None
        fit_extent = None
        if two_peak:
            two_layout = QGridLayout()
            two_layout.setHorizontalSpacing(8)
            two_layout.setVerticalSpacing(6)
            two_layout.addWidget(QLabel("Peak 2: θ min"), 0, 0)
            theta2_min = QDoubleSpinBox(page)
            theta2_min.setRange(0.0, 360.0)
            theta2_min.setDecimals(2)
            two_layout.addWidget(theta2_min, 0, 1)
            two_layout.addWidget(QLabel("θ max"), 0, 2)
            theta2_max = QDoubleSpinBox(page)
            theta2_max.setRange(0.0, 360.0)
            theta2_max.setDecimals(2)
            two_layout.addWidget(theta2_max, 0, 3)

            two_layout.addWidget(QLabel("Peak 2 side"), 1, 0)
            peak2_side = QComboBox(page)
            peak2_side.addItems(["auto", "left", "right"])
            two_layout.addWidget(peak2_side, 1, 1)

            tail_fit = QCheckBox("Extrapolate tail with LG fit", page)
            two_layout.addWidget(tail_fit, 2, 0, 1, 4)
            two_layout.addWidget(QLabel("Fit extent (0-1)"), 3, 0)
            fit_extent = QDoubleSpinBox(page)
            fit_extent.setRange(0.0, 1.0)
            fit_extent.setSingleStep(0.05)
            fit_extent.setValue(0.9)
            two_layout.addWidget(fit_extent, 3, 1)
            layout.addLayout(two_layout)

        folder_label = QLabel("Processing SAXS folder: —", page)
        folder_label.setWordWrap(True)
        layout.addWidget(folder_label)

        table = QTableWidget(page)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Sample", "Label"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setVisible(False)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.DoubleClicked | QAbstractItemView.EditTrigger.SelectedClicked)
        layout.addWidget(table, 1)

        btn_grid = QGridLayout()
        refresh_btn = QPushButton("Refresh list", page)
        refresh_btn.setToolTip("Reload SAXS samples from the reference folder.")
        refresh_btn.clicked.connect(self.refresh_list)
        btn_grid.addWidget(refresh_btn, 0, 0)
        rename_btn = QPushButton("Rename label", page)
        rename_btn.setToolTip("Rename the selected sample label (does not change the file name).")
        rename_btn.clicked.connect(self.rename_selected)
        btn_grid.addWidget(rename_btn, 0, 1)
        run_sel_btn = QPushButton("Run selected", page)
        run_sel_btn.setToolTip("Run anisotropy analysis on the selected sample.")
        run_sel_btn.clicked.connect(self.run_selected)
        btn_grid.addWidget(run_sel_btn, 1, 0)
        run_all_btn = QPushButton("Run all", page)
        run_all_btn.setToolTip("Run anisotropy analysis on all samples.")
        run_all_btn.clicked.connect(self.run_all)
        btn_grid.addWidget(run_all_btn, 1, 1)
        btn_grid.setColumnStretch(0, 1)
        btn_grid.setColumnStretch(1, 1)
        layout.addLayout(btn_grid)

        widgets = {
            "smoothing": smoothing,
            "sigma": sigma,
            "theta_min": theta_min,
            "theta_max": theta_max,
            "fast": fast_check,
            "persist": persist_defaults,
            "method_fit": method_fit,
            "method_direct": method_direct,
            "peak_side": peak_side,
            "mirror": mirror_check,
            "average": average_check,
            "average_group": average_group,
            "theta2_min": theta2_min,
            "theta2_max": theta2_max,
            "peak2_side": peak2_side,
            "tail_fit": tail_fit,
            "fit_extent": fit_extent,
            "folder_label": folder_label,
            "table": table,
        }
        method_fit.stateChanged.connect(lambda _=False, w=widgets: self._set_method("fitting", w))
        method_direct.stateChanged.connect(lambda _=False, w=widgets: self._set_method("direct", w))
        return widgets

    # ---- Actions ----
    def refresh_list(self) -> None:
        if not self._saxs_tab_widgets:
            return
        for w in self._saxs_tab_widgets.values():
            w["table"].setRowCount(0)
        self._saxs_samples = []
        project_path = self.main.project_path
        if not project_path or not os.path.isdir(project_path):
            for w in self._saxs_tab_widgets.values():
                w["folder_label"].setText("Processing SAXS folder: —")
            return
        saxs_folder = os.path.join(project_path, "SAXS")
        if not os.path.isdir(saxs_folder):
            for w in self._saxs_tab_widgets.values():
                w["folder_label"].setText("Processing SAXS folder: (not found)")
            return
        for w in self._saxs_tab_widgets.values():
            w["folder_label"].setText(f"Processing SAXS folder: {os.path.basename(project_path)}")

        sample_folders = [
            f for f in os.listdir(saxs_folder)
            if os.path.isdir(os.path.join(saxs_folder, f)) and not f.startswith("_") and not f.startswith(".")
        ]
        samples = []
        if sample_folders:
            for s in sorted(sample_folders, key=str.lower):
                samples.append({"name": s, "path": os.path.join(saxs_folder, s), "kind": "folder"})
        else:
            flat = [f for f in os.listdir(saxs_folder) if f.lower().endswith(".dat") and not f.startswith("_")]
            for f in sorted(flat, key=str.lower):
                samples.append({"name": f, "path": os.path.join(saxs_folder, f), "kind": "file"})
        self._saxs_samples = samples

        for w in self._saxs_tab_widgets.values():
            table = w["table"]
            for idx, s in enumerate(samples):
                table.insertRow(idx)
                item_name = QTableWidgetItem(s["name"])
                item_name.setFlags(item_name.flags() & ~Qt.ItemFlag.ItemIsEditable)
                table.setItem(idx, 0, item_name)
                item_label = QTableWidgetItem(self._clean_label(s["name"]))
                table.setItem(idx, 1, item_label)

    def rename_selected(self) -> None:
        w = self._active()
        table = w.get("table")
        if table is None:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self._saxs_samples):
            return
        new_label = (table.item(row, 1).text() if table.item(row, 1) else "").strip()
        if not new_label:
            self.main.logger.warning("SAXS rename: new name is empty.")
            return
        entry = self._saxs_samples[row]
        old_path = entry["path"]
        new_path = os.path.join(os.path.dirname(old_path), new_label)
        if os.path.exists(new_path):
            self.main.logger.error("SAXS rename: target already exists.")
            return
        try:
            os.rename(old_path, new_path)
            self.main.logger.info(f"SAXS renamed: {os.path.basename(old_path)} -> {new_label}")
            self.refresh_list()
        except Exception as e:
            self.main._log_exception("SAXS rename failed", e)

    def run_selected(self) -> None:
        w = self._active()
        table = w.get("table")
        if table is None:
            return
        row = table.currentRow()
        if row < 0 or row >= len(self._saxs_samples):
            return
        entry = self._saxs_samples[row]
        label_item = table.item(row, 1)
        label = label_item.text().strip() if label_item else self._clean_label(entry["name"])
        self.run_on_sample(entry["path"], label)

    def run_all(self) -> None:
        w = self._active()
        table = w.get("table")
        if table is None:
            return
        for idx, entry in enumerate(self._saxs_samples):
            label_item = table.item(idx, 1)
            label = label_item.text().strip() if label_item else self._clean_label(entry["name"])
            self.run_on_sample(entry["path"], label)

    def run_on_sample(self, sample_path: str, label: str) -> None:
        try:
            os.environ["MUDPAW_EMBED"] = "1"
            if "MUDRAW_SAXS_NO_PLOTS" in os.environ:
                os.environ.pop("MUDRAW_SAXS_NO_PLOTS", None)
            w = self._active()
            two_peak = w.get("theta2_min") is not None
            secondary = (
                (w["theta2_min"].value(), w["theta2_max"].value())
                if two_peak and w.get("theta2_min") is not None and w.get("theta2_max") is not None
                else None
            )
            from _Routines.SAXS.SAXS_data_processor_v4 import SAXS_data_processor
            fig = SAXS_data_processor(
                sample_path,
                label,
                float(w["smoothing"].value()),
                float(w["sigma"].value()),
                float(w["theta_min"].value()),
                float(w["theta_max"].value()),
                plots=not w["fast"].isChecked(),
                method="fitting" if w["method_fit"].isChecked() else "direct",
                mirror=w["mirror"].isChecked(),
                two_peak=two_peak,
                secondary_limits=secondary,
                flatten_tail_fit=w["tail_fit"].isChecked() if w.get("tail_fit") is not None else False,
                flatten_tail_extent=float(w["fit_extent"].value()) if w.get("fit_extent") is not None else 0.9,
                average_columns=w["average"].isChecked(),
                average_group=int(w["average_group"].value()),
                centering_mode="auto",
                weak_metric="A2",
                weak_threshold=None,
                side_mode=w["peak_side"].currentText(),
                side_mode2=w["peak2_side"].currentText() if w.get("peak2_side") is not None else "auto",
            )
            if fig is None and not w["fast"].isChecked():
                try:
                    if two_peak:
                        from _Routines.SAXS import P2468OAS_v4_2pk as hop2
                        if hop2.plt.get_fignums():
                            fig = hop2.plt.gcf()
                    else:
                        from _Routines.SAXS import P2468OAS_v4 as hop
                        if hop.plt.get_fignums():
                            fig = hop.plt.gcf()
                except Exception:
                    pass
            if fig is None and not w["fast"].isChecked():
                try:
                    import matplotlib._pylab_helpers as pylab_helpers
                    mgrs = pylab_helpers.Gcf.get_all_fig_managers()
                    if mgrs:
                        fig = mgrs[-1].canvas.figure
                except Exception:
                    pass
            if w["persist"].isChecked():
                self._save_defaults()
            if not w["fast"].isChecked() and fig is not None:
                self._display_figure(fig)
        except Exception as e:
            self.main._log_exception("SAXS processing failed", e)

    # ---- Helpers ----
    def _active(self) -> dict:
        if self.saxsTabs is not None and self.saxsTabs.currentIndex() == 1:
            return self._saxs_tab_widgets.get("two", {})
        return self._saxs_tab_widgets.get("single", {})

    def _set_method(self, method: str, widgets: dict | None = None) -> None:
        if widgets is None:
            widgets = self._active()
        method = (method or "fitting").lower()
        fit = (method == "fitting")
        try:
            widgets["method_fit"].blockSignals(True)
            widgets["method_direct"].blockSignals(True)
            widgets["method_fit"].setChecked(fit)
            widgets["method_direct"].setChecked(not fit)
        finally:
            widgets["method_fit"].blockSignals(False)
            widgets["method_direct"].blockSignals(False)

    def _clean_label(self, raw: str) -> str:
        stem = os.path.splitext(raw)[0]
        cleaned = re.sub(r"^azi[_-]?saxs[_-]?", "", stem, flags=re.IGNORECASE)
        cleaned = cleaned.lstrip("_.- ")
        return cleaned or stem

    def _load_defaults(self) -> None:
        defaults = {
            "smoothing": 0.04,
            "sigma": 0.05,
            "theta_min": 70.0,
            "theta_max": 170.0,
            "method": "fitting",
            "mirror": False,
            "lower2": 70.0,
            "upper2": 170.0,
            "tail_fit": False,
            "fit_extent": 0.9,
            "average": False,
            "average_group": 4,
            "centering_mode": "auto",
            "weak_metric": "A2",
            "weak_threshold": "",
            "side_mode": "auto",
            "side_mode2": "auto",
        }
        try:
            if self._saxs_defaults_path.exists():
                data = json.loads(self._saxs_defaults_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    defaults.update(data)
        except Exception:
            pass
        for w in self._saxs_tab_widgets.values():
            w["smoothing"].setValue(float(defaults["smoothing"]))
            w["sigma"].setValue(float(defaults["sigma"]))
            w["theta_min"].setValue(float(defaults["theta_min"]))
            w["theta_max"].setValue(float(defaults["theta_max"]))
            self._set_method(defaults["method"], w)
            w["mirror"].setChecked(bool(defaults["mirror"]))
            if w.get("theta2_min") is not None:
                w["theta2_min"].setValue(float(defaults["lower2"]))
            if w.get("theta2_max") is not None:
                w["theta2_max"].setValue(float(defaults["upper2"]))
            if w.get("tail_fit") is not None:
                w["tail_fit"].setChecked(bool(defaults["tail_fit"]))
            if w.get("fit_extent") is not None:
                w["fit_extent"].setValue(float(defaults["fit_extent"]))
            w["average"].setChecked(bool(defaults["average"]))
            w["average_group"].setValue(int(defaults["average_group"]))
            w["peak_side"].setCurrentText(str(defaults["side_mode"]))
            if w.get("peak2_side") is not None:
                w["peak2_side"].setCurrentText(str(defaults["side_mode2"]))

    def _save_defaults(self) -> None:
        w = self._active()
        payload = {
            "smoothing": float(w["smoothing"].value()),
            "sigma": float(w["sigma"].value()),
            "theta_min": float(w["theta_min"].value()),
            "theta_max": float(w["theta_max"].value()),
            "method": "fitting" if w["method_fit"].isChecked() else "direct",
            "mirror": bool(w["mirror"].isChecked()),
            "lower2": float(w.get("theta2_min").value()) if w.get("theta2_min") is not None else float(w["theta_min"].value()),
            "upper2": float(w.get("theta2_max").value()) if w.get("theta2_max") is not None else float(w["theta_max"].value()),
            "tail_fit": bool(w.get("tail_fit").isChecked()) if w.get("tail_fit") is not None else False,
            "fit_extent": float(w.get("fit_extent").value()) if w.get("fit_extent") is not None else 0.9,
            "average": bool(w["average"].isChecked()),
            "average_group": int(w["average_group"].value()),
            "centering_mode": "auto",
            "weak_metric": "A2",
            "weak_threshold": "",
            "side_mode": w["peak_side"].currentText(),
            "side_mode2": w.get("peak2_side").currentText() if w.get("peak2_side") is not None else "auto",
        }
        try:
            (BASE_DIR / "_Miscell").mkdir(parents=True, exist_ok=True)
            self._saxs_defaults_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass
