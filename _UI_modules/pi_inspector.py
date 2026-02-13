from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QEvent, QObject, Qt, QTimer, QThread
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QGridLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QFileDialog,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)


class PIInspector(QObject):
    def __init__(self, main) -> None:
        super().__init__(main)
        self.main = main
        self.fileList: QListWidget | None = None
        self._sample_paths: list[Path] = []
        self._page: QWidget | None = None
        self._current_sample: Path | None = None
        self._current_name: str | None = None
        self._gap_entry: QLineEdit | None = None
        self._fr_gap_entry: QLineEdit | None = None
        self._ref_label: QLabel | None = None
        self._sample_label: QLabel | None = None
        self._status_label: QLabel | None = None

        # Geometry state
        self._geom_data: np.ndarray | None = None
        self._geom_center: list[float] | None = None
        self._geom_inner: int | None = None
        self._geom_outer: int | None = None

    def build_inspector(self, page: QWidget) -> None:
        self._page = page
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Processing PI folder", page))
        self.btnRefresh = QPushButton("Refresh list", page)
        layout.addWidget(self.btnRefresh)
        self.fileList = QListWidget(page)
        self.fileList.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        layout.addWidget(self.fileList)

        # Cap list height to ~1/3 of inspector height
        self._sync_list_height()
        page.installEventFilter(self)

        tabs = QTabWidget(page)
        steady = QWidget(tabs)
        flow = QWidget(tabs)
        tabs.addTab(steady, "Steady State")
        tabs.addTab(flow, "Flow Reversal")

        steady_layout = QVBoxLayout(steady)
        flow_layout = QVBoxLayout(flow)

        self._ref_label = QLabel("Reference folder: —", steady)
        self._sample_label = QLabel("Sample name: —", steady)
        steady_layout.addWidget(self._ref_label)
        steady_layout.addWidget(self._sample_label)
        steady_layout.addWidget(QLabel("Enter gap value:", steady))
        self._gap_entry = QLineEdit(steady)
        self._gap_entry.setText("1")
        steady_layout.addWidget(self._gap_entry)
        btn_center = QPushButton("Center Geometry", steady)
        btn_triggers = QPushButton("Process Triggers", steady)
        btn_extract = QPushButton("Extract Data", steady)
        btn_adjust = QPushButton("Adjust Data", steady)
        steady_layout.addWidget(btn_center)
        steady_layout.addWidget(btn_triggers)
        steady_layout.addWidget(btn_extract)
        steady_layout.addWidget(btn_adjust)
        self._status_label = QLabel("", steady)
        steady_layout.addWidget(self._status_label)

        # Geometry controls for live plot (in inspector)
        geom_grid = QGridLayout()
        geom_grid.addWidget(QLabel("Geometry controls:"), 0, 0, 1, 4)
        btn_left = QPushButton("←")
        btn_right = QPushButton("→")
        btn_up = QPushButton("↑")
        btn_down = QPushButton("↓")
        btn_inner_plus = QPushButton("Inner +")
        btn_inner_minus = QPushButton("Inner -")
        btn_outer_plus = QPushButton("Outer +")
        btn_outer_minus = QPushButton("Outer -")
        btn_save_geom = QPushButton("Save Geometry")

        geom_grid.addWidget(btn_up, 1, 1)
        geom_grid.addWidget(btn_left, 2, 0)
        geom_grid.addWidget(btn_right, 2, 2)
        geom_grid.addWidget(btn_down, 3, 1)
        geom_grid.addWidget(btn_inner_plus, 1, 4)
        geom_grid.addWidget(btn_inner_minus, 2, 4)
        geom_grid.addWidget(btn_outer_plus, 1, 5)
        geom_grid.addWidget(btn_outer_minus, 2, 5)
        geom_grid.addWidget(btn_save_geom, 3, 4, 1, 2)
        steady_layout.addLayout(geom_grid)
        steady_layout.addStretch(1)

        flow_layout.addWidget(QLabel("Enter gap value:", flow))
        self._fr_gap_entry = QLineEdit(flow)
        self._fr_gap_entry.setText("1")
        flow_layout.addWidget(self._fr_gap_entry)
        btn_center_fr = QPushButton("Center Geometry", flow)
        btn_extract_fr = QPushButton("Extract Data", flow)
        btn_adjust_fr = QPushButton("Adjust Data", flow)
        flow_layout.addWidget(btn_center_fr)
        flow_layout.addWidget(btn_extract_fr)
        flow_layout.addWidget(btn_adjust_fr)
        flow_layout.addStretch(1)

        layout.addWidget(tabs, 1)

        self.btnRefresh.clicked.connect(self.refresh_list)
        self.fileList.itemSelectionChanged.connect(self._on_sample_selected)
        btn_center.clicked.connect(lambda: self._center_geometry())
        btn_triggers.clicked.connect(self._process_triggers)
        btn_extract.clicked.connect(lambda: self._extract_data("steady_state"))
        btn_adjust.clicked.connect(lambda: self._adjust_data("steady_state"))
        btn_center_fr.clicked.connect(lambda: self._center_geometry())
        btn_extract_fr.clicked.connect(lambda: self._extract_data("flow_reversal"))
        btn_adjust_fr.clicked.connect(lambda: self._adjust_data("flow_reversal"))

        btn_left.clicked.connect(lambda: self._nudge_center(-1, 0))
        btn_right.clicked.connect(lambda: self._nudge_center(1, 0))
        btn_up.clicked.connect(lambda: self._nudge_center(0, -1))
        btn_down.clicked.connect(lambda: self._nudge_center(0, 1))
        btn_inner_plus.clicked.connect(lambda: self._nudge_radius(inner_delta=1))
        btn_inner_minus.clicked.connect(lambda: self._nudge_radius(inner_delta=-1))
        btn_outer_plus.clicked.connect(lambda: self._nudge_radius(outer_delta=1))
        btn_outer_minus.clicked.connect(lambda: self._nudge_radius(outer_delta=-1))
        btn_save_geom.clicked.connect(self._save_geometry)

    def refresh_list(self) -> None:
        pi_dir = self._resolve_pi_dir()
        if pi_dir is None:
            self._log("[PI] PI folder not found for current reference.")
            return
        if self.fileList is None:
            self._log("[PI] fileList is None (inspector not built yet?)")
            return
        try:
            self.fileList.clear()
            self._sample_paths = self._list_samples(pi_dir)
            self._log(f"[PI] Found {len(self._sample_paths)} folder(s).")
            for sample in self._sample_paths:
                item = QListWidgetItem(sample.name)
                item.setData(Qt.ItemDataRole.UserRole, str(sample))
                self.fileList.addItem(item)
            if not self._sample_paths:
                self.fileList.addItem("(empty)")
        except Exception as exc:
            self._log(f"[PI] Refresh failed: {exc}")

    def _list_samples(self, pi_dir: Path) -> list[Path]:
        if not pi_dir.exists():
            return []
        try:
            names = [n for n in os.listdir(pi_dir) if not n.startswith(".")]
        except Exception as exc:
            self._log(f"[PI] Failed to list PI folder: {exc}")
            return []
        samples = [
            pi_dir / n for n in names
            if os.path.isdir(os.path.join(pi_dir, n))
            and not n.startswith("_")
            and not n.endswith(".dgraph")
        ]
        return sorted(samples, key=lambda p: p.name.lower())

    def _selected_samples(self) -> list[Path]:
        if not self.fileList:
            return []
        items = self.fileList.selectedItems()
        if not items:
            return []
        out: list[Path] = []
        for item in items:
            p = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(p, str):
                out.append(Path(p))
            else:
                out.append(Path(item.text()))
        return out

    def _on_sample_selected(self) -> None:
        samples = self._selected_samples()
        if not samples:
            self._current_sample = None
            self._current_name = None
            if self._sample_label:
                self._sample_label.setText("Sample name: —")
            return
        sample = samples[0]
        self._current_sample = sample
        self._current_name = sample.name
        if self._sample_label:
            self._sample_label.setText(f"Sample name: {self._current_name}")
        if self._ref_label:
            self._ref_label.setText(f"Reference folder: {self._resolve_reference_root() or '—'}")

    def _log(self, msg: str) -> None:
        if QThread.currentThread() is not self.main.thread():
            QTimer.singleShot(0, lambda m=msg: self._log(m))
            return
        if hasattr(self.main, "logger"):
            self.main.logger.info(msg)

    def _resolve_pi_dir(self) -> Path | None:
        base = self.main.project_path
        if not base:
            return None
        if os.path.isdir(base) and os.path.basename(base).lower() == "pi":
            return Path(base)
        if os.path.basename(base) == "Inputs (reference)":
            base = os.path.dirname(base)
        direct = os.path.join(base, "PI")
        if os.path.isdir(direct):
            return Path(direct)
        candidate = os.path.join(base, "Inputs (reference)", "PI")
        if os.path.isdir(candidate):
            return Path(candidate)
        return None

    def _resolve_reference_root(self) -> str | None:
        base = self.main.project_path
        if not base:
            return None
        if os.path.basename(base).lower() == "pi":
            return os.path.dirname(base)
        if os.path.basename(base) == "Inputs (reference)":
            return os.path.dirname(base)
        return base

    def _sync_list_height(self) -> None:
        if not self.fileList or not self._page:
            return
        h = self._page.height() if self._page.height() > 0 else 600
        self.fileList.setMaximumHeight(max(120, int(h * 0.33)))

    def eventFilter(self, obj: QObject, event: QEvent) -> bool:
        if obj is self._page and event.type() == QEvent.Type.Resize:
            self._sync_list_height()
        return False

    # ---- Geometry (Qt-native with central canvas) ----
    def _find_csv_folder_with_files(self, base_path: Path) -> Path | None:
        for root, dirs, files in os.walk(base_path):
            if "_Temp_processed" in dirs:
                dirs.remove("_Temp_processed")
            if any(("axis" in f or "retard" in f) and f.endswith(".csv") for f in files):
                return Path(root)
        return None

    def _find_first_csv(self, folder: Path) -> Path | None:
        for f in sorted(os.listdir(folder)):
            if ("axis" in f or "retard" in f) and f.endswith(".csv"):
                return folder / f
        return None

    def _load_existing_geometry(self, output_dir: Path):
        json_file = output_dir / "_Geometry_positioning.json"
        txt_file = output_dir / "_Geometry_positioning.txt"
        if json_file.exists():
            try:
                import json as _json
                data = _json.loads(json_file.read_text())
                return (
                    int(data.get("offset_x", 0)),
                    int(data.get("offset_y", 0)),
                    int(data.get("Inner_initial", 86)),
                    int(data.get("Outer_initial", 126)),
                )
            except Exception:
                pass
        if txt_file.exists():
            try:
                vals = {}
                for line in txt_file.read_text().splitlines():
                    if ":" in line:
                        k, v = line.split(":", 1)
                        vals[k.strip()] = v.strip()
                return (
                    int(vals.get("offset_x", 0)),
                    int(vals.get("offset_y", 0)),
                    int(vals.get("Inner_initial", 86)),
                    int(vals.get("Outer_initial", 126)),
                )
            except Exception:
                pass
        return None

    def _center_geometry(self) -> None:
        sample_path, _name = self._require_sample()
        if sample_path is None:
            return
        csv_root = self._find_csv_folder_with_files(sample_path)
        if csv_root is None:
            QMessageBox.warning(self._page, "PI", "No axis/retard CSV files found.")
            return
        first_csv = self._find_first_csv(csv_root)
        if first_csv is None:
            QMessageBox.warning(self._page, "PI", "No axis/retard CSV files found.")
            return
        try:
            data = np.loadtxt(first_csv, delimiter=",")
        except Exception as exc:
            QMessageBox.critical(self._page, "PI", f"Failed to load CSV: {exc}")
            return

        height, width = data.shape
        temp = self._temp_processed(sample_path)
        loaded = self._load_existing_geometry(temp)
        if loaded:
            offset_x, offset_y, inner, outer = loaded
        else:
            offset_x, offset_y, inner, outer = 0, 0, 86, 126

        center_x = width // 2 + offset_x
        center_y = height // 2 + offset_y

        self._geom_data = data
        self._geom_center = [center_x, center_y]
        self._geom_inner = int(inner)
        self._geom_outer = int(outer)
        self._render_geometry()

    def _render_geometry(self) -> None:
        if self._geom_data is None or self._geom_center is None:
            return
        self.main.pi_show_geometry(
            self._geom_data,
            self._geom_center,
            int(self._geom_inner or 0),
            int(self._geom_outer or 0),
        )

    def _nudge_center(self, dx: int, dy: int) -> None:
        if self._geom_data is None or self._geom_center is None:
            QMessageBox.information(self._page, "PI", "Run Center Geometry first.")
            return
        height, width = self._geom_data.shape
        x, y = self._geom_center
        x = max(0, min(width, x + dx))
        y = max(0, min(height, y + dy))
        self._geom_center = [x, y]
        self._render_geometry()

    def _nudge_radius(self, inner_delta: int = 0, outer_delta: int = 0) -> None:
        if self._geom_data is None:
            QMessageBox.information(self._page, "PI", "Run Center Geometry first.")
            return
        height, width = self._geom_data.shape
        max_radius = min(height, width) // 2
        inner = int(self._geom_inner or 1)
        outer = int(self._geom_outer or inner + 1)
        inner = max(1, min(max_radius, inner + inner_delta))
        outer = max(inner, min(max_radius, outer + outer_delta))
        self._geom_inner = inner
        self._geom_outer = outer
        self._render_geometry()

    def _save_geometry(self) -> None:
        sample_path, _name = self._require_sample()
        if sample_path is None:
            return
        if self._geom_data is None or self._geom_center is None:
            QMessageBox.information(self._page, "PI", "Run Center Geometry first.")
            return
        height, width = self._geom_data.shape
        center_x, center_y = self._geom_center
        offset_x = int(center_x - width // 2)
        offset_y = int(center_y - height // 2)
        inner = int(self._geom_inner or 0)
        outer = int(self._geom_outer or 0)
        temp = self._temp_processed(sample_path)
        txt = temp / "_Geometry_positioning.txt"
        jsn = temp / "_Geometry_positioning.json"
        try:
            txt.write_text(
                f"offset_x: {offset_x}\n"
                f"offset_y: {offset_y}\n"
                f"Inner_initial: {inner}\n"
                f"Outer_initial: {outer}\n"
            )
        except Exception as exc:
            QMessageBox.critical(self._page, "PI", f"Failed to write geometry TXT: {exc}")
            return
        try:
            import json as _json
            jsn.write_text(
                _json.dumps(
                    {
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "Inner_initial": inner,
                        "Outer_initial": outer,
                    },
                    indent=2,
                )
            )
        except Exception as exc:
            QMessageBox.critical(self._page, "PI", f"Failed to write geometry JSON: {exc}")
            return
        self._log(f"[PI] Geometry saved to {txt}")

    # ---- PI actions (Qt-native) ----
    def _require_sample(self) -> tuple[Path | None, str | None]:
        if self._current_sample is None:
            QMessageBox.warning(self._page, "PI", "Select a sample first.")
            return None, None
        name = self._current_name or self._current_sample.name
        return self._current_sample, name

    def _sample_root(self, sample_path: Path) -> Path:
        cur = sample_path
        if cur.is_file():
            cur = cur.parent
        while True:
            parent = cur.parent
            if parent == cur:
                return cur
            if parent.name.lower() == "pi":
                return cur
            cur = parent

    def _temp_processed(self, sample_path: Path) -> Path:
        root = self._sample_root(sample_path)
        temp = root / "_Temp_processed"
        temp.mkdir(parents=True, exist_ok=True)
        return temp

    def _processed_root(self, sample_path: Path) -> Path:
        ref = self._resolve_reference_root() or self._sample_root(sample_path).parent
        processed = Path(ref) / "_Processed"
        processed.mkdir(parents=True, exist_ok=True)
        return processed

    def _process_triggers(self) -> None:
        sample_path, name = self._require_sample()
        if sample_path is None:
            return
        temp = self._temp_processed(sample_path)
        triggers = temp / "_Triggers"
        if triggers.exists():
            resp = QMessageBox.question(
                self._page,
                "Overwrite Triggers",
                "Triggers already exist. Overwrite?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if resp != QMessageBox.StandardButton.Yes:
                self._log("[PI] Trigger processing skipped (existing triggers).")
                return
        sample_root = self._sample_root(sample_path)
        start_dir = str(sample_root)
        picked, _ = QFileDialog.getOpenFileName(
            self._page,
            "Select Source Timestamp File (SAXS or Anton Paar)",
            start_dir,
            "All files (*.*)",
        )
        if not picked:
            self._log("[PI] Trigger processing canceled.")
            return
        script = Path(__file__).resolve().parents[1] / "_Routines" / "Time_stamper_v1.py"
        if not script.exists():
            QMessageBox.critical(self._page, "PI", f"Missing script: {script}")
            return
        self._log(f"[PI] Process triggers for {name}: {temp}")
        threading.Thread(
            target=self._run_subprocess,
            args=([sys.executable, str(script), str(sample_root), picked],),
            daemon=True,
        ).start()

    def _extract_data(self, mode: str) -> None:
        sample_path, name = self._require_sample()
        if sample_path is None:
            return
        if self._status_label is not None:
            self._status_label.setText("Extracting…")
        temp = self._temp_processed(sample_path)
        job_id = None
        job_status = None
        if mode != "flow_reversal":
            job_id = f"pi-extract-{name}"
            job_widget = QWidget(self._page)
            job_layout = QVBoxLayout(job_widget)
            job_layout.setContentsMargins(8, 8, 8, 8)
            job_label = QLabel(f"PI Extract: {name}", job_widget)
            job_status = QLabel("Running…", job_widget)
            job_layout.addWidget(job_label)
            job_layout.addWidget(job_status)
            if hasattr(self.main, "_add_processing_job"):
                self.main._add_processing_job(job_id, f"PI Extract: {name}", job_widget)
        self._log(f"[PI] Extract start ({mode}) for {name}")
        gap = self._read_gap(mode)
        if gap is None:
            self._log("[PI] Extract aborted: invalid gap")
            if job_id and hasattr(self.main, "_remove_processing_job"):
                self.main._remove_processing_job(job_id)
            return
        geometry_file = temp / "_Geometry_positioning.txt"
        if not geometry_file.exists():
            QMessageBox.warning(self._page, "PI", "Geometry positioning file not found. Run Center Geometry first.")
            self._log(f"[PI] Extract aborted: missing geometry file at {geometry_file}")
            if job_status is not None:
                job_status.setText("Missing geometry.")
            if job_id and hasattr(self.main, "_remove_processing_job"):
                self.main._remove_processing_job(job_id)
            return
        triggers = temp / "_Triggers"
        if mode != "flow_reversal" and not triggers.exists():
            QMessageBox.warning(self._page, "PI", "Triggers file not found. Run Process Triggers first.")
            self._log(f"[PI] Extract aborted: missing triggers at {triggers}")
            if job_status is not None:
                job_status.setText("Missing triggers.")
            if job_id and hasattr(self.main, "_remove_processing_job"):
                self.main._remove_processing_job(job_id)
            return
        try:
            geo = self._read_geometry(geometry_file)
        except Exception as exc:
            QMessageBox.critical(self._page, "PI", f"Failed to read geometry file: {exc}")
            self._log(f"[PI] Extract aborted: bad geometry file: {exc}")
            if job_id and hasattr(self.main, "_remove_processing_job"):
                self.main._remove_processing_job(job_id)
            return

        csv_folder = self._sample_root(sample_path)
        if mode == "flow_reversal":
            script1 = Path(__file__).resolve().parents[1] / "_Routines" / "PI" / "Data_extracter_flow_reversal.py"
            if not script1.exists():
                QMessageBox.critical(self._page, "PI", f"Missing script: {script1}")
                return

            args1 = [
                sys.executable,
                str(script1),
                str(csv_folder),
                str(gap),
                str(geo["offset_x"]),
                str(geo["offset_y"]),
                str(geo["Inner_initial"]),
                str(geo["Outer_initial"]),
            ]
            threading.Thread(target=self._run_subprocess, args=(args1,), daemon=True).start()
            return

        # Steady-state: run in-process and render in central canvas
        heartbeat_stop = threading.Event()

        def heartbeat_logger():
            elapsed = 0
            while not heartbeat_stop.wait(10):
                elapsed += 10
                self._log(f"[PI] Extract still running ({mode}) for {name}: {elapsed}s elapsed")

        elapsed_timer: QTimer | None = None

        def _stop_elapsed_timer() -> None:
            if elapsed_timer is not None:
                try:
                    elapsed_timer.stop()
                except Exception:
                    pass
            heartbeat_stop.set()

        def run():
            try:
                mod_path = Path(__file__).resolve().parents[1] / "_Routines" / "PI" / "Data_extracter_v2_1.py"
                import importlib.util
                spec = importlib.util.spec_from_file_location("pi_extract", mod_path)
                if spec is None or spec.loader is None:
                    self._log("[PI] Failed to load extractor module.")
                    return
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                start = time.time()
                self._log(
                    f"[PI] Running extract_space_time gap={gap} offsets=({geo['offset_x']},{geo['offset_y']})"
                    f" radii=({geo['Inner_initial']},{geo['Outer_initial']}) csv_root={csv_folder}"
                )
                # Force single-process to avoid silent hangs with ProcessPool in some environments.
                result = mod.extract_space_time(
                    str(csv_folder),
                    str(gap),
                    str(geo["offset_x"]),
                    str(geo["offset_y"]),
                    str(geo["Inner_initial"]),
                    str(geo["Outer_initial"]),
                    str(triggers),
                    plot=False,
                    use_multiprocessing=False,
                )
                self._log(f"[PI] extract_space_time completed in {time.time() - start:.1f}s")
                if not result:
                    def _fail():
                        _stop_elapsed_timer()
                        if self._status_label is not None:
                            self._status_label.setText("Extract failed.")
                        if job_status is not None:
                            job_status.setText("Failed.")
                        if job_id and hasattr(self.main, "_remove_processing_job"):
                            self.main._remove_processing_job(job_id)
                    self._log("[PI] Extract returned no result.")
                    QTimer.singleShot(0, _fail)
                    return
                circle = result["circle"]
                line = result["line"]
                interval = result["interval"]
                n_intervals = result["n_intervals"]

                def _render():
                    _stop_elapsed_timer()
                    self.main.pi_show_extract(circle, line, interval, n_intervals)
                    self._log("[PI] Extract render complete.")
                    if self._status_label is not None:
                        self._status_label.setText("Extract complete.")
                    if job_status is not None:
                        job_status.setText("Complete.")
                    if job_id and hasattr(self.main, "_remove_processing_job"):
                        self.main._remove_processing_job(job_id)

                QTimer.singleShot(0, _render)
            except Exception as exc:
                tb = traceback.format_exc(limit=4)
                self._log(f"[PI] Extract failed: {exc}\n{tb}")
                def _fail_exc():
                    _stop_elapsed_timer()
                    if self._status_label is not None:
                        self._status_label.setText("Extract failed.")
                    if job_status is not None:
                        job_status.setText("Failed.")
                    if job_id and hasattr(self.main, "_remove_processing_job"):
                        self.main._remove_processing_job(job_id)
                QTimer.singleShot(0, _fail_exc)

        # Start a lightweight elapsed-time ticker so the user can see progress.
        try:
            elapsed_timer = QTimer(self._page)
            start_ts = time.time()
            def _tick():
                if job_status is not None:
                    secs = int(time.time() - start_ts)
                    job_status.setText(f"Running… {secs}s")
            elapsed_timer.timeout.connect(_tick)
            elapsed_timer.start(1000)
        except Exception:
            elapsed_timer = None

        self._log("[PI] Spawning extract thread (steady_state)")
        threading.Thread(target=heartbeat_logger, daemon=True).start()
        threading.Thread(target=run, daemon=True).start()

    def _adjust_data(self, mode: str) -> None:
        sample_path, name = self._require_sample()
        if sample_path is None:
            return
        gap = self._read_gap(mode)
        if gap is None:
            return
        temp = self._temp_processed(sample_path)
        if mode == "flow_reversal":
            script = Path(__file__).resolve().parents[1] / "_Routines" / "PI" / "Data_adjuster_flow_reversal.py"
        else:
            script = Path(__file__).resolve().parents[1] / "_Routines" / "PI" / "Data_adjuster_v4.py"
        if not script.exists():
            QMessageBox.critical(self._page, "PI", f"Missing script: {script}")
            return

        def run():
            self._run_subprocess([sys.executable, str(script), str(temp), str(name), str(gap)])
            out = temp / f"_output_PI_{name}.json"
            if out.exists():
                dest = self._processed_root(sample_path) / out.name
                try:
                    dest.write_bytes(out.read_bytes())
                except Exception as exc:
                    self._log(f"[PI] Failed to mirror output: {exc}")

        threading.Thread(target=run, daemon=True).start()

    def _read_gap(self, mode: str) -> float | None:
        entry = self._fr_gap_entry if mode == "flow_reversal" else self._gap_entry
        if entry is None:
            return None
        try:
            return float(entry.text().strip())
        except Exception:
            QMessageBox.warning(self._page, "PI", "Invalid gap value. Please enter a numeric value.")
            return None

    def _read_geometry(self, path: Path) -> dict:
        geo = {}
        for line in path.read_text().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                geo[k] = int(v)
        for key in ("offset_x", "offset_y", "Inner_initial", "Outer_initial"):
            if key not in geo:
                raise ValueError(f"Missing {key} in geometry file")
        return geo

    def _run_subprocess(self, args: list[str]) -> None:
        try:
            proc = subprocess.run(args, capture_output=True, text=True)
            if proc.stdout:
                self._log(proc.stdout.strip())
            if proc.stderr:
                self._log(proc.stderr.strip())
            if proc.returncode != 0:
                self._log(f"[PI] Command failed: {args}")
        except Exception as exc:
            self._log(f"[PI] Subprocess failed: {exc}")
