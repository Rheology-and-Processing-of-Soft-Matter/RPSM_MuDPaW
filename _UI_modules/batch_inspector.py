from __future__ import annotations

import os
import subprocess
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QGridLayout,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
)

BASE_DIR = Path(__file__).resolve().parents[1]


class BatchInspector:
    def __init__(self, main) -> None:
        self.main = main

    def _default_reference_dir(self) -> str:
        ref = getattr(self.main, "project_path", None)
        if ref and Path(ref).exists():
            return str(ref)
        cand = BASE_DIR / "_Miscell" / "last_folder.txt"
        try:
            txt = cand.read_text(encoding="utf-8").strip()
            if txt:
                p = Path(txt).expanduser()
                if p.exists():
                    return str(p)
        except Exception:
            pass
        return str(Path.home())

    def _default_overfolder_dir(self) -> str:
        current = getattr(self, "editOverfolder", None)
        if current:
            txt = current.text().strip()
            if txt:
                return txt
        ref = self._default_reference_dir()
        try:
            return str(Path(ref).parent)
        except Exception:
            return ref

    def build_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(12)
        layout.setContentsMargins(12, 12, 12, 12)

        # Overfolder row
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Overfolder:", page))
        self.editOverfolder = QLineEdit(page)
        self.editOverfolder.setMinimumWidth(180)
        self.editOverfolder.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        row.addWidget(self.editOverfolder, 1)
        self.btnBrowse = QPushButton("Browse…", page)
        self.btnVerify = QPushButton("Verify setup…", page)
        row.addWidget(self.btnBrowse)
        row.addWidget(self.btnVerify)
        layout.addLayout(row)

        layout.addSpacing(4)

        # Options row
        opt_row = QGridLayout()
        opt_row.setHorizontalSpacing(10)
        opt_row.setVerticalSpacing(6)
        self.chkSaxs = QCheckBox("SAXS", page)
        self.chkSaxs.setChecked(True)
        self.chkRheo = QCheckBox("Rheology", page)
        self.chkRheo.setChecked(True)
        self.chkSkipFresh = QCheckBox("Skip if fresh", page)
        self.chkDryRun = QCheckBox("Dry-run", page)
        opt_row.addWidget(self.chkSaxs, 0, 0)
        opt_row.addWidget(self.chkRheo, 0, 1)
        opt_row.addWidget(self.chkSkipFresh, 0, 2)
        opt_row.addWidget(self.chkDryRun, 0, 3)
        opt_row.addWidget(QLabel("Log level:", page), 1, 0)
        self.comboLog = QComboBox(page)
        self.comboLog.addItems(["INFO", "DEBUG", "WARNING", "ERROR"])
        self.comboLog.setMinimumWidth(110)
        opt_row.addWidget(self.comboLog, 1, 1)
        opt_row.addWidget(QLabel("Workers:", page), 1, 2)
        self.spinWorkers = QSpinBox(page)
        self.spinWorkers.setRange(1, 64)
        self.spinWorkers.setValue(8)
        self.spinWorkers.setMinimumWidth(70)
        opt_row.addWidget(self.spinWorkers, 1, 3)
        opt_row.setColumnStretch(4, 1)
        layout.addLayout(opt_row)
        layout.addSpacing(6)

        # Advanced group
        grp = QGroupBox("Advanced — Processor parameters", page)
        g = QVBoxLayout(grp)
        g.setSpacing(10)

        # SAXS parameters
        grp_saxs = QGroupBox("SAXS parameters", grp)
        s_layout = QGridLayout(grp_saxs)
        s_layout.setHorizontalSpacing(10)
        s_layout.setVerticalSpacing(6)
        self.saxsSmoothing = QLineEdit("0.0", page)
        self.saxsSigma = QLineEdit("0.5", page)
        self.saxsThetaMin = QLineEdit("75.0", page)
        self.saxsThetaMax = QLineEdit("110.0", page)
        for field in (self.saxsSmoothing, self.saxsSigma, self.saxsThetaMin, self.saxsThetaMax):
            field.setMinimumWidth(80)
            field.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        s_layout.addWidget(QLabel("Smoothing", page), 0, 0)
        s_layout.addWidget(self.saxsSmoothing, 0, 1)
        s_layout.addWidget(QLabel("Sigma", page), 0, 2)
        s_layout.addWidget(self.saxsSigma, 0, 3)
        s_layout.addWidget(QLabel("θ min", page), 1, 0)
        s_layout.addWidget(self.saxsThetaMin, 1, 1)
        s_layout.addWidget(QLabel("θ max", page), 1, 2)
        s_layout.addWidget(self.saxsThetaMax, 1, 3)
        s_layout.setColumnStretch(4, 1)
        mrow = QGridLayout()
        mrow.setHorizontalSpacing(10)
        mrow.setVerticalSpacing(4)
        mrow.addWidget(QLabel("Method:", page), 0, 0)
        self.saxsFit = QCheckBox("Fitting", page)
        self.saxsDirect = QCheckBox("Direct", page)
        self.saxsDirect.setChecked(True)
        self.saxsNoPlots = QCheckBox("No plots", page)
        self.saxsMirror = QCheckBox("Mirror", page)
        mrow.addWidget(self.saxsFit, 0, 1)
        mrow.addWidget(self.saxsDirect, 0, 2)
        mrow.addWidget(self.saxsNoPlots, 1, 1)
        mrow.addWidget(self.saxsMirror, 1, 2)
        mrow.setColumnStretch(3, 1)
        s_layout.addLayout(mrow, 2, 0, 1, 4)
        g.addWidget(grp_saxs)

        # Rheology parameters
        grp_rheo = QGroupBox("Rheology parameters", grp)
        r_layout = QGridLayout(grp_rheo)
        r_layout.setHorizontalSpacing(10)
        r_layout.setVerticalSpacing(6)
        self.comboRheoMode = QComboBox(page)
        self.comboRheoMode.addItems(["triggered", "nontriggered", "other"])
        self.rheoSteady = QLineEdit("10.0", page)
        self.rheoNoPlots = QCheckBox("No plots", page)
        self.comboRheoMode.setMinimumWidth(140)
        self.rheoSteady.setMinimumWidth(80)
        r_layout.addWidget(QLabel("Mode", page), 0, 0)
        r_layout.addWidget(self.comboRheoMode, 0, 1)
        r_layout.addWidget(QLabel("Steady window (s)", page), 1, 0)
        r_layout.addWidget(self.rheoSteady, 1, 1)
        r_layout.addWidget(self.rheoNoPlots, 2, 0, 1, 2)
        r_layout.setColumnStretch(2, 1)
        g.addWidget(grp_rheo)
        layout.addWidget(grp)
        layout.addSpacing(6)

        # DataGraph export
        grp_dg = QGroupBox("DataGraph Export (optional)", page)
        dg = QGridLayout(grp_dg)
        dg.setHorizontalSpacing(10)
        dg.setVerticalSpacing(6)
        dg.addWidget(QLabel("Template (.dgraph):", page), 0, 0)
        self.editTemplate = QLineEdit(page)
        self.editTemplate.setMinimumWidth(200)
        self.editTemplate.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dg.addWidget(self.editTemplate, 0, 1)
        self.btnBrowseTemplate = QPushButton("Browse…", page)
        dg.addWidget(self.btnBrowseTemplate, 0, 2)

        dg.addWidget(QLabel("DG CLI:", page), 1, 0)
        self.editDgCli = QLineEdit("/Applications/DataGraph.app/Contents/Library/dgraph", page)
        self.editDgCli.setMinimumWidth(240)
        self.editDgCli.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        dg.addWidget(self.editDgCli, 1, 1, 1, 2)
        self.chkWriteDg = QCheckBox("Write to DataGraph", page)
        dg.addWidget(self.chkWriteDg, 2, 1, 1, 2)

        self.chkSingleFile = QCheckBox("Single file (combine all processed data into one export)", page)
        self.chkSingleFile.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        dg.addWidget(self.chkSingleFile, 3, 0, 1, 3)
        dg.setColumnStretch(1, 1)
        layout.addWidget(grp_dg)
        layout.addSpacing(6)

        # Footer buttons
        btns = QHBoxLayout()
        btns.setSpacing(8)
        self.btnWriteDgOnly = QPushButton("Write DataGraph only", page)
        self.btnSaveFlagged = QPushButton("Save flagged graphs", page)
        self.btnStop = QPushButton("Stop", page)
        self.btnEngage = QPushButton("Engage", page)
        btns.addWidget(self.btnWriteDgOnly)
        btns.addWidget(self.btnSaveFlagged)
        btns.addWidget(self.btnStop)
        btns.addWidget(self.btnEngage)
        layout.addLayout(btns)
        layout.addStretch(1)

        self._load_settings()

        # Wiring
        self.btnBrowse.clicked.connect(self._browse_overfolder)
        self.btnBrowseTemplate.clicked.connect(self._browse_template)
        self.btnEngage.clicked.connect(self._run_batch)
        self.btnWriteDgOnly.clicked.connect(self._run_dg_only)
        self.btnSaveFlagged.clicked.connect(lambda: self._log("Save flagged graphs not implemented yet."))
        self.btnVerify.clicked.connect(lambda: self._log("Verify setup not implemented yet."))

    def _browse_overfolder(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main,
            "Select overfolder",
            self._default_overfolder_dir(),
        )
        if folder:
            self.editOverfolder.setText(folder)

    def _browse_template(self) -> None:
        start = self._default_reference_dir()
        path, _ = QFileDialog.getOpenFileName(
            self.main,
            "Select template",
            start,
            "DataGraph (*.dgraph)",
        )
        if path:
            self.editTemplate.setText(path)

    def _run_batch(self) -> None:
        overfolder = self.editOverfolder.text().strip()
        if not overfolder:
            self._log("Select an overfolder first.")
            return

        modalities = "both"
        if self.chkSaxs.isChecked() and not self.chkRheo.isChecked():
            modalities = "saxs"
        if self.chkRheo.isChecked() and not self.chkSaxs.isChecked():
            modalities = "rheology"
        if not self.chkSaxs.isChecked() and not self.chkRheo.isChecked():
            modalities = "none"

        script = BASE_DIR / "_Batch processor" / "orchestrate_overfolder.py"
        cmd = [
            os.environ.get("PYTHON", "python3"),
            str(script),
            overfolder,
            "--modalities",
            modalities,
            "--workers",
            str(self.spinWorkers.value()),
            "--saxs-smoothing",
            self.saxsSmoothing.text().strip() or "0.04",
            "--saxs-sigma",
            self.saxsSigma.text().strip() or "0.5",
            "--saxs-theta-min",
            self.saxsThetaMin.text().strip() or "2.0",
            "--saxs-theta-max",
            self.saxsThetaMax.text().strip() or "180.0",
            "--saxs-method",
            "direct" if self.saxsDirect.isChecked() else "fitting",
        ]
        if self.chkSkipFresh.isChecked():
            cmd.append("--skip-fresh")
        if self.chkDryRun.isChecked():
            cmd.append("--dry-run")
        if self.saxsNoPlots.isChecked():
            cmd.append("--saxs-no-plots")
        if self.saxsMirror.isChecked():
            cmd.append("--saxs-mirror")
        if self.rheoNoPlots.isChecked():
            cmd.append("--rheo-no-plots")
        if self.chkWriteDg.isChecked():
            cmd.append("--write-dg")
            if self.editTemplate.text().strip():
                cmd.extend(["--dg-template", self.editTemplate.text().strip()])
            if self.editDgCli.text().strip():
                cmd.extend(["--dg-cli", self.editDgCli.text().strip()])
            if self.chkSingleFile.isChecked():
                cmd.append("--dg-single-file")

        self._save_settings()
        self._log(f"[Batch] Running: {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout or []:
                self._log(line.rstrip())
        except Exception as e:
            self._log(f"[Batch] Failed: {e}")

    def _run_dg_only(self) -> None:
        """Run DataGraph writer only (no SAXS/Rheology processing)."""
        overfolder = self.editOverfolder.text().strip()
        if not overfolder:
            self._log("Select an overfolder first.")
            return
        tpl = self.editTemplate.text().strip()
        if not tpl:
            self._log("Provide a .dgraph template to enable DataGraph writing.")
            return
        cli = self.editDgCli.text().strip()
        if not cli:
            self._log("Provide the DataGraph CLI path first.")
            return

        script = BASE_DIR / "_Batch processor" / "orchestrate_overfolder.py"
        cmd = [
            os.environ.get("PYTHON", "python3"),
            str(script),
            overfolder,
            "--modalities",
            "none",
            "--workers",
            str(self.spinWorkers.value()),
            "--write-dg",
            "--dg-template",
            tpl,
            "--dg-cli",
            cli,
        ]
        if self.chkSingleFile.isChecked():
            cmd.append("--dg-single-file")
        self._save_settings()
        self._log(f"[Batch] Running (DG only): {' '.join(cmd)}")
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in proc.stdout or []:
                self._log(line.rstrip())
        except Exception as e:
            self._log(f"[Batch] Failed: {e}")

    # ---- settings persistence ----
    def _settings_path(self) -> Path:
        return BASE_DIR / "_Miscell" / "batch_last.json"

    def _collect_settings(self) -> dict:
        return {
            "overfolder": self.editOverfolder.text().strip(),
            "saxs": bool(self.chkSaxs.isChecked()),
            "rheo": bool(self.chkRheo.isChecked()),
            "skip_fresh": bool(self.chkSkipFresh.isChecked()),
            "dry_run": bool(self.chkDryRun.isChecked()),
            "log": self.comboLog.currentText(),
            "workers": int(self.spinWorkers.value()),
            "saxs_smoothing": self.saxsSmoothing.text().strip(),
            "saxs_sigma": self.saxsSigma.text().strip(),
            "saxs_theta_min": self.saxsThetaMin.text().strip(),
            "saxs_theta_max": self.saxsThetaMax.text().strip(),
            "saxs_direct": bool(self.saxsDirect.isChecked()),
            "saxs_fit": bool(self.saxsFit.isChecked()),
            "saxs_no_plots": bool(self.saxsNoPlots.isChecked()),
            "saxs_mirror": bool(self.saxsMirror.isChecked()),
            "rheo_mode": self.comboRheoMode.currentText(),
            "rheo_steady": self.rheoSteady.text().strip(),
            "rheo_no_plots": bool(self.rheoNoPlots.isChecked()),
            "dg_template": self.editTemplate.text().strip(),
            "dg_cli": self.editDgCli.text().strip(),
            "write_dg": bool(self.chkWriteDg.isChecked()),
            "dg_single_file": bool(self.chkSingleFile.isChecked()),
        }

    def _apply_settings(self, data: dict) -> None:
        try:
            self.editOverfolder.setText(data.get("overfolder", ""))
            self.chkSaxs.setChecked(bool(data.get("saxs", True)))
            self.chkRheo.setChecked(bool(data.get("rheo", True)))
            self.chkSkipFresh.setChecked(bool(data.get("skip_fresh", True)))
            self.chkDryRun.setChecked(bool(data.get("dry_run", False)))
            self.comboLog.setCurrentText(str(data.get("log", "INFO")))
            self.spinWorkers.setValue(int(data.get("workers", 8)))
            self.saxsSmoothing.setText(str(data.get("saxs_smoothing", "0.0")))
            self.saxsSigma.setText(str(data.get("saxs_sigma", "0.5")))
            self.saxsThetaMin.setText(str(data.get("saxs_theta_min", "75.0")))
            self.saxsThetaMax.setText(str(data.get("saxs_theta_max", "110.0")))
            self.saxsDirect.setChecked(bool(data.get("saxs_direct", True)))
            self.saxsFit.setChecked(bool(data.get("saxs_fit", False)))
            self.saxsNoPlots.setChecked(bool(data.get("saxs_no_plots", False)))
            self.saxsMirror.setChecked(bool(data.get("saxs_mirror", False)))
            mode = str(data.get("rheo_mode", "triggered"))
            idx = self.comboRheoMode.findText(mode)
            if idx >= 0:
                self.comboRheoMode.setCurrentIndex(idx)
            self.rheoSteady.setText(str(data.get("rheo_steady", "10.0")))
            self.rheoNoPlots.setChecked(bool(data.get("rheo_no_plots", False)))
            self.editTemplate.setText(str(data.get("dg_template", "")))
            self.editDgCli.setText(str(data.get("dg_cli", "/Applications/DataGraph.app/Contents/Library/dgraph")))
            self.chkWriteDg.setChecked(bool(data.get("write_dg", False)))
            self.chkSingleFile.setChecked(bool(data.get("dg_single_file", False)))
        except Exception:
            pass

    def _load_settings(self) -> None:
        p = self._settings_path()
        if not p.exists():
            return
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                self._apply_settings(data)
        except Exception:
            pass

    def _save_settings(self) -> None:
        try:
            p = self._settings_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(self._collect_settings(), indent=2), encoding="utf-8")
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        if hasattr(self.main, "logger"):
            self.main.logger.info(msg)
