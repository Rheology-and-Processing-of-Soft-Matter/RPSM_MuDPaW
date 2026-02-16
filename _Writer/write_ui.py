from __future__ import annotations

import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List

from _Core.paths import get_processed_root

from _Writer.dg_runner import run_datagraph_write

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QToolButton,
    QVBoxLayout,
    QWidget,
    QComboBox,
    QGridLayout,
    QCheckBox,
    QPlainTextEdit,
    QSizePolicy,
)

BASE_DIR = Path(__file__).resolve().parent.parent


class WriteUI:
    def __init__(self, main_window):
        self.main = main_window
        self.writeGridCells: Dict[tuple[int, int], QComboBox] = {}
        self._write_outputs_by_modality: Dict[str, List[dict]] = {}
        self._write_refreshed_once = False
        self._write_processed_root: Path | None = None
        self.datagraph_path = ""
        self.current_template_path = ""
        self.custom_files: List[str] = []
        self._write_columns: List[str] = []

    # ---- UI builders ----
    def build_write_canvas(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.header = QWidget(page)
        self.header.setObjectName("writeHeaderBar")
        header_layout = QHBoxLayout(self.header)
        header_layout.setContentsMargins(8, 6, 8, 6)
        header_layout.setSpacing(8)

        self.lblWriteTitle = QLabel("Write to Template", self.header)
        self.lblWriteTitle.setObjectName("lblWriteTitle")
        header_layout.addWidget(self.lblWriteTitle)
        header_layout.addStretch(1)

        self.lblWriteProject = QLabel("—", self.header)
        self.lblWriteProject.setObjectName("lblWriteProject")
        header_layout.addWidget(self.lblWriteProject)

        self.btnWriteRefresh = QToolButton(self.header)
        self.btnWriteRefresh.setObjectName("btnWriteRefresh")
        self.btnWriteRefresh.setToolTip("Refresh")
        header_layout.addWidget(self.btnWriteRefresh)

        # Show by default so refresh is visible once a project is selected.
        self.header.setVisible(True)
        layout.addWidget(self.header)

        self.writeGridScroll = QScrollArea(page)
        self.writeGridScroll.setObjectName("writeGridScroll")
        self.writeGridScroll.setWidgetResizable(True)
        self.writeGridContainer = QWidget(self.writeGridScroll)
        self.writeGridContainer.setObjectName("writeGridContainer")
        self.writeGridLayout = QGridLayout(self.writeGridContainer)
        self.writeGridLayout.setContentsMargins(8, 6, 8, 6)
        self.writeGridLayout.setHorizontalSpacing(8)
        self.writeGridLayout.setVerticalSpacing(6)
        self.writeGridLayout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.writeGridContainer.setSizePolicy(
            self.writeGridContainer.sizePolicy().horizontalPolicy(),
            self.writeGridContainer.sizePolicy().verticalPolicy(),
        )
        self.writeGridScroll.setWidget(self.writeGridContainer)
        layout.addWidget(self.writeGridScroll)

        status = QWidget(page)
        status.setObjectName("writeTableStatusBar")
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(8, 4, 8, 4)
        status_layout.setSpacing(8)
        self.lblWriteStatus = QLabel("0 rows | 0 selections", status)
        self.lblWriteStatus.setObjectName("lblWriteStatus")
        status_layout.addWidget(self.lblWriteStatus)
        status_layout.addStretch(1)
        self.lblWriteTemplateStatus = QLabel("Template: —", status)
        self.lblWriteTemplateStatus.setObjectName("lblWriteTemplateStatus")
        status_layout.addWidget(self.lblWriteTemplateStatus)
        layout.addWidget(status)

        self._build_write_grid()

    def build_write_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(8)

        grp_template = QGroupBox("Template", page)
        grp_template.setObjectName("grpTemplate")
        tmpl_layout = QVBoxLayout(grp_template)
        tmpl_layout.setSpacing(8)

        row1 = QHBoxLayout()
        self.editTemplatePath = QLineEdit(grp_template)
        self.editTemplatePath.setObjectName("editTemplatePath")
        self.editTemplatePath.setReadOnly(True)
        row1.addWidget(self.editTemplatePath, 1)
        self.btnBrowseTemplate = QPushButton("Load", grp_template)
        self.btnBrowseTemplate.setObjectName("btnBrowseTemplate")
        row1.addWidget(self.btnBrowseTemplate)
        tmpl_layout.addLayout(row1)

        row2 = QHBoxLayout()
        self.btnValidateTemplate = QPushButton("Validate", grp_template)
        self.btnValidateTemplate.setObjectName("btnValidateTemplate")
        row2.addWidget(self.btnValidateTemplate)
        self.btnReloadTemplate = QPushButton("Reload", grp_template)
        self.btnReloadTemplate.setObjectName("btnReloadTemplate")
        row2.addWidget(self.btnReloadTemplate)
        row2.addStretch(1)
        tmpl_layout.addLayout(row2)

        row2b = QHBoxLayout()
        self.btnRefreshData = QPushButton("Refresh data", grp_template)
        self.btnRefreshData.setObjectName("btnRefreshData")
        row2b.addWidget(self.btnRefreshData)
        row2b.addStretch(1)
        tmpl_layout.addLayout(row2b)

        self.comboRecentTemplates = QComboBox(grp_template)
        self.comboRecentTemplates.setObjectName("comboRecentTemplates")
        tmpl_layout.addWidget(QLabel("Recent templates", grp_template))
        tmpl_layout.addWidget(self.comboRecentTemplates)

        row3 = QHBoxLayout()
        self.btnLoadCustomFiles = QPushButton("Load external files", grp_template)
        self.btnLoadCustomFiles.setObjectName("btnLoadCustomFiles")
        row3.addWidget(self.btnLoadCustomFiles)
        row3.addStretch(1)
        tmpl_layout.addLayout(row3)

        layout.addWidget(grp_template)

        grp_bulk = QGroupBox("Bulk operations", page)
        grp_bulk.setObjectName("grpBulk")
        bulk_layout = QGridLayout(grp_bulk)
        bulk_layout.setHorizontalSpacing(8)
        bulk_layout.setVerticalSpacing(8)
        self.btnEnableAll = QPushButton("Clear all", grp_bulk)
        self.btnEnableAll.setObjectName("btnEnableAll")
        self.btnDisableAll = QPushButton("Auto-fill by name", grp_bulk)
        self.btnDisableAll.setObjectName("btnDisableAll")
        self.btnEnableSelected = QPushButton("Enable selected", grp_bulk)
        self.btnEnableSelected.setObjectName("btnEnableSelected")
        self.btnDisableSelected = QPushButton("Disable selected", grp_bulk)
        self.btnDisableSelected.setObjectName("btnDisableSelected")
        self.btnClearSelection = QPushButton("Reset radial", grp_bulk)
        self.btnClearSelection.setObjectName("btnClearSelection")
        self.btnAutoFill = QPushButton("Auto-fill", grp_bulk)
        self.btnAutoFill.setObjectName("btnAutoFill")
        self.btnResetMappings = QPushButton("Reset mappings", grp_bulk)
        self.btnResetMappings.setObjectName("btnResetMappings")
        bulk_layout.addWidget(self.btnEnableAll, 0, 0)
        bulk_layout.addWidget(self.btnDisableAll, 0, 1)
        bulk_layout.addWidget(self.btnClearSelection, 1, 0, 1, 2)
        bulk_layout.addWidget(self.btnEnableSelected, 2, 0)
        bulk_layout.addWidget(self.btnDisableSelected, 2, 1)
        bulk_layout.addWidget(self.btnAutoFill, 3, 0)
        bulk_layout.addWidget(self.btnResetMappings, 3, 1)
        layout.addWidget(grp_bulk)
        self.btnEnableSelected.hide()
        self.btnDisableSelected.hide()
        self.btnAutoFill.hide()
        self.btnResetMappings.hide()

        grp_write = QGroupBox("Write", page)
        grp_write.setObjectName("grpWrite")
        write_layout = QVBoxLayout(grp_write)
        write_layout.setSpacing(8)
        self.btnWriteSelected = QPushButton("Write selected", grp_write)
        self.btnWriteSelected.setObjectName("btnWriteSelected")
        write_layout.addWidget(self.btnWriteSelected)
        concat_row1 = QHBoxLayout()
        self.btnWriteSaxsConcat = QPushButton("Write SAXS (concat)", grp_write)
        self.btnWriteSaxsConcat.setObjectName("btnWriteSaxsConcat")
        concat_row1.addWidget(self.btnWriteSaxsConcat)
        self.btnWritePiConcat = QPushButton("Write PI (concat)", grp_write)
        self.btnWritePiConcat.setObjectName("btnWritePiConcat")
        concat_row1.addWidget(self.btnWritePiConcat)
        write_layout.addLayout(concat_row1)

        concat_row2 = QHBoxLayout()
        self.btnWritePliConcat = QPushButton("Write PLI (concat)", grp_write)
        self.btnWritePliConcat.setObjectName("btnWritePliConcat")
        concat_row2.addWidget(self.btnWritePliConcat)
        self.btnWriteRheoConcat = QPushButton("Write Rheology (concat)", grp_write)
        self.btnWriteRheoConcat.setObjectName("btnWriteRheoConcat")
        concat_row2.addWidget(self.btnWriteRheoConcat)
        write_layout.addLayout(concat_row2)
        self.chkDryRun = QCheckBox("Dry-run (no files)", grp_write)
        self.chkDryRun.setObjectName("chkDryRun")
        write_layout.addWidget(self.chkDryRun)
        self.chkOpenAfterWrite = QCheckBox("Open output folder", grp_write)
        self.chkOpenAfterWrite.setObjectName("chkOpenAfterWrite")
        write_layout.addWidget(self.chkOpenAfterWrite)
        row_open = QHBoxLayout()
        self.btnOpenOutputFolder = QToolButton(grp_write)
        self.btnOpenOutputFolder.setObjectName("btnOpenOutputFolder")
        row_open.addWidget(self.btnOpenOutputFolder)
        row_open.addStretch(1)
        write_layout.addLayout(row_open)
        layout.addWidget(grp_write)

        layout.addStretch(1)
        self._wire_buttons()
        self._load_last_template()
        self._load_recent_templates()
        self._load_last_datagraph_path()

    # ---- Wiring ----
    def _wire_buttons(self) -> None:
        self.btnEnableAll.clicked.connect(self._clear_all_selections)
        self.btnDisableAll.clicked.connect(self._auto_fill_by_name)
        self.btnEnableSelected.clicked.connect(lambda: self.main._log_not_implemented("btnEnableSelected"))
        self.btnDisableSelected.clicked.connect(lambda: self.main._log_not_implemented("btnDisableSelected"))
        self.btnResetMappings.clicked.connect(lambda: self.main._log_not_implemented("btnResetMappings"))
        self.btnWriteSelected.clicked.connect(self._write_selected)
        self.btnWriteRheoConcat.clicked.connect(self._write_rheo_concat_only)
        self.btnOpenOutputFolder.clicked.connect(lambda: self.main._log_not_implemented("btnOpenOutputFolder"))
        self.btnWriteRefresh.clicked.connect(lambda _=False: self.refresh_write_data())
        self.btnClearSelection.clicked.connect(self._reset_radial_selections)
        self.btnAutoFill.clicked.connect(lambda: self.main._log_not_implemented("btnAutoFill"))
        self.btnReloadTemplate.clicked.connect(self._reload_template)
        self.btnRefreshData.clicked.connect(lambda _=False: self.refresh_write_data(initial=False))

        self.btnBrowseTemplate.clicked.connect(self._browse_template_path)
        self.btnValidateTemplate.clicked.connect(self._validate_datagraph_path)
        self.comboRecentTemplates.currentIndexChanged.connect(self._on_recent_template_selected)
        self.btnLoadCustomFiles.clicked.connect(self._load_custom_files)

    def _on_recent_template_selected(self, _idx: int) -> None:
        try:
            text = self.comboRecentTemplates.currentText().strip()
            if not text or text == "Select recent...":
                return
            if os.path.isfile(text):
                self.current_template_path = text
                self.editTemplatePath.setText(Path(text).name)
                self.editTemplatePath.setToolTip(text)
                self._save_last_template(text)
                self.lblWriteTemplateStatus.setText(f"Template: {Path(text).name}")
                self.main.logger.info(f"Template selected: {text}")
        except Exception as e:
            self.main._log_exception("Recent template select failed", e)

    def _load_custom_files(self) -> None:
        try:
            start_dir = self.main.project_path or str(BASE_DIR)
            paths, _ = QFileDialog.getOpenFileNames(
                self.main,
                "Select External CSV Files",
                start_dir,
                "CSV Files (*.csv);;All Files (*)",
            )
            if not paths:
                return
            # de-dup while preserving order
            existing = set(self.custom_files)
            for p in paths:
                if p not in existing:
                    self.custom_files.append(p)
                    existing.add(p)
            self.main.logger.info(f"[Write] Loaded {len(paths)} custom file(s).")
            self._refresh_write_grid()
        except Exception as e:
            self.main._log_exception("Load custom files failed", e)

    # ---- Header ----
    def update_write_header(self, project_path: str | None) -> None:
        if not project_path:
            self.lblWriteProject.setText("—")
            if hasattr(self, "header"):
                self.header.setVisible(False)
            return
        self.lblWriteProject.setText(Path(project_path).name)
        if hasattr(self, "header"):
            self.header.setVisible(True)

    # ---- Template ----
    def _browse_template_path(self) -> None:
        try:
            start_dir = None
            last_path = self.current_template_path or ""
            if last_path:
                start_dir = str(Path(last_path).expanduser().parent)
            if not start_dir:
                if self.main.project_path:
                    start_dir = str(Path(self.main.project_path))
                else:
                    start_dir = str(BASE_DIR)
            path, _ = QFileDialog.getOpenFileName(
                self.main,
                "Select Template File",
                start_dir,
                "DataGraph Template (*.dgraph *.dgtemplate *.dgl);;All Files (*)",
            )
            if not path:
                return
            self.current_template_path = path
            self.editTemplatePath.setText(Path(path).name)
            self.editTemplatePath.setToolTip(path)
            self._save_last_template(path)
            self._add_recent_template(path)
            self.lblWriteTemplateStatus.setText(f"Template: {Path(path).name}")
            self.main.logger.info(f"Template selected: {path}")
        except Exception as e:
            self.main._log_exception("Template selection failed", e)

    # ---- DataGraph validation ----
    def _validate_datagraph_path(self) -> None:
        path = (self.datagraph_path or "").strip()
        if path and self._is_datagraph_path_valid(path):
            self.main.logger.info(f"DataGraph OK: {path}")
            QMessageBox.information(self.main, "DataGraph", f"DataGraph OK:\n{path}")
            return

        candidates = [
            "/Applications/DataGraph.app/Contents/MacOS",
            "/Applications/DataGraph.app/Contents/Library",
        ]
        for c in candidates:
            if self._is_datagraph_path_valid(c):
                self.datagraph_path = c
                self._save_last_datagraph_path(c)
                self.main.logger.info(f"DataGraph OK: {c}")
                QMessageBox.information(self.main, "DataGraph", f"DataGraph OK:\n{c}")
                return

        folder = QFileDialog.getExistingDirectory(self.main, "Select DataGraph Folder")
        if folder and self._is_datagraph_path_valid(folder):
            self.datagraph_path = folder
            self._save_last_datagraph_path(folder)
            self.main.logger.info(f"DataGraph OK: {folder}")
            QMessageBox.information(self.main, "DataGraph", f"DataGraph OK:\n{folder}")
            return

        QMessageBox.warning(self.main, "DataGraph", "DataGraph not found. Please select the DataGraph folder.")

    def _is_datagraph_path_valid(self, path: str) -> bool:
        if os.path.isfile(path) and os.access(path, os.X_OK):
            return True
        if os.path.isdir(path):
            for name in ("dgraph", "DataGraph", "datagraph"):
                p = os.path.join(path, name)
                if os.path.isfile(p) and os.access(p, os.X_OK):
                    return True
        return False

    def _last_datagraph_path(self) -> Path:
        return BASE_DIR / "_Miscell" / "last_datagraph_path.txt"

    def _save_last_datagraph_path(self, path: str) -> None:
        try:
            (BASE_DIR / "_Miscell").mkdir(parents=True, exist_ok=True)
            self._last_datagraph_path().write_text(path, encoding="utf-8")
        except Exception:
            pass

    def _load_last_datagraph_path(self) -> None:
        try:
            p = self._last_datagraph_path()
            if not p.exists():
                return
            path = p.read_text(encoding="utf-8").strip()
            if path:
                self.datagraph_path = path
        except Exception:
            pass

    def _last_template_path(self) -> Path:
        return BASE_DIR / "_Miscell" / "last_template.txt"

    def _save_last_template(self, path: str) -> None:
        try:
            (BASE_DIR / "_Miscell").mkdir(parents=True, exist_ok=True)
            self._last_template_path().write_text(path, encoding="utf-8")
        except Exception:
            pass

    def _load_last_template(self) -> None:
        try:
            p = self._last_template_path()
            if not p.exists():
                return
            path = p.read_text(encoding="utf-8").strip()
            if not path:
                return
            self.current_template_path = path
            self.editTemplatePath.setText(Path(path).name)
            self.editTemplatePath.setToolTip(path)
            self.lblWriteTemplateStatus.setText(f"Template: {Path(path).name}")
            self._add_recent_template(path)
        except Exception:
            pass

    def _recent_templates_path(self) -> Path:
        return BASE_DIR / "_Miscell" / "recent_templates.json"

    def _load_recent_templates(self) -> None:
        try:
            p = self._recent_templates_path()
            if not p.exists():
                return
            data = json.loads(p.read_text(encoding="utf-8"))
            if not isinstance(data, list):
                return
            self.comboRecentTemplates.clear()
            self.comboRecentTemplates.addItem("Select recent...")
            for item in data:
                if isinstance(item, str) and item:
                    self.comboRecentTemplates.addItem(item)
        except Exception:
            pass

    def _save_recent_templates(self, items: List[str]) -> None:
        try:
            (BASE_DIR / "_Miscell").mkdir(parents=True, exist_ok=True)
            self._recent_templates_path().write_text(json.dumps(items, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _add_recent_template(self, path: str) -> None:
        if not path:
            return
        current = []
        try:
            p = self._recent_templates_path()
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    current = [x for x in data if isinstance(x, str)]
        except Exception:
            current = []
        # de-dup + keep most recent first
        current = [p for p in current if p != path]
        current.insert(0, path)
        current = current[:10]
        self._save_recent_templates(current)
        self._load_recent_templates()

    def _reload_template(self) -> None:
        """Re-apply the current template file (useful after editing it on disk)."""
        path = (self.current_template_path or "").strip()
        if not path:
            QMessageBox.information(self.main, "Reload", "No template selected.")
            return
        if not os.path.isfile(path):
            QMessageBox.warning(self.main, "Reload", f"Template not found:\n{path}")
            return
        try:
            self.editTemplatePath.setText(Path(path).name)
            self.editTemplatePath.setToolTip(path)
            self.lblWriteTemplateStatus.setText(f"Template: {Path(path).name}")
            self._save_last_template(path)
            self.main.logger.info(f"Template reloaded: {path}")
        except Exception as e:
            self.main._log_exception("Template reload failed", e)


    # ---- Grid data ----
    def refresh_write_data(self, initial: bool = False) -> None:
        if initial and self._write_refreshed_once:
            return
        self._write_refreshed_once = True

        self._collect_write_outputs()
        self._refresh_write_grid()
        self._update_write_status()

    def _collect_write_outputs(self) -> None:
        self._write_outputs_by_modality = {"SAXS": [], "PI": [], "PLI": [], "Rheology": []}
        project_path = self.main.project_path
        if not project_path:
            return
        processed = get_processed_root(project_path)
        self._write_processed_root = processed
        if not processed.exists():
            return
        # Pure filename-based discovery (no JSON dependency)
        print(f"[WRITER] processed_root={processed} -- JSON disabled; using filename scan only")
        for modality in self._write_outputs_by_modality.keys():
            self._write_outputs_by_modality[modality] = self._fallback_scan_processed(modality)

    def _fallback_scan_processed(self, modality: str) -> List[dict]:
        if not self._write_processed_root:
            return []
        prefixes = {
            # Order matters for collision avoidance: PLI before PI, explicit dg before base.
            "PLI": ("_dg_pli_", "_pli_"),
            "PI": ("_dg_pi_", "_pi_"),
            "SAXS": ("saxs_1_", "saxs_2_", "saxs_radial_"),
            "Rheology": ("_rheo_", "_visco_"),
        }
        allowed_exts = {".csv", ".dat", ".txt"}
        names_to_files: Dict[str, List[str]] = {}
        for folder in [self._write_processed_root] + [p for p in self._write_processed_root.iterdir() if p.is_dir() and p.name != "_Other"]:
            for path in folder.iterdir():
                if not path.is_file():
                    continue
                if path.suffix.lower() not in allowed_exts:
                    continue
                low = path.name.lower()
                if not any(low.startswith(pref) for pref in prefixes.get(modality, ())):
                    continue
                name = path.stem
                for pref in prefixes.get(modality, ()):
                    if name.lower().startswith(pref):
                        name = name[len(pref):]
                        break
                names_to_files.setdefault(name, []).append(str(path))
        return [{"name": name, "entry": {"files": files}} for name, files in sorted(names_to_files.items())]

    def _modality_from_output_name(self, fname: str) -> str:
        low = fname.lower()
        if "saxs" in low:
            return "SAXS"
        if "pli" in low:
            return "PLI"
        if "pi" in low and "pli" not in low:
            return "PI"
        if "rheo" in low or "viscos" in low:
            return "Rheology"
        return "Auto"

    def _clean_output_basename(self, fname: str, modality: str) -> str:
        base = os.path.splitext(fname)[0]
        low = base.lower()
        if low.startswith("_output_"):
            base = base[len("_output_"):]
        if low.startswith("output_"):
            base = base[len("output_"):]
        mod_low = modality.lower()
        if base.lower().startswith(mod_low + "_"):
            base = base[len(mod_low) + 1 :]
        return base

    def _build_write_grid(self) -> None:
        if not hasattr(self, "writeGridLayout"):
            return
        self._refresh_write_grid()

    def _write_grid_max_rows(self) -> int:
        counts = [
            len(self._write_outputs_by_modality.get("SAXS", [])),
            len(self._write_outputs_by_modality.get("PI", [])),
            len(self._write_outputs_by_modality.get("PLI", [])),
            len(self._write_outputs_by_modality.get("Rheology", [])),
            len(self.custom_files),
        ]
        return max(counts) if counts else 0

    def _refresh_write_grid(self) -> None:
        if not hasattr(self, "writeGridLayout"):
            return
        while self.writeGridLayout.count():
            item = self.writeGridLayout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)

        columns = ["SAXS", "PI", "PLI", "Rheology", "Radial Integration"]
        if self.custom_files:
            columns.append("Custom")
        self._write_columns = columns[:]
        for j, col in enumerate(columns):
            lbl = QLabel(col, self.writeGridContainer)
            lbl.setStyleSheet("font-weight: 600;")
            self.writeGridLayout.addWidget(lbl, 0, j)
            self.writeGridLayout.setColumnStretch(j, 1)

        def _names_for(modality: str) -> List[str]:
            names = []
            for e in self._write_outputs_by_modality.get(modality, []):
                name = e.get("name")
                if not isinstance(name, str):
                    continue
                names.append(name)
            return sorted(set(names))

        options = {
            "SAXS": ["None"] + _names_for("SAXS"),
            "PI": ["None"] + _names_for("PI"),
            "PLI": ["None"] + _names_for("PLI"),
            "Rheology": ["None"] + _names_for("Rheology"),
            "Radial Integration": ["None"],
        }
        if self.custom_files:
            options["Custom"] = ["None"] + [os.path.basename(p) for p in self.custom_files]

        row_count = self._write_grid_max_rows()
        self.writeGridCells = {}
        metrics = self.writeGridContainer.fontMetrics()
        custom_map = {os.path.basename(p): p for p in self.custom_files}
        for i in range(row_count):
            for j, col in enumerate(columns):
                combo = QComboBox(self.writeGridContainer)
                combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
                for item in options[col]:
                    elided = metrics.elidedText(item, Qt.TextElideMode.ElideRight, 260)
                    combo.addItem(elided)
                    if col == "Custom" and item != "None":
                        full = custom_map.get(item, item)
                        combo.setItemData(combo.count() - 1, full, Qt.ItemDataRole.ToolTipRole)
                    else:
                        combo.setItemData(combo.count() - 1, item, Qt.ItemDataRole.ToolTipRole)
                    combo.setItemData(combo.count() - 1, item, Qt.ItemDataRole.UserRole)
                combo.setCurrentText("None")
                combo.currentTextChanged.connect(lambda _val, r=i, c=col: self._on_write_cell_changed(r, c))
                self.writeGridLayout.addWidget(combo, i + 1, j)
                self.writeGridCells[(i, j)] = combo
        self.writeGridLayout.setRowStretch(row_count + 1, 1)

    def _on_write_cell_changed(self, row: int, col: str) -> None:
        col_index = 0
        if self._write_columns and col in self._write_columns:
            col_index = self._write_columns.index(col)
        combo = self.writeGridCells.get((row, col_index))
        if combo is not None:
            full = combo.currentData(Qt.ItemDataRole.UserRole)
            if isinstance(full, str):
                combo.setToolTip(full)
        if col == "SAXS":
            name = self._write_grid_value(row, 0)
            self._auto_fill_radial(row, name)
        self._update_write_status()

    def _write_grid_value(self, row: int, col_index: int) -> str:
        combo = self.writeGridCells.get((row, col_index))
        if combo is None:
            return "None"
        data = combo.currentData(Qt.ItemDataRole.UserRole)
        return data if isinstance(data, str) else combo.currentText()

    def _write_grid_selection_count(self) -> int:
        count = 0
        row_count = self._write_grid_max_rows()
        for i in range(row_count):
            for j in range(len(self._write_columns or [])):
                if self._write_grid_value(i, j) != "None":
                    count += 1
        return count

    def _update_write_status(self) -> None:
        self.lblWriteStatus.setText(f"{self._write_grid_max_rows()} rows | {self._write_grid_selection_count()} selections")

    def _write_selected(self) -> None:
        project_path = self.main.project_path
        if not project_path:
            QMessageBox.warning(self.main, "Write", "Select a reference folder first.")
            return
        template_path = self._resolve_template_path()
        if not template_path:
            msg = f"Select a valid DataGraph template first.\nCurrent: {self.current_template_path or self.editTemplatePath.text().strip()}"
            QMessageBox.warning(self.main, "Write", msg)
            return
        exec_path = self.datagraph_path or ""
        if exec_path is None:
            QMessageBox.warning(self.main, "Write", "DataGraph not found. Click Validate to locate it.")
            return

        row_count = self._write_grid_max_rows()
        if row_count == 0:
            QMessageBox.information(self.main, "Write", "No rows available to write.")
            return

        # Save all DataGraph outputs inside the current reference folder's unified
        # `_Processed` tree (rather than the template location or an old/fixed path).
        processed_root = self._write_processed_root or get_processed_root(project_path)
        self._write_processed_root = processed_root
        ref_root = Path(project_path).resolve()
        output_dir = ref_root  # write .dgraph next to the reference folder, not inside _Processed
        dry_run = self.chkDryRun.isChecked()
        self.main.logger.info(f"[Write] Using template: {template_path}")
        self.main.logger.info(f"[Write] DataGraph path: {exec_path}")
        self.main.logger.info(f"[Write] Dry-run: {dry_run}")
        if not dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)

        wrote_any = False
        base_name = Path(project_path).name
        for i in range(row_count):
            row_selections = {
                "SAXS": self._write_grid_value(i, self._write_columns.index("SAXS")),
                "PI": self._write_grid_value(i, self._write_columns.index("PI")),
                "PLI": self._write_grid_value(i, self._write_columns.index("PLI")),
                "Rheology": self._write_grid_value(i, self._write_columns.index("Rheology")),
            }
            if all(v == "None" for v in row_selections.values()):
                # allow custom-only rows
                pass

            files = []
            name_token = None
            for modality, name in row_selections.items():
                if name == "None":
                    continue
                if name_token is None:
                    name_token = name
                files.extend(self._resolve_files_for(modality, name))

            radial = self._write_grid_value(i, self._write_columns.index("Radial Integration"))
            if radial and radial != "None":
                files.append(radial)
                if name_token is None:
                    name_token = Path(radial).stem

            if "Custom" in self._write_columns:
                custom_val = self._write_grid_value(i, self._write_columns.index("Custom"))
                if custom_val and custom_val != "None":
                    match = next((p for p in self.custom_files if os.path.basename(p) == custom_val), None)
                    if match:
                        files.append(match)
                        if name_token is None:
                            name_token = Path(match).stem

            files = [f for f in files if f and os.path.isfile(f)]
            if not files:
                self.main.logger.warning(f"[Write] Row {i+1}: no files resolved.")
                continue

            safe_token = self._safe_name(name_token or f"row_{i+1}")
            output_path = output_dir / f"{base_name}_{safe_token}.dgraph"
            if dry_run:
                self.main.logger.info(f"[Write] Dry-run: {output_path}")
            else:
                try:
                    self.main.logger.info(f"[Write] Row {i+1}: {len(files)} file(s) -> {output_path}")
                    result = run_datagraph_write(exec_path, files, template_path, str(output_path), sample_name=base_name, quit_after=(i == row_count - 1))
                    if result and result.get("returncode", 0) != 0:
                        self.main.logger.error(f"[Write] DataGraph failed (code {result.get('returncode')}): {result.get('stderr')}")
                    if not output_path.exists():
                        self.main.logger.warning(f"[Write] Output not found: {output_path}")
                except Exception as e:
                    self.main._log_exception("DataGraph write failed", e)
            wrote_any = True

        if not wrote_any:
            self.main.logger.info("[Write] Nothing written.")
        else:
            self.main.logger.info(f"[Write] Output folder: {output_dir}")

    def _write_rheo_concat_only(self) -> None:
        project_path = self.main.project_path
        if not project_path:
            QMessageBox.warning(self.main, "Write", "Select a reference folder first.")
            return
        template_path = self._resolve_template_path()
        if not template_path:
            msg = f"Select a valid DataGraph template first.\nCurrent: {self.current_template_path or self.editTemplatePath.text().strip()}"
            QMessageBox.warning(self.main, "Write", msg)
            return
        exec_path = self.datagraph_path or ""
        if exec_path is None:
            QMessageBox.warning(self.main, "Write", "DataGraph not found. Click Validate to locate it.")
            return

        row_count = self._write_grid_max_rows()
        if row_count == 0:
            QMessageBox.information(self.main, "Write", "No rows available to write.")
            return

        # Concat temp stays in _Processed, but final .dgraph goes to reference root.
        processed_root = self._write_processed_root or get_processed_root(project_path)
        self._write_processed_root = processed_root
        ref_root = Path(project_path).resolve()
        output_dir = ref_root
        temp_dir = processed_root / "Write"
        dry_run = self.chkDryRun.isChecked()
        self.main.logger.info(f"[Write] Using template: {template_path}")
        self.main.logger.info(f"[Write] DataGraph path: {exec_path}")
        self.main.logger.info(f"[Write] Dry-run: {dry_run}")
        if not dry_run:
            temp_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)

        wrote_any = False
        base_name = Path(project_path).name
        for i in range(row_count):
            name = self._write_grid_value(i, 3)
            if name == "None":
                continue
            files = self._resolve_files_for("Rheology", name)
            files = [f for f in files if f and os.path.isfile(f)]
            if not files:
                self.main.logger.warning(f"[Write] Row {i+1}: no Rheology files resolved.")
                continue
            safe_token = self._safe_name(name)
            concat_path = temp_dir / f"Rheo_concat_{safe_token}.csv"
            if not dry_run:
                self._concat_csv_with_filename(files, concat_path)
            output_path = output_dir / f"{base_name}_{safe_token}.dgraph"
            if dry_run:
                self.main.logger.info(f"[Write] Dry-run: {output_path}")
            else:
                try:
                    self.main.logger.info(f"[Write] Row {i+1}: Rheology concat -> {output_path}")
                    result = run_datagraph_write(exec_path, [str(concat_path)], template_path, str(output_path), sample_name=base_name, quit_after=(i == row_count - 1))
                    if result and result.get("returncode", 0) != 0:
                        self.main.logger.error(f"[Write] DataGraph failed (code {result.get('returncode')}): {result.get('stderr')}")
                    if not output_path.exists():
                        self.main.logger.warning(f"[Write] Output not found: {output_path}")
                except Exception as e:
                    self.main._log_exception("DataGraph write failed", e)
            wrote_any = True

        if not wrote_any:
            self.main.logger.info("[Write] Nothing written.")
        else:
            self.main.logger.info(f"[Write] Output folder: {output_dir}")

    def _resolve_files_for(self, modality: str, name: str) -> List[str]:
        entries = self._write_outputs_by_modality.get(modality, [])
        for entry in entries:
            if entry.get("name") == name:
                data = entry.get("entry", entry)
                files = self._extract_files_from_entry(data)
                if modality == "SAXS":
                    files = self._ensure_saxs_export(name, files)
                return files
        if modality == "SAXS":
            # Even if no entry matched, try to surface export files for the name.
            return self._ensure_saxs_export(name, [])
        return []

    def _extract_files_from_entry(self, entry) -> List[str]:
        if entry is None:
            return []
        if isinstance(entry, list):
            return [self._normalize_path(str(x)) for x in entry if x]
        if not isinstance(entry, dict):
            return []
        keys = (
            "csv_outputs",
            "files",
            "outputs",
            "paths",
            "csv_output",
            "summary_csv",
            "datagraph_csv",
        )
        for key in keys:
            val = entry.get(key)
            if val is None:
                continue
            if isinstance(val, list):
                return [self._normalize_path(str(x)) for x in val if x]
            if isinstance(val, dict):
                return [self._normalize_path(str(x)) for x in val.values() if x]
            return [self._normalize_path(str(val))]
        return []

    def _ensure_saxs_export(self, name: str, files: List[str]) -> List[str]:
        """
        Ensure key SAXS CSVs for this dataset are included, without pulling in
        other q-windows of the same sample:
          - SAXS_1_*_export.csv (P2/P4/P6/OAS)
          - SAXS_2_* (azimuthal flattened)
          - SAXS_radial_flat_*.csv
          - SAXS_radial_matrix_*.csv
        Only files whose stem contains the dataset tag (derived from existing files
        when available, otherwise the sample name) are considered, and at most one
        match per pattern is added.
        """
        if not self._write_processed_root or not name:
            return files

        # Derive a dataset tag from existing files (preferred) or from the sample name.
        tag = None
        for f in files:
            stem = Path(f).stem.lower()
            if "saxs_" in stem:
                tag = stem.split("saxs_", 1)[-1]
                break
        if tag is None:
            tag = name.lower().replace(" ", "_")

        patterns = [
            ("export", "SAXS_1_*_export.csv"),
            ("raw_flat", "SAXS_2_*"),
            ("rad_flat", "SAXS_radial_flat_*.csv"),
            ("rad_matrix", "SAXS_radial_matrix_*.csv"),
        ]
        extras: List[str] = []
        search_dirs = [self._write_processed_root, self._write_processed_root / "SAXS"]
        seen_keys = set()

        def matches_tag(path: Path) -> bool:
            stem = path.stem.lower()
            return tag in stem

        for folder in search_dirs:
            try:
                for key, pat in patterns:
                    if key in seen_keys:
                        continue
                    for path in folder.glob(pat):
                        if path.is_file() and matches_tag(path):
                            p_resolved = path.resolve()
                            # Ensure legacy exports have P246OAS column for DataGraph templates.
                            if key == "export":
                                try:
                                    df = pd.read_csv(p_resolved)
                                    if "OAS" in df.columns and "P246OAS" not in df.columns:
                                        df["P246OAS"] = df["OAS"]
                                        df.to_csv(p_resolved, index=False)
                                        self.main.logger.info(f"[Write] Added P246OAS to {p_resolved.name}")
                                except Exception:
                                    pass
                            extras.append(str(p_resolved))
                            seen_keys.add(key)
                            break  # only first match per pattern
            except Exception:
                continue
        if not extras:
            return files
        existing = set(os.path.abspath(f) for f in files)
        for p in extras:
            if os.path.abspath(p) not in existing:
                files.append(p)
        return files

    def _normalize_path(self, path: str) -> str:
        if not path:
            return path
        p = Path(path)
        if p.is_absolute():
            return str(p)
        if self._write_processed_root:
            cand = self._write_processed_root / path
            if cand.exists():
                return str(cand)
        project_path = self.main.project_path
        if project_path:
            cand = Path(project_path) / path
            if cand.exists():
                return str(cand)
        return str(p)

    def _concat_csv_with_filename(self, files: List[str], out_path: Path) -> None:
        header = None
        try:
            with out_path.open("w", newline="", encoding="utf-8") as out_f:
                writer = None
                for fpath in files:
                    with open(fpath, "r", newline="", encoding="utf-8") as in_f:
                        reader = csv.DictReader(in_f)
                        if not reader.fieldnames:
                            continue
                        if header is None:
                            header = ["File_name"] + list(reader.fieldnames)
                            writer = csv.DictWriter(out_f, fieldnames=header)
                            writer.writeheader()
                        if reader.fieldnames != header[1:]:
                            self.main.logger.warning(f"[Write] Column mismatch, skipping: {fpath}")
                            continue
                        file_name = Path(fpath).stem
                        for row in reader:
                            row_out = {"File_name": file_name}
                            row_out.update(row)
                            writer.writerow(row_out)
        except Exception as e:
            self.main._log_exception("Rheology concatenation failed", e)

    def _safe_name(self, name: str) -> str:
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_")

    def _resolve_template_path(self) -> str | None:
        cand = (self.current_template_path or "").strip()
        if cand:
            self.main.logger.info(f"[Write] Resolve template: current={cand}")
        if cand and os.path.exists(cand):
            return cand

        name = self.editTemplatePath.text().strip()
        if name:
            self.main.logger.info(f"[Write] Resolve template: display='{name}'")
            # Try reference folder + _Processed
            project_path = self.main.project_path
            if project_path:
                for folder in (Path(project_path), Path(project_path) / "_Processed"):
                    p = folder / name
                    if p.exists():
                        self.main.logger.info(f"[Write] Resolve template: found in {folder}")
                        self.current_template_path = str(p)
                        self.editTemplatePath.setText(p.name)
                        self.editTemplatePath.setToolTip(str(p))
                        return str(p)
            # Try recent templates that match by filename
            try:
                p = self._recent_templates_path()
                if p.exists():
                    data = json.loads(p.read_text(encoding="utf-8"))
                    for item in data if isinstance(data, list) else []:
                        if isinstance(item, str) and os.path.exists(item) and os.path.basename(item) == name:
                            self.main.logger.info(f"[Write] Resolve template: found in recent list")
                            self.current_template_path = item
                            self.editTemplatePath.setText(Path(item).name)
                            self.editTemplatePath.setToolTip(item)
                            return item
            except Exception:
                pass
        return None


    def _clear_all_selections(self) -> None:
        row_count = self._write_grid_max_rows()
        for i in range(row_count):
            for j in range(5):
                combo = self.writeGridCells.get((i, j))
                if combo:
                    combo.setCurrentText("None")
        self._update_write_status()

    def _reset_radial_selections(self) -> None:
        row_count = self._write_grid_max_rows()
        for i in range(row_count):
            combo = self.writeGridCells.get((i, 4))
            if combo:
                combo.setCurrentText("None")
        self._update_write_status()

    def _auto_fill_by_name(self) -> None:
        row_count = self._write_grid_max_rows()
        for i in range(row_count):
            saxs_name = self._write_grid_value(i, 0)
            if not saxs_name or saxs_name == "None":
                continue
            for col_index in (1, 2, 3):
                combo = self.writeGridCells.get((i, col_index))
                if combo is None:
                    continue
                if combo.currentText() != "None":
                    continue
                if combo.findText(saxs_name) >= 0:
                    combo.setCurrentText(saxs_name)
        self._update_write_status()

    def _auto_fill_radial(self, row: int, saxs_name: str) -> None:
        if saxs_name == "None":
            combo = self.writeGridCells.get((row, 4))
            if combo:
                combo.setCurrentText("None")
            return
        radial = self._find_radial_for_saxs_name(saxs_name)
        if radial:
            radial = self._prefer_radial_csv(radial)
            combo = self.writeGridCells.get((row, 4))
            if combo:
                if combo.findText(radial) < 0:
                    combo.addItem(radial)
                combo.setCurrentText(radial)

    def _find_radial_for_saxs_name(self, saxs_name: str) -> str | None:
        """Filename-based radial lookup (no JSON)."""
        if not self._write_processed_root:
            return None
        tag = saxs_name.lower().replace(" ", "_")
        search_dirs = [self._write_processed_root, self._write_processed_root / "SAXS"]
        best: tuple[int, Path] | None = None
        for folder in search_dirs:
            if not folder.exists():
                continue
            for path in folder.glob("SAXS_radial_*"):
                if not path.is_file():
                    continue
                stem = path.stem.lower()
                if tag not in stem:
                    continue
                score = 0 if "matrix" in stem else 1 if "flat" in stem else 2
                if best is None or score < best[0]:
                    best = (score, path)
        return str(best[1].resolve()) if best else None

    def _prefer_radial_csv(self, path: str | Path) -> str:
        try:
            p = Path(path).expanduser()
        except Exception:
            return str(path)
        if p.suffix.lower() == ".csv":
            return str(p)
        if p.name.startswith("SAXS_radial_"):
            sample_tag = p.stem.replace("SAXS_radial_", "", 1)
            matrix_candidate = p.with_name(f"SAXS_radial_matrix_{sample_tag}.csv")
            if matrix_candidate.exists():
                return str(matrix_candidate)
            flat_candidate = p.with_name(f"SAXS_radial_flat_{sample_tag}.csv")
            if flat_candidate.exists():
                return str(flat_candidate)
        csv_candidate = p.with_suffix(".csv")
        if csv_candidate.exists():
            return str(csv_candidate)
        return str(p)
