import sys
import os
import logging
import traceback
from pathlib import Path
import json
import math
import re
import faulthandler

from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QGuiApplication, QAction, QActionGroup, QIcon, QPixmap, QColor, QBrush
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QMessageBox,
    QToolBar,
    QFileDialog,
    QDockWidget,
    QPlainTextEdit,
    QWidget,
    QStackedWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QToolButton,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QSizePolicy,
    QSpinBox,
    QLineEdit,
    QGroupBox,
    QGridLayout,
    QTreeWidget,
    QTreeWidgetItem,
    QMenu,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QAbstractItemView,
)
from PyQt6.uic import loadUi

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas 
from _Writer.write_ui import WriteUI
from _UI_modules.saxs_inspector import SAXSInspector
from _UI_modules.connector_inspector import ConnectorInspector
from _UI_modules.batch_inspector import BatchInspector
from _UI_modules.pli_inspector import PLIInspector
from _UI_modules.pi_inspector import PIInspector

# If you compiled Qt resources (e.g. icons) with pyrcc6, import them so runtime matches Designer preview.
try:
    import resources_rc  # type: ignore
except Exception:
    resources_rc = None

BASE_DIR = Path(__file__).resolve().parent

MODES = [
    ("Rheo", "Rheology"),
    ("SAXS", "SAXS"),
    ("PI", "PI"),
    ("PLI", "PLI"),
    ("Tribo", "Tribology"),
    ("WAXS", "WAXS"),
]
class QtLogHandler(logging.Handler):
    def __init__(self, widget: QPlainTextEdit):
        super().__init__()
        self.widget = widget

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            self.widget.appendPlainText(msg)
            self.widget.verticalScrollBar().setValue(self.widget.verticalScrollBar().maximum())
        except Exception:
            pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self._inspector_default_width = 300
        self.setDockOptions(self.dockOptions() | QMainWindow.DockOption.AnimatedDocks)
        self.setStyleSheet(
            "QMainWindow::separator {"
            "background: #333333;"
            "width: 0.5px;"
            "height: 0.5px;"
            "}"
        )

        ui_path = Path(__file__).resolve().parent / "MainWindow.ui"
        loadUi(str(ui_path), self)

        self.project_path: str | None = None
        self.mode_actions: dict[str, QAction] = {}
        self.canvas_pages: dict[str, QWidget] = {}
        self.inspector_pages: dict[str, QWidget] = {}
        self.logger = logging.getLogger("MuDPaW")
        # Hold strong refs to PLI controllers so their Qt slots stay alive
        self._pli_controllers: dict[str, object] = {}
        self._rheo_session_loaded_path: str | None = None
        self.rheoCanvas: FigureCanvas | None = None
        self.piCanvas: FigureCanvas | None = None
        self.piFigure = None
        self.piAx = None
        self._last_rheo_csv: str | None = None
        self._rheo_rerun_guard = False
        self._last_rheo_preview: dict | None = None
        self.write_ui = WriteUI(self)
        self.saxs_ui = SAXSInspector(self)
        self.connector_ui = ConnectorInspector(self)
        self.batch_ui = BatchInspector(self)
        self.pli_ui = PLIInspector(self)
        self.pi_ui = PIInspector(self)

        self._ensure_required_ui()
        self._setup_logging()
        self._setup_toolbars()
        self._wire_actions()
        self._restore_last_folder()
        self._show_home()
        self._shrink_inspector_width()

    # Ensure background workers and GUI-bound log handlers shut down cleanly.
    def closeEvent(self, event):  # type: ignore[override]
        try:
            # Stop connector background fetch thread (rsync) if running.
            if hasattr(self, "connector_ui") and hasattr(self.connector_ui, "shutdown"):
                self.connector_ui.shutdown()
            # Detach Qt log handler to avoid logging into destroyed widgets during teardown.
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                if isinstance(handler, QtLogHandler):
                    root_logger.removeHandler(handler)
            for handler in list(self.logger.handlers):
                if isinstance(handler, QtLogHandler):
                    self.logger.removeHandler(handler)
        except Exception:
            # We don't want close to crash because of cleanup; log to stderr silently.
            pass
        super().closeEvent(event)

    # -----------------
    # UI setup
    # -----------------

    def _ensure_required_ui(self) -> None:
        self.stackedCanvas = self.findChild(QStackedWidget, "stackedCanvas")
        if self.stackedCanvas is None:
            self.stackedCanvas = QStackedWidget(self)
            self.stackedCanvas.setObjectName("stackedCanvas")
            central = self.centralWidget() or QWidget(self)
            if self.centralWidget() is None:
                self.setCentralWidget(central)
            layout = central.layout()
            if layout is None:
                layout = QVBoxLayout(central)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.setSpacing(0)
            layout.addWidget(self.stackedCanvas)
        else:
            layout = self.stackedCanvas.parentWidget().layout()
            if layout is not None:
                layout.setContentsMargins(0, 0, 0, 0)
                layout.setSpacing(0)

        # Canvas pages
        self._ensure_canvas_page("pageRheoCanvas", "Rheo canvas ready")
        self._ensure_canvas_page("pageSAXSCanvas", "SAXS canvas (not implemented)")
        self._ensure_canvas_page("pageConnectorCanvas", "Connector canvas")
        self._ensure_canvas_page("pageBatchCanvas", "Batch canvas")
        self._ensure_canvas_page("pagePLICanvas", "PLI canvas (not implemented)")
        self._ensure_canvas_page("pagePICanvas", "PI geometry preview (run Center Geometry)")
        self._ensure_canvas_page("pageWriteCanvas", "Write canvas ready")
        self._ensure_canvas_page("pageTriboCanvas", "Tribology canvas (not implemented)")
        self._ensure_canvas_page("pageWAXSCanvas", "WAXS canvas (not implemented)")
        self._ensure_canvas_page("pageHomeCanvas", "Home")

        # Inspector dock + stack
        self.dockInspector = self.findChild(QDockWidget, "dockInspector")
        if self.dockInspector is None:
            self.dockInspector = QDockWidget("Inspector", self)
            self.dockInspector.setObjectName("dockInspector")
            self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dockInspector)

        self.stackInspector = self.dockInspector.findChild(QStackedWidget, "stackInspector")
        if self.stackInspector is None:
            self.stackInspector = QStackedWidget(self.dockInspector)
            self.stackInspector.setObjectName("stackInspector")

        self._ensure_inspector_tabs()

        self._ensure_inspector_page("pageRheoInspector", is_rheo=True)
        self._ensure_inspector_page("pageSAXSInspector")
        self._ensure_inspector_page("pageConnectorInspector")
        self._ensure_inspector_page("pageBatchInspector")
        self._ensure_inspector_page("pagePLIInspector")
        self._ensure_inspector_page("pagePIInspector")
        self._ensure_inspector_page("pageWriteInspector", is_write=True)
        self._ensure_inspector_page("pageTriboInspector")
        self._ensure_inspector_page("pageWAXSInspector")
        self._ensure_inspector_page("pageHomeInspector")

        # Console dock
        self.dockConsole = self.findChild(QDockWidget, "dockConsole")
        if self.dockConsole is None:
            self.dockConsole = QDockWidget("Console", self)
            self.dockConsole.setObjectName("dockConsole")
            self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, self.dockConsole)

        self.consoleOutput = self.dockConsole.findChild(QPlainTextEdit, "consoleOutput")
        if self.consoleOutput is None:
            self.consoleOutput = QPlainTextEdit(self.dockConsole)
            self.consoleOutput.setObjectName("consoleOutput")
            self.consoleOutput.setReadOnly(True)
            self.dockConsole.setWidget(self.consoleOutput)

    def _ensure_inspector_tabs(self) -> None:
        # Create a shared inspector+processing tab container.
        tabs = self.dockInspector.findChild(QTabWidget, "inspectorTabs")
        if tabs is None:
            tabs = QTabWidget(self.dockInspector)
            tabs.setObjectName("inspectorTabs")
        self.inspectorTabs = tabs

        # Build processing panel once.  
        if not hasattr(self, "processingPanel"):
            self._build_processing_panel()

        # Ensure the tab widget is the dock's widget.
        if self.dockInspector.widget() is not self.inspectorTabs:
            self.dockInspector.setWidget(self.inspectorTabs)

        # Add/ensure tabs.
        if self.inspectorTabs.indexOf(self.stackInspector) == -1:
            self.inspectorTabs.addTab(self.stackInspector, "Inspector")
        if self.inspectorTabs.indexOf(self.processingPanel) == -1:
            self.inspectorTabs.addTab(self.processingPanel, "Processing")
        try:
            self.inspectorTabs.currentChanged.connect(self._on_inspector_tab_changed)
        except Exception:
            pass

    def _build_processing_panel(self) -> None:
        self.processingPanel = QWidget(self.dockInspector)
        self.processingPanel.setObjectName("processingPanel")
        layout = QVBoxLayout(self.processingPanel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        title = QLabel("Active processing jobs", self.processingPanel)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self.processingList = QListWidget(self.processingPanel)
        self.processingList.setObjectName("processingList")
        self.processingList.setMinimumHeight(120)
        self.processingList.setMaximumHeight(200)
        self.processingList.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
        self.processingList.currentRowChanged.connect(self._on_processing_row_changed)
        layout.addWidget(self.processingList)

        self.processingStack = QStackedWidget(self.processingPanel)
        self.processingStack.setObjectName("processingStack")
        placeholder = QLabel("No active jobs.", self.processingPanel)
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processingStack.addWidget(placeholder)
        layout.addWidget(self.processingStack, 2)

        self._processing_widgets: dict[str, QWidget] = {}
        self._processing_ids: list[str] = []
        self._processing_controllers: list[object] = []

    def _on_processing_row_changed(self, row: int) -> None:
        if not hasattr(self, "processingStack"):
            return
        # index 0 is placeholder
        stack_index = row + 1
        if 0 <= stack_index < self.processingStack.count():
            self.processingStack.setCurrentIndex(stack_index)
        else:
            self.processingStack.setCurrentIndex(0)

    def _on_inspector_tab_changed(self, _idx: int) -> None:
        try:
            if self.inspectorTabs.currentWidget() is self.processingPanel:
                target = 520
                self.dockInspector.setMinimumWidth(target)
                self.dockInspector.setMaximumWidth(target)
                self.resizeDocks([self.dockInspector], [target], Qt.Orientation.Horizontal)
            else:
                self._shrink_inspector_width()
        except Exception:
            pass

    def _add_processing_job(self, job_id: str, title: str, widget: QWidget) -> None:
        self._clear_processing_jobs()
        if job_id in self._processing_widgets:
            return
        self.processingList.addItem(title)
        self._processing_ids.append(job_id)
        self._processing_widgets[job_id] = widget
        self.processingStack.addWidget(widget)
        self.processingList.setCurrentRow(len(self._processing_ids) - 1)
        if hasattr(self, "inspectorTabs"):
            self.inspectorTabs.setCurrentWidget(self.processingPanel)

    def _clear_processing_jobs(self) -> None:
        if not hasattr(self, "_processing_ids"):
            return
        for job_id in list(self._processing_ids):
            self._remove_processing_job(job_id)
        self._processing_ids = []
        self._processing_widgets = {}
        try:
            self.processingList.clear()
        except Exception:
            pass

    def _remove_processing_job(self, job_id: str) -> None:
        if job_id not in self._processing_widgets:
            return
        idx = self._processing_ids.index(job_id)
        self._processing_ids.pop(idx)
        w = self._processing_widgets.pop(job_id)
        self.processingStack.removeWidget(w)
        w.setParent(None)
        item = self.processingList.takeItem(idx)
        if item is not None:
            del item
        if not self._processing_ids:
            self.processingStack.setCurrentIndex(0)

    def _set_canvas_content(self, page_name: str, widget: QWidget) -> None:
        page = self.canvas_pages.get(page_name)
        if page is None:
            return
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # Clear existing widgets
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
        layout.addWidget(widget, 1)

    def _ensure_canvas_page(self, name: str, label_text: str) -> None:
        page = self.findChild(QWidget, name)
        if page is None:
            page = QWidget(self.stackedCanvas)
            page.setObjectName(name)
            layout = QVBoxLayout(page)
            if name not in {"pageWriteCanvas", "pageHomeCanvas"}:
                label = QLabel(label_text, page)
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout.addWidget(label)
            self.stackedCanvas.addWidget(page)
        self.canvas_pages[name] = page
        if name == "pageHomeCanvas":
            self._ensure_home_canvas(page)
        if name == "pageRheoCanvas":
            # Ensure a status label for Rheo canvas
            label = page.findChild(QLabel, "rheoStatusLabel")
            if label is None:
                label = QLabel(label_text, page)
                label.setObjectName("rheoStatusLabel")
                label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                layout = page.layout()
                if layout is None:
                    layout = QVBoxLayout(page)
                else:
                    while layout.count():
                        item = layout.takeAt(0)
                        w = item.widget()
                        if w is not None:
                            w.setParent(None)
                layout.addWidget(label)
            self.rheoStatusLabel = label
            self._ensure_rheo_canvas(page)
        if name == "pageSAXSCanvas":
            self._ensure_saxs_canvas(page)
        if name == "pageConnectorCanvas":
            self._ensure_connector_canvas(page)
        if name == "pagePICanvas":
            # Defer PI canvas creation until we have geometry data to show.
            pass
        if name == "pageWriteCanvas":
            self.write_ui.build_write_canvas(page)

    def _ensure_home_canvas(self, page: QWidget) -> None:
        if page.layout() is not None and page.findChild(QStackedWidget, "homeStack") is not None:
            return
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)
        self.homeStack = QStackedWidget(page)
        self.homeStack.setObjectName("homeStack")
        layout.addWidget(self.homeStack, 1)

        # Blank home page
        blank_page = QWidget(self.homeStack)
        blank_layout = QVBoxLayout(blank_page)
        blank_layout.setContentsMargins(0, 0, 0, 0)
        blank_layout.addStretch(1)
        self.homeStack.addWidget(blank_page)

        # About page
        about_page = QWidget(self.homeStack)
        about_layout = QVBoxLayout(about_page)
        about_layout.setContentsMargins(0, 0, 0, 0)
        about_layout.setSpacing(12)
        images_row = QHBoxLayout()
        images_row.setSpacing(12)
        self.aboutLogoMudpaw = QLabel(about_page)
        self.aboutLogoArtSI = QLabel(about_page)
        for lbl in (self.aboutLogoMudpaw, self.aboutLogoArtSI):
            lbl.setFixedSize(100, 100)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        images_row.addWidget(self.aboutLogoMudpaw)
        images_row.addWidget(self.aboutLogoArtSI)
        images_row.addStretch(1)
        about_layout.addLayout(images_row)

        self.aboutText = QLabel(about_page)
        self.aboutText.setWordWrap(True)
        self.aboutText.setTextFormat(Qt.TextFormat.RichText)
        self.aboutText.setOpenExternalLinks(True)
        about_layout.addWidget(self.aboutText)
        about_layout.addStretch(1)
        self.homeStack.addWidget(about_page)

        self._update_about_content()

    def _show_home(self) -> None:
        canvas = self.canvas_pages.get("pageHomeCanvas")
        inspector = self.inspector_pages.get("pageHomeInspector")
        if canvas is not None:
            self.stackedCanvas.setCurrentWidget(canvas)
            if hasattr(self, "homeStack"):
                self.homeStack.setCurrentIndex(0)
        if inspector is not None:
            self.stackInspector.setCurrentWidget(inspector)
        self.logger.info("Switched mode: Home")

    def _show_about(self) -> None:
        canvas = self.canvas_pages.get("pageHomeCanvas")
        inspector = self.inspector_pages.get("pageHomeInspector")
        if canvas is not None:
            self.stackedCanvas.setCurrentWidget(canvas)
            if hasattr(self, "homeStack"):
                self.homeStack.setCurrentIndex(1)
        if inspector is not None:
            self.stackInspector.setCurrentWidget(inspector)
        self.logger.info("Switched mode: About")

    def _show_connector(self) -> None:
        canvas = self.canvas_pages.get("pageConnectorCanvas")
        inspector = self.inspector_pages.get("pageConnectorInspector")
        if canvas is not None:
            self.stackedCanvas.setCurrentWidget(canvas)
        if inspector is not None:
            self.stackInspector.setCurrentWidget(inspector)
        self._shrink_inspector_width()
        self.logger.info("Switched mode: Connector")

    def _show_batch(self) -> None:
        canvas = self.canvas_pages.get("pageBatchCanvas")
        inspector = self.inspector_pages.get("pageBatchInspector")
        if canvas is not None:
            self.stackedCanvas.setCurrentWidget(canvas)
        if inspector is not None:
            self.stackInspector.setCurrentWidget(inspector)
        self._shrink_inspector_width()
        self.logger.info("Switched mode: Batch")

    def _update_home_canvas(self) -> None:
        if not hasattr(self, "homeTree"):
            return
        self.homeTree.clear()

        def _make_dim_item(text: str) -> QTreeWidgetItem:
            item = QTreeWidgetItem([text])
            try:
                color = QColor(160, 160, 160)
                item.setForeground(0, QBrush(color))
                font = item.font(0)
                font.setItalic(True)
                item.setFont(0, font)
                # Visually de-emphasize and prevent selection/expansion clicks.
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEnabled & ~Qt.ItemFlag.ItemIsSelectable)
            except Exception:
                pass
            return item

        if not self.project_path or not os.path.isdir(self.project_path):
            self.homeRefLabel.setText("Reference folder path: —")
            top = QTreeWidgetItem(["Reference folder not selected"])
            self.homeTree.addTopLevelItem(top)
            self.homeTree.expandAll()
            return

        ref_path = self.project_path
        if os.path.basename(ref_path) == "Inputs (reference)":
            ref_path = os.path.dirname(ref_path)
        if os.path.basename(ref_path).lower() == "pi":
            ref_path = os.path.dirname(ref_path)
        ref_name = os.path.basename(ref_path)
        self.homeRefLabel.setText(f"Reference folder path: {ref_path}")

        root = QTreeWidgetItem([ref_name])
        root.setExpanded(True)
        self.homeTree.addTopLevelItem(root)

        inputs_root = os.path.join(ref_path, "Inputs (reference)")
        if os.path.isdir(inputs_root):
            inputs_label = "Inputs (reference)"
        else:
            inputs_root = ref_path
            inputs_label = "Inputs"
        inputs = QTreeWidgetItem([inputs_label])
        inputs.setExpanded(True)
        root.addChild(inputs)

        for key, folder_name in MODES:
            mod_item = QTreeWidgetItem([key])
            mod_item.setExpanded(True)
            inputs.addChild(mod_item)
            folder = os.path.join(inputs_root, folder_name)
            if not os.path.isdir(folder):
                mod_item.addChild(_make_dim_item("(not found)"))
                continue
            folders = [
                f for f in os.listdir(folder)
                if os.path.isdir(os.path.join(folder, f)) and not f.startswith(".")
            ]
            files = [
                f for f in os.listdir(folder)
                if os.path.isfile(os.path.join(folder, f)) and not f.startswith(".")
            ]
            folders.sort(key=str.lower)
            files.sort(key=str.lower)
            if not folders and not files:
                mod_item.addChild(_make_dim_item("(empty)"))
                continue
            shown = 0
            for f in folders:
                if shown >= 200:
                    break
                mod_item.addChild(QTreeWidgetItem([f]))
                shown += 1
            for f in files:
                if shown >= 200:
                    break
                mod_item.addChild(QTreeWidgetItem([f]))
                shown += 1
            total = len(folders) + len(files)
            if total > 200:
                mod_item.addChild(QTreeWidgetItem([f"... ({total-200} more)"]))

        processed_root = os.path.join(ref_path, "_Processed")
        processed = QTreeWidgetItem(["Processed (_Processed)"])
        processed.setExpanded(True)
        root.addChild(processed)

        if not os.path.isdir(processed_root):
            processed.addChild(_make_dim_item("(not found)"))
        else:
            by_mod = {k: [] for k, _ in MODES}
            by_mod["Other"] = []
            for fname in os.listdir(processed_root):
                if fname.startswith("."):
                    continue
                full = os.path.join(processed_root, fname)
                if os.path.isdir(full):
                    # skip nested folders for brevity
                    continue
                low = fname.lower()
                if low.startswith(("saxs_", "_output_saxs")):
                    by_mod["SAXS"].append(fname)
                elif low.startswith(("rheo_", "_output_rheology")):
                    by_mod["Rheo"].append(fname)
                elif low.startswith(("pli_", "_output_pli")):
                    by_mod["PLI"].append(fname)
                elif low.startswith(("pi_", "_output_pi")):
                    by_mod["PI"].append(fname)
                elif "tribo" in low:
                    by_mod["Tribo"].append(fname)
                elif "waxs" in low:
                    by_mod["WAXS"].append(fname)
                else:
                    by_mod["Other"].append(fname)

            for key, _folder in MODES:
                item = QTreeWidgetItem([key])
                item.setExpanded(True)
                processed.addChild(item)
                files = sorted(by_mod.get(key, []), key=str.lower)
                if not files:
                    item.addChild(_make_dim_item("(none)"))
                    continue
                for f in files[:200]:
                    item.addChild(QTreeWidgetItem([f]))
                if len(files) > 200:
                    item.addChild(QTreeWidgetItem([f"... ({len(files)-200} more)"]))

            other = QTreeWidgetItem(["Other"])
            other.setExpanded(True)
            processed.addChild(other)
            files = sorted(by_mod.get("Other", []), key=str.lower)
            if not files:
                other.addChild(_make_dim_item("(none)"))
            else:
                for f in files[:200]:
                    other.addChild(QTreeWidgetItem([f]))
                if len(files) > 200:
                    other.addChild(QTreeWidgetItem([f"... ({len(files)-200} more)"]))
        self.homeTree.expandAll()

    def _update_about_content(self) -> None:
        mudpaw_path = BASE_DIR / "_Resources" / "2026_MuPaW.png"
        artsi_path = BASE_DIR / "_Resources" / "2026_artSI.png"
        if mudpaw_path.exists():
            pix = QPixmap(str(mudpaw_path)).scaled(100, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.aboutLogoMudpaw.setPixmap(pix)
        if artsi_path.exists():
            pix = QPixmap(str(artsi_path)).scaled(200, 100, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self.aboutLogoArtSI.setPixmap(pix)

        text = (
            "<b>Multimodal Data Processor and Writer (MuDPaW)</b> was created to assist the use of "
            "multihyphenated techniques developed within the Advanced Rheological Testing Science "
            "Initiative at MAX IV. For questions contact "
            "<a href='mailto:roland.kadar@chalmers.se'>roland.kadar@chalmers.se</a> and <a href='mailto:marko.bek@chalmers.se'>marko.bek@chalmers.se</a>.<br><br>"
            "<a href=''>GitHub repository</a><br>. Contributors: RK, MB and Dr. Stuart Ansell. <br><br>"  
            "<a href='https://www.maxiv.lu.se/rheology'>www.maxiv.lu.se/rheology</a>"
        )
        self.aboutText.setText(text)

    def _ensure_rheo_canvas(self, page: QWidget) -> None:
        if self.rheoCanvas is not None:
            return
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        fig = Figure(figsize=(5, 4), dpi=100)
        self.rheoCanvas = FigureCanvas(fig)
        self.rheoCanvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.rheoCanvas)

    def _ensure_pi_canvas(self, page: QWidget) -> None:
        if self.piCanvas is not None:
            return
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        else:
            while layout.count():
                item = layout.takeAt(0)
                w = item.widget()
                if w is not None:
                    w.setParent(None)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        fig = Figure(figsize=(5, 4), dpi=100)
        self.piFigure = fig
        self.piAx = fig.add_subplot(111)
        self.piCanvas = FigureCanvas(fig)
        self.piCanvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        layout.addWidget(self.piCanvas)

    def pi_show_geometry(self, data, center, inner_radius, outer_radius) -> None:
        if self.piCanvas is None or self.piAx is None or self.piFigure is None:
            page = self.canvas_pages.get("pagePICanvas")
            if page is None:
                return
            self._ensure_pi_canvas(page)
        if self.piAx is None:
            return
        ax = self.piAx
        ax.clear()
        ax.imshow(data, cmap="gray")
        height, width = data.shape
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        from matplotlib.patches import Circle
        circle_inner = Circle((center[0], center[1]), inner_radius, color="r", fill=False)
        circle_outer = Circle((center[0], center[1]), outer_radius, color="b", fill=False)
        ax.add_patch(circle_inner)
        ax.add_patch(circle_outer)
        # horizontal chord at 2/3 of outer radius below center
        import numpy as np
        y_offset = (2 / 3) * outer_radius
        y_line = center[1] + y_offset
        if outer_radius > abs(y_offset):
            half_span = np.sqrt(outer_radius ** 2 - y_offset ** 2)
            x_min = center[0] - half_span
            x_max = center[0] + half_span
            ax.plot([x_min, x_max], [y_line, y_line], "r--", linewidth=1)
        self.piCanvas.draw()

    def pi_show_extract(self, circle, line, interval, n_intervals) -> None:
        page = self.canvas_pages.get("pagePICanvas")
        if page is None:
            return
        if self.piCanvas is None or self.piFigure is None:
            self._ensure_pi_canvas(page)
        if self.piFigure is None or self.piCanvas is None:
            return
        self.piFigure.clear()
        ax1 = self.piFigure.add_subplot(1, 2, 1)
        ax2 = self.piFigure.add_subplot(1, 2, 2)
        im1 = ax1.imshow(circle.T, aspect="auto", cmap="viridis", vmin=0, vmax=270)
        ax1.set_title("Extracted Space-Time Diagram (Circle)")
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Radius")
        self.piFigure.colorbar(im1, ax=ax1)

        im2 = ax2.imshow(line.T, aspect="auto", cmap="viridis", vmin=0, vmax=270)
        ax2.set_title("Extracted Space-Time Diagram (Line)")
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Position")
        self.piFigure.colorbar(im2, ax=ax2)

        try:
            n = int(n_intervals)
        except Exception:
            n = 0
        for i in range(n):
            x = int(i * interval)
            ax1.axvline(x=x, color="white", linestyle="--")
            ax2.axvline(x=x, color="white", linestyle="--")
        self.piCanvas.draw()

    def _ensure_saxs_canvas(self, page: QWidget) -> None:
        self.saxs_ui.ensure_canvas(page)
        self._clear_rheo_plot()

    def _ensure_connector_canvas(self, page: QWidget) -> None:
        self.connector_ui.ensure_canvas(page)

    def _ensure_inspector_page(self, name: str, *, is_rheo: bool = False, is_write: bool = False) -> None:
        page = self.findChild(QWidget, name)
        if page is None:
            page = QWidget(self.stackInspector)
            page.setObjectName(name)
            self.stackInspector.addWidget(page)
        self.inspector_pages[name] = page
        if is_rheo:
            self._build_rheo_inspector(page)
        elif name == "pageSAXSInspector":
            self.saxs_ui.build_inspector(page)
        elif name == "pageConnectorInspector":
            self.connector_ui.build_inspector(page)
        elif name == "pageBatchInspector":
            self.batch_ui.build_inspector(page)
        elif name == "pagePLIInspector":
            self.pli_ui.build_inspector(page)
        elif name == "pagePIInspector":
            self.pi_ui.build_inspector(page)
        elif is_write:
            self.write_ui.build_write_inspector(page)
        elif name == "pageHomeInspector":
            self._build_home_inspector(page)
        else:
            self._build_placeholder_inspector(page, name)

    def _build_home_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.homeRefLabel = QLabel("Reference folder path: —", page)
        self.homeRefLabel.setObjectName("homeRefLabel")
        self.homeRefLabel.setWordWrap(True)
        layout.addWidget(self.homeRefLabel)
        self.homeTree = QTreeWidget(page)
        self.homeTree.setObjectName("homeTree")
        self.homeTree.setHeaderHidden(True)
        layout.addWidget(self.homeTree, 1)

    def _build_rheo_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(8)

        title = QLabel("Rheology parameters", page)
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        # Mode selector
        row_mode = QHBoxLayout()
        row_mode.addWidget(QLabel("Mode:", page))
        self.rheoModeCombo = QComboBox(page)
        self.rheoModeCombo.addItems(["triggered", "nontriggered", "other"])
        self.rheoModeCombo.currentTextChanged.connect(self._on_rheo_mode_changed)
        row_mode.addWidget(self.rheoModeCombo)
        row_mode.addStretch(1)
        layout.addLayout(row_mode)

        # Steady window
        row_steady = QHBoxLayout()
        row_steady.addWidget(QLabel("Steady window (s):", page))
        self.rheoSteadySpin = QDoubleSpinBox(page)
        self.rheoSteadySpin.setRange(0.1, 1_000_000.0)
        self.rheoSteadySpin.setDecimals(2)
        self.rheoSteadySpin.setValue(10.0)
        row_steady.addWidget(self.rheoSteadySpin)
        row_steady.addStretch(1)
        layout.addLayout(row_steady)
        self._on_rheo_mode_changed(self.rheoModeCombo.currentText())

        # Plot options (placeholder for upcoming canvas controls)
        plot_title = QLabel("Plot options", page)
        plot_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(plot_title)

        crop_row = QHBoxLayout()
        self.rheoCropCheck = QCheckBox("Apply shear-rate crop", page)
        self.rheoCropCheck.stateChanged.connect(self._rerun_last_rheo_if_any)
        crop_row.addWidget(self.rheoCropCheck)
        crop_row.addStretch(1)
        layout.addLayout(crop_row)

        crop_vals = QHBoxLayout()
        crop_vals.addWidget(QLabel("Min idx:", page))
        self.rheoCropMin = QSpinBox(page)
        self.rheoCropMin.setRange(1, 1)
        self.rheoCropMin.setValue(1)
        self.rheoCropMin.setSingleStep(1)
        self.rheoCropMin.valueChanged.connect(self._rerun_last_rheo_if_any)
        crop_vals.addWidget(self.rheoCropMin)
        crop_vals.addWidget(QLabel("Max idx:", page))
        self.rheoCropMax = QSpinBox(page)
        self.rheoCropMax.setRange(1, 1)
        self.rheoCropMax.setValue(1)
        self.rheoCropMax.setSingleStep(1)
        self.rheoCropMax.valueChanged.connect(self._rerun_last_rheo_if_any)
        crop_vals.addWidget(self.rheoCropMax)
        crop_vals.addStretch(1)
        layout.addLayout(crop_vals)

        # Reference folder label
        self.rheoFolderLabel = QLabel("Reference folder name: —", page)
        self.rheoFolderLabel.setWordWrap(True)
        layout.addWidget(self.rheoFolderLabel)

        # Sample list
        self.rheoSampleList = QListWidget(page)
        self.rheoSampleList.itemClicked.connect(self._on_rheo_item_selected)
        layout.addWidget(self.rheoSampleList)

        # Actions row
        btn_row = QHBoxLayout()
        self.rheoRefreshButton = QPushButton("Refresh list", page)
        self.rheoRefreshButton.clicked.connect(self._refresh_rheo_list)
        btn_row.addWidget(self.rheoRefreshButton)

        self.rheoRunSelectedButton = QPushButton("Run selected", page)
        self.rheoRunSelectedButton.clicked.connect(self._run_rheo_selected)
        btn_row.addWidget(self.rheoRunSelectedButton)

        self.rheoProcessAllButton = QPushButton("Process all", page)
        self.rheoProcessAllButton.clicked.connect(self._process_all_rheo)
        btn_row.addWidget(self.rheoProcessAllButton)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)
        layout.addStretch(1)

    # SAXS inspector moved to _UI_modules/saxs_inspector.py

    def _build_placeholder_inspector(self, page: QWidget, name: str) -> None:
        if page.layout() is None:
            layout = QVBoxLayout(page)
        else:
            layout = page.layout()
        label = QLabel("Not implemented yet", page)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(label)
        btn = QPushButton("Run", page)
        btn.clicked.connect(lambda _=False, n=name: self._log_not_implemented(n))
        layout.addWidget(btn)
        layout.addStretch(1)

    def _setup_logging(self) -> None:
        self.logger.setLevel(logging.INFO)
        handler = QtLogHandler(self.consoleOutput)
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.INFO)
        # Avoid duplicate Qt handlers if reloaded
        if not any(isinstance(h, QtLogHandler) for h in root_logger.handlers):
            root_logger.addHandler(handler)
        self.logger.info("Welcome to MuDPaW v6!")
        self.logger.info("Console logger ready.")

    def _setup_toolbars(self) -> None:
        for tb in self.findChildren(QToolBar):
            tb.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
            tb.setIconSize(QSize(30, 30))

        self.modeToolBar = self.findChild(QToolBar, "toolBar")
        if self.modeToolBar is None:
            self.modeToolBar = self.findChild(QToolBar, "modeToolBar")
        if self.modeToolBar is None:
            self.modeToolBar = QToolBar("Modes", self)
            self.modeToolBar.setObjectName("modeToolBar")
            self.addToolBar(Qt.ToolBarArea.LeftToolBarArea, self.modeToolBar)

        self.modeToolBar.setIconSize(QSize(30, 30))

        # Choose reference folder action
        self.actionChooseReferenceFolder = self.findChild(QAction, "actionChooseReferenceFolder")
        if self.actionChooseReferenceFolder is None:
            self.actionChooseReferenceFolder = QAction("Choose reference folder", self)
            self.actionChooseReferenceFolder.setObjectName("actionChooseReferenceFolder")
        if self.actionChooseReferenceFolder.icon().isNull():
            action_select = self.findChild(QAction, "actionSelect_Reference_Folder")
            if action_select is not None and not action_select.icon().isNull():
                self.actionChooseReferenceFolder.setIcon(action_select.icon())
            else:
                icon_path = BASE_DIR / "_Resources" / "folder.svg"
                if icon_path.exists():
                    self.actionChooseReferenceFolder.setIcon(QIcon(str(icon_path)))

        action_select = self.findChild(QAction, "actionSelect_Reference_Folder")
        select_on_toolbar = action_select is not None and self.modeToolBar.widgetForAction(action_select) is not None
        # Home action (force above reference folder)
        self.actionHome = self.findChild(QAction, "actionHome")
        if self.actionHome is None:
            self.actionHome = QAction("Home", self)
            self.actionHome.setObjectName("actionHome")

        # Remove existing placements to enforce order
        if action_select is not None and self.modeToolBar.widgetForAction(action_select) is not None:
            self.modeToolBar.removeAction(action_select)
        if self.modeToolBar.widgetForAction(self.actionChooseReferenceFolder) is not None:
            self.modeToolBar.removeAction(self.actionChooseReferenceFolder)
        if self.modeToolBar.widgetForAction(self.actionHome) is not None:
            self.modeToolBar.removeAction(self.actionHome)

        self.modeToolBar.addAction(self.actionHome)
        self.modeToolBar.addSeparator()
        if not select_on_toolbar:
            self.modeToolBar.addAction(self.actionChooseReferenceFolder)
        else:
            self.modeToolBar.addAction(action_select)
        self.modeToolBar.addSeparator()

        # Re-add connector/batch/write actions to keep them below Home + reference
        for action_name in ("actionMAX_IV_Connector2", "action_Batch_processor2", "actionData_Writer_to_Template2"):
            act = self.findChild(QAction, action_name)
            if act is not None and self.modeToolBar.widgetForAction(act) is not None:
                self.modeToolBar.removeAction(act)
            if act is not None:
                self.modeToolBar.addAction(act)
        self.modeToolBar.addSeparator()

        # Improve visual/hover/checked state to mimic Zoom-like sidebar
        self.modeToolBar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
        # Neutral light-gray palette (less blue tint)
        self.modeToolBar.setStyleSheet(
            """
            QToolBar {
                background: #f2f2f2;
                border: 0px;
            }
            QToolButton {
                background: transparent;
                margin: 4px;
                padding: 6px;
                border-radius: 10px;
                color: #444;
            }
            QToolButton:hover {
                background: #e0e0e0;
            }
            QToolButton:checked {
                background: #ffffff;
                border: 1px solid #d0d0d0;
                color: #000;
            }
            """
        )
        # Make all mode actions checkable and exclusive so the active one stays highlighted.
        self._mode_action_group = QActionGroup(self)
        self._mode_action_group.setExclusive(True)
        for act in self.modeToolBar.actions():
            if act.isSeparator():
                continue
            act.setCheckable(True)
            self._mode_action_group.addAction(act)
        # Mode actions (exclusive)
        group = QActionGroup(self)
        group.setExclusive(True)

        for key, _folder in MODES:
            action_name = f"actionMode{key}"
            act = self.findChild(QAction, action_name)
            if act is None:
                act = QAction(key, self)
                act.setObjectName(action_name)
            act.setCheckable(True)
            act.setEnabled(False)
            act.triggered.connect(lambda _=False, k=key: self.set_mode(k))
            group.addAction(act)
            self.modeToolBar.addAction(act)
            self.mode_actions[key] = act

            btn = self.modeToolBar.widgetForAction(act)
            if isinstance(btn, QToolButton):
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
                f = btn.font()
                f.setBold(True)
                btn.setFont(f)

        # Place quit action at bottom of toolbar (icon only, no text)
        quit_action = self.findChild(QAction, "actionQuit")
        if quit_action is not None:
            if self.modeToolBar.widgetForAction(quit_action) is not None:
                self.modeToolBar.removeAction(quit_action)
            spacer = QWidget(self)
            spacer.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Expanding)
            self.modeToolBar.addWidget(spacer)
            self.modeToolBar.addSeparator()
            self.modeToolBar.addAction(quit_action)
            btn = self.modeToolBar.widgetForAction(quit_action)
            if isinstance(btn, QToolButton):
                btn.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                btn.setText("")

        # Cleanup: remove redundant/leading/trailing separators
        actions = self.modeToolBar.actions()
        prev_sep = True
        for act in actions:
            if act.isSeparator():
                if prev_sep:
                    self.modeToolBar.removeAction(act)
                prev_sep = True
            else:
                prev_sep = False
        # Remove trailing separators
        actions = self.modeToolBar.actions()
        for act in reversed(actions):
            if act.isSeparator():
                self.modeToolBar.removeAction(act)
            else:
                break

    def _wire_actions(self) -> None:
        # Choose reference folder actions
        self.actionChooseReferenceFolder.triggered.connect(self._select_reference_folder)
        if hasattr(self, "actionHome"):
            self.actionHome.triggered.connect(self._show_home)
        if hasattr(self, "actionAbout"):
            self.actionAbout.triggered.connect(self._show_about)
        action_connector = self.findChild(QAction, "actionMAX_IV_Connector2")
        if action_connector is not None:
            action_connector.triggered.connect(self._show_connector)
        action_batch = self.findChild(QAction, "action_Batch_processor2")
        if action_batch is not None:
            action_batch.triggered.connect(self._show_batch)

        # Info/About menu
        menubar = self.menuBar()
        if menubar is not None:
            info_menu = menubar.findChild(QMenu, "menuInfo")
            if info_menu is None:
                info_menu = QMenu("Info", self)
                info_menu.setObjectName("menuInfo")
                menubar.addMenu(info_menu)
            if not hasattr(self, "actionAbout"):
                self.actionAbout = QAction("About", self)
                self.actionAbout.setObjectName("actionAbout")
                self.actionAbout.triggered.connect(self._show_about)
            if self.actionAbout not in info_menu.actions():
                info_menu.addAction(self.actionAbout)

        action_menu = self.findChild(QAction, "actionSelect_reference_folder")
        if action_menu is not None:
            action_menu.triggered.connect(self._select_reference_folder)
        action_toolbar = self.findChild(QAction, "actionSelect_Reference_Folder")
        if action_toolbar is not None:
            action_toolbar.triggered.connect(self._select_reference_folder)

        action_write_toolbar = self.findChild(QAction, "actionData_Writer_to_Template2")
        if action_write_toolbar is not None:
            action_write_toolbar.triggered.connect(lambda _=False: self.set_mode("Write"))
        action_write_menu = self.findChild(QAction, "actionData_Writer_to_Template")
        if action_write_menu is not None:
            action_write_menu.triggered.connect(lambda _=False: self.set_mode("Write"))

        # Quit actions
        for quit_name in ("actionQuit", "actionExit"):
            act = self.findChild(QAction, quit_name)
            if act is not None:
                act.triggered.connect(self.close)

        # Other actions: log not implemented
        for act in self.findChildren(QAction):
            if act in {self.actionChooseReferenceFolder}:
                continue
            if act.objectName().startswith("actionMode"):
                continue
            if act.objectName() in {
                "actionSelect_reference_folder",
                "actionSelect_Reference_Folder",
                "actionQuit",
                "actionExit",
                "actionData_Writer_to_Template2",
                "actionData_Writer_to_Template",
                "actionMAX_IV_Connector2",
            }:
                continue
            if hasattr(self, "actionHome") and act == self.actionHome:
                continue
            if hasattr(self, "actionAbout") and act == self.actionAbout:
                continue
            act.triggered.connect(lambda _=False, n=act.text() or act.objectName(): self._log_not_implemented(n))

    # -----------------
    # State + behavior
    # -----------------

    def _restore_last_folder(self) -> None:
        last_path = BASE_DIR / "_Miscell" / "last_folder.txt"
        if last_path.exists():
            try:
                path = last_path.read_text(encoding="utf-8").strip()
                if path:
                    self.project_path = path
                    self.logger.info(f"Restored last folder: {path}")
                    self._detect_and_update_modes(path)
                    self.write_ui.update_write_header(self.project_path)
                    self._refresh_rheo_list()
                    self._update_home_canvas()
                    self._show_home()
            except Exception:
                pass

    def _select_reference_folder(self) -> None:
        try:
            start_dir = self.project_path if self.project_path else str(Path.home())
            folder = QFileDialog.getExistingDirectory(self, "Select reference folder", start_dir)
            if folder:
                self.project_path = folder
                self.logger.info(f"Selected reference folder: {folder}")
                (BASE_DIR / "_Miscell").mkdir(parents=True, exist_ok=True)
                (BASE_DIR / "_Miscell" / "last_folder.txt").write_text(folder, encoding="utf-8")
                if self.statusBar() is not None:
                    self.statusBar().showMessage(f"Selected: {folder}", 4000)
                self._detect_and_update_modes(folder)
                self.write_ui.update_write_header(self.project_path)
                # Immediately refresh writer data so tables aren't stale until switching modes.
                self.write_ui.refresh_write_data(initial=False)
                self._refresh_rheo_list()
                self._update_home_canvas()
                self._show_home()
        except Exception as e:
            self._log_exception("Failed to select reference folder", e)

    def _detect_and_update_modes(self, folder: str) -> None:
        availability = self._detect_modalities(folder)
        for key, (enabled, reason) in availability.items():
            act = self.mode_actions.get(key)
            if not act:
                continue
            act.setEnabled(enabled)
            if reason:
                act.setToolTip(reason)
        self._update_home_canvas()

    def _detect_modalities(self, folder: str) -> dict[str, tuple[bool, str]]:
        results: dict[str, tuple[bool, str]] = {}
        if not folder or not os.path.isdir(folder):
            for key, _ in MODES:
                results[key] = (False, "Choose reference folder first")
            return results

        # Rheology detection (folder or processed outputs)
        rheo_folder = os.path.join(folder, "Rheology")
        rheo_csvs = []
        if os.path.isdir(rheo_folder):
            rheo_csvs = [f for f in os.listdir(rheo_folder) if f.lower().endswith(".csv") and not f.startswith(".")]
        processed = os.path.join(folder, "_Processed")
        processed_hits = []
        if os.path.isdir(processed):
            processed_hits = [
                f for f in os.listdir(processed)
                if f.startswith("Rheo_steady_") or f.startswith("_output_Rheology_")
            ]
        if rheo_csvs or processed_hits:
            results["Rheo"] = (True, "Rheology data detected")
        else:
            results["Rheo"] = (False, "No Rheology CSVs detected in reference folder")

        # Other modes: presence of modality folder
        for key, folder_name in MODES:
            if key == "Rheo":
                continue
            mod_path = os.path.join(folder, folder_name)
            if os.path.isdir(mod_path):
                results[key] = (True, f"{folder_name} folder detected")
            else:
                results[key] = (False, f"No {folder_name} folder detected in reference folder")

        return results

    def set_mode(self, mode_key: str) -> None:
        canvas_name = f"page{mode_key}Canvas"
        inspector_name = f"page{mode_key}Inspector"
        canvas = self.canvas_pages.get(canvas_name)
        inspector = self.inspector_pages.get(inspector_name)
        if canvas is None or inspector is None:
            self.logger.warning(f"Mode {mode_key} UI missing")
            return

        # Always move back to the Inspector tab when changing modality so the
        # user sees the relevant inspector for the newly selected mode. The
        # Processing tab can still be opened manually if needed.
        if hasattr(self, "inspectorTabs"):
            self.inspectorTabs.setCurrentWidget(self.stackInspector)

        self.stackedCanvas.setCurrentWidget(canvas)
        self.stackInspector.setCurrentWidget(inspector)
        if mode_key == "Rheo":
            self._clear_rheo_plot()
            self._shrink_inspector_width()
        if mode_key == "Write":
            self.write_ui.update_write_header(self.project_path)
            self.write_ui.refresh_write_data(initial=False)
            self._shrink_inspector_width()
        if mode_key == "SAXS":
            self.saxs_ui.refresh_list()
            self._shrink_inspector_width()
        if mode_key == "PLI":
            self.pli_ui.refresh_lists()
            self._shrink_inspector_width()
        if mode_key == "PI":
            self.pi_ui.refresh_list()
            self._shrink_inspector_width()
        self.logger.info(f"Switched mode: {mode_key}")

    def open_pli_extractor(self, video_paths: list[str]) -> None:
        try:
            from _UI_modules.pli_extract_qt import PLIExtractPreview, PLIExtractControls, PLIExtractController
        except Exception as e:
            self.logger.error(f"[PLI] Could not load Qt extractor: {e}")
            return

        if not video_paths:
            self.logger.info("[PLI] Select a video first.")
            return

        preview = PLIExtractPreview(self)
        controls = PLIExtractControls(total_frames=1, parent=self.processingPanel)
        try:
            controller = PLIExtractController(video_paths, preview, controls)
        except Exception as e:
            self._log_exception("Failed to open PLI extractor", e)
            return

        job_id = f"pli-extract-{os.path.basename(controller.preview_video)}"
        title = f"PLI Extract: {os.path.basename(controller.preview_video)}"
        self._add_processing_job(job_id, title, controls)

        # Show preview in the PLI canvas.
        self._set_canvas_content("pagePLICanvas", preview)
        self.stackedCanvas.setCurrentWidget(self.canvas_pages.get("pagePLICanvas"))
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, preview.ensure_fit)
            QTimer.singleShot(50, preview.ensure_fit)
        except Exception:
            pass

        # Auto-remove job when finished.
        def _cleanup():
            self._remove_processing_job(job_id)
            if controller in self._processing_controllers:
                self._processing_controllers.remove(controller)
        controller.on_done = _cleanup
        self._processing_controllers.append(controller)

    def open_pli_rescale(self, image_paths: list[str]) -> None:
        try:
            from _UI_modules.pli_rescale_qt import PLIRescalePreview, PLIRescaleControls, PLIRescaleController
        except Exception as e:
            self.logger.error(f"[PLI] Could not load Qt rescale: {e}")
            return

        if not image_paths:
            self.logger.info("[PLI] Select 1-2 images first.")
            return

        preview = PLIRescalePreview(self)
        controls = PLIRescaleControls(parent=self.processingPanel)
        job_id = f"pli-rescale-{os.path.basename(image_paths[0])}"
        title = f"PLI Rescale: {os.path.basename(image_paths[0])}"
        try:
            controller = PLIRescaleController(image_paths, preview, controls)
        except Exception as e:
            self._log_exception("Failed to open PLI rescale", e)
            return
        # Keep controller alive
        self._pli_controllers[job_id] = controller
        self._add_processing_job(job_id, title, controls)

        self._set_canvas_content("pagePLICanvas", preview)
        self.stackedCanvas.setCurrentWidget(self.canvas_pages.get("pagePLICanvas"))
        try:
            from PyQt6.QtCore import QTimer
            QTimer.singleShot(0, preview.ensure_fit)
            QTimer.singleShot(50, preview.ensure_fit)
        except Exception:
            pass

    def open_pli_analyzer(self, stitched_path: str) -> None:
        try:
            from _UI_modules.pli_analyzer_qt import PLIAnalyzerPreview, PLIAnalyzerControls, PLIAnalyzerController
        except Exception as e:
            self.logger.error(f"[PLI] Could not load Qt analyzer: {e}")
            return

        if not stitched_path:
            self.logger.info("[PLI] Select a stitched panel first.")
            return

        preview = PLIAnalyzerPreview(self)
        controls = PLIAnalyzerControls(parent=self.processingPanel)
        job_id = f"pli-analyzer-{os.path.basename(stitched_path)}"
        title = f"PLI Analyzer: {os.path.basename(stitched_path)}"
        try:
            controller = PLIAnalyzerController(stitched_path, preview, controls)
        except Exception as e:
            self._log_exception("Failed to open PLI analyzer", e)
            return
        # Keep controller alive
        self._pli_controllers[job_id] = controller

        self._add_processing_job(job_id, title, controls)

        self._set_canvas_content("pagePLICanvas", preview)
        self.stackedCanvas.setCurrentWidget(self.canvas_pages.get("pagePLICanvas"))

    def _shrink_inspector_width(self) -> None:
        try:
            from PyQt6.QtCore import QTimer
            def _apply():
                try:
                    page = self.stackInspector.currentWidget() if self.stackInspector else None
                    hint = page.sizeHint().width() if page else 0
                    target = max(260, min(380, hint + 20))
                    self.dockInspector.setMinimumWidth(target)
                    self.dockInspector.setMaximumWidth(target)
                    self.resizeDocks([self.dockInspector], [target], Qt.Orientation.Horizontal)
                    w = self.dockInspector.width()
                    self.logger.info(f"Inspector width set to {w}px (hint {hint}px, target {target}px)")
                except Exception:
                    pass
            QTimer.singleShot(0, _apply)
        except Exception:
            pass

    def _run_rheo(self) -> None:
        try:
            if not self.project_path:
                self.logger.warning("Select a reference folder first.")
                return

            mode = self.rheoModeCombo.currentText().strip() or "triggered"
            steady = float(self.rheoSteadySpin.value())

            from _Routines.Rheo.Read_viscosity_v3 import process_file, get_rheology_folder_from_path

            rh_dir = get_rheology_folder_from_path(self.project_path) or os.path.join(self.project_path, "Rheology")
            if not os.path.isdir(rh_dir):
                self.logger.error("Rheology folder not found in reference folder.")
                self._update_rheo_status("Rheology folder not found")
                return

            csvs = [
                os.path.join(rh_dir, f)
                for f in os.listdir(rh_dir)
                if f.lower().endswith(".csv") and not f.startswith(".")
            ]
            if not csvs:
                self.logger.error("No Rheology CSV files found.")
                self._update_rheo_status("No Rheology CSV files found")
                return

            csvs.sort(key=lambda p: os.path.getmtime(p))
            csv_path = csvs[-1]

            self.logger.info(f"Running Rheology on: {csv_path}")
            out = process_file(csv_path, mode=mode, steady_sec=steady, show_preview=False)
            out_json = out.get("csv_output") or out.get("csv_outputs")
            self.logger.info(f"Rheology complete. Output: {out_json}")
            self._update_rheo_status(f"Rheo complete: {os.path.basename(csv_path)}")
        except Exception as e:
            self._log_exception("Rheology run failed", e)

    def _load_rheo_session(self, reference_folder: str) -> None:
        try:
            sess = self._rheo_load_session(reference_folder)
            mode_default = sess.get("mode")
            steady_default = sess.get("steady_sec")
            if mode_default in {"triggered", "nontriggered", "other"}:
                self.rheoModeCombo.setCurrentText(mode_default)
            if steady_default is not None:
                try:
                    self.rheoSteadySpin.setValue(float(steady_default))
                except Exception:
                    pass
        except Exception:
            pass

    def _save_rheo_session(self, reference_folder: str) -> None:
        try:
            self._rheo_save_session(reference_folder, {
                "mode": self.rheoModeCombo.currentText(),
                "steady_sec": float(self.rheoSteadySpin.value()),
            })
        except Exception:
            pass

    def _rheo_session_path(self, reference_folder: str) -> str:
        processed_root = os.path.join(reference_folder, "_Processed")
        os.makedirs(processed_root, exist_ok=True)
        return os.path.join(processed_root, "_rheo_reader_session.json")

    def _rheo_load_session(self, reference_folder: str) -> dict:
        try:
            with open(self._rheo_session_path(reference_folder), "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _rheo_save_session(self, reference_folder: str, data: dict) -> None:
        try:
            with open(self._rheo_session_path(reference_folder), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    def _list_rheology_files(self, reference_folder: str) -> list[str]:
        rheo_folder = os.path.join(reference_folder, "Rheology")
        if not os.path.isdir(rheo_folder):
            return []
        files = [
            os.path.join(rheo_folder, f)
            for f in os.listdir(rheo_folder)
            if f.lower().endswith(".csv") and not f.startswith(".")
        ]
        files.sort(key=lambda p: os.path.basename(p).lower())
        return files

    def _refresh_rheo_list(self) -> None:
        if not hasattr(self, "rheoSampleList"):
            return
        self.rheoSampleList.clear()
        if not self.project_path or not os.path.isdir(self.project_path):
            self.rheoFolderLabel.setText("Reference folder name: —")
            return

        rheo_folder = os.path.join(self.project_path, "Rheology")
        self.rheoFolderLabel.setText(f"Reference folder name: {os.path.basename(self.project_path)}")
        files = self._list_rheology_files(self.project_path)

        if self._rheo_session_loaded_path != self.project_path:
            self._load_rheo_session(self.project_path)
            self._rheo_session_loaded_path = self.project_path

        for p in files:
            name = os.path.splitext(os.path.basename(p))[0]
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, p)
            self.rheoSampleList.addItem(item)

        self.logger.info(f"Rheology list refreshed: {len(files)} file(s)")

    def _on_rheo_item_selected(self, _item: QListWidgetItem) -> None:
        # Selection only; actual run is via "Run selected"
        return

    def _run_rheo_selected(self) -> None:
        if not hasattr(self, "rheoSampleList"):
            return
        item = self.rheoSampleList.currentItem()
        if item is None:
            self.logger.info("Select a Rheology sample first.")
            return
        self._run_rheo_file(item.data(Qt.ItemDataRole.UserRole))

    def _process_all_rheo(self) -> None:
        if not self.project_path:
            self.logger.warning("Select a reference folder first.")
            return
        files = self._list_rheology_files(self.project_path)
        if not files:
            self.logger.info("No Rheology CSV files found.")
            return
        for p in files:
            self._run_rheo_file(p)

    def _run_rheo_file(self, csv_path: str | None) -> None:
        try:
            if not csv_path or not os.path.isfile(csv_path):
                self.logger.warning("Invalid Rheology CSV selection.")
                return

            mode = self.rheoModeCombo.currentText().strip() or "triggered"
            steady = float(self.rheoSteadySpin.value())

            from _Routines.Rheo.Read_viscosity_v3 import process_file

            self.logger.info(f"Running Rheology on: {csv_path}")
            self._save_rheo_session(self.project_path or os.path.dirname(csv_path))
            self._last_rheo_csv = csv_path
            preview: dict = {}
            crop_enabled = self.rheoCropCheck.isChecked()
            # First pass (no crop) to derive crop bounds from data if needed.
            out = process_file(
                csv_path,
                mode=mode,
                steady_sec=steady,
                show_preview=False,
                crop_range=None,
                preview_payload=preview,
            )
            self._update_crop_bounds(preview)
            if crop_enabled:
                crop_range = self._compute_crop_range(preview)
                if crop_range is not None:
                    preview_cropped: dict = {}
                    out = process_file(
                        csv_path,
                        mode=mode,
                        steady_sec=steady,
                        show_preview=False,
                        crop_range=crop_range,
                        preview_payload=preview_cropped,
                    )
                    preview = preview_cropped
            out_json = out.get("csv_output") or out.get("csv_outputs")
            self.logger.info(f"Rheology complete. Output: {out_json}")
            self._update_rheo_status(f"Rheo complete: {os.path.basename(csv_path)}")
            self._last_rheo_preview = preview
            self._update_rheo_plot(preview)
        except Exception as e:
            self._log_exception("Rheology run failed", e)

    def _rerun_last_rheo_if_any(self) -> None:
        if self._rheo_rerun_guard:
            return
        if not self._last_rheo_csv:
            return
        if not self.rheoCropCheck.isChecked():
            # Rerun to remove crop
            self._rheo_rerun_guard = True
            try:
                self._run_rheo_file(self._last_rheo_csv)
            finally:
                self._rheo_rerun_guard = False
            return
        crop_min = float(self.rheoCropMin.value())
        crop_max = float(self.rheoCropMax.value())
        if crop_max <= crop_min:
            return
        self._rheo_rerun_guard = True
        try:
            self._run_rheo_file(self._last_rheo_csv)
        finally:
            self._rheo_rerun_guard = False

    def _compute_crop_range(self, preview: dict) -> tuple[float, float] | None:
        # Build shear rate list for index-based cropping
        shear_vals = []
        rates = preview.get("steady_rates_all") or preview.get("steady_rates")
        if rates:
            for v in rates:
                try:
                    if v is not None and math.isfinite(v) and v > 0:
                        shear_vals.append(float(v))
                except Exception:
                    continue
        if not shear_vals:
            rate_series = preview.get("rate_series") or []
            for v in rate_series:
                try:
                    if v is not None and math.isfinite(v) and v > 0:
                        shear_vals.append(float(v))
                except Exception:
                    continue
        if not shear_vals:
            return None
        shear_vals = sorted(shear_vals)
        max_idx = len(shear_vals)
        self._update_crop_bounds(preview)

        min_idx = int(self.rheoCropMin.value())
        max_idx_sel = int(self.rheoCropMax.value())
        if max_idx_sel < min_idx:
            return None
        # Map indices (1-based) to shear-rate values
        start_r = shear_vals[min_idx - 1]
        end_r = shear_vals[max_idx_sel - 1]
        return (start_r, end_r)

    def _update_crop_bounds(self, preview: dict) -> None:
        shear_vals = []
        rates = preview.get("steady_rates_all") or preview.get("steady_rates")
        if rates:
            for v in rates:
                try:
                    if v is not None and math.isfinite(v) and v > 0:
                        shear_vals.append(float(v))
                except Exception:
                    continue
        if not shear_vals:
            rate_series = preview.get("rate_series") or []
            for v in rate_series:
                try:
                    if v is not None and math.isfinite(v) and v > 0:
                        shear_vals.append(float(v))
                except Exception:
                    continue
        if not shear_vals:
            return
        max_idx = len(shear_vals)
        self.rheoCropMin.setRange(1, max_idx)
        self.rheoCropMax.setRange(1, max_idx)
        # Default to full range if both at 1 or out of range
        if self.rheoCropMax.value() < 1 or self.rheoCropMax.value() > max_idx:
            self.rheoCropMax.setValue(max_idx)
        if self.rheoCropMin.value() < 1 or self.rheoCropMin.value() > max_idx:
            self.rheoCropMin.setValue(1)
        if self.rheoCropMin.value() == 1 and self.rheoCropMax.value() == 1 and max_idx > 1:
            self.rheoCropMax.setValue(max_idx)

    def _update_rheo_plot(self, preview: dict) -> None:
        if self.rheoCanvas is None:
            return
        fig = self.rheoCanvas.figure
        fig.clear()

        t_sec = preview.get("t_sec")
        rate_sec = preview.get("rate_series")
        visc_sec = preview.get("visc_series")
        stress_sec = preview.get("stress_series")
        T_beg = preview.get("T_beg", [])
        S_end = preview.get("S_end", [])
        steady_rates = preview.get("steady_rates")
        steady_viscs = preview.get("steady_viscs")
        steady_rates_all = preview.get("steady_rates_all") or steady_rates
        steady_viscs_all = preview.get("steady_viscs_all") or steady_viscs
        crop_range = preview.get("crop_range")

        ax_time = fig.add_subplot(1, 2, 1)
        ax_rate = ax_time.twinx()
        ax_steady = fig.add_subplot(1, 2, 2)
        # Keep both plot boxes the same size and centered within the canvas.
        try:
            ax_time.set_box_aspect(1)
            ax_steady.set_box_aspect(1)
        except Exception:
            pass
        # Axis label size ~30% larger than tick labels
        tick_size = 10
        label_size = int(tick_size * 1.3)
        for ax in (ax_time, ax_rate, ax_steady):
            ax.tick_params(axis="both", labelsize=tick_size)
            ax.xaxis.label.set_size(label_size)
            ax.yaxis.label.set_size(label_size)

        # Left plot: time traces with dual y axes
        if t_sec is None:
            t_plot = list(range(len(rate_sec or visc_sec or stress_sec or [])))
            ax_time.set_xlabel("sample index")
        else:
            t_plot = t_sec
            ax_time.set_xlabel("time (s)")

        def _clean_pos(series, positive_only: bool = True):
            if series is None:
                return None
            out = []
            for v in series:
                try:
                    if v is None or not math.isfinite(v):
                        out.append(math.nan)
                    elif positive_only and v <= 0:
                        out.append(math.nan)
                    else:
                        out.append(float(v))
                except Exception:
                    out.append(math.nan)
            return out

        any_plotted = False
        rate_clean = _clean_pos(rate_sec)
        if rate_clean is not None and any(math.isfinite(v) for v in rate_clean):
            ax_time.plot(t_plot, rate_clean, label="Shear rate [1/s]", linewidth=1.0)
            ax_time.set_ylabel("Shear rate [1/s]")
            ax_time.set_yscale("log")
            any_plotted = True

        visc_clean = _clean_pos(visc_sec)
        if visc_clean is not None and any(math.isfinite(v) for v in visc_clean):
            ax_rate.plot(t_plot, visc_clean, label="Viscosity [Pa·s]", linewidth=1.0, alpha=0.9, color="tab:orange")
            ax_rate.set_ylabel("Viscosity [Pa·s]")
            ax_rate.set_yscale("log")
            any_plotted = True
        else:
            stress_clean = _clean_pos(stress_sec, positive_only=False)
            if stress_clean is not None and any(math.isfinite(v) for v in stress_clean):
                ax_rate.plot(t_plot, stress_clean, label="Shear stress [Pa]", linewidth=1.0, linestyle="--", alpha=0.9, color="tab:green")
                ax_rate.set_ylabel("Shear stress [Pa]")
                finite_vals = [v for v in stress_clean if math.isfinite(v) and abs(v) > 0]
                if finite_vals and (max(finite_vals) / max(min(abs(v) for v in finite_vals), 1e-12) > 50):
                    ax_rate.set_yscale("log")
                any_plotted = True

        if not any_plotted:
            ax_time.text(0.5, 0.5, "No plottable data detected", transform=ax_time.transAxes, ha="center", va="center")

        ax_time.grid(True, which="both", alpha=0.2)
        try:
            ax_time.legend(loc="upper left")
        except Exception:
            pass
        try:
            ax_rate.legend(loc="upper right")
        except Exception:
            pass

        for x in T_beg:
            if x is not None:
                ax_time.axvline(x, color="cyan", linewidth=1.0)
        for x in S_end:
            if x is not None:
                ax_time.axvline(x, color="red", linewidth=1.0)

        ax_time.set_title("Time traces — cyan=T_begin, red=S_end")

        # Right plot: steady scatter
        scatter_points = []
        if steady_rates and steady_viscs:
            for r, v in zip(steady_rates, steady_viscs):
                try:
                    if r is not None and v is not None and math.isfinite(r) and math.isfinite(v) and r > 0 and v > 0:
                        scatter_points.append((r, v))
                except Exception:
                    continue
        if scatter_points:
            xs, ys = zip(*scatter_points)
            ax_steady.scatter(xs, ys, color="tab:blue")
            ax_steady.set_xscale("log")
            ax_steady.set_yscale("log")
            ax_steady.set_xlabel("Shear rate [1/s]")
            ax_steady.set_ylabel("Steady viscosity [Pa·s]")
            ax_steady.set_title("Steady-state viscosity vs shear rate")
            ax_steady.grid(True, which="both", alpha=0.2)
            # Crop markers (vertical lines)
            try:
                if crop_range is not None:
                    crop_min, crop_max = crop_range
                    if crop_max > crop_min and crop_min > 0 and crop_max > 0:
                        ax_steady.axvline(crop_min, color="#5E8C31", linestyle="--", linewidth=1.2)
                        ax_steady.axvline(crop_max, color="#5E8C31", linestyle="--", linewidth=1.2)
            except Exception:
                pass
        else:
            ax_steady.text(0.5, 0.5, "No steady averages computed", transform=ax_steady.transAxes, ha="center", va="center")

        # Preserve original steady plot ranges even when crop hides points
        try:
            if steady_rates_all and steady_viscs_all:
                xs_all = []
                ys_all = []
                for r, v in zip(steady_rates_all, steady_viscs_all):
                    if r is None or v is None:
                        continue
                    if not (math.isfinite(r) and math.isfinite(v)):
                        continue
                    if r <= 0 or v <= 0:
                        continue
                    xs_all.append(r)
                    ys_all.append(v)
                if xs_all and ys_all:
                    xmin, xmax = min(xs_all), max(xs_all)
                    ymin, ymax = min(ys_all), max(ys_all)
                    # Add 10% padding to x/y ranges (log-space if positive)
                    if xmin > 0 and xmax > 0:
                        log_min = math.log10(xmin)
                        log_max = math.log10(xmax)
                        pad = 0.1 * max(1e-9, log_max - log_min)
                        ax_steady.set_xlim(10 ** (log_min - pad), 10 ** (log_max + pad))
                    else:
                        pad = 0.1 * max(1e-9, xmax - xmin)
                        ax_steady.set_xlim(xmin - pad, xmax + pad)

                    if ymin > 0 and ymax > 0:
                        log_min = math.log10(ymin)
                        log_max = math.log10(ymax)
                        pad = 0.1 * max(1e-9, log_max - log_min)
                        ax_steady.set_ylim(10 ** (log_min - pad), 10 ** (log_max + pad))
                    else:
                        pad = 0.1 * max(1e-9, ymax - ymin)
                        ax_steady.set_ylim(ymin - pad, ymax + pad)
        except Exception:
            pass

        fig.subplots_adjust(left=0.07, right=0.98, top=0.92, bottom=0.08, wspace=0.35)
        self.rheoCanvas.draw_idle()
        if hasattr(self, "rheoStatusLabel"):
            self.rheoStatusLabel.setVisible(False)

    def _clear_rheo_plot(self) -> None:
        if self.rheoCanvas is None:
            return
        fig = self.rheoCanvas.figure
        fig.clear()
        ax = fig.add_subplot(111)
        try:
            bg = self.palette().window().color().name()
            fig.patch.set_facecolor(bg)
            ax.set_facecolor(bg)
        except Exception:
            pass
        ax.axis("off")
        fig.tight_layout()
        self.rheoCanvas.draw_idle()
        if hasattr(self, "rheoStatusLabel"):
            self.rheoStatusLabel.setText("")
            self.rheoStatusLabel.setVisible(False)

    def _update_rheo_status(self, text: str) -> None:
        if hasattr(self, "rheoStatusLabel"):
            self.rheoStatusLabel.setText(text)

    def _on_rheo_mode_changed(self, mode: str) -> None:
        # Triggered mode uses internal segmentation; steady window is not used.
        if not hasattr(self, "rheoSteadySpin"):
            return
        use_steady = mode in {"nontriggered", "other"}
        self.rheoSteadySpin.setEnabled(use_steady)

    def _log_not_implemented(self, name: str) -> None:
        self.logger.info(f"Not implemented yet: {name}")

    def _log_exception(self, context: str, exc: Exception) -> None:
        self.logger.error(f"{context}: {exc}")
        self.logger.error(traceback.format_exc())
        QMessageBox.warning(self, "Error", f"{context}\n{exc}")


def main() -> None:
    faulthandler.enable()
    # HiDPI: prevent scale-factor rounding that can make SVG icons look pixelated on Retina
    QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    # Qt 6 / PyQt6: AA_UseHighDpiPixmaps may be unavailable (HiDPI pixmaps are typically default).
    if hasattr(Qt.ApplicationAttribute, "AA_UseHighDpiPixmaps"):
        QApplication.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)

    # macOS default is a global (top-of-screen) menu bar. Disable it so the menu shows in-window.
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)
    try:
        app = QApplication(sys.argv)

        window = MainWindow()
        # Window sizing: default to laptop resolution, but never exceed screen.
        default_w, default_h = 1728, 1117
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            width = min(default_w, avail.width())
            height = min(default_h, avail.height())
            window.resize(width, height)
        window.show()

        sys.exit(app.exec())
    except Exception:
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
