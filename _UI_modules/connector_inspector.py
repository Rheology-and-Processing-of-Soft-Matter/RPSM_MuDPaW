from __future__ import annotations

import json
import os
import re
import csv
import shutil
from pathlib import Path
from typing import List

import numpy as np

from PyQt6.QtCore import Qt, QThread, QObject, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLineEdit,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QSizePolicy,
    QDialog,
    QPlainTextEdit,
    QMessageBox,
    QTextEdit,
    QCheckBox,
    QTableWidget,
    QTableWidgetItem,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from _Connector.formax_preview import detect_scan_ids, group_continuous_ranges, plot_azint_overview
from _Connector import maxiv_connect
from _Routines.Connector.azint_processing import AzintProcessConfig, process_scans, compute_background
from UI.formax_config_dialog import ForMAXConfigDialog, DEFAULT_FORMAX
from UI.cosaxs_config_dialog import CoSAXSConfigDialog, DEFAULT_COSAXS

BASE_DIR = Path(__file__).resolve().parents[1]
STATE_PATH = BASE_DIR / "_Miscell" / "_connector_state.json"
CONFIG_DIR = BASE_DIR / "_Miscell" / "ConnectorConfigs"


def _load_connector_state() -> dict:
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _parse_scan_input(text: str) -> List[int]:
    text = (text or "").strip()
    if not text:
        return []
    ids: List[int] = []
    for part in re.split(r"[,\s]+", text):
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                lo = int(a)
                hi = int(b)
            except ValueError:
                continue
            ids.extend(list(range(min(lo, hi), max(lo, hi) + 1)))
        else:
            try:
                ids.append(int(part))
            except ValueError:
                continue
    return sorted(set(ids))


def _load_scan_arrays(local_azint: Path, scan_id: int):
    import h5py
    import numpy as np

    # Try multiple filename patterns (eiger, pilatus, generic) for integrated files.
    candidates = [
        local_azint / f"scan-{scan_id:04d}_eiger_integrated.h5",
        local_azint / f"scan-{scan_id:04d}_pilatus_integrated.h5",
    ]
    # Generic fallback: any scan-####_*integrated.h5
    candidates.extend(sorted(local_azint.glob(f"scan-{scan_id:04d}_*integrated.h5")))
    filepath = None
    for cand in candidates:
        if cand.exists():
            filepath = cand
            break
    if filepath is None:
        existing = [p.name for p in sorted(local_azint.glob(f"scan-{scan_id:04d}_*"))]
        raise FileNotFoundError(
            f"Integrated scan file not found for scan {scan_id} in {local_azint}; "
            f"found: {', '.join(existing) or 'none'}"
        )

    def _load_primary(f):
        azi = np.array(f["entry/azimuthal/azi"])
        A0 = np.array(f["entry/azimuthal/A0"])
        A1 = np.array(f["entry/azimuthal/A1"])
        q_plot = np.array(f["entry/radial/q"])
        R = np.array(f["entry/radial/R"])
        cake = np.array(f["entry/azimuthal/cake"]) if "entry/azimuthal/cake" in f else None
        norm = np.array(f["entry/azimuthal/norm"]) if "entry/azimuthal/norm" in f else None
        qmin0 = float(f["entry/azimuthal/A0"].attrs["qmin"])
        qmax0 = float(f["entry/azimuthal/A0"].attrs["qmax"])
        qmin1 = float(f["entry/azimuthal/A1"].attrs["qmin"])
        qmax1 = float(f["entry/azimuthal/A1"].attrs["qmax"])
        return azi, q_plot, cake, R, norm, A0, A1, qmin0, qmax0, qmin1, qmax1

    def _load_azint2d(f):
        azi = np.array(f["entry/azint2d/data/azimuthal_axis"])
        q_plot = np.array(f["entry/azint2d/data/radial_axis"])
        cake = np.array(f["entry/azint2d/data/I"])
        R = np.array(f["entry/azint1d/data/I"])
        norm = np.array(f["entry/azint2d/data/norm"]) if "entry/azint2d/data/norm" in f else None
        if cake.ndim == 2:
            cake = cake[np.newaxis, ...]
        if R.ndim == 1:
            R = R[np.newaxis, ...]
        mid = max(1, q_plot.shape[0] // 2)
        A0 = cake[:, :, :mid].mean(axis=2)
        A1 = cake[:, :, mid:].mean(axis=2)
        qmin0 = float(q_plot[0])
        qmax0 = float(q_plot[mid - 1])
        qmin1 = float(q_plot[mid])
        qmax1 = float(q_plot[-1])
        return azi, q_plot, cake, R, norm, A0, A1, qmin0, qmax0, qmin1, qmax1

    def _load_data2d(f):
        azi = np.array(f["entry/data2d/azi"])
        q_plot = np.array(f["entry/data2d/q"])
        cake = np.array(f["entry/data2d/cake"])
        R = np.array(f["entry/data1d/I"])
        norm = np.array(f["entry/data2d/norm"]) if "entry/data2d/norm" in f else None
        if cake.ndim == 2:
            cake = cake[np.newaxis, ...]
        if R.ndim == 1:
            R = R[np.newaxis, ...]
        mid = max(1, q_plot.shape[0] // 2)
        A0 = cake[:, :, :mid].mean(axis=2)
        A1 = cake[:, :, mid:].mean(axis=2)
        qmin0 = float(q_plot[0])
        qmax0 = float(q_plot[mid - 1])
        qmin1 = float(q_plot[mid])
        qmax1 = float(q_plot[-1])
        return azi, q_plot, cake, R, norm, A0, A1, qmin0, qmax0, qmin1, qmax1

    errors = []
    with h5py.File(filepath, "r") as f:
        norm = None
        for loader in (_load_primary, _load_azint2d, _load_data2d):
            try:
                (
                    azi,
                    q_plot,
                    cake,
                    R,
                    norm,
                    A0,
                    A1,
                    qmin0,
                    qmax0,
                    qmin1,
                    qmax1,
                ) = loader(f)
                break
            except Exception as exc:  # noqa: BLE001
                errors.append(str(exc))
                continue
        else:
            joined = " | ".join(errors)
            raise RuntimeError(
                f"Failed to read integrated file {filepath}. Tried layouts: primary/azint2d/data2d. Errors: {joined}"
            )

    if cake is None:
        raise RuntimeError("No azimuthal cake data found in the integrated file.")
    if norm is None:
        norm = np.ones((cake.shape[0], cake.shape[2]), dtype=cake.dtype)

    qmin_r = float(q_plot.min())
    qmax_r = float(q_plot.max())

    return {
        "A0": A0,
        "A1": A1,
        "R": R,
        "azi": azi,
        "q_plot": q_plot,
        "norm": norm,
        "qmin0": qmin0,
        "qmax0": qmax0,
        "qmin1": qmin1,
        "qmax1": qmax1,
        "qmin_r": qmin_r,
        "qmax_r": qmax_r,
        "cake": cake,
    }


class ConnectorInspector:
    def __init__(self, main) -> None:
        self.main = main
        self.canvas: FigureCanvas | None = None
        # Use a module-level loader to avoid rare initialization issues if the
        # instance method is not yet bound during early imports.
        self._state = _load_connector_state()
        self.embed_rows = list(self._state.get("embed_rows", []))
        self._config_formax = self._state.get("config_formax", DEFAULT_FORMAX.copy())
        self._config_cosaxs = self._state.get("config_cosaxs", DEFAULT_COSAXS.copy())
        self._last_bg_scan = self._state.get("bg_scan", "")
        self._last_bg_frames = self._state.get("bg_frames", "")
        self._last_scan_frames = self._state.get("scan_frames", "")
        self._bg_subtract = bool(self._state.get("bg_subtract", False))
        try:
            self._bg_coeff = float(self._state.get("bg_coeff", 1.0))
        except Exception:
            self._bg_coeff = 1.0
        self._last_q_range = self._state.get("q_range", "")
        self._last_scans_text = self._state.get("scans_text", "")
        try:
            self._last_bins = int(self._state.get("bin_frames", 0))
        except Exception:
            self._last_bins = 0
        self._pending_preview_ids: list[int] | None = None
        self._fetch_thread: QThread | None = None
        self._fetch_worker: QObject | None = None
        self._remote_azint_override: Path | None = None
        last_out = self._state.get("last_output_dir", "")
        self._last_output_dir: Path | None = Path(last_out) if last_out else None

    def _default_browse_dir(self) -> str:
        """
        Starting directory for QFileDialog pickers.
        Priority:
        1) Current reference folder (`main.project_path`) if available.
        2) Last reference saved in _Miscell/last_folder.txt.
        3) User home.
        """
        ref = getattr(self.main, "project_path", None)
        if ref:
            try:
                p = Path(ref)
                if p.exists():
                    return str(p)
            except Exception:
                pass
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

    def shutdown(self) -> None:
        """Gracefully stop background threads to avoid QThread destruction crashes on exit."""
        try:
            if self._fetch_thread is not None:
                try:
                    if self._fetch_worker is not None:
                        try:
                            self._fetch_worker.finished.disconnect()
                        except Exception:
                            pass
                    self._fetch_thread.requestInterruption()
                    self._fetch_thread.quit()
                    self._fetch_thread.wait(200)
                except Exception:
                    pass
            self._fetch_thread = None
            self._fetch_worker = None
        except Exception:
            pass

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
        page = self.main.canvas_pages.get("pageConnectorCanvas")
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

    def build_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(8)

        # Connection group
        grp_conn = QGroupBox("Connection", page)
        conn_layout = QGridLayout(grp_conn)
        conn_layout.setColumnStretch(1, 1)
        self.editHost = QLineEdit(self._state.get("hostname", "offline-fe1.maxiv.lu.se"), grp_conn)
        self.editUser = QLineEdit(self._state.get("username", ""), grp_conn)
        self.comboBeamline = QComboBox(grp_conn)
        self.comboBeamline.addItems(["ForMAX", "CoSAXS"])
        self.comboBeamline.setCurrentText(self._state.get("beamline", "ForMAX"))
        self.editProposal = QLineEdit(self._state.get("proposal", ""), grp_conn)
        self.editVisit = QLineEdit(self._state.get("visit", ""), grp_conn)
        conn_layout.addWidget(QLabel("Hostname", grp_conn), 0, 0)
        conn_layout.addWidget(self.editHost, 0, 1)
        conn_layout.addWidget(QLabel("Username", grp_conn), 1, 0)
        conn_layout.addWidget(self.editUser, 1, 1)
        conn_layout.addWidget(QLabel("Beamline", grp_conn), 2, 0)
        conn_layout.addWidget(self.comboBeamline, 2, 1)
        conn_layout.addWidget(QLabel("Proposal", grp_conn), 3, 0)
        conn_layout.addWidget(self.editProposal, 3, 1)
        conn_layout.addWidget(QLabel("Visit", grp_conn), 4, 0)
        conn_layout.addWidget(self.editVisit, 4, 1)

        btn_row = QHBoxLayout()
        self.btnConnect = QPushButton("Connect", grp_conn)
        self.btnListScans = QPushButton("List scans", grp_conn)
        self.btnForMAX = QPushButton("ForMAX config", grp_conn)
        self.btnCoSAXS = QPushButton("CoSAXS config", grp_conn)
        btn_row.addWidget(self.btnConnect)
        btn_row.addWidget(self.btnListScans)
        btn_row.addWidget(self.btnForMAX)
        btn_row.addWidget(self.btnCoSAXS)
        conn_layout.addLayout(btn_row, 5, 0, 1, 2)

        self.lblVpn = QLabel("● VPN disconnected", grp_conn)
        self.btnVpn = QPushButton("Reconnect VPN", grp_conn)
        vpn_row = QHBoxLayout()
        vpn_row.addWidget(self.lblVpn)
        vpn_row.addStretch(1)
        vpn_row.addWidget(self.btnVpn)
        conn_layout.addLayout(vpn_row, 6, 0, 1, 2)
        layout.addWidget(grp_conn)

        # Scan overview
        grp_over = QGroupBox("Scan overview", page)
        over_layout = QVBoxLayout(grp_over)
        self.lblRemoteAzint = QLabel("Remote azint: —", grp_over)
        self.lblScanRanges = QLabel("Detected scan ranges: —", grp_over)
        over_layout.addWidget(self.lblRemoteAzint)
        over_layout.addWidget(self.lblScanRanges)
        layout.addWidget(grp_over)

        # Preview
        grp_prev = QGroupBox("Preview", page)
        prev_layout = QGridLayout(grp_prev)
        prev_layout.setColumnStretch(1, 1)
        self.editScans = QLineEdit(grp_prev)
        if self._last_scans_text:
            self.editScans.setText(self._last_scans_text)
        self.spinAvg = QSpinBox(grp_prev)
        self.spinAvg.setRange(1, 999)
        self.spinAvg.setValue(4)
        self.editBgScan = QLineEdit(grp_prev)
        self.editBgScan.setPlaceholderText("e.g. 570")
        self.editBgScan.setText(str(self._last_bg_scan))
        self.editBgFrames = QLineEdit(grp_prev)
        self.editBgFrames.setPlaceholderText("start-end (optional)")
        self.editBgFrames.setText(str(self._last_bg_frames))
        self.editBgFrames.setVisible(False)
        self.editBgFrames.setEnabled(False)
        self.editScanFrames = QLineEdit(grp_prev)
        self.editScanFrames.setPlaceholderText("start-end (optional)")
        self.editScanFrames.setText(str(self._last_scan_frames))
        self.editScanFrames.setVisible(False)
        self.editScanFrames.setEnabled(False)
        self.editQRange = QLineEdit(grp_prev)
        self.editQRange.setPlaceholderText("qmin-qmax (optional)")
        self.editQRange.setText(str(self._last_q_range))
        self.spinBinFrames = QSpinBox(grp_prev)
        self.spinBinFrames.setRange(1, 999)
        self.spinBinFrames.setValue(max(1, self._last_bins or 1))
        self.spinBinFrames.setToolTip("Split q-range into this many slices for plotting (>=1)")
        self.chkSubtractBg = QCheckBox("Subtract background", grp_prev)
        self.chkSubtractBg.setChecked(self._bg_subtract)
        self.spinBgCoeff = QDoubleSpinBox(grp_prev)
        self.spinBgCoeff.setRange(0.0, 1000.0)
        self.spinBgCoeff.setSingleStep(0.1)
        self.spinBgCoeff.setDecimals(3)
        self.spinBgCoeff.setValue(self._bg_coeff)
        self.spinBgCoeff.setToolTip("Multiplier applied to background before subtraction")
        self.btnPreview = QPushButton("Preview", grp_prev)
        self.btnProcessPreview = QPushButton("Process preview", grp_prev)
        self.btnProcessList = QPushButton("Process list", grp_prev)
        self.btnEmbed = QPushButton("Embed", grp_prev)
        self.btnViewList = QPushButton("View list", grp_prev)
        self.btnLoadList = QPushButton("Load list", grp_prev)
        self.editRawScan = QLineEdit(grp_prev)
        self.btnPreviewRaw = QPushButton("Preview raw pattern", grp_prev)

        prev_layout.addWidget(QLabel("Scans:", grp_prev), 0, 0)
        prev_layout.addWidget(self.editScans, 0, 1)
        prev_layout.addWidget(QLabel("Avg frames:", grp_prev), 1, 0)
        prev_layout.addWidget(self.spinAvg, 1, 1)
        prev_layout.addWidget(QLabel("Background scan:", grp_prev), 2, 0)
        prev_layout.addWidget(self.editBgScan, 2, 1)
        # Keep layout rows consistent; hide BG/scan frames controls
        self.lblBgFrames = QLabel("BG frames:", grp_prev)
        self.lblBgFrames.setVisible(False)
        self.lblBgFrames.setEnabled(False)
        self.lblScanFrames = QLabel("Scan frames:", grp_prev)
        self.lblScanFrames.setVisible(False)
        self.lblScanFrames.setEnabled(False)

        prev_layout.addWidget(self.lblBgFrames, 3, 0)
        prev_layout.addWidget(self.editBgFrames, 3, 1)
        prev_layout.addWidget(self.lblScanFrames, 4, 0)
        prev_layout.addWidget(self.editScanFrames, 4, 1)
        prev_layout.addWidget(QLabel("q range:", grp_prev), 5, 0)
        prev_layout.addWidget(self.editQRange, 5, 1)
        prev_layout.addWidget(QLabel("Bins (q slices):", grp_prev), 6, 0)
        prev_layout.addWidget(self.spinBinFrames, 6, 1)
        prev_layout.addWidget(self.chkSubtractBg, 7, 0)
        prev_layout.addWidget(self.spinBgCoeff, 7, 1)
        prev_layout.addWidget(self.btnPreview, 8, 0, 1, 2)

        # Stack Process preview under Preview; keep list controls in right column
        prev_layout.addWidget(self.btnProcessPreview, 9, 0, 1, 2)
        prev_layout.addWidget(self.btnProcessList, 10, 1)
        prev_layout.addWidget(self.btnEmbed, 10, 0)
        prev_layout.addWidget(self.btnViewList, 11, 1)
        prev_layout.addWidget(self.btnLoadList, 12, 1)
        prev_layout.addWidget(QLabel("Raw scan:", grp_prev), 12, 0)
        prev_layout.addWidget(self.editRawScan, 12, 1)
        prev_layout.addWidget(self.btnPreviewRaw, 13, 0, 1, 2)

        layout.addWidget(grp_prev)

        # Local azint folder (for preview)
        self.localAzintEdit = QLineEdit(self._state.get("local_azint", ""), page)
        self.btnBrowseAzint = QPushButton("Browse azint…", page)
        self.btnClearCache = QPushButton("Clear cache", page)
        local_row = QHBoxLayout()
        local_row.addWidget(QLabel("Local azint:", page))
        local_row.addWidget(self.localAzintEdit, 1)
        local_row.addWidget(self.btnBrowseAzint)
        local_row.addWidget(self.btnClearCache)
        layout.addLayout(local_row)
        layout.addStretch(1)

        # Wiring
        self.btnBrowseAzint.clicked.connect(self._browse_azint)
        self.btnClearCache.clicked.connect(self._clear_cache)
        self.btnConnect.clicked.connect(self._connect)
        self.btnListScans.clicked.connect(self._list_scans)
        self.btnForMAX.clicked.connect(self._apply_formax)
        self.btnCoSAXS.clicked.connect(self._apply_cosaxs)
        self.btnVpn.clicked.connect(self._vpn_reconnect)
        self.btnPreview.clicked.connect(self._preview)
        self.btnProcessPreview.clicked.connect(self._process_preview)
        self.btnProcessList.clicked.connect(self._process_list)
        self.btnEmbed.clicked.connect(self._embed_current)
        self.btnViewList.clicked.connect(self._view_embed_list)
        self.btnLoadList.clicked.connect(self._load_embed_list)
        self.btnPreviewRaw.clicked.connect(self._preview_raw)

    def _connect(self) -> None:
        self._save_state()
        host = self.editHost.text().strip()
        user = self.editUser.text().strip()
        beamline = self.comboBeamline.currentText().strip() or "ForMAX"
        try:
            proposal = int(self.editProposal.text().strip())
            visit = int(self.editVisit.text().strip())
        except ValueError:
            self._log("Enter numeric Proposal and Visit to connect.")
            return
        if not host or not user:
            self._log("Enter hostname and username to connect.")
            return

        local_root = None
        local_azint = Path(self.localAzintEdit.text().strip())
        if local_azint.exists():
            # If the user picked .../<beamline>/<proposal>/<visit>/azint, infer local root.
            try:
                local_root = local_azint.parents[3]
            except Exception:
                local_root = None

        self._log("Connecting to MAX IV (SSH)...")
        try:
            remote_azint, scan_ids = maxiv_connect.list_remote_scans(
                host,
                user,
                beamline,
                proposal,
                visit,
                log_cb=lambda m: self._log(m.strip()),
                local_root=local_root,
            )
            self.lblRemoteAzint.setText(f"Remote azint: {remote_azint}")
            ranges = group_continuous_ranges(scan_ids)
            if ranges:
                ranges_txt = ", ".join([f"{a}-{b}" if a != b else f"{a}" for a, b in ranges])
            else:
                ranges_txt = "—"
            self.lblScanRanges.setText(f"Detected scan ranges: {ranges_txt}")
            self._log(f"Connected. Scans detected: {len(scan_ids)}")
        except Exception as exc:
            self._log(f"Connect failed: {exc}")
            # Helper: list proposal folder to guide the user when visit path is missing
            try:
                proposal_root = Path("/data/visitors") / beamline.lower() / str(proposal)
                listing = maxiv_connect.run_ssh(
                    host,
                    user,
                    f"ls -1 {proposal_root}",
                    log_cb=None,
                )
                if listing.strip():
                    self._log(
                        f"Contents of proposal folder {proposal_root}:\n{listing.strip()}"
                    )
                else:
                    self._log(f"Proposal folder {proposal_root} is empty or inaccessible.")
            except Exception as exc2:  # noqa: BLE001
                self._log(f"Could not list proposal folder: {exc2}")

    def _apply_formax(self) -> None:
        self.comboBeamline.setCurrentText("ForMAX")
        initial = self._load_reference_config("formax") or self._config_formax
        dlg = ForMAXConfigDialog(self.main, initial=initial)
        if dlg.exec():
            if dlg.result_config is not None:
                self._config_formax = dlg.result_config
                self._save_reference_config("formax", dlg.result_config)
                self._save_state()
                self._log("ForMAX config applied.")

    def _apply_cosaxs(self) -> None:
        self.comboBeamline.setCurrentText("CoSAXS")
        initial = self._load_reference_config("cosaxs") or self._config_cosaxs
        dlg = CoSAXSConfigDialog(self.main, initial=initial)
        if dlg.exec():
            if dlg.result_config is not None:
                self._config_cosaxs = dlg.result_config
                self._save_reference_config("cosaxs", dlg.result_config)
                self._save_state()
                self._log("CoSAXS config applied.")

    def _vpn_reconnect(self) -> None:
        self._log("Reconnect VPN not implemented yet.")

    def _browse_azint(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self.main,
            "Select azint folder",
            self._default_browse_dir(),
        )
        if folder:
            self.localAzintEdit.setText(folder)
            self._save_state()

    def _list_scans(self) -> None:
        local_azint = self._require_local_azint()
        if local_azint is None:
            return
        try:
            ids = detect_scan_ids(local_azint)
            ranges = group_continuous_ranges(ids)
            if ranges:
                ranges_txt = ", ".join([f"{a}-{b}" if a != b else f"{a}" for a, b in ranges])
            else:
                ranges_txt = "—"
            self.lblScanRanges.setText(f"Detected scan ranges: {ranges_txt}")
        except Exception as e:
            self._log(f"Failed to list scans: {e}")

    def _preview(self) -> None:
        scan_ids = _parse_scan_input(self.editScans.text())
        if not scan_ids:
            self._log("Enter scan IDs to preview (e.g., 569-597).")
            return
        local_azint = self._require_local_azint(allow_missing=True)
        if local_azint is None:
            return
        if not self._has_scan_files(local_azint, scan_ids):
            self._pending_preview_ids = scan_ids
            self._start_fetch_thread(scan_ids)
            return
        self._render_preview(local_azint, scan_ids)

    def _render_preview(self, local_azint: Path, scan_ids: list[int]) -> None:
        local_azint = Path(local_azint)
        if not local_azint.exists():
            self._log(f"Local azint not found: {local_azint}")
            alt = QFileDialog.getExistingDirectory(
                self.main,
                "Select azint folder",
                self._default_browse_dir(),
            )
            if not alt:
                return
            local_azint = Path(alt)
            self.localAzintEdit.setText(str(local_azint))
            self._save_state()
        try:
            cfg = self._build_config()
            q_range = self._parse_q_range(self.editQRange.text())
            bin_frames = int(self.spinAvg.value()) if self.spinAvg is not None else 1
            if len(scan_ids) == 1:
                data = _load_scan_arrays(local_azint, scan_ids[0])
            else:
                data = self._load_and_stack_scans(local_azint, scan_ids)

            # Helper for safe divide to mirror postdoc script behaviour
            def _safe_divide(a, b):
                return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0)

            # Base arrays (frames, theta, q)
            cake_full = np.asarray(data["cake"])
            q_full = np.asarray(data["q_plot"])
            norm_full = data.get("norm")
            frames, azi_len, q_len = cake_full.shape

            def _prepare_norm(n_arr, frames_count, azi_len_local, q_len_local):
                """Shape norm to (frames, q); accept per-azi or per-q inputs."""
                if n_arr is None:
                    return np.ones((frames_count, q_len_local), dtype=cake_full.dtype)
                n = np.asarray(n_arr)
                # If norm is (frames, azi, q) -> average over theta
                if n.ndim == 3 and n.shape[1] == azi_len_local and n.shape[2] == q_len_local:
                    n = n.mean(axis=1)
                elif n.ndim == 2:
                    # (frames, q)
                    if n.shape == (frames_count, q_len_local):
                        pass
                    # (azi, q): average over theta then broadcast frames
                    elif n.shape == (azi_len_local, q_len_local):
                        n = np.broadcast_to(n.mean(axis=0)[None, :], (frames_count, q_len_local))
                    # (q, azi) swapped
                    elif n.shape == (q_len_local, azi_len_local):
                        n = np.broadcast_to(n.T.mean(axis=0)[None, :], (frames_count, q_len_local))
                    # single frame with q
                    elif n.shape == (1, q_len_local):
                        n = np.broadcast_to(n, (frames_count, q_len_local))
                elif n.ndim == 1 and n.shape[0] == q_len_local:
                    n = np.broadcast_to(n[None, :], (frames_count, q_len_local))
                else:
                    # Fallback: uniform weights
                    n = np.ones((frames_count, q_len_local), dtype=cake_full.dtype)

                # Align frame dimension
                if n.shape[0] != frames_count:
                    if n.shape[0] == 1:
                        n = np.broadcast_to(n, (frames_count, q_len_local))
                    elif n.shape[0] > frames_count:
                        n = n[:frames_count, :]
                    else:
                        pad = np.broadcast_to(n[-1:, :], (frames_count - n.shape[0], q_len_local))
                        n = np.vstack([n, pad])
                return n

            norm_full = _prepare_norm(norm_full, frames, azi_len, q_len)

            frame_counts = data.get("counts")

            # Frame slicing (applies to cake/norm)
            if cfg.scan_frames:
                cake_full = self._slice_frames_local(cake_full, cfg.scan_frames)
                norm_full = self._slice_frames_local(norm_full, cfg.scan_frames)

            # Background subtraction on cakes (before binning)
            if self.chkSubtractBg.isChecked():
                coeff = float(self.spinBgCoeff.value()) if self.spinBgCoeff is not None else 1.0
                bg_scan_text = self.editBgScan.text().strip()
                if not bg_scan_text.isdigit():
                    self._log("Background subtraction skipped: enter a numeric background scan.")
                else:
                    raw_folder = local_azint.parent / "raw"
                    if not raw_folder.exists():
                        raw_folder = None
                    cfg.bg_scan = int(bg_scan_text)
                    bg_id = cfg.bg_scan
                    bg_local = local_azint / f"scan-{bg_id:04d}_eiger_integrated.h5"
                    if not bg_local.exists():
                        self._log("Background scan missing locally; attempting download from current remote path...")
                        try:
                            host, user, beamline, proposal, visit, inferred_root = self._get_connection_fields(
                                require_local_root=False
                            )
                            remote_override = self._remote_azint_override
                            target_root = None
                            try:
                                cand = local_azint.parents[3]
                                if cand.exists():
                                    target_root = cand
                            except Exception:
                                target_root = None
                            if target_root is None:
                                target_root = inferred_root or self._default_local_root()
                            maxiv_connect.rsync_scans(
                                hostname=host,
                                username=user,
                                beamline=beamline,
                                proposal=proposal,
                                visit=visit,
                                scan_ids=[bg_id],
                                local_root=target_root,
                                log_cb=None,
                                remote_azint_override=remote_override,
                            )
                            # refresh local paths to the mirror destination
                            local_azint = Path(target_root) / beamline.lower() / str(proposal) / str(visit) / "azint"
                            raw_folder = local_azint.parent / "raw"
                            if not raw_folder.exists():
                                raw_folder = None
                            self.localAzintEdit.setText(str(local_azint))
                            self._save_state()
                        except Exception as dl_exc:  # noqa: BLE001
                            self._log(f"Background download failed: {dl_exc}")
                    try:
                        bg = compute_background(cfg.bg_scan, cfg, local_azint, raw_folder)
                        cake_bg = bg.get("saxs", {}).get("bg2d")
                        if cake_bg is not None:
                            cake_full = cake_full - coeff * cake_bg[None, :, :]
                            # Avoid LogNorm issues after subtraction
                            cake_full = np.maximum(cake_full, 1e-12)
                        self._log(f"Background subtraction applied: Inew = Iscan - {coeff} * Ibg (bg scan {cfg.bg_scan}).")
                    except Exception as exc:
                        self._log(f"Background subtraction failed: {exc}")

            # Frame binning with norm weights
            if bin_frames > 1 and cake_full.shape[0] >= bin_frames:
                splits = np.array_split(np.arange(cake_full.shape[0]), bin_frames)
                cake_binned = []
                norm_binned = []
                for idxs in splits:
                    c = cake_full[idxs]
                    n = norm_full[idxs]
                    n_sum = n.sum(axis=0)  # q
                    num = (c * n[:, None, :]).sum(axis=0)  # theta, q
                    c_bin = _safe_divide(num, n_sum[None, :])
                    cake_binned.append(c_bin)
                    norm_binned.append(n_sum)
                cake_full = np.stack(cake_binned, axis=0)
                norm_full = np.stack(norm_binned, axis=0)

            # Keep copies for full-range radial plot (full q)
            norm_for_full = norm_full.copy()
            cake_for_full = cake_full.copy()

            # Apply q-range filter for the preview (cake + norm)
            cake_filtered = cake_full
            norm_filtered = norm_full
            q_filtered = q_full
            if q_range is not None:
                qmin, qmax = q_range
                mask = (q_full >= min(qmin, qmax)) & (q_full <= max(qmin, qmax))
                if mask.any():
                    q_filtered = q_full[mask]
                    cake_filtered = cake_full[:, :, mask]
                    norm_filtered = norm_full[:, mask]
                else:
                    self._log("q-range filter ignored (no overlap with data).")

            # Weighted radial using full q-range (matches postdoc routine)
            norm_sum_full = norm_for_full.sum(axis=0)  # q
            num_full = (cake_for_full * norm_for_full[:, None, :]).sum(axis=1)  # frames, q
            radial_full = _safe_divide(num_full, norm_sum_full[None, :])

            # Weighted radial for filtered q (for bin slices)
            norm_sum_filt = norm_filtered.sum(axis=0) if norm_filtered.size else np.array([1.0])
            num_filt = (cake_filtered * norm_filtered[:, None, :]).sum(axis=1)
            radial_filtered = _safe_divide(num_filt, norm_sum_filt[None, :])

            q_int_min = float(q_filtered.min())
            q_int_max = float(q_filtered.max())

            q_bins = max(1, int(self.spinBinFrames.value()) if self.spinBinFrames is not None else 1)
            fig = self._plot_preview_layout(
                cake_filtered,
                radial_filtered,
                data["azi"],
                q_filtered,
                scan_ids,
                q_bins=q_bins,
                radial_full=radial_full,
                q_full=q_full,
                q_int_range=(q_int_min, q_int_max),
                frame_counts=frame_counts,
                scan_ids_full=scan_ids,
                norm=norm_filtered,
            )
            self._display_figure(fig)
        except FileNotFoundError as e:
            self._log(f"Preview failed: {e}")
            self._offer_missing_scan_actions(scan_ids)
        except Exception as e:
            self._log(f"Preview failed: {e}")

    def _offer_missing_scan_actions(self, scan_ids: list[int]) -> None:
        """
        When a scan file is missing, offer to download or pick another azint folder.
        """
        msg = QMessageBox(self.main)
        msg.setWindowTitle("Missing scan files")
        msg.setText(
            "Integrated scan file(s) not found locally.\n"
            "Do you want to download the requested scans from the remote beamline,\n"
            "browse the remote azint folder, or pick another local azint folder?"
        )
        btn_dl = msg.addButton("Download scans", QMessageBox.ButtonRole.AcceptRole)
        btn_remote = msg.addButton("Browse remote…", QMessageBox.ButtonRole.ActionRole)
        btn_pick = msg.addButton("Choose local folder…", QMessageBox.ButtonRole.ActionRole)
        msg.addButton(QMessageBox.StandardButton.Cancel)
        msg.exec()
        clicked = msg.clickedButton()
        if clicked == btn_dl:
            self._pending_preview_ids = scan_ids
            self._start_fetch_thread(scan_ids)
        elif clicked == btn_remote:
            self._browse_remote_azint()
        elif clicked == btn_pick:
            alt = QFileDialog.getExistingDirectory(
                self.main,
                "Select azint folder",
                self._default_browse_dir(),
            )
            if alt:
                self.localAzintEdit.setText(alt)
                self._save_state()
                self._render_preview(Path(alt), scan_ids)

    def _browse_remote_azint(self) -> None:
        """Fetch and display remote azint folder listing over SSH."""
        try:
            host, user, beamline, proposal, visit, _local_root = self._get_connection_fields(
                require_local_root=False
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self.main, "Remote browse", str(exc))
            return
        remote_azint = maxiv_connect.build_remote_azint_path(beamline, proposal, visit)
        remote_parent = remote_azint.parent
        try:
            listing_azint = maxiv_connect.run_ssh(
                host,
                user,
                f"ls -1 {remote_azint}",
                log_cb=None,
            )
            listing_parent = maxiv_connect.run_ssh(
                host,
                user,
                f"ls -1 {remote_parent}",
                log_cb=None,
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self.main, "Remote browse", f"Failed to list remote azint:\n{exc}")
            return

        dlg = QDialog(self.main)
        dlg.setWindowTitle(f"Remote azint: {remote_azint}")
        layout = QVBoxLayout(dlg)
        text = QTextEdit(dlg)
        text.setReadOnly(True)
        text.setPlainText(
            f"Parent folder: {remote_parent}\n"
            f"{'-'*60}\n"
            f"{listing_parent.strip()}\n\n"
            f"Azint folder: {remote_azint}\n"
            f"{'-'*60}\n"
            f"{listing_azint.strip()}\n"
        )
        layout.addWidget(text)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("Use remote folder:", dlg))
        path_edit = QLineEdit(str(remote_azint), dlg)
        path_row.addWidget(path_edit, 1)
        btn_use = QPushButton("Set for downloads", dlg)
        path_row.addWidget(btn_use)
        layout.addLayout(path_row)

        def _use_path():
            p = path_edit.text().strip()
            if not p:
                QMessageBox.information(dlg, "Remote path", "Path is empty.")
                return
            self._remote_azint_override = Path(p)
            self._log(f"Remote azint override set to: {p}")
            dlg.accept()

        btn_use.clicked.connect(_use_path)

        find_row = QHBoxLayout()
        find_row.addWidget(QLabel("Find scan #", dlg))
        scan_edit = QLineEdit(dlg)
        find_row.addWidget(scan_edit, 1)
        btn_find = QPushButton("Search", dlg)
        find_row.addWidget(btn_find)
        layout.addLayout(find_row)

        def _do_find():
            val = scan_edit.text().strip()
            if not val.isdigit():
                QMessageBox.information(dlg, "Search", "Enter a numeric scan ID.")
                return
            try:
                hits = maxiv_connect.run_ssh(
                    host,
                    user,
                    f"find {remote_parent} -maxdepth 3 -name 'scan-{int(val):04d}*'",
                    log_cb=None,
                )
                if not hits.strip():
                    QMessageBox.information(dlg, "Search", f"No matches for scan {val}.")
                else:
                    QMessageBox.information(dlg, "Search results", hits)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(dlg, "Search", f"Search failed:\n{exc}")

        btn_find.clicked.connect(_do_find)

        btn_close = QPushButton("Close", dlg)
        btn_close.clicked.connect(dlg.accept)
        layout.addWidget(btn_close, alignment=Qt.AlignmentFlag.AlignRight)
        dlg.resize(700, 520)
        dlg.exec()

    def _process_preview(self) -> None:
        scan_ids = _parse_scan_input(self.editScans.text())
        if not scan_ids:
            self._log("Enter scan IDs to process (e.g., 569-597).")
            return
        row = {
            "Name": f"scan_{scan_ids[0]}-{scan_ids[-1]}",
            "Scan interval": self.editScans.text().strip(),
        }
        out_dir = None
        ref = getattr(self.main, "project_path", None)
        if ref:
            try:
                base = Path(ref) / "SAXS"
                base.mkdir(parents=True, exist_ok=True)
                out_dir = base / row["Name"]
                if out_dir.exists():
                    resp = QMessageBox.question(
                        self.main,
                        "Overwrite?",
                        f"Folder {out_dir} already exists. Overwrite?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                        QMessageBox.StandardButton.No,
                    )
                    if resp == QMessageBox.StandardButton.No:
                        qtxt = (self.editQRange.text().strip() or "qfull").replace(" ", "_")
                        safe_q = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in qtxt)
                        alt = base / f"{row['Name']}_{safe_q}"
                        # if still exists, add numeric suffix
                        idx = 1
                        while alt.exists():
                            idx += 1
                            alt = base / f"{row['Name']}_{safe_q}_{idx}"
                        out_dir = alt
                out_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:  # noqa: BLE001
                self._log(f"Could not create preview output folder, using default: {exc}")
                out_dir = None
        self._run_process_row(row, output_override=out_dir, flat_output=True)

    def _process_list(self) -> None:
        if not self.embed_rows:
            self._log("Embed list is empty. Use Embed or Load list first.")
            return
        base_dir = None
        ref = getattr(self.main, "project_path", None)
        path = QFileDialog.getExistingDirectory(
            self.main,
            "Select output folder for processed data",
            str(ref) if ref else self._default_browse_dir(),
        )
        if not path:
            self._log("Process list canceled: no output folder selected.")
            return
        base_dir = Path(path)
        base_dir.mkdir(parents=True, exist_ok=True)
        failures = 0
        for row in list(self.embed_rows):
            # Use the user-selected folder directly; _run_process_row will create SAXS/scan_xxxx underneath.
            self._log(f"[Connector] Processing list row → base output: {base_dir}")
            ok = self._run_process_row(row, output_override=base_dir, flat_output=False)
            if not ok:
                failures += 1
        self._log(f"Process list complete. Failures: {failures}")

    def _embed_current(self) -> None:
        scan_ids = _parse_scan_input(self.editScans.text())
        if not scan_ids:
            self._log("Enter scan IDs to embed (e.g., 569-597).")
            return
        name = f"scan_{scan_ids[0]}-{scan_ids[-1]}"
        row = {
            "Name": name,
            "Detail": "",
            "Scan interval": self.editScans.text().strip(),
            "background": "",
            "bg_scale": "",
            "q_range1": "",
        }
        self.embed_rows.append(row)
        self._save_state()
        self._log(f"Embed list: added {name} (total {len(self.embed_rows)})")

    def _view_embed_list(self) -> None:
        self._open_embed_editor()

    def _save_embed_list_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self.main,
            "Save embed list",
            str(Path(self._default_browse_dir()) / "embed_list.csv"),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        fieldnames = self._compute_embed_columns(self.embed_rows)
        try:
            with open(path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
                for row in self.embed_rows:
                    writer.writerow(row)
            self._log(f"Embed list saved: {path}")
        except Exception as exc:
            self._log(f"Failed to save embed list: {exc}")

    def _load_embed_list(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.main,
            "Load embed list",
            self._default_browse_dir(),
            "CSV files (*.csv);;All files (*.*)",
        )
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                rows = []
                for row in reader:
                    if not row:
                        continue
                    norm: dict[str, str] = {}
                    for key, val in row.items():
                        if key is None:
                            continue
                        key_clean = str(key).strip()
                        if not key_clean:
                            continue
                        key_simple = re.sub(r"[\s_-]+", "", key_clean.lower())
                        value = str(val).strip() if val is not None else ""
                        if key_simple == "name":
                            norm["Name"] = value
                            continue
                        if key_simple == "detail":
                            norm["Detail"] = value
                            continue
                        if key_simple in {"scaninterval", "firstlast", "firrstandlast", "range"}:
                            norm["Scan interval"] = value.replace("scan_", "")
                            continue
                        if key_simple in {"background", "backgrounds", "bkg"}:
                            norm["background"] = value
                            continue
                        if key_simple in {"bgscale", "scaling"}:
                            norm["bg_scale"] = value
                            continue
                        if key_simple.startswith("qrange") or key_simple.startswith("interval"):
                            m = re.search(r"(\d+)", key_simple)
                            idx = m.group(1) if m else "1"
                            norm[f"q_range{idx}"] = value
                            continue
                        norm[key_clean] = value
                    if not any(str(v).strip() for v in norm.values()):
                        continue
                    rows.append(norm)
            self.embed_rows = rows
            self._save_state()
            self._log(f"Embed list loaded: {len(rows)} row(s)")
        except Exception as exc:
            self._log(f"Failed to load embed list: {exc}")

    # ---- Embed list editor (legacy-like) ----
    def _compute_embed_columns(self, rows: list[dict]) -> list[str]:
        base = ["Name", "Detail", "Scan interval", "background", "bg_scale"]
        extra = set()
        for row in rows:
            for k in row.keys():
                if k is None:
                    continue
                key_simple = re.sub(r"[\\s_-]+", "", str(k).strip().lower())
                if key_simple.startswith("qrange"):
                    extra.add(k)
        if not extra:
            extra = {"q_range1"}

        def _q_index(name: str) -> int:
            m = re.search(r"\\d+", name)
            return int(m.group()) if m else 0

        ordered_extra = sorted(extra, key=_q_index)
        return base + ordered_extra

    def _open_embed_editor(self) -> None:
        cols = self._compute_embed_columns(self.embed_rows)
        dlg = QDialog(self.main)
        dlg.setWindowTitle("Embed list")
        dlg.resize(820, 480)
        layout = QVBoxLayout(dlg)

        table = QTableWidget(dlg)
        table.setColumnCount(len(cols))
        table.setHorizontalHeaderLabels(cols)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        layout.addWidget(table)

        def _load_rows():
            table.setRowCount(len(self.embed_rows))
            for r, row in enumerate(self.embed_rows):
                for c, col in enumerate(cols):
                    item = QTableWidgetItem(str(row.get(col, "")))
                    table.setItem(r, c, item)

        _load_rows()

        btn_row = QHBoxLayout()
        btn_add = QPushButton("Add", dlg)
        btn_del = QPushButton("Delete", dlg)
        btn_savefile = QPushButton("Save to file…", dlg)
        btn_close = QPushButton("Close", dlg)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_del)
        btn_row.addStretch(1)
        btn_row.addWidget(btn_savefile)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

        def _collect_rows():
            rows = []
            for r in range(table.rowCount()):
                row = {}
                empty = True
                for c, col in enumerate(cols):
                    item = table.item(r, c)
                    val = item.text().strip() if item else ""
                    if val:
                        empty = False
                    row[col] = val
                if not empty:
                    rows.append(row)
            return rows

        def _add_row():
            table.insertRow(table.rowCount())

        def _del_row():
            idx = table.currentRow()
            if idx >= 0:
                table.removeRow(idx)

        def _save_file():
            path, _ = QFileDialog.getSaveFileName(
                self.main,
                "Save embed list",
                str(Path(self._default_browse_dir()) / "embed_list.csv"),
                "CSV files (*.csv);;All files (*.*)",
            )
            if not path:
                return
            rows = _collect_rows()
            try:
                with open(path, "w", newline="", encoding="utf-8") as fh:
                    writer = csv.DictWriter(fh, fieldnames=cols)
                    writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
                self._log(f"Embed list saved: {path}")
            except Exception as exc:
                self._log(f"Failed to save embed list: {exc}")

        def _close():
            self.embed_rows = _collect_rows()
            self._save_state()
            dlg.accept()

        btn_add.clicked.connect(_add_row)
        btn_del.clicked.connect(_del_row)
        btn_savefile.clicked.connect(_save_file)
        btn_close.clicked.connect(_close)

        dlg.exec()

    def _preview_raw(self) -> None:
        raw_val = (self.editRawScan.text().strip() or self.editScans.text().strip())
        scan_ids = _parse_scan_input(raw_val)
        if not scan_ids:
            self._log("Enter a raw scan ID to preview.")
            return
        scan_id = scan_ids[0]
        try:
            fields = self._get_connection_fields(require_local_root=True)
        except Exception as exc:
            self._log(str(exc))
            return
        hostname, username, beamline, proposal, visit, local_root = fields
        self._log(f"Fetching raw scan {scan_id}...")
        try:
            maxiv_connect.rsync_raw_scans(
                hostname=hostname,
                username=username,
                beamline=beamline,
                proposal=proposal,
                visit=visit,
                scan_ids=[scan_id],
                local_root=local_root,
                log_cb=lambda m: self._log(m.strip()),
            )
            local_raw = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "raw"
            img = self._load_raw_image(local_raw, scan_id)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
            ax.imshow(img, cmap="gray")
            ax.set_title(f"Raw scan {scan_id}")
            ax.set_axis_off()
            self._display_figure(fig)
        except Exception as exc:
            self._log(f"Preview raw failed: {exc}")

    def _log(self, msg: str) -> None:
        if hasattr(self.main, "logger"):
            self.main.logger.info(f"[Connector] {msg}")

    def _infer_local_root(self) -> Path | None:
        local_azint = Path(self.localAzintEdit.text().strip())
        if local_azint.exists():
            try:
                return local_azint.parents[3]
            except Exception:
                return None
        return None

    def _get_reference_saxs_dir(self) -> Path | None:
        ref = getattr(self.main, "project_path", None)
        if not ref:
            return None
        try:
            ref_path = Path(ref)
        except Exception:
            return None
        if not ref_path.exists():
            return None
        saxs_dir = ref_path / "SAXS"
        saxs_dir.mkdir(parents=True, exist_ok=True)
        return saxs_dir

    def _save_reference_config(self, key: str, cfg: dict) -> None:
        saxs_dir = self._get_reference_saxs_dir()
        if saxs_dir is None:
            return
        proposal = (self.editProposal.text().strip() or "").strip()
        visit = (self.editVisit.text().strip() or "").strip()
        if proposal and visit:
            name = f"connector_{key}_{proposal}_{visit}.json"
        else:
            name = f"connector_{key}_config.json"
        try:
            (saxs_dir / name).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _load_reference_config(self, key: str) -> dict | None:
        saxs_dir = self._get_reference_saxs_dir()
        proposal = (self.editProposal.text().strip() or "").strip()
        visit = (self.editVisit.text().strip() or "").strip()
        candidates = []
        if proposal and visit:
            candidates.append(f"connector_{key}_{proposal}_{visit}.json")
        candidates.append(f"connector_{key}_config.json")
        paths = []
        if saxs_dir is not None:
            paths.extend([saxs_dir / name for name in candidates])
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        paths.extend([CONFIG_DIR / name for name in candidates])
        for path in paths:
            if path.exists():
                try:
                    return json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
        return None

    def _get_connection_fields(self, require_local_root: bool = False):
        host = self.editHost.text().strip()
        user = self.editUser.text().strip()
        beamline = self.comboBeamline.currentText().strip() or "ForMAX"
        if not host or not user:
            raise ValueError("Hostname and username are required.")
        try:
            proposal = int(self.editProposal.text().strip())
            visit = int(self.editVisit.text().strip())
        except ValueError as exc:
            raise ValueError("Proposal and Visit must be numeric.") from exc
        local_root = self._infer_local_root()
        if require_local_root and local_root is None:
            folder = QFileDialog.getExistingDirectory(
                self.main,
                "Select local root",
                self._default_browse_dir(),
            )
            if folder:
                local_root = Path(folder)
        if require_local_root and local_root is None:
            raise ValueError("Select a valid local azint folder or local root first.")
        return host, user, beamline, proposal, visit, local_root

    def _require_local_azint(self, allow_missing: bool = False) -> Path | None:
        """Return the azint folder path.

        If allow_missing is True, only validate when a path is provided; this
        lets us auto-download when the user hasn't synced yet.
        """
        text = self.localAzintEdit.text().strip()
        if not text:
            if allow_missing:
                return Path()
            self._log("Select a valid local azint folder first.")
            return None
        local_azint = Path(text)
        if not allow_missing and (not local_azint.exists() or not local_azint.is_dir()):
            self._log(f"Local azint path is invalid: {local_azint}")
            return None
        return local_azint

    def _has_scan_files(self, azint_folder: Path, scan_ids: list[int]) -> bool:
        """Return True only if *all* requested scan files exist locally."""
        if not azint_folder or not azint_folder.exists():
            return False
        for scan_id in scan_ids:
            fname = azint_folder / f"scan-{scan_id:04d}_eiger_integrated.h5"
            if not fname.exists():
                return False
        return True

    def _fetch_scans(self, scan_ids: list[int]) -> bool:
        """Try to rsync scans into a local mirror if connection info is present."""
        try:
            host, user, beamline, proposal, visit, local_root = self._get_connection_fields(
                require_local_root=False
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Cannot download scans: {exc}")
            return False

        # Default cache if no local root inferred
        if local_root is None:
            local_root = self._default_local_root()

        try:
            self._log(f"Fetching scans via rsync to {local_root}...")
            maxiv_connect.rsync_scans(
                hostname=host,
                username=user,
                beamline=beamline,
                proposal=proposal,
                visit=visit,
                scan_ids=scan_ids,
                local_root=local_root,
                log_cb=lambda m: self._log(m.strip()),
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Download failed: {exc}")
            return False

        azint_folder = local_root / beamline.lower() / str(proposal) / str(visit) / "azint"
        self.localAzintEdit.setText(str(azint_folder))
        self._save_state()
        self._log(f"Scans downloaded. Using local azint: {azint_folder}")
        return True

    # ---------- Non-blocking fetch helpers ----------
    def _start_fetch_thread(self, scan_ids: list[int]) -> None:
        if self._fetch_thread is not None and self._fetch_thread.isRunning():
            self._log("Download already in progress.")
            return
        try:
            host, user, beamline, proposal, visit, local_root = self._get_connection_fields(
                require_local_root=False
            )
        except Exception as exc:  # noqa: BLE001
            self._log(f"Cannot download scans: {exc}")
            return

        if local_root is None:
            local_root = self._default_local_root()

        worker = _FetchWorker(
            host,
            user,
            beamline,
            proposal,
            visit,
            scan_ids,
            local_root,
            log_cb=None,  # avoid GUI logging from worker thread
            remote_azint_override=self._remote_azint_override,
        )
        thread = QThread(self.main)
        self._fetch_thread = thread
        self._fetch_worker = worker
        worker.moveToThread(thread)
        worker.finished.connect(self._on_fetch_finished)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.started.connect(worker.run)
        thread.start()
        self._log(
            f"Downloading scans {scan_ids} via rsync to {local_root} in background... (you can keep working)"
        )

    def _on_fetch_finished(self, success: bool, azint_path: str, error: str) -> None:
        self._fetch_thread = None
        self._fetch_worker = None
        if not success:
            self._log(f"Download failed: {error}")
            return
        self.localAzintEdit.setText(azint_path)
        self._save_state()
        self._log(f"Scans downloaded. Using local azint: {azint_path}")
        if self._pending_preview_ids:
            scan_ids = self._pending_preview_ids
            self._pending_preview_ids = None
            self._render_preview(Path(azint_path), scan_ids)


    def _parse_frame_range(self, text: str):
        text = (text or "").strip()
        if not text:
            return None
        if "-" not in text:
            try:
                val = int(text)
                return (val, val)
            except Exception:
                return None
        a, b = text.split("-", 1)
        try:
            return (int(a.strip()), int(b.strip()))
        except Exception:
            return None

    def _build_config(self) -> AzintProcessConfig:
        beamline = self.comboBeamline.currentText().strip() or "ForMAX"
        cfg_dict = self._config_formax if beamline == "ForMAX" else self._config_cosaxs
        cfg = AzintProcessConfig.from_dict(cfg_dict)
        bg_scan_text = self.editBgScan.text().strip()
        cfg.bg_scan = int(bg_scan_text) if bg_scan_text.isdigit() else None
        # BG/scan frame slicing disabled in UI; always use full frames
        cfg.bg_frames = None
        cfg.scan_frames = None
        # If the user supplied a q-range in the connector UI, apply it to both SAXS/WAXS
        q_range = self._parse_q_range(self.editQRange.text())
        if q_range:
            qmin, qmax = q_range
            cfg.saxs_azi_windows = [(qmin, qmax)]
            cfg.waxs_azi_windows = [(qmin, qmax)]
            cfg.saxs_rad_range = (qmin, qmax)
            cfg.waxs_rad_range = (qmin, qmax)
        return cfg

    # ----------- Multi-scan loader (parity with v5.8.1) -----------
    @staticmethod
    def _orient_azimuthal(arr, azi):
        """Ensure azimuthal array shape is [frames, azi]."""
        import numpy as np

        azi_len = azi.shape[0]
        if arr.ndim == 1:
            if arr.shape[0] != azi_len:
                raise ValueError("Azimuth array length mismatch.")
            return arr[np.newaxis, :]
        if arr.shape[1] == azi_len:
            return arr
        if arr.shape[0] == azi_len:
            return arr.T
        raise ValueError(f"Azimuth array shape {arr.shape} does not align with azi length {azi_len}")

    @staticmethod
    def _orient_radial(arr, q_vals):
        """Ensure radial array shape is [frames, q]."""
        import numpy as np

        q_len = q_vals.shape[0]
        if arr.ndim == 1:
            if arr.shape[0] != q_len:
                raise ValueError("Radial array length mismatch.")
            return arr[np.newaxis, :]
        if arr.shape[1] == q_len:
            return arr
        if arr.shape[0] == q_len:
            return arr.T
        raise ValueError(f"Radial array shape {arr.shape} does not align with q length {q_len}")

    @staticmethod
    def _orient_cake(arr, azi, q_vals):
        """Ensure cake array shape is [frames, azi, q]."""
        import numpy as np

        azi_len = azi.shape[0]
        q_len = q_vals.shape[0]
        if arr.ndim == 2:
            if arr.shape != (azi_len, q_len):
                raise ValueError("Cake array shape mismatch.")
            return arr[np.newaxis, ...]
        if arr.ndim == 3:
            f, a, q = arr.shape
            if a == azi_len and q == q_len:
                return arr
            if a == q_len and q == azi_len:
                return arr.transpose(0, 2, 1)
        raise ValueError(f"Cake array shape {arr.shape} does not align with azi/q lengths")

    @staticmethod
    def _orient_norm(arr, q_vals):
        """Ensure norm array shape is [frames, q]."""
        import numpy as np

        if arr is None:
            return None
        a = np.asarray(arr)
        q_len = q_vals.shape[0]
        if a.ndim == 1:
            if a.shape[0] != q_len:
                raise ValueError("Norm array length mismatch.")
            return a[np.newaxis, :]
        if a.ndim == 2:
            if a.shape[1] == q_len:
                return a
            if a.shape[0] == q_len:
                return a.T
        raise ValueError(f"Norm array shape {a.shape} does not align with q length {q_len}")

    def _load_and_stack_scans(self, local_azint: Path, scan_ids: list[int]) -> dict:
        """
        Load multiple scans and concatenate frames across scans (no per-scan averaging).
        Keeps full frame axis so cake shows all frames in the range.
        """
        import numpy as np

        cakes = []
        radials = []
        norms = []
        frame_counts: list[int] = []
        azi_ref = None
        q_ref = None

        for scan_id in scan_ids:
            self._log(f"Loading scan {scan_id} for preview...")
            scan = _load_scan_arrays(local_azint, scan_id)
            cake = scan["cake"]
            R = scan["R"]
            azi = scan["azi"]
            q_plot = scan["q_plot"]
            norm_arr = scan.get("norm")

            if azi_ref is None:
                azi_ref = azi
                q_ref = q_plot
            else:
                if azi.shape[0] != azi_ref.shape[0]:
                    raise ValueError(
                        f"Scan {scan_id} azimuth bins differ from first scan ({azi.shape[0]} vs {azi_ref.shape[0]})"
                    )
                if q_plot.shape[0] != q_ref.shape[0]:
                    raise ValueError(
                        f"Scan {scan_id} q bins differ from first scan ({q_plot.shape[0]} vs {q_ref.shape[0]})"
                    )

            cakes.append(self._orient_cake(cake, azi_ref, q_ref))
            radials.append(self._orient_radial(R, q_ref))
            norm_oriented = (
                self._orient_norm(norm_arr, q_ref)
                if norm_arr is not None
                else np.ones((cake.shape[0] if cake.ndim == 3 else 1, q_ref.shape[0]), dtype=cake.dtype)
            )
            norms.append(norm_oriented)
            frame_counts.append(cake.shape[0] if cake.ndim == 3 else 1)

        if not cakes or not radials:
            raise ValueError("No scan data loaded.")

        cake_cat = np.concatenate(cakes, axis=0)
        radial_cat = np.concatenate(radials, axis=0)
        norm_cat = np.concatenate(norms, axis=0) if norms else None

        return {
            "cake": cake_cat,
            "R": radial_cat,
            "azi": azi_ref,
            "q_plot": q_ref,
            "counts": frame_counts,
            "norm": norm_cat,
        }

    def _run_process_row(self, row: dict, output_override: Path | None = None, flat_output: bool = False) -> bool:
        try:
            host, user, beamline, proposal, visit, local_root = self._get_connection_fields(
                require_local_root=True
            )
        except Exception as exc:
            self._log(str(exc))
            return False
        cfg = self._build_config()
        self._last_bg_scan = self.editBgScan.text().strip()
        self._last_bg_frames = self.editBgFrames.text().strip()
        self._last_scan_frames = self.editScanFrames.text().strip()
        self._bg_subtract = self.chkSubtractBg.isChecked()
        self._bg_coeff = float(self.spinBgCoeff.value()) if self.spinBgCoeff is not None else 1.0

        scan_ids = _parse_scan_input(row.get("Scan interval", ""))
        if not scan_ids:
            self._log("No scan IDs found in row.")
            return False
        local_root = Path(local_root)
        azint_folder = local_root / beamline.lower() / str(proposal) / str(visit) / "azint"
        raw_folder = local_root / beamline.lower() / str(proposal) / str(visit) / "raw"
        output_root = output_override or (local_root / beamline.lower() / f"{proposal}_{visit}" / "processed")

        # Ensure required scans are present locally; if missing, try to fetch via rsync.
        missing_ids = list(scan_ids)
        if cfg.bg_scan is not None:
            try:
                missing_ids.append(int(cfg.bg_scan))
            except Exception:
                pass
        if not self._has_scan_files(azint_folder, missing_ids):
            self._log("Local azint missing requested scans; attempting download...")
            if not self._fetch_scans(missing_ids):
                self._log("Download failed or not configured; process row aborted.")
                return False
            # Refresh azint/raw paths after download (local root might change)
            text = self.localAzintEdit.text().strip()
            azint_folder = Path(text) if text else azint_folder
            raw_folder = azint_folder.parent / "raw"

        if not azint_folder.exists():
            self._log(f"Azint folder not found: {azint_folder}")
            return False

        # Pre-validate background; if missing, warn and continue without subtraction.
        if cfg.bg_scan is not None:
            try:
                compute_background(cfg.bg_scan, cfg, azint_folder, raw_folder)
            except FileNotFoundError as exc:
                self._log(f"Background scan missing ({exc}); proceeding without background subtraction.")
                cfg.bg_scan = None
            except Exception as exc:
                self._log(f"Background check failed ({exc}); proceeding without background subtraction.")
                cfg.bg_scan = None

        # For list processing (flat_output=False), nest under SAXS/scan_xxxx and allow overwrite
        proc_output_root = output_root
        if not flat_output:
            sax_root = output_root / "SAXS"
            sax_root.mkdir(parents=True, exist_ok=True)
            for sid in scan_ids:
                scan_dir = sax_root / f"scan_{sid:04d}"
                if scan_dir.exists():
                    shutil.rmtree(scan_dir, ignore_errors=True)
            proc_output_root = sax_root

        outputs, failures = process_scans(
            scan_ids,
            cfg,
            azint_folder,
            raw_folder,
            proc_output_root,
            log_cb=self._log,
            flat_output=flat_output,
        )
        if failures:
            self._log(f"Failures: {len(failures)}")
            return False
        return True

    def _load_raw_image(self, local_raw: Path, scan_id: int):
        import numpy as np
        try:
            import h5py
        except Exception as exc:
            raise RuntimeError(f"h5py is required to load raw scans: {exc}") from exc
        candidates = [
            local_raw / f"scan-{scan_id:04d}_eiger_master.h5",
            local_raw / f"scan-{scan_id:04d}_master.h5",
        ]
        file_path = None
        for cand in candidates:
            if cand.exists():
                file_path = cand
                break
        if file_path is None:
            raise FileNotFoundError(f"No raw file found for scan {scan_id} in {local_raw}")
        with h5py.File(file_path, "r") as f:
            datasets: list[tuple[str, np.ndarray]] = []

            def _visitor(name, obj):
                if not isinstance(obj, h5py.Dataset):
                    return
                if obj.ndim < 2:
                    return
                datasets.append((name, obj))

            f.visititems(_visitor)
            if not datasets:
                raise RuntimeError(f"No 2D/3D datasets found in {file_path}")
            datasets.sort(key=lambda item: item[1].size, reverse=True)
            data = np.array(datasets[0][1])
        if data.ndim >= 3:
            data = data[0]
        return data

    def _load_state(self) -> dict:
        # Retained for backward compatibility; delegates to module-level helper.
        return _load_connector_state()

    def _save_state(self) -> None:
        payload = {
            "hostname": self.editHost.text().strip(),
            "username": self.editUser.text().strip(),
            "beamline": self.comboBeamline.currentText(),
            "proposal": self.editProposal.text().strip(),
            "visit": self.editVisit.text().strip(),
            "local_azint": self.localAzintEdit.text().strip(),
            "scans_text": self.editScans.text().strip(),
            "embed_rows": self.embed_rows,
            "config_formax": self._config_formax,
            "config_cosaxs": self._config_cosaxs,
            "bg_scan": self.editBgScan.text().strip(),
            "bg_frames": "",
            "scan_frames": "",
            "q_range": self.editQRange.text().strip(),
            "bin_frames": int(self.spinBinFrames.value()) if self.spinBinFrames is not None else 1,
            "bg_subtract": self.chkSubtractBg.isChecked(),
            "bg_coeff": float(self.spinBgCoeff.value()) if self.spinBgCoeff is not None else 1.0,
            "last_output_dir": str(self._last_output_dir) if self._last_output_dir else "",
        }
        STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        try:
            STATE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        except Exception:
            pass

    def _default_local_root(self) -> Path:
        """
        Choose a writable cache root:
        - Prefer hidden '_Temp' inside the current project/reference folder, if available.
        - Fallback to ~/.mudpaw_cache.
        """
        ref = getattr(self.main, "project_path", None)
        if ref:
            try:
                ref_path = Path(ref)
                if ref_path.exists() and ref_path.is_dir():
                    temp = ref_path / "_Temp"
                    temp.mkdir(parents=True, exist_ok=True)
                    return temp
            except Exception:
                pass
        cache = Path.home() / ".mudpaw_cache"
        cache.mkdir(parents=True, exist_ok=True)
        return cache

    def _clear_cache(self) -> None:
        """
        Remove the current cache root after confirmation.
        """
        azint_txt = self.localAzintEdit.text().strip()
        target_root = None
        if azint_txt:
            p = Path(azint_txt)
            try:
                target_root = p.parents[3]
            except Exception:
                target_root = p.parent if p.exists() else None
        if target_root is None or not target_root.exists():
            target_root = self._default_local_root()
        msg = QMessageBox(self.main)
        msg.setWindowTitle("Clear cache")
        msg.setText(
            f"This will delete all cached connector data under:\n{target_root}\n\n"
            "Proceed?"
        )
        msg.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        msg.setIcon(QMessageBox.Icon.Warning)
        if msg.exec() != QMessageBox.StandardButton.Yes:
            return
        try:
            shutil.rmtree(target_root)
            self._log(f"Cache cleared: {target_root}")
            target_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            self._log(f"Failed to clear cache: {exc}")

    # ---------- Helpers for preview math ----------
    @staticmethod
    def _parse_q_range(text: str):
        text = (text or "").strip()
        if not text:
            return (0.002, 0.5)
        if "-" not in text:
            parts = text.split()
        else:
            parts = text.replace(",", " ").split("-")
        try:
            qmin = float(parts[0])
            qmax = float(parts[1]) if len(parts) > 1 else qmin
            return qmin, qmax
        except Exception:
            return None

    @staticmethod
    def _slice_frames_local(arr: np.ndarray, frame_range):
        if frame_range is None:
            return arr
        start, end = frame_range
        start = max(0, int(start))
        end = int(end)
        if end < start:
            start, end = end, start
        return arr[start : end + 1]

    @staticmethod
    def _recompute_azi_bands(cake: np.ndarray, q_vals: np.ndarray):
        mid = max(1, q_vals.shape[0] // 2)
        A0 = cake[:, :, :mid].mean(axis=2)
        A1 = cake[:, :, mid:].mean(axis=2)
        return A0, A1

    @staticmethod
    def _q_range_titles(q_vals: np.ndarray) -> dict:
        qmin_r = float(np.min(q_vals))
        qmax_r = float(np.max(q_vals))
        mid = max(1, q_vals.shape[0] // 2)
        qmin0 = float(q_vals[0])
        qmax0 = float(q_vals[mid - 1])
        qmin1 = float(q_vals[mid])
        qmax1 = float(q_vals[-1])
        return {
            "qmin0": qmin0,
            "qmax0": qmax0,
            "qmin1": qmin1,
            "qmax1": qmax1,
            "qmin_r": qmin_r,
            "qmax_r": qmax_r,
        }

    def _plot_preview_layout(
        self,
        cake: np.ndarray,
        radial: np.ndarray,
        azi: np.ndarray,
        q_plot: np.ndarray,
        scan_ids: list[int],
        q_bins: int = 1,
        radial_full: np.ndarray | None = None,
        q_full: np.ndarray | None = None,
        q_int_range: tuple[float, float] | None = None,
        frame_counts: list[int] | None = None,
        scan_ids_full: list[int] | None = None,
        norm: np.ndarray | None = None,
    ):
        """Preview layout: single band by default; when q_bins>1, show per-bin cakes side by side plus overlayed azimuthal lines and one radial plot."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import LogNorm
        import numpy as np

        cake = np.asarray(cake)                       # already sliced by q-range
        radial_bins = np.asarray(radial)              # already sliced by q-range
        if radial_full is not None and q_full is not None:
            radial_plot_src = np.asarray(radial_full)
            q_for_rad = np.asarray(q_full)
        else:
            radial_plot_src = np.asarray(radial_full) if radial_full is not None else radial_bins
            # Use the actually plotted q-range for radial x-axis to avoid empty leading range
            q_for_rad = np.asarray(q_plot)
        if cake.ndim != 3:
            raise ValueError("cake must be [frames, azi, q]")
        if radial_bins.ndim == 1:
            radial_bins = radial_bins[np.newaxis, :]
        if radial_plot_src.ndim == 1:
            radial_plot_src = radial_plot_src[np.newaxis, :]

        def _split_q_bins(cake_arr, radial_arr, q_vals, norm_arr, n_bins):
            bins = []
            n = max(1, n_bins)
            edges = np.linspace(q_vals.min(), q_vals.max(), n + 1)
            for i in range(n):
                mask = (q_vals >= edges[i]) & (q_vals <= edges[i + 1])
                if not mask.any():
                    continue
                if norm_arr is not None:
                    norm_mask = norm_arr[:, mask]  # (frames, qbin)
                    norm_sum_q = norm_mask.sum(axis=1)  # (frames,)
                    cake_num = (cake_arr[:, :, mask] * norm_mask[:, None, :]).sum(axis=2)  # (frames, azi)
                    cake_bin = np.divide(
                        cake_num,
                        norm_sum_q[:, None],
                        out=np.zeros_like(cake_num, dtype=float),
                        where=norm_sum_q[:, None] != 0,
                    )
                    # Radial bin representative (sum over frames, theta) / sum(norm over frames)
                    num = (cake_arr[:, :, mask] * norm_mask[:, None, :]).sum(axis=1)  # (frames, qbin)
                    denom = norm_mask.sum(axis=0)  # (qbin)
                    rad_bin = np.divide(
                        num.sum(axis=0),
                        np.maximum(denom, 1e-12),
                    )
                else:
                    cake_bin = cake_arr[:, :, mask].mean(axis=2)
                    rad_bin = radial_arr[:, mask].mean(axis=0)
                bins.append(
                    {
                        "cake": cake_bin,
                        "rad": rad_bin,
                        "label": f"{edges[i]:.4f}–{edges[i+1]:.4f} Å⁻¹",
                        "qmin": float(edges[i]),
                        "qmax": float(edges[i + 1]),
                    }
                )
            return bins

        bins = _split_q_bins(cake, radial_bins, q_plot, norm, q_bins)
        if not bins:
            if norm is not None:
                norm_sum_q = norm.sum(axis=1)  # (frames,)
                cake_num = (cake * norm[:, None, :]).sum(axis=2)
                cake_one = np.divide(
                    cake_num,
                    norm_sum_q[:, None],
                    out=np.zeros_like(cake_num, dtype=float),
                    where=norm_sum_q[:, None] != 0,
                )
            else:
                cake_one = cake.mean(axis=2)
            bins = [
                {
                    "cake": cake_one,
                    "rad": radial.mean(axis=0),
                    "label": f"{q_plot.min():.4f}–{q_plot.max():.4f} Å⁻¹",
                    "qmin": float(q_plot.min()),
                    "qmax": float(q_plot.max()),
                }
            ]

        n_bins = len(bins)
        fig = plt.figure(figsize=(4 * max(n_bins, 2), 6))
        gs = fig.add_gridspec(2, max(n_bins, 2), height_ratios=[2, 1.2])

        def _auto_log_limits(arr):
            """Compute per-plot log limits that adapt to current data (for q-range changes)."""
            finite = np.isfinite(arr)
            pos = arr[(arr > 0) & finite]
            if pos.size == 0:
                return 1e-6, 1.0
            vmin = np.percentile(pos, 1.0)
            vmax = np.percentile(pos, 99.5)
            if vmin <= 0:
                vmin = max(pos.min(), 1e-6)
            if vmax <= vmin:
                vmax = vmin * 10
            return float(vmin), float(vmax)

        # Row 1: cakes side by side
        for idx, b in enumerate(bins):
            ax = fig.add_subplot(gs[0, idx])
            frames_total = b["cake"].shape[0]
            vmin, vmax = _auto_log_limits(b["cake"])
            im = ax.imshow(
                b["cake"].T,
                extent=[0, frames_total, azi[0], azi[-1]],
                origin="lower",
                cmap="viridis",
                aspect="auto",
                norm=LogNorm(vmin=vmin, vmax=vmax),
            )
            ax.set_title(b["label"], fontsize=9)
            ax.set_xlabel("Frame #", fontsize=9)
            ax.set_ylabel(r"Theta [deg]", fontsize=9)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Choose a consistent palette per frame (match cake viridis)
        cmap = plt.cm.viridis
        n_frames_total = radial_plot_src.shape[0]
        frame_colors = cmap(np.linspace(0, 1, max(n_frames_total, 2)))

        # Row 2: azimuthal lines (left)
        ax_azi = fig.add_subplot(gs[1, 0])
        frame_idx = 0
        for b in bins:
            for frame in b["cake"]:
                color = frame_colors[min(frame_idx, frame_colors.shape[0] - 1)]
                ax_azi.plot(azi, frame, color=color, alpha=0.35, linewidth=0.9)
                frame_idx += 1
        ax_azi.set_xlim(0, 360)
        ax_azi.set_xlabel(r"Theta [deg]")
        ax_azi.set_ylabel("I [a.u.]")
        if n_bins > 1:
            ax_azi.legend(fontsize=8)

        # Row 2: radial (right)
        ax_rad = fig.add_subplot(gs[1, 1])
        # Radial: plot every frame's I(q) (theta-averaged) for the full stack with same palette
        for idx, frame in enumerate(radial_plot_src):
            color = frame_colors[min(idx, frame_colors.shape[0] - 1)]
            ax_rad.plot(q_for_rad, frame, color=color, alpha=0.25, linewidth=0.9)
        # Optional mean overlay for reference (darker tone)
        ax_rad.plot(q_for_rad, radial_plot_src.mean(axis=0), color="black", linewidth=1.4, alpha=0.8)
        ax_rad.set_xscale("log")
        ax_rad.set_yscale("log")
        ax_rad.set_xlabel("q [Å⁻¹]")
        ax_rad.set_ylabel("I [a.u.]")
        q_min_eff = max(float(q_for_rad.min()), 1e-4)
        q_max_eff = float(q_for_rad.max())
        ax_rad.set_xlim(q_min_eff, q_max_eff)
        # show bin ranges on radial plot
        if n_bins > 1:
            for b in bins:
                ax_rad.axvspan(b["qmin"], b["qmax"], color="gray", alpha=0.08)
                ax_rad.axvline(b["qmin"], color="gray", ls="--", lw=0.7)
                ax_rad.axvline(b["qmax"], color="gray", ls="--", lw=0.7)
        else:
            # single range: mark full q-range with dotted lines
            ql, qr = q_int_range if q_int_range is not None else (q_min_eff, q_max_eff)
            ax_rad.axvline(ql, color="gray", ls="--", lw=0.8)
            ax_rad.axvline(qr, color="gray", ls="--", lw=0.8)

        first_scan, last_scan = scan_ids[0], scan_ids[-1]
        scan_title = f"Scan {first_scan}" if first_scan == last_scan else f"Scans {first_scan}-{last_scan}"
        fig.suptitle(scan_title, fontsize=10)
        fig.tight_layout()
        return fig

    @staticmethod
    def _bin_frames(arr: np.ndarray, bins: int):
        frames = arr.shape[0]
        if bins <= 1 or bins == frames:
            return arr
        bins = min(bins, frames)
        split = np.array_split(np.arange(frames), bins)
        out = []
        for idxs in split:
            out.append(arr[idxs].mean(axis=0))
        return np.stack(out, axis=0)


class _FetchWorker(QObject):
    finished = pyqtSignal(bool, str, str)  # success, azint_path, error_msg

    def __init__(
        self,
        host: str,
        user: str,
        beamline: str,
        proposal: int,
        visit: int,
        scan_ids: list[int],
        local_root: Path,
        log_cb,
        remote_azint_override: Path | None = None,
    ) -> None:
        super().__init__()
        self.host = host
        self.user = user
        self.beamline = beamline
        self.proposal = proposal
        self.visit = visit
        self.scan_ids = scan_ids
        self.local_root = local_root
        self.log_cb = log_cb
        self.remote_azint_override = remote_azint_override

    def run(self) -> None:
        try:
            maxiv_connect.rsync_scans(
                hostname=self.host,
                username=self.user,
                beamline=self.beamline,
                proposal=self.proposal,
                visit=self.visit,
                scan_ids=self.scan_ids,
                local_root=self.local_root,
                log_cb=None,  # suppress cross-thread logging to avoid crashes
                remote_azint_override=self.remote_azint_override,
            )
            azint_folder = (
                Path(self.local_root)
                / self.beamline.lower()
                / str(self.proposal)
                / str(self.visit)
                / "azint"
            )
            self.finished.emit(True, str(azint_folder), "")
        except Exception as exc:  # noqa: BLE001
            self.finished.emit(False, "", str(exc))
