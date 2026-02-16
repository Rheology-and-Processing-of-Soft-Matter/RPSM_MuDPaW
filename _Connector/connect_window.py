from pathlib import Path
from typing import Any, Iterable, List, Optional, Tuple
import subprocess
import json
import re
import csv

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from _Connector import maxiv_connect
from _Connector.formax_preview import (
    detect_scan_ids,
    group_continuous_ranges,
    plot_azint_overview,
)
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from _Connector.embed_list_window import EmbedListWindow


def load_scan_arrays(local_azint: Path, scan_id: int):
    """
    Load ForMAX integrated HDF5 file for the given scan and extract a dict:
        {
            "A0", "A1", "R", "azi", "q_plot",
            "qmin0", "qmax0", "qmin1", "qmax1", "qmin_r", "qmax_r",
            "cake"  # full azimuthal stack shaped (frames, azi, q)
        }
    """
    import h5py
    import numpy as np
    from pathlib import Path

    fname = f"scan-{scan_id:04d}_eiger_integrated.h5"
    filepath = local_azint / fname

    if not filepath.exists():
        raise FileNotFoundError(f"Integrated scan file not found: {filepath}")

    with h5py.File(filepath, "r") as f:
        try:
            # Preferred ForMAX layout
            azi = np.array(f["entry/azimuthal/azi"])
            A0 = np.array(f["entry/azimuthal/A0"])
            A1 = np.array(f["entry/azimuthal/A1"])
            q_plot = np.array(f["entry/radial/q"])
            R = np.array(f["entry/radial/R"])
            cake = np.array(f["entry/azimuthal/cake"]) if "entry/azimuthal/cake" in f else None
            qmin0 = float(f["entry/azimuthal/A0"].attrs["qmin"])
            qmax0 = float(f["entry/azimuthal/A0"].attrs["qmax"])
            qmin1 = float(f["entry/azimuthal/A1"].attrs["qmin"])
            qmax1 = float(f["entry/azimuthal/A1"].attrs["qmax"])
        except KeyError:
            # Fallback to azint2d/azint1d structure
            azi = np.array(f["entry/azint2d/data/azimuthal_axis"])
            q_plot = np.array(f["entry/azint2d/data/radial_axis"])
            cake = np.array(f["entry/azint2d/data/I"])  # shape (F, A, Q)
            R = np.array(f["entry/azint1d/data/I"])     # shape (F, Q)

            # Normalize shapes to [frame, ...]
            if cake.ndim == 3:
                pass
            elif cake.ndim == 2:
                cake = cake[np.newaxis, ...]
            else:
                raise RuntimeError(f"Unexpected azint2d/data/I shape: {cake.shape}")

            if R.ndim == 1:
                R = R[np.newaxis, ...]

            # Split q-range into two bands for A0/A1 preview
            mid = max(1, q_plot.shape[0] // 2)
            A0 = cake[:, :, :mid].mean(axis=2)
            A1 = cake[:, :, mid:].mean(axis=2)

            qmin0 = float(q_plot[0])
            qmax0 = float(q_plot[mid - 1])
            qmin1 = float(q_plot[mid])
            qmax1 = float(q_plot[-1])

    if cake is None:
        raise RuntimeError("No azimuthal cake data found in the integrated file.")

    qmin_r = float(q_plot.min())
    qmax_r = float(q_plot.max())

    return {
        "A0": A0,
        "A1": A1,
        "R": R,
        "azi": azi,
        "q_plot": q_plot,
        "qmin0": qmin0,
        "qmax0": qmax0,
        "qmin1": qmin1,
        "qmax1": qmax1,
        "qmin_r": qmin_r,
        "qmax_r": qmax_r,
        "cake": cake,
    }


class ConnectWindow(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Connect to MAX IV (ForMAX)")
        self.minsize(960, 640)
        self._state_file = Path.home() / ".mudpaw_connect.json"

        self.hostname_var = tk.StringVar(value="login.maxiv.lu.se")
        self.username_var = tk.StringVar()
        self.beamline_var = tk.StringVar(value="ForMAX")
        self.proposal_var = tk.StringVar()
        self.visit_var = tk.StringVar()
        self.local_root_var = tk.StringVar()
        self.scan_no_var = tk.StringVar()
        self.raw_scan_var = tk.StringVar()
        self.preview_progress = None

        self.remote_azint_label_var = tk.StringVar(value="Remote azint: -")
        self.scan_ranges_var = tk.StringVar(value="Detected scan ranges: -")
        self.preview_state: Optional[dict] = None
        self.embed_rows: list[dict] = []
        self.embed_counter = 1
        self.embed_window = None
        self.embed_list_path = None
        self.avg_frames_var = tk.IntVar(value=4)
        self.cosaxs_config = {
            "dark_i0": -0.015692761504974766,
            "dark_it": -0.004680580198485629,
            "empty_i0": 0.7040761073201496,
            "empty_it": 0.17054912055292087,
            "qSumStart": 0.01,
            "qSumEnd": 0.08,
            "absolute_scaling_saxs": 0.00833716555970972,
            "absolute_scaling_waxs": 0.000003,
            "thickness": 1.0,
            "normalization": "none",
            "average": False,
            "pictures": "None",
            "cormap": False,
            "auto_buffer": False,
            "waxs": False,
            "save": True,
            "format": "dat",
        }
        self.formax_config = {
            "empty_i0": 0.172828152,
            "empty_it": 0.411628054,
            "qSumStart": 0.01,
            "qSumEnd": 0.08,
            "absolute_scaling_saxs": 1.0,
            "absolute_scaling_waxs": 1.0,
            "thickness": 1.0,
            "normalization": "transmission",
            "save": True,
            "format": "dat",
        }
        self.connected = False

        self.vpn_status_var = tk.StringVar(value="VPN status: unknown")
        self.vpn_status_label = None
        self.vpn_check_target = "192.168.19.100"  # internal MAX IV address, reachable only via VPN
        self.vpn_profile_id = "cc82faff8b573e4d"

        self._build_ui()
        self._load_state()
        self.after(2000, self._check_vpn_status)
        self.protocol("WM_DELETE_WINDOW", self._on_close)
        # style for connected indicator
        style = ttk.Style(self)
        style.configure("Connected.TButton", foreground="white", background="#2e8b57")
        style.map("Connected.TButton", background=[("active", "#2e8b57")], foreground=[("active", "white")])

    def _default_browse_dir(self) -> str:
        """
        Best-effort starting directory for file/directory dialogs:
        - Prefer the app-wide reference folder stored in _Miscell/last_folder.txt.
        - Fallback to the user's home directory.
        """
        candidates = [
            Path(__file__).resolve().parents[1] / "_Miscell" / "last_folder.txt",
            Path(__file__).resolve().parents[1] / "last_folder.txt",
        ]
        for cand in candidates:
            try:
                txt = cand.read_text(encoding="utf-8").strip()
                if txt:
                    p = Path(txt).expanduser()
                    if p.exists():
                        return str(p)
            except Exception:
                continue
        return str(Path.home())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        # Controls row (multiple columns) and plot row
        container.columnconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.columnconfigure(2, weight=1)
        container.rowconfigure(0, weight=0)
        container.rowconfigure(1, weight=1)

        controls_row = ttk.Frame(container)
        controls_row.grid(row=0, column=0, columnspan=3, sticky="nw")
        for c in range(3):
            controls_row.columnconfigure(c, weight=1)

        self._build_connection_frame(controls_row)
        self._build_scan_frame(controls_row)
        self._build_bottom_frames(controls_row)

        # Plot area across full width
        self._build_plot_host(container)

    def _build_connection_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Connection")
        frame.grid(row=0, column=0, sticky="nw", padx=(0, 10), pady=(0, 6))

        # Two fixed-width columns: 0 = fields, 1 = buttons
        frame.columnconfigure(0, weight=0)
        frame.columnconfigure(1, weight=0)

        # --- Left column: all fields stacked in their own subframe ---
        fields = ttk.Frame(frame)
        fields.grid(row=0, column=0, sticky="w", padx=4, pady=2)

        # Make labels and entries align nicely inside the fields frame
        fields.columnconfigure(0, weight=0)  # labels
        fields.columnconfigure(1, weight=0)  # entries (fixed width via width=...)

        # Row 0: Hostname
        ttk.Label(fields, text="Hostname").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(fields, textvariable=self.hostname_var, width=24).grid(
            row=0, column=1, sticky="w", padx=4, pady=2
        )

        # Row 1: Username
        ttk.Label(fields, text="Username").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(fields, textvariable=self.username_var, width=24).grid(
            row=1, column=1, sticky="w", padx=4, pady=2
        )

        # Row 2: Beamline
        ttk.Label(fields, text="Beamline").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        beamline_menu = ttk.OptionMenu(
            fields, self.beamline_var, self.beamline_var.get(), "ForMAX", "CoSAXS"
        )
        beamline_menu.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        # Row 3: Proposal
        ttk.Label(fields, text="Proposal").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(fields, textvariable=self.proposal_var, width=24).grid(
            row=3, column=1, sticky="w", padx=4, pady=2
        )

        # Row 4: Visit
        ttk.Label(fields, text="Visit").grid(row=4, column=0, sticky="w", padx=4, pady=2)
        ttk.Entry(fields, textvariable=self.visit_var, width=24).grid(
            row=4, column=1, sticky="w", padx=4, pady=2
        )

        # Row 5: Local root + Browse button on the right column
        # ttk.Label(fields, text="Local root").grid(row=5, column=0, sticky="w", padx=4, pady=2)
        # ttk.Entry(fields, textvariable=self.local_root_var, width=24).grid(
        #    row=5, column=1, sticky="w", padx=4, pady=2
        # x)

        # --- Right column: buttons, one per row or just the big connect button ---

        btns = ttk.Frame(frame)
        btns.grid(row=5, column=0, columnspan=2, sticky="nw", padx=4, pady=2)
        self._connect_btn = ttk.Button(btns, text="Connect", command=self._on_connect_only, style="TButton")
        self._connect_btn.grid(row=0, column=0, sticky="w", padx=(0, 6), pady=2)
        ttk.Button(btns, text="List scans", command=self._on_list_scans).grid(
            row=0, column=1, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Button(btns, text="ForMAX config", command=self._open_formax_config).grid(
            row=1, column=0, sticky="w", padx=(0, 6), pady=2
        )
        ttk.Button(btns, text="CoSAXS config", command=self._open_cosaxs_config).grid(
            row=1, column=1, sticky="w", padx=(0, 6), pady=2
        )

        # VPN status indicator and reconnect button
        status_frame = ttk.Frame(frame)
        status_frame.grid(row=6, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))

        self.vpn_status_label = ttk.Label(
            status_frame,
            textvariable=self.vpn_status_var,
            foreground="orange",
        )
        self.vpn_status_label.grid(row=0, column=0, sticky="w")

        ttk.Button(
            status_frame,
            text="Reconnect VPN",
            command=self._on_reconnect_vpn,
        ).grid(row=0, column=1, padx=10, sticky="w")


    def _build_scan_frame(self, parent: ttk.Frame) -> None:
        frame = ttk.LabelFrame(parent, text="Scan overview")
        frame.grid(row=0, column=1, sticky="nw", padx=(0, 10), pady=(0, 6))

        ttk.Label(
            frame,
            textvariable=self.remote_azint_label_var,
            wraplength=400,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=4, pady=(4, 2))

        ttk.Label(
            frame,
            textvariable=self.scan_ranges_var,
        ).grid(row=1, column=0, sticky="w", padx=4, pady=(4, 2))

    def _build_bottom_frames(self, parent: ttk.Frame) -> None:
        bottom = ttk.Frame(parent)
        bottom.grid(row=0, column=2, sticky="nw", pady=(0, 6))
        bottom.columnconfigure(0, weight=0)
        bottom.columnconfigure(1, weight=0)

        # --- Preview panel (two columns only) ---
        preview = ttk.LabelFrame(bottom, text="Preview")
        preview.grid(row=0, column=0, columnspan=2, sticky="w", padx=(0, 6))
        preview.columnconfigure(0, weight=0)   # left column: labels + entries
        preview.columnconfigure(1, weight=0)   # right column: buttons

        # Row 0: Scans field and Preview button
        ttk.Label(preview, text="Scans:").grid(
            row=0, column=0, padx=4, pady=2, sticky="w"
        )
        ttk.Entry(preview, textvariable=self.scan_no_var, width=20).grid(
            row=0, column=1, padx=4, pady=2, sticky="w"
        )

        # Put the Preview button on its own line in col1 (still two columns total)
        ttk.Button(preview, text="Preview", command=self._on_preview).grid(
            row=1, column=1, padx=4, pady=(2, 2), sticky="w"
        )

        # Avg frames row
        avg_row = ttk.Frame(preview)
        avg_row.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 2))
        ttk.Label(avg_row, text="Avg frames:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Entry(avg_row, textvariable=self.avg_frames_var, width=8).grid(
            row=0, column=1, padx=4, pady=2, sticky="w"
        )
        # Process buttons
        proc_row = ttk.Frame(preview)
        proc_row.grid(row=3, column=0, columnspan=2, sticky="w", padx=4, pady=(2, 2))
        ttk.Button(proc_row, text="Process preview", command=self._on_process_preview).grid(
            row=0, column=0, padx=(0, 6), pady=2, sticky="w"
        )
        ttk.Button(proc_row, text="Process list", command=self._on_process_list).grid(
            row=0, column=1, padx=(0, 6), pady=2, sticky="w"
        )

        # --- Embed bar spans both columns ---
        embed_bar = ttk.Frame(preview)
        embed_bar.grid(row=4, column=0, columnspan=2, sticky="w", padx=4, pady=(4, 2))
        embed_bar.columnconfigure(0, weight=0)
        embed_bar.columnconfigure(1, weight=0)
        embed_bar.columnconfigure(2, weight=0)

        ttk.Button(embed_bar, text="Embed", command=self._on_embed).grid(
            row=0, column=0, padx=(0, 6), pady=2, sticky="w"
        )
        ttk.Button(embed_bar, text="View list", command=self._on_view_embed_list).grid(
            row=0, column=1, padx=6, pady=2, sticky="w"
        )
        ttk.Button(embed_bar, text="Load list", command=self._on_load_embed_list).grid(
            row=0, column=2, padx=(6, 0), pady=2, sticky="w"
        )

        # Raw preview row spans both columns
        raw_row = ttk.Frame(preview)
        raw_row.grid(row=5, column=0, columnspan=2, sticky="w", padx=4, pady=(6, 2))
        ttk.Label(raw_row, text="Raw scan:").grid(row=0, column=0, padx=4, pady=2, sticky="w")
        ttk.Entry(raw_row, textvariable=self.raw_scan_var, width=10).grid(
            row=0, column=1, padx=4, pady=2, sticky="w"
        )
        ttk.Button(raw_row, text="Preview raw pattern", command=self._on_preview_raw).grid(
            row=0, column=2, padx=(2, 0), pady=2, sticky="w"
        )

        # Hidden log widget
        self.log_text = tk.Text(bottom, height=1, width=1)
        self.log_text.grid_forget()

    def _build_plot_host(self, parent: ttk.Frame) -> None:
        self.plot_host = ttk.Frame(parent)
        # Occupy the second row across all columns; this area expands with the window
        self.plot_host.grid(row=1, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(2, weight=1)
        parent.rowconfigure(1, weight=1)

    def _on_close(self):
        self._save_state()
        self.destroy()

    # ------------------------------------------------------------------ Actions

    def _set_connected_state(self, ok: bool) -> None:
        self.connected = ok
        if hasattr(self, "_connect_btn"):
            self._connect_btn.configure(style="Connected.TButton" if ok else "TButton")

    def _fetch_remote_scans(self) -> tuple[str, list[int]]:
        hostname, username, beamline, proposal, visit, local_root = self._read_connection_fields(
            require_local_root=False
        )
        #if beamline != "ForMAX":
        #    raise ValueError("Only ForMAX is supported at the moment.")
        list_kwargs = dict(
            hostname=hostname,
            username=username,
            beamline=beamline,
            proposal=proposal,
            visit=visit,
            log_cb=self._log_line,
        )
        if local_root:
            list_kwargs["local_root"] = local_root
        remote_path, scan_ids = maxiv_connect.list_remote_scans(**list_kwargs)
        return remote_path, scan_ids

    def _on_connect_only(self) -> None:
        self._log_line("Connecting...\n")
        try:
            remote_path, scan_ids = self._fetch_remote_scans()
        except Exception as exc:  # noqa: BLE001
            self._set_connected_state(False)
            messagebox.showerror("Connection error", str(exc), parent=self)
            return
        self.remote_azint_label_var.set(f"Remote azint:\n {remote_path}")
        # Do not alter detected ranges here; just confirm connectivity
        self._set_connected_state(True)
        self._save_state()

    def _on_list_scans(self) -> None:
        self._log_line("Listing scans...\n")
        try:
            remote_path, scan_ids = self._fetch_remote_scans()
        except Exception as exc:  # noqa: BLE001
            self._set_connected_state(False)
            messagebox.showerror("List scans", str(exc), parent=self)
            return
        self.remote_azint_label_var.set(f"Remote azint:\n {remote_path}")
        self._populate_scan_list(scan_ids)
        self._set_connected_state(True)
        self._save_state()

    def _open_terminal_ssh(self) -> None:
        """Open macOS Terminal with an ssh command to allow password entry."""
        hostname = self.hostname_var.get().strip()
        username = self.username_var.get().strip()
        if not hostname or not username:
            messagebox.showerror("SSH", "Hostname and username are required to open SSH.", parent=self)
            return
        cmd = ["open", "-a", "Terminal", f"ssh {username}@{hostname}"]
        try:
            subprocess.Popen(cmd)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("SSH", f"Failed to open Terminal:\n{exc}", parent=self)

    def _open_cosaxs_config(self) -> None:
        """Open a simple editor for CoSAXS configuration values."""
        dlg = tk.Toplevel(self)
        dlg.title("CoSAXS configuration")
        dlg.transient(self)
        dlg.grab_set()
        fields = [
            ("dark_i0", float),
            ("dark_it", float),
            ("empty_i0", float),
            ("empty_it", float),
            ("qSumStart", float),
            ("qSumEnd", float),
            ("absolute_scaling_saxs", float),
            ("absolute_scaling_waxs", float),
            ("thickness", float),
            ("normalization", str),
            ("average", bool),
            ("pictures", str),
            ("cormap", bool),
            ("auto_buffer", bool),
            ("waxs", bool),
            ("save", bool),
            ("format", str),
        ]
        entries = {}
        for idx, (key, typ) in enumerate(fields):
            ttk.Label(dlg, text=key).grid(row=idx, column=0, padx=6, pady=3, sticky="w")
            if typ is bool:
                var = tk.BooleanVar(value=bool(self.cosaxs_config.get(key, False)))
                chk = ttk.Checkbutton(dlg, variable=var)
                chk.grid(row=idx, column=1, padx=6, pady=3, sticky="w")
                entries[key] = var
            else:
                var = tk.StringVar(value=str(self.cosaxs_config.get(key, "")))
                ent = ttk.Entry(dlg, textvariable=var, width=18)
                ent.grid(row=idx, column=1, padx=6, pady=3, sticky="w")
                entries[key] = var

        def _save_to_json():
            path = filedialog.asksaveasfilename(
                parent=dlg,
                title="Save CoSAXS config",
                initialdir=self._default_browse_dir(),
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            updated = {}
            for key, var in entries.items():
                if isinstance(var, tk.BooleanVar):
                    updated[key] = bool(var.get())
                else:
                    val = var.get()
                    # try numeric conversion where appropriate
                    if key in {"format", "normalization", "pictures"}:
                        updated[key] = val
                    else:
                        try:
                            updated[key] = float(val)
                        except Exception:
                            updated[key] = val
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(updated, fh, indent=2)
            except Exception as exc:
                messagebox.showerror("Save config", f"Failed to save config:\n{exc}", parent=dlg)

        def _load_from_json():
            path = filedialog.askopenfilename(
                parent=dlg,
                title="Load CoSAXS config",
                initialdir=self._default_browse_dir(),
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                loaded = json.loads(Path(path).read_text())
                for key, var in entries.items():
                    if key in loaded:
                        if isinstance(var, tk.BooleanVar):
                            var.set(bool(loaded.get(key)))
                        else:
                            var.set(str(loaded.get(key)))
            except Exception as exc:
                messagebox.showerror("Load config", f"Failed to load config:\n{exc}", parent=dlg)

        def _apply_and_close():
            updated = {}
            for key, var in entries.items():
                if isinstance(var, tk.BooleanVar):
                    updated[key] = bool(var.get())
                else:
                    val = var.get()
                    if key in {"format", "normalization", "pictures"}:
                        updated[key] = val
                    else:
                        try:
                            updated[key] = float(val)
                        except Exception:
                            updated[key] = val
            self.cosaxs_config = updated
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="Load", command=_load_from_json).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Save as...", command=_save_to_json).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="Apply", command=_apply_and_close).grid(row=0, column=2, padx=4)

        dlg.columnconfigure(1, weight=1)

    def _open_formax_config(self) -> None:
        """Open a simple editor for ForMAX configuration values."""
        dlg = tk.Toplevel(self)
        dlg.title("ForMAX configuration")
        dlg.transient(self)
        dlg.grab_set()
        fields = [
            ("empty_i0", float),
            ("empty_it", float),
            ("qSumStart", float),
            ("qSumEnd", float),
            ("absolute_scaling_saxs", float),
            ("absolute_scaling_waxs", float),
            ("thickness", float),
            ("normalization", str),
            ("save", bool),
            ("format", str),
        ]
        entries = {}
        for idx, (key, typ) in enumerate(fields):
            ttk.Label(dlg, text=key).grid(row=idx, column=0, padx=6, pady=3, sticky="w")
            if typ is bool:
                var = tk.BooleanVar(value=bool(self.formax_config.get(key, False)))
                chk = ttk.Checkbutton(dlg, variable=var)
                chk.grid(row=idx, column=1, padx=6, pady=3, sticky="w")
                entries[key] = var
            else:
                var = tk.StringVar(value=str(self.formax_config.get(key, "")))
                ent = ttk.Entry(dlg, textvariable=var, width=18)
                ent.grid(row=idx, column=1, padx=6, pady=3, sticky="w")
                entries[key] = var

        def _save_to_json():
            path = filedialog.asksaveasfilename(
                parent=dlg,
                title="Save ForMAX config",
                initialdir=self._default_browse_dir(),
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            updated = {}
            for key, var in entries.items():
                if isinstance(var, tk.BooleanVar):
                    updated[key] = bool(var.get())
                else:
                    val = var.get()
                    if key in {"format", "normalization"}:
                        updated[key] = val
                    else:
                        try:
                            updated[key] = float(val)
                        except Exception:
                            updated[key] = val
            try:
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(updated, fh, indent=2)
            except Exception as exc:
                messagebox.showerror("Save config", f"Failed to save config:\n{exc}", parent=dlg)

        def _load_from_json():
            path = filedialog.askopenfilename(
                parent=dlg,
                title="Load ForMAX config",
                initialdir=self._default_browse_dir(),
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )
            if not path:
                return
            try:
                loaded = json.loads(Path(path).read_text())
                for key, var in entries.items():
                    if key in loaded:
                        if isinstance(var, tk.BooleanVar):
                            var.set(bool(loaded.get(key)))
                        else:
                            var.set(str(loaded.get(key)))
            except Exception as exc:
                messagebox.showerror("Load config", f"Failed to load config:\n{exc}", parent=dlg)

        def _apply_and_close():
            updated = {}
            for key, var in entries.items():
                if isinstance(var, tk.BooleanVar):
                    updated[key] = bool(var.get())
                else:
                    val = var.get()
                    if key in {"format", "normalization"}:
                        updated[key] = val
                    else:
                        try:
                            updated[key] = float(val)
                        except Exception:
                            updated[key] = val
            self.formax_config = updated
            dlg.destroy()

        btn_frame = ttk.Frame(dlg)
        btn_frame.grid(row=len(fields), column=0, columnspan=2, pady=8)
        ttk.Button(btn_frame, text="Load", command=_load_from_json).grid(row=0, column=0, padx=4)
        ttk.Button(btn_frame, text="Save as...", command=_save_to_json).grid(row=0, column=1, padx=4)
        ttk.Button(btn_frame, text="Apply", command=_apply_and_close).grid(row=0, column=2, padx=4)

        dlg.columnconfigure(1, weight=1)

    def _check_vpn_status(self) -> None:
        """Ping an internal MAX IV address to infer whether the VPN is up and update the status label."""
        import subprocess

        target = self.vpn_check_target
        try:
            # On macOS, -c 1 = 1 packet, -W 1 = 1 second timeout
            result = subprocess.run(
                ["ping", "-c", "1", "-W", "1", target],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if result.returncode == 0:
                self.vpn_status_var.set("● VPN connected")
                if self.vpn_status_label is not None:
                    self.vpn_status_label.configure(foreground="green")
            else:
                self.vpn_status_var.set("● VPN disconnected")
                if self.vpn_status_label is not None:
                    self.vpn_status_label.configure(foreground="red")
        except Exception:
            self.vpn_status_var.set("● VPN status error")
            if self.vpn_status_label is not None:
                self.vpn_status_label.configure(foreground="orange")

        # Schedule next check if the window is still alive
        if self.winfo_exists():
            self.after(10000, self._check_vpn_status)

    def _on_reconnect_vpn(self) -> None:
        """Attempt to restart the Pritunl VPN profile using the local CLI client."""
        import subprocess

        # Full path to the Pritunl client on macOS installations
        client_path = "/Applications/Pritunl.app/Contents/Resources/pritunl-client"
        profile_id = self.vpn_profile_id

        try:
            # Stop (ignore failures)
            subprocess.run(
                [client_path, "stop", profile_id],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            # Start
            result = subprocess.run(
                [client_path, "start", profile_id, "--mode=ovpn"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                messagebox.showerror(
                    "VPN",
                    f"Failed to restart VPN profile:\n{result.stderr or result.stdout}",
                    parent=self,
                )
            else:
                # Give the tunnel a brief moment to come up before the next status check
                self.vpn_status_var.set("● VPN reconnect requested...")
                if self.vpn_status_label is not None:
                    self.vpn_status_label.configure(foreground="orange")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("VPN", f"Error while restarting VPN:\n{exc}", parent=self)

    def _populate_scan_list(self, scan_ids: Iterable[int]) -> None:
        ids = sorted({int(s) for s in scan_ids})
        ranges = group_continuous_ranges(ids)
        if ranges:
            ranges_text = ", ".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)
        else:
            ranges_text = "-"
        self.scan_ranges_var.set(f"Detected scan ranges: \n {ranges_text}")

    def _parse_scan_input(self, value: str) -> List[int]:
        """
        Accept inputs like:
            "1201-1205"
            "1201-1205, 1210 1212"
            "1201,1203,1205"
        and return a sorted, deduplicated list of ints.
        """
        text = value.strip()
        if not text:
            raise ValueError("Please enter at least one scan number or range.")

        scan_ids: List[int] = []
        for part in re.split(r"[;,\\s]+", text):
            if not part:
                continue
            if "-" in part:
                start_str, end_str = part.split("-", 1)
                try:
                    start = int(start_str)
                    end = int(end_str)
                except ValueError as exc:
                    raise ValueError(f"Invalid range segment: '{part}'") from exc
                if end < start:
                    start, end = end, start
                scan_ids.extend(range(start, end + 1))
            else:
                try:
                    scan_ids.append(int(part))
                except ValueError as exc:
                    raise ValueError(f"Invalid scan number: '{part}'") from exc

        unique_ids = sorted(set(scan_ids))
        if not unique_ids:
            raise ValueError("No valid scan numbers found.")
        return unique_ids

    @staticmethod
    def _format_scan_ids(scan_ids: Iterable[int]) -> str:
        ids = sorted(set(int(s) for s in scan_ids))
        if not ids:
            return "-"
        ranges = group_continuous_ranges(ids)
        parts = []
        for a, b in ranges:
            parts.append(f"{a}-{b}" if a != b else f"{a}")
        return ", ".join(parts)

    def _on_preview(self) -> None:
        self._progress_start()
        try:
            scan_ids = self._parse_scan_input(self.scan_no_var.get())
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self)
            self._progress_stop()
            return

        try:
            hostname, username, beamline, proposal, visit, local_root = self._read_connection_fields(
                require_local_root=False
            )
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self)
            self._progress_stop()
            return

        if not local_root:
            local_root = Path.home() / ".mudpaw_cache"
            local_root.mkdir(parents=True, exist_ok=True)

        # Ensure we have the scan locally (rsync from JupyterHub/login host)
        self._log_line(f"Fetching scans {self._format_scan_ids(scan_ids)} via rsync...\n")
        try:
            maxiv_connect.rsync_scans(
                hostname=hostname,
                username=username,
                beamline=beamline,
                proposal=proposal,
                visit=visit,
                scan_ids=scan_ids,
                local_root=local_root,
                log_cb=self._log_line,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Download error", str(exc), parent=self)
            self._progress_stop()
            return

        local_azint = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "azint"
        self._log_line(f"Loading scans from {local_azint}\n")

        try:
            plot_args = self._load_and_average_scans(local_azint, scan_ids)
        except NotImplementedError as exc:
            messagebox.showwarning("Not implemented", str(exc), parent=self)
            self._progress_stop()
            return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load error", str(exc), parent=self)
            self._progress_stop()
            return

        self.last_scan_ids = scan_ids
        self._open_azint_window(plot_args, local_azint)
        self.preview_state = {"scan_ids": scan_ids, "intervals": [], "background_scan": None}
        self._save_state()
        self._progress_stop()

    def _on_preview_raw(self) -> None:
        self._progress_start()
        try:
            raw_val = self.raw_scan_var.get().strip() or self.scan_no_var.get().strip()
            scan_no = int(raw_val)
        except ValueError:
            messagebox.showerror("Invalid input", "Raw scan number must be an integer.", parent=self)
            self._progress_stop()
            return

        try:
            hostname, username, beamline, proposal, visit, local_root = self._read_connection_fields(
                require_local_root=False
            )
        except ValueError as exc:
            messagebox.showerror("Invalid input", str(exc), parent=self)
            self._progress_stop()
            return

        if not local_root:
            local_root = Path.home() / ".mudpaw_cache"
            local_root.mkdir(parents=True, exist_ok=True)

        # Download raw file
        self._log_line(f"Fetching raw scan {scan_no} via rsync...\n")
        try:
            maxiv_connect.rsync_raw_scans(
                hostname=hostname,
                username=username,
                beamline=beamline,
                proposal=proposal,
                visit=visit,
                scan_ids=[scan_no],
                local_root=local_root,
                log_cb=self._log_line,
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Download error", str(exc), parent=self)
            self._progress_stop()
            return

        local_raw = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "raw"
        try:
            img = self._load_raw_image(local_raw, scan_no)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Load error", str(exc), parent=self)
            self._progress_stop()
            return

        RawPreviewWindow(self, img, local_raw, loader=self._load_raw_image)
        self._save_state()
        self._progress_stop()

    def _open_azint_window(self, data: dict, local_azint: Path) -> None:
        """Show interactive azimuth/radial preview with interval controls embedded."""
        # create plot host if missing
        if not hasattr(self, "plot_host") or self.plot_host is None:
            messagebox.showerror("Preview", "Plot area is not available in this window.", parent=self)
            return
        # clear previous
        for child in self.plot_host.winfo_children():
            child.destroy()
        panel = AzintInteractiveWindow(self.plot_host, self, data, local_azint)
        panel.pack(fill=tk.BOTH, expand=True)
        self.azint_panel = panel

    def _load_and_average_scans(self, local_azint: Path, scan_ids: List[int]) -> dict:
        """
        Load multiple scans, orient axes, average over frames per scan, and stack by scan.
        Returns a dict of arrays ready for plot_azint_overview.
        """
        A0_rows: List[np.ndarray] = []
        A1_rows: List[np.ndarray] = []
        R_rows: List[np.ndarray] = []
        cake_rows: List[np.ndarray] = []

        azi_ref: Optional[np.ndarray] = None
        q_ref: Optional[np.ndarray] = None
        q_params: Optional[Tuple[float, float, float, float, float, float]] = None

        for scan_id in scan_ids:
            self._log_line(f"  Loading scan {scan_id}")
            scan = load_scan_arrays(local_azint, scan_id)
            A0 = scan["A0"]
            A1 = scan["A1"]
            R = scan["R"]
            azi = scan["azi"]
            q_plot = scan["q_plot"]
            qmin0 = scan["qmin0"]
            qmax0 = scan["qmax0"]
            qmin1 = scan["qmin1"]
            qmax1 = scan["qmax1"]
            qmin_r = scan["qmin_r"]
            qmax_r = scan["qmax_r"]
            cake = scan["cake"]

            if azi_ref is None:
                azi_ref = np.asarray(azi)
                q_ref = np.asarray(q_plot)
                q_params = (qmin0, qmax0, qmin1, qmax1, qmin_r, qmax_r)
            else:
                if azi.shape[0] != azi_ref.shape[0]:
                    raise ValueError(
                        f"Scan {scan_id} azimuth bins differ from first scan ({azi.shape[0]} vs {azi_ref.shape[0]})"
                    )
                if q_plot.shape[0] != q_ref.shape[0]:
                    raise ValueError(
                        f"Scan {scan_id} q bins differ from first scan ({q_plot.shape[0]} vs {q_ref.shape[0]})"
                    )

            A0_oriented = self._orient_azimuthal(A0, azi_ref)
            A1_oriented = self._orient_azimuthal(A1, azi_ref)
            R_oriented = self._orient_radial(R, q_ref)
            cake_oriented = self._orient_cake(cake, azi_ref, q_ref)

            A0_rows.append(A0_oriented.mean(axis=0))
            A1_rows.append(A1_oriented.mean(axis=0))
            R_rows.append(R_oriented.mean(axis=0))
            cake_rows.append(cake_oriented.mean(axis=0))

        if not (A0_rows and A1_rows and R_rows):
            raise ValueError("No scan data loaded.")

        assert azi_ref is not None and q_ref is not None and q_params is not None  # for type-checkers

        return {
            "A0": np.vstack(A0_rows),
            "A1": np.vstack(A1_rows),
            "R": np.vstack(R_rows),
            "azi": azi_ref,
            "q_plot": q_ref,
            "scan_ids": scan_ids,
            "qmin0": q_params[0],
            "qmax0": q_params[1],
            "qmin1": q_params[2],
            "qmax1": q_params[3],
            "qmin_r": q_params[4],
            "qmax_r": q_params[5],
            "cake": np.stack(cake_rows, axis=0),  # (scans, azi, q)
        }

    @staticmethod
    def _orient_azimuthal(arr: np.ndarray, azi: np.ndarray) -> np.ndarray:
        """Ensure azimuthal array is shaped [frames, angles]."""
        azi_len = azi.shape[0]
        if arr.ndim == 1:
            if arr.shape[0] != azi_len:
                raise ValueError(f"Azimuthal array length {arr.shape[0]} does not match azi {azi_len}")
            return arr[np.newaxis, :]
        if arr.shape[1] == azi_len:
            return arr
        if arr.shape[0] == azi_len:
            return arr.T
        raise ValueError(f"Azimuthal array shape {arr.shape} does not align with azi length {azi_len}")

    @staticmethod
    def _orient_radial(arr: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
        """Ensure radial array is shaped [frames, q]."""
        q_len = q_vals.shape[0]
        if arr.ndim == 1:
            if arr.shape[0] != q_len:
                raise ValueError(f"Radial array length {arr.shape[0]} does not match q length {q_len}")
            return arr[np.newaxis, :]
        if arr.shape[1] == q_len:
            return arr
        if arr.shape[0] == q_len:
            return arr.T
        raise ValueError(f"Radial array shape {arr.shape} does not align with q length {q_len}")

    @staticmethod
    def _orient_cake(arr: np.ndarray, azi: np.ndarray, q_vals: np.ndarray) -> np.ndarray:
        """Ensure cake array is shaped [frames, azi, q]."""
        azi_len = azi.shape[0]
        q_len = q_vals.shape[0]
        if arr.ndim == 2:
            # assume [azi, q]
            if arr.shape != (azi_len, q_len):
                raise ValueError(f"Cake array shape {arr.shape} does not align with azi/q lengths")
            return arr[np.newaxis, ...]
        if arr.ndim == 3:
            f, a, q = arr.shape
            if a == azi_len and q == q_len:
                return arr
            if a == q_len and q == azi_len:
                return arr[:, :, :].transpose(0, 2, 1)
        raise ValueError(f"Cake array shape {arr.shape} does not align with azi/q lengths")

    # ------------------------------------------------------------------ Helpers
    def _read_connection_fields(self, require_local_root: bool = True) -> Tuple[str, str, str, int, int, Optional[Path]]:
        hostname = self.hostname_var.get().strip()
        username = self.username_var.get().strip()
        beamline = self.beamline_var.get().strip()
        local_root_val = self.local_root_var.get().strip()
        local_root = Path(local_root_val) if local_root_val else None

        if not hostname:
            raise ValueError("Hostname is required.")
        if not username:
            raise ValueError("Username is required.")
        if require_local_root and not local_root:
            raise ValueError("Local root is required.")

        try:
            proposal = int(self.proposal_var.get())
        except ValueError as exc:
            raise ValueError("Proposal must be an integer.") from exc

        try:
            visit = int(self.visit_var.get())
        except ValueError as exc:
            raise ValueError("Visit must be an integer.") from exc

        return hostname, username, beamline, proposal, visit, local_root

    def _log_line(self, line: str) -> None:
        text = line if line.endswith("\n") else f"{line}\n"
        if hasattr(self, "log_text") and self.log_text:
            try:
                self.log_text.configure(state="normal")
                self.log_text.insert("end", text)
                self.log_text.see("end")
                self.log_text.configure(state="disabled")
                return
            except Exception:
                pass
        print(text, end="")

    def _load_raw_image(self, local_raw: Path, scan_id: int) -> np.ndarray:
        """
        Load a raw Eiger image from the local raw mirror.
        """
        import h5py

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

            # choose largest dataset
            datasets.sort(key=lambda item: item[1].size, reverse=True)
            data = np.array(datasets[0][1])

        if data.ndim >= 3:
            data = data[0]
        data = np.squeeze(data)
        return data

    def _progress_start(self) -> None:
        if hasattr(self, "preview_progress") and self.preview_progress:
            self.preview_progress.start(10)

    def _progress_stop(self) -> None:
        if hasattr(self, "preview_progress") and self.preview_progress:
            self.preview_progress.stop()

    # ---------------------- Embed list management ----------------------
    def _on_embed(self) -> None:
        if not self.preview_state:
            messagebox.showerror("Embed", "Preview data not available yet. Run Preview first.", parent=self)
            return
        scan_ids = self.preview_state.get("scan_ids", [])
        if not scan_ids:
            messagebox.showerror("Embed", "No scan IDs in preview state.", parent=self)
            return
        ranges = group_continuous_ranges(sorted(set(scan_ids)))
        ranges_str = ";".join(f"{a}-{b}" if a != b else f"{a}" for a, b in ranges)
        bg = self.preview_state.get("background_scan")
        bg_scale = ""
        panel = getattr(self, "azint_panel", None)
        if panel and getattr(panel, "bg_scale_var", None) is not None:
            try:
                bg_scale = f"{float(panel.bg_scale_var.get()):.4g}"
            except Exception:
                bg_scale = ""
        detail = f"Scan_{ranges_str}"
        intervals = self.preview_state.get("intervals", [])
        interval_strs = []
        for (a, b) in intervals:
            interval_strs.append(f"{a:.4g}-{b:.4g}")
        detail = f"{detail} intervals {';'.join(interval_strs)}" if interval_strs else detail
        row = {
            "Name": f"No {self.embed_counter}",
            "Detail": detail,
            "Scan interval": ranges_str,
            "background": str(bg) if bg is not None else "",
            "bg_scale": bg_scale,
        }
        for idx, rng in enumerate(interval_strs, start=1):
            row[f"q_range{idx}"] = rng
        self.embed_counter += 1
        self.embed_rows.append(row)
        if self.embed_window and tk.Toplevel.winfo_exists(self.embed_window):
            try:
                self.embed_window.refresh()
            except Exception:
                pass
        messagebox.showinfo("Embed", f"Added entry: {row['Name']}", parent=self)
        if not (self.embed_window and tk.Toplevel.winfo_exists(self.embed_window)):
            self._on_view_embed_list()

    def _on_view_embed_list(self) -> None:
        if not self.embed_rows:
            messagebox.showinfo("Embed list", "No entries yet. Click Embed after a preview to add one.", parent=self)
            return
        if self.embed_window and tk.Toplevel.winfo_exists(self.embed_window):
            try:
                self.embed_window.refresh()
            except Exception:
                pass
            self.embed_window.lift()
            return
        self.embed_window = EmbedListWindow(self, self.embed_rows, on_save=self._save_embed_rows)

    def _load_embed_list_from_path(self, path: str, show_message: bool = True) -> bool:
        try:
            rows = []
            with open(path, newline="", encoding="utf-8") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    if not row:
                        continue
                    # Normalize headers into current schema
                    norm = {}
                    for key, val in row.items():
                        if key is None:
                            continue
                        key_clean = str(key).strip()
                        if not key_clean or key_clean.lower() == "detail":
                            continue
                        key_simple = re.sub(r"[\\s_-]+", "", key_clean.lower())
                        m = re.search(r"(\\d+)", key_simple)
                        idx = m.group(1) if m else "1"
                        if key_simple.startswith("qrange") or key_simple.startswith("interval"):
                            norm[f"q_range{idx}"] = val
                        else:
                            norm[key_clean] = val
                    # scan interval
                    if not norm.get("Scan interval"):
                        rng = norm.get("First-last") or norm.get("Firrst and last") or norm.get("range") or ""
                        norm["Scan interval"] = rng.replace("scan_", "").strip()
                    # background
                    if not norm.get("background"):
                        bkg = norm.get("backgrounds") or norm.get("Background") or norm.get("bkg") or ""
                        norm["background"] = bkg.strip()
                    # background scale
                    if not norm.get("bg_scale"):
                        norm["bg_scale"] = norm.get("Scaling", "").strip()
                    # q ranges
                    if not any(k.lower().startswith("q_range") for k in norm.keys()):
                        q_rng = norm.get("Q-range") or norm.get("q_range") or ""
                        if q_rng:
                            norm["q_range1"] = q_rng.strip()
                    rows.append(norm)
            self.embed_rows = rows
            self.embed_counter = len(rows) + 1
            self.embed_list_path = path
            if self.embed_window and tk.Toplevel.winfo_exists(self.embed_window):
                try:
                    self.embed_window.refresh()
                except Exception:
                    pass
            if show_message:
                messagebox.showinfo("Embed", f"Loaded {len(rows)} entries.", parent=self)
            return True
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Embed", f"Failed to load list:\n{exc}", parent=self)
            return False

    def _on_load_embed_list(self) -> None:
        path = filedialog.askopenfilename(
            title="Load embed list CSV",
            initialdir=self._default_browse_dir(),
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not path:
            return
        self._load_embed_list_from_path(path)

    def _save_embed_rows(self, rows: list[dict]) -> None:
        self.embed_rows = rows
        self.embed_counter = len(rows) + 1

    # ---------------------- Processing / export ----------------------
    @staticmethod
    def _avg_every_n_rows(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 1:
            return arr
        F = arr.shape[0]
        k = F // n
        if k <= 0:
            return arr
        trimmed = arr[: k * n, ...]
        new_shape = (k, n) + arr.shape[1:]
        return trimmed.reshape(new_shape).mean(axis=1)

    @staticmethod
    def _frame_headers(ncols: int, first_label: str) -> str:
        n_frames = max(0, ncols - 1)
        frames = [f"f{i}" for i in range(1, n_frames + 1)]
        return "\t".join([first_label] + frames)

    @staticmethod
    def _sanitize_sample_name(name: str, detail: str = "", fallback: str = "sample") -> str:
        raw = f"{(name or '').strip()}_{(detail or '').strip()}"
        s = re.sub(r"\s+", "_", raw).strip("_")
        # allow +, -, ., _, parentheses; put hyphen at end to avoid range
        s = re.sub(r"[^A-Za-z0-9_.()+-]", "_", s)
        s = re.sub(r"_+", "_", s)
        return s or fallback

    def _parse_scan_tokens(self, text: str) -> list[int]:
        ids: list[int] = []
        for tok in re.split(r"[;,\\s]+", text.strip()):
            if not tok:
                continue
            if "-" in tok:
                try:
                    a, b = tok.split("-", 1)
                    a_i, b_i = int(a), int(b)
                    if b_i < a_i:
                        a_i, b_i = b_i, a_i
                    ids.extend(list(range(a_i, b_i + 1)))
                except Exception:
                    continue
            else:
                try:
                    ids.append(int(tok))
                except Exception:
                    continue
        return sorted(set(ids))

    def _ensure_local_scans(
        self,
        scan_ids: list[int],
        hostname: str,
        username: str,
        beamline: str,
        proposal: int,
        visit: int,
        local_root: Path,
    ) -> Path:
        maxiv_connect.rsync_scans(
            hostname=hostname,
            username=username,
            beamline=beamline,
            proposal=proposal,
            visit=visit,
            scan_ids=scan_ids,
            local_root=local_root,
            log_cb=self._log_line,
        )
        local_azint = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "azint"
        local_azint.mkdir(parents=True, exist_ok=True)
        return local_azint

    def _load_full_scan(
        self, local_azint: Path, scan_id: int, ref_azi: Optional[np.ndarray] = None, ref_q: Optional[np.ndarray] = None
    ) -> dict:
        scan = load_scan_arrays(local_azint, scan_id)
        azi = ref_azi if ref_azi is not None else np.asarray(scan["azi"])
        q_plot = ref_q if ref_q is not None else np.asarray(scan["q_plot"])
        R = self._orient_radial(scan["R"], q_plot)
        cake = self._orient_cake(scan["cake"], azi, q_plot)
        return {"azi": azi, "q": q_plot, "R": R, "cake": cake}

    def _process_and_save(
        self,
        scan_ids: list[int],
        intervals: list[tuple[float, float]],
        bg_scan: Optional[int],
        avg_frames: int,
        sample_name: str,
        bg_scale: float = 1.0,
        base_dir: Optional[Path] = None,
    ) -> Path:
        try:
            hostname, username, beamline, proposal, visit, local_root = self._read_connection_fields(
                require_local_root=False
            )
        except ValueError as exc:
            raise RuntimeError(str(exc))

        if not local_root:
            local_root = Path.home() / ".mudpaw_cache"
            local_root.mkdir(parents=True, exist_ok=True)
        else:
            local_root = Path(local_root)
        if base_dir is None:
            base_dir = local_root

        all_fetch = list(scan_ids)
        if bg_scan is not None:
            all_fetch.append(int(bg_scan))
        local_azint = self._ensure_local_scans(
            all_fetch, hostname, username, beamline, int(proposal), int(visit), local_root
        )

        # Reference axes from first scan
        first_scan = self._load_full_scan(local_azint, scan_ids[0])
        azi = first_scan["azi"]
        q = first_scan["q"]

        bg_R = None
        bg_cake = None
        if bg_scan is not None:
            bg_data = self._load_full_scan(local_azint, int(bg_scan), ref_azi=azi, ref_q=q)
            bg_R = bg_data["R"].mean(axis=0)
            bg_cake = bg_data["cake"].mean(axis=0)
        scale = float(bg_scale or 1.0)

        rad_frames: list[np.ndarray] = []
        interval_frames: list[list[np.ndarray]] = [[] for _ in intervals]

        for sid in scan_ids:
            sdata = self._load_full_scan(local_azint, sid, ref_azi=azi, ref_q=q)
            R = sdata["R"]
            cake = sdata["cake"]
            if bg_R is not None:
                R = R - scale * bg_R[np.newaxis, :]
            rad_frames.append(R)
            for idx, (qmin, qmax) in enumerate(intervals):
                mask = (q >= qmin) & (q <= qmax)
                if not np.any(mask):
                    if q.size == 0:
                        continue
                    idx_min = int(np.clip(np.searchsorted(q, qmin, side="left"), 0, q.size - 1))
                    idx_max = int(np.clip(np.searchsorted(q, qmax, side="right"), idx_min + 1, q.size))
                    mask = np.zeros_like(q, dtype=bool)
                    mask[idx_min:idx_max] = True
                    print(
                        f"[list] Interval [{qmin}, {qmax}] not in q grid; "
                        f"using nearest bins [{q[idx_min]}, {q[idx_max-1]}]."
                    )
                seg = cake[:, :, mask].mean(axis=2)
                if bg_cake is not None:
                    bg_seg = bg_cake[:, mask].mean(axis=1)
                    seg = seg - scale * bg_seg
                interval_frames[idx].append(seg)

        rad_all = np.vstack(rad_frames)
        rad_avg = self._avg_every_n_rows(rad_all, avg_frames)

        interval_avg: list[np.ndarray] = []
        for buf in interval_frames:
            if not buf:
                interval_avg.append(np.empty((0, azi.size)))
            else:
                stacked = np.vstack(buf)
                interval_avg.append(self._avg_every_n_rows(stacked, avg_frames))

        q_mask = q > 0
        if not np.any(q_mask):
            q_mask = np.ones_like(q, dtype=bool)
        q_axis = q[q_mask]
        rad_out = np.hstack([q_axis.reshape(-1, 1), rad_avg[:, q_mask].T])

        first, last = scan_ids[0], scan_ids[-1]
        base_name = sample_name or f"scan_{first}-{last}"
        safe_name = self._sanitize_sample_name(base_name, fallback=f"scan_{first}-{last}")
        session_folder = f"{proposal}_{visit}"
        save_dir = (
            Path(base_dir)
            / beamline.lower()
            / session_folder
            / safe_name
            / "SAXS"
            / f"scan_{first}-{last}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        np.savetxt(
            save_dir / f"rad_{first}-{last}.txt",
            rad_out,
            delimiter="\t",
            header=self._frame_headers(rad_out.shape[1], "q"),
            comments="",
        )

        for idx, arr in enumerate(interval_avg, start=1):
            if arr.size == 0:
                continue
            out = np.hstack([azi.reshape(-1, 1), arr.T])
            np.savetxt(
                save_dir / f"azi_interval{idx}_{first}-{last}.dat",
                out,
                delimiter="\t",
                header=self._frame_headers(out.shape[1], "angle"),
                comments="",
            )
        return save_dir

    def _on_process_preview(self) -> None:
        panel = getattr(self, "azint_panel", None)
        if panel is None:
            messagebox.showerror("Process", "Preview has not been generated yet.", parent=self)
            return
        scan_ids = getattr(self, "last_scan_ids", []) or getattr(panel, "scan_ids", [])
        if not scan_ids:
            messagebox.showerror("Process", "No scan IDs found for preview.", parent=self)
            return
        intervals = panel._current_bounds() if hasattr(panel, "_current_bounds") else []
        if not intervals:
            intervals = [(panel.qmin_r, panel.qmax_r)] if hasattr(panel, "qmin_r") else []
        bg_scan = panel.bg_loaded_scan if getattr(panel, "bg_enable_var", tk.BooleanVar(value=False)).get() else None
        avg_frames = max(1, int(self.avg_frames_var.get() or 1))
        scale = panel.bg_scale_var.get() if hasattr(panel, "bg_scale_var") else 1.0
        sample_name = f"scan_{scan_ids[0]}-{scan_ids[-1]}"
        dest = filedialog.askdirectory(
            title="Select destination for processed preview",
            initialdir=self._default_browse_dir(),
        )
        if not dest:
            return
        try:
            save_dir = self._process_and_save(
                scan_ids, intervals, bg_scan, avg_frames, sample_name, bg_scale=scale, base_dir=Path(dest)
            )
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Process", f"Failed to process preview:\n{exc}", parent=self)
            return
        messagebox.showinfo("Process", f"Saved processed data to:\n{save_dir}", parent=self)

    def _on_process_list(self) -> None:
        if not self.embed_rows and self.embed_list_path:
            self._load_embed_list_from_path(self.embed_list_path, show_message=False)
        if not self.embed_rows:
            path = filedialog.askopenfilename(
                title="Load embed list CSV",
                initialdir=self._default_browse_dir(),
                filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            )
            if not path:
                return
            if not self._load_embed_list_from_path(path, show_message=False):
                return
            if not self.embed_rows:
                messagebox.showerror("Process", "Embed list is empty.", parent=self)
                return
        dest = filedialog.askdirectory(
            title="Select destination for processed list",
            initialdir=self._default_browse_dir(),
        )
        if not dest:
            return
        base_dir = Path(dest)
        avg_frames = max(1, int(self.avg_frames_var.get() or 1))
        successes = 0
        failures = []
        for row in self.embed_rows:
            scans_text = row.get("Scan interval", "")
            scan_ids = self._parse_scan_tokens(scans_text)
            if not scan_ids:
                failures.append(f"{row.get('Name','?')}: invalid scan interval")
                continue
            bg_raw = str(row.get("background", "")).strip()
            bg_scan = int(bg_raw) if bg_raw.isdigit() else None
            bg_scale_raw = row.get("bg_scale", "")
            try:
                bg_scale_val = float(bg_scale_raw)
            except Exception:
                bg_scale_val = 1.0
            intervals: list[tuple[float, float]] = []
            for key, val in row.items():
                if key is None:
                    continue
                key_norm = re.sub(r"[\\s_-]+", "", str(key).strip().lower())
                if (key_norm.startswith("qrange") or key_norm.startswith("interval")) and val:
                    nums = re.findall(r"[-+]?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][-+]?\\d+)?", str(val))
                    if len(nums) >= 2:
                        try:
                            intervals.append((float(nums[0]), float(nums[1])))
                        except Exception:
                            continue
            if not intervals and hasattr(self, "azint_panel"):
                intervals = self.azint_panel._current_bounds()
            if not intervals:
                intervals = []
            name = row.get("Name") or f"scan_{scan_ids[0]}-{scan_ids[-1]}"
            try:
                self._process_and_save(
                    scan_ids, intervals, bg_scan, avg_frames, name, bg_scale=bg_scale_val, base_dir=base_dir
                )
                successes += 1
            except Exception as exc:  # noqa: BLE001
                failures.append(f"{name}: {exc}")
        msg = f"Processed {successes} entr{'y' if successes==1 else 'ies'}."
        if failures:
            msg += "\nFailed:\n- " + "\n- ".join(failures)
        messagebox.showinfo("Process", msg, parent=self)

    def _load_state(self) -> None:
        """Restore last-used connection fields from disk."""
        if not self._state_file.exists():
            return
        try:
            data = json.loads(self._state_file.read_text())
        except Exception:
            return
        self.hostname_var.set(data.get("hostname", self.hostname_var.get()))
        self.username_var.set(data.get("username", ""))
        self.beamline_var.set(data.get("beamline", self.beamline_var.get()))
        self.proposal_var.set(str(data.get("proposal", "")))
        self.visit_var.set(str(data.get("visit", "")))
        self.local_root_var.set(data.get("local_root", ""))
        self.scan_no_var.set(str(data.get("scan_no", "")))
        self.cosaxs_config = data.get("cosaxs_config", self.cosaxs_config)
        self.formax_config = data.get("formax_config", self.formax_config)

    def _save_state(self) -> None:
        """Persist connection fields to disk."""
        data = {
            "hostname": self.hostname_var.get().strip(),
            "username": self.username_var.get().strip(),
            "beamline": self.beamline_var.get().strip(),
            "proposal": self.proposal_var.get().strip(),
            "visit": self.visit_var.get().strip(),
            "local_root": self.local_root_var.get().strip(),
            "scan_no": self.scan_no_var.get().strip(),
            "cosaxs_config": self.cosaxs_config,
            "formax_config": self.formax_config,
        }
        try:
            self._state_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    # Called by preview window to share current state
    def _set_preview_state(self, state: dict) -> None:
        self.preview_state = state

    def _current_intervals_from_rows(self) -> list[tuple[float, float]]:
        """Best-effort accessor for current interval bounds from the embedded preview panel."""
        panel = getattr(self, "azint_panel", None)
        if panel and hasattr(panel, "_current_bounds"):
            try:
                return list(panel._current_bounds())
            except Exception:
                return []
        return []


class AzintInteractiveWindow(tk.Frame):
    """Interactive azimuth/radial preview with interval controls."""

    def __init__(self, master: tk.Widget, owner: ConnectWindow, data: dict, local_azint: Path):
        super().__init__(master)
        self.master_ref = master
        self.connect_owner = owner
        self.local_azint = local_azint

        self.data_R = np.array(data["R"], copy=True)  # (scans, q)
        self.data_cake = np.array(data["cake"], copy=True)  # (scans, azi, q)
        self.azi = np.array(data["azi"], copy=True)
        self.q_plot = np.array(data["q_plot"], copy=True)
        self.scan_ids = list(data["scan_ids"])
        self.qmin_r = float(data["qmin_r"])
        self.qmax_r = float(data["qmax_r"])

        self.bg_image = None
        self.bg_R = None
        self.bg_cake = None
        self.bg_loaded_scan: Optional[int] = None

        self.interval_rows: list[dict] = []
        self.last_bounds: list[tuple[float, float]] = []

        self.bg_scan_var = tk.StringVar()
        self.bg_scale_var = tk.DoubleVar(value=1.0)
        self.bg_enable_var = tk.BooleanVar(value=False)
        self.interval_count_var = tk.IntVar(value=1)
        self.bg_status_var = tk.StringVar(value="No background loaded")
        self._clamp_guard = False
        self.last_scan_ids: list[int] = []

        self._build_ui()
        self._apply_interval_count()
        self._emit_state()

    def _build_ui(self) -> None:
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(1, weight=1)
        container.rowconfigure(0, weight=1)

        controls = ttk.Frame(container)
        controls.grid(row=0, column=0, sticky="nsw", padx=(0, 10))

        # Background controls
        bg_frame = ttk.LabelFrame(controls, text="Background")
        bg_frame.grid(row=0, column=0, sticky="w", pady=(0, 10))
        for c in range(3):
            bg_frame.columnconfigure(c, weight=1 if c == 1 else 0)

        ttk.Label(bg_frame, text="Scan #:").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(bg_frame, textvariable=self.bg_scan_var, width=10).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Button(bg_frame, text="Load", command=self._on_load_background).grid(
            row=0, column=2, padx=4, pady=4, sticky="w"
        )

        ttk.Label(bg_frame, text="Scale:").grid(row=1, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(bg_frame, textvariable=self.bg_scale_var, width=8).grid(
            row=1, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Scale(
            bg_frame,
            from_=0.0,
            to=2.0,
            orient="horizontal",
            variable=self.bg_scale_var,
            command=lambda _e=None: self._update_plots(),
        ).grid(row=1, column=2, padx=4, pady=4, sticky="ew")
        ttk.Checkbutton(
            bg_frame,
            text="Apply subtraction",
            variable=self.bg_enable_var,
            command=self._update_plots,
        ).grid(row=2, column=2, padx=4, pady=4, sticky="w")

        ttk.Label(bg_frame, textvariable=self.bg_status_var, foreground="gray").grid(
            row=3, column=0, columnspan=3, padx=4, pady=(0, 6), sticky="w"
        )

        # Interval controls
        interval_frame = ttk.LabelFrame(controls, text="Azimuthal intervals")
        interval_frame.grid(row=1, column=0, sticky="new")
        interval_frame.columnconfigure(1, weight=1)
        ttk.Label(interval_frame, text="Count (1-15):").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Entry(interval_frame, textvariable=self.interval_count_var, width=6).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Button(interval_frame, text="Apply", command=self._apply_interval_count).grid(
            row=0, column=2, padx=4, pady=4, sticky="w"
        )

        self.interval_rows_container = ttk.Frame(interval_frame)
        self.interval_rows_container.grid(row=1, column=0, columnspan=3, sticky="w", padx=4, pady=(4, 0))
        self.interval_rows_container.columnconfigure(1, weight=1)
        self.interval_rows_container.columnconfigure(3, weight=1)

        # Plot area (simple fill)
        self.plot_container = ttk.Frame(container)
        self.plot_container.grid(row=0, column=1, sticky="nsew")
        self.plot_container.rowconfigure(0, weight=1)
        self.plot_container.columnconfigure(0, weight=1)

        self.canvas = None
        self.fig = None
        self.ax_rad_img = None
        self.ax_rad_lines = None
        self.interval_axes = []
        self._build_plot_area()

    # -------------------- Background --------------------
    def _on_load_background(self) -> None:
        text = self.bg_scan_var.get().strip()
        if not text:
            messagebox.showerror("Background", "Enter a background scan number.", parent=self)
            return
        try:
            scan_id = int(text)
        except ValueError:
            messagebox.showerror("Background", "Background scan must be an integer.", parent=self)
            return
        try:
            scan = load_scan_arrays(self.local_azint, scan_id)
        except FileNotFoundError:
            # Try to fetch the background scan into the current cache
            owner = self.connect_owner if isinstance(self.connect_owner, ConnectWindow) else None
            if owner:
                try:
                    hostname, username, beamline, proposal, visit, local_root = owner._read_connection_fields(
                        require_local_root=False
                    )
                except Exception as exc:  # noqa: BLE001
                    messagebox.showerror(
                        "Background",
                        f"Background not found locally and cannot fetch because connection details are incomplete:\n{exc}",
                        parent=self,
                    )
                    return
                if not local_root:
                    # derive from current azint path
                    local_root = self.local_azint.parent.parent.parent.parent
                try:
                    maxiv_connect.rsync_scans(
                        hostname=hostname,
                        username=username,
                        beamline=beamline,
                        proposal=proposal,
                        visit=visit,
                        scan_ids=[scan_id],
                        local_root=local_root,
                        log_cb=owner._log_line,
                    )
                except Exception as exc:  # noqa: BLE001
                    messagebox.showerror(
                        "Background",
                        f"Background not found locally and download failed:\n{exc}",
                        parent=self,
                    )
                    return
                # Retry after fetch
                try:
                    scan = load_scan_arrays(self.local_azint, scan_id)
                except Exception as exc:  # noqa: BLE001
                    messagebox.showerror("Background", f"Failed to load background scan after download:\n{exc}", parent=self)
                    return
            else:
                messagebox.showerror("Background", "Failed to load background scan:\nIntegrated scan file not found locally.")
                return
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Background", f"Failed to load background scan:\n{exc}", parent=self)
            return

        azi = np.asarray(scan["azi"])
        q_plot = np.asarray(scan["q_plot"])
        if azi.shape[0] != self.azi.shape[0] or q_plot.shape[0] != self.q_plot.shape[0]:
            messagebox.showerror(
                "Background",
                "Background scan geometry does not match current data.",
                parent=self,
            )
            return

        cake = ConnectWindow._orient_cake(scan["cake"], self.azi, self.q_plot)
        R = ConnectWindow._orient_radial(scan["R"], self.q_plot)
        self.bg_cake = cake.mean(axis=0)  # (azi, q)
        self.bg_R = R.mean(axis=0)        # (q,)
        self.bg_loaded_scan = scan_id
        self.bg_status_var.set(f"Loaded scan {scan_id}")
        self._update_plots()

    # -------------------- Interval handling --------------------
    def _apply_interval_count(self) -> None:
        try:
            count = int(self.interval_count_var.get())
        except ValueError:
            messagebox.showerror("Intervals", "Interval count must be an integer.", parent=self)
            return
        # Capture current bounds so we can reuse min/max when rebuilding rows.
        if self.interval_rows:
            self.last_bounds = self._current_bounds()
        count = max(1, min(15, count))
        self.interval_count_var.set(count)
        self._build_interval_rows(count)
        self._build_plot_area()

    def _build_interval_rows(self, count: int) -> None:
        for child in self.interval_rows_container.winfo_children():
            child.destroy()
        self.interval_rows.clear()

        defaults = self._default_intervals(count)
        for idx, (qmin, qmax) in enumerate(defaults):
            row = idx
            lbl = ttk.Label(self.interval_rows_container, text=f"Interval {idx + 1}")
            lbl.grid(row=row, column=0, padx=4, pady=2, sticky="w")

            min_var = tk.DoubleVar(value=qmin)
            max_var = tk.DoubleVar(value=qmax)

            min_entry = ttk.Entry(self.interval_rows_container, textvariable=min_var, width=8)
            min_entry.grid(row=row, column=1, padx=4, pady=2, sticky="ew")
            max_entry = ttk.Entry(self.interval_rows_container, textvariable=max_var, width=8)
            max_entry.grid(row=row, column=3, padx=4, pady=2, sticky="ew")

            min_slider = ttk.Scale(
                self.interval_rows_container,
                from_=self.qmin_r,
                to=self.qmax_r,
                orient="horizontal",
                variable=min_var,
                command=lambda _e=None, mv=min_var, xv=max_var: self._clamp_interval(mv, xv),
            )
            min_slider.grid(row=row, column=2, padx=4, pady=2, sticky="ew")

            max_slider = ttk.Scale(
                self.interval_rows_container,
                from_=self.qmin_r,
                to=self.qmax_r,
                orient="horizontal",
                variable=max_var,
                command=lambda _e=None, mv=min_var, xv=max_var: self._clamp_interval(mv, xv),
            )
            max_slider.grid(row=row, column=4, padx=4, pady=2, sticky="ew")

            min_entry.bind("<FocusOut>", lambda _e, mv=min_var, xv=max_var: self._clamp_interval(mv, xv))
            max_entry.bind("<FocusOut>", lambda _e, mv=min_var, xv=max_var: self._clamp_interval(mv, xv))

            self.interval_rows.append(
                {"min": min_var, "max": max_var, "min_scale": min_slider, "max_scale": max_slider}
            )

    def _safe_float(self, value: Any, default: float = 0.0) -> float:
        """Best-effort float parser that tolerates malformed strings like '0.0.12'."""
        try:
            return float(value)
        except Exception:
            pass
        try:
            s = str(value)
            m = re.search(r"-?\d+(?:\.\d+)?", s)
            if m:
                return float(m.group(0))
        except Exception:
            pass
        return float(default)

    def _clamp_interval(self, min_var: tk.DoubleVar, max_var: tk.DoubleVar) -> None:
        if self._clamp_guard:
            return
        self._clamp_guard = True
        try:
            try:
                raw_min = min_var.get()
            except Exception:
                try:
                    raw_min = min_var._tk.globalgetvar(min_var._name)
                except Exception:
                    raw_min = ""
            try:
                raw_max = max_var.get()
            except Exception:
                try:
                    raw_max = max_var._tk.globalgetvar(max_var._name)
                except Exception:
                    raw_max = ""

            qmin = self._safe_float(raw_min, self.qmin_r)
            qmax = self._safe_float(raw_max, self.qmax_r)
            qmin = max(self.qmin_r, min(qmin, self.qmax_r))
            qmax = max(self.qmin_r, min(qmax, self.qmax_r))
            min_gap = max((self.qmax_r - self.qmin_r) * 0.01, 1e-6)
            if qmax < qmin + min_gap:
                qmax = min(self.qmax_r, qmin + min_gap)
            min_var.set(qmin)
            max_var.set(qmax)
            self._on_interval_change()
        finally:
            self._clamp_guard = False

    def _default_intervals(self, count: int) -> list[tuple[float, float]]:
        if self.last_bounds and len(self.last_bounds) == count:
            return self.last_bounds

        def _evenly_spaced(start: float, end: float, n: int) -> list[tuple[float, float]]:
            width = (end - start) / max(n, 1)
            return [(start + i * width, start + (i + 1) * width) for i in range(n)]

        if self.last_bounds:
            mins = [min(a, b) for a, b in self.last_bounds]
            maxs = [max(a, b) for a, b in self.last_bounds]
            low = max(self.qmin_r, min(mins))
            high = min(self.qmax_r, max(maxs))
            if high > low and count >= 1:
                return _evenly_spaced(low, high, count)

        span = self.qmax_r - self.qmin_r
        inner_start = self.qmin_r + span / 3.0
        inner_end = self.qmin_r + 2 * span / 3.0
        inner_span = max(inner_end - inner_start, span * 0.25)

        if count == 1:
            return [(inner_start, inner_end)]

        gap = 0.1 * inner_span / max(count - 1, 1)
        width = (inner_span - gap * (count - 1)) / count
        if width <= 0:
            width = inner_span / count
            gap = 0.0

        bounds = []
        pos = inner_start
        for _ in range(count):
            qmin = pos
            qmax = min(inner_end, qmin + width)
            bounds.append((qmin, qmax))
            pos += width + gap
        return bounds

    def _current_bounds(self) -> list[tuple[float, float]]:
        bounds: list[tuple[float, float]] = []
        for row in self.interval_rows:
            qmin = self._safe_float(row["min"].get(), self.qmin_r)
            qmax = self._safe_float(row["max"].get(), self.qmax_r)
            if qmax < qmin:
                qmin, qmax = qmax, qmin
            bounds.append((qmin, qmax))
        self.last_bounds = bounds
        return bounds

    def _on_interval_change(self) -> None:
        self._update_plots()

    # -------------------- Plot scaffolding --------------------
    def _build_plot_area(self) -> None:
        # Destroy old figure if any
        for child in self.plot_container.winfo_children():
            child.destroy()

        count = max(1, min(15, int(self.interval_count_var.get() or 1)))
        gs_cols = 1 + count  # radial lines + one azimuthal heatmap per interval
        self.fig = Figure(figsize=(3.0 * gs_cols, 3.2))
        self.ax_rad_img = None
        self.ax_rad_lines = self.fig.add_subplot(1, gs_cols, 1)

        self.interval_axes = []
        for i in range(count):
            ax_img = self.fig.add_subplot(1, gs_cols, 2 + i)
            self.interval_axes.append(ax_img)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_container)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.plot_container.rowconfigure(0, weight=1)
        self.plot_container.columnconfigure(0, weight=1)
        self._update_plots()

    # -------------------- Plotting --------------------
    def _get_active_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        R = np.array(self.data_R, copy=True)
        cake = np.array(self.data_cake, copy=True)

        if self.bg_enable_var.get() and self.bg_R is not None and self.bg_cake is not None:
            scale = float(self.bg_scale_var.get() or 1.0)
            R = R - scale * self.bg_R[np.newaxis, :]
            cake = cake - scale * self.bg_cake[np.newaxis, :, :]
        return R, cake

    def _update_plots(self) -> None:
        R, cake = self._get_active_arrays()
        bounds = self._current_bounds()
        expected_intervals = len(self.interval_axes)
        if len(bounds) < expected_intervals:
            # Pad with defaults so every interval axis gets data
            defaults = self._default_intervals(expected_intervals)
            bounds = (bounds + defaults)[:expected_intervals]
        colors = plt.cm.tab20.colors if len(bounds) <= 20 else plt.cm.nipy_spectral(np.linspace(0, 1, len(bounds)))

        frames = R.shape[0]
        # Derive q plotting range from finite, non-zero columns
        q_valid = self.q_plot > 0
        r_finite = np.any(np.isfinite(R) & (R != 0), axis=0)
        mask = q_valid & r_finite
        if np.any(mask):
            q_min_plot = float(np.nanmin(self.q_plot[mask]))
            q_max_plot = float(np.nanmax(self.q_plot[mask]))
        else:
            q_pos = self.q_plot[q_valid]
            q_min_plot = float(np.nanmin(q_pos)) if q_pos.size else float(self.q_plot[0])
            q_max_plot = float(np.nanmax(q_pos)) if q_pos.size else float(self.q_plot[-1])

        # Radial lines: q vs intensity per frame
        ax_lines = self.ax_rad_lines
        ax_lines.clear()
        for idx in range(frames):
            ax_lines.plot(self.q_plot, R[idx, :], color=colors[idx % len(colors)], alpha=0.8)
        ax_lines.set_xlabel("q [1/A]")
        ax_lines.set_ylabel("I [a.u.]")
        ax_lines.set_xscale("log")
        ax_lines.set_yscale("log")
        ax_lines.set_xlim(q_min_plot, q_max_plot)
        for color, (qmin, qmax) in zip(colors, bounds):
            ax_lines.axvline(qmin, color=color, ls="--", lw=1.0)
            ax_lines.axvline(qmax, color=color, ls="--", lw=1.0)

        # Interval heatmaps only (no 1D azimuthal lines)
        for ax_im, (qmin, qmax), color in zip(self.interval_axes, bounds, colors):
            mask = (self.q_plot >= qmin) & (self.q_plot <= qmax)
            ax_im.clear()
            if not np.any(mask):
                # Fallback: use nearest bins to requested bounds
                idx_min = int(np.clip(np.searchsorted(self.q_plot, qmin, side="left"), 0, self.q_plot.size - 1))
                idx_max = int(np.clip(np.searchsorted(self.q_plot, qmax, side="right"), 0, self.q_plot.size))
                if idx_max <= idx_min:
                    idx_max = min(self.q_plot.size, idx_min + 1)
                mask = np.zeros_like(self.q_plot, dtype=bool)
                mask[idx_min:idx_max] = True
                ax_im.set_title(f"Adjusted to nearest bins [{self.q_plot[idx_min]:.3g}, {self.q_plot[idx_max-1]:.3g}]")
            else:
                ax_im.set_title(f"Interval [{qmin:.4g}, {qmax:.4g}]")

            seg = cake[:, :, mask]  # (frames, azi, q)
            azi_image = seg.mean(axis=2)  # (frames, azi)
            ax_im.imshow(
                azi_image.T,
                extent=[0, frames, self.azi[0], self.azi[-1]],
                origin="lower",
                aspect="auto",
                cmap="viridis",
            )
            ax_im.set_ylabel("Theta [deg]")
            ax_im.set_xlabel("Frame #")

        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._emit_state()

    def _emit_state(self) -> None:
        state = {
            "scan_ids": getattr(self.connect_owner, "last_scan_ids", []),
            "intervals": self._current_bounds(),
            "background_scan": self.bg_loaded_scan,
        }
        self.connect_owner._set_preview_state(state)


class RawPreviewWindow(tk.Toplevel):
    """Interactive raw scattering preview with background subtraction."""

    def __init__(self, master: ConnectWindow, image: np.ndarray, local_raw: Path, loader):
        super().__init__(master)
        self.title("Raw scattering preview")
        self.minsize(720, 640)
        self.image = np.array(image, copy=True)
        self.local_raw = local_raw
        self.loader = loader
        self.bg_image = None

        self.bg_scan_var = tk.StringVar()
        self.bg_scale_var = tk.DoubleVar(value=1.0)
        self.bg_enable_var = tk.BooleanVar(value=False)
        self.bg_status_var = tk.StringVar(value="No background loaded")

        self._build_ui()
        self._update_plot()

    def _build_ui(self) -> None:
        container = ttk.Frame(self)
        container.grid(row=0, column=0, sticky="n", padx=10, pady=10)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        container.columnconfigure(0, weight=1)
        container.rowconfigure(1, weight=1)

        ctrl = ttk.LabelFrame(container, text="Background")
        ctrl.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        for c in range(4):
            ctrl.columnconfigure(c, weight=1 if c == 1 else 0)

        ttk.Label(ctrl, text="Scan #:").grid(row=0, column=0, padx=4, pady=4, sticky="n")
        ttk.Entry(ctrl, textvariable=self.bg_scan_var, width=10).grid(row=0, column=1, padx=4, pady=4, sticky="ew")
        ttk.Label(ctrl, text="Scale:").grid(row=0, column=2, padx=4, pady=4, sticky="n")
        ttk.Entry(ctrl, textvariable=self.bg_scale_var, width=8).grid(row=0, column=3, padx=4, pady=4, sticky="ew")
        ttk.Checkbutton(ctrl, text="Apply", variable=self.bg_enable_var, command=self._update_plot).grid(
            row=1, column=0, padx=4, pady=4, sticky="n"
        )
        ttk.Button(ctrl, text="Load background", command=self._on_load_background).grid(
            row=1, column=1, padx=4, pady=4, sticky="n"
        )
        ttk.Label(ctrl, textvariable=self.bg_status_var, foreground="gray").grid(
            row=1, column=2, columnspan=2, padx=4, pady=4, sticky="n"
        )

        plot_frame = ttk.Frame(container)
        plot_frame.grid(row=1, column=0, sticky="n")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.fig = Figure(figsize=(6, 5))
        self.ax = self.fig.add_subplot(1, 1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="n")

    def _on_load_background(self) -> None:
        text = self.bg_scan_var.get().strip()
        if not text:
            messagebox.showerror("Background", "Enter a background scan number.", parent=self)
            return
        try:
            scan_id = int(text)
        except ValueError:
            messagebox.showerror("Background", "Background scan must be an integer.", parent=self)
            return
        try:
            self.bg_image = self.loader(self.local_raw, scan_id)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Background", f"Failed to load background:\n{exc}", parent=self)
            return
        self.bg_loaded_scan = scan_id
        self.bg_status_var.set(f"Loaded scan {scan_id}")
        self._update_plot()

    def _update_plot(self) -> None:
        img = np.array(self.image, copy=True)
        if self.bg_enable_var.get() and self.bg_image is not None:
            scale = float(self.bg_scale_var.get() or 1.0)
            img = img - scale * self.bg_image
        vmin, vmax = self._auto_levels(img)
        self.ax.clear()
        self.ax.imshow(img, cmap="magma", vmin=vmin, vmax=vmax, origin="lower")
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.tight_layout()
        self.canvas.draw_idle()

    @staticmethod
    def _auto_levels(img: np.ndarray) -> tuple[float, float]:
        finite = img[np.isfinite(img)]
        if finite.size == 0:
            return 0.0, 1.0
        vmin = float(np.nanpercentile(finite, 1))
        vmax = float(np.nanpercentile(finite, 99))
        if vmax <= vmin:
            vmax = vmin + 1.0
        return vmin, vmax


def open_connect_window(master) -> ConnectWindow:
    return ConnectWindow(master)
