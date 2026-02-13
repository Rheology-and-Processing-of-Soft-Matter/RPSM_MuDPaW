from __future__ import annotations
import os
import sys
import math
import threading
import subprocess
import queue
import json
import shlex
import signal
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

BASE_DIR = Path(__file__).resolve().parent
SETTINGS_PATH = BASE_DIR / "_batch_window_settings.json"
DEFAULT_SAXS_SMOOTHING = 0.04
DEFAULT_SAXS_SIGMA = 0.5
DEFAULT_SAXS_TMIN = 2.0
DEFAULT_SAXS_TMAX = 180.0
DEFAULT_SAXS_METHOD = "fitting"
_SAXS_METHOD_CHOICES = {"fitting", "direct"}
DEFAULT_RHEO_STEADY = 10.0


class BatchProcessingWindow(tk.Toplevel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.title("Batch Processing — Orchestrator")
        self.geometry("980x600")
        self.minsize(820, 520)
        self._q = queue.Queue()
        self._stop_event = threading.Event()
        self._worker = None
        self._active_process = None

        # ---- state ----
        self.var_overfolder = tk.StringVar(value="")
        self.var_modal_saxs = tk.BooleanVar(value=True)
        self.var_modal_rheo = tk.BooleanVar(value=True)
        self.var_skip_fresh = tk.BooleanVar(value=True)
        self.var_dryrun = tk.BooleanVar(value=False)
        self.var_log_level = tk.StringVar(value="INFO")  # for display only
        self.var_dg_template = tk.StringVar(value="")
        self.var_dg_cli = tk.StringVar(value="/Applications/DataGraph.app/Contents/Library/dgraph")
        self.var_run_writer = tk.BooleanVar(value=False)
        self.var_single_file = tk.BooleanVar(value=False)
        self.var_workers = tk.IntVar(value=1)
        # Advanced processor params
        self.var_saxs_smoothing = tk.DoubleVar(value=DEFAULT_SAXS_SMOOTHING)
        self.var_saxs_sigma = tk.DoubleVar(value=DEFAULT_SAXS_SIGMA)
        self.var_saxs_tmin = tk.DoubleVar(value=DEFAULT_SAXS_TMIN)
        self.var_saxs_tmax = tk.DoubleVar(value=DEFAULT_SAXS_TMAX)
        self.var_saxs_method = tk.StringVar(value=DEFAULT_SAXS_METHOD)
        self.var_saxs_no_plots = tk.BooleanVar(value=True)
        self.var_saxs_mirror = tk.BooleanVar(value=False)
        self.var_rheo_mode = tk.StringVar(value="triggered")
        self.var_rheo_steady = tk.DoubleVar(value=DEFAULT_RHEO_STEADY)
        self.var_rheo_no_plots = tk.BooleanVar(value=True)

        self._load_settings()
        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _default_reference_dir(self) -> str:
        """Best-effort reference root for file dialogs.

        Prefer the app-wide last_folder.txt under _Miscell; otherwise fall back to the user's home.
        """
        candidates = [
            BASE_DIR.parent / "_Miscell" / "last_folder.txt",
            BASE_DIR.parent / "last_folder.txt",
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

    def _default_overfolder_dir(self) -> str:
        """Use current overfolder, else the parent of the reference folder, else home."""
        existing = self.var_overfolder.get().strip()
        if existing:
            return existing
        ref = self._default_reference_dir()
        try:
            return str(Path(ref).parent)
        except Exception:
            return ref

    # ---------------- UI ----------------
    def _build_ui(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True)
        main = ttk.Frame(container)
        main.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        side = ttk.Frame(container)
        side.pack(side=tk.RIGHT, fill=tk.Y, padx=(6, 6), pady=10)
        ttk.Button(side, text="Main menu", command=self._on_close).pack(fill=tk.X)

        top = ttk.Frame(main)
        top.pack(fill=tk.X, pady=(8, 6), padx=8)
        ttk.Label(top, text="Overfolder:").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.var_overfolder, width=80).grid(row=0, column=1, sticky="ew", padx=(6, 6))
        ttk.Button(top, text="Browse…", command=self._pick_overfolder).grid(row=0, column=2, padx=(0, 6))
        top.columnconfigure(1, weight=1)

        opts = ttk.Frame(main)
        opts.pack(fill=tk.X, pady=(0, 6), padx=8)
        ttk.Checkbutton(opts, text="SAXS", variable=self.var_modal_saxs).grid(row=0, column=0, padx=(0, 10), sticky="w")
        ttk.Checkbutton(opts, text="Rheology", variable=self.var_modal_rheo).grid(row=0, column=1, padx=(0, 10), sticky="w")
        ttk.Checkbutton(opts, text="Skip if fresh", variable=self.var_skip_fresh).grid(row=0, column=2, padx=(0, 10), sticky="w")
        ttk.Checkbutton(opts, text="Dry-run", variable=self.var_dryrun).grid(row=0, column=3, padx=(0, 10), sticky="w")
        ttk.Label(opts, text="Log level:").grid(row=0, column=4, padx=(8, 4), sticky="e")
        ttk.Combobox(opts, textvariable=self.var_log_level, values=["DEBUG", "INFO", "WARN", "ERROR"], width=8, state="readonly").grid(row=0, column=5, sticky="w")
        ttk.Label(opts, text="Workers:").grid(row=0, column=6, padx=(12, 4), sticky="e")
        tk.Spinbox(opts, from_=1, to=32, width=4, textvariable=self.var_workers).grid(row=0, column=7, sticky="w")

        # Advanced params
        adv = ttk.LabelFrame(main, text="Advanced — Processor parameters")
        adv.pack(fill=tk.X, padx=8, pady=(0, 8))
        sax = ttk.Frame(adv); sax.pack(fill=tk.X, padx=6, pady=(4, 2))
        ttk.Label(sax, text="SAXS:").pack(side=tk.LEFT)
        ttk.Label(sax, text="Smoothing").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(sax, width=7, textvariable=self.var_saxs_smoothing).pack(side=tk.LEFT)
        ttk.Label(sax, text="Sigma").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(sax, width=7, textvariable=self.var_saxs_sigma).pack(side=tk.LEFT)
        ttk.Label(sax, text="θ min").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(sax, width=7, textvariable=self.var_saxs_tmin).pack(side=tk.LEFT)
        ttk.Label(sax, text="θ max").pack(side=tk.LEFT, padx=(12, 4))
        ttk.Entry(sax, width=7, textvariable=self.var_saxs_tmax).pack(side=tk.LEFT)
        sax_opts = ttk.Frame(adv); sax_opts.pack(fill=tk.X, padx=6, pady=(0, 4))
        ttk.Label(sax_opts, text="SAXS method:").pack(side=tk.LEFT)
        ttk.Radiobutton(sax_opts, text="Fitting", value="fitting", variable=self.var_saxs_method).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Radiobutton(sax_opts, text="Direct", value="direct", variable=self.var_saxs_method).pack(side=tk.LEFT, padx=(6, 0))
        ttk.Checkbutton(sax_opts, text="No plots", variable=self.var_saxs_no_plots).pack(side=tk.LEFT, padx=(12, 0))
        ttk.Checkbutton(sax_opts, text="Mirror", variable=self.var_saxs_mirror).pack(side=tk.LEFT, padx=(12, 0))

        rhe = ttk.Frame(adv); rhe.pack(fill=tk.X, padx=6, pady=(2, 4))
        ttk.Label(rhe, text="Rheology:").pack(side=tk.LEFT)
        ttk.Label(rhe, text="Mode").pack(side=tk.LEFT, padx=(12, 4))
        self._rheo_mode_cb = ttk.Combobox(rhe, width=14, state="readonly",
                                          textvariable=self.var_rheo_mode,
                                          values=["triggered", "nontriggered", "other"])
        self._rheo_mode_cb.pack(side=tk.LEFT)
        ttk.Label(rhe, text="Steady window (s)").pack(side=tk.LEFT, padx=(12, 4))
        self._rheo_steady_entry = ttk.Entry(rhe, width=8, textvariable=self.var_rheo_steady)
        self._rheo_steady_entry.pack(side=tk.LEFT)
        ttk.Checkbutton(rhe, text="No plots", variable=self.var_rheo_no_plots).pack(side=tk.LEFT, padx=(12, 0))

        def _toggle_steady(*_):
            self._rheo_steady_entry.configure(
                state=("normal" if self.var_rheo_mode.get() == "nontriggered" else "disabled"))
        _toggle_steady()
        self._rheo_mode_cb.bind("<<ComboboxSelected>>", _toggle_steady)

        # DataGraph options
        dg = ttk.LabelFrame(main, text="DataGraph Export (optional)")
        dg.pack(fill=tk.X, padx=8, pady=(0, 8))
        row1 = ttk.Frame(dg)
        row1.pack(fill=tk.X, pady=(4, 2))
        ttk.Label(row1, text="Template (.dgraph):").pack(side=tk.LEFT)
        ttk.Entry(row1, textvariable=self.var_dg_template).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        ttk.Button(row1, text="Browse…", command=self._pick_dg_template).pack(side=tk.LEFT)
        row2 = ttk.Frame(dg)
        row2.pack(fill=tk.X, pady=(2, 6))
        ttk.Label(row2, text="DG CLI:").pack(side=tk.LEFT)
        ttk.Entry(row2, textvariable=self.var_dg_cli, width=60).pack(side=tk.LEFT, padx=6)
        ttk.Checkbutton(row2, text="Write to DataGraph", variable=self.var_run_writer).pack(side=tk.LEFT, padx=(12, 0))
        row3 = ttk.Frame(dg)
        row3.pack(fill=tk.X, pady=(0, 8))
        ttk.Checkbutton(row3, text="Single file (combine all processed data into one export)", variable=self.var_single_file).pack(side=tk.LEFT, padx=(24, 0))

        # Control bar
        bar = ttk.Frame(main)
        bar.pack(fill=tk.X, padx=8, pady=(0, 6))
        ttk.Button(bar, text="Engage", command=self._on_engage).pack(side=tk.RIGHT)
        self.stop_button = ttk.Button(bar, text="Stop", command=self._on_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(bar, text="Save flagged graphs", command=self._on_save_flagged_graphs).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(bar, text="Write DataGraph only", command=self._on_write_dg_only).pack(side=tk.RIGHT, padx=(0, 6))
        ttk.Button(bar, text="Close", command=self._on_close).pack(side=tk.RIGHT, padx=(0, 6))

        # Log view
        self.txt = tk.Text(main, height=20, wrap="word")
        self.txt.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))
        self._log("Welcome to MuDPaW — Batch Processing.")
        self._log("Ready. Pick an overfolder and press 'Engage'. Uses orchestrate_overfolder.py.")
        self.after(100, self._drain_queue)

    # ---------------- actions ----------------
    def _pick_overfolder(self):
        p = filedialog.askdirectory(
            title="Select Overfolder",
            initialdir=self._default_overfolder_dir(),
        )
        if p:
            self.var_overfolder.set(p)

    def _pick_dg_template(self):
        """Choose a DataGraph template (file or bundle) with macOS-friendly fallbacks.
        1) Try an unrestricted file picker so nothing is greyed out.
        2) If user cancels or the selection is a bundle that cannot be chosen, try a directory picker.
        3) Accept either a regular file ending in .dgraph or a folder whose name ends with .dgraph.
        """
        try:
            initdir = str(Path(self.var_dg_template.get()).expanduser().resolve().parent)
        except Exception:
            initdir = self._default_reference_dir()

        # Step 1: Unrestricted file chooser (no filetype filter to avoid macOS bundle filtering)
        p = filedialog.askopenfilename(parent=self,
                                       title="Choose DataGraph template (.dgraph file or bundle)",
                                       initialdir=initdir)
        if p:
            sel = Path(p)
            # If user picked a .dgraph file (some installs package as a single file), accept directly
            if sel.suffix.lower() == ".dgraph" and sel.is_file():
                self.var_dg_template.set(str(sel))
                return
            # If user navigated *into* a bundle and picked a file inside, accept the bundle root if discoverable
            if ".dgraph" in str(sel):
                # walk up until a parent endswith .dgraph
                for ancestor in [sel] + list(sel.parents):
                    if str(ancestor).lower().endswith(".dgraph") and ancestor.exists():
                        self.var_dg_template.set(str(ancestor))
                        return

        # Step 2: Directory chooser as fallback for bundle selection
        d = filedialog.askdirectory(parent=self,
                                    title="Choose DataGraph template (.dgraph bundle)",
                                    initialdir=initdir)
        if d:
            dp = Path(d)
            if dp.suffix.lower() == ".dgraph" and dp.is_dir():
                self.var_dg_template.set(str(dp))
                return
            else:
                messagebox.showerror(
                    "Invalid selection",
                    "Please choose a DataGraph template (.dgraph file or .dgraph bundle)."
                )

    def _on_write_dg_only(self):
        over = self.var_overfolder.get().strip()
        if not over:
            messagebox.showerror("Batch", "Please choose an overfolder first.")
            return
        tpl = self.var_dg_template.get().strip()
        if not tpl:
            messagebox.showerror("Batch", "Provide a .dgraph template to run the DataGraph writer.")
            return
        self._save_settings()
        orch = BASE_DIR / "orchestrate_overfolder.py"
        if not orch.exists():
            messagebox.showerror("Batch", f"Missing orchestrator:\n{orch}")
            return

        cmd = [sys.executable, str(orch), over, "--modalities", "none", "--write-dg", "--dg-template", tpl]
        cli = self.var_dg_cli.get().strip()
        if cli:
            cmd += ["--dg-cli", cli]
        if self.var_dryrun.get():
            cmd += ["--dry-run"]
        if not self.var_skip_fresh.get():
            cmd += ["--no-skip-fresh"]

        self._log(f"$ {' '.join(cmd)}")
        self._start_worker(cmd)

    def _on_save_flagged_graphs(self):
        over = self.var_overfolder.get().strip()
        if not over:
            messagebox.showerror("Batch", "Please choose an overfolder first.")
            return
        cli = self.var_dg_cli.get().strip()
        if not cli:
            messagebox.showerror("Batch", "Provide the DataGraph CLI path first.")
            return
        helper = BASE_DIR / "save_flagged_graphs.py"
        if not helper.exists():
            messagebox.showerror("Batch", f"Missing helper script:\n{helper}")
            return
        self._save_settings()
        cmd = [sys.executable, str(helper), over, "--dg-cli", cli]
        self._log(f"$ {shlex.join(cmd)}")
        self._start_worker(cmd)

    def _on_engage(self):
        over = self.var_overfolder.get().strip()
        if not over:
            messagebox.showerror("Batch", "Please choose an overfolder first.")
            return
        self._save_settings()
        orch = BASE_DIR / "orchestrate_overfolder.py"
        if not orch.exists():
            messagebox.showerror("Batch", f"Missing orchestrator:\n{orch}")
            return

        cmd = [sys.executable, str(orch), over]
        if self.var_modal_saxs.get() and self.var_modal_rheo.get():
            modalities = "both"
        elif self.var_modal_saxs.get():
            modalities = "saxs"
        elif self.var_modal_rheo.get():
            modalities = "rheology"
        else:
            modalities = "none"

        if modalities == "none" and not self.var_run_writer.get():
            messagebox.showerror("Batch", "Select at least one modality or enable DataGraph writing.")
            return

        cmd += ["--modalities", modalities]

        if self.var_dryrun.get():
            cmd += ["--dry-run"]
        if not self.var_skip_fresh.get():
            cmd += ["--no-skip-fresh"]

        cmd += ["--saxs-smoothing", str(self.var_saxs_smoothing.get()),
                "--saxs-sigma", str(self.var_saxs_sigma.get()),
                "--saxs-theta-min", str(self.var_saxs_tmin.get()),
                "--saxs-theta-max", str(self.var_saxs_tmax.get()),
                "--saxs-method", self._get_saxs_method()]
        if self.var_saxs_no_plots.get():
            cmd += ["--saxs-no-plots"]
        if self.var_saxs_mirror.get():
            cmd += ["--saxs-mirror"]
        cmd += ["--rheo-mode", self.var_rheo_mode.get(),
                "--rheo-steady-window", str(self.var_rheo_steady.get())]
        if self.var_rheo_no_plots.get():
            cmd += ["--rheo-no-plots"]

        if self.var_run_writer.get() or self.var_single_file.get():
            tpl = self.var_dg_template.get().strip()
            if not tpl:
                messagebox.showerror("Batch", "Provide a .dgraph template to enable DataGraph writing.")
                return
            cmd += ["--dg-template", tpl]
        if self.var_run_writer.get() or self.var_single_file.get():
            cli = self.var_dg_cli.get().strip()
            if cli:
                cmd += ["--dg-cli", cli]
        if self.var_run_writer.get():
            cmd.append("--write-dg")
        if self.var_single_file.get():
            cmd.append("--dg-single-file")

        workers = max(1, int(self.var_workers.get() or 1))
        cmd += ["--workers", str(workers)]

        self._log(f"$ {' '.join(cmd)}")
        self._start_worker(cmd)

    def _start_worker(self, cmd):
        if self._worker and self._worker.is_alive():
            messagebox.showinfo("Batch", "A run is already in progress. Please stop it before starting a new one.")
            return
        self._stop_event.clear()
        self.stop_button.config(state=tk.NORMAL)
        self._worker = threading.Thread(target=self._run_subprocess, args=(cmd,), daemon=True)
        self._worker.start()

    def _on_stop(self):
        if self._worker and self._worker.is_alive():
            self._stop_event.set()
            proc = self._active_process
            if proc and proc.poll() is None:
                try:
                    if hasattr(os, "getpgid"):
                        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                        self._log("[stop] Sent SIGTERM to subprocess group. Waiting for exit…")
                    else:
                        raise OSError("process groups not supported")
                except Exception:
                    try:
                        proc.terminate()
                        self._log("[stop] Terminated subprocess. Waiting for it to exit…")
                    except Exception as exc:
                        self._log(f"[stop] Failed to terminate subprocess: {exc}")
            else:
                self._log("[stop] Requested stop… waiting for current subprocess to exit.")
        else:
            self._stop_event.clear()
            self.stop_button.config(state=tk.DISABLED)

    # ---------------- plumbing ----------------
    def _run_subprocess(self, cmd):
        proc: subprocess.Popen[str] | None = None
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(BASE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                start_new_session=True,
            )
            self._active_process = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                self._q.put(line.rstrip("\n"))
            proc.wait()
            stamp = datetime.now().strftime('%H:%M:%S')
            self._q.put(f"[{stamp}] (exit code {proc.returncode})")
        except FileNotFoundError:
            self._q.put("ERROR: Python or script not found.")
        except Exception as e:
            self._q.put(f"ERROR: {e}")
        finally:
            if proc and proc.stdout:
                proc.stdout.close()
            self._active_process = None
            self._worker = None
            self._q.put({"event": "worker_done"})

    def _drain_queue(self):
        try:
            while True:
                payload = self._q.get_nowait()
                if isinstance(payload, dict):
                    self._handle_queue_event(payload)
                else:
                    self._log(payload)
        except queue.Empty:
            pass
        self.after(100, self._drain_queue)

    def _handle_queue_event(self, payload: dict):
        if payload.get("event") == "worker_done":
            self.stop_button.config(state=tk.DISABLED)
            self._stop_event.clear()

    def _log(self, msg: str):
        self.txt.insert(tk.END, f"{msg}\n")
        self.txt.see(tk.END)
        if self.parent and hasattr(self.parent, "log_text"):
            try:
                parent_log = self.parent.log_text
                parent_log.config(state=tk.NORMAL)
                parent_log.insert(tk.END, f"[Batch] {msg}\n")
                parent_log.see(tk.END)
                parent_log.config(state=tk.DISABLED)
            except Exception:
                pass

    # ---------------- persistence ----------------
    def _load_settings(self):
        if not SETTINGS_PATH.exists():
            return
        try:
            data = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            return

        def _bool(val, default=False):
            if isinstance(val, bool):
                return val
            if isinstance(val, (int, float)):
                return bool(val)
            if isinstance(val, str):
                return val.strip().lower() in ("1", "true", "yes", "on")
            return default

        def _float(val, default):
            try:
                return float(val)
            except Exception:
                return default
        def _int(val, default):
            try:
                iv = int(val)
                return iv if iv >= 1 else default
            except Exception:
                return default
        def _method(val, default):
            try:
                txt = str(val).strip().lower()
            except Exception:
                txt = ""
            return txt if txt in _SAXS_METHOD_CHOICES else default

        self.var_overfolder.set(data.get("overfolder", self.var_overfolder.get()))
        self.var_modal_saxs.set(_bool(data.get("modal_saxs"), self.var_modal_saxs.get()))
        self.var_modal_rheo.set(_bool(data.get("modal_rheo"), self.var_modal_rheo.get()))
        self.var_skip_fresh.set(_bool(data.get("skip_fresh"), self.var_skip_fresh.get()))
        self.var_dryrun.set(_bool(data.get("dryrun"), self.var_dryrun.get()))
        self.var_log_level.set(data.get("log_level", self.var_log_level.get()))
        self.var_dg_template.set(data.get("dg_template", self.var_dg_template.get()))
        self.var_dg_cli.set(data.get("dg_cli", self.var_dg_cli.get()))
        self.var_run_writer.set(_bool(data.get("run_writer"), self.var_run_writer.get()))
        self.var_single_file.set(_bool(data.get("single_file"), self.var_single_file.get()))
        self.var_workers.set(_int(data.get("workers"), self.var_workers.get()))
        self.var_saxs_smoothing.set(_float(data.get("saxs_smoothing"), self.var_saxs_smoothing.get()))
        self.var_saxs_sigma.set(_float(data.get("saxs_sigma"), self.var_saxs_sigma.get()))
        self.var_saxs_tmin.set(_float(data.get("saxs_tmin"), self.var_saxs_tmin.get()))
        self.var_saxs_tmax.set(_float(data.get("saxs_tmax"), self.var_saxs_tmax.get()))
        self.var_saxs_method.set(_method(data.get("saxs_method"), self.var_saxs_method.get()))
        self.var_saxs_no_plots.set(_bool(data.get("saxs_no_plots"), self.var_saxs_no_plots.get()))
        self.var_saxs_mirror.set(_bool(data.get("saxs_mirror"), self.var_saxs_mirror.get()))
        self.var_rheo_mode.set(data.get("rheo_mode", self.var_rheo_mode.get()))
        self.var_rheo_steady.set(_float(data.get("rheo_steady"), self.var_rheo_steady.get()))
        self.var_rheo_no_plots.set(_bool(data.get("rheo_no_plots"), self.var_rheo_no_plots.get()))

    def _save_settings(self):
        data = {
            "overfolder": self.var_overfolder.get().strip(),
            "modal_saxs": bool(self.var_modal_saxs.get()),
            "modal_rheo": bool(self.var_modal_rheo.get()),
            "skip_fresh": bool(self.var_skip_fresh.get()),
            "dryrun": bool(self.var_dryrun.get()),
            "log_level": self.var_log_level.get(),
            "dg_template": self.var_dg_template.get().strip(),
            "dg_cli": self.var_dg_cli.get().strip(),
            "run_writer": bool(self.var_run_writer.get()),
            "single_file": bool(self.var_single_file.get()),
            "workers": max(1, int(self.var_workers.get() or 1)),
            "saxs_smoothing": self._float_from_var(self.var_saxs_smoothing, DEFAULT_SAXS_SMOOTHING),
            "saxs_sigma": self._float_from_var(self.var_saxs_sigma, DEFAULT_SAXS_SIGMA),
            "saxs_tmin": self._float_from_var(self.var_saxs_tmin, DEFAULT_SAXS_TMIN),
            "saxs_tmax": self._float_from_var(self.var_saxs_tmax, DEFAULT_SAXS_TMAX),
            "saxs_method": self._get_saxs_method(),
            "saxs_no_plots": bool(self.var_saxs_no_plots.get()),
            "saxs_mirror": bool(self.var_saxs_mirror.get()),
            "rheo_mode": self.var_rheo_mode.get(),
            "rheo_steady": self._float_from_var(self.var_rheo_steady, DEFAULT_RHEO_STEADY),
            "rheo_no_plots": bool(self.var_rheo_no_plots.get()),
        }
        try:
            SETTINGS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            print(f"[BatchWindow] Failed to save settings: {e}")

    def _float_from_var(self, tk_var, fallback):
        """Best-effort float extraction that keeps the UI responsive even if the entry is blank."""
        try:
            value = float(tk_var.get())
            if not math.isfinite(value):
                raise ValueError
            return value
        except Exception:
            try:
                tk_var.set(str(fallback))
            except Exception:
                pass
            return fallback

    def _get_saxs_method(self) -> str:
        """Return the selected SAXS HoP method, falling back to the default if invalid."""
        try:
            choice = str(self.var_saxs_method.get()).strip().lower()
        except Exception:
            choice = ""
        if choice not in _SAXS_METHOD_CHOICES:
            choice = DEFAULT_SAXS_METHOD
            try:
                self.var_saxs_method.set(choice)
            except Exception:
                pass
        return choice

    def _on_close(self):
        self._save_settings()
        self.destroy()
