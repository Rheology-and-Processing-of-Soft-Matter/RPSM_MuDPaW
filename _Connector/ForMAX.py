"""
Batch processor for ForMAX SAXS azint data.

Reads an embed/list CSV (columns: Name, Detail, Scan interval, background, q_range1...)
Downloads missing scans, applies optional background subtraction, averages frames,
and writes outputs to <local_root>/<beamline>/<proposal>_<visit>/<sample>/SAXS/scan_<first-last>/.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

from _Connector import maxiv_connect
from _Connector.connect_window import load_scan_arrays, ConnectWindow


def _parse_scan_tokens(text: str) -> List[int]:
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
                ids.extend(range(a_i, b_i + 1))
            except Exception:
                continue
        else:
            try:
                ids.append(int(tok))
            except Exception:
                continue
    return sorted(set(ids))

def _parse_range_value(val: str) -> tuple[float, float] | None:
    tokens = re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", str(val))
    if len(tokens) < 2:
        return None
    try:
        return float(tokens[0]), float(tokens[1])
    except Exception:
        return None


def _sanitize_sample_name(name: str, detail: str = "", fallback: str = "sample") -> str:
    raw = f"{(name or '').strip()}_{(detail or '').strip()}"
    s = re.sub(r"\\s+", "_", raw).strip("_")
    s = re.sub(r"[^A-Za-z0-9_.()+-]", "_", s)
    s = re.sub(r"_+", "_", s)
    return s or fallback


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


def _frame_headers(ncols: int, first_label: str) -> str:
    n_frames = max(0, ncols - 1)
    frames = [f"f{i}" for i in range(1, n_frames + 1)]
    return "\\t".join([first_label] + frames)


def _ensure_local(
    hostname: str,
    username: str,
    beamline: str,
    proposal: int,
    visit: int,
    scan_ids: Iterable[int],
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
        log_cb=None,
    )
    local_azint = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "azint"
    local_azint.mkdir(parents=True, exist_ok=True)
    return local_azint


def _load_full_scan(local_azint: Path, scan_id: int, ref_azi=None, ref_q=None) -> dict:
    scan = load_scan_arrays(local_azint, scan_id)
    azi = ref_azi if ref_azi is not None else np.asarray(scan["azi"])
    q_plot = ref_q if ref_q is not None else np.asarray(scan["q_plot"])
    R = ConnectWindow._orient_radial(scan["R"], q_plot)
    cake = ConnectWindow._orient_cake(scan["cake"], azi, q_plot)
    return {"azi": azi, "q": q_plot, "R": R, "cake": cake}


def process_entry(
    row: dict,
    args,
    avg_frames: int,
    bg_scale: float,  # kept for signature compatibility; UI typically provides frac directly
    ) -> Path:
        """
    Processes a user-defined range of SAXS scans and exports normalized,
    background-corrected radial and azimuthal intensity profiles.

    Workflow
    --------
    1. Data loading
       Loads azimuthally integrated 2D SAXS data ("cake") together with
       azimuthal weighting factors (azint2d norm) and the corresponding
       raw transmission signals (dt, i0, it) for each frame.

    2. Frame-wise normalization
       Applies per-frame normalization according to the UI-selected method:
           - transmission
           - transmittance
           - none
       using the provided empty_i0 and empty_it values.
       Normalization is applied directly to the 2D cake data.

    3. Background construction (optional)
       If a background scan is specified:
           - The background is constructed from ALL frames of that scan.
           - A 2D mean background (azi × q) is computed.
           - A 1D background (q) is computed via weighted azimuthal collapse.
       If no background scan is provided, no subtraction is performed
       (mathematically equivalent to frac = 0).

    4. Per-scan processing
       For each scan in the selected range:
           - The normalized cake is reduced to a 1D radial profile
             using azimuthal weighting over the full measured q range.
           - The scaled 1D background (if available) is subtracted
             after azimuthal integration.
           - For the UI-selected q-interval, 2D background subtraction
             is applied before q-integration to compute the azimuthal
             intensity profile (angle × frame).

    5. Frame aggregation
       Frames from all selected scans are concatenated and optionally
       averaged in blocks of size `avg_frames`.

    6. Outputs
       Writes to disk:
           - Full-q radial intensity matrix (q × frame_index)
           - Azimuthal intensity matrix for the selected q-interval
             (angle × frame_index)
    """

    # ---------------------------
    # helpers
    # ---------------------------
        def safe_divide(a, b):
            return np.divide(a, b, out=np.zeros_like(a, dtype=float), where=b != 0.0)

        def _compute_norm_factor(dt, i0, it, cfg):
            mode = cfg["normalization"]
            if mode == "transmittance":
                empty = cfg["empty_it"] / cfg["empty_i0"]
                return (it / i0) * (dt / empty)
            elif mode == "transmission":
                empty = cfg["empty_it"]
                return it * dt / empty
            elif mode == "none":
                return np.ones_like(it, dtype=float)
            raise RuntimeError(
                f"Unknown normalization: {mode}. Allowed: transmission, transmittance, none"
            )

        def _radial_from_cake(cake_faq, norm2d_aq):
            num = np.sum(cake_faq * norm2d_aq[None, :, :], axis=1)  # (F,Q)
            den = np.sum(norm2d_aq, axis=0)                         # (Q,)
            return safe_divide(num, den[None, :])

        def _azi_interval(cake_faq, q_axis, norm2d_aq, qmin, qmax, bg2d_aq=None, frac=1.0):
            a, b = np.searchsorted(q_axis, [qmin, qmax])
            sig = cake_faq[:, :, a:b]  # (F,A,Qsub)

            if bg2d_aq is not None:
                sig = sig - frac * bg2d_aq[None, :, a:b]

            w = norm2d_aq[:, a:b]  # (A,Qsub)
            num = np.sum(sig * w[None, :, :], axis=2)  # (F,A)
            den = np.sum(w, axis=1)                    # (A,)
            return safe_divide(num, den[None, :])

        def _avg_every_n_rows(arr, n):
            if n <= 1:
                return arr
            F = arr.shape[0]
            k = F // n
            if k <= 0:
                return arr
            trimmed = arr[: k * n, ...]
            new_shape = (k, n) + arr.shape[1:]
            return trimmed.reshape(new_shape).mean(axis=1)

        def _frame_headers(ncols: int, first_label: str) -> str:
            n_frames = max(0, ncols - 1)
            frames = [f"f{i}" for i in range(1, n_frames + 1)]
            return "\t".join([first_label] + frames)

    # ---------------------------
    # UI-driven inputs
    # ---------------------------
        scan_start = int(row["scan_start"])
        scan_end = int(row["scan_end"])
        if scan_end < scan_start:
            scan_start, scan_end = scan_end, scan_start
        scan_ids = list(range(scan_start, scan_end + 1))

        qmin = float(row["qmin"])
        qmax = float(row["qmax"])
        if qmax < qmin:
            qmin, qmax = qmax, qmin

        frac = float(row["frac"])
        frac_eff = (float(bg_scale) * frac) if bg_scale is not None else frac  # harmless if bg_scale==1

        norm_mode = str(row["normalization"]).strip().lower()
        cfg = {
            "empty_i0": float(row["empty_i0"]),
            "empty_it": float(row["empty_it"]),
            "normalization": norm_mode,
        }

        # Background scan is (optional))
        bg_raw = str(row.get("background", "")).strip()
        bg_scan = int(bg_raw) if bg_raw.isdigit() else None

        # ---------------------------
        # download/ensure local
        # ---------------------------
        hostname = args.hostname
        username = args.username
        beamline = args.beamline
        proposal = int(args.proposal)
        visit = int(args.visit)

        local_root = Path(args.local_root).expanduser()
        local_root.mkdir(parents=True, exist_ok=True)

        fetch_ids = list(scan_ids)
        if bg_scan is not None:
            fetch_ids.append(bg_scan)

        local_azint = _ensure_local(
            hostname=hostname,
            username=username,
            beamline=beamline,
            proposal=proposal,
            visit=visit,
            scan_ids=fetch_ids,
            local_root=local_root,
        )

        # ---------------------------
        # loader: must provide cake, azi, q_plot, norm2d/norm, dt, i0/i_0, it/i_t
        # ---------------------------
        def _load_full_scan_saxs(local_azint: Path, scan_id: int, ref_azi=None, ref_q=None) -> dict:
            scan = load_scan_arrays(local_azint, scan_id)

            azi = ref_azi if ref_azi is not None else np.asarray(scan["azi"])
            q_plot = ref_q if ref_q is not None else np.asarray(scan["q_plot"])

            cake = ConnectWindow._orient_cake(scan["cake"], azi, q_plot)  # (F,A,Q)

            norm2d = np.asarray(scan.get("norm2d", scan.get("norm", None)))
            if norm2d is None:
                raise KeyError(f"Scan {scan_id} missing azint2d norm weights (norm2d/norm).")
            if norm2d.ndim == 2:
                norm2d = ConnectWindow._orient_cake(norm2d[None, :, :], azi, q_plot)[0, :, :]

            dt = np.asarray(scan.get("dt", None))
            i0 = np.asarray(scan.get("i0", scan.get("i_0", None)))
            it = np.asarray(scan.get("it", scan.get("i_t", None)))
            if dt is None or i0 is None or it is None:
                raise KeyError(f"Scan {scan_id} missing dt/i0/it (needed for normalization).")

            return {"azi": azi, "q": q_plot, "cake": cake, "norm2d": norm2d, "dt": dt, "i0": i0, "it": it}

        # reference axes from first scan
        s0 = _load_full_scan_saxs(local_azint, scan_ids[0])
        azi = s0["azi"]
        q = s0["q"]

        # ---------------------------
        # optional background (ALL frames)
        # ---------------------------
        saxs_bg_2d = None
        saxs_bg_1d = None

        if bg_scan is not None:
            bg = _load_full_scan_saxs(local_azint, bg_scan, ref_azi=azi, ref_q=q)

            nf_bg = _compute_norm_factor(bg["dt"], bg["i0"], bg["it"], cfg)
            F_bg = min(bg["cake"].shape[0], len(nf_bg))
            bg_cake = bg["cake"][:F_bg] / nf_bg[:F_bg, None, None]

            saxs_bg_2d = bg_cake.mean(axis=0)  # (A,Q)
            saxs_bg_1d = _radial_from_cake(bg_cake, bg["norm2d"]).mean(axis=0)  # (Q,)

        # If no background, behave like frac=0
        if saxs_bg_1d is None:
            frac_eff = 0.0

        # ---------------------------
        # process scans: normalize -> radial full q -> subtract 1D bg; also azimuthal in [qmin,qmax]
        # ---------------------------
        rad_frames = []  # list of (F, Q)
        azi_frames = []  # list of (F, A)

        for sid in scan_ids:
            s = _load_full_scan_saxs(local_azint, sid, ref_azi=azi, ref_q=q)

            nf = _compute_norm_factor(s["dt"], s["i0"], s["it"], cfg)
            F = min(s["cake"].shape[0], len(nf))
            cake = s["cake"][:F] / nf[:F, None, None]  # normalized cake (F,A,Q)

            # radial (full q), subtract 1D bg after azimuthal collapse
            rad_full = _radial_from_cake(cake, s["norm2d"])
            if saxs_bg_1d is not None:
                rad_full = rad_full - frac_eff * saxs_bg_1d[None, :]
            rad_frames.append(rad_full)

            # azimuthal for UI-selected interval, subtract 2D bg before q-integration
            azi_prof = _azi_interval(
                cake_faq=cake,
                q_axis=q,
                norm2d_aq=s["norm2d"],
                qmin=qmin,
                qmax=qmax,
                bg2d_aq=saxs_bg_2d,
                frac=frac_eff,
            )  # (F,A)
            azi_frames.append(azi_prof)

        rad_all = np.vstack(rad_frames) if rad_frames else np.empty((0, 0))
        rad_avg = _avg_every_n_rows(rad_all, avg_frames)

        azi_all = np.vstack(azi_frames) if azi_frames else np.empty((0, 0))
        azi_avg = _avg_every_n_rows(azi_all, avg_frames)

        # ---------------------------
        # save outputs
        # ---------------------------
        first, last = scan_ids[0], scan_ids[-1]
        base_name = row.get("Name") or f"scan_{first}-{last}"
        detail = row.get("Detail", "")
        safe_name = _sanitize_sample_name(base_name, detail, fallback=f"scan_{first}-{last}")
        session_folder = f"{proposal}_{visit}"

        save_dir = (
            local_root
            / beamline.lower()
            / session_folder
            / safe_name
            / "SAXS"
            / f"scan_{first}-{last}"
        )
        save_dir.mkdir(parents=True, exist_ok=True)

        # radial full q (keep q>0 like your earlier batch code)
        if rad_avg.size:
            q_mask = q > 0
            if not np.any(q_mask):
                q_mask = np.ones_like(q, dtype=bool)
            q_axis = q[q_mask]
            rad_out = np.hstack([q_axis.reshape(-1, 1), rad_avg[:, q_mask].T])

            np.savetxt(
                save_dir / f"rad_{first}-{last}.txt",
                rad_out,
                delimiter="\t",
                header=_frame_headers(rad_out.shape[1], "q"),
                comments="",
            )

        # azimuthal angle × frame for the selected interval
        if azi_avg.size:
            azi_out = np.hstack([azi.reshape(-1, 1), azi_avg.T])
            np.savetxt(
                save_dir / f"azi_{first}-{last}.dat",
                azi_out,
                delimiter="\t",
                header=_frame_headers(azi_out.shape[1], "ang"),
                comments="",
            )

        return save_dir


def main():
    ap = argparse.ArgumentParser(description="ForMAX batch processor using embed list CSV.")
    ap.add_argument("--list", dest="list_path", default="embed_list.csv", help="Embed/list CSV file.")
    ap.add_argument("--hostname", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--beamline", default="ForMAX")
    ap.add_argument("--proposal", required=True)
    ap.add_argument("--visit", required=True)
    ap.add_argument("--local-root", default=str(Path.home() / ".mudpaw_cache"))
    ap.add_argument("--avg-frames", type=int, default=4, help="Average every N frames (default 4).")
    ap.add_argument("--bg-scale", type=float, default=1.0, help="Background scale factor.")
    args = ap.parse_args()

    rows: list[dict] = []
    with open(args.list_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row:
                rows.append(row)
    if not rows:
        print("No rows found in list.")
        return

    successes = 0
    failures: list[str] = []
    for row in rows:
        try:
            dest = process_entry(row, args, args.avg_frames, args.bg_scale)
            print(f"[OK] {row.get('Name','?')} -> {dest}")
            successes += 1
        except Exception as exc:
            failures.append(f"{row.get('Name','?')}: {exc}")
    print(f"Done. OK={successes}, ERR={len(failures)}")
    if failures:
        print("Failures:")
        for f in failures:
            print(" -", f)


if __name__ == "__main__":
    main()
