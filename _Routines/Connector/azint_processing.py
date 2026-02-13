from __future__ import annotations

from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Iterable, Optional, Tuple
import json

import numpy as np
import h5py
import pandas as pd


@dataclass
class AzintProcessConfig:
    empty_i0: float = 1.0
    empty_it: float = 1.0
    normalization: str = "transmittance"  # 'transmittance' | 'transmission' | 'none'
    bg_scan: Optional[int] = None
    bg_frames: Optional[Tuple[int, int]] = None
    scan_frames: Optional[Tuple[int, int]] = None

    frac: float = 1.0
    frac_r: float = 1.0
    frac_waxs: float = 1.0
    frac_waxs_r: float = 1.0

    # Default to a single SAXS azimuthal window; add more tuples to export multiple ranges.
    saxs_azi_windows: list[Tuple[float, float]] = field(
        default_factory=lambda: [(0.01, 0.10)]
    )
    saxs_rad_range: Tuple[float, float] = (0.01, 0.10)
    waxs_azi_windows: list[Tuple[float, float]] = field(
        default_factory=lambda: [(0.2, 0.4), (0.4, 0.6)]
    )
    waxs_rad_range: Tuple[float, float] = (0.2, 1.2)

    raw_dt_path: str = "entry/measurement/dt"
    raw_i0_path: str = "entry/measurement/I0"
    raw_it_path: str = "entry/measurement/It"

    saxs_integrated_suffix: str = "_eiger_integrated.h5"
    waxs_integrated_suffix: str = "_lambda_integrated.h5"

    output_format: str = "dat"  # azimuthal outputs
    radial_output_ext: str = "txt"  # radial outputs (customizable, e.g., 'txt' or 'dat')
    export_saxs_azi: bool = True
    export_saxs_rad: bool = True
    export_waxs_azi: bool = True
    export_waxs_rad: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "AzintProcessConfig":
        return cls(**{k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__})


def safe_divide(a, b, default=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        out = np.true_divide(a, b)
        if np.isscalar(out):
            return out if np.isfinite(out) else default
        out[~np.isfinite(out)] = default
        return out


def _read_dataset(f: h5py.File, path: str):
    if path in f:
        return np.array(f[path])
    return None


def _ensure_frames(arr: np.ndarray, target_frames: Optional[int] = None) -> np.ndarray:
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim == 2 and target_frames is not None and arr.shape[0] != target_frames:
        # assume (azi, q) or (q,) -> add frame axis
        arr = arr[None, :, :]
    if arr.ndim == 2:
        arr = arr[None, :, :]
    return arr


def load_saxs_azint(azint_file: Path):
    suffix = azint_file.suffix.lower()
    # Connector path: expect HDF5 integrated azimuthal data
    with h5py.File(azint_file, "r") as f:
        q = _read_dataset(f, "entry/radial/q")
        I1d = _read_dataset(f, "entry/radial/R")
        cake = _read_dataset(f, "entry/azimuthal/cake")
        azi = _read_dataset(f, "entry/azimuthal/azi")
        azint_norm = _read_dataset(f, "entry/azimuthal/norm")
        if q is None:
            q = _read_dataset(f, "entry/azint2d/data/radial_axis")
        if azi is None:
            azi = _read_dataset(f, "entry/azint2d/data/azimuthal_axis")
        if cake is None:
            cake = _read_dataset(f, "entry/azint2d/data/I")
        if I1d is None:
            I1d = _read_dataset(f, "entry/azint1d/data/I")
        if azint_norm is None:
            azint_norm = _read_dataset(f, "entry/azint2d/data/norm")
        # CoSAXS / data2d layout fallback
        if q is None:
            q = _read_dataset(f, "entry/data2d/q")
        if azi is None:
            azi = _read_dataset(f, "entry/data2d/azi")
        if cake is None:
            cake = _read_dataset(f, "entry/data2d/cake")
        if I1d is None:
            I1d = _read_dataset(f, "entry/data1d/I")

    if q is None or I1d is None or cake is None or azi is None:
        raise RuntimeError(f"Missing SAXS azint datasets in {azint_file}")
    if cake.ndim == 2:
        cake = cake[None, :, :]
    if I1d.ndim == 1:
        I1d = I1d[None, :]
    return q, I1d, cake, azi, azint_norm


def load_waxs_azint(azint_file: Path):
    return load_saxs_azint(azint_file)


def load_raw_signals(raw_master_file: Path, cfg: AzintProcessConfig):
    if raw_master_file is None or not raw_master_file.exists():
        return None, None, None
    with h5py.File(raw_master_file, "r") as f:
        dt = _read_dataset(f, cfg.raw_dt_path)
        i0 = _read_dataset(f, cfg.raw_i0_path)
        it = _read_dataset(f, cfg.raw_it_path)
    return dt, i0, it


def normalize_cakes(cake: np.ndarray, I1d: np.ndarray, azint_norm, cfg: AzintProcessConfig, dt, i0, it):
    frames = cake.shape[0]
    if dt is not None:
        dt = np.asarray(dt).reshape(-1)
    if i0 is not None:
        i0 = np.asarray(i0).reshape(-1)
    if it is not None:
        it = np.asarray(it).reshape(-1)

    if cfg.normalization.lower() in ("none", "off"):
        norm_factor = np.ones(frames)
    else:
        if dt is not None and i0 is not None:
            i0 = safe_divide(i0, dt, default=0.0)
        if dt is not None and it is not None:
            it = safe_divide(it, dt, default=0.0)
        trans = safe_divide(it, i0, default=0.0) if (i0 is not None and it is not None) else np.ones(frames)
        empty_trans = safe_divide(cfg.empty_it, cfg.empty_i0, default=1.0)
        norm_factor = safe_divide(empty_trans, trans, default=1.0)
        if np.size(norm_factor) == 1:
            norm_factor = np.full(frames, float(norm_factor))
        if len(norm_factor) != frames:
            norm_factor = np.resize(norm_factor, frames)

    cake_n = cake * norm_factor[:, None, None]
    I1d_n = I1d * norm_factor[:, None]

    if azint_norm is not None:
        az = np.asarray(azint_norm)
        if az.ndim == 1 and az.shape[0] == I1d_n.shape[1]:
            I1d_n = safe_divide(I1d_n, az[None, :], default=0.0)
            cake_n = safe_divide(cake_n, az[None, None, :], default=0.0)

    return norm_factor, cake_n, I1d_n


def _slice_frames(arr: np.ndarray, frame_range: Optional[Tuple[int, int]]):
    if frame_range is None:
        return arr
    start, end = frame_range
    start = max(0, int(start))
    end = int(end)
    if end < start:
        start, end = end, start
    return arr[start : end + 1]


def compute_background(bg_scan: int, cfg: AzintProcessConfig, azint_folder: Path, raw_folder: Path):
    saxs_file = azint_folder / f"scan-{bg_scan:04d}{cfg.saxs_integrated_suffix}"
    waxs_file = azint_folder / f"scan-{bg_scan:04d}{cfg.waxs_integrated_suffix}"

    if not saxs_file.exists():
        raise FileNotFoundError(f"Background SAXS azint file not found: {saxs_file}")

    q_s, I1d_s, cake_s, azi_s, norm_s = load_saxs_azint(saxs_file)
    raw_master = _find_raw_master(raw_folder, bg_scan)
    dt, i0, it = load_raw_signals(raw_master, cfg)
    _, cake_s, I1d_s = normalize_cakes(cake_s, I1d_s, norm_s, cfg, dt, i0, it)
    cake_s = _slice_frames(cake_s, cfg.bg_frames)
    I1d_s = _slice_frames(I1d_s, cfg.bg_frames)
    saxs_bg_2d = np.mean(cake_s, axis=0)
    saxs_bg_1d = np.mean(I1d_s, axis=0)

    waxs_bg_2d = None
    waxs_bg_1d = None
    q_w = None
    azi_w = None
    if waxs_file.exists():
        q_w, I1d_w, cake_w, azi_w, norm_w = load_waxs_azint(waxs_file)
        dtw, i0w, itw = load_raw_signals(raw_master, cfg)
        _, cake_w, I1d_w = normalize_cakes(cake_w, I1d_w, norm_w, cfg, dtw, i0w, itw)
        cake_w = _slice_frames(cake_w, cfg.bg_frames)
        I1d_w = _slice_frames(I1d_w, cfg.bg_frames)
        waxs_bg_2d = np.mean(cake_w, axis=0)
        waxs_bg_1d = np.mean(I1d_w, axis=0)

    return {
        "saxs": {"q": q_s, "azi": azi_s, "bg2d": saxs_bg_2d, "bg1d": saxs_bg_1d},
        "waxs": {"q": q_w, "azi": azi_w, "bg2d": waxs_bg_2d, "bg1d": waxs_bg_1d},
    }


def _find_raw_master(raw_folder: Path, scan_id: int) -> Optional[Path]:
    if raw_folder is None:
        return None
    candidates = [
        raw_folder / f"scan-{scan_id:04d}_eiger_master.h5",
        raw_folder / f"scan-{scan_id:04d}_master.h5",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def _mask_q(q: np.ndarray, qmin: float, qmax: float):
    qmin = float(qmin)
    qmax = float(qmax)
    if qmax < qmin:
        qmin, qmax = qmax, qmin
    mask = (q >= qmin) & (q <= qmax)
    if not np.any(mask):
        # fallback: nearest two points
        idx_min = int(np.clip(np.searchsorted(q, qmin, side="left"), 0, q.size - 1))
        idx_max = int(np.clip(np.searchsorted(q, qmax, side="right"), idx_min + 1, q.size))
        mask = np.zeros_like(q, dtype=bool)
        mask[idx_min:idx_max] = True
    return mask


def binq_azi(cake: np.ndarray, q: np.ndarray, qmin: float, qmax: float, bg2d: Optional[np.ndarray], frac: float):
    cake_adj = cake.copy()
    if bg2d is not None:
        cake_adj = cake_adj - frac * bg2d[None, :, :]
    mask = _mask_q(q, qmin, qmax)
    return np.mean(cake_adj[:, :, mask], axis=2)


def binq_rad(I1d: np.ndarray, q: np.ndarray, qmin: float, qmax: float, bg1d: Optional[np.ndarray], frac_r: float):
    I_adj = I1d.copy()
    if bg1d is not None:
        I_adj = I_adj - frac_r * bg1d[None, :]
    mask = _mask_q(q, qmin, qmax)
    return q[mask], I_adj[:, mask]


def write_azi_file(path: Path, azi: np.ndarray, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = data.shape[0]
    header = "azi" + "\t" + "\t".join([f"f{i}" for i in range(1, n_frames + 1)])
    out = np.column_stack([azi, data.T])
    np.savetxt(path, out, header=header, comments="")


def write_rad_file(path: Path, q: np.ndarray, data: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = data.shape[0]
    header = "q" + "\t" + "\t".join([f"f{i}" for i in range(1, n_frames + 1)])
    out = np.column_stack([q, data.T])
    np.savetxt(path, out, header=header, comments="")


def process_scan(
    scan_id: int,
    cfg: AzintProcessConfig,
    azint_folder: Path,
    raw_folder: Path,
    output_root: Path,
    bg_cache=None,
    flat_output: bool = False,
):
    saxs_file = azint_folder / f"scan-{scan_id:04d}{cfg.saxs_integrated_suffix}"
    if not saxs_file.exists():
        raise FileNotFoundError(f"SAXS azint file not found: {saxs_file}")

    q_s, I1d_s, cake_s, azi_s, norm_s = load_saxs_azint(saxs_file)
    raw_master = _find_raw_master(raw_folder, scan_id)
    dt, i0, it = load_raw_signals(raw_master, cfg)
    _, cake_s, I1d_s = normalize_cakes(cake_s, I1d_s, norm_s, cfg, dt, i0, it)
    cake_s = _slice_frames(cake_s, cfg.scan_frames)
    I1d_s = _slice_frames(I1d_s, cfg.scan_frames)

    waxs_data = None
    waxs_file = azint_folder / f"scan-{scan_id:04d}{cfg.waxs_integrated_suffix}"
    if waxs_file.exists():
        q_w, I1d_w, cake_w, azi_w, norm_w = load_waxs_azint(waxs_file)
        dtw, i0w, itw = load_raw_signals(raw_master, cfg)
        _, cake_w, I1d_w = normalize_cakes(cake_w, I1d_w, norm_w, cfg, dtw, i0w, itw)
        cake_w = _slice_frames(cake_w, cfg.scan_frames)
        I1d_w = _slice_frames(I1d_w, cfg.scan_frames)
        waxs_data = (q_w, I1d_w, cake_w, azi_w)

    bg = bg_cache
    if bg is None and cfg.bg_scan is not None:
        bg = compute_background(cfg.bg_scan, cfg, azint_folder, raw_folder)

    out_dir = output_root if flat_output else output_root / f"scan_{scan_id:04d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "scan_id": scan_id,
        "cfg": cfg.to_dict(),
        "bg_scan": cfg.bg_scan,
        "bg_frames": cfg.bg_frames,
        "scan_frames": cfg.scan_frames,
        "saxs_azi_windows": cfg.saxs_azi_windows,
        "saxs_rad_range": cfg.saxs_rad_range,
        "waxs_azi_windows": cfg.waxs_azi_windows,
        "waxs_rad_range": cfg.waxs_rad_range,
    }

    if cfg.export_saxs_azi:
        for idx, (qmin, qmax) in enumerate(cfg.saxs_azi_windows):
            data = binq_azi(
                cake_s,
                q_s,
                qmin,
                qmax,
                None if bg is None else bg["saxs"]["bg2d"],
                cfg.frac,
            )
            fname = out_dir / f"azi_saxs{idx}_scan{scan_id:04d}.{cfg.output_format}"
            write_azi_file(fname, azi_s, data)
    if cfg.export_saxs_rad:
        q_sub, data = binq_rad(
            I1d_s,
            q_s,
            cfg.saxs_rad_range[0],
            cfg.saxs_rad_range[1],
            None if bg is None else bg["saxs"]["bg1d"],
            cfg.frac_r,
        )
        ext = cfg.radial_output_ext.lstrip(".") if cfg.radial_output_ext else "dat"
        fname = out_dir / f"rad_saxs_scan{scan_id:04d}.{ext}"
        write_rad_file(fname, q_sub, data)

    if waxs_data and cfg.export_waxs_azi:
        q_w, I1d_w, cake_w, azi_w = waxs_data
        for idx, (qmin, qmax) in enumerate(cfg.waxs_azi_windows):
            data = binq_azi(
                cake_w,
                q_w,
                qmin,
                qmax,
                None if bg is None else bg["waxs"]["bg2d"],
                cfg.frac_waxs,
            )
            fname = out_dir / f"azi_waxs{idx}_scan{scan_id:04d}.{cfg.output_format}"
            write_azi_file(fname, azi_w, data)
    if waxs_data and cfg.export_waxs_rad:
        q_w, I1d_w, cake_w, azi_w = waxs_data
        q_sub, data = binq_rad(
            I1d_w,
            q_w,
            cfg.waxs_rad_range[0],
            cfg.waxs_rad_range[1],
            None if bg is None else bg["waxs"]["bg1d"],
            cfg.frac_waxs_r,
        )
        ext = cfg.radial_output_ext.lstrip(".") if cfg.radial_output_ext else "dat"
        fname = out_dir / f"rad_waxs_scan{scan_id:04d}.{ext}"
        write_rad_file(fname, q_sub, data)

    meta_path = out_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out_dir


def process_scans(scan_ids: Iterable[int], cfg: AzintProcessConfig, azint_folder: Path, raw_folder: Path, output_root: Path, log_cb=None, flat_output: bool = False):
    scan_ids = list(scan_ids)
    if not scan_ids:
        return [], []
    bg_cache = None
    if cfg.bg_scan is not None:
        if log_cb:
            log_cb(f"[Connector] Computing background from scan {cfg.bg_scan}")
        bg_cache = compute_background(cfg.bg_scan, cfg, azint_folder, raw_folder)

    failures = []
    outputs = []
    for sid in scan_ids:
        try:
            out_dir = process_scan(
                sid,
                cfg,
                azint_folder,
                raw_folder,
                output_root,
                bg_cache=bg_cache,
                flat_output=flat_output,
            )
            outputs.append(out_dir)
            if log_cb:
                log_cb(f"[Connector] Processed scan {sid} -> {out_dir}")
        except Exception as exc:
            failures.append((sid, str(exc)))
            if log_cb:
                log_cb(f"[Connector] Scan {sid} failed: {exc}")
    return outputs, failures
