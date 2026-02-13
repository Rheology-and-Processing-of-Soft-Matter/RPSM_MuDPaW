"""Signal reconstruction utilities for azimuthal SAXS traces.

Implements a masked Fourier fit that respects nematic symmetry (π-periodic)
and can be used to fill missing angular sectors caused by beamstops.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class ReconstructionResult:
    phi: np.ndarray  # (N,) radians
    I_original: np.ndarray  # (N,)
    mask: np.ndarray  # (N,), True = valid
    I_reconstructed: np.ndarray  # (N,) model over full range
    I_filled: np.ndarray  # (N,) original with gaps filled
    hop: float | None  # Hermans parameter (optional)
    hwhm_deg: float | None  # HWHM of main peak (optional)


def _validate_even_order(max_order: int) -> int:
    if max_order < 2:
        return 2
    if max_order % 2:
        return max_order + 1
    return max_order


def _compute_hermans(phi: np.ndarray, intensity: np.ndarray) -> float | None:
    """Estimate Hermans orientation parameter using the reconstructed profile."""
    valid = np.isfinite(phi) & np.isfinite(intensity)
    if not np.any(valid):
        return None
    phi_v = phi[valid]
    I_v = np.clip(intensity[valid], a_min=0.0, a_max=None)
    area = np.trapz(I_v, phi_v)
    if area <= 0:
        return None
    pdf = I_v / area
    cos2 = np.cos(phi_v) ** 2
    hop = (3 * np.trapz(pdf * cos2, phi_v) - 1) / 2
    return float(hop)


def _compute_hwhm_deg(phi: np.ndarray, intensity: np.ndarray) -> float | None:
    """Half width at half max around the dominant peak (degrees)."""
    valid = np.isfinite(phi) & np.isfinite(intensity)
    if np.count_nonzero(valid) < 2:
        return None
    phi_v = phi[valid]
    I_v = intensity[valid]
    peak_idx = int(np.argmax(I_v))
    peak_val = I_v[peak_idx]
    if peak_val <= 0:
        return None
    half = 0.5 * peak_val

    def _interp_left(idx: int) -> float | None:
        for i in range(idx - 1, -1, -1):
            y1, y2 = I_v[i], I_v[i + 1]
            if (y1 - half) == 0:
                return phi_v[i]
            if (y1 - half) * (y2 - half) <= 0:
                t = (half - y1) / (y2 - y1 + 1e-12)
                return float(phi_v[i] + t * (phi_v[i + 1] - phi_v[i]))
        return None

    def _interp_right(idx: int) -> float | None:
        for i in range(idx, len(I_v) - 1):
            y1, y2 = I_v[i], I_v[i + 1]
            if (y1 - half) == 0:
                return phi_v[i]
            if (y1 - half) * (y2 - half) <= 0:
                t = (half - y1) / (y2 - y1 + 1e-12)
                return float(phi_v[i] + t * (phi_v[i + 1] - phi_v[i]))
        return None

    left = _interp_left(peak_idx)
    right = _interp_right(peak_idx)
    if left is None or right is None:
        return None
    return float(np.degrees(0.5 * ((phi_v[peak_idx] - left) + (right - phi_v[peak_idx]))))


def reconstruct_azimuthal(
    phi: np.ndarray,
    I: np.ndarray,
    mask: np.ndarray | None = None,
    symmetry: str = "nematic",
    max_order: int = 4,
    ridge_lambda: float = 0.02,
    edge_emphasis: bool = True,
    compute_hop: bool = True,
    compute_hwhm: bool = True,
) -> ReconstructionResult:
    """
    Reconstruct azimuthal intensity in masked regions using a symmetry-aware
    Fourier fit.
    """
    phi = np.asarray(phi, dtype=float).reshape(-1)
    I = np.asarray(I, dtype=float).reshape(-1)
    if phi.shape != I.shape:
        raise ValueError("phi and I must have identical shapes.")
    if mask is None:
        mask = np.ones_like(phi, dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool).reshape(-1)
        if mask.shape != phi.shape:
            raise ValueError("mask must be the same shape as phi/I.")

    if symmetry not in {"nematic", "isotropic"}:
        raise ValueError(f"Unsupported symmetry '{symmetry}'.")

    max_order = _validate_even_order(int(max_order))

    cols = [np.ones_like(phi)]
    for n in range(2, max_order + 1, 2):
        cols.append(np.cos(n * phi))
    A = np.column_stack(cols)

    A_valid = A[mask]
    I_valid = I[mask]
    weights = np.ones_like(I_valid)
    if edge_emphasis and (~mask).any():
        idx = np.arange(mask.size)
        bad = idx[~mask]
        if bad.size:
            dist = np.full(mask.size, np.inf)
            for b in bad:
                np.minimum(dist, np.abs(idx - b), out=dist)
            decay = max(1.0, 0.05 * mask.size)
            full_weights = 1.0 + 3.0 * np.exp(-dist / decay)
            weights = full_weights[mask]

    ridge = max(0.0, float(ridge_lambda))
    if A_valid.shape[0] < A_valid.shape[1]:
        # Under-determined; fall back to simple mean to avoid blowing up the fit
        coef = np.zeros(A.shape[1])
        coef[0] = float(np.nanmean(I_valid)) if I_valid.size else 0.0
    else:
        Aw = A_valid * weights[:, None] ** 0.5
        Iw = I_valid * weights ** 0.5
        ata = Aw.T @ Aw
        if ridge > 0:
            reg = np.eye(ata.shape[0]) * ridge
            reg[0, 0] = 0.0  # do not damp the DC term
            ata = ata + reg
        try:
            coef = np.linalg.solve(ata, Aw.T @ Iw)
        except np.linalg.LinAlgError:
            coef, *_ = np.linalg.lstsq(Aw, Iw, rcond=None)

    I_reconstructed = A @ coef
    I_filled = I.copy()
    I_filled[~mask] = I_reconstructed[~mask]

    hop = _compute_hermans(phi, I_reconstructed) if compute_hop else None
    hwhm_deg = _compute_hwhm_deg(phi, I_reconstructed) if compute_hwhm else None

    return ReconstructionResult(
        phi=phi,
        I_original=I,
        mask=mask,
        I_reconstructed=I_reconstructed,
        I_filled=I_filled,
        hop=hop,
        hwhm_deg=hwhm_deg,
    )


# --- File helpers for quick CLI/GUI integrations ---

def _looks_like_header(line: str) -> bool:
    stripped = (line or "").strip()
    if not stripped:
        return True
    tokens = stripped.replace(",", " ").split()
    if not tokens:
        return True
    try:
        float(tokens[0])
        return False
    except ValueError:
        return True


def _scan_header_info(path: Path, max_lines: int = 20) -> tuple[str, int]:
    header_line = ""
    skiprows = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for _ in range(max_lines):
                line = fh.readline()
                if not line:
                    break
                raw = line.rstrip("\r\n")
                stripped = raw.strip()
                if not stripped:
                    skiprows += 1
                    continue
                if stripped.startswith("#"):
                    header_line = stripped
                    skiprows += 1
                    continue
                if _looks_like_header(stripped):
                    header_line = stripped
                    skiprows += 1
                    continue
                break
    except Exception:
        header_line = ""
        skiprows = 0
    return header_line, skiprows


def _load_dat_matrix(path: Path) -> tuple[np.ndarray, str]:
    header_line, skiprows = _scan_header_info(path)
    try:
        data = np.loadtxt(path, skiprows=skiprows)
    except ValueError:
        data = np.loadtxt(path, delimiter=",", skiprows=skiprows)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Unexpected data shape in {path}: {data.shape}")
    return data, header_line


def _auto_mask_column(col: np.ndarray) -> np.ndarray:
    finite = np.isfinite(col)
    positive = col > 0
    if positive.any() and (~positive & finite).any():
        return finite & positive
    return finite


def _pick_azimuthal_file(sample_path: Path) -> Path | None:
    if sample_path.is_file():
        return sample_path
    if not sample_path.exists():
        return None
    priority_patterns: Iterable[str] = ("azi*.*", "*azi*.*", "saxs*.*", "*.dat")
    candidates: list[Path] = []
    for pat in priority_patterns:
        candidates.extend(sorted(sample_path.glob(pat)))
    for cand in candidates:
        if cand.is_file():
            return cand
    return None


def _ensure_temp_folder(sample_path: Path) -> Path:
    sample_dir = sample_path if sample_path.is_dir() else sample_path.parent
    temp_dir = sample_dir / "_Temp"
    try:
        temp_dir.mkdir(parents=True, exist_ok=True)
        return temp_dir
    except Exception:
        pass
    sax_root = None
    for parent in sample_dir.parents:
        if parent.name.upper() == "SAXS":
            sax_root = parent
            break
    if sax_root:
        fallback = sax_root / "_Temp"
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback
    sample_dir.mkdir(parents=True, exist_ok=True)
    return sample_dir


def reconstruct_file_to_temp(
    sample_path: str,
    symmetry: str = "nematic",
    max_order: int = 4,
    ridge_lambda: float = 0.02,
    edge_emphasis: bool = True,
) -> tuple[Path, Path]:
    """Reconstruct the azimuthal curve(s) in a sample folder/file and write to _Temp.

    Returns (output_path, source_path).
    """
    src = _pick_azimuthal_file(Path(sample_path).expanduser())
    if src is None:
        raise FileNotFoundError(f"No azimuthal SAXS file found under {sample_path}")

    data, header_line = _load_dat_matrix(src)
    phi = np.radians(data[:, 0]) if data[:, 0].max() > 2 * np.pi else data[:, 0]
    intensity_block = data[:, 1:]

    filled_cols = []
    hop = None
    hwhm = None
    for idx in range(intensity_block.shape[1]):
        col = intensity_block[:, idx]
        mask = _auto_mask_column(col)
        res = reconstruct_azimuthal(
            phi,
            col,
            mask=mask,
            symmetry=symmetry,
            max_order=max_order,
            ridge_lambda=ridge_lambda,
            edge_emphasis=edge_emphasis,
            compute_hop=(idx == 0),
            compute_hwhm=(idx == 0),
        )
        filled_cols.append(res.I_filled)
        if idx == 0:
            hop = res.hop
            hwhm = res.hwhm_deg

    filled_matrix = np.column_stack([phi, np.column_stack(filled_cols)])
    if phi.max() > 2 * np.pi:
        filled_matrix[:, 0] = np.degrees(phi)

    temp_dir = _ensure_temp_folder(src)
    out_name = f"{src.stem}_reconstructed{src.suffix or '.dat'}"
    out_path = temp_dir / out_name
    header = "# Reconstructed via masked Fourier fit"
    if header_line and not header_line.startswith("#"):
        header = f"{header}\n# Original header: {header_line}"
    if hop is not None:
        header += f"\n# Estimated HoP: {hop:.5f}"
    if hwhm is not None:
        header += f"\n# Estimated HWHM (deg): {hwhm:.3f}"

    np.savetxt(out_path, filled_matrix, delimiter="\t", header=header, comments="")
    return out_path, src


def preview_reconstruction(src_path: Path, recon_path: Path, frame_index: int = 0):
    """Pop up a quick Matplotlib overlay: original vs reconstructed for one frame."""
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[Reconstruction] Preview unavailable (matplotlib import failed): {exc}")
        return

    def _load(path: Path) -> np.ndarray:
        hdr, skip = _scan_header_info(path)
        try:
            arr = np.loadtxt(path, skiprows=skip)
        except ValueError:
            arr = np.loadtxt(path, delimiter=",", skiprows=skip)
        return arr

    src = _load(src_path)
    recon = _load(recon_path)
    if src.shape[0] == 0 or src.shape[1] < 2 or recon.shape[1] < 2:
        print("[Reconstruction] Preview skipped: unexpected shapes.")
        return
    frame = max(0, min(frame_index, src.shape[1] - 2))
    x_src = src[:, 0]
    x_recon = recon[:, 0]
    y_src = src[:, frame + 1]
    y_recon = recon[:, frame + 1]

    plt.figure(figsize=(6, 4))
    plt.plot(x_src, y_src, "o", ms=3, alpha=0.6, label=f"Original frame {frame+1}")
    plt.plot(x_recon, y_recon, "-", lw=2, alpha=0.9, label="Reconstructed")
    plt.xlabel("Angle")
    plt.ylabel("Intensity")
    plt.title(f"Reconstruction preview — frame {frame+1}")
    plt.legend()
    plt.tight_layout()
    plt.show()
