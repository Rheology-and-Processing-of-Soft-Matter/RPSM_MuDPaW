from pathlib import Path
from typing import List, Tuple, Sequence
import re

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def get_azint_folder(
    proposal: int,
    visit: int,
    base: Path = Path("/data/visitors/formax"),
) -> Path:
    """
    Construct the azint/ folder path for a given proposal and visit.

    Example:
        proposal = 20250354
        visit    = 2025110408
        -> /data/visitors/formax/20250354/2025110408/process/azint/
    """
    return Path(base) / str(proposal) / str(visit) / "process" / "azint"


def detect_scan_ids(azint_folder: Path) -> List[int]:
    """
    Detect scan IDs from the contents of an azint folder.

    Works with both:
      - folder names like 'scan_0001', 'scan_0012'
      - filenames like '0001_azimuthal.dat', 'scan_0012_rad.h5', etc.

    The function should:
      - raise FileNotFoundError if azint_folder does not exist
      - search for 3-5 digit numbers in each filename using regex
      - collect all such numbers as integer scan IDs
      - return a sorted list of unique integers, e.g. [1, 2, 3, 6, 7]
    """
    folder = Path(azint_folder)
    if not folder.exists():
        raise FileNotFoundError(f"azint folder does not exist: {folder}")

    pattern = re.compile(r"(?<!\d)(\d{3,5})(?!\d)")
    scan_ids = set()

    for entry in folder.iterdir():
        for match in pattern.findall(entry.name):
            scan_ids.add(int(match))

    return sorted(scan_ids)


def group_continuous_ranges(scan_ids: List[int]) -> List[Tuple[int, int]]:
    """
    Group a sorted list of scan IDs into continuous ranges.

    Example:
        [1, 2, 3, 6, 7] -> [(1, 3), (6, 7)]
        [1201, 1202, 1203, 1207, 1208] -> [(1201, 1203), (1207, 1208)]

    If scan_ids is empty, return an empty list.
    """
    if not scan_ids:
        return []

    sorted_ids = sorted(scan_ids)
    ranges: List[Tuple[int, int]] = []

    start = prev = sorted_ids[0]
    for scan_id in sorted_ids[1:]:
        if scan_id == prev + 1:
            prev = scan_id
            continue

        ranges.append((start, prev))
        start = prev = scan_id

    ranges.append((start, prev))
    return ranges


def print_scan_overview(proposal: int, visit: int) -> None:
    """
    Convenience helper:
      - Calls get_azint_folder(proposal, visit)
      - Uses detect_scan_ids(...) and group_continuous_ranges(...)
      - Prints:
          azint folder path
          list of detected scan IDs
          continuous ranges in a human-readable form

    This is mainly for interactive use in a notebook or terminal.
    """
    azint_folder = get_azint_folder(proposal, visit)
    scan_ids = detect_scan_ids(azint_folder)
    ranges = group_continuous_ranges(scan_ids)

    print(f"azint folder: {azint_folder}")
    print(f"detected scan IDs: {scan_ids if scan_ids else 'none'}")

    if ranges:
        ranges_text = ", ".join(
            f"{start}-{end}" if start != end else f"{start}"
            for start, end in ranges
        )
    else:
        ranges_text = "none"

    print(f"continuous ranges: {ranges_text}")


def plot_azint_overview(
    A0: np.ndarray,
    A1: np.ndarray,
    R: np.ndarray,
    azi: np.ndarray,
    q_plot: np.ndarray,
    scan_ids: Sequence[int],
    qmin0: float,
    qmax0: float,
    qmin1: float,
    qmax1: float,
    qmin_r: float,
    qmax_r: float,
    xlab: str = "Frame #",
):
    """
    Make the 3x2 overview plot for azimuthal and radial data.

    Parameters:
        A0, A1 : 2D arrays [frame, azi] (can be transposed internally as needed)
        R      : 2D array [frame, q] or [q, frame] depending on convention
        azi    : 1D array of azimuthal angles (0-360 deg)
        q_plot : 1D array of q values
        scan_ids : sequence of scan numbers (for figure title only)
        qmin0, qmax0 : q-range for first azimuthal band (for title + vertical lines)
        qmin1, qmax1 : q-range for second azimuthal band
        qmin_r, qmax_r : q-range for radial image title
        xlab   : label for x-axis in the image plots (usually frame index or time)

    The implementation should use the plotting code I provide below,
    wrapped into this function and slightly parameterized.
    """
    if not scan_ids:
        raise ValueError("scan_ids must contain at least one element")

    def _orient_azimuthal(arr: np.ndarray, angles: np.ndarray) -> np.ndarray:
        """Ensure the azimuthal array shape is [frame, azi]."""
        if arr.shape[1] == angles.shape[0]:
            return arr
        if arr.shape[0] == angles.shape[0]:
            return arr.T
        raise ValueError(
            "Azimuthal array shape does not match azi length"
        )

    def _orient_radial(radial: np.ndarray, q_vals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return image data [frame, q] and line data [q, frame] aligned to q_plot."""
        q_len = q_vals.shape[0]
        if radial.shape[1] == q_len:
            return radial, radial.T
        if radial.shape[0] == q_len:
            return radial.T, radial
        raise ValueError("R shape does not align with q_plot length")

    A0_oriented = _orient_azimuthal(A0, azi)
    A1_oriented = _orient_azimuthal(A1, azi)
    radial_img, radial_lines = _orient_radial(R, q_plot)

    fig, axs = plt.subplots(3, 2, figsize=(11, 9))

    first_scan, last_scan = scan_ids[0], scan_ids[-1]
    scan_title = f"Scan {first_scan}" if first_scan == last_scan else f"Scans {first_scan}-{last_scan}"
    fig.suptitle(f"{scan_title}", fontsize=10)
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.9, wspace=0.18, hspace=0.4)
    for ax in fig.axes:
        ax.tick_params(labelsize=10)

    # Azimuthal
    ax1 = axs[0, 0]
    ax1.set_title(f"{qmin0} <azi< {qmax0}", fontsize=9)
    ax1.imshow(
        A0_oriented.T,
        extent=[0, A0_oriented.shape[0], azi[0], azi[-1]],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        norm=LogNorm(),
    )
    ax1.set_xlabel(xlab, fontsize=9)
    ax1.set_ylabel(r"$\Theta$ [$^\circ$]", fontsize=9)

    ax2 = axs[0, 1]
    ax2.plot(azi, A0_oriented.T)
    ax2.set_ylabel("I [a.u.]", fontsize=9)
    ax2.set_xlim(0, 360)
    ax2.set_xlabel(r"$\Theta$ [$^\circ$]", fontsize=9)

    ax3 = axs[1, 0]
    ax3.set_title(f"{qmin1} <azi< {qmax1}", fontsize=9)
    ax3.imshow(
        A1_oriented.T,
        extent=[0, A1_oriented.shape[0], azi[0], azi[-1]],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        norm=LogNorm(),
    )
    ax3.set_xlabel(xlab, fontsize=9)
    ax3.set_ylabel(r"$\Theta$ [$^\circ$]", fontsize=9)

    ax4 = axs[1, 1]
    ax4.plot(azi, A1_oriented.T)
    ax4.set_ylabel("I [a.u.]", fontsize=9)
    ax4.set_xlim(0, 360)
    ax4.set_xlabel(r"$\Theta$ [$^\circ$]", fontsize=9)

    # Radial
    ax5 = axs[2, 0]
    ax5.set_title(f"{qmin_r} <rad< {qmax_r}", fontsize=9)
    ax5.imshow(
        radial_img,
        extent=[q_plot[0], q_plot[-1], 0, radial_img.shape[0]],
        origin="lower",
        cmap="viridis",
        aspect="auto",
        norm=LogNorm(),
    )
    ax5.set_xlabel("q [1/A]", fontsize=9)
    ax5.set_ylabel(xlab, fontsize=9)
    ax5.set_xlim(q_plot[0], 0.08)

    ax6 = axs[2, 1]
    ax6.plot(q_plot, radial_lines)
    ax6.set_xlim(q_plot[0], 0.08)
    ax6.axvline(x=qmin0, color="r", ls="--", lw=0.85)
    ax6.axvline(x=qmax0, color="r", ls="--", lw=0.85)
    ax6.axvline(x=qmin1, color="b", ls="--", lw=0.85)
    ax6.axvline(x=qmax1, color="b", ls="--", lw=0.85)
    ax6.set_xscale("log")
    ax6.set_yscale("log")
    ax6.set_ylabel("I [a.u.]", fontsize=9)
    ax6.set_xlabel("q [1/A]", fontsize=9)

    plt.show()
    return fig, axs
