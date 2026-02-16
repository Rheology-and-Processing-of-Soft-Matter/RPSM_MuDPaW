#!/usr/bin/env python3
"""
Simple installer for MuDPaW dependencies.
Run with: python Install.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REQUIREMENTS_FILE = Path(__file__).with_name("requirements.txt")
DEFAULT_PACKAGES = [
    # Core GUI + numerics
    "PyQt6",
    "numpy",
    "scipy",
    "matplotlib",
    "pandas",
    # Imaging / video
    "opencv-python",
    "Pillow",
    "imageio",
    "imageio-ffmpeg",
    # File formats
    "h5py",
    "pyyaml",
    # Utilities / stats
    "psutil",
    "plotly",
    "statsmodels",
    "moepy",
    "seaborn",
    "fitter",
]



def unique(seq):
    seen = set()
    out = []
    for item in seq:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def load_packages():
    if REQUIREMENTS_FILE.exists():
        lines = REQUIREMENTS_FILE.read_text(encoding="utf-8").splitlines()
        pkgs = [
            line.strip()
            for line in lines
            if line.strip() and not line.strip().startswith("#")
        ]
        if pkgs:
            print(f"→ Loaded packages from {REQUIREMENTS_FILE.name}")
            return unique(pkgs)
        print(f"→ {REQUIREMENTS_FILE.name} is empty, falling back to defaults")
    else:
        print(f"→ {REQUIREMENTS_FILE.name} not found, falling back to defaults")
    return DEFAULT_PACKAGES


def ensure_pip():
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "--version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        print("→ pip not detected, installing via ensurepip …")
        subprocess.check_call([sys.executable, "-m", "ensurepip", "--upgrade"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])


def install_package(package):
    print(f"→ Installing {package}")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--upgrade", package]
        )
        return True
    except subprocess.CalledProcessError as err:
        print(f"  ! Failed to install {package}: {err}")
        return False


def main():
    packages = load_packages()
    if not packages:
        print("No packages to install.")
        return

    print(f"→ Using interpreter: {sys.executable}")
    ensure_pip()

    successes = []
    failures = []
    for pkg in packages:
        if install_package(pkg):
            successes.append(pkg)
        else:
            failures.append(pkg)

    print("\n=== Installation summary ===")
    if successes:
        print(f"  ✓ Installed: {', '.join(successes)}")
    if failures:
        print(f"  ✗ Failed: {', '.join(failures)}")
        print("    You can retry failed packages manually once issues are resolved.")
    else:
        print("  All packages installed successfully.")


if __name__ == "__main__":
    main()
