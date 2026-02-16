# Created by Roland Kádár on 2024-06-10.
# Copyright (c) 2024 Chalmers University of Technology. All rights reserved.
# Based on an earlier code made for Matlab by Roland Kádár in 2021.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from scipy.special import legendre
import seaborn as sns
import matplotlib
import array
import os
import csv
import cv2
from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy import integrate
from scipy.signal import savgol_filter
from itertools import count
import random
from moepy import lowess, eda
from fitter import Fitter, get_common_distributions, get_distributions
from scipy.signal import find_peaks


# The code assumes that steady state has been extracted and that the input data has been stitched into equal size


def LabHWHM(stitched_path: str, n_intervals: int):
    """
    Load a stitched space–time image, split into CIE-Lab channels,
    average each channel column-wise within equal-width intervals, and return
    (L_avg, a_avg, b_avg) arrays of shape (rows, n_intervals).
    """
    img = cv2.imread(stitched_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image at: {stitched_path}")

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, a, b = cv2.split(img_lab)

    L = L.astype(np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)

    height, width = L.shape
    if n_intervals <= 0:
        raise ValueError("n_intervals must be > 0")
    if n_intervals > width:
        raise ValueError("n_intervals cannot exceed image width in pixels")

    interval_size = width // n_intervals

    File_avg_L = np.zeros((height, n_intervals), dtype=np.float64)
    File_avg_a = np.zeros((height, n_intervals), dtype=np.float64)
    File_avg_b = np.zeros((height, n_intervals), dtype=np.float64)

    for i in range(n_intervals):
        start_ = i * interval_size
        stop_ = (i + 1) * interval_size if i < n_intervals - 1 else width
        Temp_L = L[:, start_:stop_]
        Temp_a = a[:, start_:stop_]
        Temp_b = b[:, start_:stop_]
        File_avg_L[:, i] = np.mean(Temp_L, axis=1)
        File_avg_a[:, i] = np.mean(Temp_a, axis=1)
        File_avg_b[:, i] = np.mean(Temp_b, axis=1)

    return File_avg_L, File_avg_a, File_avg_b


def ExPLIDat(trim_, x_axis=None, *, model="gaussian", q_init=1.1, use_baseline=True):
    """
    Fit each column of the provided 2D array (m x n) and return the Half-Width at Half-Maximum (HWHM)
    and a Hermans-like orientation parameter (HUH ∈ [0,1]) derived from that width under a wrapped-Gaussian
    assumption.
    """
    if trim_ is None:
        raise ValueError("trim_ cannot be None")
    if trim_.ndim != 2:
        raise ValueError("trim_ must be a 2D array of shape (m, n)")

    m, n = trim_.shape
    x = np.asarray(x_axis, dtype=float).reshape(-1) if x_axis is not None else np.arange(m, dtype=float)
    if x.size != m:
        x = np.linspace(0, m - 1, m)

    xr = float(np.nanmax(x) - np.nanmin(x))
    if 5.5 <= xr <= 7.5:
        _x_units = "radians"
    elif 300.0 <= xr <= 400.0:
        _x_units = "degrees"
    elif 0.9 <= xr <= 1.1:
        _x_units = "unit"
    else:
        _x_units = "unit"

    def _dx(xarr: np.ndarray) -> float:
        xd = np.diff(xarr)
        return float(np.nanmedian(xd)) if xd.size else 1.0

    def _hwhm_to_hermans(hwhm_val: float) -> float:
        if not np.isfinite(hwhm_val) or hwhm_val <= 0:
            return np.nan
        if _x_units == "radians":
            hwhm_rad = float(hwhm_val)
        elif _x_units == "degrees":
            hwhm_rad = float(hwhm_val) * (np.pi / 180.0)
        elif _x_units == "unit":
            span_local = float(np.nanmax(x) - np.nanmin(x))
            if not np.isfinite(span_local) or span_local <= 0:
                return np.nan
            hwhm_norm = float(hwhm_val) / span_local
            hwhm_rad = hwhm_norm * (2.0 * np.pi)
        else:
            return np.nan
        S = np.exp(-(hwhm_rad ** 2) / np.log(2.0))
        return float(np.clip(S, 0.0, 1.0))

    def _half_widths_from_curve(xg: np.ndarray, yg: np.ndarray, mu: float, half_level: float):
        if xg.size < 3 or yg.size != xg.size:
            return np.nan, np.nan, np.nan, np.nan

        def _cross(xarr, yarr):
            res = yarr - half_level
            sign_change = np.where(res[:-1] * res[1:] <= 0)[0]
            if sign_change.size > 0:
                i = sign_change[-1]
                x0, x1 = xarr[i], xarr[i + 1]
                y0, y1 = yarr[i], yarr[i + 1]
                if y1 == y0:
                    return float(x0)
                return float(x0 + (half_level - y0) * (x1 - x0) / (y1 - y0))
            k = int(np.argmin(np.abs(res)))
            return float(xarr[k])

        left_mask = xg <= mu
        right_mask = xg >= mu
        hwhm_left = hwhm_right = np.nan
        if np.any(left_mask):
            hwhm_left = abs(mu - _cross(xg[left_mask], yg[left_mask]))
        if np.any(right_mask):
            hwhm_right = abs(_cross(xg[right_mask], yg[right_mask]) - mu)
        fwhm_eff = hwhm_left + hwhm_right if np.isfinite(hwhm_left) and np.isfinite(hwhm_right) else (
            2 * hwhm_left if np.isfinite(hwhm_left) else (2 * hwhm_right if np.isfinite(hwhm_right) else np.nan)
        )
        hwhm_eff = fwhm_eff * 0.5 if np.isfinite(fwhm_eff) else np.nan
        return hwhm_left, hwhm_right, fwhm_eff, hwhm_eff

    def _gauss_baseline(xarr, A, x0, sigma, C):
        return C + A * np.exp(-((xarr - x0) ** 2) / (2.0 * (sigma ** 2)))

    def _q_gauss_asym(xarr, A, x0, sigma_L, sigma_R, q, C):
        sigma = np.where(xarr < x0, sigma_L, sigma_R)
        z2 = (xarr - x0) ** 2 / (sigma ** 2 + 1e-18)
        if abs(q - 1.0) < 1e-6:
            core = np.exp(-0.5 * z2)
        else:
            core = np.power(np.maximum(1.0 + (q - 1.0) * 0.5 * z2, 1e-18), -1.0 / (q - 1.0))
        return C + A * core

    def _peak_window(xarr: np.ndarray, yarr: np.ndarray, j0: int, frac: float = 0.2, pad_pts: int = 2):
        yb = yarr - np.nanpercentile(yarr, 10.0)
        yb = np.where(np.isfinite(yb), yb, 0.0)
        peak = float(np.nanmax(yb)) if np.isfinite(np.nanmax(yb)) else 0.0
        thr = frac * peak
        m = yb >= thr
        if not np.any(m):
            return 0, yarr.size
        lo = j0
        while lo > 0 and m[lo - 1]:
            lo -= 1
        hi = j0
        n_ = yarr.size
        while hi < n_ - 1 and m[hi + 1]:
            hi += 1
        lo = max(0, lo - pad_pts)
        hi = min(n_, hi + 1 + pad_pts)
        return int(lo), int(hi)

    HWHM = np.full(n, np.nan, dtype=np.float64)
    HUH = np.full(n, np.nan, dtype=np.float64)
    FWHM = np.full(n, np.nan, dtype=np.float64)
    AUC = np.full(n, np.nan, dtype=np.float64)
    params = []
    fits = []
    r2_list = []

    sqrt_2ln2 = np.sqrt(2.0 * np.log(2.0))
    dx_est = _dx(x)

    for j in range(n):
        y = np.asarray(trim_[:, j], dtype=np.float64)
        j_max = int(np.nanargmax(y)) if y.size else 0
        C0 = float(np.nanpercentile(y, 10.0))
        A0_raw = float(np.nanmax(y) - C0)
        A0 = max(A0_raw, 1e-9)
        x0 = float(x[j_max]) if np.isfinite(x[j_max]) else float(np.nanmedian(x))
        span = float(np.nanmax(x) - np.nanmin(x)) if np.isfinite(np.nanmax(x) - np.nanmin(x)) else (dx_est * max(3, y.size))
        y_above = y - C0
        half = 0.5 * (np.nanmax(y_above))
        left = j_max
        while left > 0 and y_above[left] > half:
            left -= 1
        right = j_max
        n_ = y.size
        while right < n_ - 1 and y_above[right] > half:
            right += 1
        fwhm_guess_pts = max(3, right - left)
        fwhm_guess = fwhm_guess_pts * dx_est
        sigma0 = max(fwhm_guess / (2.0 * sqrt_2ln2), dx_est * 0.5)
        lo, hi = _peak_window(x, y, j_max, frac=0.2, pad_pts=2)
        x_fit_local = x[lo:hi]
        y_fit_local = y[lo:hi]
        p0 = (A0, x0, sigma0, C0)

        def _project_p0(p0_, lb_, ub_):
            p = list(p0_)
            for i in range(len(p)):
                lo = lb_[i]
                hi = ub_[i]
                if not np.isfinite(p[i]):
                    p[i] = (lo + hi) * 0.5
                if p[i] <= lo:
                    p[i] = lo + 1e-9 * (abs(hi - lo) + 1.0)
                if p[i] >= hi:
                    p[i] = hi - 1e-9 * (abs(hi - lo) + 1.0)
            return tuple(p)

        if str(model).lower() == "asym_qgauss":
            sigmaL0 = max(dx_est * 0.5, fwhm_guess / (2.0 * sqrt_2ln2))
            sigmaR0 = sigmaL0
            q0 = float(q_init) if np.isfinite(q_init) else 1.1
            if q0 <= 0.0:
                q0 = 1.1
            p0_q = (A0, x0, sigmaL0, sigmaR0, q0, C0 if use_baseline else 0.0)
            lb = (0.0,
                  float(np.nanmin(x_fit_local)),
                  dx_est * 0.25,
                  dx_est * 0.25,
                  0.8,
                  0.0 if not use_baseline else float(np.nanmin(y_fit_local)))
            ub = (np.inf,
                  float(np.nanmax(x_fit_local)),
                  span,
                  span,
                  3.0,
                  0.0 if not use_baseline else float(np.nanmax(y_fit_local)))
            bounds = (lb, ub)
            p0_q = _project_p0(p0_q, lb, ub)

            try:
                popt, _ = curve_fit(
                    _q_gauss_asym, x_fit_local, y_fit_local,
                    p0=p0_q, bounds=bounds, method="trf", maxfev=60000
                )
                A_fit, x0_fit, sigmaL_fit, sigmaR_fit, q_fit, C_fit = popt
                yhat = _q_gauss_asym(x, *popt)
                ss_res = float(np.nansum((y - yhat) ** 2))
                ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                x_dense = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), max(400, 4 * x.size))
                y_dense = _q_gauss_asym(x_dense, *popt)
                half_level = (C_fit if use_baseline else 0.0) + 0.5 * A_fit
                hL, hR, fwhm_eff, hwhm_eff = _half_widths_from_curve(x_dense, y_dense, x0_fit, half_level)

                if not np.isfinite(hwhm_eff):
                    hwhm_eff = sqrt_2ln2 * float(np.nanmean([abs(sigmaL_fit), abs(sigmaR_fit)]))
                    fwhm_eff = 2.0 * hwhm_eff

                HWHM[j] = hwhm_eff
                FWHM[j] = fwhm_eff
                HUH[j] = _hwhm_to_hermans(hwhm_eff)

                auc = float(np.trapz(y_dense - (0.0 if not use_baseline else C_fit), x_dense))
                AUC[j] = auc

                params.append((float(A_fit), float(x0_fit), float(sigmaL_fit), float(sigmaR_fit), float(q_fit), float(C_fit)))
                fits.append(yhat)
                r2_list.append(r2)
            except Exception as e:
                print(f"Column {j}: asymmetric q-Gaussian fit failed ({e}).")
                params.append((np.nan, np.nan, np.nan, np.nan, np.nan, np.nan))
                fits.append(np.full_like(x, np.nan, dtype=float))
                r2_list.append(np.nan)
        else:
            lb = (0.0, float(np.nanmin(x_fit_local)), dx_est * 0.25, 0.0 if not use_baseline else float(np.nanmin(y_fit_local)))
            ub = (np.inf, float(np.nanmax(x_fit_local)), span, 0.0 if not use_baseline else float(np.nanmax(y_fit_local)))
            bounds = (lb, ub)
            p0 = (A0, x0, sigma0, 0.0 if not use_baseline else C0)
            p0 = _project_p0(p0, lb, ub)

            try:
                popt, _ = curve_fit(
                    _gauss_baseline, x_fit_local, y_fit_local,
                    p0=p0, bounds=bounds, method="trf", maxfev=50000
                )
                A_fit, x0_fit, sigma_fit, C_fit = popt

                hwhm = sqrt_2ln2 * abs(sigma_fit)
                HWHM[j] = hwhm
                HUH[j] = _hwhm_to_hermans(hwhm)
                fwhm = 2.0 * hwhm
                FWHM[j] = fwhm
                AUC[j] = float(A_fit) * np.sqrt(2.0 * np.pi) * abs(sigma_fit)

                yhat = _gauss_baseline(x, A_fit, x0_fit, sigma_fit, C_fit)
                ss_res = float(np.nansum((y - yhat) ** 2))
                ss_tot = float(np.nansum((y - np.nanmean(y)) ** 2))
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
                params.append((float(A_fit), float(x0_fit), float(sigma_fit), float(C_fit)))
                fits.append(yhat)
                r2_list.append(r2)
            except Exception as e:
                print(f"Column {j}: Gaussian fit failed ({e}).")
                params.append((np.nan, np.nan, np.nan, np.nan))
                fits.append(np.full_like(x, np.nan, dtype=float))
                r2_list.append(np.nan)

    details = {
        "params": params,
        "fits": fits,
        "r2": r2_list,
        "x": x,
        "HWHM": HWHM,
        "FWHM": FWHM,
        "AUC": AUC,
        "model": str(model),
    }
    return HWHM, HUH, details
