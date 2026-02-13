#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Based on a Matlab code from 2022, based on an earlier DataGraph file

Created on Sun Dec 29 09:06:37 2024

@author: Roland Kádár
"""
import numpy as np
import os
import pandas as pd
import matplotlib
if os.environ.get("MUDPAW_EMBED") == "1":
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
from matplotlib.lines import Line2D
from scipy import ndimage
import statsmodels.api as sm
from scipy.optimize import curve_fit
from scipy.special import legendre
from scipy import integrate
from scipy.signal import savgol_filter
import warnings
#matplotlib.rcParams['figure.figsize'] = (5, 10)
from itertools import count
import random
from moepy import lowess, eda
#import _Main_custom_LAOS_PLI_SAXS_v1 as main

def P2468OAS_v4(
    method,
    mirror,
    Angles,
    File_part_,
    lims,
    smoo_,
    sigma_zero,
    plot=True,
    preview_limit=1000,
    save_prefix=None,
    window_title=None,
    flatten_minima=True,
    flatten_tail_fit=False,
    flatten_tail_extent=0.9,
    centering_mode="auto",
    weak_metric="A2",
    weak_threshold=None,
    side_mode="auto",
):
    method = str(method or "fitting").strip().lower()
    if method not in {"fitting", "direct"}:
        method = "fitting"
    raw_input = np.array(File_part_, copy=True)
    File = raw_input.copy()
    lo_li = int(lims[0])
    up_li = int(lims[1])
    #preview_limit=1000
    
    #Initializations and corrections
    #Baseline = 0
    #File = File-Baseline
    Threshold = 0.0001
    [DataM,DataN] =  np.shape(File)
    #XRaww = File[:,0]*2*np.pi/DataM
    #File=File[:,1:DataN]
    #File=File[:,0:DataN]
    #DataN=DataN-1
    #print(DataM)
    Tot_ang=DataM
    UnitAngleRad = 2*np.pi/DataM
    # Angle axis: treat input as degrees; if it looks like radians (<= 2π), convert to degrees.
    # If the range is implausibly large for degrees, fall back to index-based degrees.
    angle_raw = np.asarray(Angles, dtype=float).ravel()
    if angle_raw.size != DataM or not np.isfinite(angle_raw).any():
        angle_deg = np.linspace(0.0, 360.0, DataM, endpoint=False)
        angle_rad = np.deg2rad(angle_deg)
    else:
        amax = np.nanmax(angle_raw)
        amin = np.nanmin(angle_raw)
        if amax <= (2 * np.pi + 0.1) and amin >= 0.0:
            angle_rad = angle_raw
            angle_deg = np.rad2deg(angle_raw)
        else:
            angle_deg = angle_raw
            angle_rad = np.deg2rad(angle_raw)
        # If the values don't look like degrees (outside 0..360), treat as no angle axis.
        if np.nanmin(angle_deg) < -0.1 or np.nanmax(angle_deg) > 360.1:
            angle_deg = np.linspace(0.0, 360.0, DataM, endpoint=False)
            angle_rad = np.deg2rad(angle_deg)
    XRaww = angle_rad
    
    
    Half = int(DataM/2)
    Quarter = int(Half/2)
    #print(Quarter)
            
    DataShift=np.zeros((DataM,DataN))
    DataB=np.zeros((DataM,DataN))
    DataShift=File.copy()
    # Robust half-swap that also works for odd DataM (e.g., 359 angles)
    upper_len = Tot_ang - Half  # equals Half for even, Half+1 for odd
    if mirror == True:
        # Mirror the first block into the second block; lengths now match
        mirror_src = DataShift[:upper_len, :]
        DataShift[Half:Half + upper_len, :] = mirror_src[::-1, :]
        Criterium = 45
    else:
        # Rotate halves without size mismatch for odd lengths
        DataShift[0:upper_len, :] = File[Half:Tot_ang, :]
        DataShift[upper_len:Tot_ang, :] = File[0:Half, :]
        Criterium = 0
    DataIn_=np.zeros((DataM,DataN))
    DataIn=DataShift
    smooth=np.zeros((DataM,DataN))
    MAXX=np.zeros((DataN))
    Index_MAXX=np.zeros((DataN))
    smooth_bak=np.zeros((DataM,DataN))
    Fit_Lor=np.zeros((int(Quarter),DataN))
    Fit_L0146=np.zeros((int(Quarter),DataN))
    Fit_L0123456=np.zeros((int(Quarter),DataN))
    FIT=np.zeros((int(Quarter),DataN))
    Fit_G = np.zeros((int(Quarter),DataN))
    Integration_denominator=np.zeros(DataN)
    Integration_numerator_P2=np.zeros(DataN)
    Integration_numerator_P4=np.zeros(DataN)
    Integration_numerator_P6=np.zeros(DataN)
    AvgP2=np.zeros(DataN)
    AvgP4=np.zeros(DataN)
    AvgP6=np.zeros(DataN)
    HoP=np.zeros(DataN)
    P2=np.zeros(DataN)
    P4=np.zeros(DataN)
    P6=np.zeros(DataN)
    Index_MAXX_store=np.zeros(DataN)
    display_idx = np.zeros(DataN, dtype=int)
    display_max = np.zeros(DataN)
    metrics = np.zeros(DataN)
    c2_values = np.zeros(DataN, dtype=complex)
    cmap=plt.get_cmap('turbo')
    # Degree window mask
    try:
        lo_deg, up_deg = float(lims[0]), float(lims[1])
    except Exception:
        lo_deg, up_deg = 0.0, 360.0
    window_idx = np.where((angle_deg >= lo_deg) & (angle_deg <= up_deg))[0]
    if window_idx.size == 0:
        window_idx = np.arange(DataM)
    window_min = int(window_idx.min())
    window_max = int(window_idx.max())
    #Y=[np.zeros((int(Half/2),DataN));]
    
    #Fit=np.zeros((6,DataN))
    
    #Smoothing
    DataB=DataShift
    #DataShift = File


    for i in range(0,DataN):
        if smoo_>0:
            #DataB[:,i] = pd.DataFrame(File[:,i])
            #DataB[:,i].replace(0, np.nan, inplace=True)
            #if i<35: 
            #    Frac=0.01
            #else: Frac=0.01
            lowess_model = lowess.Lowess()
            #smooth = sm.nonparametric.lowess(exog=XRaww, endog=DataB[:,i], frac=0.04, it=1000)
            lowess_model.fit(XRaww, DataB[:,i], frac=smoo_, robust_iters=0)
            smooth = lowess_model.predict(XRaww)
            #smooth = savgol_filter(DataB[:,i], window_length=10, polyorder=2, deriv=1)
            #smooth[:,1] = DataB[:,i]
            smooth_bak[:,i]=smooth#[:,1]
            MAXX[i] = smooth[lo_li:up_li].max(axis=0)
            Index_MAXX[i] = int(smooth[lo_li:up_li].argmax(axis=0))+lo_li
            Index_MAXX_store[i]=((Index_MAXX[i])*360/DataM)-90
            Index_MAXX_store[i] = ((Index_MAXX_store[i] + 90) % 180) - 90
            #    Index_MAXX_store[i]=Index_MAXX_store[i]+360
            #print(np.shape(Index_MAXX))
            DataNorm= smooth_bak
            #print("here")
        else:
            #$print("here")
            smooth  = DataB[:, i]
            smooth_bak[:,i]=smooth
            sub = smooth[window_idx]
            MAXX[i] = np.nanmax(sub)
            Index_MAXX[i] = int(window_idx[int(np.nanargmax(sub))])
            Index_MAXX_store[i]=((Index_MAXX[i])*360/DataM)-90
            Index_MAXX_store[i] = ((Index_MAXX_store[i] + 90) % 180) - 90
            DataNorm=smooth_bak
            #print(DataNorm)

        # Alignment metric
        col = smooth_bak[:, i]
        col = np.nan_to_num(col, nan=0.0)
        baseline = np.nanmin(col) if np.isfinite(np.nanmin(col)) else 0.0
        w = np.maximum(col - baseline, 0.0)
        eps = 1e-12
        c2 = np.sum(w * np.exp(1j * 2 * XRaww))
        c2_values[i] = c2
        denom_w = np.sum(w) + eps
        A2_val = np.abs(c2) / denom_w
        cos2_mean = np.sum(w * (np.cos(XRaww) ** 2)) / denom_w
        S2_val = (3 * cos2_mean - 1) / 2
        metrics[i] = A2_val if str(weak_metric).lower() == "a2" else abs(S2_val)

    #plt.plot(DataNorm)
    #plt.show()
    # Compute dataset-level weak threshold and global center
    finite_metrics = metrics[np.isfinite(metrics)]
    auto_threshold = (weak_threshold is None) or (str(weak_threshold).strip() == "")
    if auto_threshold:
        weak_thresh_val = -1.0  # disable weak detection → use argmax in auto mode
    else:
        try:
            weak_thresh_val = float(weak_threshold)
        except Exception:
            weak_thresh_val = 0.0

    mean_curve = np.nanmean(smooth_bak, axis=1)
    mean_curve = np.nan_to_num(mean_curve, nan=0.0)
    baseline_mean = np.nanmin(mean_curve) if np.isfinite(np.nanmin(mean_curve)) else 0.0
    w_mean = np.maximum(mean_curve - baseline_mean, 0.0)
    c2_global = np.sum(w_mean * np.exp(1j * 2 * XRaww))
    if np.abs(c2_global) > 1e-12:
        phi0_global = 0.5 * np.angle(c2_global) % (2 * np.pi)
        center_global_idx = int(np.round((phi0_global / (2 * np.pi)) * DataM)) % DataM
    else:
        center_global_idx = int(np.nanargmax(mean_curve[lo_li:up_li]) + lo_li)

    def _harmonic_center(c2_val, prev_center):
        if not np.isfinite(c2_val.real) or not np.isfinite(c2_val.imag) or (abs(c2_val) < 1e-12):
            return center_global_idx
        phi0 = 0.5 * np.angle(c2_val) % (2 * np.pi)
        idx0 = int(np.round((phi0 / (2 * np.pi)) * DataM)) % DataM
        idx1 = (idx0 + Half) % DataM
        def circ_dist(a, b):
            d = abs(a - b)
            return min(d, DataM - d)
        return idx0 if circ_dist(idx0, prev_center) <= circ_dist(idx1, prev_center) else idx1

    cmode = str(centering_mode or "auto").lower()
    wmode = str(weak_metric or "a2").lower()
    prev_center = center_global_idx
    for i in range(DataN):
        metric_val = metrics[i]
        weak = metric_val < weak_thresh_val
        if cmode == "argmax":
            center_idx = int(np.nanargmax(smooth_bak[lo_li:up_li, i]) + lo_li)
        elif cmode == "harmonic2":
            center_idx = _harmonic_center(c2_values[i], prev_center)
        elif cmode == "xcorr":
            center_idx = int(np.nanargmax(smooth_bak[lo_li:up_li, i]) + lo_li)
        else:  # auto
            if weak:
                center_idx = _harmonic_center(c2_values[i], prev_center)
            else:
                center_idx = int(np.nanargmax(smooth_bak[lo_li:up_li, i]) + lo_li)
        Index_MAXX[i] = center_idx
        prev_center = center_idx
        Index_MAXX_store[i]=((Index_MAXX[i])*360/DataM)-90
        Index_MAXX_store[i] = ((Index_MAXX_store[i] + 90) % 180) - 90

        # For display: show the peak inside the user window if possible
        if window_min < window_max:
            seg_disp = smooth_bak[window_min:window_max+1, i]
            local_idx = int(np.nanargmax(seg_disp)) + window_min
            disp_idx = local_idx
            disp_val = seg_disp[local_idx - window_min]
        else:
            disp_idx = center_idx % DataM
            disp_val = smooth_bak[disp_idx, i]
        display_idx[i] = disp_idx
        display_max[i] = disp_val if np.isfinite(disp_val) else MAXX[i]

    #smooth_bak=DataB
    
    #DataNorm=DataB #meaning, we determine the max on smootheed data but we process raw data for Legendre fitting
    #DataNorm=DataShift
    
    # === Combined overview figure (optional) ===
    fig = None
    ax_preview = None
    ax_pvals = None
    ax_hop = None
    preview_x_min = np.inf
    preview_x_max = -np.inf
    preview_y_min = np.inf
    preview_y_max = -np.inf
    if plot:
        fig = plt.figure(figsize=(13.0, 8.5), constrained_layout=False)
        if window_title and hasattr(fig.canvas.manager, "set_window_title"):
            fig.canvas.manager.set_window_title(window_title)
        gs = fig.add_gridspec(
            4,
            3,
            height_ratios=[1.0, 0.7, 1.0, 1.0],
            width_ratios=[1.0, 1.0, 1.0],
            wspace=0.35,
            hspace=0.85,
        )

        shared_cmap = plt.get_cmap("turbo")
        color_cycle = shared_cmap(np.linspace(0, 1, max(10, DataN)))

        ax_heat = fig.add_subplot(gs[0, 0])
        heat_data = np.flip(raw_input, axis=0)
        im = ax_heat.imshow(heat_data, aspect="auto", cmap="viridis")
        fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
        ax_heat.set_title("Raw Input Data")
        ax_heat.set_xlabel("Frame")
        ax_heat.set_ylabel("Angle idx")

        ax_raw = fig.add_subplot(gs[0, 1])
        ax_raw.set_prop_cycle(color=color_cycle)
        ax_raw.plot(angle_deg, raw_input, linewidth=0.4)
        ax_raw.set_title("Raw traces")
        ax_raw.set_xlabel("Angle [deg]")

        ax_shift = fig.add_subplot(gs[0, 2])
        ax_shift.set_prop_cycle(color=color_cycle)
        ax_shift.plot(angle_deg, DataB, linewidth=0.4)
        ax_shift.set_title("Data after shifting")
        ax_shift.set_xlabel("Angle [deg]")

        ax_smooth = fig.add_subplot(gs[1, :])
        ax_smooth.set_prop_cycle(color=color_cycle)
        ax_smooth.plot(angle_deg, smooth_bak, linewidth=0.4, zorder=1)
        idx_safe = np.clip(display_idx.astype(int), 0, DataM - 1)
        ax_smooth.scatter(
            angle_deg[idx_safe],
            display_max,
            s=10,
            marker="o",
            zorder=2,
            facecolor="none",
            edgecolor="k",
            linewidth=0.4,
        )
        ax_smooth.set_title("Smooth data with maxima")
        ax_smooth.set_xlabel("Angle [deg]", labelpad=8)
        ax_smooth.set_xlim(angle_deg.min(), angle_deg.max())

        ax_pvals = fig.add_subplot(gs[2:, 0])
        ax_pvals.set_ylabel("P2, P4, P6")
        ax_pvals.set_xlabel("Step index")
        ax_pvals.set_box_aspect(1)

        ax_preview = fig.add_subplot(gs[2:, 1])
        ax_preview.set_title("P02468 preview (data vs fit)")
        ax_preview.set_xlabel("Angle [deg]")
        ax_preview.set_ylabel("Norm. intensity")
        ax_preview.set_box_aspect(1)

        ax_hop = fig.add_subplot(gs[2:, 2])
        ax_hop.set_title("OAS vs step")
        ax_hop.set_xlabel("Step index")
        ax_hop.set_ylabel("OAS [deg]")
    
    def Legendre_series(x,a0,a2,a4,a6,a8):
        L0=legendre(0)
        P0=L0(np.cos(x))
        L2=legendre(2)
        P2=L2(np.cos(x))
        L4=legendre(4)
        P4=L4(np.cos(x))
        L6=legendre(6)
        P6=L6(np.cos(x))
        L8=legendre(8)
        P8=L6(np.cos(x))
        Legendre_series=a0*P0+a2*P2+a4*P4+a6*P6+a8*P8
        return Legendre_series
    
    def Legendre_series_full(x,a0,a1,a2,a3):
        L0=legendre(0)
        P0=L0(np.cos(x))
        L1=legendre(1)
        P1=L1(np.cos(x))
        L2=legendre(2)
        P2=L2(np.cos(x))
        L3=legendre(3)
        P3=L3(np.cos(x))
        L4=legendre(4)
        P4=L4(np.cos(x))
        L5=legendre(5)
        P5=L5(np.cos(x))
        L6=legendre(6)
        P6=L6(np.cos(x))
        #Legendre_series_full=a0*P0+a1*P1+a2*P2+a3*P3+a4*P4+a5*P5+a6*P6
        Legendre_series_full=a0*P0+a1*P1+a2*P2+a3*P3
        return Legendre_series_full
    
    def Lorentzian_Gaussian(x,a1,Ic2,xc2,omega12_2,a2,Ic1,xc1,omega12_1,):
        LG=a1*(Ic2*(1+(np.sqrt(2)-1)*((x-xc2)/omega12_2)**2)**(-2))+a2*(Ic1*np.e**(-np.log(2)*((x-xc1)/omega12_1)**2))
        return LG
    
    def Gaussian(x,Ic1,xc1,omega12_1,):
        G=Ic1*np.e**(-np.log(2)*((x-xc1)/omega12_1)**2)
        return G

    side_mode = str(side_mode or "auto").strip().lower()
    if side_mode not in {"auto", "left", "right"}:
        side_mode = "auto"

    def _pick_side(idx_max: int) -> str:
        if side_mode == "auto":
            return "right" if idx_max < Quarter + Criterium else "left"
        return side_mode
    
    color_map = shared_cmap if plot else plt.get_cmap('turbo')
    colors = color_map(np.linspace(0, 1, DataN))

    def _flatten_after_min(seg):
        if not flatten_minima or seg.size < 4:
            return seg
        try:
            sm = ndimage.uniform_filter1d(seg.astype(float), size=3, mode="nearest")
            grad = np.diff(sm)
            cut_idx = None
            for k in range(1, grad.size):
                if grad[k-1] < 0 and grad[k] >= 0:
                    cut_idx = k
                    break
            if cut_idx is None:
                return seg
            out = seg.copy()
            if flatten_tail_fit:
                frac = float(flatten_tail_extent)
                frac = min(max(frac, 0.05), 1.0)
                fit_len = max(4, int(np.ceil((cut_idx + 1) * frac)))
                fit_len = min(fit_len, cut_idx + 1)
                fit_slice = out[:fit_len]
                xloc = np.arange(len(out), dtype=float)
                xfit = np.arange(len(fit_slice), dtype=float)
                try:
                    popt, _ = curve_fit(Lorentzian_Gaussian, xfit, fit_slice, sigma=np.ones_like(fit_slice))
                    fitted = Lorentzian_Gaussian(xloc, *popt)
                    # Ensure continuity at the cut point and avoid negative tails
                    if fitted[cut_idx] != 0:
                        scale = out[cut_idx] / fitted[cut_idx]
                        fitted = fitted * scale
                    fitted = np.clip(fitted, a_min=0, a_max=None)
                    out[cut_idx+1:] = fitted[cut_idx+1:]
                except Exception:
                    out[cut_idx+1:] = out[cut_idx]
            else:
                out[cut_idx+1:] = out[cut_idx]
            return out
        except Exception:
            return seg

    if method=='fitting':

        for j in range(0,DataN):    
            #print(j)
            #if Index_MAXX[j] - Quarter <0:
            #    Y=DataNorm[int(Index_MAXX[j]+Half-Quarter):int(Index_MAXX[j]+Half+Quarter),j]/DataNorm[int(Index_MAXX[j]+Half-Quarter):int(Index_MAXX[j]+Half+Quarter),j].max(axis=0)
            #else:
            #    Y=DataNorm[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]+Quarter),j]/DataNorm[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]+Quarter),j].max(axis=0)
    
            if _pick_side(Index_MAXX[j]) == "right":
                    start_idx = int(Index_MAXX[j])
                    stop_idx = int(Index_MAXX[j]+Quarter)
                    seg = DataNorm[start_idx:stop_idx, j]
                    if np.isnan(seg).any():
                        start_idx = int(Index_MAXX[j]-Quarter)
                        stop_idx = int(Index_MAXX[j])
                        seg = DataNorm[start_idx:stop_idx, j]
                    seg = _flatten_after_min(seg)
                    peak = np.nanmax(seg)
                    if not np.isfinite(peak) or peak <= 0:
                        Y = np.zeros_like(seg)
                    else:
                        Y = seg / peak
            else:   
                    start_idx = int(Index_MAXX[j]-Quarter)
                    stop_idx = int(Index_MAXX[j])
                    seg = DataNorm[start_idx:stop_idx, j]
                    if np.isnan(seg).any():
                        start_idx = int(Index_MAXX[j])
                        stop_idx = int(Index_MAXX[j]+Quarter)
                        seg = DataNorm[start_idx:stop_idx, j]
                    seg = _flatten_after_min(seg)
                    peak = np.nanmax(seg)
                    if not np.isfinite(peak) or peak <= 0:
                        Y = np.zeros_like(seg)
                    else:
                        Y = seg / peak
                    Y = np.flip(Y)
            XRaw1 = np.arange(0,np.size(Y),1)     
            X=(XRaw1)*np.pi/Half   
            print(np.size(X))
            print(np.size(Y))
            
            
            #print(Legendre_series(X,coeff[0],coeff[1],coeff[2],coeff[3]))
            
            sigma = np.ones(len(X))
            #sigma[0] = 0.05
            sigma[0] = sigma_zero
            #sigma[1] = 0.05
            try:
                parametersL, _ = curve_fit(Lorentzian_Gaussian,X,Y, sigma=sigma)
                #print(Y)
                coeffL = parametersL
                Fit_Lor[:,j]=Lorentzian_Gaussian(X, coeffL[0], coeffL[1], coeffL[2], coeffL[3], coeffL[4], coeffL[5], coeffL[6], coeffL[7])
                FIT[:,j] = Fit_Lor[:,j]
            except RuntimeError:
                print("Lorentzian-Gaussian fit failed; switching to Legendre series")
                #popt, pcov = curve_fit(model_func, x, y, p0=(0.1 ,1e-3, 0.1), sigma=sigma)
                parameters, _ = curve_fit(Legendre_series,X,Y, sigma=sigma)
                coeff = parameters
                Fit_L0146[:,j]=Legendre_series(X,coeff[0],coeff[1],coeff[2],coeff[3],coeff[4])
                FIT[:,j] = Fit_L0146[:,j]
            #print(coeffL)
            
            #parametersG, _ = curve_fit(Gaussian,X,Y, sigma=sigma)
            #coeffG = parametersG
            #Fit_G[:,j]=Legendre_series(X,coeff[0],coeff[1],coeff[2]) 
            
            #FIT[:,j]=Fit_G;
            
            FIT[:,j]/FIT[:,j].max(axis=0)
            deg=6
            parameters_full=np.polynomial.legendre.legfit(X,Y,deg)
            coeff_full=parameters_full
            #print(coeff_full)
            Fit_L0123456[:,j]=Legendre_series_full(X,coeff_full[0],coeff_full[1],coeff_full[2],coeff_full[3])
            Int_L012=np.polynomial.legendre.legint(Y,2)
            #print(Int_L012)
            #legvander3d - FOOD FOR THOUGHT
            #for i, ax in enumerate(axs.flatten()):
            #    ax.hist(data[i])
            #    ax.set_title(f'Dataset {i+1}')
            
            
            #If the fitting moves the peak... EXTREME CASES
            #MAXX[j] = Fit_L0123456[lo_li:up_li,1].max(axis=0)
            #Index_MAXX[j] = int(Fit_L0123456[lo_li:up_li,1].argmax(axis=0))+lo_li
            #Index_MAXX_store[j]=Index_MAXX[j]
            
            #if Index_MAXX[j] < Quarter+1:
            #      Y2=Fit_L0146[int(Index_MAXX[j]):int(Index_MAXX[j]+Quarter),j]/Fit_L0146[int(Index_MAXX[j]):int(Index_MAXX[j]+Quarter),j].max(axis=0)
                    #Y=np.flip(Y)
            #else:   
            #      Y2=Fit_L0146[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]),j]/Fit_L0146[int(Index_MAXX[j]-Quarter):int(Index_MAXX[j]),j].max(axis=0)
            #      Y2=np.flip(Y)  
            #print(np.size(Y))
            #XRaw2 = np.arange(0,np.size(Y2),1)     
            #X=XRaw2*np.pi/Half
            
            
            if plot and ax_preview is not None:
                print(f"Plotting preview for column {j}, len(X)={len(X)}, len(FIT[:,j])={len(FIT[:,j])}")
                color = colors[j]
                x_preview = np.degrees(X)
                preview_x_min = min(preview_x_min, np.min(x_preview))
                preview_x_max = max(preview_x_max, np.max(x_preview))
                preview_y_min = min(preview_y_min, np.min(Y))
                preview_y_max = max(preview_y_max, np.max(Y))
                ax_preview.scatter(
                    x_preview,
                    Y,
                    s=22,
                    linewidth=0.7,
                    facecolors="none",
                    edgecolors=color,
                )
                ax_preview.plot(
                    x_preview,
                    FIT[:, j],
                    color="black",
                    linewidth=1.0,
                    alpha=0.9,
                )
        
        
            Integration_denominator[j]=integrate.trapezoid(FIT[:,j]*np.sin(X),X);
            
            Integration_numerator_P2[j]=integrate.trapezoid(FIT[:,j]*np.cos(X)**2*np.sin(X),X)
            AvgP2[j]=Integration_numerator_P2[j]/Integration_denominator[j]
        
            Integration_numerator_P4[j]=integrate.trapezoid(FIT[:,j]*np.cos(X)**4*np.sin(X),X)
            AvgP4[j]=Integration_numerator_P4[j]/Integration_denominator[j];
        
            Integration_numerator_P6[j]=integrate.trapezoid(FIT[:,j]*np.cos(X)**6*np.sin(X),X)
            AvgP6[j]=Integration_numerator_P6[j]/Integration_denominator[j]
            
            HoP[j]= (3*AvgP2[j]-1)/2; 
            P2[j] = HoP[j]
            P4[j] = (35*AvgP4[j]-30*AvgP2[j]+3)/8
            P6[j] = (231*AvgP6[j]-315*AvgP4[j]+ 105*AvgP2[j]-5)/16;
            #P8(j) = 
        
            if HoP[j]<Threshold:
                HoP[j]=0
                P4[j]=0
                P6[j]=0
              #P8[j]=
          #print(HoP[j])
    #fig, ax = plt.subplots(1,1)
    #ax.set(xlim=(0,20), ylim=(0, 5))
    #line, = ax.plot([], [], 'r-', lw=3)

    #ani = animate(fig, X, Y animate, frames=19, interval=200, repeat=False)

    elif method=='direct':
        for j in range(0, DataN):
            if _pick_side(Index_MAXX[j]) == "right":
                start_idx = int(Index_MAXX[j])
                stop_idx = int(Index_MAXX[j] + Quarter)
                seg = DataNorm[start_idx:stop_idx, j]
                Y = seg
                if np.isnan(seg).any():
                    start_idx = int(Index_MAXX[j]-Quarter)
                    stop_idx = int(Index_MAXX[j])
                    seg = DataNorm[start_idx:stop_idx, j]
                    Y = seg
                    Y = np.flip(Y)
            else:
                start_idx = int(Index_MAXX[j] - Quarter)
                stop_idx = int(Index_MAXX[j])
                seg = DataNorm[start_idx:stop_idx, j]
                if np.isnan(seg).any():
                    start_idx = int(Index_MAXX[j])
                    stop_idx = int(Index_MAXX[j]+Quarter)
                    seg = DataNorm[start_idx:stop_idx, j]
                Y = np.flip(seg)
            Y = _flatten_after_min(Y)
            Y = np.nan_to_num(Y, nan=0.0, posinf=0.0, neginf=0.0)
            XRaw1 = np.arange(0, np.size(Y), 1)
            X = XRaw1 * np.pi / Half

            if plot and ax_preview is not None:
                color = colors[j]
                x_preview = np.degrees(X)
                preview_x_min = min(preview_x_min, np.min(x_preview))
                preview_x_max = max(preview_x_max, np.max(x_preview))
                preview_y_min = min(preview_y_min, np.min(Y))
                preview_y_max = max(preview_y_max, np.max(Y))
                ax_preview.scatter(
                    x_preview,
                    Y,
                    s=22,
                    linewidth=0.7,
                    facecolors="none",
                    edgecolors=color,
                )
            Y_proc = Y

            denominator = integrate.trapezoid(Y_proc * np.sin(X), X)
            if not np.isfinite(denominator) or denominator <= 0:
                HoP[j] = 0.0
                P2[j] = 0.0
                P4[j] = 0.0
                P6[j] = 0.0
                continue

            avgP2 = integrate.trapezoid(Y_proc * np.cos(X) ** 2 * np.sin(X), X) / denominator
            avgP4 = integrate.trapezoid(Y_proc * np.cos(X) ** 4 * np.sin(X), X) / denominator
            avgP6 = integrate.trapezoid(Y_proc * np.cos(X) ** 6 * np.sin(X), X) / denominator

            # preview already plotted above with raw Y

            HoP[j] = (3 * avgP2 - 1) / 2
            P2[j] = HoP[j]
            P4[j] = (35 * avgP4 - 30 * avgP2 + 3) / 8
            P6[j] = (231 * avgP6 - 315 * avgP4 + 105 * avgP2 - 5) / 16

            if HoP[j] < Threshold:
                HoP[j] = 0.0
                P2[j] = 0.0
                P4[j] = 0.0
                P6[j] = 0.0

    
    All_in_1 = np.zeros((DataN,4))
    
    All_in_1[:,0]=HoP
    All_in_1[:,1]=P4
    All_in_1[:,2]=P6
    All_in_1[:,3]=Index_MAXX_store

    # === Finalize plotting ===
    if plot and fig is not None:
        x_time = np.arange(DataN)
        if ax_preview is not None and np.isfinite(preview_y_min):
            span_y = max(1e-6, preview_y_max - preview_y_min)
            pad_y = span_y * 0.15
            ax_preview.set_ylim(preview_y_min - pad_y, preview_y_max + pad_y)
            if np.isfinite(preview_x_min):
                span_x = max(1e-3, preview_x_max - preview_x_min)
                pad_x = span_x * 0.1
                ax_preview.set_xlim(preview_x_min - pad_x, preview_x_max + pad_x)
        if ax_pvals is not None:
            # Keep P2 colored consistently with raw/preview traces for cross-checking
            color_slice = colors[: len(x_time)]
            p2_scatter = ax_pvals.scatter(
                x_time,
                P2,
                s=27,
                c=color_slice,
                label="P2",
                edgecolors="black",
                linewidths=0.5,
            )
            p4_scatter = ax_pvals.scatter(
                x_time,
                P4,
                s=45,
                c=color_slice,
                marker="s",
                label="P4",
                edgecolors="black",
                linewidths=0.5,
            )
            p6_scatter = ax_pvals.scatter(
                x_time,
                P6,
                s=82.5,
                c=color_slice,
                marker="*",
                label="P6",
                edgecolors="black",
                linewidths=0.5,
            )
            legend_color = color_slice[0] if len(color_slice) else "black"
            legend_handles = [
                Line2D([], [], marker="o", linestyle="", markersize=10.5, markerfacecolor=legend_color, markeredgecolor="black", label="P2"),
                Line2D([], [], marker="s", linestyle="", markersize=8.5, markerfacecolor=legend_color, markeredgecolor="black", label="P4"),
                Line2D([], [], marker="*", linestyle="", markersize=11, markerfacecolor=legend_color, markeredgecolor="black", label="P6"),
            ]
            ax_pvals.legend(handles=legend_handles, loc="upper right")
        if ax_hop is not None:
            ax_hop.scatter(
                x_time,
                Index_MAXX_store,
                s=52.5,
                c=color_slice,
                edgecolors="black",
                linewidths=0.5,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fig.tight_layout()
        if save_prefix is not None:
            try:
                fig.savefig(f"{save_prefix}_P2468_overview.png", dpi=150)
            except Exception as e:
                print(f"Warning: could not save P2468 overview figure: {e}")

        # --- Force preview display ---
        try:
            plt.ion()
            fig.canvas.draw()
            fig.canvas.flush_events()
            plt.show(block=True)
            print("Interactive preview displayed.")
        except Exception as e:
            print(f"Warning: could not display interactive preview: {e}")

    return All_in_1
