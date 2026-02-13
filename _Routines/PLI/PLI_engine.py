#!/usr/bin/env python3
"""
PLI_engine.py — headless processing core for PLI_rescale_space_time_diagram

This module intentionally contains **no UI / Tk imports**. It exposes a small
API that the window file can call. Every function here is pure I/O +
computation and accepts an optional `override` dict mapping {path: time_col_idx}
so the UI can force a column without circular imports.
"""
from __future__ import annotations

import os
import csv
import codecs
import math
import re
from typing import List, Tuple, Optional, Dict

import numpy as np

# -----------------------------
# Low-level CSV + header helpers
# -----------------------------

def read_csv_rows(path: str) -> Tuple[List[List[str]], bool, str]:
    """Return (rows, prefer_comma, delimiter) reading Anton Paar CSV robustly.
    Detect UTF‑16 LE via BOM or NUL cadence; prefer TAB delimiter for Anton Paar.
    prefer_comma==True means numbers like 1,23 should be treated as 1.23.
    """
    rows: List[List[str]] = []
    prefer_comma = False
    delimiter = ','

    # sniff encoding from a small binary sample
    with open(path, 'rb') as fb:
        sampleb = fb.read(8192)

    enc = 'utf-8'
    if sampleb.startswith(codecs.BOM_UTF16_LE):
        enc = 'utf-16-le'
    elif sampleb.startswith(codecs.BOM_UTF16_BE):
        enc = 'utf-16-be'
    elif b'\x00' in sampleb[:200]:
        enc = 'utf-16-le'

    try:
        samplet = sampleb.decode(enc, errors='replace')
    except Exception:
        enc = 'utf-8'
        samplet = sampleb.decode(enc, errors='replace')

    tabs = samplet.count('\t'); semis = samplet.count(';'); commas = samplet.count(',')
    if tabs >= max(semis, commas):
        delimiter = '\t'
    elif semis >= commas:
        delimiter = ';'
    else:
        delimiter = ','
    prefer_comma = (delimiter != ',')

    with open(path, 'r', encoding=enc, errors='replace', newline='') as fh:
        reader = csv.reader(fh, delimiter=delimiter)
        rows = list(reader)

    try:
        human_delim = 'TAB' if delimiter == '\t' else delimiter
        print(f"[DBG] Detected encoding={enc}, delimiter={human_delim}, rows={len(rows)}")
    except Exception:
        pass
    return rows, prefer_comma, delimiter


def trim_leading_empty_columns(rows: List[List[str]], sample_rows: int = 300) -> List[List[str]]:
    """Remove leading columns that are empty ("", "[]", "[ ]") in the first `sample_rows` rows."""
    if not rows:
        return rows
    maxlen = max(len(r) for r in rows)
    first_non_empty = None
    for j in range(maxlen):
        for r in rows[:sample_rows]:
            cell = (r[j] if j < len(r) else '').strip()
            if cell not in ('', '[]', '[ ]'):
                first_non_empty = j
                break
        if first_non_empty is not None:
            break
    if first_non_empty in (None, 0):
        return rows
    return [ (r[first_non_empty:] if isinstance(r, list) else r) for r in rows ]


def normalize_headers(rows: List[List[str]], max_header_rows: int = 2) -> Tuple[List[str], List[List[str]]]:
    """Collapse up to `max_header_rows` header lines into a single header row, then return
    (header, data_rows)."""
    if not rows:
        return [], []
    header_rows = rows[:max_header_rows]
    data_rows = rows[max_header_rows:] if len(rows) > max_header_rows else []
    width = max(len(r) for r in header_rows) if header_rows else 0
    header: List[str] = []
    for c in range(width):
        parts = []
        for r in header_rows:
            parts.append((r[c] if c < len(r) else '').strip())
        cell = ' '.join([p for p in parts if p]).strip()
        header.append(cell)
    return header, data_rows


_tod_pat = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")


def _detect_header_start(rows: List[List[str]], seek_rows: int = 200) -> int:
    """Find the first header-like row by looking for known tokens."""
    max_i = min(seek_rows, len(rows))
    token_res = [
        re.compile(r"time\s*of\s*day", re.I),
        re.compile(r"\btime\b", re.I),
        re.compile(r"shear\s*rate", re.I),
        re.compile(r"interval\s*time", re.I),
        re.compile(r"point\s*no", re.I),
        re.compile(r"torque", re.I),
        re.compile(r"status", re.I),
    ]

    def _token_count(cells: List[str]) -> int:
        cells = [str(c) for c in cells if str(c).strip()]
        cnt = 0
        for pat in token_res:
            if any(pat.search(c) for c in cells):
                cnt += 1
        return cnt

    for i in range(max_i):
        if _token_count(rows[i]) >= 2:
            return i
    return 0

def parse_float_cell(cell: str, prefer_comma_as_decimal: bool) -> float:
    """Parse a CSV cell to float seconds.
    Supports plain floats, decimal comma, and HH:MM:SS(.sss). Returns NaN if not parseable.
    """
    if cell is None:
        return float('nan')
    s = str(cell).strip()
    if not s or s in ('[]', '[ ]', 'NA', 'NaN', 'nan'):
        return float('nan')
    try:
        return float(s)
    except Exception:
        pass
    if prefer_comma_as_decimal and (',' in s) and ('.' not in s) and (':' not in s):
        try:
            return float(s.replace(',', '.'))
        except Exception:
            return float('nan')
    if ':' in s:
        try:
            parts = s.split(':')
            if len(parts) >= 2:
                hh = int(parts[0]); mm = int(parts[1])
                ss = 0.0
                if len(parts) >= 3 and parts[2] != '':
                    ss = float(parts[2].replace(',', '.'))
                return hh*3600.0 + mm*60.0 + ss
        except Exception:
            return float('nan')
    return float('nan')


def detect_anton_paar_time_column(rows: List[List[str]], header_skip: int = 8) -> Tuple[List[List[str]], Optional[int]]:
    """Locate a header containing 'Time of Day' and return (trimmed_rows, time_idx).
    Uses a heuristic header start detector; still honors legacy header_skip as a hint.
    """
    if not rows:
        return rows, None
    start_idx = _detect_header_start(rows[max(0, header_skip - 2):])  # scan near the usual offset
    body = rows[max(0, header_skip - 2) + start_idx:]
    if not body:
        return rows, None

    header_row_idx = None
    for i, r in enumerate(body[:10]):
        if any('time of day' in str(c).lower() for c in r):
            header_row_idx = i
            break
    if header_row_idx is None:
        return rows, None
    header = body[header_row_idx]
    time_idx = None
    for j, c in enumerate(header):
        if 'time of day' in str(c).lower():
            time_idx = j
            break
    if time_idx is None:
        return rows, None
    # if two rows below is empty → try right neighbor
    try:
        if header_row_idx + 2 < len(body):
            below = body[header_row_idx + 2]
            if time_idx < len(below) and str(below[time_idx]).strip() in ('', '[]', '[ ]') and time_idx + 1 < len(header):
                time_idx = time_idx + 1
    except Exception:
        pass
    # verify by HH:MM:SS hits
    def _hits(col):
        cnt = 0
        for r in body[header_row_idx+1: header_row_idx+11]:
            if col < len(r) and _tod_pat.search(str(r[col]).strip()):
                cnt += 1
        return cnt
    hits = _hits(time_idx)
    if hits == 0 and time_idx - 1 >= 0:
        if _hits(time_idx - 1) > 0:
            time_idx = time_idx - 1
    trimmed = body[header_row_idx:]
    return trimmed, time_idx


def auto_convert_anton_paar(path: str, header_skip: int = 8) -> str:
    """If the file is Anton Paar (TAB/UTF-16LE), convert to a clean UTF‑8, comma-separated CSV
    with collapsed headers and an added Time_s_rel_t0 column. Return new path or original.
    """
    try:
        rows, prefer_comma, delim = read_csv_rows(path)
        if not rows or delim != '\t':  # only convert Anton Paar TAB files
            return path
        rows = trim_leading_empty_columns(rows, sample_rows=300)
        rows2, pre_time_idx = detect_anton_paar_time_column(rows, header_skip=header_skip)
        if not rows2:
            return path
        header, data_rows = normalize_headers(rows2, max_header_rows=2)
        time_idx = pre_time_idx
        if time_idx is None:
            for i, h in enumerate(header):
                hl = (h or '').lower(); hl = ' '.join(hl.split())
                if ('time of day' in hl) or ('time' in hl and 'deriv' not in hl):
                    time_idx = i; break
        if time_idx is None or not data_rows:
            return path
        tsec = []
        for r in data_rows:
            v = parse_float_cell(r[time_idx] if time_idx < len(r) else '', prefer_comma)
            tsec.append(v if np.isfinite(v) else np.nan)
        finite_idx = np.where(np.isfinite(tsec))[0]
        if finite_idx.size == 0:
            return path
        t0 = float(tsec[int(finite_idx[0])])
        time_rel = [(float(v) - t0) if np.isfinite(v) else '' for v in tsec]
        out_path = os.path.splitext(path)[0] + '_clean.csv'
        clean_header = header[:] + ['Time_s_rel_t0']
        clean_rows = []
        width = len(header)
        for r, tr in zip(data_rows, time_rel):
            row = r[:] + [''] * max(0, width - len(r))
            row.append(f"{tr:.6f}" if isinstance(tr, (int, float)) and tr != '' else '')
            clean_rows.append(row)
        with open(out_path, 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(clean_header)
            w.writerows(clean_rows)
        print(f"[DBG] Wrote clean Anton Paar CSV → {out_path}")
        return out_path
    except Exception:
        return path


# -----------------------------
# Time-of-Day column selection
# -----------------------------

def pick_time_column_around(idx: int, data_rows: List[List[str]], prefer_comma: bool,
                             scan_rows: int = 300, min_hits: int = 5) -> Optional[int]:
    """Try [idx-1, idx, idx+1] and pick the best column by (colon_count, finite_count, median_value)."""
    if not isinstance(idx, int) or not data_rows:
        return None
    ncols = max(len(r) for r in data_rows)
    candidates = [c for c in (idx-1, idx, idx+1) if 0 <= c < ncols]
    best = None
    best_key = (-1, -1, -1.0)
    rows = data_rows[:max(1, scan_rows)]
    for j in candidates:
        colon = 0; finite = 0; vals = []
        for r in rows:
            s = str(r[j] if j < len(r) else '').strip()
            if ':' in s:
                colon += 1
            v = parse_float_cell(s, prefer_comma)
            if np.isfinite(v):
                finite += 1; vals.append(v)
        med = float(np.median(vals)) if vals else -1.0
        key = (colon, finite, med)
        if key > best_key and finite >= min_hits:
            best_key = key
            best = j
    return best


def detect_time_column_index(path: str, header_skip: int = 8) -> Optional[int]:
    """Best-effort automatic detection of the Time-of-Day column index for the given CSV."""
    try:
        rows, prefer_comma, _delim = read_csv_rows(path)
    except Exception as exc:
        print(f"[DBG] detect_time_column_index read failed for {path}: {exc}")
        return None
    rows = trim_leading_empty_columns(rows, sample_rows=300)
    if not rows:
        return None

    header_start = _detect_header_start(rows, seek_rows=200)
    rows2 = rows[header_start:]
    pre_rows, pre_time_idx = detect_anton_paar_time_column(rows, header_skip=header_skip)
    # Prefer the Anton Paar detector if it found something; else use header_start slice
    if pre_time_idx is not None:
        rows2 = pre_rows

    header, data_rows = normalize_headers(rows2, max_header_rows=2)

    time_idx: Optional[int] = pre_time_idx
    if time_idx is None:
        for i, h in enumerate(header):
            hl = (h or '').lower(); hl = ' '.join(hl.split())
            if ('time of day' in hl) or ('time' in hl and 'deriv' not in hl):
                time_idx = i
                break

    # Fallback: pick column with most HH:MM:SS hits
    if time_idx is None and data_rows:
        width = max(len(r) for r in data_rows[:50])
        best = None; best_hits = -1
        for j in range(width):
            hits = 0
            for r in data_rows[:300]:
                if j < len(r) and _tod_pat.search(str(r[j]).strip()):
                    hits += 1
            if hits > best_hits and hits >= 3:
                best_hits = hits; best = j
        if best is not None:
            time_idx = best

    if time_idx is None:
        return None

    if data_rows:
        best = pick_time_column_around(time_idx, data_rows, prefer_comma, scan_rows=300, min_hits=3)
        if isinstance(best, int):
            time_idx = best
    return time_idx


# -----------------------------
# Interval extraction (Triggered / Reference) and advances
# -----------------------------

def _times_from_column(rows: List[List[str]], time_idx: int, prefer_comma: bool) -> List[float]:
    t: List[float] = []
    for r in rows:
        s = r[time_idx] if time_idx < len(r) else ''
        v = parse_float_cell(s, prefer_comma)
        t.append(v if np.isfinite(v) else float('nan'))
    return t


def _relative_seconds(times: List[float]) -> List[float]:
    finite = [v for v in times if np.isfinite(v)]
    if not finite:
        return [float('nan')] * len(times)
    t0 = finite[0]
    return [ (v - t0) if np.isfinite(v) else float('nan') for v in times ]


def extract_triggered_pairs_from_time_column(path: str, override: Optional[Dict[str, int]] = None
                                            ) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Return (T_beg, T_end, S_beg, S_end) in seconds (relative t0) for Triggered mode.
    Uses the Time-of-Day column only; splits numeric runs by NaN gaps into (transient, steady) pairs.
    If override is provided and contains path, that column index is used.
    """
    T_beg: List[float] = []; T_end: List[float] = []; S_beg: List[float] = []; S_end: List[float] = []
    rows, prefer_comma, _delim = read_csv_rows(path)
    rows = trim_leading_empty_columns(rows, sample_rows=300)
    if not rows:
        return T_beg, T_end, S_beg, S_end
    rows2, pre_time_idx = detect_anton_paar_time_column(rows, header_skip=8)
    header, data_rows = normalize_headers(rows2, max_header_rows=2)

    time_idx = pre_time_idx
    if time_idx is None:
        for i, h in enumerate(header):
            hl = (h or '').lower(); hl = ' '.join(hl.split())
            if ('time of day' in hl) or ('time' in hl and 'deriv' not in hl):
                time_idx = i; break

    # Manual override (takes precedence)
    if override and path in override:
        try:
            time_idx = int(override[path])
            print("[DBG] time_idx overridden by user →", time_idx)
        except Exception:
            pass

    # Refine ±1 if not overridden
    if (not (override and path in override)) and time_idx is not None:
        best = pick_time_column_around(time_idx, data_rows, prefer_comma, scan_rows=300, min_hits=5)
        if isinstance(best, int):
            time_idx = best

    if time_idx is None:
        return T_beg, T_end, S_beg, S_end

    # Build relative seconds
    t_abs = _times_from_column(data_rows, time_idx, prefer_comma)
    t_rel = _relative_seconds(t_abs)

    # Build numeric runs separated by NaN (raw)
    raw_runs: List[Tuple[int,int]] = []
    i = 0; N = len(t_rel)
    while i < N:
        if not np.isfinite(t_rel[i]):
            i += 1; continue
        a = i
        while i < N and np.isfinite(t_rel[i]):
            i += 1
        b = i - 1
        raw_runs.append((a, b))

    # Helper: does the gap between two runs contain any non-numeric text in the time column?
    def _gap_contains_text(p_end: int, n_start: int) -> bool:
        if p_end + 1 >= n_start:
            return False
        for r in range(p_end + 1, n_start):
            s = str(data_rows[r][time_idx] if time_idx < len(data_rows[r]) else '').strip()
            if not s:
                continue  # pure blank → ignore
            v = parse_float_cell(s, prefer_comma)
            if not np.isfinite(v):
                # non-numeric, non-empty cell → treat as header/text separator
                return True
        return False

    # Merge consecutive runs unless the gap contains header/text
    merged: List[Tuple[int,int]] = []
    for a, b in raw_runs:
        if not merged:
            merged.append((a, b)); continue
        pa, pb = merged[-1]
        if _gap_contains_text(pb, a):
            merged.append((a, b))  # keep boundary
        else:
            # merge with previous run (ignore blank-only gaps)
            merged[-1] = (pa, b)

    # Drop ultra-short noise segments (duration < 0.5 s)
    runs: List[Tuple[int,int]] = []
    for a, b in merged:
        if np.isfinite(t_rel[a]) and np.isfinite(t_rel[b]) and (t_rel[b] - t_rel[a]) >= 0.5:
            runs.append((a, b))

    # Pair runs: (transient, steady)
    for k in range(0, len(runs) - 1, 2):
        a0, b0 = runs[k]
        a1, b1 = runs[k+1]
        tb = t_rel[a0]; te = t_rel[b0]
        sb = t_rel[a1]; se = t_rel[b1]
        if np.isfinite(tb) and np.isfinite(te) and np.isfinite(sb) and np.isfinite(se) and te > tb and se > sb:
            T_beg.append(float(tb)); T_end.append(float(te)); S_beg.append(float(sb)); S_end.append(float(se))

    try:
        n_pairs = min(len(T_beg), len(T_end), len(S_beg), len(S_end))
        print(f"[DBG] EXTRACTED Triggered pairs (s, relative t0) — total={n_pairs}")
        print(f"  T_beg: head={T_beg[:5]}  tail={T_beg[-5:] if n_pairs>5 else T_beg}")
        print(f"  T_end: head={T_end[:5]}  tail={T_end[-5:] if n_pairs>5 else T_end}")
        print(f"  S_beg: head={S_beg[:5]}  tail={S_beg[-5:] if n_pairs>5 else S_beg}")
        print(f"  S_end: head={S_end[:5]}  tail={S_end[-5:] if n_pairs>5 else S_end}")
    except Exception:
        pass
    return T_beg, T_end, S_beg, S_end


def parse_reference_steps_from_csv(path: str, override: Optional[Dict[str, int]] = None
                                  ) -> Tuple[List[float], List[float]]:
    """Return (begins_s, ends_s) for Reference mode.
    This uses contiguous numeric runs in the Time-of-Day column as segments.
    If override is provided and contains path, that column index is used.
    """
    begins_s: List[float] = []; ends_s: List[float] = []
    rows, prefer_comma, _delim = read_csv_rows(path)
    rows = trim_leading_empty_columns(rows, sample_rows=300)
    if not rows:
        return begins_s, ends_s
    rows2, pre_time_idx = detect_anton_paar_time_column(rows, header_skip=8)
    header, data_rows = normalize_headers(rows2, max_header_rows=2)

    time_idx = pre_time_idx
    if time_idx is None:
        for i, h in enumerate(header):
            hl = (h or '').lower(); hl = ' '.join(hl.split())
            if ('time of day' in hl) or ('time' in hl and 'deriv' not in hl):
                time_idx = i; break

    if override and path in override:
        try:
            time_idx = int(override[path])
            print("[DBG] time_idx overridden by user →", time_idx)
        except Exception:
            pass

    if time_idx is None:
        return begins_s, ends_s

    t_abs = _times_from_column(data_rows, time_idx, prefer_comma)
    t_rel = _relative_seconds(t_abs)

    # Build numeric runs (raw)
    raw_runs: List[Tuple[int,int]] = []
    i = 0; N = len(t_rel)
    while i < N:
        if not np.isfinite(t_rel[i]):
            i += 1; continue
        a = i
        while i < N and np.isfinite(t_rel[i]):
            i += 1
        b = i - 1
        raw_runs.append((a, b))

    # Helper: does the gap between two runs contain any non-numeric text in the time column?
    def _gap_contains_text(p_end: int, n_start: int) -> bool:
        if p_end + 1 >= n_start:
            return False
        for r in range(p_end + 1, n_start):
            s = str(data_rows[r][time_idx] if time_idx < len(data_rows[r]) else '').strip()
            if not s:
                continue  # pure blank → ignore
            v = parse_float_cell(s, prefer_comma)
            if not np.isfinite(v):
                # header/units or any non-numeric token
                return True
        return False

    # Merge runs unless the gap contains header/text
    merged: List[Tuple[int,int]] = []
    for a, b in raw_runs:
        if not merged:
            merged.append((a, b)); continue
        pa, pb = merged[-1]
        if _gap_contains_text(pb, a):
            merged.append((a, b))
        else:
            merged[-1] = (pa, b)

    # Drop ultra-short noise segments (< 0.5 s)
    runs: List[Tuple[int,int]] = []
    for a, b in merged:
        if np.isfinite(t_rel[a]) and np.isfinite(t_rel[b]) and (t_rel[b] - t_rel[a]) >= 0.5:
            runs.append((a, b))

    for a, b in runs:
        begins_s.append(float(t_rel[a]))
        ends_s.append(float(t_rel[b]))
    return begins_s, ends_s


def compute_pair_advances_from_times(path: str, fps: float, override: Optional[Dict[str, int]] = None
                                    ) -> Tuple[List[float], List[float]]:
    """Compute cumulative pair advances and boundary seconds from Time-of-Day column.
    Uses the same text-aware gap merge as the extractors so segments reflect real steps.
    Returns (advances_in_frames_per_pair, boundary_seconds). If no time column found, returns ([], []).
    """
    rows, prefer_comma, _delim = read_csv_rows(path)
    rows = trim_leading_empty_columns(rows, sample_rows=300)
    if not rows:
        return [], []
    rows2, pre_time_idx = detect_anton_paar_time_column(rows, header_skip=8)
    header, data_rows = normalize_headers(rows2, max_header_rows=2)

    time_idx = pre_time_idx
    if time_idx is None:
        for i, h in enumerate(header):
            hl = (h or '').lower(); hl = ' '.join(hl.split())
            if ('time of day' in hl) or ('time' in hl and 'deriv' not in hl):
                time_idx = i; break

    if override and path in override:
        try:
            time_idx = int(override[path])
            print("[DBG] time_idx overridden by user →", time_idx)
        except Exception:
            pass

    if time_idx is None:
        return [], []

    t_abs = _times_from_column(data_rows, time_idx, prefer_comma)
    t_rel = _relative_seconds(t_abs)

    # Build numeric runs (raw)
    raw_runs: List[Tuple[int,int]] = []
    i = 0; N = len(t_rel)
    while i < N:
        if not np.isfinite(t_rel[i]):
            i += 1; continue
        a = i
        while i < N and np.isfinite(t_rel[i]):
            i += 1
        b = i - 1
        raw_runs.append((a, b))

    # Helper: gap contains any non-numeric text in the time column?
    def _gap_contains_text(p_end: int, n_start: int) -> bool:
        if p_end + 1 >= n_start:
            return False
        for r in range(p_end + 1, n_start):
            s = str(data_rows[r][time_idx] if time_idx < len(data_rows[r]) else '').strip()
            if not s:
                continue
            v = parse_float_cell(s, prefer_comma)
            if not np.isfinite(v):
                return True
        return False

    # Merge runs unless the gap contains header/text; drop ultra-short noise (<0.5 s)
    merged: List[Tuple[int,int]] = []
    for a, b in raw_runs:
        if not merged:
            merged.append((a, b)); continue
        pa, pb = merged[-1]
        if _gap_contains_text(pb, a):
            merged.append((a, b))
        else:
            merged[-1] = (pa, b)

    runs: List[Tuple[int,int]] = []
    for a, b in merged:
        if np.isfinite(t_rel[a]) and np.isfinite(t_rel[b]) and (t_rel[b] - t_rel[a]) >= 0.5:
            runs.append((a, b))

    # Now compute per-segment advances (frames) and boundaries (seconds)
    advances_frames: List[float] = []
    boundaries_sec: List[float] = []
    for a, b in runs:
        begin = t_rel[a]; end = t_rel[b]
        if np.isfinite(begin) and np.isfinite(end) and end > begin:
            advances_frames.append((end - begin) * float(fps))
            boundaries_sec.append(begin)

    return advances_frames, boundaries_sec

def compute_reference_intervals_with_steady(b_ref: List[float], e_ref: List[float], steady_duration: float) -> Tuple[List[float], List[float], List[float], List[float]]:
    """Compute (T_beg, T_end, S_beg, S_end) for Reference mode by subtracting fixed steady width (seconds) from each interval end.
    steady_duration: duration of steady state in seconds (converted from scaled width px / fps)."""
    T_beg: List[float] = []
    T_end: List[float] = []
    S_beg: List[float] = []
    S_end: List[float] = []
    n = min(len(b_ref), len(e_ref))
    for i in range(n):
        bb = float(b_ref[i]); ee = float(e_ref[i])
        sb = max(bb, ee - steady_duration)
        se = ee
        T_beg.append(bb)
        T_end.append(sb)
        S_beg.append(sb)
        S_end.append(se)
    return T_beg, T_end, S_beg, S_end
