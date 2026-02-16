import os, csv, codecs, re, json, math, sys
from bisect import bisect_left, bisect_right
from typing import List, Tuple, Optional, Dict

STEADY_DELAY_SEC = 2.0  # seconds trimmed from end in non-triggered steady averaging

# --- UI/plotting imports ---
import matplotlib
if os.environ.get("MUDPAW_EMBED") == "1" or "PyQt6" in sys.modules:
    matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

def _finite_count(a: Optional[List[float]]) -> int:
    if a is None:
        return 0
    cnt = 0
    for v in a:
        try:
            if math.isfinite(v):
                cnt += 1
        except Exception:
            pass
    return cnt

# (launcher removed; this module is a standalone CLI reader)

# -----------------
# Path helpers
# -----------------

def _split_parts(path: str):
    norm = os.path.normpath(path)
    return [p for p in norm.split(os.sep) if p not in ("", ".")]


def _sanitize_misjoined_user_path(p: str) -> str:
    try:
        tok = os.sep + "Users" + os.sep
        first = p.find(tok)
        if first == -1:
            return p
        second = p.find(tok, first + 1)
        if second == -1:
            return p
        fixed = p[second:]
        if not fixed.startswith(os.sep):
            fixed = os.sep + fixed
        return fixed
    except Exception:
        return p


def get_reference_folder_from_path(path: str) -> str:
    markers = {"PLI", "PI", "SAXS", "Rheology"}
    abspath = _sanitize_misjoined_user_path(os.path.abspath(os.path.realpath(path)))
    if os.path.isfile(abspath):
        abspath = os.path.dirname(abspath)
    parts = _split_parts(abspath)
    if "_Processed" in parts:
        idx = len(parts) - 1 - parts[::-1].index("_Processed")
        return os.path.join(os.sep, *parts[:idx])
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] in markers:
            return os.path.join(os.sep, *parts[:i])
    return os.path.dirname(abspath)


def get_rheology_folder_from_path(path: str) -> Optional[str]:
    abspath = _sanitize_misjoined_user_path(os.path.abspath(os.path.realpath(path)))
    if os.path.isfile(abspath):
        abspath = os.path.dirname(abspath)
    parts = _split_parts(abspath)
    if parts and parts[-1] == "Rheology" and os.path.isdir(abspath):
        return abspath
    if "_Processed" in parts:
        idx = len(parts) - 1 - parts[::-1].index("_Processed")
        ref = os.path.join(os.sep, *parts[:idx])
        cand = os.path.join(ref, "Rheology")
        return cand if os.path.isdir(cand) else None
    ref = get_reference_folder_from_path(abspath)
    cand = os.path.join(ref, "Rheology")
    return cand if os.path.isdir(cand) else None


# -----------------
# CSV open + header detection
# -----------------

def _read_csv_rows(path: str) -> Tuple[List[List[str]], str, str]:
    """Return (rows, encoding, delimiter). Detect UTF-16LE and TAB/;/,"""
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
        enc = 'utf-8'; samplet = sampleb.decode(enc, errors='replace')
    tabs = samplet.count('\t'); semis = samplet.count(';'); commas = samplet.count(',')
    if tabs >= max(semis, commas):
        delim = '\t'
    elif semis >= commas:
        delim = ';'
    else:
        delim = ','
    rows: List[List[str]] = []
    with open(path, 'r', encoding=enc, errors='replace', newline='') as fh:
        rdr = csv.reader(fh, delimiter=delim)
        rows = list(rdr)
    return rows, enc, delim


# --- Preview plot helper ---
def _show_plot_window(path: str,
                      t_sec: Optional[List[float]],
                      rate_sec: Optional[List[float]],
                      visc_sec: Optional[List[float]],
                      stress_sec: Optional[List[float]],
                      T_beg: List[float], T_end: List[float],
                      S_beg: List[float], S_end: List[float],
                      *,
                      steady_rates: Optional[List[Optional[float]]] = None,
                      steady_viscs: Optional[List[Optional[float]]] = None,
                      steady_sec: Optional[float] = None,
                      allow_refresh: bool = False,
                      refresh_callback = None,
                      allow_crop: bool = False,
                      crop_callback = None,
                      crop_limits: Optional[Tuple[float, float]] = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax_time = axes[0]
    ax_rate = ax_time.twinx()
    ax_steady = axes[1]

    if t_sec is None:
        t_plot = list(range(len(rate_sec or visc_sec or [])))
        ax_time.set_xlabel("sample index")
    else:
        t_plot = t_sec
        ax_time.set_xlabel("time (s)")

    def _clean_pos(series, positive_only=True):
        if series is None:
            return None
        out = []
        for v in series:
            try:
                ok = (v is not None and math.isfinite(v))
                if positive_only:
                    ok = ok and (v > 0)
                out.append(v if ok else math.nan)
            except Exception:
                out.append(math.nan)
        return out

    rate_clean = _clean_pos(rate_sec)
    visc_clean = _clean_pos(visc_sec)

    any_plotted = False
    if rate_clean is not None and any(math.isfinite(v) for v in rate_clean):
        ax_time.plot(t_plot, rate_clean, label="Shear rate [1/s]", linewidth=1.0)
        ax_time.set_ylabel("Shear rate [1/s]")
        ax_time.set_yscale('log')
        any_plotted = True
    if visc_clean is not None and any(math.isfinite(v) for v in visc_clean):
        ax_rate.plot(t_plot, visc_clean, label="Viscosity [Pa·s]", linewidth=1.0, alpha=0.9, color="tab:orange")
        ax_rate.set_ylabel("Viscosity [Pa·s]")
        ax_rate.set_yscale('log')
        any_plotted = True
    if stress_sec is not None and any(math.isfinite(v) for v in stress_sec):
        stress_clean = _clean_pos(stress_sec, positive_only=False)
        ax_rate.plot(t_plot, stress_clean, label="Shear stress [Pa]", linewidth=1.0, linestyle='--', alpha=0.9, color="tab:green")
        any_plotted = True

    if not any_plotted:
        ax_time.text(0.5, 0.5, "No plottable data detected", transform=ax_time.transAxes, ha='center', va='center')

    try:
        ax_time.legend(loc='upper left')
    except Exception:
        pass
    try:
        ax_rate.legend(loc='upper right')
    except Exception:
        pass

    for x in T_beg:
        if x is not None:
            ax_time.axvline(x, color='cyan', linewidth=1.0)
    for x in S_end:
        if x is not None:
            ax_time.axvline(x, color='red', linewidth=1.0)

    ax_time.set_title("Time traces — cyan=T_begin, red=S_end")

    scatter_points = []
    if steady_rates and steady_viscs:
        for r, v in zip(steady_rates, steady_viscs):
            try:
                if r is not None and v is not None and math.isfinite(r) and math.isfinite(v) and r > 0 and v > 0:
                    scatter_points.append((r, v))
            except Exception:
                continue
    if scatter_points:
        xs, ys = zip(*scatter_points)
        try:
            import numpy as _np
            cmap = plt.get_cmap("turbo")
            colors = cmap(_np.linspace(0, 1, max(10, len(xs))))
            ax_steady.scatter(xs, ys, c=colors)
        except Exception:
            ax_steady.scatter(xs, ys, color="tab:blue")
        ax_steady.set_xscale('log')
        ax_steady.set_yscale('log')
        ax_steady.set_xlabel("Shear rate [1/s]")
        ax_steady.set_ylabel("Steady viscosity [Pa·s]")
        ax_steady.set_title("Steady-state viscosity vs shear rate")
        ax_steady.grid(True, which='both', alpha=0.2)
    else:
        ax_steady.text(0.5, 0.5, "No steady averages computed", transform=ax_steady.transAxes,
                       ha='center', va='center')

    fig.tight_layout()
    try:
        if not (os.environ.get("MUDPAW_EMBED") == "1" or "PyQt6" in sys.modules):
            plt.show()
    except Exception:
        pass

def _normalize_headers(rows: List[List[str]], max_header_rows: int = 2) -> Tuple[List[str], List[List[str]]]:
    if not rows:
        return [], []
    hdr_rows = rows[:max_header_rows]
    data_rows = rows[max_header_rows:] if len(rows) > max_header_rows else []
    width = max(len(r) for r in hdr_rows) if hdr_rows else 0
    header: List[str] = []
    for c in range(width):
        parts = []
        for r in hdr_rows:
            parts.append((r[c] if c < len(r) else '').strip())
        cell = ' '.join([p for p in parts if p]).strip()
        header.append(cell)
    return header, data_rows

# --- Additional helpers to handle Anton Paar exports ---
def _trim_leading_empty_columns(rows: List[List[str]], sample_rows: int = 20) -> List[List[str]]:
    """Remove leading columns that are entirely empty (or "[]") across the first `sample_rows` lines.
    This handles the common Anton Paar case where the first column is blank.
    """
    if not rows:
        return rows
    maxw = max((len(r) for r in rows), default=0)
    start_col = 0
    for j in range(maxw):
        all_empty = True
        for r in rows[: min(len(rows), sample_rows)]:
            if j < len(r):
                s = str(r[j]).strip()
                if s not in ("", "[]"):
                    all_empty = False
                    break
        if all_empty:
            start_col += 1
        else:
            break
    if start_col > 0:
        return [r[start_col:] for r in rows]
    return rows

def _detect_header_start(rows: List[List[str]], seek_rows: int = 400) -> int:
    """Find the first *label* header row of the Anton Paar table.
    Heuristics:
      1) If a row contains 'Interval data:' **and** ≥2 known label tokens (Point No., Time of Day, Shear Rate, ...),
         treat that same row as the header (labels often live on the marker line).
      2) Otherwise, search forward for the next row that contains ≥2 known label tokens.
      3) Skip rows that look like *units* only (mostly brackets like [s], [Pa], [1/s]) or numeric-only rows.
    Returns the index of the chosen header row.
    """
    max_i = min(seek_rows, len(rows))

    label_tokens = [
        re.compile(r"point\s*no\.?", re.I),
        re.compile(r"time\s*of\s*day", re.I),
        re.compile(r"shear\s*rate", re.I),
        re.compile(r"shear\s*stress", re.I),
        re.compile(r"viscos", re.I),
        re.compile(r"interval\s*time", re.I),
        re.compile(r"test\s*time", re.I),
        re.compile(r"torque", re.I),
        re.compile(r"status", re.I),
    ]

    def is_units_row(cells: List[str]) -> bool:
        cells = [str(c).strip() for c in cells if str(c).strip()]
        if not cells:
            return False
        # Units rows have lots of brackets and few letters
        brackety = sum(1 for c in cells if ('[' in c or ']' in c))
        letters  = sum(1 for c in cells if re.search(r"[A-Za-z]", c))
        digits   = sum(1 for c in cells if re.search(r"^[0-9eE+\-.,]+$", c))
        return brackety >= max(2, len(cells)//2) and letters <= len(cells)//2 and digits <= len(cells)//2

    def token_count(cells: List[str]) -> int:
        cells = [str(c) for c in cells if str(c).strip()]
        cnt = 0
        for pat in label_tokens:
            if any(pat.search(c) for c in cells):
                cnt += 1
        return cnt

    # Pass 1: look for marker row that *already* contains labels
    for i in range(max_i):
        row = rows[i]
        has_marker = any(re.search(r"\binterval\s*data\b", str(c), re.I) for c in row)
        if has_marker and token_count(row) >= 2:
            return i

    # Pass 2: look for next label row; skip pure units rows
    for i in range(max_i):
        row = rows[i]
        if token_count(row) >= 2 and not is_units_row(row):
            return i

    # Fallback: original heuristic on any token
    keys = [re.compile(r"time\s*of\s*day", re.I), re.compile(r"\bshear\b", re.I), re.compile(r"viscos", re.I)]
    for i in range(max_i):
        line = " ".join([str(c) for c in rows[i] if str(c).strip()])
        if any(k.search(line) for k in keys):
            return i
    return 0


# -----------------
# Column auto-detection
# -----------------

_COL_PATTERNS = {
    'time_of_day': re.compile(r'time\s*of\s*day', re.I),
    'time': re.compile(r'^(time)(?!\s*of\s*day)', re.I),
    'shear_rate': re.compile(r'(shear\s*rate|\b1\s*/\s*s\b|s\s*\^\s*-?1)', re.I),
    'viscosity': re.compile(r'(viscos|η|\beta\b|nu\b)', re.I),  # η or 'eta' or 'nu' as rare variants
    'shear_stress': re.compile(r'(shear\s*stress|\btau\b)', re.I),
}
def _fallback_find_cols(header: List[str], idxs: Dict[str, Optional[int]]) -> Dict[str, Optional[int]]:
    # If viscosity not found, look for unit tokens and avoid stress columns
    if idxs.get('viscosity') is None:
        for i, h in enumerate(header):
            hl = (h or '').lower()
            if ('pa·s' in hl) or ('pa*s' in hl) or ('pas' in hl) or ('mpa·s' in hl) or ('mpa*s' in hl) or ('mpas' in hl) or (' η' in hl) or (' eta' in hl):
                if 'stress' not in hl:  # avoid shear stress
                    idxs['viscosity'] = i
                    break
    # If shear rate not found, look for [1/s] or s^-1 tokens
    if idxs.get('shear_rate') is None:
        for i, h in enumerate(header):
            hl = (h or '').lower()
            if ('[1/s]' in hl) or ('1/s' in hl) or ('s^-1' in hl) or ('s-1' in hl):
                idxs['shear_rate'] = i
                break
    return idxs


def _find_columns(header: List[str]) -> Dict[str, Optional[int]]:
    idxs: Dict[str, Optional[int]] = {k: None for k in _COL_PATTERNS.keys()}
    for i, h in enumerate(header):
        hl = (h or '').strip()
        for key, pat in _COL_PATTERNS.items():
            if idxs[key] is None and pat.search(hl):
                idxs[key] = i
    return idxs


def _to_float(s: str) -> float:
    s = (s or '').strip()
    if not s or s in ('[]', '[ ]', 'NaN', 'nan'):
        return math.nan
    try:
        return float(s)
    except Exception:
        pass
    # time-of-day like HH:MM:SS(.sss)
    if ':' in s:
        try:
            hh, mm, ss = s.split(':')[0], s.split(':')[1], s.split(':')[2]
            return int(hh)*3600 + int(mm)*60 + float(ss.replace(',', '.'))
        except Exception:
            return math.nan
    # decimal comma fallback
    if ',' in s and '.' not in s:
        try:
            return float(s.replace(',', '.'))
        except Exception:
            return math.nan
    return math.nan


# -----------------
# Segmentation helpers
# -----------------

def _build_time_series(data_rows: List[List[str]], col: int) -> List[float]:
    out = []
    for r in data_rows:
        v = _to_float(r[col] if col < len(r) else '')
        out.append(v if math.isfinite(v) else math.nan)
    # make relative to first finite
    finite = [v for v in out if math.isfinite(v)]
    if finite:
        t0 = finite[0]
        out = [(v - t0) if math.isfinite(v) else math.nan for v in out]
    return out


# --- DataGraph-friendly steady CSV writer ---
def _write_steady_csv(path: str, rates: List[Optional[float]], taus: List[Optional[float]], viscs: List[Optional[float]]):
    ref = get_reference_folder_from_path(path)
    out_dir = os.path.join(ref, "_Processed")
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    csv_path = os.path.join(out_dir, f"Rheo_steady_{base}.csv")

    def _fmt(v: Optional[float], keep_blank: bool = False):
        try:
            if v is None:
                return "" if keep_blank else "NaN"
            if math.isfinite(float(v)):
                return v
        except Exception:
            pass
        return "NaN"

    try:
        with open(csv_path, 'w', encoding='utf-8', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(["ShearRate [1/s]", "ShearStress [Pa]", "Viscosity [Pa·s]"])
            for r, t, v in zip(rates, taus, viscs):
                w.writerow([_fmt(r, keep_blank=True), _fmt(t), _fmt(v)])
        print("[Rheology] wrote steady CSV:", csv_path)
    except Exception as e:
        print("[Rheology] steady CSV write failed:", e)


def _gap_contains_text(data_rows: List[List[str]], time_idx: int, p_end: int, n_start: int) -> bool:
    if p_end + 1 >= n_start:
        return False
    for r in range(p_end + 1, n_start):
        s = (data_rows[r][time_idx] if time_idx < len(data_rows[r]) else '').strip()
        if not s:
            continue
        v = _to_float(s)
        if not math.isfinite(v):
            return True
    return False


def _segments_triggered(data_rows: List[List[str]], time_idx: int, min_run_sec: float = 0.5) -> List[Tuple[int, int]]:
    # raw runs
    t = _build_time_series(data_rows, time_idx)
    raw: List[Tuple[int,int]] = []
    i = 0; N = len(t)
    while i < N:
        if not math.isfinite(t[i]):
            i += 1; continue
        a = i
        while i < N and math.isfinite(t[i]):
            i += 1
        b = i - 1
        raw.append((a, b))
    # merge blank-only gaps, keep text gaps, drop very short runs (<0.5 s)
    merged: List[Tuple[int,int]] = []
    for a, b in raw:
        if not merged:
            merged.append((a, b)); continue
        pa, pb = merged[-1]
        if _gap_contains_text(data_rows, time_idx, pb, a):
            merged.append((a, b))
        else:
            merged[-1] = (pa, b)
    runs: List[Tuple[int,int]] = []
    for a, b in merged:
        if math.isfinite(t[a]) and math.isfinite(t[b]) and (t[b] - t[a]) >= min_run_sec:
            runs.append((a, b))
    return runs


def _steps_nontriggered(data_rows: List[List[str]], rate_idx: int, time_idx: Optional[int], rate_thresh: float = 5.0) -> List[Tuple[int, int]]:
    # basic step detection by |Δrate| > threshold; return [ (begin_idx, end_idx) ] per step
    vals = []
    for r in data_rows:
        v = _to_float(r[rate_idx] if rate_idx < len(r) else '')
        vals.append(v if math.isfinite(v) else math.nan)
    # compute diff on finite subsequence
    diffs = []
    last = float('nan')
    for v in vals:
        if math.isfinite(v) and math.isfinite(last):
            diffs.append(abs(v - last))
        else:
            diffs.append(0.0)
        if math.isfinite(v):
            last = v
    # segment at large diffs
    segments = []
    N = len(vals)
    a = 0
    for i, d in enumerate(diffs):
        if d >= rate_thresh and i > a:
            segments.append((a, i-1))
            a = i
    if a < N-1:
        segments.append((a, N-1))
    return [(aa, bb) for (aa, bb) in segments if bb > aa]



# -----------------
# Marker segmentation helper
# -----------------

def _segments_by_markers(data_rows: List[List[str]]) -> List[Tuple[int, int]]:
    """Detect segments by repeated Anton Paar interval/table headers inside the body.
    We look for rows containing 'Interval data:' or that look like column headers
    (contain ≥2 known tokens), and treat the data between two successive markers
    as one segment. We also skip the 1–2 header/unit rows immediately following
    each marker.
    Returns a list of (start_idx, end_idx) within data_rows.
    """
    if not data_rows:
        return []
    # Compile patterns once
    marker_pat = re.compile(r"\binterval\s*data\b", re.I)
    token_res = [
        re.compile(r"point\s*no\.?", re.I),
        re.compile(r"time\s*of\s*day", re.I),
        re.compile(r"shear\s*rate", re.I),
        re.compile(r"shear\s*stress", re.I),
        re.compile(r"viscos", re.I),
        re.compile(r"interval\s*time", re.I),
        re.compile(r"torque", re.I),
        re.compile(r"status", re.I),
    ]

    def _is_header_like(cells: List[str]) -> bool:
        cells = [str(c) for c in cells if str(c).strip()]
        if not cells:
            return False
        # Hard marker
        if any(marker_pat.search(c) for c in cells):
            return True
        # Count tokens
        count = 0
        for pat in token_res:
            if any(pat.search(c) for c in cells):
                count += 1
        return count >= 2

    # Find all marker rows within data_rows
    markers: List[int] = []
    for i, row in enumerate(data_rows):
        if _is_header_like(row):
            markers.append(i)

    if not markers:
        return []

    # Build (start, end) segments between markers
    segs: List[Tuple[int, int]] = []
    def _after_header(idx: int) -> int:
        # Skip 1–3 rows that look like header/units (contain brackets or tokens)
        j = idx + 1
        steps = 0
        while j < len(data_rows) and steps < 3:
            line = " ".join([str(c) for c in data_rows[j] if str(c).strip()]).lower()
            if not line:
                j += 1; steps += 1; continue
            if any(tok in line for tok in ("[", "]", "1/s", "pa", "point no", "time of day", "shear rate", "shear stress", "viscos")):
                j += 1; steps += 1; continue
            break
        return j

    starts: List[int] = []
    for m in markers:
        starts.append(_after_header(m))
    # Close with EOF as an implicit marker
    bounds = markers + [len(data_rows)]
    for si, m in zip(starts, bounds[1:]):
        end = max(si, m - 1)
        if end > si:
            # trim trailing blank lines within the segment
            while end >= si:
                line = "".join([str(c).strip() for c in data_rows[end]])
                if line:
                    break
                end -= 1
            if end > si:
                segs.append((si, end))
    return segs

# -----------------
# Main processing
# -----------------

def process_file(path: str, mode: str = 'triggered', steady_sec: float = 10.0, rate_thresh: float = 5.0,
                 show_preview: bool = True, preview_payload: Optional[Dict] = None,
                 crop_range: Optional[Tuple[float, float]] = None) -> Dict:
    rows, enc, delim = _read_csv_rows(path)
    # Trim leading empty columns, detect header row and normalize header lines
    rows = _trim_leading_empty_columns(rows, sample_rows=50)
    header_start = _detect_header_start(rows, seek_rows=400)
    rows = rows[header_start:]
    header, data_rows = _normalize_headers(rows, max_header_rows=2)
    try:
        print("[Rheology] header (normalized):", header)
    except Exception:
        pass
    cols = _find_columns(header)
    cols = _fallback_find_cols(header, cols)
    time_idx = cols.get('time_of_day') if cols.get('time_of_day') is not None else cols.get('time')
    rate_idx = cols.get('shear_rate')
    visc_idx = cols.get('viscosity')
    stress_idx = cols.get('shear_stress')
    print(f"[Rheology] detected columns → time_idx={time_idx}, rate_idx={rate_idx}, visc_idx={visc_idx}, stress_idx={stress_idx}")

    # Build initial time axis and series and show a quick preview without intervals
    t_sec_initial = _build_time_series(data_rows, time_idx) if time_idx is not None else None

    def _series(col_idx: Optional[int]) -> Optional[List[float]]:
        if col_idx is None:
            return None
        vals = []
        for r in data_rows:
            v = _to_float(r[col_idx] if col_idx < len(r) else '')
            vals.append(v if math.isfinite(v) else math.nan)
        return vals

    rate_series_initial = _series(rate_idx)
    visc_series_initial = _series(visc_idx)
    stress_series_initial = _series(stress_idx)

    # If viscosity column missing/empty but stress & rate exist, compute an apparent viscosity series for plotting
    if (visc_series_initial is None or _finite_count(visc_series_initial) == 0) and rate_series_initial is not None and stress_series_initial is not None:
        app = []
        for r, t in zip(rate_series_initial, stress_series_initial):
            if r is not None and t is not None and math.isfinite(r) and r != 0 and math.isfinite(t):
                app.append(t / r)
            else:
                app.append(math.nan)
        visc_series_initial = app

    try:
        print(f"[Rheology] initial finite counts → time={_finite_count(t_sec_initial)}, rate={_finite_count(rate_series_initial)}, visc={_finite_count(visc_series_initial)}")
    except Exception:
        pass

    # (preview is shown inline in the options dialog; no standalone window here)

    if time_idx is None and mode == 'triggered':
        raise RuntimeError("Triggered mode requires a 'Time of Day' or 'Time' column")
    if rate_idx is None:
        print("[WARN] No shear rate column detected; nontriggered mode may fail")

    # segment selection
    if mode == 'triggered':
        runs = _segments_triggered(data_rows, time_idx)
    elif mode == 'nontriggered':
        # Use the SAME run detection as triggered (continuous time runs split by text/gaps)
        # because the uncertainty is only the steady window length, not the step boundaries.
        if time_idx is not None:
            runs = _segments_triggered(data_rows, time_idx, min_run_sec=0.0)
        else:
            # If no time column, fall back to embedded header markers
            runs = _segments_by_markers(data_rows)
        # (No shear-rate based segmentation here.)
    else:  # other
        # fallback: single segment across all data
        runs = [(0, len(data_rows)-1)] if data_rows else []

    print(f"[Rheology] segments detected: {len(runs)} (mode={mode})")

    # Build time axis
    t_sec = _build_time_series(data_rows, time_idx) if time_idx is not None else None

    # Assemble interval arrays (cyan/red markers)
    T_beg: List[float] = []
    T_end: List[float] = []
    S_beg: List[float] = []
    S_end: List[float] = []

    if mode == 'triggered':
        # assume runs alternate Transient/Steady
        for k in range(0, len(runs) - 1, 2):
            a0, b0 = runs[k]
            a1, b1 = runs[k+1]
            tb = (t_sec[a0] if t_sec else float(a0))
            te = (t_sec[b0] if t_sec else float(b0))
            sb = (t_sec[a1] if t_sec else float(a1))
            se = (t_sec[b1] if t_sec else float(b1))
            T_beg.append(tb); T_end.append(te); S_beg.append(sb); S_end.append(se)
    elif mode == 'nontriggered':
        # each run is a step; steady = last steady_sec of the run
        for (a, b) in runs:
            if a >= b:
                continue
            if t_sec is not None and math.isfinite(t_sec[b]):
                raw_end = t_sec[b]
                se = max(t_sec[a], raw_end - STEADY_DELAY_SEC)
                sb = max(t_sec[a], se - steady_sec)
                tb = t_sec[a]
                te = sb
            else:
                raw_end = float(b)
                se = max(float(a), raw_end - STEADY_DELAY_SEC)
                sb = max(float(a), se - steady_sec)
                tb = float(a)
                te = sb
            T_beg.append(tb); T_end.append(te); S_beg.append(sb); S_end.append(se)
    else:
        # single interval across all data, steady = last steady_sec
        if runs:
            a, b = runs[0]
            if t_sec is not None and math.isfinite(t_sec[b]):
                raw_end = t_sec[b]
                se = max(t_sec[a], raw_end - STEADY_DELAY_SEC)
                sb = max(t_sec[a], se - steady_sec)
                tb = t_sec[a]
                te = sb
            else:
                raw_end = float(b)
                se = max(float(a), raw_end - STEADY_DELAY_SEC)
                sb = max(float(a), se - steady_sec)
                tb = float(a)
                te = sb
            T_beg.append(tb); T_end.append(te); S_beg.append(sb); S_end.append(se)

    # Compute steady averages over [S_beg, S_end]
    interval_rows: List[Dict] = []
    steady_rates: List[Optional[float]] = []
    steady_taus:  List[Optional[float]] = []
    steady_viscs: List[Optional[float]] = []

    for (tb, te, sb, se) in zip(T_beg, T_end, S_beg, S_end):
        # Resolve indices for averaging window
        if t_sec is not None:
            idxs = [i for i in range(len(data_rows)) if math.isfinite(t_sec[i]) and (t_sec[i] >= sb) and (t_sec[i] <= se)]
            if not idxs:
                end_idx = max([i for i in range(len(data_rows)) if math.isfinite(t_sec[i]) and t_sec[i] <= se] or [0])
                idxs = list(range(max(0, end_idx - 9), end_idx + 1))
        else:
            end_idx = int(se)
            idxs = list(range(max(0, end_idx - 9), end_idx + 1))

        # Compute steady window size
        steady_npts = len(idxs)
        steady_dur_sec: Optional[float] = None
        try:
            if t_sec is not None and math.isfinite(se) and math.isfinite(sb):
                steady_dur_sec = max(0.0, float(se) - float(sb))
        except Exception:
            steady_dur_sec = None

        def avg_col(col_idx: Optional[int]) -> Optional[float]:
            if col_idx is None:
                return None
            vals = []
            for i in idxs:
                v = _to_float(data_rows[i][col_idx] if col_idx < len(data_rows[i]) else '')
                if math.isfinite(v):
                    vals.append(v)
            return (sum(vals) / len(vals)) if vals else None

        r_avg = avg_col(rate_idx)
        v_avg = avg_col(visc_idx)
        t_avg = avg_col(stress_idx)
        if (v_avg is None) and (t_avg is not None) and (r_avg is not None) and (r_avg != 0):
            v_avg = t_avg / r_avg

        steady_rates.append(r_avg)
        steady_taus.append(t_avg)
        steady_viscs.append(v_avg)

        interval_rows.append({
            'T_beg': tb, 'T_end': te, 'S_beg': sb, 'S_end': se,
            'avg_shear_rate': r_avg, 'avg_shear_stress': t_avg, 'avg_viscosity': v_avg,
            'steady_window_sec': steady_dur_sec,
            'steady_npoints': steady_npts,
        })

    # Optional crop on steady data (by shear rate): keep rates but blank out stress/visc outside range
    if crop_range is not None:
        start_r, end_r = crop_range
        dropped = 0
        for idx, r in enumerate(list(steady_rates)):
            try:
                keep = (r is not None) and math.isfinite(r) and (start_r <= float(r) <= end_r)
            except Exception:
                keep = False
            if not keep:
                dropped += 1
                steady_taus[idx] = math.nan
                steady_viscs[idx] = math.nan
                try:
                    interval_rows[idx]['avg_shear_stress'] = None
                    interval_rows[idx]['avg_viscosity'] = None
                except Exception:
                    pass
        print(f"[Rheology] applied steady crop by shear rate: {start_r}–{end_r}; blanked {dropped} intervals")

    # JSON output next to CSV
    out = {
        'file': path,
        'mode': mode,
        'steady_sec': steady_sec,
        'rate_thresh': rate_thresh,
        'columns': {'time_idx': time_idx, 'shear_rate_idx': rate_idx, 'viscosity_idx': visc_idx, 'shear_stress_idx': stress_idx},
        'n_intervals': len(interval_rows),
        'intervals': interval_rows[:50],
    }
    # Write steady CSV only; skip JSON metadata
    try:
        ref = get_reference_folder_from_path(path)
        processed = os.path.join(ref, "_Processed")
        os.makedirs(processed, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        csv_path = os.path.join(processed, f"Rheo_steady_{base}.csv")
        _write_steady_csv(path, steady_rates, steady_taus, steady_viscs)
    except Exception as e:
        print("[Rheology] write outputs failed:", e)

    # Console summary
    print(f"[Rheology] {os.path.basename(path)} → intervals={len(interval_rows)} mode={mode} steady={steady_sec}s")
    for i, seg in enumerate(interval_rows[:10], 1):
        print(f"  {i:02d}: rate={seg['avg_shear_rate']}, visc={seg['avg_viscosity']}, tau={seg['avg_shear_stress']}, steady_sec={seg.get('steady_window_sec')}, npts={seg.get('steady_npoints')}")

    # Prepare preview series for plotting / refresh UI
    def _series(col_idx: Optional[int]) -> Optional[List[float]]:
        if col_idx is None:
            return None
        vals = []
        for r in data_rows:
            v = _to_float(r[col_idx] if col_idx < len(r) else '')
            vals.append(v if math.isfinite(v) else math.nan)
        return vals

    rate_series = _series(rate_idx)
    visc_series = _series(visc_idx)
    stress_series = _series(stress_idx)

    # If viscosity column missing/empty but stress & rate exist, compute an apparent viscosity series for plotting
    if (visc_series is None or _finite_count(visc_series) == 0) and rate_series is not None and stress_series is not None:
        app = []
        for r, t in zip(rate_series, stress_series):
            if r is not None and t is not None and math.isfinite(r) and r != 0 and math.isfinite(t):
                app.append(t / r)
            else:
                app.append(math.nan)
        try:
            print("[Rheology] viscosity series missing → plotting apparent viscosity tau/rate instead")
        except Exception:
            pass
        visc_series = app

    # Debug finite counts
    try:
        print(f"[Rheology] finite counts → time={_finite_count(t_sec)}, rate={_finite_count(rate_series)}, visc={_finite_count(visc_series)}")
    except Exception:
        pass

    preview_data = {
        't_sec': t_sec,
        'rate_series': rate_series,
        'visc_series': visc_series,
        'stress_series': stress_series,
        'T_beg': T_beg,
        'T_end': T_end,
        'S_beg': S_beg,
        'S_end': S_end,
        'steady_rates': steady_rates,
        'steady_viscs': steady_viscs,
        'crop_range': crop_range,
    }
    if preview_payload is not None:
        preview_payload.clear()
        preview_payload.update(preview_data)

    if show_preview:
        try:
            _show_plot_window(
                path,
                t_sec,
                rate_series,
                visc_series,
                stress_series,
                T_beg,
                T_end,
                S_beg,
                S_end,
                steady_rates=steady_rates,
                steady_viscs=steady_viscs,
                steady_sec=steady_sec,
            )
        except Exception as e:
            try:
                print("[Rheology] preview failed:", e)
            except Exception:
                pass

    return out


def _interactive_process(csv_path: str, mode: str, steady_initial: float, rate_thresh: float) -> None:
    """Process a CSV with an interactive preview that supports steady-window refreshes and cropping."""
    steady_value = steady_initial
    crop_range: Optional[Tuple[float, float]] = None
    while True:
        payload: Dict = {}
        process_file(
            csv_path,
            mode=mode,
            steady_sec=steady_value,
            rate_thresh=rate_thresh,
            show_preview=False,
            preview_payload=payload,
            crop_range=crop_range,
        )

        refresh_request: Dict[str, float] = {}

        def _handle_refresh(new_val: float):
            refresh_request['steady'] = new_val

        def _handle_crop(start: float, end: float):
            refresh_request['crop'] = (start, end)

        _show_plot_window(
            csv_path,
            payload.get('t_sec'),
            payload.get('rate_series'),
            payload.get('visc_series'),
            payload.get('stress_series'),
            payload.get('T_beg', []),
            payload.get('T_end', []),
            payload.get('S_beg', []),
            payload.get('S_end', []),
            steady_rates=payload.get('steady_rates'),
            steady_viscs=payload.get('steady_viscs'),
            steady_sec=steady_value,
            allow_refresh=True,
            refresh_callback=_handle_refresh,
            allow_crop=True,
            crop_callback=_handle_crop,
            crop_limits=crop_range,
        )

        if ('steady' not in refresh_request) and ('crop' not in refresh_request):
            break
        steady_value = refresh_request.get('steady', steady_value)
        crop_range = refresh_request.get('crop', crop_range)


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Read viscosity (Anton Paar) with robust header & mode-aware segmentation")
    ap.add_argument("path", help="Path to Rheology CSV or folder containing Rheology/")
    ap.add_argument("--mode", choices=["triggered", "nontriggered", "other"], default="triggered")
    ap.add_argument("--steady-sec", type=float, default=10.0)
    ap.add_argument("--rate-threshold", type=float, default=5.0, help="DEPRECATED: ignored unless no internal interval markers are found (kept for compatibility)")
    ap.add_argument("--no-preview", action="store_false", dest="show_preview", help="Do not show preview plot window")
    ap.set_defaults(show_preview=True)
    args = ap.parse_args()

    p = args.path
    if os.path.isdir(p):
        rh = get_rheology_folder_from_path(p) or p
        csvs = [os.path.join(rh, f) for f in os.listdir(rh) if f.lower().endswith('.csv') and not f.startswith('.')]
        if not csvs:
            raise SystemExit("No CSV files found in Rheology")
        csvs.sort(key=lambda x: os.path.getmtime(x))
        p = csvs[-1]
        print("[Rheology] using latest CSV:", p)
    # quick peek at encoding/delimiter
    try:
        _rows, _enc, _delim = _read_csv_rows(p)
        delim_label = 'TAB' if _delim == '\t' else _delim
        print(f"[Rheology] CSV open: encoding={_enc}, delimiter={delim_label}")
    except Exception as _e:
        print("[Rheology] CSV open preview failed:", _e)

    if args.show_preview:
        _interactive_process(p, args.mode, args.steady_sec, args.rate_threshold)
    else:
        process_file(p, mode=args.mode, steady_sec=args.steady_sec, rate_thresh=args.rate_threshold, show_preview=False)


if __name__ == "__main__":
    main()
