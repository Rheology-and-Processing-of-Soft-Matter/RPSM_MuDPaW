#!/usr/bin/env python3
import csv, sys, codecs, re, argparse, math, os

def detect_encoding_and_open(path):
    with open(path, "rb") as fb:
        sample = fb.read(8192)
    enc = "utf-8"
    if sample.startswith(codecs.BOM_UTF16_LE): enc = "utf-16-le"
    elif sample.startswith(codecs.BOM_UTF16_BE): enc = "utf-16-be"
    elif b"\x00" in sample[:200]: enc = "utf-16-le"
    # pick delimiter by frequency
    try: st = sample.decode(enc, errors="replace")
    except Exception:
        enc = "utf-8"; st = sample.decode(enc, errors="replace")
    delim = "\t" if st.count("\t") >= max(st.count(";"), st.count(",")) else (";" if st.count(";") >= st.count(",") else ",")
    return enc, delim

def read_rows(path):
    enc, delim = detect_encoding_and_open(path)
    with open(path, "r", encoding=enc, errors="replace", newline="") as fh:
        reader = csv.reader(fh, delimiter=delim)
        rows = list(reader)
    return rows, delim

def collapse_header(rows, header_skip=8, max_header_rows=2):
    """Skip metadata rows, then collapse header + units rows to one clean header."""
    body = rows[header_skip:] if len(rows) > header_skip else rows
    if not body: return [], []
    # find header row containing 'Time of Day'
    hdr_idx = None
    for i, r in enumerate(body[:5]):  # within first few lines after skip
        if any("time of day" in str(c).lower() for c in r):
            hdr_idx = i; break
    if hdr_idx is None:  # fallback: use first row as header
        hdr_rows = body[:max_header_rows]
        data_rows = body[max_header_rows:] if len(body) > max_header_rows else body[1:]
    else:
        hdr_rows = body[hdr_idx: hdr_idx+max_header_rows]
        data_rows = body[hdr_idx+max_header_rows:] if len(body) > hdr_idx+max_header_rows else body[hdr_idx+1:]
    width = max(len(r) for r in hdr_rows)
    header = []
    for c in range(width):
        parts = []
        for r in hdr_rows:
            parts.append((r[c] if c < len(r) else "").strip())
        # join header parts like: "Shear Rate" + "[1/s]" → "Shear Rate [1/s]"
        cell = " ".join([p for p in parts if p]).strip()
        header.append(cell)
    return header, data_rows

def parse_tod_to_seconds(s):
    s = (s or "").strip()
    if not s or s in ("[]", "[ ]"): return math.nan
    try:
        return float(s)  # already seconds
    except Exception:
        pass
    if ":" in s:
        try:
            h, m, sec = s.split(":")[0], s.split(":")[1], s.split(":")[2]
            return int(h)*3600 + int(m)*60 + float(sec.replace(",", "."))
        except Exception:
            return math.nan
    return math.nan

def find_time_col(header, data_rows):
    # prefer exact header
    cand = None
    for i, h in enumerate(header):
        hl = (h or "").lower()
        if "time of day" in hl:
            cand = i; break
    # verify by HH:MM:SS pattern or parseable seconds
    pat = re.compile(r"\b\d{2}:\d{2}:\d{2}(?:[.,]\d+)?\b")
    def score(j):
        colon = 0; finite = 0; med = -1.0
        vals = []
        for r in data_rows[:300]:
            s = (r[j] if j < len(r) else "").strip()
            if pat.search(s): colon += 1
            v = parse_tod_to_seconds(s)
            if math.isfinite(v):
                finite += 1; vals.append(v)
        if vals:
            med = sorted(vals)[len(vals)//2]
        return (colon, finite, med)
    if cand is not None:
        s = score(cand)
        # handle empty-leading-column case: try right then left
        sr = score(cand+1) if cand+1 < len(header) else (-1,-1,-1)
        sl = score(cand-1) if cand-1 >= 0 else (-1,-1,-1)
        best = max([(cand, s), (cand+1, sr), (cand-1, sl)], key=lambda x: x[1])
        if best[1][0] > 0 or best[1][1] >= 5:
            return max(0, min(best[0], len(header)-1))
    # fallback: scan all columns and pick best by (colon,finite,median)
    bestj, bests = None, (-1,-1,-1)
    ncols = max(len(r) for r in data_rows[:50]) if data_rows else 0
    for j in range(ncols):
        sc = score(j)
        if sc > bests and sc[1] >= 5:
            bests = sc; bestj = j
    return bestj

def write_clean_csv(in_path, out_path, keep_columns=None, header_skip=8):
    rows, delim = read_rows(in_path)
    header, data_rows = collapse_header(rows, header_skip=header_skip, max_header_rows=2)
    if not data_rows: raise RuntimeError("No data rows after header collapse.")
    tcol = find_time_col(header, data_rows)
    if tcol is None: raise RuntimeError("Could not locate 'Time of Day' column.")
    # compute Time_s_rel_t0 and optionally keep a subset
    time_s = [parse_tod_to_seconds(r[tcol] if tcol < len(r) else "") for r in data_rows]
    # t0 as first finite
    t0 = next((v for v in time_s if math.isfinite(v)), None)
    if t0 is None: raise RuntimeError("No finite times in the chosen column.")
    time_rel = [ (v - t0) if math.isfinite(v) else "" for v in time_s ]

    # Build output header & rows
    out_header = header[:] + ["Time_s_rel_t0"]
    out_rows = []
    for r, tr in zip(data_rows, time_rel):
        # pad row to header length
        row = r[:] + [""] * max(0, len(header) - len(r))
        row.append(f"{tr:.6f}" if tr != "" else "")
        out_rows.append(row)

    # Optional: reduce columns (e.g., only Time of Day, Shear Rate, Shear Stress, Time_s_rel_t0)
    if keep_columns:
        # map header names to indices
        name_to_idx = { (h or "").strip().lower(): i for i,h in enumerate(out_header) }
        idxs = []
        for nm in keep_columns:
            # allow loose matching
            low = nm.lower()
            j = next((i for i,h in enumerate(out_header) if low in (h or "").lower()), None)
            if j is not None: idxs.append(j)
        out_header = [out_header[j] for j in idxs]
        out_rows   = [[row[j] if j < len(row) else "" for j in idxs] for row in out_rows]

    # Write UTF-8, comma-separated
    with open(out_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(out_header)
        w.writerows(out_rows)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Convert Anton Paar CSV to clean UTF-8 CSV (comma).")
    ap.add_argument("input", help="Anton Paar CSV (UTF-16 LE + TAB).")
    ap.add_argument("-o", "--output", help="Output CSV path (UTF-8). Default: <input>_clean.csv")
    ap.add_argument("--skip", type=int, default=8, help="Header lines to skip before column header (default: 8).")
    ap.add_argument("--keep", nargs="*", default=[], help="Optional list of columns to keep (name fragments).")
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output or (os.path.splitext(in_path)[0] + "_clean.csv")
    write_clean_csv(in_path, out_path, keep_columns=args.keep or None, header_skip=args.skip)
    print(f"✅ Wrote: {out_path}")