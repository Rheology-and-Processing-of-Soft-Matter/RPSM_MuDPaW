"""
Batch processor for CoSAXS azint data.

This reuses the ForMAX processing helpers but defaults the beamline to CoSAXS.
Reads an embed/list CSV (Name, Detail, Scan interval, background, bg_scale, q_range*),
downloads missing scans, applies optional background subtraction, averages frames,
and writes outputs to <local_root>/cosaxs/<proposal>_<visit>/<sample>/SAXS/scan_<first-last>/.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from _Connector.ForMAX import process_entry  # reuse the same processing pipeline


def main():
    ap = argparse.ArgumentParser(description="CoSAXS batch processor using embed list CSV.")
    ap.add_argument("--list", dest="list_path", default="embed_list.csv", help="Embed/list CSV file.")
    ap.add_argument("--hostname", required=True)
    ap.add_argument("--username", required=True)
    ap.add_argument("--beamline", default="CoSAXS")
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
