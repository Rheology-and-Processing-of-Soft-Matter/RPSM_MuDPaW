#!/usr/bin/env python3
"""Batch-export flagged graphs for every .dgraph document under an overfolder."""
from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_DG_CLI = "/Applications/DataGraph.app/Contents/Library/dgraph"
FORMAT_SEQUENCE = ["jpg", "png", "tiff", "pdf", "svg"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Iterate over .dgraph files and export flagged graphs using the DataGraph CLI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("overfolder", help="Folder that contains the generated .dgraph files")
    parser.add_argument("--dg-cli", default=DEFAULT_DG_CLI, help="Path to the DataGraph CLI binary")
    parser.add_argument("--format", default="jpeg", help="Image format to export")
    parser.add_argument("--dpi", type=int, default=300, help="Dots per inch for the export")
    parser.add_argument("--quality", type=float, default=0.9, help="Image quality (for lossy formats)")
    parser.add_argument("--limit", type=int, help="Optional limit for number of documents to process")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    return parser.parse_args()


def discover_dgraph_files(root: Path) -> list[Path]:
    matches: list[Path] = []
    for entry in sorted(root.rglob("*.dgraph")):
        if entry.is_file() or entry.is_dir():
            matches.append(entry)
    return matches


def resolve_cli_path(raw_cli: str) -> Path:
    path = Path(raw_cli).expanduser()
    if path.is_dir():
        for candidate in ("dgraph", "DataGraph", "datagraph"):
            guess = path / candidate
            if guess.exists():
                return guess
    if path.exists():
        return path
    resolved = shutil.which(raw_cli)
    return Path(resolved).expanduser() if resolved else path


def build_command(cli: Path, doc: Path, args: argparse.Namespace, export_format: str) -> list[str]:
    dest = doc.parent
    return [
        str(cli),
        str(doc),
        "-flagged",
        str(dest),
        "-format",
        export_format,
        "-dpi",
        str(args.dpi),
        "-quality",
        str(args.quality),
    ]


def main() -> int:
    args = parse_args()
    overfolder = Path(args.overfolder).expanduser()
    if not overfolder.is_dir():
        print(f"[flagged] Overfolder not found: {overfolder}", file=sys.stderr)
        return 2

    cli = resolve_cli_path(args.dg_cli)
    if not cli.exists():
        print(f"[flagged] DataGraph CLI not found at: {args.dg_cli}", file=sys.stderr)
        return 3

    docs = discover_dgraph_files(overfolder)
    if not docs:
        print(f"[flagged] No .dgraph files found under {overfolder}")
        return 0

    total = len(docs)
    if args.limit is not None and args.limit < total:
        docs = docs[: args.limit]
        total = len(docs)

    print(f"[flagged] Using CLI: {cli}")
    print(f"[flagged] Found {total} .dgraph document(s) to process.")

    # Determine per-folder export formats (cycle sequence if necessary)
    format_overrides: dict[Path, str] = {}
    folders: dict[Path, list[Path]] = {}
    for doc in docs:
        folders.setdefault(doc.parent, []).append(doc)
    for folder, items in folders.items():
        if len(items) <= 1:
            continue
        for order, doc in enumerate(sorted(items)):
            fmt = FORMAT_SEQUENCE[order % len(FORMAT_SEQUENCE)]
            format_overrides[doc] = fmt

    for idx, doc in enumerate(docs, start=1):
        fmt = format_overrides.get(doc, args.format)
        cmd = build_command(cli, doc, args, fmt)
        human_cmd = shlex.join(cmd)
        note = f" [{fmt}]" if fmt else ""
        print(f"[flagged] ({idx}/{total}) Exporting flagged graph for: {doc}{note}")
        if args.dry_run:
            print(f"[flagged] Dry-run → {human_cmd}")
            continue
        try:
            subprocess.run(cmd, check=True)
            print(f"[flagged] ({idx}/{total}) Saved flagged view into {doc.parent}")
        except subprocess.CalledProcessError as exc:
            print(f"[flagged] ERROR for {doc}: {exc}", file=sys.stderr)
        except Exception as exc:
            print(f"[flagged] ERROR for {doc}: {exc}", file=sys.stderr)

    print("[flagged] Completed flagged exports.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
