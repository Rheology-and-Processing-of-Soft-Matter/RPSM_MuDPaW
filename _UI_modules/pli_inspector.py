from __future__ import annotations

import os
import re
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

import importlib


class PLIInspector:
    def __init__(self, main) -> None:
        self.main = main
        self.videoList: QListWidget | None = None
        self.imageList: QListWidget | None = None
        self.tempList: QListWidget | None = None

    def build_inspector(self, page: QWidget) -> None:
        layout = page.layout()
        if layout is None:
            layout = QVBoxLayout(page)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Processing PLI folder", page))

        self.btnRefreshVideo = QPushButton("Refresh video list", page)
        layout.addWidget(self.btnRefreshVideo)
        self.videoList = QListWidget(page)
        self.videoList.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.videoList.setFixedHeight(140)
        layout.addWidget(self.videoList, 0)

        row = QHBoxLayout()
        self.btnMerge = QPushButton("Merge selected videos", page)
        self.btnExtract = QPushButton("Extract space-time", page)
        row.addWidget(self.btnMerge)
        row.addWidget(self.btnExtract)
        layout.addLayout(row)

        self.btnRefreshImages = QPushButton("Refresh image list", page)
        layout.addWidget(self.btnRefreshImages)
        layout.addWidget(QLabel("Detected image files (select up to two):", page))
        self.imageList = QListWidget(page)
        # Intervals/rescale works on a single raw ST image; force single-select to
        # prevent accidental double rendering (e.g., selecting a "copy" variant).
        self.imageList.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.imageList.setFixedHeight(140)
        layout.addWidget(self.imageList, 0)

        self.btnIntervals = QPushButton("Intervals and re-scaling", page)
        layout.addWidget(self.btnIntervals)

        self.btnRefreshTemp = QPushButton("Refresh unscaled stitched (Temp)", page)
        layout.addWidget(self.btnRefreshTemp)
        layout.addWidget(QLabel("Unscaled stitched panels (PLI/_Temp):", page))
        self.tempList = QListWidget(page)
        self.tempList.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tempList.setFixedHeight(140)
        layout.addWidget(self.tempList, 0)

        self.btnProcessData = QPushButton("Process data", page)
        layout.addWidget(self.btnProcessData)

        layout.addStretch(1)

        self.btnRefreshVideo.clicked.connect(self.refresh_lists)
        self.btnRefreshImages.clicked.connect(self.refresh_lists)
        self.btnRefreshTemp.clicked.connect(self.refresh_lists)
        self.btnMerge.clicked.connect(self._merge_selected_videos)
        self.btnExtract.clicked.connect(self._extract_space_time)
        self.btnIntervals.clicked.connect(self._open_intervals_rescale)
        self.btnProcessData.clicked.connect(self._process_selected_temp)

    def refresh_lists(self) -> None:
        pli_dir = self._get_pli_dir()
        if pli_dir is None:
            return
        self._log(f"[PLI] Scanning folder: {pli_dir}")
        if self.videoList is not None:
            self.videoList.clear()
            videos = self._list_videos(pli_dir)
            for f in videos:
                self._add_item(self.videoList, f)
            if videos:
                self._log(f"[PLI] Videos found: {len(videos)}")
            else:
                self._log(f"[PLI] No video files found in: {pli_dir}")
        if self.imageList is not None:
            self.imageList.clear()
            images = self._list_images(pli_dir)
            for f in images:
                self._add_item(self.imageList, f)
            if images:
                self._log(f"[PLI] Images found: {len(images)}")
            else:
                self._log(f"[PLI] No image files found in: {pli_dir}")
        if self.tempList is not None:
            self.tempList.clear()
            temps = self._list_temp_images(pli_dir)
            for f in temps:
                self._add_item(self.tempList, f)
            if temps:
                self._log(f"[PLI] Temp stitched found: {len(temps)}")
            else:
                self._log(f"[PLI] No temp stitched files found in: {pli_dir / '_Temp'}")

    def _log(self, msg: str) -> None:
        if hasattr(self.main, "logger"):
            self.main.logger.info(msg)

    def _get_pli_dir(self) -> Path | None:
        base = self._resolve_reference_root()
        if not base or not os.path.isdir(base):
            self._log("[PLI] Select a reference folder first.")
            return None
        pli_dir = Path(base) / "PLI"
        if not pli_dir.exists():
            self._log(f"[PLI] PLI folder not found in reference folder: {base}")
            return None
        if not pli_dir.is_dir():
            self._log(f"[PLI] PLI path is not a directory: {pli_dir}")
            return None
        return pli_dir

    def _resolve_reference_root(self) -> str | None:
        base = self.main.project_path
        if not base:
            return None
        if os.path.isdir(base) and os.path.basename(base).lower() == "pli":
            return os.path.dirname(base)
        if os.path.basename(base) == "Inputs (reference)":
            return os.path.dirname(base)
        return base

    def _add_item(self, widget: QListWidget, path: Path) -> None:
        item = QListWidgetItem(path.name)
        item.setData(Qt.ItemDataRole.UserRole, str(path))
        widget.addItem(item)

    def _list_videos(self, pli_dir: Path) -> list[Path]:
        results: list[Path] = []
        for ext in (".mp4", ".avi", ".mov"):
            results.extend(sorted(pli_dir.glob(f"*{ext}")))
            results.extend(sorted(pli_dir.glob(f"*{ext.upper()}")))
        # De-dup while preserving order
        seen = set()
        uniq: list[Path] = []
        for p in results:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        return uniq

    def _list_images(self, pli_dir: Path) -> list[Path]:
        helper = self._import_helper()
        if helper is None:
            return sorted(pli_dir.glob("*.png"))
        results = [pli_dir / f for f in helper.list_raw_st_images(str(pli_dir))]
        if results:
            return results
        return sorted(pli_dir.glob("*.png"))

    def _list_temp_images(self, pli_dir: Path) -> list[Path]:
        helper = self._import_helper()
        if helper is None:
            temp = pli_dir / "_Temp"
            if not temp.exists():
                return []
            return sorted(temp.glob("*.png"))
        paths = [Path(p) for p in helper.list_unscaled_temp_stitched_outputs(str(pli_dir))]
        if paths:
            return paths
        temp = pli_dir / "_Temp"
        if not temp.exists():
            return []
        return sorted(temp.glob("*.png"))

    def _selected_paths(self, widget: QListWidget | None) -> list[str]:
        if widget is None:
            return []
        items = widget.selectedItems()
        out: list[str] = []
        for item in items:
            p = item.data(Qt.ItemDataRole.UserRole) or item.text()
            if isinstance(p, str):
                out.append(p)
        return out

    def _merge_selected_videos(self) -> None:
        pli_dir = self._get_pli_dir()
        if pli_dir is None:
            return
        selected = self._selected_paths(self.videoList)
        if len(selected) < 2:
            self._log("[PLI] Select at least two videos to merge.")
            return
        try:
            import cv2  # Lazy import so missing OpenCV surfaces as a clear log message
        except Exception as e:
            self._log(f"[PLI] OpenCV not available: {e}")
            return
        # Natural sort so merged order matches v5.8.1 behaviour
        def _nat_key(s: str):
            return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", os.path.basename(s))]

        videos = [str(Path(p)) for p in selected]
        videos = sorted(videos, key=_nat_key)

        try:
            cap0 = cv2.VideoCapture(videos[0])
            if not cap0.isOpened():
                self._log(f"[PLI] Could not open {os.path.basename(videos[0])}")
                return
            fps = cap0.get(cv2.CAP_PROP_FPS) or 0.0
            if fps <= 0:
                fps = 30.0
            width = int(cap0.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap0.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap0.release()
            if width <= 0 or height <= 0:
                self._log("[PLI] Could not determine video dimensions.")
                return

            def _next_merge_path():
                base = pli_dir / f"merged_{len(videos)}videos"
                cand = base.with_suffix(".mp4")
                idx = 1
                while cand.exists():
                    cand = base.with_name(f"{base.name}_{idx}").with_suffix(".mp4")
                    idx += 1
                return cand

            out_path = _next_merge_path()
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                self._log(f"[PLI] Could not create output file: {out_path}")
                return

            try:
                for vp in videos:
                    name = os.path.basename(vp)
                    cap = cv2.VideoCapture(vp)
                    if not cap.isOpened():
                        raise RuntimeError(f"Could not open {name}")
                    vf = cap.get(cv2.CAP_PROP_FPS) or 0.0
                    if vf and abs(vf - fps) > 0.5:
                        self._log(f"[PLI] Warning: {name} fps={vf:.2f} differs from base fps={fps:.2f}; using base fps.")
                    while True:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            break
                        h, w = frame.shape[:2]
                        if w != width or h != height:
                            frame = cv2.resize(frame, (width, height))
                        writer.write(frame)
                    cap.release()
            except Exception as merge_err:
                writer.release()
                if out_path.exists():
                    try:
                        out_path.unlink()
                    except Exception:
                        pass
                self._log(f"[PLI] Merge failed: {merge_err}")
                return

            writer.release()
            self._log(f"[PLI] Merged {len(videos)} videos → {out_path.name}")
            self.refresh_lists()
        except Exception as e:
            self._log(f"[PLI] Merge exception: {e}")

    def _extract_space_time(self) -> None:
        pli_dir = self._get_pli_dir()
        if pli_dir is None:
            return
        selected = self._selected_paths(self.videoList)
        if not selected:
            self._log("[PLI] Select a video first.")
            return
        if len(selected) > 1:
            self._log("[PLI] Multiple videos selected; using the first. Use 'Merge selected videos' for batch extraction.")
        if hasattr(self.main, "open_pli_extractor"):
            self.main.open_pli_extractor([selected[0]])
            return
        helper = self._import_helper()
        if helper is None:
            return
        try:
            helper.extract_space_time_pli(str(pli_dir), [selected[0]])
        except Exception as e:
            self._log(f"[PLI] Extract space-time failed: {e}")

    def _open_intervals_rescale(self) -> None:
        pli_dir = self._get_pli_dir()
        if pli_dir is None:
            return
        selected = self._selected_paths(self.imageList)
        if not selected:
            self._log("[PLI] Select 1-2 space-time images first.")
            return
        if len(selected) > 2:
            self._log("[PLI] Select at most two images. Using the first two.")
            selected = selected[:2]
        if hasattr(self.main, "open_pli_rescale"):
            self.main.open_pli_rescale(selected)
            return
        helper = self._import_helper()
        if helper is None:
            return
        try:
            helper.rescale_space_time_pli(str(pli_dir), selected)
        except Exception as e:
            self._log(f"[PLI] Intervals/rescale failed: {e}")

    def _process_selected_temp(self) -> None:
        pli_dir = self._get_pli_dir()
        if pli_dir is None:
            return
        selected = self._selected_paths(self.tempList)
        if not selected:
            self._log("[PLI] Select an unscaled stitched panel first.")
            return
        self._log(f"[PLI] Processing stitched panel: {selected[0]}")
        if hasattr(self.main, "open_pli_analyzer"):
            try:
                self.main.open_pli_analyzer(selected[0])
                return
            except Exception as e:
                self._log(f"[PLI] Qt analyzer failed: {e}")
                return
        self._log("[PLI] Qt analyzer hook missing; no action taken.")

    def _import_helper(self):
        try:
            return importlib.import_module("_UI_modules.PLI_helper")
        except Exception as e:
            self._log(f"[PLI] PLI helper not available: {e}")
            return None
