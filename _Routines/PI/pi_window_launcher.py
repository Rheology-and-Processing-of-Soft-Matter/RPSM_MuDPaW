import os
import sys
from pathlib import Path


def _bootstrap_path() -> None:
    # Ensure project root is on sys.path so _UI_modules can be imported.
    here = Path(__file__).resolve()
    root = here.parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def main() -> int:
    if len(sys.argv) < 3:
        print("Usage: pi_window_launcher.py <working_folder> <sample_name>")
        return 1
    working_folder = sys.argv[1]
    sample_name = sys.argv[2]
    _bootstrap_path()
    try:
        import tkinter as tk
        from _UI_modules import PI_helper
    except Exception as exc:
        print(f"[PI] Failed to import Tk/PI_helper: {exc}")
        return 1

    root = tk.Tk()
    root.withdraw()
    try:
        PI_helper.process_sample_PI(root, working_folder, "PI", sample_name)
    except Exception as exc:
        print(f"[PI] Failed to open PI window: {exc}")
        return 1
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
