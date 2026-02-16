from __future__ import annotations

import os
import subprocess
from typing import Iterable, Tuple


def _find_datagraph_exec(dir_path: str) -> Tuple[str | None, str | None]:
    candidates = ["dgraph", "DataGraph", "datagraph"]
    for name in candidates:
        p = os.path.join(dir_path, name)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return dir_path, name
    return None, None


def resolve_datagraph_exec(path: str) -> Tuple[str | None, str | None]:
    """
    Return (exec_dir, exec_name) for a DataGraph executable.
    Accepts either a directory or a full path to the executable.
    """
    if not path:
        path = ""
    if os.path.isfile(path) and os.access(path, os.X_OK):
        return os.path.dirname(path), os.path.basename(path)
    if os.path.isdir(path):
        exec_dir, exec_name = _find_datagraph_exec(path)
        if exec_name:
            return exec_dir, exec_name
        if path.endswith("/Contents/Library"):
            alt = path.replace("/Contents/Library", "/Contents/MacOS")
            exec_dir, exec_name = _find_datagraph_exec(alt)
            if exec_name:
                return exec_dir, exec_name
    for candidate in ("/Applications/DataGraph.app/Contents/MacOS", "/Applications/DataGraph.app/Contents/Library"):
        exec_dir, exec_name = _find_datagraph_exec(candidate)
        if exec_name:
            return exec_dir, exec_name
    return None, None


def run_datagraph_write(datagraph_path: str, files: Iterable[str], template_path: str, output_path: str,
                        sample_name: str, quit_after: bool = False):
    """
    Run DataGraph headlessly with a template and input files.
    """
    exec_dir, exec_name = resolve_datagraph_exec(datagraph_path)
    if not exec_name:
        raise FileNotFoundError("DataGraph executable not found.")
    files = [f for f in files if f]
    cmd = [f"./{exec_name}"] + files + ["-script", template_path, "-v", f"Sample_1={sample_name}", "-output", output_path]
    if quit_after:
        cmd.append("-quitAfterScript")
    prev_cwd = os.getcwd()
    try:
        os.chdir(exec_dir)
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "cmd": cmd,
        }
    finally:
        os.chdir(prev_cwd)
