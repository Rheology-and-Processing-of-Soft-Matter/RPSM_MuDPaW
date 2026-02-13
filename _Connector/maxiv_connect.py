"""SSH/rsync backend for accessing ForMAX azint data on MAX IV from a local GUI."""

from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
import subprocess
import shlex
import re

from _Connector.formax_preview import detect_scan_ids, get_azint_folder

LogCb = Optional[Callable[[str], None]]


def _log(cb: LogCb, msg: str) -> None:
    if cb:
        cb(msg if msg.endswith("\n") else f"{msg}\n")


def build_remote_azint_path(beamline: str, proposal: int, visit: int) -> Path:
    """
    Build the remote azint path for the given beamline/proposal/visit.

    For ForMAX, the verified path format is:
        /data/visitors/formax/{proposal}/{visit}/process/azint

    Implemented via get_azint_folder with a base under /data/visitors.
    """
    beamline_lower = beamline.lower()
    #if beamline_lower != "formax":
        #raise ValueError(f"Unsupported beamline: {beamline}. Only 'ForMAX' is supported.")
    remote_base = Path("/data/visitors") / beamline_lower
    return get_azint_folder(proposal, visit, base=remote_base)


def build_remote_raw_path(beamline: str, proposal: int, visit: int) -> Path:
    """
    Build the remote raw path for the given beamline/proposal/visit.
    Expected structure (ForMAX):
        /data/visitors/formax/{proposal}/{visit}/raw
    """
    beamline_lower = beamline.lower()
    if beamline_lower != "formax":
        raise ValueError(f"Unsupported beamline: {beamline}. Only 'ForMAX' is supported.")
    return Path("/data/visitors") / beamline_lower / str(proposal) / str(visit) / "raw"


def run_ssh(
    hostname: str,
    username: str,
    remote_cmd: str,
    log_cb: LogCb = None,
) -> str:
    """
    Run a command on the remote host via ssh using key-based authentication only.
    """
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",  # do NOT prompt for passwords
        f"{username}@{hostname}",
        remote_cmd,
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout_chunks: List[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        stdout_chunks.append(line)
        _log(log_cb, line)

    stderr = proc.stderr.read() if proc.stderr is not None else ""
    ret = proc.wait()
    if ret == 0:
        return "".join(stdout_chunks)

    if ("permission denied" in stderr.lower()) or ("authentication failed" in stderr.lower()):
        raise RuntimeError(
            "SSH authentication failed.\n"
            "The MuDPaW Connect module requires key-based SSH access to this host.\n"
            "Please configure an SSH key for this account (e.g. via ssh-keygen and ssh-copy-id)\n"
            "so that 'ssh "
            f"{username}@{hostname}"
            "' works without asking for a password."
        )

    raise RuntimeError(
        f"SSH command failed with code {ret}.\n"
        f"Command: {' '.join(cmd)}\n"
        f"Stderr:\n{stderr}"
    )


def list_remote_scans(
    hostname: str,
    username: str,
    beamline: str,
    proposal: int,
    visit: int,
    log_cb: LogCb = None,
    local_root: Optional[Path] = None,
) -> Tuple[Path, List[int]]:
    """
    Return (remote_azint_path, sorted_scan_ids).
    """
    remote_azint = build_remote_azint_path(beamline, proposal, visit)
    _log(log_cb, f"[maxiv_connect] Remote azint: {remote_azint}")

    remote_cmd = f"ls {shlex.quote(str(remote_azint))}"
    stdout = run_ssh(hostname, username, remote_cmd, log_cb=log_cb)

    pattern = re.compile(r"(\d{3,5})")
    scan_ids_remote: set[int] = set()
    for line in stdout.splitlines():
        match = pattern.search(line)
        if not match:
            continue
        try:
            scan_ids_remote.add(int(match.group(1)))
        except ValueError:
            continue

    if local_root:
        local_azint = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "azint"
        if local_azint.exists():
            try:
                local_ids = detect_scan_ids(local_azint)
                scan_ids_remote.update(local_ids)
                _log(log_cb, f"[maxiv_connect] Found {len(local_ids)} scans in local mirror.")
            except Exception as exc:  # noqa: BLE001
                _log(log_cb, f"[maxiv_connect] Failed to read local scans: {exc}")
        else:
            _log(log_cb, f"[maxiv_connect] Local mirror not found: {local_azint}")

    sorted_ids = sorted(scan_ids_remote)
    _log(log_cb, f"[maxiv_connect] Total scans detected: {len(sorted_ids)}")
    return remote_azint, sorted_ids


def rsync_scans(
    hostname: str,
    username: str,
    beamline: str,
    proposal: int,
    visit: int,
    scan_ids: Iterable[int],
    local_root: Path,
    log_cb: LogCb = None,
    remote_azint_override: Optional[Path] = None,
) -> None:
    """
    Download specified scans from the remote azint folder to a local mirror using rsync.
    """
    remote_azint = remote_azint_override or build_remote_azint_path(beamline, proposal, visit)
    local_azint = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "azint"
    local_azint.mkdir(parents=True, exist_ok=True)

    for scan_id in scan_ids:
        pattern = f"{str(remote_azint)}/scan-{scan_id:04d}_*.h5"
        cmd = [
            "rsync",
            "-avzP",
            f"{username}@{hostname}:{pattern}",
            str(local_azint) + "/",
        ]
        _log(log_cb, f"[maxiv_connect] rsync scan {scan_id}: {' '.join(cmd)}")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        stdout_lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            stdout_lines.append(line)
            _log(log_cb, line)
        stdout = "".join(stdout_lines)
        stderr = proc.stderr.read() if proc.stderr is not None else ""
        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(
                f"rsync failed for scan {scan_id} (rc={rc}): {stderr.strip() or stdout.strip()}"
            )


def rsync_raw_scans(
    hostname: str,
    username: str,
    beamline: str,
    proposal: int,
    visit: int,
    scan_ids: Iterable[int],
    local_root: Path,
    log_cb: LogCb = None,
) -> None:
    """
    Download raw Eiger files for specified scans to a local mirror.
    """
    remote_raw = build_remote_raw_path(beamline, proposal, visit)
    local_raw = Path(local_root) / beamline.lower() / str(proposal) / str(visit) / "raw"
    local_raw.mkdir(parents=True, exist_ok=True)

    for scan_id in scan_ids:
        last_err = ""
        patterns = [
            f"{str(remote_raw)}/scan-{scan_id:04d}_eiger_master.h5",
            f"{str(remote_raw)}/scan-{scan_id:04d}_master.h5",
            f"{str(remote_raw)}/scan-{scan_id:04d}_eiger_*.h5",
        ]
        for pattern in patterns:
            cmd = [
                "rsync",
                "-avzP",
                f"{username}@{hostname}:{pattern}",
                str(local_raw) + "/",
            ]
            _log(log_cb, f"[maxiv_connect] rsync raw scan {scan_id}: {' '.join(cmd)}")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            stdout_lines: List[str] = []
            assert proc.stdout is not None
            for line in proc.stdout:
                stdout_lines.append(line)
                _log(log_cb, line)
            stdout = "".join(stdout_lines)
            stderr = proc.stderr.read() if proc.stderr is not None else ""
            rc = proc.wait()
            if rc == 0:
                break
            last_err = stderr.strip() or stdout.strip()
        else:
            raise RuntimeError(
                f"rsync failed for raw scan {scan_id}: {last_err}"
            )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test MAX IV connect backend.")
    parser.add_argument("--hostname", required=True)
    parser.add_argument("--username", required=True)
    parser.add_argument("--beamline", default="ForMAX")
    parser.add_argument("--proposal", type=int, required=True)
    parser.add_argument("--visit", type=int, required=True)
    parser.add_argument("--local-root", type=Path, required=False)
    parser.add_argument("--download", nargs="*", type=int, help="Scan IDs to download")
    args = parser.parse_args()

    def log(line: str) -> None:
        print(line, end="")

    remote_azint, scan_ids = list_remote_scans(
        hostname=args.hostname,
        username=args.username,
        beamline=args.beamline,
        proposal=args.proposal,
        visit=args.visit,
        log_cb=log,
        local_root=args.local_root,
    )
    print(f"Remote azint: {remote_azint}")
    print(f"Scan IDs: {scan_ids}")

    if args.download:
        if args.local_root is None:
            raise SystemExit("--local-root is required when using --download")
        rsync_scans(
            hostname=args.hostname,
            username=args.username,
            beamline=args.beamline,
            proposal=args.proposal,
            visit=args.visit,
            scan_ids=args.download,
            local_root=args.local_root,
            log_cb=log,
        )
