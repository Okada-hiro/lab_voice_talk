import argparse
import os
import sys
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser(description="Upload local files to faster_main.py sync endpoint.")
    p.add_argument("--url", required=True, help="RunPod base URL (e.g. https://...proxy.runpod.net)")
    p.add_argument("--token", required=True, help="Token that matches RUNPOD_SYNC_TOKEN on server")
    p.add_argument(
        "--remote-root",
        default="",
        help="Remote subdirectory under RUNPOD_SYNC_ROOT (e.g. lab_voice_talk)",
    )
    p.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Local file paths to upload",
    )
    p.add_argument(
        "--base-dir",
        default=".",
        help="Base dir used to compute relative remote paths",
    )
    p.add_argument("--timeout", type=int, default=120, help="Request timeout seconds")
    return p.parse_args()


def normalize_remote_path(base_dir: Path, local_file: Path, remote_root: str) -> str:
    rel = local_file.resolve().relative_to(base_dir.resolve())
    rel_posix = rel.as_posix()
    root = remote_root.strip().strip("/")
    return f"{root}/{rel_posix}" if root else rel_posix


def upload_one(session: requests.Session, endpoint: str, token: str, local_file: Path, remote_path: str, timeout: int):
    with local_file.open("rb") as f:
        files = {"file": (local_file.name, f, "application/octet-stream")}
        data = {"relative_path": remote_path}
        headers = {"x-sync-token": token}
        r = session.post(endpoint, headers=headers, data=data, files=files, timeout=timeout)
    if r.status_code >= 300:
        raise RuntimeError(f"Upload failed ({r.status_code}) {local_file}: {r.text}")
    return r.json()


def main():
    args = parse_args()
    base_url = args.url.rstrip("/")
    endpoint = f"{base_url}/admin/upload-file"

    base_dir = Path(args.base_dir)
    file_paths = [Path(x) for x in args.files]
    for p in file_paths:
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")

    with requests.Session() as session:
        for p in file_paths:
            remote_path = normalize_remote_path(base_dir, p, args.remote_root)
            result = upload_one(session, endpoint, args.token, p, remote_path, args.timeout)
            print(f"OK: {p} -> {remote_path} ({result.get('bytes')} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
