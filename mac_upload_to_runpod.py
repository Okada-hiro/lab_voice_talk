#!/usr/bin/env python3
"""
macOS -> RunPod file uploader (port 8000).

This script uploads local files to sample_main.py endpoint:
  POST /admin/upload-file
with:
  - multipart file field: "file"
  - form field: "relative_path"
  - header: "x-sync-token"
"""

import argparse
import os
import sys
from pathlib import Path

import requests


def parse_args():
    p = argparse.ArgumentParser(
        description="Upload files from macOS to RunPod (:8000) sync endpoint."
    )
    p.add_argument(
        "--host",
        required=True,
        help="RunPod host (example: abcdefg-8000.proxy.runpod.net or https://... )",
    )
    p.add_argument(
        "--token",
        default=os.getenv("RUNPOD_SYNC_TOKEN", ""),
        help="Sync token for x-sync-token header (or set RUNPOD_SYNC_TOKEN)",
    )
    p.add_argument(
        "--remote-root",
        default="",
        help="Remote subdirectory under RUNPOD_SYNC_ROOT (example: lab_voice_talk)",
    )
    p.add_argument(
        "--base-dir",
        default=".",
        help="Base directory to compute relative paths for remote placement",
    )
    p.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="Local file paths to upload",
    )
    p.add_argument("--timeout", type=int, default=120, help="Request timeout in seconds")
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress logs",
    )
    return p.parse_args()


def normalize_base_url(host: str) -> str:
    h = host.strip().rstrip("/")
    if h.startswith("http://") or h.startswith("https://"):
        return h
    # Default to https for RunPod proxy domains.
    return f"https://{h}"


def make_remote_path(base_dir: Path, file_path: Path, remote_root: str) -> str:
    rel = file_path.resolve().relative_to(base_dir.resolve())
    rel_posix = rel.as_posix()
    root = remote_root.strip().strip("/")
    return f"{root}/{rel_posix}" if root else rel_posix


def upload_one(
    session: requests.Session,
    endpoint: str,
    token: str,
    local_file: Path,
    remote_path: str,
    timeout: int,
):
    headers = {"x-sync-token": token}
    data = {"relative_path": remote_path}
    with local_file.open("rb") as f:
        files = {"file": (local_file.name, f, "application/octet-stream")}
        resp = session.post(endpoint, headers=headers, data=data, files=files, timeout=timeout)
    if resp.status_code >= 300:
        raise RuntimeError(f"{local_file} -> {remote_path} failed: {resp.status_code} {resp.text}")
    return resp.json()


def main():
    args = parse_args()
    if not args.token:
        raise ValueError("token is required. Use --token or RUNPOD_SYNC_TOKEN.")

    base_url = normalize_base_url(args.host)
    # Do NOT force :8000 for RunPod proxy hostnames like:
    #   xxxx-8000.proxy.runpod.net
    # They already encode the port in the hostname.
    host_part = base_url.split("://", 1)[1].split("/", 1)[0]
    if ":" not in host_part and ".proxy.runpod.net" not in host_part:
        base_url = f"{base_url}:8000"
    endpoint = f"{base_url}/admin/upload-file"

    if args.verbose:
        print(f"[INFO] endpoint={endpoint}")

    base_dir = Path(args.base_dir)
    local_files = [Path(x) for x in args.files]
    for p in local_files:
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Not a file: {p}")
        # Ensure file is under base_dir for relative mapping.
        _ = p.resolve().relative_to(base_dir.resolve())

    with requests.Session() as session:
        for p in local_files:
            remote_path = make_remote_path(base_dir, p, args.remote_root)
            if args.verbose:
                print(f"[INFO] uploading {p} -> {remote_path}")
            result = upload_one(session, endpoint, args.token, p, remote_path, args.timeout)
            size = result.get("bytes", "?")
            print(f"OK {p} -> {remote_path} ({size} bytes)")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)
