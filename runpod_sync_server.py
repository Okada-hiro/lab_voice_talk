#!/usr/bin/env python3
"""
RunPod sync server (upload/download only, port 8000 by default).

Endpoints:
  - GET  /health
  - POST /admin/upload-file
  - GET  /admin/list-files?relative_dir=...
  - GET  /download/<path>
"""

import logging
import os
import sys
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


SYNC_ROOT_DIR = Path(os.path.abspath(os.getenv("RUNPOD_SYNC_ROOT", ".")))
SYNC_TOKEN = os.getenv("RUNPOD_SYNC_TOKEN", "").strip()
PORT = int(os.getenv("PORT", "8000"))

SYNC_ROOT_DIR.mkdir(parents=True, exist_ok=True)
logger.info("[SYNC] root=%s", SYNC_ROOT_DIR)

app = FastAPI(title="RunPod Sync Server", version="1.0.0")
app.mount("/download", StaticFiles(directory=str(SYNC_ROOT_DIR)), name="download")


def _verify_sync_token(token: str | None):
    if not SYNC_TOKEN:
        raise HTTPException(status_code=503, detail="RUNPOD_SYNC_TOKEN is not configured on server.")
    if token != SYNC_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid sync token.")


def _resolve_sync_path(relative_path: str) -> Path:
    rel = (relative_path or "").strip()
    if not rel:
        raise HTTPException(status_code=400, detail="relative_path is required.")
    rel = rel.lstrip("/").replace("\\", "/")
    target = (SYNC_ROOT_DIR / rel).resolve()

    # Prevent path traversal.
    try:
        target.relative_to(SYNC_ROOT_DIR)
    except ValueError:
        raise HTTPException(status_code=400, detail="Path traversal is not allowed.")
    return target


@app.get("/health")
async def health():
    return {
        "ok": True,
        "root": str(SYNC_ROOT_DIR),
        "token_configured": bool(SYNC_TOKEN),
        "port": PORT,
    }


@app.post("/admin/upload-file")
async def upload_file(
    relative_path: str = Form(...),
    file: UploadFile = File(...),
    x_sync_token: str | None = Header(default=None),
):
    _verify_sync_token(x_sync_token)
    target = _resolve_sync_path(relative_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    data = await file.read()
    target.write_bytes(data)
    logger.info("[SYNC] uploaded %s (%d bytes)", target, len(data))
    return {
        "ok": True,
        "path": str(target),
        "bytes": len(data),
        "download_url": f"/download/{target.relative_to(SYNC_ROOT_DIR).as_posix()}",
    }


@app.get("/admin/list-files")
async def list_files(
    relative_dir: str = Query(default="", description="Directory under RUNPOD_SYNC_ROOT"),
    x_sync_token: str | None = Header(default=None),
):
    _verify_sync_token(x_sync_token)
    base = _resolve_sync_path(relative_dir) if relative_dir else SYNC_ROOT_DIR
    if not base.exists():
        return JSONResponse({"ok": True, "dir": str(base), "files": []})
    if not base.is_dir():
        raise HTTPException(status_code=400, detail="relative_dir must be a directory.")

    rows = []
    for p in sorted(base.rglob("*")):
        if not p.is_file():
            continue
        st = p.stat()
        rel = p.relative_to(SYNC_ROOT_DIR).as_posix()
        rows.append(
            {
                "relative_path": rel,
                "size_bytes": int(st.st_size),
                "modified_ts": float(st.st_mtime),
                "download_url": f"/download/{rel}",
            }
        )

    return JSONResponse({"ok": True, "dir": str(base), "files": rows})


if __name__ == "__main__":
    logger.info("[SYNC] starting server on 0.0.0.0:%d", PORT)
    uvicorn.run(app, host="0.0.0.0", port=PORT)
