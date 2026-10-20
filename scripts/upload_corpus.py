"""Upload a documentation folder to a workspace in the local API."""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import os
from pathlib import Path

import httpx


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SUPPORTED_EXTENSIONS = {".md", ".txt", ".pdf"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload documentation files to a workspace."
    )
    parser.add_argument(
        "--folder",
        default="data/fastapi-docs/docs/en/docs",
        help="Folder containing the fetched documentation.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL of the running API.",
    )
    parser.add_argument(
        "--workspace-id",
        default=os.getenv("WORKSPACE_ID"),
        help="Workspace UUID, or set the WORKSPACE_ID environment variable.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upload files even if their relative names are already listed.",
    )
    return parser.parse_args()


async def upload_folder(args: argparse.Namespace) -> None:
    token = os.getenv("API_TOKEN", "").strip()
    if not token:
        raise SystemExit("Set API_TOKEN to a valid bearer token first.")
    if not args.workspace_id:
        raise SystemExit("Provide --workspace-id or set WORKSPACE_ID.")

    root = Path(args.folder).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Documentation folder does not exist: {root}")

    endpoint = (
        f"{args.base_url.rstrip('/')}/v1/workspaces/"
        f"{args.workspace_id}/documents"
    )
    list_endpoint = (
        f"{args.base_url.rstrip('/')}/v1/workspaces/"
        f"{args.workspace_id}/documents"
    )
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient(timeout=120) as client:
        existing_names: set[str] = set()
        if not args.force:
            response = await client.get(list_endpoint, headers=headers)
            response.raise_for_status()
            existing = response.json()
            if isinstance(existing, list):
                for document in existing:
                    if isinstance(document, dict):
                        name = document.get("original_filename") or document.get(
                            "filename"
                        )
                        if isinstance(name, str):
                            existing_names.add(name.replace("\\", "/"))

        files = sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            raise SystemExit(f"No supported documents found under {root}")

        uploaded = 0
        skipped = 0
        failed = 0

        for path in files:
            relative_name = path.relative_to(root).as_posix()
            if relative_name in existing_names:
                skipped += 1
                continue

            size = path.stat().st_size
            if size > MAX_UPLOAD_BYTES:
                print(f"SKIP {relative_name}: larger than 10 MiB")
                skipped += 1
                continue

            content_type = (
                mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            )
            with path.open("rb") as file:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    files={
                        "file": (
                            relative_name,
                            file,
                            content_type,
                        )
                    },
                )

            if response.is_success:
                uploaded += 1
                print(f"UPLOADED {relative_name}")
            else:
                failed += 1
                print(
                    f"FAILED {relative_name}: "
                    f"HTTP {response.status_code} {response.text[:500]}"
                )

        print(
            f"Finished: uploaded={uploaded}, skipped={skipped}, failed={failed}"
        )


if __name__ == "__main__":
    asyncio.run(upload_folder(parse_args()))