"""Download all checkpoint files from the public Hugging Face repo.

Mirrors the remote layout into the local repo root, so e.g.
    m3m4/diffusion/notebook_ckpts/m3_mlp_adamw.pt
on the Hub lands at the matching path under this repo.

Stdlib only — no `huggingface_hub` dependency, since the source repo is
public and the Hub serves files over plain HTTPS.

Usage:
    python scripts/download_checkpoints.py                       # default repo
    python scripts/download_checkpoints.py --repo-id owner/name  # different repo
    python scripts/download_checkpoints.py --dry-run             # just list
    python scripts/download_checkpoints.py --force               # re-download
    python scripts/download_checkpoints.py --filter m3m4         # only m3m4/**
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path


DEFAULT_REPO = "az2296/prelim-checkpoints"
CHECKPOINT_SUFFIXES = (".pt", ".pth", ".ckpt", ".safetensors")


def list_repo_files(repo_id: str, revision: str = "main") -> list[dict]:
    url = (
        f"https://huggingface.co/api/models/{repo_id}"
        f"/tree/{revision}?recursive=true"
    )
    with urllib.request.urlopen(url) as resp:
        return json.load(resp)


def resolve_url(repo_id: str, path: str, revision: str = "main") -> str:
    return f"https://huggingface.co/{repo_id}/resolve/{revision}/{path}"


def download(url: str, dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as out:
        total = 0
        while True:
            chunk = resp.read(1 << 20)  # 1 MiB
            if not chunk:
                break
            out.write(chunk)
            total += len(chunk)
    tmp.replace(dest)
    return total


def human(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--revision", default="main")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Local directory to mirror into (defaults to repo root).",
    )
    parser.add_argument(
        "--filter",
        default="",
        help="Only download paths starting with this prefix (e.g. 'm3m4').",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists locally.",
    )
    args = parser.parse_args()

    entries = list_repo_files(args.repo_id, args.revision)
    files = [
        e for e in entries
        if e.get("type") == "file"
        and e["path"].endswith(CHECKPOINT_SUFFIXES)
        and e["path"].startswith(args.filter)
    ]
    if not files:
        print(f"No checkpoints matched (repo={args.repo_id}, filter={args.filter!r}).")
        return

    print(f"Found {len(files)} checkpoint(s) in {args.repo_id}@{args.revision}.")
    total_bytes = 0
    skipped = 0
    for entry in files:
        path = entry["path"]
        dest = args.root / path
        url = resolve_url(args.repo_id, path, args.revision)

        if dest.exists() and not args.force:
            print(f"[skip] exists: {path}")
            skipped += 1
            continue

        if args.dry_run:
            print(f"[dry]  {url} -> {dest}")
            continue

        print(f"[get]  {path} ...", end=" ", flush=True)
        try:
            n = download(url, dest)
        except Exception as exc:  # noqa: BLE001
            print(f"FAILED ({exc})")
            continue
        total_bytes += n
        print(human(n))

    if not args.dry_run:
        print(
            f"Done. Downloaded {human(total_bytes)} "
            f"({len(files) - skipped} files; {skipped} already present)."
        )


if __name__ == "__main__":
    sys.exit(main())
