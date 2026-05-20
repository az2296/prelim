"""Upload local training checkpoints to the Hugging Face Hub."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


CHECKPOINT_PATTERNS = ("*.pt", "*.pth", "*.ckpt", "*.safetensors")


def _load_hf_api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with "
            "`pip install huggingface_hub`."
        ) from exc

    return HfApi


def configured_repo(repo_id: str | None = None) -> str | None:
    return repo_id or os.environ.get("HF_CHECKPOINT_REPO")


def default_prefix(prefix: str | None, module_dir: Path | None = None) -> str:
    if prefix is not None:
        return prefix.strip("/")
    if module_dir is None:
        return ""
    return str(module_dir).strip("/")


def path_in_repo(
    local_path: str | Path,
    *,
    prefix: str | None = None,
    seed: int | None = None,
) -> str:
    path = Path(local_path)
    name = path.name

    if seed is not None:
        suffix = "".join(path.suffixes)
        stem = name[: -len(suffix)] if suffix else name
        name = f"{stem}_seed{seed:04d}{suffix}"

    clean_prefix = default_prefix(prefix)
    if clean_prefix:
        return f"{clean_prefix}/{name}"
    return name


def upload_checkpoint(
    local_path: str | Path,
    *,
    repo_id: str | None = None,
    repo_type: str | None = None,
    private: bool = False,
    prefix: str | None = None,
    seed: int | None = None,
    commit_message: str | None = None,
) -> str | None:
    repo_id = configured_repo(repo_id)
    if not repo_id:
        return None

    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"Checkpoint does not exist: {local_path}")

    repo_type = repo_type or os.environ.get("HF_CHECKPOINT_REPO_TYPE", "model")
    api = _load_hf_api()()

    api.create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
    )

    remote_path = path_in_repo(local_path, prefix=prefix, seed=seed)
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=remote_path,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message or f"Upload checkpoint {remote_path}",
    )
    return remote_path


def try_upload_checkpoint(*args, **kwargs) -> str | None:
    try:
        return upload_checkpoint(*args, **kwargs)
    except Exception as exc:
        print(f"Warning: checkpoint upload skipped: {exc}")
        return None


def find_checkpoints(root: Path) -> list[Path]:
    checkpoints: list[Path] = []
    for pattern in CHECKPOINT_PATTERNS:
        checkpoints.extend(root.rglob(pattern))
    return sorted(path for path in checkpoints if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload checkpoints to Hugging Face.")
    parser.add_argument("paths", nargs="*", help="Checkpoint files or directories to upload.")
    parser.add_argument("--repo-id", default=os.environ.get("HF_CHECKPOINT_REPO"))
    parser.add_argument(
        "--repo-type",
        default=os.environ.get("HF_CHECKPOINT_REPO_TYPE", "model"),
        choices=["model", "dataset", "space"],
    )
    parser.add_argument("--prefix", default=os.environ.get("HF_CHECKPOINT_PREFIX", ""))
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.repo_id:
        raise SystemExit(
            "Set HF_CHECKPOINT_REPO=owner/repo or pass --repo-id owner/repo."
        )

    roots = [Path(path) for path in args.paths] or [Path.cwd()]
    files: list[Path] = []
    for root in roots:
        if root.is_dir():
            files.extend(find_checkpoints(root))
        elif root.is_file():
            files.append(root)
        else:
            raise SystemExit(f"Not found: {root}")

    if not files:
        print("No checkpoint files found.")
        return

    for checkpoint in sorted(set(files)):
        try:
            relative_checkpoint = checkpoint.relative_to(Path.cwd())
        except ValueError:
            relative_checkpoint = checkpoint

        remote_path = str(Path(args.prefix) / relative_checkpoint) if args.prefix else str(relative_checkpoint)
        if args.dry_run:
            print(f"would upload {checkpoint} -> {args.repo_id}:{remote_path}")
            continue

        api = _load_hf_api()()
        api.create_repo(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            private=args.private,
            exist_ok=True,
        )
        api.upload_file(
            path_or_fileobj=str(checkpoint),
            path_in_repo=remote_path,
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            commit_message=f"Upload checkpoint {remote_path}",
        )
        print(f"uploaded {checkpoint} -> {args.repo_id}:{remote_path}")


if __name__ == "__main__":
    main()
