#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError


def _default_token() -> str | None:
    return (
        os.environ.get("HF_TOKEN")
        or os.environ.get("HUGGINGFACE_HUB_TOKEN")
        or os.environ.get("HF_HUB_TOKEN")
    )


def _build_repo_id(owner: str, repo: str) -> str:
    if "/" in repo:
        return repo
    return f"{owner}/{repo}"


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Upload a local folder to Hugging Face Hub (repo-type: model by default)."
    )
    parser.add_argument(
        "--owner",
        default=os.environ.get("HF_OWNER", "newshawn"),
        help="Hugging Face username/org (default: env HF_OWNER or 'newshawn').",
    )
    parser.add_argument(
        "--repo",
        default=os.environ.get("HF_REPO", "Qwen2.5-7B-MATH-1EPOCH"),
        help="Repo name (or full repo_id like owner/name).",
    )
    parser.add_argument(
        "--path",
        default=os.environ.get(
            "HF_LOCAL_PATH",
            "/run/determined/NAS1/public/xuexiang/H800/best_7B/checkpoint-20",
        ),
        help="Local folder to upload.",
    )
    parser.add_argument(
        "--repo-type",
        default="model",
        choices=("model", "dataset", "space"),
        help="Hub repository type.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create repo as private (only used if creating the repo).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token (prefer env HF_TOKEN / HUGGINGFACE_HUB_TOKEN / HF_HUB_TOKEN).",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload files",
        help="Commit message for the upload.",
    )
    parser.add_argument(
        "--no-create-repo",
        action="store_true",
        help="Skip repo creation (assumes repo already exists).",
    )
    parser.add_argument(
        "--enable-hf-transfer",
        action="store_true",
        help="Enable hf_transfer acceleration (requires hf_transfer installed).",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Extra ignore pattern (glob). Can be repeated.",
    )

    args = parser.parse_args(argv)

    if args.enable_hf_transfer:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    token = args.token or _default_token()
    if not token:
        print(
            "Missing Hugging Face token.\n"
            "- Recommended: `export HF_TOKEN=...` (or set HUGGINGFACE_HUB_TOKEN)\n"
            "- Or: `huggingface-cli login` (then rerun)\n",
            file=sys.stderr,
        )
        return 2

    local_path = Path(args.path).expanduser()
    if not local_path.exists():
        print(f"Local path not found: {local_path}", file=sys.stderr)
        return 2
    if not local_path.is_dir():
        print(f"Local path is not a directory: {local_path}", file=sys.stderr)
        return 2

    repo_id = _build_repo_id(args.owner, args.repo)
    api = HfApi(token=token)

    try:
        if not args.no_create_repo:
            api.create_repo(
                repo_id=repo_id,
                repo_type=args.repo_type,
                private=bool(args.private),
                exist_ok=True,
            )

        ignore_patterns = [
            "**/.git/**",
            "**/__pycache__/**",
            "**/.DS_Store",
            "**/.uv-cache/**",
            "**/.venv/**",
        ] + list(args.ignore)

        commit_info = api.upload_folder(
            repo_id=repo_id,
            repo_type=args.repo_type,
            folder_path=str(local_path),
            commit_message=args.commit_message,
            ignore_patterns=ignore_patterns,
        )
    except HfHubHTTPError as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        print(
            "Tips:\n"
            "- Ensure the token has `write` permission\n"
            "- Check your network/proxy settings\n"
            "- If repo exists and you lack create permission, pass --no-create-repo\n",
            file=sys.stderr,
        )
        return 1

    url = getattr(commit_info, "commit_url", None) or f"https://huggingface.co/{repo_id}"
    print(f"Uploaded to: {url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
