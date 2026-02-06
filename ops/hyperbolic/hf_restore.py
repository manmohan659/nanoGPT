#!/usr/bin/env python3
"""
Restore nanochat artifacts from a Hugging Face repo into NANOCHAT_BASE_DIR.
"""

import argparse
import glob
import os
import sys
from typing import List

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError


DEFAULT_ALLOW_PATTERNS = [
    "base_checkpoints/**",
    "chatsft_checkpoints/**",
    "chatrl_checkpoints/**",
    "tokenizer/**",
    "report/**",
    "report.md",
    "identity_conversations.jsonl",
]


def has_local_checkpoints(local_dir: str) -> bool:
    pattern = os.path.join(local_dir, "base_checkpoints", "*", "model_*.pt")
    return bool(glob.glob(pattern))


def main() -> int:
    parser = argparse.ArgumentParser(description="Restore nanochat checkpoints from Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/nanochat-checkpoints")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--local-dir", required=True, help="NANOCHAT_BASE_DIR path")
    parser.add_argument("--allow-pattern", action="append", default=[], help="Can be repeated")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    parser.add_argument("--skip-if-local", action="store_true", help="Skip restore if local base checkpoints exist")
    args = parser.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)

    if args.skip_if_local and has_local_checkpoints(args.local_dir):
        print("[hf-restore] local base checkpoints already exist, skipping restore")
        return 0

    allow_patterns: List[str] = args.allow_pattern if args.allow_pattern else list(DEFAULT_ALLOW_PATTERNS)

    try:
        print(f"[hf-restore] restoring from {args.repo_type}:{args.repo_id} -> {args.local_dir}")
        snapshot_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            token=args.token,
            local_dir=args.local_dir,
            local_dir_use_symlinks=False,
            allow_patterns=allow_patterns,
            resume_download=True,
        )
    except HfHubHTTPError as err:
        # Repo may not exist yet on first run; this should not fail the launch.
        print(f"[hf-restore] hub error (continuing): {err}", file=sys.stderr)
        return 0
    except Exception as err:  # pylint: disable=broad-except
        print(f"[hf-restore] unexpected restore error (continuing): {err}", file=sys.stderr)
        return 0

    print("[hf-restore] restore complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
