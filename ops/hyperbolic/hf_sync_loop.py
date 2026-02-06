#!/usr/bin/env python3
"""
Continuously sync selected nanochat artifacts to a Hugging Face repo.

Usage:
  python ops/hyperbolic/hf_sync_loop.py --repo-id user/repo --local-dir /data/nanochat
"""

import argparse
import datetime as dt
import glob
import os
import sys
import time
from typing import Iterable, List

from huggingface_hub import HfApi
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

DEFAULT_IGNORE_PATTERNS = [
    "**/*.tmp",
    "**/*.lock",
]


def parse_bool(value: str) -> bool:
    return value.lower() in {"1", "true", "yes", "y"}


def any_matches(local_dir: str, allow_patterns: Iterable[str]) -> bool:
    for pattern in allow_patterns:
        if glob.glob(os.path.join(local_dir, pattern), recursive=True):
            return True
    return False


def sync_once(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    private: bool,
    local_dir: str,
    allow_patterns: List[str],
    ignore_patterns: List[str],
) -> None:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=private, exist_ok=True)

    if not any_matches(local_dir, allow_patterns):
        print("[hf-sync] no matching files yet, skipping upload")
        return

    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    commit_message = f"checkpoint sync {timestamp}"
    print(f"[hf-sync] uploading artifacts from {local_dir} -> {repo_type}:{repo_id}")
    api.upload_folder(
        repo_id=repo_id,
        repo_type=repo_type,
        folder_path=local_dir,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        commit_message=commit_message,
    )
    print("[hf-sync] upload complete")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync nanochat checkpoints to Hugging Face")
    parser.add_argument("--repo-id", required=True, help="Hugging Face repo id, e.g. user/nanochat-checkpoints")
    parser.add_argument("--repo-type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--private", default="true", help="Create repo private if it does not exist")
    parser.add_argument("--local-dir", required=True, help="NANOCHAT_BASE_DIR path")
    parser.add_argument("--interval-seconds", type=int, default=1200, help="Sync interval for loop mode")
    parser.add_argument("--once", action="store_true", help="Run one upload and exit")
    parser.add_argument("--allow-pattern", action="append", default=[], help="Can be repeated")
    parser.add_argument("--ignore-pattern", action="append", default=[], help="Can be repeated")
    parser.add_argument("--token", default=os.environ.get("HF_TOKEN"), help="HF token (or set HF_TOKEN)")
    args = parser.parse_args()

    if not os.path.isdir(args.local_dir):
        print(f"[hf-sync] local directory does not exist: {args.local_dir}", file=sys.stderr)
        return 2

    allow_patterns = args.allow_pattern if args.allow_pattern else list(DEFAULT_ALLOW_PATTERNS)
    ignore_patterns = args.ignore_pattern if args.ignore_pattern else list(DEFAULT_IGNORE_PATTERNS)
    private = parse_bool(str(args.private))

    api = HfApi(token=args.token)
    try:
        if args.once:
            sync_once(
                api=api,
                repo_id=args.repo_id,
                repo_type=args.repo_type,
                private=private,
                local_dir=args.local_dir,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
            )
            return 0

        while True:
            try:
                sync_once(
                    api=api,
                    repo_id=args.repo_id,
                    repo_type=args.repo_type,
                    private=private,
                    local_dir=args.local_dir,
                    allow_patterns=allow_patterns,
                    ignore_patterns=ignore_patterns,
                )
            except HfHubHTTPError as err:
                print(f"[hf-sync] hub error: {err}", file=sys.stderr)
            except Exception as err:  # pylint: disable=broad-except
                print(f"[hf-sync] unexpected error: {err}", file=sys.stderr)
            time.sleep(max(30, args.interval_seconds))
    except KeyboardInterrupt:
        print("[hf-sync] interrupted")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
