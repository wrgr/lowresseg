"""
Upload a neuroglancer precomputed directory to a Hugging Face dataset repo
so it can be served directly to neuroglancer browsers.

After upload, the neuroglancer source URL is:
  precomputed://https://huggingface.co/datasets/<owner>/<repo>/resolve/main/<subdir>

Usage:
    # First time — creates the repo if it doesn't exist:
    python scripts/upload_hf.py --token hf_xxx --repo myname/minnie65-seg

    # Re-upload after rebuilding:
    python scripts/upload_hf.py --token hf_xxx --repo myname/minnie65-seg

    # Upload a specific precomputed dir (default: data/minnie65_precomputed):
    python scripts/upload_hf.py --token hf_xxx --repo myname/minnie65-seg \\
        --local data/minnie65_precomputed --subdir seg

    # Use HF_TOKEN env var instead of --token:
    export HF_TOKEN=hf_xxx
    python scripts/upload_hf.py --repo myname/minnie65-seg
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True,
                        help="HF dataset repo in owner/name format, e.g. myname/minnie65-seg")
    parser.add_argument("--token", default=None,
                        help="Hugging Face write token (or set HF_TOKEN env var)")
    parser.add_argument("--local", default="data/minnie65_precomputed",
                        help="Local precomputed directory to upload")
    parser.add_argument("--subdir", default="",
                        help="Subdirectory inside the HF repo (leave empty for root)")
    parser.add_argument("--public", action="store_true", default=True,
                        help="Make the repo public (default: true)")
    parser.add_argument("--private", dest="public", action="store_false",
                        help="Make the repo private")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        log.error("Provide --token or set HF_TOKEN environment variable")
        return

    local_dir = Path(args.local)
    if not local_dir.exists():
        log.error("Local directory not found: %s", local_dir)
        log.error("Run build_precomputed.py first.")
        return

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    # Create repo if it doesn't exist
    try:
        create_repo(
            repo_id=args.repo,
            repo_type="dataset",
            private=not args.public,
            exist_ok=True,
            token=token,
        )
        log.info("Dataset repo: https://huggingface.co/datasets/%s", args.repo)
    except Exception as e:
        log.error("Failed to create/access repo: %s", e)
        return

    # Count files for progress reporting
    all_files = list(local_dir.rglob("*"))
    all_files = [f for f in all_files if f.is_file()]
    log.info("Uploading %d files from %s ...", len(all_files), local_dir)
    log.info("(Large volumes may take several minutes)")

    path_in_repo = args.subdir or None

    api.upload_folder(
        folder_path=str(local_dir),
        repo_id=args.repo,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        ignore_patterns=["*.log", "__pycache__", ".DS_Store"],
        commit_message="Upload neuroglancer precomputed segmentation",
        token=token,
    )

    base_url = f"https://huggingface.co/datasets/{args.repo}/resolve/main"
    if args.subdir:
        base_url = f"{base_url}/{args.subdir}"

    log.info("Upload complete!")
    log.info("")
    log.info("Neuroglancer precomputed URL:")
    log.info("  precomputed://%s", base_url)
    log.info("")
    log.info("Add to a neuroglancer state as a segmentation layer source.")
    log.info("Or open directly:")
    log.info("  https://neuroglancer-demo.appspot.com/#!")
    log.info('  + add source: precomputed://%s', base_url)


if __name__ == "__main__":
    main()
