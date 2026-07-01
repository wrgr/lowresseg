"""
Update the NeuroGlass study with a new glance pointing at the predicted segmentation.

Usage:
    python scripts/update_neuroglass.py
    python scripts/update_neuroglass.py --study-id 06a43f00-3207-7c96-8000-fae320380bcf
    python scripts/update_neuroglass.py --hf-repo wrgr2026/minnietest1 --hf-subdir seg
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import urllib.parse
from pathlib import Path

import requests

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

BASE = "https://www.neuroglass.io"
NG_DEMO = "https://neuroglancer-demo.appspot.com"


def _session(token: str) -> requests.Session:
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    s.verify = os.environ.get("REQUESTS_CA_BUNDLE", "/root/.ccr/ca-bundle.crt")
    # Disable redirects globally — neuroglass returns 303 to login if token is missing;
    # a 200 with allow_redirects=False means auth succeeded.
    s.max_redirects = 0
    return s


def _get(s: requests.Session, path: str) -> dict:
    r = s.get(f"{BASE}{path}", allow_redirects=False)
    r.raise_for_status()
    return r.json()


def _post(s: requests.Session, path: str, payload: dict) -> dict:
    r = s.post(f"{BASE}{path}", json=payload, allow_redirects=False)
    if not r.ok:
        log.error("POST %s → %s: %s", path, r.status_code, r.text[:300])
    r.raise_for_status()
    return r.json()


def _patch(s: requests.Session, path: str, payload: dict) -> dict:
    r = s.patch(f"{BASE}{path}", json=payload, allow_redirects=False)
    if not r.ok:
        log.error("PATCH %s → %s: %s", path, r.status_code, r.text[:300])
    r.raise_for_status()
    return r.json()


def build_ng_state(hf_repo: str, hf_subdir: str | None) -> dict:
    """Build a Neuroglancer state dict with EM + predicted seg layers from HF."""
    base_url = f"https://huggingface.co/datasets/{hf_repo}/resolve/main"
    if hf_subdir:
        base_url = f"{base_url}/{hf_subdir}"
    pred_src = f"precomputed://{base_url}"

    return {
        "layers": [
            {
                "name": "EM",
                "type": "image",
                "source": "precomputed://https://storage.googleapis.com/iarpa_microns/minnie/minnie65/em",
                "opacity": 1,
            },
            {
                "name": "seg (predicted, 1µm)",
                "type": "segmentation",
                "source": pred_src,
                "selectedAlpha": 0.5,
                "notSelectedAlpha": 0.05,
            },
        ],
        "layout": "4panel",
        "position": [880, 704, 200],
        "dimensions": [
            {"name": "x", "scale": [1.024e-6, "m"]},
            {"name": "y", "scale": [1.024e-6, "m"]},
            {"name": "z", "scale": [1.28e-6, "m"]},
        ],
        "projectionScale": 2048,
        "crossSectionScale": 8,
    }


def state_to_url(state: dict) -> str:
    fragment = urllib.parse.quote(json.dumps(state, separators=(",", ":")))
    return f"{NG_DEMO}#!{fragment}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="NeuroGlass bearer token (or NEUROGLASS_TOKEN env)")
    parser.add_argument("--study-id", default="06a43f00-3207-7c96-8000-fae320380bcf",
                        help="Study public_id to update")
    parser.add_argument("--hf-repo", default=None, help="HF dataset repo (or HF_REPO env)")
    parser.add_argument("--hf-subdir", default="seg", help="Subdirectory within HF repo")
    parser.add_argument("--glance-name", default="pred seg (1µm lowresseg model)")
    args = parser.parse_args()

    # Load token
    token = args.token or os.environ.get("NEUROGLASS_TOKEN")
    if not token:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            m = re.search(r"NEUROGLASS_TOKEN=(\S+)", env_path.read_text())
            if m:
                token = m.group(1)
    if not token:
        log.error("No NEUROGLASS_TOKEN found. Set env var or pass --token.")
        return

    hf_repo = args.hf_repo or os.environ.get("HF_REPO")
    if not hf_repo:
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            m = re.search(r"HF_REPO=(\S+)", env_path.read_text())
            if m:
                hf_repo = m.group(1)
    if not hf_repo:
        log.error("No HF_REPO found. Set env var or pass --hf-repo.")
        return

    s = _session(token)

    # Verify token
    study = _get(s, f"/api/studies/{args.study_id}")
    log.info("Study: %s (id=%s)", study["name"], study["public_id"])

    # Build neuroglancer state
    ng_state = build_ng_state(hf_repo, args.hf_subdir)
    ng_url = state_to_url(ng_state)

    # Check if a glance with this name already exists
    glances = _get(s, f"/api/studies/{args.study_id}/glances")
    existing = next((g for g in glances["items"] if g["name"] == args.glance_name), None)

    if existing:
        log.info("Updating existing glance: %s (%s)", existing["name"], existing["public_id"])
        result = _patch(s, f"/api/studies/glances/{existing['public_id']}", {
            "name": args.glance_name,
            "description": f"Predicted segmentation from lowresseg model — source: {hf_repo}",
            "url": ng_url,
            "state": ng_state,
        })
        log.info("Glance updated: %s", result["public_id"])
    else:
        log.info("Creating new glance: %s", args.glance_name)
        result = _post(s, f"/api/studies/{args.study_id}/glances", {
            "name": args.glance_name,
            "description": f"Predicted segmentation from lowresseg model — source: {hf_repo}",
            "url": ng_url,
            "state": ng_state,
            "automatic": False,
        })
        log.info("Glance created: %s", result["public_id"])

    # Update study description to mention pred seg
    _patch(s, f"/api/studies/{args.study_id}", {
        "description": (
            "MICrONS minnie65 EM + segmentation at 1µm isotropic (mip7). "
            "Includes ground-truth seg and a predicted segmentation from a "
            "trained AffinityUNet (lowresseg model)."
        ),
        "tags": ["microns", "em", "segmentation", "1um", "lowresseg", "predicted-seg", "affinityunet"],
    })
    log.info("Study description updated.")

    log.info("")
    log.info("View in NeuroGlass: https://www.neuroglass.io/studies/%s", args.study_id)
    log.info("Direct Neuroglancer URL: %s", ng_url[:120] + "...")


if __name__ == "__main__":
    main()
