"""
Serve MICrONS zarr data (EM + optionally seg/affinities) in a browser
via neuroglancer.

Usage:
    python scripts/view_neuroglancer.py                    # EM only
    python scripts/view_neuroglancer.py --seg              # EM + ground-truth seg
    python scripts/view_neuroglancer.py --seg --aff        # + predicted affinities
    python scripts/view_neuroglancer.py --tunnel           # expose via public ngrok URL
    python scripts/view_neuroglancer.py --zarr data/minnie65_1um.zarr
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import neuroglancer
import numpy as np
import zarr


def add_zarr_layer(txn, name: str, arr: zarr.Array, layer_type: str,
                   res_nm: list[int] | None = None) -> None:
    res = res_nm or arr.attrs.get("resolution_nm", [1000, 1000, 1000])
    data = np.array(arr)
    if data.ndim == 3:
        data = data.T  # neuroglancer wants (Z, Y, X)

    vol = neuroglancer.LocalVolume(
        data=data,
        dimensions=neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units="nm",
            scales=res,
        ),
    )

    if layer_type == "image":
        txn.layers[name] = neuroglancer.ImageLayer(source=vol)
    else:
        txn.layers[name] = neuroglancer.SegmentationLayer(source=vol)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/minnie65_1um.zarr")
    parser.add_argument("--seg", action="store_true", help="Show ground-truth seg layer")
    parser.add_argument("--aff", action="store_true", help="Show predicted affinity layer (if present)")
    parser.add_argument("--pred-seg", action="store_true", help="Show predicted segmentation (if present)")
    parser.add_argument("--pred-seg-filtered", action="store_true", help="Show size-filtered predicted seg (if present)")
    parser.add_argument("--port", type=int, default=9090)
    parser.add_argument("--tunnel", action="store_true",
                        help="Open a public ngrok tunnel (needed when running remotely)")
    parser.add_argument("--ngrok-token", default=None,
                        help="ngrok authtoken (or set NGROK_AUTHTOKEN env var)")
    args = parser.parse_args()

    # Bind to 0.0.0.0 so the tunnel can reach it
    bind = "0.0.0.0" if args.tunnel else "127.0.0.1"
    neuroglancer.set_server_bind_address(bind, bind_port=args.port)
    viewer = neuroglancer.Viewer()

    store_path = Path(args.zarr)
    if not store_path.exists():
        print(f"zarr not found: {store_path}. Run download_full_volume.py first.")
        return

    z = zarr.open_group(str(store_path), mode="r")
    res_em = list(z["em"].attrs.get("resolution_nm", [1024, 1024, 1280])) if "em" in z else [1024, 1024, 1280]

    with viewer.txn() as txn:
        if "em" in z:
            print("Loading EM into memory...")
            add_zarr_layer(txn, "EM", z["em"], "image", res_nm=res_em)

        if args.seg and "seg" in z:
            print("Loading ground-truth seg...")
            add_zarr_layer(txn, "seg (ground truth)", z["seg"], "segmentation",
                           res_nm=list(z["seg"].attrs.get("resolution_nm", res_em)))

        if args.pred_seg and "pred_seg" in z:
            print("Loading predicted seg...")
            add_zarr_layer(txn, "seg (predicted)", z["pred_seg"], "segmentation", res_nm=res_em)

        if args.pred_seg_filtered and "pred_seg_filtered" in z:
            print("Loading size-filtered predicted seg...")
            add_zarr_layer(txn, "seg (predicted, filtered)", z["pred_seg_filtered"], "segmentation", res_nm=res_em)

        if args.aff and "affinities" in z:
            aff_data = np.array(z["affinities"])  # (3, X, Y, Z)
            # Show mean affinity as an image
            mean_aff = aff_data.mean(axis=0)
            vol = neuroglancer.LocalVolume(
                data=mean_aff.T,
                dimensions=neuroglancer.CoordinateSpace(
                    names=["x", "y", "z"], units="nm", scales=res_em),
            )
            txn.layers["affinities (mean)"] = neuroglancer.ImageLayer(source=vol)

    local_url = str(viewer)

    if args.tunnel:
        try:
            from pyngrok import ngrok, conf
            if args.ngrok_token:
                conf.get_default().auth_token = args.ngrok_token
            tunnel = ngrok.connect(args.port, "http")
            # Replace the local host in the viewer URL with the tunnel URL
            public_base = tunnel.public_url
            viewer_path = local_url.split(f":{args.port}", 1)[1]
            public_url = public_base + viewer_path
            print()
            print("=" * 60)
            print(f"  Public URL (open in browser):")
            print(f"  {public_url}")
            print("=" * 60)
        except Exception as e:
            print(f"Tunnel failed: {e}")
            print(f"Local URL (not reachable from outside): {local_url}")
    else:
        print()
        print("=" * 60)
        print(f"  Neuroglancer: {local_url}")
        print("  (use --tunnel to get a public URL when running remotely)")
        print("=" * 60)

    print("\nPress Ctrl+C to stop.\n")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
