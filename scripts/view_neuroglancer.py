"""
Serve MICrONS zarr data (EM + optionally seg/affinities) in a browser
via neuroglancer.

Usage:
    python scripts/view_neuroglancer.py                    # EM only
    python scripts/view_neuroglancer.py --seg              # EM + ground-truth seg
    python scripts/view_neuroglancer.py --seg --aff        # + predicted affinities (if present)
    python scripts/view_neuroglancer.py --zarr data/minnie65_1um.zarr
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import neuroglancer
import numpy as np
import zarr


def add_zarr_layer(viewer_txn, name: str, arr: zarr.Array, layer_type: str,
                   res_nm: list[int] | None = None, shader: str | None = None) -> None:
    res = res_nm or arr.attrs.get("resolution_nm", [1000, 1000, 1000])
    # Wrap zarr array as a LocalVolume
    data = np.array(arr)  # load into RAM; <1 GB is fine
    if data.ndim == 3:
        data = data.T  # neuroglancer expects (Z, Y, X)
    elif data.ndim == 4:
        data = np.moveaxis(data, -1, 0)  # (C, X, Y, Z) → keep as is after T

    vol = neuroglancer.LocalVolume(
        data=data,
        dimensions=neuroglancer.CoordinateSpace(
            names=["x", "y", "z"],
            units="nm",
            scales=res,
        ),
    )

    if layer_type == "image":
        layer = neuroglancer.ImageLayer(source=vol, shader=shader or "#uicontrol float brightness\nvoid main() { emitGrayscale(brightness + toNormalized(getDataValue())); }")
    else:
        layer = neuroglancer.SegmentationLayer(source=vol)

    viewer_txn.layers[name] = layer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/minnie65_1um.zarr")
    parser.add_argument("--seg", action="store_true", help="Show ground-truth seg layer")
    parser.add_argument("--aff", action="store_true", help="Show predicted affinity layer (if present)")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9090)
    args = parser.parse_args()

    neuroglancer.set_server_bind_address(args.bind, bind_port=args.port)
    viewer = neuroglancer.Viewer()

    store_path = Path(args.zarr)
    if not store_path.exists():
        print(f"zarr not found: {store_path}. Run download_full_volume.py first.")
        return

    z = zarr.open_group(str(store_path), mode="r")

    with viewer.txn() as txn:
        if "em" in z:
            add_zarr_layer(txn, "EM", z["em"], "image",
                           res_nm=z["em"].attrs.get("resolution_nm", [1024, 1024, 1280]))

        if args.seg and "seg" in z:
            add_zarr_layer(txn, "seg (ground truth)", z["seg"], "segmentation",
                           res_nm=z["seg"].attrs.get("resolution_nm", [1024, 1024, 1280]))

        if args.aff and "affinities" in z:
            aff = z["affinities"]
            # Show x-affinity channel as image
            add_zarr_layer(txn, "affinities (x)", aff, "image",
                           res_nm=z["em"].attrs.get("resolution_nm", [1024, 1024, 1280]))

    url = str(viewer)
    print()
    print("=" * 60)
    print(f"  Neuroglancer viewer: {url}")
    print("=" * 60)
    print()
    print("Open the URL above in your browser.")
    print("Press Ctrl+C to stop.")
    print()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
