"""
Skeletonize and mesh neuron segments from pred_seg_filtered.

Outputs (written inside the zarr store directory):
  <zarr>/skeletons/      SWC files, one per segment ID
  <zarr>/meshes/         OBJ files, one per segment ID

Usage:
    python scripts/postprocess.py
    python scripts/postprocess.py --zarr data/minnie65_1um.zarr --seg pred_seg_filtered
    python scripts/postprocess.py --seg pred_seg_filtered --max-segments 200
    python scripts/postprocess.py --no-skeletons   # meshes only
    python scripts/postprocess.py --no-meshes       # skeletons only
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import zarr

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


def skeletonize(seg: np.ndarray, seg_ids: np.ndarray, res_nm: list[int],
                out_dir: Path) -> None:
    import kimimaro

    log.info("Skeletonizing %d segments with TEASAR...", len(seg_ids))
    # TEASAR scale/const are tuned for ~1µm resolution (units: nm).
    # scale=4 → 4× cross-section radius; const=500nm minimum radius.
    skels = kimimaro.skeletonize(
        seg,
        teasar_params={
            "scale": 4,
            "const": 500,          # nm
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_detection_threshold": 1100,
            "soma_acceptance_threshold": 3500,
            "soma_invalidation_scale": 1.0,
            "soma_invalidation_const": 300,
            "max_paths": None,
        },
        anisotropy=tuple(res_nm),  # voxel size in nm (x, y, z)
        object_ids=seg_ids.tolist(),
        fix_branching=True,
        fix_borders=True,
        progress=False,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    n_written = 0
    for seg_id, skel in skels.items():
        if len(skel.vertices) == 0:
            continue
        swc_path = out_dir / f"{seg_id}.swc"
        _write_swc(skel, swc_path)
        n_written += 1

    log.info("Wrote %d skeleton SWC files to %s", n_written, out_dir)


def _write_swc(skel, path: Path) -> None:
    """Write a kimimaro Skeleton as a minimal SWC file."""
    lines = ["# SWC from kimimaro TEASAR"]
    # vertices: (N, 3) in nm; radii stored per vertex if available
    radii = skel.radii if hasattr(skel, "radii") and skel.radii is not None else np.ones(len(skel.vertices))
    # build parent map from edges
    parent = {int(e[1]): int(e[0]) for e in skel.edges}
    for i, (v, r) in enumerate(zip(skel.vertices, radii)):
        p = parent.get(i, -1)
        lines.append(f"{i+1} 0 {v[0]:.2f} {v[1]:.2f} {v[2]:.2f} {r:.2f} {p if p < 0 else p+1}")
    path.write_text("\n".join(lines) + "\n")


def mesh(seg: np.ndarray, seg_ids: np.ndarray, res_nm: list[int],
         out_dir: Path) -> None:
    import zmesh

    log.info("Meshing %d segments...", len(seg_ids))
    out_dir.mkdir(parents=True, exist_ok=True)

    mesher = zmesh.Mesher(tuple(res_nm))  # voxel → nm scaling
    mesher.mesh(seg if seg.dtype == np.uint32 else seg.astype(np.uint32))

    n_written = 0
    for seg_id in seg_ids:
        m = mesher.get(int(seg_id), normals=False)
        if m is None or len(m.vertices) == 0:
            continue
        obj_path = out_dir / f"{seg_id}.obj"
        _write_obj(m, obj_path)
        n_written += 1

    mesher.erase_buffer()
    log.info("Wrote %d mesh OBJ files to %s", n_written, out_dir)


def _write_obj(m, path: Path) -> None:
    lines = []
    for v in m.vertices:
        lines.append(f"v {v[0]:.3f} {v[1]:.3f} {v[2]:.3f}")
    faces = m.faces.reshape(-1, 3) + 1  # OBJ is 1-indexed
    for f in faces:
        lines.append(f"f {f[0]} {f[1]} {f[2]}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/minnie65_1um.zarr")
    parser.add_argument("--seg", default="pred_seg_filtered",
                        help="Which seg array to process (pred_seg or pred_seg_filtered)")
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Process only the N largest segments (useful for testing)")
    parser.add_argument("--min-size", type=int, default=100,
                        help="Skip segments smaller than this many voxels")
    parser.add_argument("--no-skeletons", action="store_true")
    parser.add_argument("--no-meshes", action="store_true")
    args = parser.parse_args()

    store_path = Path(args.zarr)
    store = zarr.open_group(str(store_path), mode="r")

    if args.seg not in store:
        log.error("Array '%s' not found in %s. Available: %s",
                  args.seg, args.zarr, list(store.keys()))
        return

    seg_arr = store[args.seg]
    res_nm = list(seg_arr.attrs.get("resolution_nm", [1024, 1024, 1280]))
    sx, sy, sz = seg_arr.shape
    log.info("Counting segments in %s (shape=%s) via Z-slabs to save RAM...", args.seg, seg_arr.shape)

    # Count segment sizes without loading full volume
    from collections import Counter
    counts_map: Counter = Counter()
    slab = 32
    for z0 in range(0, sz, slab):
        z1 = min(z0 + slab, sz)
        chunk = np.array(seg_arr[:, :, z0:z1])
        ids, cnts = np.unique(chunk, return_counts=True)
        for i, c in zip(ids, cnts):
            counts_map[int(i)] += int(c)
        del chunk
    counts_map.pop(0, None)  # remove background

    seg_ids = np.array(sorted(counts_map.keys()), dtype=np.uint64)
    counts = np.array([counts_map[int(i)] for i in seg_ids], dtype=np.int64)
    mask = counts >= args.min_size
    seg_ids, counts = seg_ids[mask], counts[mask]

    if args.max_segments is not None:
        order = np.argsort(counts)[::-1][: args.max_segments]
        seg_ids = seg_ids[order]
        counts = counts[order]

    log.info("Processing %d segments (min_size=%d, largest=%d voxels)",
             len(seg_ids), args.min_size, int(counts.max()) if len(counts) else 0)

    skel_dir = store_path / "skeletons"
    mesh_dir = store_path / "meshes"

    # Load full volume — uint32 to save RAM (1664×1408×409×4 ≈ 3.8 GB)
    log.info("Loading full seg array (~%.1f GB)...", sx * sy * sz * 4 / 1e9)
    seg = np.array(seg_arr).astype(np.uint32)
    del seg_arr  # free zarr reference

    if not args.no_skeletons:
        skeletonize(seg, seg_ids, res_nm, skel_dir)

    if not args.no_meshes:
        mesh(seg, seg_ids, res_nm, mesh_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
