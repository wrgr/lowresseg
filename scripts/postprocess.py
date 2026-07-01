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
                out_dir: Path, downsample: int = 2) -> None:
    """Skeletonize segments with coarse TEASAR params tuned for 1µm resolution.

    downsample: factor to downsample crops before TEASAR (2 = 2× faster, fewer nodes).
    """
    import kimimaro

    out_dir.mkdir(parents=True, exist_ok=True)
    eff_res = [r * downsample for r in res_nm]  # effective resolution after downsampling

    # Coarse params for 1µm data — scale=10/const=2000nm gives ~50-150 nodes per fragment
    teasar_params = {
        "scale": 10,
        "const": 2000,        # nm — minimum invalidation radius
        "pdrf_scale": 100000,
        "pdrf_exponent": 4,
        "soma_detection_threshold": 1100,
        "soma_acceptance_threshold": 3500,
        "soma_invalidation_scale": 1.0,
        "soma_invalidation_const": 300,
        "max_paths": None,
    }

    log.info("Skeletonizing %d segments (downsample=%dx, coarse TEASAR)...", len(seg_ids), downsample)
    n_written = 0
    PAD = 2
    for i, seg_id in enumerate(seg_ids):
        if i % 50 == 0:
            log.info("  skeletonizing %d/%d...", i, len(seg_ids))
        mask = np.where(seg == seg_id)
        if len(mask[0]) == 0:
            continue
        x0 = max(mask[0].min() - PAD, 0); x1 = min(mask[0].max() + PAD + 1, seg.shape[0])
        y0 = max(mask[1].min() - PAD, 0); y1 = min(mask[1].max() + PAD + 1, seg.shape[1])
        z0 = max(mask[2].min() - PAD, 0); z1 = min(mask[2].max() + PAD + 1, seg.shape[2])

        crop = seg[x0:x1, y0:y1, z0:z1]
        if downsample > 1:
            crop = crop[::downsample, ::downsample, ::downsample]

        skels = kimimaro.skeletonize(
            crop, teasar_params=teasar_params,
            anisotropy=tuple(eff_res),
            object_ids=[int(seg_id)],
            fix_branching=True, fix_borders=True, progress=False,
        )
        skel = skels.get(int(seg_id))
        if skel is None or len(skel.vertices) == 0:
            continue
        # Shift vertices to global nm coordinates (eff_res already baked in by kimimaro)
        skel.vertices[:, 0] += x0 * res_nm[0]
        skel.vertices[:, 1] += y0 * res_nm[1]
        skel.vertices[:, 2] += z0 * res_nm[2]
        _write_swc(skel, out_dir / f"{seg_id}.swc")
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

    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Meshing %d segments one-by-one (bbox crop to save RAM)...", len(seg_ids))

    n_written = 0
    PAD = 1
    for i, seg_id in enumerate(seg_ids):
        if i % 50 == 0:
            log.info("  meshing %d/%d...", i, len(seg_ids))
        mask = np.where(seg == seg_id)
        if len(mask[0]) == 0:
            continue
        x0, x1 = max(mask[0].min() - PAD, 0), min(mask[0].max() + PAD + 1, seg.shape[0])
        y0, y1 = max(mask[1].min() - PAD, 0), min(mask[1].max() + PAD + 1, seg.shape[1])
        z0, z1 = max(mask[2].min() - PAD, 0), min(mask[2].max() + PAD + 1, seg.shape[2])

        crop = seg[x0:x1, y0:y1, z0:z1]
        mesher = zmesh.Mesher(tuple(res_nm))
        mesher.mesh(crop)
        m = mesher.get(int(seg_id), normals=False)
        if hasattr(mesher, 'erase_buffer'):
            mesher.erase_buffer()
        if m is None or len(m.vertices) == 0:
            continue
        # Shift vertices to global nm coordinates
        m.vertices[:, 0] += x0 * res_nm[0]
        m.vertices[:, 1] += y0 * res_nm[1]
        m.vertices[:, 2] += z0 * res_nm[2]
        _write_obj(m, out_dir / f"{seg_id}.obj")
        n_written += 1

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
    parser.add_argument("--min-size", type=int, default=300,
                        help="Skip segments smaller than this many voxels")
    parser.add_argument("--max-size", type=int, default=2_000_000,
                        help="Skip segments larger than this many voxels — likely merge errors "
                             "(default 2M voxels ≈ 2000 µm³ at 1µm iso)")
    parser.add_argument("--downsample", type=int, default=2,
                        help="Downsample factor for skeletonization crops (default 2 = 2× faster, fewer nodes)")
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
    mask = (counts >= args.min_size) & (counts <= args.max_size)
    n_merge_errors = int((counts > args.max_size).sum())
    seg_ids, counts = seg_ids[mask], counts[mask]

    if n_merge_errors:
        log.info("Skipped %d likely merge errors (>%d voxels)", n_merge_errors, args.max_size)

    if args.max_segments is not None:
        order = np.argsort(counts)[::-1][: args.max_segments]
        seg_ids = seg_ids[order]
        counts = counts[order]

    log.info("Processing %d segments (min_size=%d, max_size=%d, largest=%d voxels)",
             len(seg_ids), args.min_size, args.max_size, int(counts.max()) if len(counts) else 0)

    skel_dir = store_path / "skeletons"
    mesh_dir = store_path / "meshes"

    # Load full volume — uint32 to save RAM (1664×1408×409×4 ≈ 3.8 GB)
    log.info("Loading full seg array (~%.1f GB)...", sx * sy * sz * 4 / 1e9)
    seg = np.array(seg_arr).astype(np.uint32)
    del seg_arr  # free zarr reference

    if not args.no_skeletons:
        skeletonize(seg, seg_ids, res_nm, skel_dir, downsample=args.downsample)

    if not args.no_meshes:
        mesh(seg, seg_ids, res_nm, mesh_dir)

    log.info("Done.")


if __name__ == "__main__":
    main()
