"""
Convert pred_seg_filtered (zarr) → neuroglancer precomputed directory with:
  - Multiscale segmentation pyramid  (mip 0 = 1µm, mip 1 = 2µm, mip 2 = 4µm)
  - Precomputed meshes               (neuroglancer legacy mesh format)
  - Precomputed skeletons            (neuroglancer skeleton binary format)

Output:
  <out_dir>/
    info                    ← precomputed seg info
    <mip0_chunk_dir>/       ← full-res chunks
    <mip1_chunk_dir>/       ← 2× downsampled
    <mip2_chunk_dir>/       ← 4× downsampled
    mesh/
      info
      <seg_id>:0            ← binary mesh fragment per segment
    skeletons/
      info
      <seg_id>              ← binary skeleton per segment

Serve locally with:
    python -m http.server 9191 --directory <out_dir>
Then add as a neuroglancer precomputed source:
    precomputed://http://localhost:9191

Usage:
    python scripts/build_precomputed.py
    python scripts/build_precomputed.py --seg pred_seg --out data/precomputed
    python scripts/build_precomputed.py --max-segments 50   # quick test
    python scripts/build_precomputed.py --no-skeletons --no-meshes  # pyramid only
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
from pathlib import Path

import numpy as np
import zarr

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")


# ---------------------------------------------------------------------------
# Neuroglancer precomputed info helpers
# ---------------------------------------------------------------------------

def _scale_info(res_nm: list[int], size: list[int], mip: int,
                chunk: list[int]) -> dict:
    factor = 2 ** mip
    return {
        "chunk_sizes": [chunk],
        "encoding": "compressed_segmentation",
        "compressed_segmentation_block_size": [8, 8, 8],
        "key": f"{'_'.join(str(r * factor) for r in res_nm)}",
        "resolution": [r * factor for r in res_nm],
        "size": [max(1, s // factor) for s in size],
        "voxel_offset": [0, 0, 0],
    }


def write_seg_info(out_dir: Path, res_nm: list[int], size: list[int],
                   num_mips: int = 3, chunk: list[int] | None = None) -> list[dict]:
    chunk = chunk or [128, 128, 128]
    scales = [_scale_info(res_nm, size, m, chunk) for m in range(num_mips)]
    info = {
        "type": "segmentation",
        "data_type": "uint64",
        "num_channels": 1,
        "scales": scales,
    }
    (out_dir / "info").write_text(json.dumps(info, indent=2))
    return scales


# ---------------------------------------------------------------------------
# Chunk writing (compressed_segmentation via cloud-volume encoder)
# ---------------------------------------------------------------------------

def write_scale(seg: np.ndarray, scale_info: dict, out_dir: Path) -> None:
    import compressed_segmentation as cseg

    key = scale_info["key"]
    size = scale_info["size"]     # [x, y, z]
    chunk = scale_info["chunk_sizes"][0]  # [cx, cy, cz]
    scale_dir = out_dir / key
    scale_dir.mkdir(parents=True, exist_ok=True)

    xs, ys, zs = size
    cx, cy, cz = chunk

    x_starts = range(0, xs, cx)
    y_starts = range(0, ys, cy)
    z_starts = range(0, zs, cz)

    total = len(x_starts) * len(y_starts) * len(z_starts)
    done = 0
    for x0 in x_starts:
        xe = min(x0 + cx, xs)
        for y0 in y_starts:
            ye = min(y0 + cy, ys)
            for z0 in z_starts:
                ze = min(z0 + cz, zs)
                chunk_data = seg[x0:xe, y0:ye, z0:ze].astype(np.uint32)
                # Pad to full chunk size (required by compressed_segmentation encoder)
                pad = [(0, cx - (xe - x0)), (0, cy - (ye - y0)), (0, cz - (ze - z0))]
                chunk_data = np.pad(chunk_data, pad, mode="constant")
                encoded = cseg.compress(chunk_data, order='C')
                fname = scale_dir / f"{x0}-{xe}_{y0}-{ye}_{z0}-{ze}"
                fname.write_bytes(encoded)
                done += 1
                if done % max(1, total // 10) == 0 or done == total:
                    log.info("  [%s] %d/%d chunks", key, done, total)


def downsample(seg: np.ndarray, factor: int = 2) -> np.ndarray:
    """Majority-label downsample by factor in each dimension."""
    sx, sy, sz = seg.shape
    nx = sx // factor * factor
    ny = sy // factor * factor
    nz = sz // factor * factor
    cropped = seg[:nx, :ny, :nz]
    # Reshape into (nx/f, f, ny/f, f, nz/f, f) then pick mode along axes 1,3,5
    r = cropped.reshape(nx // factor, factor, ny // factor, factor, nz // factor, factor)
    # Use first voxel of each block (fast; for seg "nearest" is fine at 2× steps)
    return r[:, 0, :, 0, :, 0].copy()


# ---------------------------------------------------------------------------
# Meshes — neuroglancer legacy precomputed format
# ---------------------------------------------------------------------------

def write_mesh_info(mesh_dir: Path) -> None:
    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "info").write_text(json.dumps({"@type": "neuroglancer_legacy_mesh"}))


def _encode_mesh(vertices: np.ndarray, faces: np.ndarray) -> bytes:
    """Encode as neuroglancer legacy mesh binary (little-endian)."""
    # vertices: (N, 3) float32 nm coords; faces: (M, 3) uint32 indices
    n_verts = len(vertices)
    verts_f32 = vertices.astype(np.float32)
    faces_u32 = faces.astype(np.uint32)
    buf = struct.pack("<I", n_verts)
    buf += verts_f32.tobytes()
    buf += faces_u32.tobytes()
    return buf


def write_meshes(seg: np.ndarray, seg_ids: np.ndarray, res_nm: list[int],
                 mesh_dir: Path) -> None:
    import zmesh

    log.info("Meshing %d segments...", len(seg_ids))
    mesher = zmesh.Mesher(tuple(res_nm))
    mesher.mesh(seg.astype(np.uint32))

    written = 0
    for sid in seg_ids:
        m = mesher.get(int(sid), normals=False)
        if m is None or len(m.vertices) == 0:
            continue
        encoded = _encode_mesh(m.vertices, m.faces.reshape(-1, 3))
        (mesh_dir / f"{sid}:0").write_bytes(encoded)
        written += 1

    if hasattr(mesher, 'erase_buffer'):
            mesher.erase_buffer()
    log.info("Wrote %d mesh fragments to %s", written, mesh_dir)


# ---------------------------------------------------------------------------
# Skeletons — neuroglancer precomputed binary format
# ---------------------------------------------------------------------------

def write_skeleton_info(skel_dir: Path, res_nm: list[int]) -> None:
    skel_dir.mkdir(parents=True, exist_ok=True)
    info = {
        "@type": "neuroglancer_skeletons",
        "transform": [
            res_nm[0], 0, 0, 0,
            0, res_nm[1], 0, 0,
            0, 0, res_nm[2], 0,
        ],
        "vertex_attributes": [],
    }
    (skel_dir / "info").write_text(json.dumps(info))


def _encode_skeleton(vertices: np.ndarray, edges: np.ndarray) -> bytes:
    """Encode neuroglancer precomputed skeleton binary."""
    # Format: num_vertices (uint32), num_edges (uint32),
    #         vertices (float32 x,y,z * N), edges (uint32 pair * E)
    n_v = len(vertices)
    n_e = len(edges)
    buf = struct.pack("<II", n_v, n_e)
    buf += vertices.astype(np.float32).tobytes()
    buf += edges.astype(np.uint32).tobytes()
    return buf


def write_skeletons(seg: np.ndarray, seg_ids: np.ndarray, res_nm: list[int],
                    skel_dir: Path) -> None:
    import kimimaro

    log.info("Skeletonizing %d segments with TEASAR...", len(seg_ids))
    skels = kimimaro.skeletonize(
        seg,
        teasar_params={
            "scale": 4,
            "const": 500,
            "pdrf_scale": 100000,
            "pdrf_exponent": 4,
            "soma_detection_threshold": 1100,
            "soma_acceptance_threshold": 3500,
            "soma_invalidation_scale": 1.0,
            "soma_invalidation_const": 300,
            "max_paths": None,
        },
        anisotropy=tuple(res_nm),
        object_ids=seg_ids.tolist(),
        fix_branching=True,
        fix_borders=True,
        progress=False,
    )

    written = 0
    for sid, skel in skels.items():
        if len(skel.vertices) == 0:
            continue
        # vertices in nm (kimimaro returns physical coords when anisotropy is set)
        encoded = _encode_skeleton(skel.vertices, skel.edges)
        (skel_dir / str(sid)).write_bytes(encoded)
        written += 1

    log.info("Wrote %d skeletons to %s", written, skel_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zarr", default="data/minnie65_1um.zarr")
    parser.add_argument("--seg", default="pred_seg_filtered")
    parser.add_argument("--out", default="data/minnie65_precomputed")
    parser.add_argument("--num-mips", type=int, default=3,
                        help="Number of resolution levels (0=full, 1=2×, 2=4×, ...)")
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--max-segments", type=int, default=None,
                        help="Process only the N largest segments")
    parser.add_argument("--no-skeletons", action="store_true")
    parser.add_argument("--no-meshes", action="store_true")
    parser.add_argument("--no-pyramid", action="store_true",
                        help="Skip writing the segmentation pyramid (meshes/skels only)")
    args = parser.parse_args()

    store = zarr.open_group(args.zarr, mode="r")
    if args.seg not in store:
        log.error("'%s' not in %s. Available: %s", args.seg, args.zarr, list(store.keys()))
        return

    seg_arr = store[args.seg]
    res_nm = list(seg_arr.attrs.get("resolution_nm", [1024, 1024, 1280]))
    sx, sy, sz = seg_arr.shape

    # Count via Z-slabs to avoid double-loading the full volume
    from collections import Counter
    counts_map: Counter = Counter()
    log.info("Counting segments in %s via Z-slabs...", args.seg)
    slab = 32
    for z0 in range(0, sz, slab):
        z1 = min(z0 + slab, sz)
        chunk_data = np.array(seg_arr[:, :, z0:z1])
        ids, cnts = np.unique(chunk_data, return_counts=True)
        for i, c in zip(ids, cnts):
            counts_map[int(i)] += int(c)
        del chunk_data
    counts_map.pop(0, None)

    seg_ids = np.array(sorted(counts_map.keys()), dtype=np.uint64)
    counts = np.array([counts_map[int(i)] for i in seg_ids], dtype=np.int64)
    mask = counts >= args.min_size
    seg_ids, counts = seg_ids[mask], counts[mask]
    if args.max_segments:
        order = np.argsort(counts)[::-1][: args.max_segments]
        seg_ids, counts = seg_ids[order], counts[order]
    log.info("%d segments to process (largest=%d voxels)",
             len(seg_ids), counts.max() if len(counts) else 0)

    log.info("Loading %s %s...", args.seg, seg_arr.shape)
    seg = np.array(seg_arr).astype(np.uint32)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    chunk = [args.chunk] * 3

    # --- Segmentation pyramid ---
    if not args.no_pyramid:
        log.info("Writing segmentation pyramid (%d mips)...", args.num_mips)
        scales = write_seg_info(out_dir, res_nm, [sx, sy, sz],
                                num_mips=args.num_mips, chunk=chunk)
        current = seg
        for mip, scale in enumerate(scales):
            log.info("MIP %d: %s", mip, scale["size"])
            write_scale(current, scale, out_dir)
            if mip < args.num_mips - 1:
                current = downsample(current, factor=2)

    # --- Meshes ---
    if not args.no_meshes:
        mesh_dir = out_dir / "mesh"
        write_mesh_info(mesh_dir)
        write_meshes(seg, seg_ids, res_nm, mesh_dir)

    # --- Skeletons ---
    if not args.no_skeletons:
        skel_dir = out_dir / "skeletons"
        write_skeleton_info(skel_dir, res_nm)
        write_skeletons(seg, seg_ids, res_nm, skel_dir)

    log.info("Done. Serve with:")
    log.info("  python -m http.server 9191 --directory %s", out_dir)
    log.info("Then open in neuroglancer as: precomputed://http://localhost:9191")


if __name__ == "__main__":
    main()
