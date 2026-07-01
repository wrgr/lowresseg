# lowresseg

Low-resolution (~1 µm isotropic) neuron instance segmentation for large EM volumes (MICrONS minnie65 and compatible).

## Overview

A 3D UNet (MONAI backbone) predicts short-range voxel affinities, which are agglomerated into neuron instances. The full pipeline runs end-to-end from raw EM to interactive neuroglancer visualization.

```
EM volume (zarr)
      │
      ▼
  AffinityUNet  ← scripts/infer_full_volume.py
      │ 3-channel affinities
      ▼
  CC3D agglomeration → pred_seg
      │
      ▼
  size filter    → pred_seg_filtered
      │
      ▼
  skeletons + meshes  ← scripts/postprocess.py
      │
      ▼
  neuroglancer precomputed  ← scripts/build_precomputed.py
      │
      ▼
  HuggingFace dataset  ← scripts/upload_hf.py
      │
      ▼
  NeuroGlass study  ← scripts/update_neuroglass.py
```

## Setup

### Requirements

- Python ≥ 3.10
- CUDA GPU recommended for inference (CPU works but is slow)
- ~20 GB disk for the full minnie65 1µm volume + outputs

### Install

```bash
pip install -e ".[dev]"

# Additional runtime deps not in pyproject.toml:
pip install kimimaro zmesh connected-components-3d crackle-codec compressed_segmentation
```

### Environment variables

Create a `.env` file (gitignored) or export these in your shell:

```bash
HF_TOKEN=hf_...           # HuggingFace write token
HF_REPO=owner/repo        # HuggingFace dataset repo to publish to
NEUROGLASS_TOKEN=eyJ...   # NeuroGlass bearer token
CAVE_TOKEN=...            # CAVE/FANC connectivity token (optional)
```

Add to `~/.bashrc` for persistence across sessions:

```bash
export HF_TOKEN=hf_...
export HF_REPO=owner/repo
export NEUROGLASS_TOKEN=eyJ...
```

## End-to-end pipeline

### 1. Download the EM volume

```bash
# Small sample chunk (~50 MB, good for development):
python scripts/download_microns_chunk.py --size 256 --output data/microns_sample.zarr

# Full minnie65 1µm volume (~6 GB):
python scripts/download_full_volume.py --output data/minnie65_1um.zarr
```

### 2. Train the model

```bash
# From local zarr (offline):
python scripts/train_from_zarr.py data/microns_sample.zarr --steps 5000

# Full pipeline via CLI entry point:
lowresseg-train
```

A checkpoint is saved to `runs/<experiment>/checkpoint_final.pt`.

### 3. Run inference

```bash
# One-shot launcher (starts inference + keepalive, survives container restarts):
bash scripts/run_inference.sh

# Or run directly:
python scripts/infer_full_volume.py \
    --checkpoint runs/microns_256/checkpoint_final.pt \
    --zarr data/minnie65_1um.zarr \
    --threshold 0.5 --patch 96 --overlap 16 --min-size 100
```

Outputs written to the zarr store:
- `affinities` — 3-channel float32 affinity predictions
- `pred_seg` — raw agglomerated segments (uint64)
- `pred_seg_filtered` — segments with small objects zeroed out

Inference checkpoints progress to `data/minnie65_1um.zarr/infer_checkpoint.json` and resumes automatically on restart.

**Monitoring:**

```bash
tail -f /tmp/infer_session/infer.log
```

### 4. Postprocess (skeletons + meshes)

```bash
# All segments ≥500 voxels, top 100 largest (quick iteration):
python scripts/postprocess.py \
    --zarr data/minnie65_1um.zarr \
    --seg pred_seg_filtered \
    --min-size 500 --max-size 2000000 \
    --max-segments 100

# Full run (660 segments at default thresholds):
python scripts/postprocess.py \
    --zarr data/minnie65_1um.zarr \
    --seg pred_seg_filtered \
    --min-size 500

# Meshes only (faster, skip skeletons):
python scripts/postprocess.py --no-skeletons

# Skeletons only:
python scripts/postprocess.py --no-meshes
```

Outputs:
- `data/minnie65_1um.zarr/skeletons/<id>.swc` — one SWC file per segment
- `data/minnie65_1um.zarr/meshes/<id>.obj` — one OBJ file per segment

Postprocess resumes automatically — already-done segments are skipped.

**Tuning knobs:**

| Flag | Default | Effect |
|------|---------|--------|
| `--min-size` | 500 | Skip segments smaller than N voxels |
| `--max-size` | 2000000 | Skip likely merge errors (>N voxels) |
| `--downsample` | 2 | Downsample crops before TEASAR (2× faster, fewer skeleton nodes) |
| `--max-segments` | None | Process only the N largest (for iteration) |

### 5. Build neuroglancer precomputed

```bash
python scripts/build_precomputed.py \
    --zarr data/minnie65_1um.zarr \
    --seg pred_seg_filtered \
    --out data/minnie65_precomputed \
    --max-segments 100

# Skip pyramid rebuild if already done:
python scripts/build_precomputed.py --no-pyramid
```

Output: `data/minnie65_precomputed/` — neuroglancer precomputed directory with:
- Multiscale segmentation pyramid (MIP 0–2)
- Binary mesh fragments (legacy format)
- Binary skeletons (from existing SWC files)

**Local preview:**

```bash
python -m http.server 9191 --directory data/minnie65_precomputed
# Then in neuroglancer: add source precomputed://http://localhost:9191
```

Or use the viewer script:

```bash
python scripts/view_neuroglancer.py --zarr data/minnie65_1um.zarr
```

### 6. Upload to HuggingFace

```bash
python scripts/upload_hf.py \
    --repo wrgr2026/minnietest1 \
    --local data/minnie65_precomputed \
    --subdir seg
```

After upload, the neuroglancer source URL is:
```
precomputed://https://huggingface.co/datasets/wrgr2026/minnietest1/resolve/main/seg
```

### 7. Update NeuroGlass

```bash
python scripts/update_neuroglass.py \
    --study-id 06a43f00-3207-7c96-8000-fae320380bcf \
    --hf-repo wrgr2026/minnietest1 \
    --hf-subdir seg
```

This creates or updates a "glance" in the NeuroGlass study pointing at the HF-hosted precomputed data.

View: https://www.neuroglass.io/studies/06a43f00-3207-7c96-8000-fae320380bcf

## Iterating

The pipeline is designed for incremental iteration:

1. Run postprocess with `--max-segments 100` for a quick preview
2. Build precomputed with `--no-pyramid` to skip the slow pyramid rebuild
3. Upload and update NeuroGlass to see the result
4. Increase `--max-segments` or remove the cap for the full run

Postprocess and build_precomputed both skip already-completed work on restart.

## Key design choices

| Choice | Rationale |
|--------|-----------|
| 1 µm MIP7 | Covers full MICrONS volume at manageable size; resolves soma/dendrite topology |
| Short-range affinities (3ch) | Proven for EM; easy to extend with long-range channels |
| Balanced BCE loss | Boundaries are sparse; balancing prevents collapse to all-foreground |
| Hanning overlap blending | Suppresses tile-edge artefacts without expensive test-time augmentation |
| CC3D agglomeration | Fast, dependency-light fallback when waterz unavailable |
| Per-segment bbox crop | Avoids OOM in kimimaro/zmesh on large volumes |
| Z-slab counting | Counts segment sizes without loading the full 3.8 GB array |
| Resume via file existence | SWC/OBJ files act as implicit checkpoints; no separate state file needed |

## Extending

- **Long-range affinities**: set `out_channels: 12` in `configs/model/unet_affinity.yaml`
- **Semantic classes**: add a second head (soma / axon / dendrite / glia) with cross-entropy loss
- **Larger patches**: increase `patch_size` — adjust `channels` for GPU memory budget
- **More segments**: remove `--max-segments` cap and run overnight
- **Better skeletons**: reduce `--downsample` to 1 and tighten TEASAR `scale`/`const` params

## Scripts reference

| Script | Purpose |
|--------|---------|
| `scripts/download_microns_chunk.py` | Download small EM+seg chunk for dev/testing |
| `scripts/download_full_volume.py` | Download full minnie65 1µm EM volume |
| `scripts/train_from_zarr.py` | Train AffinityUNet from local zarr |
| `scripts/infer_full_volume.py` | Affinity inference + agglomeration on full volume |
| `scripts/run_inference.sh` | One-shot launcher with keepalive for container environments |
| `scripts/postprocess.py` | Skeletonize and mesh segments (SWC + OBJ) |
| `scripts/build_precomputed.py` | Convert zarr seg → neuroglancer precomputed format |
| `scripts/view_neuroglancer.py` | Local neuroglancer viewer for the zarr data |
| `scripts/upload_hf.py` | Upload precomputed directory to HuggingFace dataset |
| `scripts/update_neuroglass.py` | Update NeuroGlass study with new visualization glance |
