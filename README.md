# lowresseg

Low-resolution (~1 µm isotropic) neuron instance segmentation for large EM volumes (MICrONS and compatible).

## Approach

**Affinity prediction + waterz agglomeration**

A 3D UNet (MONAI backbone) predicts short-range voxel affinities (±1 voxel in x/y/z = 3 channels). Affinities are agglomerated into neuron instances using [waterz](https://github.com/funkey/waterz) (mean-affinity watershed). Post-processing falls back to connected components when waterz is unavailable.

At 1 µm the MICrONS minnie65 volume is ~1400 × 870 × 840 voxels — small enough to hold in RAM and process in a single tiled inference pass (~128³ patches with Hanning overlap-blending).

## Pipeline

```
EM (CloudVolume / zarr)
      │
      ▼
  3D UNet (AffinityUNet)
      │ (3, X, Y, Z) affinities
      ▼
  waterz agglomeration
      │ (X, Y, Z) uint64 seg
      ▼
  evaluation (Rand / VI)
```

## Quick start

### Install

```bash
pip install -e ".[dev]"
```

### Download a sample chunk from MICrONS

Requires Google Cloud credentials (the bucket is public, anonymous access works):

```bash
python scripts/download_microns_chunk.py --size 256 --output data/microns_sample.zarr
```

### Train offline from local zarr

```bash
python scripts/train_from_zarr.py data/microns_sample.zarr --steps 5000
```

### Train from CloudVolume (full pipeline)

```bash
lowresseg-train
```

Config is managed by [Hydra](https://hydra.cc/). Override any field on the CLI:

```bash
lowresseg-train train.batch_size=4 train.max_steps=100000
```

### Predict affinities (whole volume)

```bash
lowresseg-predict checkpoint=runs/20240101_120000/checkpoint_0200000.pt output_zarr=affs.zarr
```

### Agglomerate → segmentation

```bash
lowresseg-segment affs.zarr --output seg.zarr --method waterz --threshold 0.5
```

### Evaluate

```bash
lowresseg-eval seg.zarr gt.zarr
```

## Key design choices

| Choice | Rationale |
|---|---|
| 1 µm MIP7 | Covers full MICrONS volume at manageable size; resolves soma/dendrite topology |
| Short-range affinities (3ch) | Proven for EM; easy to extend with long-range channels |
| Balanced BCE loss | Boundaries are sparse; balancing prevents collapse to all-foreground |
| Hanning overlap blending | Suppresses tile-edge artefacts without expensive test-time augmentation |
| waterz agglomeration | Deterministic, fast, tunable via single threshold |

## Extending

- **Long-range affinities**: set `out_channels: 12` in `configs/model/unet_affinity.yaml` and pass `_OFFSETS_LONG` to `seg_to_affinities`.
- **Semantic classes**: add a second head (soma / axon / dendrite / glia) with cross-entropy loss.
- **Larger patches**: increase `patch_size` — the UNet depth is set so a 128³ patch fits on 24 GB; adjust `channels` for smaller GPUs.
