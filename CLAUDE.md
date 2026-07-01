# CLAUDE.md — session context for lowresseg

## What this repo is

End-to-end pipeline for low-resolution (~1 µm isotropic) neuron segmentation of the MICrONS minnie65 EM volume. Trains an AffinityUNet, runs full-volume inference, postprocesses into skeletons/meshes, publishes to HuggingFace, and updates a NeuroGlass study.

## Environment

Running in a remote Claude Code container. Key facts:
- Data lives in `data/minnie65_1um.zarr` (persistent disk, ~20 GB)
- Model checkpoint: `runs/microns_256/checkpoint_final.pt`
- `/tmp` is wiped on container restart; `data/` persists
- Env vars are set in `~/.bashrc` and `~/.profile` AND in `.env` (gitignored)
- Runtime deps not in pyproject.toml must be reinstalled on restart: `pip install kimimaro zmesh connected-components-3d crackle-codec compressed_segmentation`

## Credentials (in .env, gitignored)

```
HF_TOKEN=<your HuggingFace write token>
HF_REPO=wrgr2026/minnietest1
NEUROGLASS_TOKEN=<NeuroGlass bearer JWT>
CAVE_TOKEN=<CAVE connectivity token>
```

Load with: `env $(cat .env | xargs) python scripts/...`

## zarr store layout

```
data/minnie65_1um.zarr/
  em                    float32 (1664, 1408, 409)  resolution_nm=[1024,1024,1280]
  affinities            float32 (3, 1664, 1408, 409)
  pred_seg              uint64  raw agglomerated segments
  pred_seg_filtered     uint32  small objects zeroed (floor=300 vox)
  skeletons/            SWC files, one per segment id
  meshes/               OBJ files, one per segment id
  infer_checkpoint.json z-slab progress for inference resume
```

## Key pipeline commands

```bash
# Start/resume inference (keepalive included):
bash scripts/run_inference.sh

# Postprocess top-100 segments:
python scripts/postprocess.py --zarr data/minnie65_1um.zarr --seg pred_seg_filtered \
    --min-size 500 --max-segments 100

# Build precomputed (skip pyramid if already done):
python scripts/build_precomputed.py --zarr data/minnie65_1um.zarr \
    --out data/minnie65_precomputed --max-segments 100 --no-pyramid

# Upload to HF:
env $(cat .env | xargs) python scripts/upload_hf.py \
    --repo wrgr2026/minnietest1 --local data/minnie65_precomputed --subdir seg

# Update NeuroGlass:
env $(cat .env | xargs) python scripts/update_neuroglass.py
```

## NeuroGlass

- Study ID: `06a43f00-3207-7c96-8000-fae320380bcf`
- Glance ID: `06a44e82-9d8c-7cce-8000-19224f22401f`
- URL: https://www.neuroglass.io/studies/06a43f00-3207-7c96-8000-fae320380bcf
- Auth: Bearer token, `allow_redirects=False` (303 = auth failure, 200 = success)

## HuggingFace

- Repo: `wrgr2026/minnietest1`
- Precomputed at: `https://huggingface.co/datasets/wrgr2026/minnietest1/resolve/main/seg`
- Neuroglancer source: `precomputed://https://huggingface.co/datasets/wrgr2026/minnietest1/resolve/main/seg`

## Known issues / gotchas

- `build_precomputed.py` used to call `cloudvolume.datasource.precomputed.image.common.compressed_segmentation_encode` which no longer exists — fixed to use `compressed_segmentation.compress()` directly
- `zmesh.Mesher` has no `erase_buffer()` in current version — guarded with `hasattr`
- `build_precomputed.py` `write_skeletons` reads existing SWC files (fast) rather than re-running kimimaro (slow + OOM)
- Always count segment sizes via Z-slabs (`slab=32`) to avoid loading the full 3.8 GB array twice
- kimimaro TEASAR params: `scale=10, const=2000, downsample=2` — coarse but fast at 1µm

## Git branch

Development branch: `claude/low-res-segmentation-microns-8nyp2p`
