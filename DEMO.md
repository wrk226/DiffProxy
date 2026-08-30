# DiffProxy one-command demo

`demo.py` exports Stage-1 and Stage-2 artifacts in one consistent, release-friendly layout:

- raw segmentation and UV proxies;
- raw segmentation/UV uncertainty and a joint color heatmap;
- full-resolution fitted-mesh RGBA;
- white fitted-mesh material with a per-view key light placed at the camera center;
- segmentation, UV, uncertainty, and mesh overlays in original image coordinates;
- per-view panels, per-subject summaries, a global `index.png`, and SHA-256 manifests.

The visualizer uses the same OpenCV camera convention as `fit_smplx.py`: OBJ vertices are in metres,
while camera translations are converted from millimetres to metres. It does not load GT meshes, GT
joints, or evaluation scores, and it never selects a view or result using GT.

## Fast render from existing artifacts

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python demo.py \
  --base-data-dir /path/to/evaluate \
  --dataset data_example_4v \
  --subjects 0000,0003 \
  --tts-tag 03_v00 \
  --mesh-prefix pred-tts03_v00 \
  --output-dir demo_outputs/example_existing
```

## Full image-to-visualization pipeline

This runs Stage 1 and Stage 2 only for the explicitly listed subjects, then renders all artifacts:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python demo.py \
  --base-data-dir /path/to/evaluate \
  --dataset data_example_4v \
  --subjects 0000,0003 \
  --run-pipeline \
  --output-dir demo_outputs/example
```

The full pipeline uses deterministic Stage-2 seed `123`, residual hand-view filtering, DPoser-X hand
multiplier `0.5`, and complete Hands-A/C refinement by default. Override these with CLI options when
needed. Stage 1 and Stage 2 also expose `--subjects` directly for arbitrary non-contiguous IDs.

Run `python demo.py --help` to see all options. `--dataset-root` is normally inferred as the parent of
`--base-data-dir`; pass it explicitly if the directory containing `evaluate/` has a different layout.

## Output contract

```text
demo_outputs/<name>/
├── manifest.json                  # configuration, code SHA-256, execution mode
├── index.png                      # all selected subjects
└── <subject>/
    ├── manifest.json              # source artifact SHA-256 and output paths
    ├── inputs/
    ├── proxy/                     # raw segmentation and UV proxies
    ├── uncertainty/               # raw segmentation/UV + joint heatmaps
    ├── mesh/                      # transparent full-frame mesh renders
    ├── overlays/                  # artifacts composited over original images
    ├── panels/                    # one comparison panel per camera
    └── summary.png
```

Existing Stage-1 and Stage-2 artifacts are never overwritten by the visualizer. Demo outputs go only
under the explicit `--output-dir`; an existing demo directory is rejected unless
`--overwrite` is provided.

## Subject selection

Always pass `--subjects` explicitly. The demo never selects examples or views using ground truth.

## Rendering provenance

The release retains only the GT-free rendering behavior needed to place proxy, uncertainty, and
SMPL-X outputs back into original image coordinates. Mesh rasterization uses the same
PyTorch3D/OpenCV camera convention as Stage 2, including conversion of camera translation from
millimetres to metres. Research-only evaluators, competing-method branches, local paths, and
notification hooks are not included.

Every top-level manifest records SHA-256 hashes for `demo.py`, `demo_visualize.py`, `demo_utils.py`,
`infer_proxy.py`, and `fit_smplx.py`. Subject manifests record the exact source image, camera, proxy,
uncertainty, and mesh hashes used for each exported visualization.
