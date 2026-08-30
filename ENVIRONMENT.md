# DiffProxy release environment

## Files

- `requirements.txt`: minimum non-interactive runtime for render-only, Stage-1 MV-Adapter inference, and Stage-2 SMPL-X fitting.
- `requirements-optional.txt`: optional Open3D support for interactive fitting visualization.

The main file intentionally does **not** include legacy debug/acceleration packages such as Open3D, TorchViz, VizTracer, Apex, BitsAndBytes, Tiny-CUDA-NN, or nvdiffrast. Open3D remains available as a separate optional extra; the TorchViz/VizTracer imports were unused debug imports and were removed; the remaining packages belong to optional research branches that the release entrypoints do not import.

`scikit-image` and `scipy` are retained because `fit_smplx.py` imports `dense_body.uv_map_generator`, which imports both packages. `chumpy` is retained for the external `human_body_prior` dependency.

## Fresh environment

The validated Linux/WSL baseline is Python 3.10.20, PyTorch 2.5.1 + CUDA 12.1, torchvision 0.20.1 + CUDA 12.1, and the matching PyTorch3D wheel pinned in `requirements.txt`.

```bash
conda create -n diffproxy python=3.10.20 pip -y
conda activate diffproxy
python -m pip install --upgrade --no-build-isolation -r requirements.txt
```

`--no-build-isolation` is required for the legacy `chumpy==0.70` build, whose setup script imports `pip` while building. It is not a CUDA workaround.

For optional interactive visualization:

```bash
python -m pip install -r requirements-optional.txt
```

The fitting entrypoint expects third-party source trees under `dependents/` (`DPoser-X` and, for VPoser, `human_body_prior`). They are not redistributed; obtain them from the upstream projects listed in the README. SMPL-X models, DiffProxy weights, DPoser checkpoints, datasets, and caches are likewise external assets.

## Smoke checks

```bash
python -m demo --help
PYTHONDONTWRITEBYTECODE=1 python -m unittest tests.test_demo -v
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 python demo.py \
  --base-data-dir /path/to/evaluate \
  --dataset data_example_4v \
  --subjects 0000 --views 0 \
  --output-dir demo_outputs/environment_smoke
```

The last command is render-only when the existing Stage-1 proxy and Stage-2 mesh artifacts are present. `--run-pipeline` additionally requires the external diffusion and body-model assets described in `DEMO.md`.
