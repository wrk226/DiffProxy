"""One-command DiffProxy release demo.

Examples:
  # Fast render from existing stage artifacts
  python demo.py --base-data-dir /data/evaluate --dataset data_example \
      --subjects 0000,0001 --output-dir demo_outputs/example

  # Full image -> proxy/uncertainty -> fit -> visualization pipeline
  python demo.py --base-data-dir /data/evaluate --dataset data_example \
      --subjects 0000,0001 --run-pipeline --output-dir demo_outputs/example
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable

from demo_utils import parse_subject_ids, sha256_file, write_json_atomic
from demo_visualize import render_demo


PROJECT_ROOT = Path(__file__).resolve().parent


def _tts_count(tts_tag: str) -> int:
    if not tts_tag.endswith("_v00") or len(tts_tag) < 2:
        raise ValueError(
            f"Full-pipeline demo currently requires a v00 sampling tag, got {tts_tag!r}"
        )
    try:
        return int(tts_tag.split("_", 1)[0])
    except ValueError as exc:
        raise ValueError(f"Invalid --tts-tag: {tts_tag!r}") from exc


def _run(command: list[str], env: dict[str, str]) -> dict[str, object]:
    printable = shlex.join(command)
    print(f"[demo] running: {printable}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    record = {"command": command, "display": printable, "returncode": completed.returncode}
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run selected DiffProxy stages and export original-image proxy, uncertainty, "
            "mesh overlays, per-view panels, and traceable manifests."
        )
    )
    parser.add_argument("--base-data-dir", required=True, help="directory containing <dataset>/")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", default=None, help="parent of evaluate/ for fit_smplx.py")
    parser.add_argument("--exp", default="sd21-ep30")
    parser.add_argument("--ours-suffix", default="_4v")
    parser.add_argument(
        "--subjects",
        required=True,
        help="explicit comma-separated subject IDs, e.g. 0000,0003",
    )
    parser.add_argument("--views", default="0,1,2,3")
    parser.add_argument("--tts-tag", default="03_v00")
    parser.add_argument("--output-prefix", default="pred", help="fit_smplx output prefix")
    parser.add_argument(
        "--mesh-prefix",
        default=None,
        help="effective mesh prefix; default <output-prefix>-tts<tts-tag>",
    )
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--gpu", default="0", help="physical CUDA ID exposed to both stages")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--dposer-hand-mult", type=float, default=0.5)
    parser.add_argument("--run-stage1", action="store_true")
    parser.add_argument("--run-stage2", action="store_true")
    parser.add_argument("--run-pipeline", action="store_true", help="run stage 1 and stage 2")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    subject_ids = parse_subject_ids(args.subjects)
    if not subject_ids:
        raise ValueError("--subjects cannot be empty")
    views = [int(token.strip()) for token in args.views.split(",") if token.strip()]
    if not views or any(view < 0 for view in views):
        raise ValueError("--views must contain non-negative comma-separated integers")

    base_data_dir = Path(args.base_data_dir).expanduser().resolve()
    dataset_dir = base_data_dir / args.dataset
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    dataset_root = (
        Path(args.dataset_root).expanduser().resolve()
        if args.dataset_root
        else base_data_dir.parent
    )
    mesh_prefix = args.mesh_prefix or f"{args.output_prefix}-tts{args.tts_tag}"
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else PROJECT_ROOT / "demo_outputs" / args.dataset / args.exp / mesh_prefix
    )
    run_stage1 = args.run_stage1 or args.run_pipeline
    run_stage2 = args.run_stage2 or args.run_pipeline
    subject_csv = ",".join(subject_ids)

    environment = os.environ.copy()
    environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    # demo_visualize imports torch lazily, so pinning here also covers render-only mode.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    commands: list[dict[str, object]] = []
    if run_stage1:
        commands.append(
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "infer_proxy.py"),
                    "--base_data_dir",
                    str(base_data_dir),
                    "--dataset",
                    args.dataset,
                    "--exp",
                    args.exp,
                    "--tts_list",
                    str(_tts_count(args.tts_tag)),
                    "--body_views",
                    str(len(views)),
                    "--subjects",
                    subject_csv,
                ],
                environment,
            )
        )
    if run_stage2:
        commands.append(
            _run(
                [
                    sys.executable,
                    str(PROJECT_ROOT / "fit_smplx.py"),
                    "--dataset_root",
                    str(dataset_root),
                    "--dataset",
                    args.dataset,
                    "--exp",
                    args.exp,
                    "--ours_suffix",
                    args.ours_suffix,
                    "--output_prefix",
                    args.output_prefix,
                    "--tts_count",
                    args.tts_tag,
                    "--body_views",
                    str(len(views)),
                    "--dposer_hand_mult",
                    str(args.dposer_hand_mult),
                    "--seed",
                    str(args.seed),
                    "--subjects",
                    subject_csv,
                    "--gpu",
                    "0",
                ],
                environment,
            )
        )

    manifest = render_demo(
        base_data_dir=base_data_dir,
        dataset=args.dataset,
        exp=args.exp,
        subjects=subject_ids,
        tts_tag=args.tts_tag,
        mesh_prefix=mesh_prefix,
        output_dir=output_dir,
        ours_suffix=args.ours_suffix,
        views=views,
        device=args.device,
        overwrite=args.overwrite,
    )
    manifest["execution"] = {
        "mode": (
            "full_pipeline"
            if run_stage1 and run_stage2
            else "stage1_then_render"
            if run_stage1
            else "stage2_then_render"
            if run_stage2
            else "render_existing"
        ),
        "commands": commands,
        "seed": args.seed,
        "dposer_hand_mult": args.dposer_hand_mult,
        "cuda_visible_devices": str(args.gpu),
    }
    manifest["code"].update(
        {
            "demo.py": sha256_file(Path(__file__)),
            "infer_proxy.py": sha256_file(PROJECT_ROOT / "infer_proxy.py"),
            "fit_smplx.py": sha256_file(PROJECT_ROOT / "fit_smplx.py"),
        }
    )
    write_json_atomic(output_dir / "manifest.json", manifest)
    print(f"[demo] index: {output_dir / 'index.png'}")
    print(f"[demo] manifest: {output_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
