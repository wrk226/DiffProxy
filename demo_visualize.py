"""Release-grade DiffProxy artifact visualizer.

Consumes shipped stage-1 fixed proxies/uncertainty maps and a fitted SMPL-X OBJ,
then renders all artifacts in the original image coordinate system. No GT mesh,
metric, or GT-based result selection is used.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
from PIL import Image, ImageDraw

from demo_utils import parse_subject_ids, sha256_file, write_json_atomic


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")
DEFAULT_MESH_COLOR = (255, 255, 255)


def read_rgb(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def read_gray(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def save_rgb(path: str | Path, image: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB").save(destination)


def save_rgba(path: str | Path, image: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGBA").save(destination)


def save_gray(path: str | Path, image: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(image, dtype=np.uint8), mode="L").save(destination)


def alpha_composite_rgb(base_rgb: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
    """Alpha-composite an RGBA image over RGB without mutating either input."""
    if base_rgb.shape[:2] != overlay_rgba.shape[:2]:
        raise ValueError(
            f"Image size mismatch: base={base_rgb.shape[:2]}, overlay={overlay_rgba.shape[:2]}"
        )
    base = base_rgb.astype(np.float32)
    overlay = overlay_rgba[..., :3].astype(np.float32)
    alpha = overlay_rgba[..., 3:4].astype(np.float32) / 255.0
    return np.clip(overlay * alpha + base * (1.0 - alpha), 0.0, 255.0).astype(np.uint8)


def combine_uncertainty(uv_uncertainty: np.ndarray, seg_uncertainty: np.ndarray) -> np.ndarray:
    """Return the bounded arithmetic mean of two uint8/float uncertainty maps."""
    uv = np.asarray(uv_uncertainty, dtype=np.float32)
    seg = np.asarray(seg_uncertainty, dtype=np.float32)
    if uv.shape != seg.shape:
        raise ValueError(f"Uncertainty size mismatch: uv={uv.shape}, seg={seg.shape}")
    if uv.ndim != 2:
        raise ValueError(f"Expected 2D uncertainty maps, got shape {uv.shape}")
    if uv.size and max(float(uv.max()), float(seg.max())) > 1.0:
        uv = uv / 255.0
        seg = seg / 255.0
    return np.clip((uv + seg) * 0.5, 0.0, 1.0).astype(np.float32)


def load_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with Path(path).open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("v "):
                values = line.split()[1:4]
                vertices.append([float(value) for value in values])
            elif line.startswith("f "):
                indices = [int(value.split("/")[0]) - 1 for value in line.split()[1:4]]
                faces.append(indices)
    vertices_array = np.asarray(vertices, dtype=np.float32)
    faces_array = np.asarray(faces, dtype=np.int64)
    if vertices_array.ndim != 2 or vertices_array.shape[1:] != (3,):
        raise ValueError(f"Invalid or empty OBJ vertices: {path}")
    if faces_array.ndim != 2 or faces_array.shape[1:] != (3,):
        raise ValueError(f"Invalid or empty triangular OBJ faces: {path}")
    if not np.isfinite(vertices_array).all():
        raise ValueError(f"OBJ contains non-finite vertices: {path}")
    return vertices_array, faces_array


def _find_image(image_dir: Path, key: str) -> Path:
    for extension in IMAGE_EXTENSIONS:
        candidate = image_dir / f"{key}{extension}"
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing source image for {key} under {image_dir}")


def _load_camera(camera_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(camera_path) as camera:
        missing = {"K", "R", "T"} - set(camera.files)
        if missing:
            raise KeyError(f"Camera {camera_path} is missing keys: {sorted(missing)}")
        intrinsic = np.asarray(camera["K"], dtype=np.float32)
        rotation = np.asarray(camera["R"], dtype=np.float32)
        translation_m = np.asarray(camera["T"], dtype=np.float32) / 1000.0
    for name, value, shape in (
        ("K", intrinsic, (3, 3)),
        ("R", rotation, (3, 3)),
        ("T", translation_m, (3,)),
    ):
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"Invalid camera {name} in {camera_path}: shape={value.shape}")
    return intrinsic, rotation, translation_m


def camera_center_world(rotation: np.ndarray, translation_m: np.ndarray) -> np.ndarray:
    """Return the camera center for an OpenCV world-to-camera transform."""
    rotation = np.asarray(rotation, dtype=np.float32)
    translation_m = np.asarray(translation_m, dtype=np.float32)
    if rotation.shape != (3, 3) or translation_m.shape != (3,):
        raise ValueError(
            f"Expected R=(3,3), T=(3,), got R={rotation.shape}, T={translation_m.shape}"
        )
    center = -rotation.T @ translation_m
    if not np.isfinite(center).all():
        raise ValueError("Camera center is non-finite")
    return center


def render_mesh_rgba(
    vertices: np.ndarray,
    faces: np.ndarray,
    intrinsic: np.ndarray,
    rotation: np.ndarray,
    translation_m: np.ndarray,
    image_hw: tuple[int, int],
    device: str = "cuda",
    mesh_color: tuple[int, int, int] = DEFAULT_MESH_COLOR,
    mesh_alpha: float = 0.78,
) -> np.ndarray:
    """Render a world-space mesh using the release fitter's OpenCV camera convention."""
    import torch
    from pytorch3d.renderer import (
        BlendParams,
        MeshRasterizer,
        MeshRenderer,
        PointLights,
        RasterizationSettings,
        SoftPhongShader,
        TexturesVertex,
    )
    from pytorch3d.structures import Meshes
    from pytorch3d.utils import cameras_from_opencv_projection

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for mesh visualization but is unavailable")
    torch_device = torch.device(device)
    height, width = image_hw
    camera = cameras_from_opencv_projection(
        R=torch.from_numpy(rotation).to(torch_device)[None],
        tvec=torch.from_numpy(translation_m).to(torch_device)[None],
        camera_matrix=torch.from_numpy(intrinsic).to(torch_device)[None],
        image_size=torch.tensor([[height, width]], dtype=torch.float32, device=torch_device),
    )
    vertices_tensor = torch.from_numpy(vertices).to(torch_device)
    faces_tensor = torch.from_numpy(faces).to(torch_device)
    color = torch.tensor(mesh_color, dtype=torch.float32, device=torch_device) / 255.0
    textures = TexturesVertex(verts_features=color.view(1, 1, 3).repeat(1, len(vertices), 1))
    mesh = Meshes(verts=[vertices_tensor], faces=[faces_tensor], textures=textures)
    camera_center = camera_center_world(rotation, translation_m)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=camera,
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1,
                # Match the research renderer: coarse binning can overflow on full-frame SMPL-X.
                bin_size=0,
            ),
        ),
        shader=SoftPhongShader(
            device=torch_device,
            cameras=camera,
            lights=PointLights(
                device=torch_device,
                # PyTorch3D lights are world-space; put the key light at this view's camera.
                location=camera_center[None].tolist(),
                # Keep a white albedo without clipping the camera-facing planes.
                ambient_color=((0.32, 0.32, 0.32),),
                diffuse_color=((0.55, 0.55, 0.55),),
                specular_color=((0.10, 0.10, 0.10),),
            ),
            blend_params=BlendParams(background_color=(0.0, 0.0, 0.0)),
        ),
    )
    with torch.no_grad():
        rendered = renderer(mesh)[0].detach().cpu().numpy()
    rgba = np.clip(rendered * 255.0, 0.0, 255.0).astype(np.uint8)
    visible = rgba[..., 3] > 0
    rgba[..., 3] = np.where(visible, round(255 * mesh_alpha), 0).astype(np.uint8)
    return rgba


def _proxy_rgba(proxy_rgb: np.ndarray, foreground_mask: np.ndarray, alpha: float) -> np.ndarray:
    if proxy_rgb.shape[:2] != foreground_mask.shape:
        raise ValueError("Proxy and foreground mask sizes differ")
    opacity = np.where(foreground_mask, round(255 * alpha), 0).astype(np.uint8)
    return np.dstack([proxy_rgb, opacity])


def _colorize_uncertainty(joint_uncertainty: np.ndarray) -> np.ndarray:
    gray = np.clip(joint_uncertainty * 255.0, 0.0, 255.0).astype(np.uint8)
    bgr = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _uncertainty_rgba(
    colorized: np.ndarray, joint_uncertainty: np.ndarray, support: np.ndarray, alpha: float
) -> np.ndarray:
    opacity = np.where(
        support,
        np.clip((0.2 + 0.8 * joint_uncertainty) * alpha * 255.0, 0.0, 255.0),
        0.0,
    ).astype(np.uint8)
    return np.dstack([colorized, opacity])


def _labeled_tile(image: np.ndarray, label: str, tile_width: int = 300, tile_height: int = 410) -> np.ndarray:
    source = Image.fromarray(image.astype(np.uint8), mode="RGB")
    content_height = tile_height - 34
    scale = min(tile_width / source.width, content_height / source.height)
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))),
        Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (tile_width, tile_height), (22, 25, 31))
    x = (tile_width - resized.width) // 2
    y = 34 + (content_height - resized.height) // 2
    canvas.paste(resized, (x, y))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 9), label, fill=(238, 242, 249))
    return np.asarray(canvas, dtype=np.uint8)


def _sha_record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _validate_image(path: Path, expected_hw: tuple[int, int] | None = None) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"Visualization artifact is missing or empty: {path}")
    with Image.open(path) as image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise RuntimeError(f"Visualization artifact has invalid dimensions: {path}")
        if expected_hw is not None and (height, width) != expected_hw:
            raise RuntimeError(
                f"Visualization artifact has unexpected size {height}x{width}, expected {expected_hw}: {path}"
            )


def make_grid(
    images: Sequence[np.ndarray], columns: int = 2, fill_value: int = 22
) -> np.ndarray:
    """Arrange equally sized RGB images in a compact grid."""
    if not images:
        raise ValueError("make_grid requires at least one image")
    if columns <= 0:
        raise ValueError("make_grid columns must be positive")
    shape = images[0].shape
    if len(shape) != 3 or shape[2] != 3:
        raise ValueError(f"make_grid expects RGB images, got {shape}")
    if any(image.shape != shape for image in images):
        raise ValueError("make_grid images must have identical shapes")
    rows = []
    for start in range(0, len(images), columns):
        row = list(images[start : start + columns])
        while len(row) < columns:
            row.append(np.full(shape, fill_value, dtype=np.uint8))
        rows.append(np.hstack(row))
    return np.vstack(rows)


def render_subject(
    dataset_dir: Path,
    exp_root: Path,
    proxy_dir: Path,
    mesh_prefix: str,
    subject_id: str,
    output_root: Path,
    views: Sequence[int],
    device: str,
    proxy_alpha: float,
    uncertainty_alpha: float,
    mesh_alpha: float,
) -> dict[str, object]:
    gt_root = dataset_dir / "gt"
    image_dir = gt_root / "image_ori"
    camera_dir = gt_root / "camera"
    mesh_path = exp_root / f"{mesh_prefix}_smplx" / f"{subject_id}.obj"
    if not mesh_path.exists():
        raise FileNotFoundError(f"Missing fitted mesh: {mesh_path}")
    vertices, faces = load_obj(mesh_path)

    subject_root = output_root / subject_id
    input_root = subject_root / "inputs"
    proxy_root = subject_root / "proxy"
    uncertainty_root = subject_root / "uncertainty"
    mesh_root = subject_root / "mesh"
    overlay_root = subject_root / "overlays"
    panel_root = subject_root / "panels"
    for directory in (input_root, proxy_root, uncertainty_root, mesh_root, overlay_root, panel_root):
        directory.mkdir(parents=True, exist_ok=True)

    source_records: dict[str, object] = {"mesh": _sha_record(mesh_path), "views": {}}
    output_records: dict[str, object] = {}
    summary_rows: list[np.ndarray] = []

    for view in views:
        key = f"{subject_id}_{view:02d}"
        image_path = _find_image(image_dir, key)
        camera_path = camera_dir / f"{key}.npz"
        proxy_paths = {
            "seg": proxy_dir / f"seg_{key}_pred.png",
            "uv": proxy_dir / f"uv_{key}_pred.png",
            "seg_uncertainty": proxy_dir / f"seg_uncertainty_{key}_pred.png",
            "uv_uncertainty": proxy_dir / f"uv_uncertainty_{key}_pred.png",
        }
        missing = [str(path) for path in (camera_path, *proxy_paths.values()) if not path.exists()]
        if missing:
            raise FileNotFoundError("Missing demo inputs:\n" + "\n".join(missing))

        original = read_rgb(image_path)
        height, width = original.shape[:2]
        seg = read_rgb(proxy_paths["seg"])
        uv = read_rgb(proxy_paths["uv"])
        seg_uncertainty = read_gray(proxy_paths["seg_uncertainty"])
        uv_uncertainty = read_gray(proxy_paths["uv_uncertainty"])
        for name, value in (
            ("seg", seg),
            ("uv", uv),
            ("seg_uncertainty", seg_uncertainty),
            ("uv_uncertainty", uv_uncertainty),
        ):
            if value.shape[:2] != (height, width):
                raise ValueError(
                    f"{key} {name} is {value.shape[:2]}, expected original image size {(height, width)}"
                )

        foreground = np.any(seg > 3, axis=2)
        joint_uncertainty = combine_uncertainty(uv_uncertainty, seg_uncertainty)
        uncertainty_support = (uv_uncertainty > 0) | (seg_uncertainty > 0) | foreground
        uncertainty_color = _colorize_uncertainty(joint_uncertainty)
        uncertainty_map = uncertainty_color.copy()
        uncertainty_map[~uncertainty_support] = np.array([22, 25, 31], dtype=np.uint8)
        intrinsic, rotation, translation_m = _load_camera(camera_path)
        mesh_rgba = render_mesh_rgba(
            vertices,
            faces,
            intrinsic,
            rotation,
            translation_m,
            (height, width),
            device=device,
            mesh_alpha=mesh_alpha,
        )
        if not np.any(mesh_rgba[..., 3] > 0):
            raise RuntimeError(f"Mesh is not visible in view {key}; check camera convention and units")

        seg_overlay = alpha_composite_rgb(original, _proxy_rgba(seg, foreground, proxy_alpha))
        uv_overlay = alpha_composite_rgb(original, _proxy_rgba(uv, foreground, proxy_alpha))
        uncertainty_overlay = alpha_composite_rgb(
            original,
            _uncertainty_rgba(
                uncertainty_color, joint_uncertainty, uncertainty_support, uncertainty_alpha
            ),
        )
        mesh_overlay = alpha_composite_rgb(original, mesh_rgba)

        output_paths = {
            "input": input_root / f"{view:02d}.png",
            "proxy_seg": proxy_root / f"seg_{view:02d}.png",
            "proxy_uv": proxy_root / f"uv_{view:02d}.png",
            "uncertainty_seg": uncertainty_root / f"seg_{view:02d}.png",
            "uncertainty_uv": uncertainty_root / f"uv_{view:02d}.png",
            "uncertainty_joint": uncertainty_root / f"joint_{view:02d}.png",
            "mesh_rgba": mesh_root / f"mesh_{view:02d}.png",
            "proxy_seg_overlay": overlay_root / f"seg_{view:02d}.png",
            "proxy_uv_overlay": overlay_root / f"uv_{view:02d}.png",
            "uncertainty_overlay": overlay_root / f"uncertainty_{view:02d}.png",
            "mesh_overlay": overlay_root / f"mesh_{view:02d}.png",
        }
        save_rgb(output_paths["input"], original)
        save_rgb(output_paths["proxy_seg"], seg)
        save_rgb(output_paths["proxy_uv"], uv)
        save_gray(output_paths["uncertainty_seg"], seg_uncertainty)
        save_gray(output_paths["uncertainty_uv"], uv_uncertainty)
        save_rgb(output_paths["uncertainty_joint"], uncertainty_map)
        save_rgba(output_paths["mesh_rgba"], mesh_rgba)
        save_rgb(output_paths["proxy_seg_overlay"], seg_overlay)
        save_rgb(output_paths["proxy_uv_overlay"], uv_overlay)
        save_rgb(output_paths["uncertainty_overlay"], uncertainty_overlay)
        save_rgb(output_paths["mesh_overlay"], mesh_overlay)
        for path in output_paths.values():
            _validate_image(path, (height, width))

        panel = np.hstack(
            [
                _labeled_tile(original, f"{key} | input"),
                _labeled_tile(seg_overlay, "seg proxy overlay"),
                _labeled_tile(uv_overlay, "UV proxy overlay"),
                _labeled_tile(uncertainty_map, "joint uncertainty | blue low, red high"),
                _labeled_tile(mesh_overlay, "fitted mesh overlay"),
            ]
        )
        panel_path = panel_root / f"{view:02d}.png"
        save_rgb(panel_path, panel)
        _validate_image(panel_path)
        summary_rows.append(panel)

        source_records["views"][f"{view:02d}"] = {
            "image": _sha_record(image_path),
            "camera": _sha_record(camera_path),
            **{name: _sha_record(path) for name, path in proxy_paths.items()},
        }
        output_records[f"{view:02d}"] = {
            name: str(path.relative_to(output_root)) for name, path in output_paths.items()
        }
        output_records[f"{view:02d}"]["panel"] = str(panel_path.relative_to(output_root))
        output_records[f"{view:02d}"]["uncertainty_range"] = [
            float(joint_uncertainty.min()),
            float(joint_uncertainty.max()),
        ]
        output_records[f"{view:02d}"]["visible_mesh_pixels"] = int(
            np.count_nonzero(mesh_rgba[..., 3])
        )

    summary = np.vstack(summary_rows)
    summary_path = subject_root / "summary.png"
    save_rgb(summary_path, summary)
    _validate_image(summary_path)
    manifest = {
        "schema_version": 1,
        "subject": subject_id,
        "views": [int(view) for view in views],
        "sources": source_records,
        "outputs": output_records,
        "summary": str(summary_path.relative_to(output_root)),
        "mesh_vertices": int(vertices.shape[0]),
        "mesh_faces": int(faces.shape[0]),
    }
    write_json_atomic(subject_root / "manifest.json", manifest)
    return manifest


def render_demo(
    base_data_dir: str | Path,
    dataset: str,
    exp: str,
    subjects: Sequence[int | str] | str,
    tts_tag: str,
    mesh_prefix: str,
    output_dir: str | Path,
    ours_suffix: str = "_4v",
    views: Sequence[int] = (0, 1, 2, 3),
    device: str = "cuda",
    proxy_alpha: float = 0.58,
    uncertainty_alpha: float = 0.72,
    mesh_alpha: float = 0.78,
    overwrite: bool = False,
) -> dict[str, object]:
    subject_ids = parse_subject_ids(subjects)
    if not subject_ids:
        raise ValueError("At least one subject is required")
    base_data_path = Path(base_data_dir).expanduser().resolve()
    dataset_dir = base_data_path / dataset
    exp_root = dataset_dir / f"ours{ours_suffix}" / exp
    proxy_dir = exp_root / f"tempt_tts{tts_tag}_fixed"
    output_root = Path(output_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")
    if not proxy_dir.is_dir():
        raise FileNotFoundError(f"Fixed proxy directory does not exist: {proxy_dir}")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Demo output already exists: {output_root}. Use --overwrite or a new --output-dir."
            )
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    subject_manifests = []
    summaries = []
    for subject_id in subject_ids:
        print(f"[demo] rendering subject {subject_id}")
        manifest = render_subject(
            dataset_dir=dataset_dir,
            exp_root=exp_root,
            proxy_dir=proxy_dir,
            mesh_prefix=mesh_prefix,
            subject_id=subject_id,
            output_root=output_root,
            views=views,
            device=device,
            proxy_alpha=proxy_alpha,
            uncertainty_alpha=uncertainty_alpha,
            mesh_alpha=mesh_alpha,
        )
        subject_manifests.append(manifest)
        summaries.append(read_rgb(output_root / manifest["summary"]))

    index_image = make_grid(summaries, columns=2 if len(summaries) > 1 else 1)
    index_path = output_root / "index.png"
    save_rgb(index_path, index_image)
    _validate_image(index_path)
    top_manifest = {
        "schema_version": 1,
        "created_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pipeline": "DiffProxy release demo visualization",
        "selection_policy": "explicit subject IDs and body views; no GT-based routing or view selection",
        "config": {
            "base_data_dir": str(base_data_path),
            "dataset": dataset,
            "exp": exp,
            "ours_suffix": ours_suffix,
            "subjects": subject_ids,
            "views": [int(view) for view in views],
            "tts_tag": tts_tag,
            "mesh_prefix": mesh_prefix,
            "device": device,
            "proxy_alpha": proxy_alpha,
            "uncertainty_alpha": uncertainty_alpha,
            "mesh_alpha": mesh_alpha,
            "mesh_color": list(DEFAULT_MESH_COLOR),
            "mesh_light": "point_at_camera_center",
        },
        "code": {
            "demo_visualize.py": sha256_file(Path(__file__)),
            "demo_utils.py": sha256_file(Path(__file__).with_name("demo_utils.py")),
        },
        "subjects": [manifest["subject"] for manifest in subject_manifests],
        "index": str(index_path.relative_to(output_root)),
        "subject_manifests": [f"{manifest['subject']}/manifest.json" for manifest in subject_manifests],
    }
    write_json_atomic(output_root / "manifest.json", top_manifest)
    return top_manifest


def _parse_views(value: str) -> list[int]:
    views = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not views or any(view < 0 for view in views):
        raise argparse.ArgumentTypeError("--views must contain non-negative comma-separated integers")
    return views


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render DiffProxy proxies, uncertainty, and fitted mesh onto original images."
    )
    parser.add_argument("--base-data-dir", required=True, help="directory containing <dataset>/")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--exp", default="sd21-ep30")
    parser.add_argument("--ours-suffix", default="_4v")
    parser.add_argument("--subjects", default="0000,0001,0003,0004")
    parser.add_argument("--views", type=_parse_views, default=[0, 1, 2, 3])
    parser.add_argument("--tts-tag", default="03_v00")
    parser.add_argument("--mesh-prefix", default="pred-tts03_v00")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--proxy-alpha", type=float, default=0.58)
    parser.add_argument("--uncertainty-alpha", type=float, default=0.72)
    parser.add_argument("--mesh-alpha", type=float, default=0.78)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    render_demo(
        base_data_dir=args.base_data_dir,
        dataset=args.dataset,
        exp=args.exp,
        subjects=args.subjects,
        tts_tag=args.tts_tag,
        mesh_prefix=args.mesh_prefix,
        output_dir=args.output_dir,
        ours_suffix=args.ours_suffix,
        views=args.views,
        device=args.device,
        proxy_alpha=args.proxy_alpha,
        uncertainty_alpha=args.uncertainty_alpha,
        mesh_alpha=args.mesh_alpha,
        overwrite=args.overwrite,
    )
    print(f"[demo] complete: {Path(args.output_dir).resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
