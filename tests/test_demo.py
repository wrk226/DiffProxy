import tempfile
import os
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

from demo_utils import filter_subject_items, parse_subject_ids
from demo_visualize import (
    DEFAULT_MESH_COLOR,
    alpha_composite_rgb,
    camera_center_world,
    combine_uncertainty,
    load_obj,
    make_grid,
)


class SubjectFilterTests(unittest.TestCase):
    def test_parse_subject_ids_normalizes_and_deduplicates(self):
        self.assertEqual(parse_subject_ids("3,0001,3,0"), ["0003", "0001", "0000"])

    def test_filter_subject_items_preserves_requested_order(self):
        items = [("0000", ["a"]), ("0001", ["b"]), ("0003", ["c"])]
        self.assertEqual(
            filter_subject_items(items, ["0003", "0000"]),
            [("0003", ["c"]), ("0000", ["a"])],
        )

    def test_filter_subject_items_rejects_missing_subject(self):
        with self.assertRaisesRegex(ValueError, "0009"):
            filter_subject_items([("0000", [])], ["0000", "0009"])


class VisualizationMathTests(unittest.TestCase):
    def test_stage2_non_visual_import_does_not_require_open3d(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules['open3d'] = None; import test_utils; assert test_utils.o3d is None",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_stage1_import_does_not_require_debug_visual_packages(self):
        root = Path(__file__).resolve().parents[1]
        env = dict(os.environ, PYTHONPATH=str(root), PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; sys.modules.update({name: None for name in ('open3d', 'torchviz', 'viztracer')}); import test_utils; import mvadapter.systems.mvadapter_image_sd",
            ],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_alpha_composite_rgb(self):
        base = np.zeros((1, 1, 3), dtype=np.uint8)
        overlay = np.array([[[200, 100, 0, 128]]], dtype=np.uint8)
        result = alpha_composite_rgb(base, overlay)
        np.testing.assert_allclose(result[0, 0], [100, 50, 0], atol=1)

    def test_combine_uncertainty_is_bounded_mean(self):
        uv = np.array([[0, 255]], dtype=np.uint8)
        seg = np.array([[255, 255]], dtype=np.uint8)
        joint = combine_uncertainty(uv, seg)
        np.testing.assert_allclose(joint, [[0.5, 1.0]], atol=1e-6)
        self.assertEqual(joint.dtype, np.float32)

    def test_load_obj_parses_vertices_and_zero_based_faces(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mesh.obj"
            path.write_text(
                "v 0 0 1\nv 1 0 1\nv 0 1 1\nf 1/1 2/2 3/3\n",
                encoding="utf-8",
            )
            vertices, faces = load_obj(path)
        self.assertEqual(vertices.shape, (3, 3))
        self.assertEqual(faces.tolist(), [[0, 1, 2]])

    def test_make_grid_uses_fill_tile_for_incomplete_row(self):
        images = [np.full((2, 3, 3), value, dtype=np.uint8) for value in (1, 2, 3)]
        result = make_grid(images, columns=2, fill_value=7)
        self.assertEqual(result.shape, (4, 6, 3))
        self.assertTrue(np.all(result[2:, 3:] == 7))

    def test_camera_center_is_derived_from_world_to_camera_rt(self):
        rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        translation = np.array([1.0, 2.0, 3.0])
        np.testing.assert_allclose(
            camera_center_world(rotation, translation), np.array([-2.0, 1.0, -3.0])
        )

    def test_default_mesh_material_is_white(self):
        self.assertEqual(DEFAULT_MESH_COLOR, (255, 255, 255))


if __name__ == "__main__":
    unittest.main()
