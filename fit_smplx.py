import os

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import numpy as np
import imageio
from tqdm import tqdm
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    OrthographicCameras,
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    SoftPhongShader,
    TexturesVertex,
    PointLights,
    BlendParams,
    look_at_view_transform,
)
from pytorch3d.ops import interpolate_face_attributes
import pickle
import smplx
import torch.optim as optim
import time
import cv2
from collections import defaultdict
from dense_body.uv_map_generator import UV_Map_Generator
from dense_body.render_body_p3d_densepose_parallel_demo import group_faces_and_normalize_uvs
from dense_body.model.smplmodel.joint_names import SMPLH_JOINT_NAMES
try:
    import open3d as o3d  # 仅在交互可视化路径上使用；缺包时降级为 None
except ImportError:
    o3d = None
import matplotlib.pyplot as plt
import math  # 用于可视化网格计算
import json
import test_utils
from pytorch3d.transforms import (
    axis_angle_to_matrix,
    matrix_to_axis_angle,
    rotation_6d_to_matrix,
    matrix_to_rotation_6d,
)
import glob
import sys
from pytorch3d.utils import cameras_from_opencv_projection
from dataclasses import dataclass, field
from typing import Tuple, Dict, List, Optional
import re
import itertools
import copy
import gc
import random
import traceback
from demo_utils import parse_subject_ids

import torch.nn.functional as F
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# DPoser-X 根目录

# 路径变量（将在 __main__ 块中设置实际值）
DPOSER_ROOT = None
HBP_SRC_PATH = None

# 数据集相关路径（将在 __main__ 块中设置）
DATASET_ROOT = None  # 数据集根目录（包含 evaluate 等子目录）
UV_TEMPLATE_DIR = None  # UV 模板文件目录（包含 obj 文件）
CACHE_DIR = None  # 缓存目录

# DPoser-X 缓存目录（用于加速 checkpoint 加载）
DPOSER_CACHE_DIR = None  # 若设置，会将 DPoser-X 的 ckpt 和 data 复制到此目录

# SMPLX 模型缓存目录（用于加速模型加载）
SMPLX_CACHE_DIR = None  # 若设置，会将 SMPLX 模型复制到此目录

# DPoser 和 VPoser 模块（延迟导入，由 _setup_module_paths 设置路径后导入）
WholeBodyDPoser = None
HandDPoser = None
load_model = None
VPoser = None

def _setup_module_paths():
    """设置模块搜索路径并导入依赖模块，由 main 调用"""
    global DPOSER_ROOT, HBP_SRC_PATH
    global WholeBodyDPoser, HandDPoser, load_model, VPoser
    
    if DPOSER_ROOT and os.path.isdir(DPOSER_ROOT) and DPOSER_ROOT not in sys.path:
        sys.path.insert(0, DPOSER_ROOT)
    if HBP_SRC_PATH and os.path.isdir(HBP_SRC_PATH) and HBP_SRC_PATH not in sys.path:
        sys.path.insert(0, HBP_SRC_PATH)
    
    # DPoser-X whole-body diffusion prior
    from run.tester.wholebody.smplify import DPoser as _WholeBodyDPoser
    from run.tester.hand.smplify import DPoser as _HandDPoser
    WholeBodyDPoser = _WholeBodyDPoser
    HandDPoser = _HandDPoser
    
    # VPoser 相关
    from human_body_prior.tools.model_loader import load_model as _load_model
    from human_body_prior.models.vposer_model import VPoser as _VPoser
    load_model = _load_model
    VPoser = _VPoser


def _cache_dposer_files():
    """
    将 DPoser-X 的 checkpoint 和 data 文件缓存到本地快速存储。
    如果缓存已存在则跳过，否则从 DPOSER_ROOT 复制到 DPOSER_CACHE_DIR。
    """
    import shutil
    
    if DPOSER_CACHE_DIR is None or DPOSER_ROOT is None:
        return
    
    if not os.path.isdir(DPOSER_ROOT):
        print(f"[Cache] DPoser-X 源目录不存在: {DPOSER_ROOT}，跳过缓存")
        return
    
    os.makedirs(DPOSER_CACHE_DIR, exist_ok=True)
    
    # 需要缓存的文件/目录列表（相对于 DPOSER_ROOT）
    # DPoser-X 内部使用相对路径加载，需要缓存完整的目录结构
    files_to_cache = [
        # pretrained_models 整个目录（包含 body, hand, face, wholebody 等）
        "pretrained_models",
        # data 目录（body/hand normalizer 等）
        "data",
        # configs 目录（DPoser 初始化需要）
        "configs",
    ]
    
    for rel_path in files_to_cache:
        src = os.path.join(DPOSER_ROOT, rel_path)
        dst = os.path.join(DPOSER_CACHE_DIR, rel_path)
        
        if os.path.exists(dst):
            continue  # 已缓存，跳过
        
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"[Cache] 缓存文件: {rel_path}")
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            print(f"[Cache] 缓存目录: {rel_path}")
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            print(f"[Cache] 警告：源不存在 {src}")
    
    print(f"[Cache] DPoser-X 缓存完成: {DPOSER_CACHE_DIR}")


def get_dposer_effective_root():
    """返回 DPoser-X 的有效根目录（优先使用缓存目录）"""
    if DPOSER_CACHE_DIR and os.path.isdir(DPOSER_CACHE_DIR):
        # 检查缓存是否有效（至少有一个 ckpt 文件）
        ckpt_check = os.path.join(DPOSER_CACHE_DIR, "pretrained_models", "wholebody", "mixed", "last.ckpt")
        if os.path.isfile(ckpt_check):
            return DPOSER_CACHE_DIR
    return DPOSER_ROOT


def _cache_smplx_files():
    """
    将 SMPLX 模型文件缓存到本地快速存储。
    如果缓存已存在则跳过，否则从 MODEL_FOLDER 复制到 SMPLX_CACHE_DIR。
    """
    import shutil
    
    if SMPLX_CACHE_DIR is None or MODEL_FOLDER is None:
        return
    
    if not os.path.isdir(MODEL_FOLDER):
        print(f"[Cache] SMPLX 源目录不存在: {MODEL_FOLDER}，跳过缓存")
        return
    
    os.makedirs(SMPLX_CACHE_DIR, exist_ok=True)
    
    # 需要缓存的文件列表（相对于 MODEL_FOLDER）
    files_to_cache = [
        "smplx/SMPLX_NEUTRAL.pkl",
        "smplx/SMPLX_NEUTRAL.npz",
        "smplx/SMPLX_MALE.npz",
        "smplx/SMPLX_FEMALE.npz",
    ]
    
    for rel_path in files_to_cache:
        src = os.path.join(MODEL_FOLDER, rel_path)
        dst = os.path.join(SMPLX_CACHE_DIR, rel_path)
        
        if os.path.exists(dst):
            continue  # 已缓存，跳过
        
        if os.path.isfile(src):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f"[Cache] 缓存 SMPLX 文件: {rel_path}")
            shutil.copy2(src, dst)
        else:
            print(f"[Cache] 警告：SMPLX 文件不存在 {src}")
    
    print(f"[Cache] SMPLX 缓存完成: {SMPLX_CACHE_DIR}")


def get_smplx_effective_folder():
    """返回 SMPLX 模型的有效目录（优先使用缓存目录）"""
    if SMPLX_CACHE_DIR and os.path.isdir(SMPLX_CACHE_DIR):
        # 检查缓存是否有效
        check_file = os.path.join(SMPLX_CACHE_DIR, "smplx", "SMPLX_NEUTRAL.npz")
        if os.path.isfile(check_file):
            return SMPLX_CACHE_DIR
    return MODEL_FOLDER


# 新增基于图像的可视化开关（独立于3D可视化）






# ----------------------------------------



# --- 设置设备 ---
# device_id = 1
# if torch.cuda.is_available():
#     try:
#         device = torch.device(f"cuda:{device_id}")
#         torch.cuda.set_device(device)
#         print(f"使用设备: {torch.cuda.get_device_name(device)}")
#     except Exception as e:
#         print(f"无法设置 CUDA 设备 cuda:{device_id} ({e}), 回退到 CPU.")
#         device = torch.device("cpu")
#         print("使用设备: CPU")
# else:
#     device = torch.device("cpu")
#     print("使用设备: CPU")


def B2P(B: np.ndarray):
    B2P_R1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    B2P_R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    B2P_T = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    R = B2P_R1 @ B[:3, :3] @ B2P_R2
    T = B2P_T @ B[:3, 3] @ R
    return R, T


def render_mesh(verts, faces, cameras, image_size, device):
    """使用PyTorch3D渲染mesh，光源位于相机中心"""
    # 创建纹理
    verts_rgb = torch.ones_like(verts)[None] * torch.tensor([0.8, 0.8, 0.8], device=device)
    textures = TexturesVertex(verts_features=verts_rgb)
    
    # 创建mesh
    meshes = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )
    
    # 计算相机中心作为光源位置
    cam_center = cameras.get_camera_center()  # (N, 3)
    lights = PointLights(
        device=device,
        location=cam_center,
        ambient_color=[[0.6, 0.6, 0.6]],
        diffuse_color=[[0.6, 0.6, 0.6]],
        specular_color=[[0.0, 0.0, 0.0]],
    )
    
    # 渲染设置
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-10,
        # faces_per_pixel=1,
        # ------- 针对 overflow 警告的调节 -------
        bin_size=0,                # 0 表示启用 naive coarse rasterizer，最安全但可能稍慢
        # max_faces_per_bin=150000   # 或根据需要调大
    )
    
    # 设置blend参数（用于mask）
    blend_params = BlendParams(background_color=[0, 0, 0])
    
    # 创建渲染器
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=device, cameras=cameras, lights=lights, blend_params=blend_params)
    )
    
    # 渲染
    images = renderer(meshes)
    # 提取RGB和alpha mask
    rgb = images[0, ..., :3]  # (H, W, 3)
    mask = (images[0, ..., 3] > 1e-4).float()  # 二值mask，1为mesh像素
    
    return rgb, mask 


def save_obj(verts, faces, filename):
    """保存顶点和面到OBJ文件"""
    if torch.is_tensor(verts):
        verts = verts.detach().cpu().numpy()
    if torch.is_tensor(faces):
        faces = faces.detach().cpu().numpy()
    
    with open(filename, 'w') as f:
        # 写入顶点（确保使用高精度浮点数格式）
        for v in verts:
            f.write(f'v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n')
        # 写入面（注意OBJ索引从1开始）
        for face in faces:
            f.write(f'f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n')
    
    print(f"保存模型到 {filename}")


def save_joints_2d(joints_2d, filename):
    """保存关节点2D坐标到JSON文件"""
    if torch.is_tensor(joints_2d):
        joints_2d = joints_2d.detach().cpu().numpy().tolist()
    
    with open(filename, 'w') as f:
        json.dump({"joints": joints_2d}, f, indent=2)
    
    print(f"保存关节点坐标到 {filename}")


 

def generate_class_masks(pred_img, num_classes=12):
    """基于RGB颜色距离生成逐类掩码，不直接 .cuda()，由调用者决定设备"""
    # 如果通道在第一维，调整到最后一维
    if pred_img.shape[0] == 3:
        pred_img = pred_img.permute(1, 2, 0)
    
    # 转换为numpy进行处理
    if isinstance(pred_img, torch.Tensor):
        pred_img_np = pred_img.cpu().float().numpy()
    else:
        pred_img_np = pred_img.astype(np.float32)
    
    # 确保值在0-255范围内
    if pred_img_np.max() <= 1.0:
        pred_img_np = pred_img_np * 255.0
    
    H, W = pred_img_np.shape[:2]
    
    # 使用_SEG_PALETTE进行颜色匹配
    palette = _SEG_PALETTE.astype(np.float32)
    num_classes = min(num_classes, len(palette))
    
    # 重塑图像为(H*W, 3)进行批量计算
    img_flat = pred_img_np.reshape(-1, 3)
    
    # 计算每个像素与调色板中每种颜色的欧氏距离
    distances = np.linalg.norm(img_flat[:, None, :] - palette[None, :num_classes, :], axis=2)
    
    # 找到最近的颜色类别
    nearest_class = np.argmin(distances, axis=1)
    
    # 重塑回原图像尺寸
    class_map = nearest_class.reshape(H, W)
    
    # 生成每个类别的掩码
    class_masks = []
    for i in range(num_classes):
        mask = (class_map == i).astype(np.uint8)
        # 形态学操作去噪（是否启用由外部 flag 控制）
        # 实际启用/禁用由 _prepare_subject_views 里的 do_erode 进行，
        # 这里仅保留基础实现，具体开关逻辑在调用端处理
        class_masks.append(torch.from_numpy(mask).int())
    
    return class_masks


def generate_class_masks_subset(pred_img, allowed_ids):
    """仅在给定 allowed_ids 的颜色子集中进行最近颜色匹配，返回与全局 seg id 对齐的掩码列表。
    - 返回列表长度与 _SEG_PALETTE 相同，未被允许的 id 对应掩码为全零。
    - pred_img 支持 torch.Tensor 或 numpy，值域 [0,255] 或 [0,1]。
    """
    # 调整通道顺序至最后一维
    if isinstance(pred_img, torch.Tensor) and pred_img.shape[0] == 3:
        pred_img = pred_img.permute(1, 2, 0)

    # 转 numpy float32，值域归一
    if isinstance(pred_img, torch.Tensor):
        pred_img_np = pred_img.detach().cpu().float().numpy()
    else:
        pred_img_np = pred_img.astype(np.float32)
    if pred_img_np.max() <= 1.0:
        pred_img_np = pred_img_np * 255.0

    H, W = pred_img_np.shape[:2]
    palette = _SEG_PALETTE.astype(np.float32)

    # 规范 allowed ids
    allowed = sorted({int(i) for i in allowed_ids if 0 <= int(i) < len(palette)})
    if len(allowed) == 0:
        # 若无允许 id，返回全零掩码
        return [torch.zeros((H, W), dtype=torch.int32) for _ in range(len(palette))]

    # 在子集内做最近颜色匹配
    sub_colors = palette[np.asarray(allowed, dtype=np.int64)]  # (K,3)
    img_flat = pred_img_np.reshape(-1, 3)
    distances = np.linalg.norm(img_flat[:, None, :] - sub_colors[None, :, :], axis=2)  # (N,K)
    nearest_sub_idx = np.argmin(distances, axis=1)  # (N,)
    nearest_global_id = np.asarray(allowed, dtype=np.int64)[nearest_sub_idx]  # (N,)
    seg_map = nearest_global_id.reshape(H, W)

    # 生成对齐全局 seg id 的掩码列表
    class_masks = []
    for gid in range(len(palette)):
        if gid in allowed:
            mask = (seg_map == gid).astype(np.uint8)
            class_masks.append(torch.from_numpy(mask).int())
        else:
            class_masks.append(torch.zeros((H, W), dtype=torch.int32))
    return class_masks

def load_group_uv_info(
    obj_path: str,
    uv_size: int = 64,
    cache_dir: str = './cache',
    extra_uv_payload: bool = False,
) -> Tuple[Dict[int, Dict[str, torch.Tensor]], int]:
    """
    Build per-part UV lookup tables from a user-provided topology template.
    The caller is responsible for providing templates whose SMPL-X face/UV
    indices are compatible with the fitted model and whose license permits use.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_tag = 'hand' if extra_uv_payload else 'body'
    safe_name = os.path.splitext(os.path.basename(obj_path))[0].replace(os.sep, '_')
    cache_file = os.path.join(cache_dir, f'group_uv_info_{cache_tag}_{safe_name}_{uv_size}.pkl')
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as rf:
            return pickle.load(rf)

    body_template_path = os.path.join(UV_TEMPLATE_DIR, "diffproxy_uv_template.obj")

    if extra_uv_payload:
        filtered_obj, kept_seg_ids, kept_face_orig_indices = _create_hand_only_obj(
            hand_obj_path=obj_path,
            body_obj_path=body_template_path,
            cache_dir=cache_dir,
        )
        obj_for_generator = filtered_obj
        result = group_faces_and_normalize_uvs(filtered_obj)
    else:
        kept_seg_ids = None
        kept_face_orig_indices = None
        obj_for_generator = obj_path
        result = group_faces_and_normalize_uvs(obj_path)

    _, _, _, face_group, uvs_new_arr, faces_uvs_arr, group_bbox = result[:7]
    face_group = np.asarray(face_group)
    max_gid = int(face_group.max())

    generator = UV_Map_Generator(
        UV_height=uv_size,
        data_root=os.path.dirname(obj_for_generator) or '.',
        file_name=os.path.splitext(os.path.basename(obj_for_generator))[0]
    )
    pixel_face_full = generator.face_id  # (uv, uv) → local face idx
    pixel_bary_full = generator.bary_weights  # (uv, uv, 3)
    valid_mask = generator.valid_face_mask

    valid_face_indices = np.where(valid_mask)[0]
    pixel_face_tmp = pixel_face_full.copy()
    pixel_face_tmp[pixel_face_tmp >= 0] = valid_face_indices[pixel_face_tmp[pixel_face_tmp >= 0]]

    if extra_uv_payload:
        kept_face_orig_indices_arr = np.asarray(kept_face_orig_indices, dtype=np.int64)
        pixel_face_full_orig = pixel_face_tmp.copy()
        mask_valid_pf = pixel_face_full_orig >= 0
        pixel_face_full_orig[mask_valid_pf] = kept_face_orig_indices_arr[pixel_face_full_orig[mask_valid_pf]]
        pixel_face_full = pixel_face_full_orig
    else:
        pixel_face_full = pixel_face_tmp

    group_info: Dict[int, Dict[str, torch.Tensor]] = {}
    local_gid_to_seg_id: Dict[int, int] = {}
    local_gid_faces: Dict[int, np.ndarray] = {}

    for gid in range(1, max_gid + 1):
        bbox = group_bbox[gid - 1]
        group_faces_local = np.where((face_group == gid) & valid_mask)[0]
        if extra_uv_payload:
            group_faces = kept_face_orig_indices_arr[group_faces_local]
        else:
            group_faces = group_faces_local
        mask_face = np.isin(pixel_face_full, group_faces)
        pf = np.where(mask_face, pixel_face_full, -1)
        bw = pixel_bary_full.copy()
        bw[~mask_face] = 0
        group_info[gid] = {
            'pixel_face': torch.from_numpy(pf.astype(np.int64)),
            'pixel_bary': torch.from_numpy(bw.astype(np.float32)),
            'bbox': bbox,
        }
        if extra_uv_payload and kept_seg_ids is not None:
            seg_ids_np = np.asarray(kept_seg_ids, dtype=np.int64)
            local_face_ids = group_faces_local
            local_gid_faces[gid] = local_face_ids
            if local_face_ids.size > 0:
                segs_in_group = seg_ids_np[local_face_ids]
                binc = np.bincount(segs_in_group)
                mapped_seg = int(np.argmax(binc))  # 5 或 9
                local_gid_to_seg_id[gid] = mapped_seg

    # 将局部组映射到细分手部语义（12~17 左手；18~23 右手）
    local_gid_to_fine_seg_id: Dict[int, int] = {}
    if extra_uv_payload and kept_seg_ids is not None:
        left_locals = [gid for gid, sid in local_gid_to_seg_id.items() if sid == 5]
        right_locals = [gid for gid, sid in local_gid_to_seg_id.items() if sid == 9]
        left_locals.sort(); right_locals.sort()
        left_ids = list(LEFT_HAND_FINE_SEG_IDS)
        right_ids = list(RIGHT_HAND_FINE_SEG_IDS)
        for i, gid in enumerate(left_locals):
            local_gid_to_fine_seg_id[gid] = left_ids[i % len(left_ids)]
        for i, gid in enumerate(right_locals):
            local_gid_to_fine_seg_id[gid] = right_ids[i % len(right_ids)]

    if extra_uv_payload:
        group_info['metadata'] = {
            'local_gid_to_seg_id': local_gid_to_seg_id,
            'local_gid_to_fine_seg_id': local_gid_to_fine_seg_id,
            'kept_seg_ids': np.asarray(kept_seg_ids) if kept_seg_ids is not None else None,
            'kept_face_orig_indices': np.asarray(kept_face_orig_indices) if kept_face_orig_indices is not None else None,
            'face_group_local': face_group.copy(),
            'uvs_new': np.asarray(uvs_new_arr),
            'faces_uvs': np.asarray(faces_uvs_arr),
        }

    with open(cache_file, 'wb') as wf:
        pickle.dump((group_info, max_gid), wf)
        print(f"已缓存 {len(group_info)} 个 group UV 信息到 {cache_file}")

    return group_info, max_gid


def get_smplx_vertices(poses_param, betas_param, trans_param, gender="neutral",
                       expression_param=None, scale_param=None, smplx_model_neutral=None,
                       smplx_model_male=None, smplx_model_female=None,
                       use_hand_pca: bool = False, hand_pca_comps: int = 45):
    """
    获取SMPLX顶点坐标
    """
    expression_tensor = (expression_param.unsqueeze(0)
                         if expression_param is not None and hasattr(expression_param, 'unsqueeze')
                         else None)
    model_dict = {
        'male': smplx_model_male,
        'female': smplx_model_female,
        'neutral': smplx_model_neutral
    }
    # 计算手部切片（与 flat_poses 拼接顺序一致）
    base_hand_start = 75
    hand_dim = hand_pca_comps if use_hand_pca else 45
    l_start = base_hand_start
    l_end = base_hand_start + hand_dim
    r_start = l_end
    r_end = l_end + hand_dim

    model_out = model_dict[gender](  # type: ignore
        betas=betas_param.unsqueeze(0),
        global_orient=poses_param[:3].unsqueeze(0),
        body_pose=poses_param[3:66].unsqueeze(0),
        left_hand_pose=poses_param[l_start:l_end].unsqueeze(0),
        right_hand_pose=poses_param[r_start:r_end].unsqueeze(0),
        jaw_pose=poses_param[66:69].unsqueeze(0),
        leye_pose=poses_param[69:72].unsqueeze(0),
        reye_pose=poses_param[72:75].unsqueeze(0),
        transl=trans_param.unsqueeze(0),
        expression=expression_tensor
    )
    verts = model_out.vertices[0]
    return verts * scale_param if scale_param is not None else verts 


def get_smplx_vertices_batch(poses_flat_base, betas_param, trans_param,
                             global_orient_batch,
                             gender="neutral", scale_param=None,
                             smplx_model_neutral=None,
                             smplx_model_male=None, smplx_model_female=None,
                             use_hand_pca: bool = False, hand_pca_comps: int = 45):
    """
    Batch 版 SMPLX 前向：给定 B 组 global_orient，其余参数不变，返回 (B, V, 3)。
    global_orient_batch: (B, 3)
    """
    B = global_orient_batch.shape[0]
    model_dict = {'male': smplx_model_male, 'female': smplx_model_female, 'neutral': smplx_model_neutral}
    model = model_dict[gender]
    num_expression = model.num_expression_coeffs if hasattr(model, 'num_expression_coeffs') else 10
    expression_tensor = torch.zeros(B, num_expression, device=global_orient_batch.device, dtype=global_orient_batch.dtype)

    base_hand_start = 75
    hand_dim = hand_pca_comps if use_hand_pca else 45
    l_start, l_end = base_hand_start, base_hand_start + hand_dim
    r_start, r_end = l_end, l_end + hand_dim

    model_out = model(
        betas=betas_param.unsqueeze(0).expand(B, -1),
        global_orient=global_orient_batch,
        body_pose=poses_flat_base[3:66].unsqueeze(0).expand(B, -1),
        left_hand_pose=poses_flat_base[l_start:l_end].unsqueeze(0).expand(B, -1),
        right_hand_pose=poses_flat_base[r_start:r_end].unsqueeze(0).expand(B, -1),
        jaw_pose=poses_flat_base[66:69].unsqueeze(0).expand(B, -1),
        leye_pose=poses_flat_base[69:72].unsqueeze(0).expand(B, -1),
        reye_pose=poses_flat_base[72:75].unsqueeze(0).expand(B, -1),
        transl=trans_param.unsqueeze(0).expand(B, -1),
        expression=expression_tensor,
    )
    verts = model_out.vertices
    if scale_param is not None:
        verts = verts * scale_param
    return verts


# 模型文件夹路径（将在 __main__ 块中设置实际值）
MODEL_FOLDER = None
# **全局模型加载已移至 DenseBodyReconstructor._load_models**

# ----------------- VPoser 设置 -----------------
# (已迁移至 DenseBodyReconstructor._load_models 按 GPU 动态加载)

# ====================  语义分割可视化辅助  ====================
# 调色板：seg id -> RGB
_SEG_PALETTE = np.array([
    [0, 0, 0],              # 0: 背景
    [0, 0, 255],            # 1-11: 与旧版 body palette 对齐
    [0, 255, 0],
    [255, 0, 0],
    [255, 255, 0],
    [255, 0, 255],
    [128, 128, 0],
    [255, 255, 255],
    [0, 255, 255],
    [0, 128, 128],
    [128, 0, 128],
    [128, 128, 255],
    [64, 255, 128],         # 12-17: 左手细分语义 (1-based 13-18)
    [255, 64, 128],
    [192, 255, 128],
    [0, 0, 128],
    [0, 128, 0],
    [0, 128, 255],
    [128, 0, 0],            # 18-23: 右手细分语义 (1-based 19-24)
    [128, 0, 255],
    [128, 255, 0],
    [128, 255, 255],
    [255, 128, 0],
    [255, 128, 255],
    [128, 128, 128],       # 24: 灰色（手部视角中非手的人体）
], dtype=np.uint8)

def colorize_seg(seg: np.ndarray) -> np.ndarray:
    """(H,W) seg id -> (H,W,3) 彩色"""
    seg_clip = np.clip(seg, 0, len(_SEG_PALETTE) - 1)
    return _SEG_PALETTE[seg_clip]

def colorize_uv(uv_2ch: np.ndarray) -> np.ndarray:
    """将 (H,W,2) 的 UV (0-255) 可视化为 (H,W,3) 彩色图。
    - U 映射到 R 通道，V 映射到 G 通道；B 通道用 U、V 的均值增强对比。
    - 输入类型可为 uint8 或 float，内部都会裁剪到 [0,255]。
    """
    if uv_2ch.ndim != 3 or uv_2ch.shape[-1] < 2:
        raise ValueError("uv_2ch 需要形状为 (H,W,2) 或 (H,W,>=2)")
    uv_uint8 = uv_2ch[..., :2]
    if uv_uint8.dtype != np.uint8:
        uv_uint8 = np.clip(uv_uint8, 0, 255).astype(np.uint8)
    u = uv_uint8[..., 0]
    v = uv_uint8[..., 1]
    b = np.zeros(u.shape, dtype=np.uint8)
    vis = np.stack([u, v, b], axis=-1)
    return vis

def build_seg_id_map(mask_list):
    """根据 mask 列表构造整型 seg id 图 (torch.long)"""
    seg_map = torch.zeros_like(mask_list[0], dtype=torch.long)
    for sid, m in enumerate(mask_list):
        if sid == 0:
            continue  # 跳过背景
        seg_map[m.bool()] = sid
    return seg_map

DEFAULT_BBOX_RATIO = 1.25
HAND_FOOT_BBOX_RATIO = 1.5

# 语义分割索引（与 _SEG_PALETTE / generate_class_masks 对齐）
LEFT_HAND_FINE_SEG_IDS = (12, 13, 14, 15, 16, 17)
RIGHT_HAND_FINE_SEG_IDS = (18, 19, 20, 21, 22, 23)
LEFT_HAND_SEG_IDS = LEFT_HAND_FINE_SEG_IDS
RIGHT_HAND_SEG_IDS = RIGHT_HAND_FINE_SEG_IDS
GRAY_HAND_SEG_ID = 24
BODY_SEG_IDS = tuple(range(1, 12))

# ----------------- 关节角度约束（基于 pose_flat 的单关节单轴限制） -----------------
# 这里不再用 3D 关节点三点夹角，而是直接对 axis-angle 中每个关节每个轴的旋转范围做约束。
# pose_flat 的结构与 get_smplx_vertices 保持一致：
# [0:3]   global_orient
# [3:66]  body_pose (21*3, 关节顺序为 SMPL-X 主体关节)
# [66:69] jaw_pose
# [69:72] leye_pose
# [72:75] reye_pose
# [75:120] left_hand_pose  (15*3)
# [120:165] right_hand_pose (15*3)

# SMPL-X 主体关节名称（与 body_pose 的 21 个关节一一对应）
SMPLX_BODY_JOINT_NAMES: List[str] = [
    "left_hip",      # 0  → 索引1
    "right_hip",     # 1
    "spine1",        # 2
    "left_knee",     # 3
    "right_knee",    # 4
    "spine2",        # 5
    "left_ankle",    # 6
    "right_ankle",   # 7
    "spine3",        # 8
    "left_foot",     # 9
    "right_foot",    # 10
    "neck",          # 11
    "left_collar",   # 12
    "right_collar",  # 13
    "head",          # 14
    "left_shoulder", # 15
    "right_shoulder",# 16
    "left_elbow",    # 17
    "right_elbow",   # 18
    "left_wrist",    # 19
    "right_wrist",   # 20
]
SMPLX_BODY_JOINT_TO_IDX: Dict[str, int] = {
    name: i for i, name in enumerate(SMPLX_BODY_JOINT_NAMES)
}

HAND_JOINT_COUNT = 15  # 每只手 15 个关节 * 3 轴 = 45 维
LEFT_HAND_JOINT_NAMES: List[str] = [
    f"left_hand_{i:02d}" for i in range(HAND_JOINT_COUNT)
]
RIGHT_HAND_JOINT_NAMES: List[str] = [
    f"right_hand_{i:02d}" for i in range(HAND_JOINT_COUNT)
]


def _deg_pair(lo: float, hi: float) -> Tuple[float, float]:
    return math.radians(lo), math.radians(hi)


# JOINT_ANGLE_LIMITS: 关节名 → 每个轴 (x,y,z) 的 (min_rad, max_rad)
JOINT_ANGLE_LIMITS: Dict[str, List[Tuple[float, float]]] = {}


def _set_joint_limits(jname: str, axis_limits_deg: List[Tuple[float, float]]):
    """
    axis_limits_deg: [(min_deg, max_deg)] * 3
    """
    assert len(axis_limits_deg) == 3
    JOINT_ANGLE_LIMITS[jname] = [
        _deg_pair(lo, hi) for (lo, hi) in axis_limits_deg
    ]


# ---- 身体大关节 ----
_set_joint_limits("left_hip", [(-180.0, 90.0), (-90.0, 90.0), (-45.0, 180.0)])
_set_joint_limits("right_hip", [(-180.0, 90.0), (-90.0, 90.0), (-180.0, 45.0)])

_set_joint_limits("left_knee", [(0.0, 180.0), (-90.0, 90.0), (0.0, 0.0)])
_set_joint_limits("right_knee", [(0.0, 180.0), (-90.0, 90.0), (0.0, 0.0)])

_set_joint_limits(
    "left_shoulder", [(-90.0, 90.0), (-180.0, 90.0), (-180.0, 90.0)]
)
_set_joint_limits(
    "right_shoulder", [(-90.0, 90.0), (-90.0, 180.0), (-90.0, 180.0)]
)

_set_joint_limits(
    "left_ankle", [(-90.0, 90.0), (-90.0, 90.0), (-45.0, 45.0)]
)
_set_joint_limits(
    "right_ankle", [(-90.0, 90.0), (-90.0, 90.0), (-45.0, 45.0)]
)

_set_joint_limits("left_foot", [(-45.0, 45.0), (0.0, 0.0), (0.0, 0.0)])
_set_joint_limits("right_foot", [(-45.0, 45.0), (0.0, 0.0), (0.0, 0.0)])

_set_joint_limits(
    "left_wrist", [(-90.0, 90.0), (-30.0, 60.0), (-90.0, 90.0)]
)
_set_joint_limits(
    "right_wrist", [(-90.0, 90.0), (-60.0, 30.0), (-90.0, 90.0)]
)

_set_joint_limits("spine1", [(-90.0, 90.0), (-60.0, 60.0), (-60.0, 60.0)])
_set_joint_limits("spine2", [(-90.0, 90.0), (-60.0, 60.0), (-60.0, 60.0)])
_set_joint_limits("spine3", [(-90.0, 90.0), (-60.0, 60.0), (-60.0, 60.0)])

_set_joint_limits("left_elbow", [(-90.0, 90.0), (-180.0, 0.0), (0.0, 0.0)])
_set_joint_limits("right_elbow", [(-90.0, 90.0), (0.0, 180.0), (0.0, 0.0)])

_set_joint_limits("neck", [(-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0)])
_set_joint_limits("head", [(-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0)])

_set_joint_limits("left_collar", [(-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)])
_set_joint_limits("right_collar", [(-30.0, 30.0), (-30.0, 30.0), (-30.0, 30.0)])

# pelvis 明确“无限制”，不在 JOINT_ANGLE_LIMITS 中登记

# 下颚
_set_joint_limits("jaw", [(0.0, 30.0), (0.0, 0.0), (0.0, 0.0)])

# ---- 手指分组（左右对称）----
for side in ["left", "right"]:
    # 00,03,06,09
    for idx in (0, 3, 6, 9):
        _set_joint_limits(
            f"{side}_hand_{idx:02d}",
            [(0.0, 0.0), (-30.0, 30.0), (-90.0, 90.0)],
        )
    # 01,02,04,05,07,08,10,11
    for idx in (1, 2, 4, 5, 7, 8, 10, 11):
        _set_joint_limits(
            f"{side}_hand_{idx:02d}",
            [(0.0, 0.0), (0.0, 0.0), (-90.0, 90.0)],
        )
    # 12,13
    for idx in (12, 13):
        _set_joint_limits(
            f"{side}_hand_{idx:02d}",
            [(-45.0, 45.0), (-45.0, 45.0), (-45.0, 45.0)],
        )
    # 14
    _set_joint_limits(
        f"{side}_hand_14",
        [(0.0, 0.0), (-90.0, 90.0), (0.0, 0.0)],
    )


def compute_pose_joint_limit_loss(poses_flat: torch.Tensor) -> torch.Tensor:
    """
    基于 pose_flat（axis-angle）直接对每个关节每个轴的角度范围做限制。
    poses_flat: (..., 165+)，支持 [165] 或 [B,165]
    返回标量 loss。
    """
    if not JOINT_ANGLE_LIMITS:
        return poses_flat.new_tensor(0.0)

    if poses_flat.ndim == 1:
        poses_eval = poses_flat.unsqueeze(0)
    elif poses_flat.ndim == 2:
        poses_eval = poses_flat
    else:
        return poses_flat.new_tensor(0.0)

    B = poses_eval.shape[0]
    penalties: List[torch.Tensor] = []

    base_hand_start = 75
    l_start = base_hand_start
    r_start = base_hand_start + HAND_JOINT_COUNT * 3  # 75 + 45 = 120

    for jname, limits in JOINT_ANGLE_LIMITS.items():
        # 计算该关节在 pose_flat 中的起始索引
        if jname == "pelvis":
            base_idx = 0
        elif jname in SMPLX_BODY_JOINT_TO_IDX:
            local_idx = SMPLX_BODY_JOINT_TO_IDX[jname]  # 0~20
            base_idx = 3 + local_idx * 3
        elif jname == "jaw":
            base_idx = 66
        elif jname.startswith("left_hand_"):
            j_id = int(jname.split("_")[-1])
            base_idx = l_start + j_id * 3
        elif jname.startswith("right_hand_"):
            j_id = int(jname.split("_")[-1])
            base_idx = r_start + j_id * 3
        else:
            # 未知关节，直接跳过
            continue

        for axis_idx, (amin, amax) in enumerate(limits):
            angle = poses_eval[:, base_idx + axis_idx]
            lower_violation = torch.relu(amin - angle)
            upper_violation = torch.relu(angle - amax)
            v = lower_violation + upper_violation
            if v.numel() == 0:
                continue
            penalties.append(v ** 2)

    if not penalties:
        return poses_eval.new_tensor(0.0)
    stacked = torch.stack(penalties, dim=0)  # [N_constraints, B]
    return stacked.mean()

def _create_hand_only_obj(hand_obj_path: str, body_obj_path: str, cache_dir: str) -> Tuple[str, List[int], List[int]]:
    """生成仅包含手部相关三角形的 OBJ，并返回：
    - 过滤后的 hand-only obj 路径
    - kept_seg_ids: 与过滤后每个三角形对应的原始 body 语义ID（来自模板 face_group）
    - kept_face_orig_indices: 过滤后每个三角形对应的原始全局面索引（与 SMPL 面索引对齐）
    """
    os.makedirs(cache_dir, exist_ok=True)
    filtered_path = os.path.join(
        cache_dir,
        f"{os.path.splitext(os.path.basename(hand_obj_path))[0]}_handonly.obj",
    )
    seg_ids_path = filtered_path + ".seg.npy"
    face_ids_path = filtered_path + ".faceidx.npy"

    # 模板分组（用于识别手部面片索引）——使用 body 模板中的手部ID：5(左手)、9(右手)
    _, _, _, face_group_body, _, _, _ = group_faces_and_normalize_uvs(body_obj_path)
    face_group_body = np.asarray(face_group_body)
    allowed_ids = np.array([5, 9], dtype=np.int64)
    allowed_mask = np.isin(face_group_body, allowed_ids)

    kept_seg_ids: List[int] = []
    kept_face_orig_indices: List[int] = []
    with open(hand_obj_path, 'r') as src, open(filtered_path, 'w') as dst:
        face_idx = 0
        for line in src:
            if line.startswith('f '):
                if face_idx < allowed_mask.shape[0] and allowed_mask[face_idx]:
                    dst.write(line)
                    kept_seg_ids.append(int(face_group_body[face_idx]))
                    kept_face_orig_indices.append(face_idx)
                face_idx += 1
            else:
                dst.write(line)

    np.save(seg_ids_path, np.asarray(kept_seg_ids, dtype=np.int64))
    np.save(face_ids_path, np.asarray(kept_face_orig_indices, dtype=np.int64))
    return filtered_path, kept_seg_ids, kept_face_orig_indices
def crop_scale_center_tensor(tensor: torch.Tensor, bbox, target_height: int, target_width: int,
                             default_ratio: float = DEFAULT_BBOX_RATIO):
    """
    依据 bbox_v2 坐标裁剪并缩放 tensor：
    1. 将 bbox 四边以 default_ratio 做等比扩张；
    2. 若扩张后任意边触及图像边界，则返回 None（不做处理）；
    3. 取扩张后区域，保持长边与目标尺寸一致等比缩放；
    4. 将缩放后的 patch 置于黑色背景中心。
    返回处理后的 tensor，若返回 None 表示跳过。
    """
    C, H, W = tensor.shape
    x_min, y_min, x_max, y_max = bbox

    # 计算 bbox 中心与宽高
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = x_max - x_min
    bh = y_max - y_min

    # 扩张 bbox（与 multiview_v3 一致）
    bw *= default_ratio
    bh *= default_ratio
    side = max(bw, bh)
    new_xmin = int(round(cx - side / 2.0))
    new_xmax = int(round(cx + side / 2.0))
    new_ymin = int(round(cy - side / 2.0))
    new_ymax = int(round(cy + side / 2.0))
    # crop在原图的相交部分的坐标
    new_xmin_clip = max(0, new_xmin)
    new_ymin_clip = max(0, new_ymin)
    new_xmax_clip = min(W, new_xmax)
    new_ymax_clip = min(H, new_ymax)
    # 原图在crop的相交部分的坐标
    x_min_crop = (-new_xmin if new_xmin < 0 else 0) * target_height / side
    y_min_crop = (-new_ymin if new_ymin < 0 else 0) * target_height / side
    x_max_crop = (side - (new_xmax - W) if new_xmax > W else side) * target_height / side
    y_max_crop = (side - (new_ymax - H) if new_ymax > H else side) * target_height / side


    # patch_scaled = F.interpolate(patch.unsqueeze(0), size=(y_max_crop-y_min_crop,x_max_crop-x_min_crop), mode='bilinear', align_corners=False).squeeze(0)
    patch_scaled = F.interpolate(tensor[:,new_ymin_clip:new_ymax_clip,new_xmin_clip:new_xmax_clip].unsqueeze(0).to(dtype=torch.float32),
                                 size=(int(y_max_crop)-int(y_min_crop),int(x_max_crop)-int(x_min_crop)), mode='bilinear', align_corners=False).squeeze(0)

    result = torch.zeros((C, target_height, target_width), dtype=tensor.dtype, device=tensor.device)
    result[:, int(y_min_crop):int(y_max_crop), int(x_min_crop):int(x_max_crop)] = patch_scaled
    return result


def load_bbox(path_like: str):
    """
    给定任意同主体同视角的路径（image/camera/kpts），在同一组的 bbox 目录中查找同名的 .json 文件，
    并从中提取 "tlwh" 格式的 bbox，转换为 [xmin, ymin, xmax, ymax]。
    """
    # 允许传入 camera/kpts 路径：仅使用上级上级目录和同名基名
    parent_dir = os.path.dirname(os.path.dirname(path_like))
    bbox_dir = os.path.join(parent_dir, "bbox")
    bbox_path = os.path.join(bbox_dir, os.path.splitext(os.path.basename(path_like))[0] + ".json")


    with open(bbox_path, 'r') as f:
        data = json.load(f)
    tlwh = data.get("tlwh")
    t, l, w, h = tlwh
    return [t, l, t + w, l + h]


# ============================================================

# ----------------- 语义分割 Face Group 加载 -----------------
# 模板路径（将在 __main__ 块中设置）
TEMPLATE_OBJ_PATH = None  # 身体模板 obj 文件路径
HAND_SPLIT_OBJ_PATH = None  # 手部分割 obj 文件路径
# 这些全局张量用于在可视化阶段从网格插值预测UV（body 模板）
faces_uvs_tensor = None
uvs_new_tensor = None
face_group_tensor = None

def _init_uv_templates():
    """初始化 UV 模板张量，由 main 调用（在路径设置后）"""
    global TEMPLATE_OBJ_PATH, HAND_SPLIT_OBJ_PATH
    global face_group_tensor, uvs_new_tensor, faces_uvs_tensor
    
    if TEMPLATE_OBJ_PATH is None:
        raise RuntimeError("TEMPLATE_OBJ_PATH 未设置，请先在 main 中配置路径")
    
    _, _, _, _face_group_arr, _uvs_new, _faces_uvs, _ = group_faces_and_normalize_uvs(TEMPLATE_OBJ_PATH)
    face_group_tensor = torch.tensor(np.asarray(_face_group_arr), dtype=torch.long)
    uvs_new_tensor = torch.tensor(np.asarray(_uvs_new), dtype=torch.float32)
    faces_uvs_tensor = torch.tensor(np.asarray(_faces_uvs), dtype=torch.long)
    print(f"已载入 body face_group/uvs，面数: {face_group_tensor.shape[0]}, UV数: {uvs_new_tensor.shape[0]}")
# -----------------------------------------------------------
# ------------------------ Refactor Skeleton ------------------------
@dataclass
class Config:
    MODEL_FOLDER: str
    VPOSER_MODEL_DIR: str
    exp: str
    data: str
    ours_suffix: str = '_bg'
    TARGET_SIZE: int = 512
    device_id: int = 1
    RANDOM_SEED: Optional[int] = None
    # ---------- 身体/手部先验类型配置 ----------
    # BODY_PRIOR: "dposerx"（DPoser-X wholebody）或 "vposer"（VPoser）
    # HAND_PRIOR: "dposerx"（DPoser-X hand prior, 45 维轴角）或 "mano"（MANO PCA 空间）
    BODY_PRIOR: str = "dposerx"
    HAND_PRIOR: str = "dposerx"
    HAND_PCA_COMPS: int = 12  # MANO PCA 主成分数量（仅在 HAND_PRIOR="mano" 时生效）
    # MANO PCA L2 正则化权重（仅在 HAND_PRIOR="mano" 时生效）
    HAND_MANO_WEIGHT: float = 0.1
    ENABLE_VIS: bool = False
    ENABLE_IMG_VIS: bool = True
    output_prefix: str = "pred"   # 新增：输出目录前缀
    OPTIMIZER: str = "adam"  # 新增：优化器选项 ['adam', 'lbfgs']
    # TTS：若设置则读取 tempt_ttsXX_single，并为输出追加 -ttsXX 后缀
    tts_count: Optional[int] = None
    # 是否跳过手部拟合阶段（Hands-A / Hands-C）
    SKIP_HAND_FIT: bool = False
    # 是否仅跳过 Hands-C: fingers refine 阶段
    SKIP_HANDS_C: bool = False
    # 是否启用“手部独立拟合”（单独的 betas / scale / trans，并在顶点级与身体 mesh 合并）
    # 默认关闭；打开时，在 Hands-* 阶段只优化手部的 betas/scale/trans，身体保持上一阶段结果。
    SPLIT_HAND_FIT: bool = False
    # 是否启用像素置信度权重：(1-uv_uncertainty)*(1-seg_uncertainty)
    USE_UNCERTAINTY_WEIGHT: bool = False
    # 视角质量筛选
    FILTER_VIEWS: bool = False
    MIN_VALID_FACE_RATIO_BODY: float = 0.05
    MIN_VALID_FACE_RATIO_HAND: float = 0.15
    MAX_AVG_UNCERTAINTY: float = 0.1
    # 完成身体拟合后，以实际 reprojection residual 过滤灾难性手部视角。
    FILTER_HAND_VIEW_RESIDUALS: bool = True
    HAND_VIEW_RESIDUAL_MAX_RATIO: float = 4.0
    HAND_VIEW_RESIDUAL_ABS_FLOOR: float = 50.0
    HAND_VIEW_RESIDUAL_MIN_KEEP: int = 2
    HAND_VIEW_WEIGHT: float = 1. #0.1 #1.
    JOINT_LIMIT_WEIGHT: float = 0.0
    # 每个可学习参数的全局 learning rate multiplier（键名与 pose_params / 其它参数 key 对齐）
    PARAM_LR_MULTIPLIERS: Dict[str, float] = field(default_factory=dict)
    # 手部阶段（Hands-*）中手部相关参数的额外倍率（在 PARAM_LR_MULTIPLIERS 之后再乘）
    HAND_LR_MULTIPLIER: float = 1.0
    # 身体视角周期大小（默认4）。当不是4视角时，可在 task 中覆盖此值
    BODY_VIEWS: int = 4
    # global pose 失败判定：相对上升阈值与最大组合大小
    GLOBAL_POSE_FAIL_REL_INCREASE: float = 1.
    GLOBAL_POSE_MAX_COMB_SIZE: int = 8
    # DPoser-X 先验权重（当使用 dposerx 时生效）
    DPOSER_WEIGHT: float = 1.0
    # DPoser-X 分部位倍率（用于控制 body 与 hands 的相对强度）
    DPOSER_BODY_MULT: float = 1.0
    DPOSER_HAND_MULT: float = 1.0
    # ---------- 手部阶段的"冻结身体参数"开关 ----------
    # 目的：在 Hands-* 阶段只优化手相关参数，不允许动身体的根部/整体参数（避免看起来“身体大小变了”）。
    # - 当 SPLIT_HAND_FIT=True 时：trans/scale/betas 会切到 hand 版本，但 global_orient 仍是共享的，必须冻结。
    FREEZE_BODY_IN_HAND_STAGES: bool = True
    # ---------- 粗网格搜索时的 2D Procrustes 配准 ----------
    # 在粗网格搜索评估 global_orient 候选时，先对 2D 投影点做 Procrustes 配准（平移+缩放，不旋转），
    # 然后再计算 loss，以更公平地比较不同旋转角度下的形状匹配程度。
    USE_PROCRUSTES_IN_COARSE_SEARCH: bool = True
    # ---------- scale 参数优化开关 ----------
    # 是否优化 scale 参数。若为 False，则各阶段不优化 scale（使用初始值）
    OPTIMIZE_SCALE: bool = True
    # ---------- 性能优化开关 ----------
    BATCH_COARSE_SEARCH: bool = True        # global orient 粗搜索 batch 化（216 次前向 → 分批 batch forward）
    CACHE_BODY_VERTS_IN_HAND: bool = True   # 手部阶段缓存 body verts（避免重复 SMPLX forward）
    BATCH_VIEW_PROJECTION: bool = True      # 多视角投影 batch 化（消除 Python for-loop）
    COARSE_SEARCH_BATCH_SIZE: int = 64      # batch 粗搜索每批大小（控制显存）
    # ---------- [Step 0-H · EXP-6] SMPLest-X 回归参数作为 fitting 初值 ----------
    # 是否启用 SMPLest-X init（仅注入 betas + body_pose, 不动 global_orient/trans/scale 以避免坐标系冲突）
    INIT_FROM_SMPLESTX: bool = False
    # SMPLest-X init 参数 npz 所在目录（若为 None 则自动推断为 <DATASET_ROOT>/evaluate/<data>/smplestx/params）
    SMPLESTX_INIT_DIR: Optional[str] = None
    # 选用哪个视角的 SMPLest-X 输出作为 init（0~3，单视角回归模型对每个视角给出独立估计）
    SMPLESTX_INIT_VIEW: int = 0
    # 是否同时注入 hand pose（默认 False，按 EXPERIMENT_PLAN 风险分析“不污染 hand-refinement”）
    SMPLESTX_INIT_INJECT_HANDS: bool = False
    # 是否固定 SMPLest-X hand0 + left/right hand，并跳过手部优化阶段
    SMPLESTX_FIX_HANDS: bool = False
    # body_pose 注入模式：
    #   "off"        — 不注入 body_pose, 只注 betas（Plan 2，最安全）
    #   "warmstart"  — 注入 body_pose 但仅给 global pose / scale 用，full pose 启动前 reset 到 zero（Plan 1，Codex 推荐）
    #   "full"       — 注入 body_pose 全程保留（原始 v0h 行为，已知会拖慢 full pose / Fine-tune All）
    SMPLESTX_INIT_BODY_POSE_MODE: str = "warmstart"


class DenseBodyReconstructor:
    """
    重构骨架：负责加载模型、遍历数据、优化、渲染保存。
    目前仅提供接口占位，后续会逐步迁移旧版逻辑。
    """
    def __init__(self, cfg: Config):
        self.cfg = cfg
        # 直接复用脚本顶部已初始化的全局 device，避免重复 set_device 调用
        self.device = self._setup_device(cfg.device_id)

        # 模型与资源
        self._load_models()

        # DPoser 时间调度相关状态（用于按 stage 选择 t）
        self._dposer_total_steps: Optional[int] = None
        self._dposer_step_idx: int = 0
        self._dposer_t_min: Optional[float] = None
        self._dposer_t_max: Optional[float] = None

    # ----------------- DPoser 时间调度（以 stage 为单位） -----------------
    def _init_dposer_schedule(self, stage_name: str):
        """
        初始化当前 stage 的 DPoser 时间策略：
        - full pose          → 较大的 t（约 0.12）
        - Fine-tune All      → 中等 t（约 0.10）
        - Hands-A            → 偏小 t（约 0.09）
        - Hands-C            → 更小 t（约 0.08）
        其余 stage 不做特殊处理，退回到 [eps, T] 上的随机采样。
        """
        # 仅当使用 DPoser-X 身体先验时才初始化调度
        if getattr(self, "dposer_prior", None) is None or getattr(self, "_body_prior", "") != "dposerx":
            self._dposer_total_steps = None
            self._dposer_step_idx = 0
            self._dposer_t_min = None
            self._dposer_t_max = None
            return

        name_l = (stage_name or "").lower()
        target_t: Optional[float] = None
        # 仅对指定的几个 stage 使用固定 t，其余保持随机
        if "full pose" in name_l:
            target_t = 0.12
        elif "fine-tune all" in name_l or "fine-tune" in name_l:
            target_t = 0.10
        elif "hands-a" in name_l:
            target_t = 0.09
        elif "hands-c" in name_l:
            target_t = 0.08

        if target_t is None:
            # 不启用固定调度，后续 _sample_dposer_t 会退回到随机策略
            self._dposer_total_steps = None
            self._dposer_step_idx = 0
            self._dposer_t_min = None
            self._dposer_t_max = None
            return

        # 将目标 t 限制在 [eps, T] 之内
        with torch.no_grad():
            eps = float(self.dposer_prior.eps)
            T = float(self.dposer_prior.sde.T)
        t_clamped = max(min(target_t, T), eps)

        # 设定为「单步退火」：t_min == t_max == t_clamped，total_steps=1
        self._dposer_t_min = float(t_clamped)
        self._dposer_t_max = float(t_clamped)
        self._dposer_total_steps = 1
        self._dposer_step_idx = 0

    def _sample_dposer_t(self, device: torch.device) -> torch.Tensor:
        """
        返回当前步的 t：
        - 若已初始化调度，则按 time_strategy=3 从 t_max 线性退火到 t_min；
        - 若未初始化，则退回到 [eps, T] 上均匀随机采样（保持兼容性）。
        """
        if getattr(self, "dposer_prior", None) is None:
            # 退回到默认范围
            return 0.01 + torch.rand(1, device=device) * 0.99

        total_steps = self._dposer_total_steps
        idx = self._dposer_step_idx
        t_min = self._dposer_t_min
        t_max = self._dposer_t_max

        # 若没有有效的调度信息，则退回到统一随机采样
        if (
            total_steps is None
            or total_steps <= 0
            or t_min is None
            or t_max is None
        ):
            with torch.no_grad():
                t_min_raw = float(self.dposer_prior.eps)
                t_max_raw = float(self.dposer_prior.sde.T)
            return t_min_raw + torch.rand(1, device=device) * (t_max_raw - t_min_raw)

        # 计算当前 frac = (total_steps - iter - 1) / total_steps，超界做裁剪
        if idx >= total_steps:
            frac = 0.0
        else:
            frac = float(total_steps - idx - 1) / float(total_steps)
        frac = max(min(frac, 1.0), 0.0)

        t_val = t_min + frac * (t_max - t_min)
        t = torch.tensor([t_val], dtype=torch.float32, device=device)

        # 步数自增
        self._dposer_step_idx = idx + 1
        return t

    def _compute_dposer_loss(self, poses_flat: torch.Tensor, betas: torch.Tensor) -> Optional[torch.Tensor]:
        """
        使用 DPoser-X 先验计算当前姿态的先验 loss：
        - 身体：使用 whole-body prior（当 BODY_PRIOR="dposerx" 时）
        - 手部：使用 hand prior（当 HAND_PRIOR="dposerx" 时）
        - poses_flat: 轴角参数（包含 global_orient, body, hands, jaw, eyes...）
        - betas: [10]
        返回: 标量 loss，若未启用任何 DPoser 则返回 None。
        """
        whole_prior = getattr(self, "dposer_prior", None)
        hand_prior = getattr(self, "hand_dposer_prior", None)
        if whole_prior is None and hand_prior is None:
            return None
        # Debug：若输入已经包含 NaN/Inf，直接在这里抛错，便于定位是外层优化的问题
        if not torch.isfinite(poses_flat).all():
            raise RuntimeError("[NaN] poses_flat 在进入 DPoser 前已含 NaN/Inf")
        if not torch.isfinite(betas).all():
            raise RuntimeError("[NaN] betas 在进入 DPoser 前已含 NaN/Inf")
        device = poses_flat.device
        
        # 获取先验配置
        body_prior_type = getattr(self, "_body_prior", "dposerx")
        hand_prior_type = getattr(self, "_hand_prior", "dposerx")
        use_hand_pca = getattr(self, "_use_hand_pca", False)
        
        body_pose = poses_flat[3:66].unsqueeze(0)       # 63
        jaw_pose = poses_flat[66:69].unsqueeze(0)       # 3
        
        # 手部参数：若使用 MANO PCA，则用零填充（不影响 DPoser 因为不会用）
        if use_hand_pca:
            lhand_pose = torch.zeros(1, 45, dtype=poses_flat.dtype, device=device)
            rhand_pose = torch.zeros(1, 45, dtype=poses_flat.dtype, device=device)
        else:
            lhand_pose = poses_flat[75:120].unsqueeze(0)    # 45
            rhand_pose = poses_flat[120:165].unsqueeze(0)   # 45

        betas_batch = betas.unsqueeze(0)

        # 统一采样一个 t
        if whole_prior is not None:
            t = self._sample_dposer_t(device=device)
        elif hand_prior is not None:
            with torch.no_grad():
                eps_h = float(hand_prior.eps)
                T_h = float(hand_prior.sde.T)
            t = eps_h + torch.rand(1, device=device) * (T_h - eps_h)
        else:
            return None

        total_loss = poses_flat.new_tensor(0.0)
        # 注意：此处不乘 DPOSER_WEIGHT，权重由外部主循环统一控制，避免重复相乘
        body_mult = float(getattr(self.cfg, "DPOSER_BODY_MULT", 1.0))
        hand_mult = float(getattr(self.cfg, "DPOSER_HAND_MULT", 1.0))

        # -------- 1) wholebody prior：身体约束（当 BODY_PRIOR="dposerx" 时）--------
        if whole_prior is not None and body_prior_type == "dposerx":
            base_dim = 63 + 90 + 3
            total_dim = int(whole_prior.part_weights.shape[1])
            expr_dim = max(total_dim - base_dim, 0)

            if expr_dim > 0:
                expr_zeros = torch.zeros((1, expr_dim), dtype=poses_flat.dtype, device=device)
                wholebody_params = torch.cat([body_pose, lhand_pose, rhand_pose, jaw_pose, expr_zeros], dim=1)
            else:
                wholebody_params = torch.cat([body_pose, lhand_pose, rhand_pose, jaw_pose], dim=1)

            # 关闭 hands 权重（手部由 hand_prior 单独处理）
            pw = getattr(whole_prior, "part_weights", None)
            if pw is None:
                raise AttributeError("dposer_prior 缺少 part_weights")
            pw_backup = None
            try:
                with torch.no_grad():
                    pw_backup = pw.detach().clone()
                    pw_scaled = pw_backup.clone()
                    dim = int(pw_scaled.shape[-1])
                    h0, h1 = min(63, dim), min(63 + 90, dim)
                    if h1 > h0:
                        pw_scaled[..., h0:h1] = 0.0  # 关闭 hands 权重
                    pw.data.copy_(pw_scaled)
                loss_whole = whole_prior(wholebody_params, betas_batch, t)
            finally:
                if pw_backup is not None:
                    with torch.no_grad():
                        pw.data.copy_(pw_backup)
            if not torch.isfinite(loss_whole).all():
                print(f"[Warn][DPoser] wholebody prior NaN/Inf, set to 0 (t={float(t.item()):.6f}).")
                loss_whole = total_loss.new_tensor(0.0)
            total_loss = total_loss + loss_whole * body_mult

        # -------- 2) hand prior：手部约束（当 HAND_PRIOR="dposerx" 时）--------
        if hand_prior is not None and hand_prior_type == "dposerx":
            with torch.no_grad():
                eps_h = float(hand_prior.eps)
                T_h = float(hand_prior.sde.T)
            t_hand = torch.clamp(t, min=eps_h, max=T_h)
            loss_l = hand_prior(lhand_pose, betas_batch, t_hand)
            loss_r = hand_prior(rhand_pose, betas_batch, t_hand)
            loss_hand = loss_l + loss_r
            if not torch.isfinite(loss_hand).all():
                print(f"[Warn][DPoser] hand prior NaN/Inf, set to 0 (t={float(t_hand.item()):.6f}).")
                loss_hand = total_loss.new_tensor(0.0)
            total_loss = total_loss + loss_hand * hand_mult

        return total_loss if total_loss.abs() > 0 else None

    # ----------------- 初始化 -----------------
    def _setup_device(self, device_id: int):
        """根据 device_id 返回设备对象，不依赖全局变量"""
        if torch.cuda.is_available():
            dev = torch.device(f"cuda:{device_id}")
        else:
            dev = torch.device("cpu")
        print(f"[Init] 使用设备: {dev}")
        return dev

    def _load_models(self):
        """将脚本中已加载的全局模型句柄挂到实例属性。"""
        # 使用缓存目录（如果可用）
        model_folder = get_smplx_effective_folder()
        print(f"[Model] SMPLX 使用路径: {model_folder}")
        
        # 解析先验配置
        # BODY_PRIOR: "dposerx" 或 "vposer"
        # HAND_PRIOR: "dposerx" 或 "mano"（mano 时自动使用 PCA 空间）
        body_prior = getattr(self.cfg, "BODY_PRIOR", "dposerx").lower()
        hand_prior = getattr(self.cfg, "HAND_PRIOR", "dposerx").lower()
        
        # 手部使用 mano 时自动开启 PCA，使用 dposerx 时自动关闭 PCA
        use_hand_pca = (hand_prior == "mano")
        hand_pca_comps = getattr(self.cfg, "HAND_PCA_COMPS", 12) if use_hand_pca else 45
        
        # 保存到实例属性，供优化阶段使用
        self._body_prior = body_prior
        self._hand_prior = hand_prior
        self._use_hand_pca = use_hand_pca
        self._hand_pca_comps = hand_pca_comps
        
        print(f"[Model] 身体先验: {body_prior}, 手部先验: {hand_prior} → {'PCA (comps=' + str(hand_pca_comps) + ')' if use_hand_pca else '轴角 (45D)'}")
        
        self.smplx_model_male = smplx.create(model_folder, model_type='smplx',
                                gender='male',
                                ext='npz',
                                num_betas=10,
                                flat_hand_mean=False,
                                use_pca=use_hand_pca,
                                num_pca_comps=hand_pca_comps).to(self.device)

        self.smplx_model_female = smplx.create(model_folder, model_type='smplx',
                                  gender='female',
                                  ext='npz',
                                  num_betas=10,
                                  flat_hand_mean=False,
                                  use_pca=use_hand_pca,
                                  num_pca_comps=hand_pca_comps).to(self.device)

        self.smplx_model_neutral = smplx.create(model_folder, model_type='smplx',
                                   gender='neutral',
                                   ext='npz',
                                   flat_hand_mean=False,
                                   num_betas=10,
                                   use_pca=use_hand_pca,
                                   num_pca_comps=hand_pca_comps).to(self.device)

        # -------- DPoser-X 先验加载（根据 body_prior/hand_prior 决定是否加载）--------
        self.dposer_prior = None
        self.hand_dposer_prior = None
        need_dposer_body = (body_prior == "dposerx")
        need_dposer_hand = (hand_prior == "dposerx")
        
        if need_dposer_body or need_dposer_hand:
            assert os.path.isdir(DPOSER_ROOT), f"DPoser-X 根目录不存在: {DPOSER_ROOT}"
            
            # 获取有效的 DPoser 根目录（优先使用缓存）
            dposer_effective_root = get_dposer_effective_root()
            print(f"[Model] DPoser-X 使用路径: {dposer_effective_root}")
            os.chdir(dposer_effective_root)
            from types import SimpleNamespace
            
            # 加载 wholebody prior（用于身体约束）
            if need_dposer_body:
                expected_body_ckpt = os.path.join(
                    dposer_effective_root, "pretrained_models", "wholebody", "mixed", "last.ckpt"
                )
                assert os.path.isfile(expected_body_ckpt), (
                    f"DPoser-X wholebody 先验权重不存在: {expected_body_ckpt}，请先在该目录下运行 "
                    f"python download_models.py 下载预训练模型。"
                )
                dposer_args = SimpleNamespace(
                    device=self.device,
                    data_path=os.path.join(dposer_effective_root, "data"),
                    config_path="configs.wholebody.subvp.mixed.get_config",
                    ckpt_path=expected_body_ckpt,
                )
                self.dposer_prior = WholeBodyDPoser(
                    batch_size=1,
                    config_path=dposer_args.config_path,
                    args=dposer_args,
                )
                print("[Model] DPoser-X whole-body 先验已加载")

            # 加载 hand prior（用于手部约束）
            if need_dposer_hand:
                expected_hand_ckpt = os.path.join(
                    dposer_effective_root, "pretrained_models", "hand", "BaseMLP", "last.ckpt"
                )
                assert os.path.isfile(expected_hand_ckpt), (
                    f"DPoser-X hand 先验权重不存在: {expected_hand_ckpt}，请先在该目录下运行 "
                    f"python download_models.py 下载预训练模型。"
                )
                expected_hand_norm = os.path.join(
                    dposer_effective_root, "data", "hand_data", "hand_normalizer", "axis_normalize2.pt"
                )
                assert os.path.isfile(expected_hand_norm), (
                    f"DPoser-X hand normalizer 不存在: {expected_hand_norm}。"
                    f"请确认 DPoser-X/data/hand_data 已完整下载。"
                )
                hand_args = SimpleNamespace(
                    device=self.device,
                    data_path=os.path.join(dposer_effective_root, "data", "hand_data"),
                    config_path="configs.hand.subvp.timefc.get_config",
                    ckpt_path=expected_hand_ckpt,
                )
                self.hand_dposer_prior = HandDPoser(
                    batch_size=1,
                    config_path=hand_args.config_path,
                    args=hand_args,
                )
                print("[Model] DPoser-X hand 先验已加载")

        # -------- VPoser 加载（当 body_prior="vposer" 时）--------
        self.vp_model = None
        if body_prior == "vposer":
            print(f"[GPU{self.cfg.device_id}] 加载 VPoser: {self.cfg.VPOSER_MODEL_DIR}")
            vp_model_inst, _ = load_model(
                self.cfg.VPOSER_MODEL_DIR,
                model_code=VPoser,
                remove_words_in_model_weights='vp_model.',
                disable_grad=True,
            )
            self.vp_model = vp_model_inst.to(self.device)
            self.vp_model.eval()

        # UV 分辨率
        self.uv_size = 128  # 与 render/seg 输入保持一致（按需调试）

        # 简要日志
        print("[Model] SMPL-X & DPoser-X 先验句柄已就绪")

        # Face index tensor（常量）
        self.smplx_faces_tensor = torch.from_numpy(self.smplx_model_neutral.faces.astype(np.int64)).to(self.device)

        # UV group 预处理（一次即可）：body + hand（细分UV，非手面被剔除）
        body_uv_info, self.max_gid = load_group_uv_info(
            obj_path=TEMPLATE_OBJ_PATH, uv_size=self.uv_size, cache_dir=os.path.join(CACHE_DIR, 'cache_body'), extra_uv_payload=False
        )
        hand_obj_path = HAND_SPLIT_OBJ_PATH
        hand_uv_info, _ = load_group_uv_info(
            obj_path=hand_obj_path, uv_size=self.uv_size, cache_dir=os.path.join(CACHE_DIR, 'cache_hand'), extra_uv_payload=True
        )

        # 映射到 seg_id → UV 查找表
        self.group_uv_lookup: Dict[int, Dict[str, torch.Tensor]] = {}
        for sid, info in body_uv_info.items():
            if isinstance(sid, int):
                self.group_uv_lookup[int(sid)] = info

        meta = hand_uv_info.get('metadata', {}) if isinstance(hand_uv_info, dict) else {}
        local2seg: Dict[int, int] = meta.get('local_gid_to_seg_id', {}) if isinstance(meta, dict) else {}
        local2fine: Dict[int, int] = meta.get('local_gid_to_fine_seg_id', {}) if isinstance(meta, dict) else {}
        kept_face_orig_indices = meta.get('kept_face_orig_indices') if isinstance(meta, dict) else None
        hand_uvs_new_np = meta.get('uvs_new') if isinstance(meta, dict) else None
        hand_faces_uvs_np = meta.get('faces_uvs') if isinstance(meta, dict) else None
        hand_face_group_local = meta.get('face_group_local') if isinstance(meta, dict) else None
        hand_keys = [k for k in hand_uv_info.keys() if isinstance(k, int)]
        for local_gid in hand_keys:
            seg_id = int(local2fine.get(local_gid, local2seg.get(local_gid, local_gid)))
            self.group_uv_lookup[seg_id] = hand_uv_info[local_gid]
        import time as _time

        # ---------- 手部可视化所需：构建全局 face → 手部UV 三角索引与语义 ----------
        nfaces_global = self.smplx_faces_tensor.shape[0]
        self.hand_uvs_new_tensor = None
        self.hand_tri_uv_idx_global = None
        self.hand_face_group_global = None
        if kept_face_orig_indices is None or hand_uvs_new_np is None or hand_faces_uvs_np is None or hand_face_group_local is None:
            self.hand_face_mask_global = None
        else:
            kept_face_orig_indices_t = torch.tensor(np.asarray(kept_face_orig_indices), dtype=torch.long)
            faces_uvs_hand_t = torch.tensor(np.asarray(hand_faces_uvs_np), dtype=torch.long)
            uvs_new_hand_t = torch.tensor(np.asarray(hand_uvs_new_np), dtype=torch.float32)
            face_group_local_np = np.asarray(hand_face_group_local)

            tri_uv_idx_global = torch.full((nfaces_global, 3), -1, dtype=torch.long)
            tri_uv_idx_global[kept_face_orig_indices_t] = faces_uvs_hand_t
            self.hand_tri_uv_idx_global = tri_uv_idx_global
            self.hand_face_mask_global = (tri_uv_idx_global[:, 0] >= 0)

            # 语义映射：局部gid -> 全局细分seg id（优先使用细分映射）
            seg_map_global = torch.zeros(nfaces_global, dtype=torch.long)
            for i in range(kept_face_orig_indices_t.shape[0]):
                gface = int(kept_face_orig_indices_t[i].item())
                local_gid = int(face_group_local_np[i])
                mapped = int(local2fine.get(local_gid, local2seg.get(local_gid, 0)))
                seg_map_global[gface] = mapped
            self.hand_face_group_global = seg_map_global
            self.hand_uvs_new_tensor = uvs_new_hand_t

        # ---------- 从手部面片构建全局手部顶点掩码（用于手/身独立拟合后在顶点级合并） ----------
        verts_count = int(self.smplx_model_neutral.v_template.shape[0])
        hand_vertex_mask = torch.zeros(verts_count, dtype=torch.bool)
        if getattr(self, "hand_face_mask_global", None) is not None:
            face_idx_hand = torch.nonzero(self.hand_face_mask_global, as_tuple=False).squeeze(1)
            if face_idx_hand.numel() > 0:
                verts_in_hand = self.smplx_faces_tensor[face_idx_hand].unique()
                hand_vertex_mask[verts_in_hand] = True
        self.hand_vertex_mask_global = hand_vertex_mask

        # ---------- 额外：左右手/掌心顶点掩码（用于“以掌心为中心的局部缩放”） ----------
        # 说明：
        # - seg id 定义来自本脚本的细分手部语义：LEFT_HAND_FINE_SEG_IDS=(12..17), RIGHT_HAND_FINE_SEG_IDS=(18..23)
        # - 且在 Hands-B 的 palm 选择中，默认使用 12（左掌心）与 18（右掌心）
        self.left_hand_vertex_mask_global = torch.zeros(verts_count, dtype=torch.bool)
        self.right_hand_vertex_mask_global = torch.zeros(verts_count, dtype=torch.bool)
        self.left_palm_vertex_mask_global = torch.zeros(verts_count, dtype=torch.bool)
        self.right_palm_vertex_mask_global = torch.zeros(verts_count, dtype=torch.bool)
        if (getattr(self, "hand_face_mask_global", None) is not None) and (getattr(self, "hand_face_group_global", None) is not None):
            seg_face = self.hand_face_group_global
            if torch.is_tensor(seg_face):
                seg_face = seg_face.detach().cpu()
            # 左右手面片（按 seg id 范围）
            left_face_mask = torch.zeros_like(seg_face, dtype=torch.bool)
            for sid in LEFT_HAND_SEG_IDS:
                left_face_mask |= (seg_face == int(sid))
            right_face_mask = torch.zeros_like(seg_face, dtype=torch.bool)
            for sid in RIGHT_HAND_SEG_IDS:
                right_face_mask |= (seg_face == int(sid))
            # 掌心面片（约定：左掌心=12，右掌心=18）
            left_palm_face_mask = (seg_face == 12)
            right_palm_face_mask = (seg_face == 18)

            left_face_idx = torch.nonzero(left_face_mask, as_tuple=False).squeeze(1)
            right_face_idx = torch.nonzero(right_face_mask, as_tuple=False).squeeze(1)
            left_palm_face_idx = torch.nonzero(left_palm_face_mask, as_tuple=False).squeeze(1)
            right_palm_face_idx = torch.nonzero(right_palm_face_mask, as_tuple=False).squeeze(1)

            if left_face_idx.numel() > 0:
                v_left = self.smplx_faces_tensor[left_face_idx].unique()
                self.left_hand_vertex_mask_global[v_left] = True
            if right_face_idx.numel() > 0:
                v_right = self.smplx_faces_tensor[right_face_idx].unique()
                self.right_hand_vertex_mask_global[v_right] = True
            if left_palm_face_idx.numel() > 0:
                v_lp = self.smplx_faces_tensor[left_palm_face_idx].unique()
                self.left_palm_vertex_mask_global[v_lp] = True
            if right_palm_face_idx.numel() > 0:
                v_rp = self.smplx_faces_tensor[right_palm_face_idx].unique()
                self.right_palm_vertex_mask_global[v_rp] = True

        def _merge_verts_hand_body(self_ref, verts_body: torch.Tensor, verts_hand: torch.Tensor) -> torch.Tensor:
            """按全局手部顶点掩码，将两次前向的顶点结果在手部区域进行合并。

            - 仅在 Config.SPLIT_HAND_FIT=True 且 hand_vertex_mask_global 有效时使用；
            - 身体顶点保持 verts_body，手部顶点来自 verts_hand。
            """
            if getattr(self_ref, "hand_vertex_mask_global", None) is None:
                return verts_body
            mask = self_ref.hand_vertex_mask_global.to(verts_body.device)
            if mask.dtype != torch.bool:
                mask = mask.bool()
            merged = verts_body.clone()
            merged[mask] = verts_hand[mask]
            return merged

        # 绑定为实例方法，供优化阶段在启用 SPLIT_HAND_FIT 时调用
        self._merge_verts_hand_body = _merge_verts_hand_body.__get__(self, self.__class__)

        # ------------- Load J_regressor -------------
        smplx_model_path = os.path.join(get_smplx_effective_folder(), 'smplx', 'SMPLX_NEUTRAL.pkl')
        with open(smplx_model_path, 'rb') as f:
            smplx_data = pickle.load(f, encoding='latin1')
        self.J_regressor_smplx = torch.tensor(np.array(smplx_data['J_regressor']), dtype=torch.float32).to(self.device)

    # ----------------- 数据准备 -----------------
    def prepare_dataset(self):
        """根据 cfg 生成路径并扫描主体列表。"""
        cfg = self.cfg

        # 生成关键路径（使用 DATASET_ROOT 作为数据集根目录）
        self.CAMERAS_GT_PATH = os.path.join(DATASET_ROOT, "evaluate", cfg.data, "gt", "camera")
        self.base_path = os.path.join(
            DATASET_ROOT,
            "evaluate",
            cfg.data,
            f"ours{cfg.ours_suffix}",
            cfg.exp,
        )
        # 仅使用裁剪后的单视角输入；若配置了 tts_count，则读取对应 tts 目录
        if cfg.tts_count is not None:
            tts_suffix = f"tts{cfg.tts_count}"
            self.input_path = os.path.join(self.base_path, f"tempt_{tts_suffix}_single")
        else:
            self.input_path = os.path.join(self.base_path, "tempt_single")




        def _collect_subjects(prefix: str):
            if not os.path.isdir(self.input_path):
                print(f"[Dataset] 输入目录不存在: {self.input_path}")
                return set()
            pat = re.compile(r'^uv_(?P<sid>\d{4})_\d{2}_pred\.png$')
            c_files = [f for f in os.listdir(self.input_path) if f.startswith(prefix) and f.endswith('.png')]
            subjects = set()
            for f in c_files:
                m = pat.match(f)
                if m:
                    subjects.add(int(m.group('sid')))
            return subjects

        self.subjects = sorted(_collect_subjects("uv_"))

        # 记录
        print(f"[Dataset] 共找到 {len(self.subjects)} 个主体")

        # 预留 view_data 容器（后续在 optimize_subject 内填充）
        self.view_data = {}

    # ------- 参考 test_script_i2mv_v6.1_fix_bbox 的内参更新（仅内参） -------
    def _update_intrinsic_by_bbox(self,
                                  K_orig: torch.Tensor,
                                  bbox_norm: Tuple[float, float, float, float],
                                  h_ori: int,
                                  w_ori: int,
                                  h_tgt: int,
                                  w_tgt: int,
                                  default_ratio: float = DEFAULT_BBOX_RATIO) -> torch.Tensor:
        x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_norm

        cx = (x_min_norm + x_max_norm) / 2.0
        cy = (y_min_norm + y_max_norm) / 2.0
        bw = x_max_norm - x_min_norm
        bh = y_max_norm - y_min_norm
        side = max(bw, bh) * default_ratio

        new_xmin_tmp = cx - side / 2.0
        new_xmax_tmp = cx + side / 2.0
        new_ymin_tmp = cy - side / 2.0
        new_ymax_tmp = cy + side / 2.0

        new_xmin = max(new_xmin_tmp, 0.)
        new_xmax = min(new_xmax_tmp, 1.)
        new_ymin = max(new_ymin_tmp, 0.)
        new_ymax = min(new_ymax_tmp, h_ori / w_ori)

        tgt_xmin = round((new_xmin - new_xmin_tmp) * w_tgt / side)
        tgt_xmax = round((new_xmax - new_xmin_tmp) * w_tgt / side)
        tgt_ymin = round((new_ymin - new_ymin_tmp) * h_tgt / side)
        tgt_ymax = round((new_ymax - new_ymin_tmp) * h_tgt / side)

        actual_crop_x_start = round(new_xmin * w_ori)
        actual_crop_x_end = round(new_xmax * w_ori)
        actual_crop_y_start = round(new_ymin * w_ori)
        actual_crop_y_end = round(new_ymax * w_ori)

        actual_patch_w = actual_crop_x_end - actual_crop_x_start
        actual_patch_h = actual_crop_y_end - actual_crop_y_start

        target_w = tgt_xmax - tgt_xmin
        target_h = tgt_ymax - tgt_ymin

        scale_x = target_w / actual_patch_w if actual_patch_w != 0 else 1.0
        scale_y = target_h / actual_patch_h if actual_patch_h != 0 else 1.0

        offset_x = tgt_xmin - scale_x * actual_crop_x_start
        offset_y = tgt_ymin - scale_y * actual_crop_y_start

        A = torch.tensor([
            [scale_x, 0.0, offset_x],
            [0.0, scale_y, offset_y],
            [0.0, 0.0, 1.0]
        ], dtype=K_orig.dtype, device=K_orig.device)

        return A @ K_orig

    def _update_intrinsic_by_bbox_rect(self,
                                       K_orig: torch.Tensor,
                                       bbox_norm: Tuple[float, float, float, float],
                                       h_ori: int,
                                       w_ori: int,
                                       h_tgt: int,
                                       w_tgt: int) -> torch.Tensor:
        """使用原始矩形 bbox（不扩张、不方形化）更新内参。
        - bbox_norm 为基于原始宽度归一化的 (xmin, ymin, xmax, ymax)
        - 输出返回 A @ K_orig，其中 A 将原图坐标仿射到目标分辨率。
        """
        x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_norm

        # 直接使用给定矩形，无扩张、无方形化
        new_xmin_tmp = x_min_norm
        new_xmax_tmp = x_max_norm
        new_ymin_tmp = y_min_norm
        new_ymax_tmp = y_max_norm

        # 边界裁剪（注意 y 仍按宽度归一化，最大为 h_ori/w_ori）
        new_xmin = max(new_xmin_tmp, 0.)
        new_xmax = min(new_xmax_tmp, 1.)
        new_ymin = max(new_ymin_tmp, 0.)
        new_ymax = min(new_ymax_tmp, h_ori / w_ori)

        # 目标画布上的占位范围（考虑越界被裁剪后的留白）
        bw = max(new_xmax_tmp - new_xmin_tmp, 1e-8)
        bh = max(new_ymax_tmp - new_ymin_tmp, 1e-8)
        tgt_xmin = round((new_xmin - new_xmin_tmp) * w_tgt / bw)
        tgt_xmax = round((new_xmax - new_xmin_tmp) * w_tgt / bw)
        tgt_ymin = round((new_ymin - new_ymin_tmp) * h_tgt / bh)
        tgt_ymax = round((new_ymax - new_ymin_tmp) * h_tgt / bh)

        # 实际在原图中的裁剪像素范围
        actual_crop_x_start = round(new_xmin * w_ori)
        actual_crop_x_end = round(new_xmax * w_ori)
        actual_crop_y_start = round(new_ymin * w_ori)
        actual_crop_y_end = round(new_ymax * w_ori)

        actual_patch_w = max(actual_crop_x_end - actual_crop_x_start, 1)
        actual_patch_h = max(actual_crop_y_end - actual_crop_y_start, 1)

        target_w = max(tgt_xmax - tgt_xmin, 1)
        target_h = max(tgt_ymax - tgt_ymin, 1)

        scale_x = target_w / actual_patch_w
        scale_y = target_h / actual_patch_h

        offset_x = tgt_xmin - scale_x * actual_crop_x_start
        offset_y = tgt_ymin - scale_y * actual_crop_y_start

        A = torch.tensor([
            [scale_x, 0.0, offset_x],
            [0.0, scale_y, offset_y],
            [0.0, 0.0, 1.0]
        ], dtype=K_orig.dtype, device=K_orig.device)

        return A @ K_orig

    # ------------------------------------------------------------------
    # 每个主体视图解析
    # ------------------------------------------------------------------

    def _prepare_subject_views(self, subject: int):
        """读取 subject 的所有视图图片和相机参数，返回 view_data dict。"""
        cfg = self.cfg

        both_lst = []  # (H,W,4) uv+seg or combined

        # 严格匹配该 subject 下的 20 视角 uv 切片，避免匹配到 uv_uncertainty_*；
        # 仅保留 uv/seg 都存在的成对视角，并使用文件名中的 vid_code 作为视角ID
        pat_uv = re.compile(rf'^uv_{subject:04d}_(?P<vid>\d{{2}})_pred\.png$')
        uv_candidates = [p for p in glob.glob(os.path.join(self.input_path, f"uv_{subject:04d}_*_pred.png"))
                         if pat_uv.match(os.path.basename(p))]
        pairs: List[Tuple[int, str, str]] = []  # (vid_code, uv_path, seg_path)
        for upath in uv_candidates:
            m = pat_uv.match(os.path.basename(upath))
            if not m:
                continue
            vid_code = int(m.group('vid'))
            spath = upath.replace("uv_", "seg_")
            if not os.path.exists(spath):
                # 缺少对应 seg，跳过该视角
                continue
            pairs.append((vid_code, upath, spath))
        pairs.sort(key=lambda x: x[0])

        if len(pairs) == 0:
            print(f"[Prepare] 主体 {subject} 无可用视角，跳过")
            return None

        for _, upath, spath in pairs:
            uv_img = imageio.v2.imread(upath)
            seg_img = imageio.v2.imread(spath)
            uv_2ch = uv_img[..., :2]
            both_img = np.concatenate([uv_2ch, seg_img], axis=2)
            both_lst.append(both_img)
            # 强制保存输入UV，供可视化 GT-UV 使用
            #（注意：真正写入 view_data 在后面统一进行）

        if len(both_lst) == 0:
            print(f"[Prepare] 主体 {subject} 无有效图片，跳过")
            return None

        # 使用文件名中的 vid_code 作为视角ID，避免因缺失视角导致的索引错位
        view_ids = [vid for vid, _, _ in pairs]
        view_data = {}

        # 视角周期大小：默认4；如需非4视角，可在任务配置 BODY_VIEWS 中覆盖
        cycle_size = int(getattr(self.cfg, 'BODY_VIEWS', 4))

        # 遍历视角生成详细信息
        for _idx, (vid_code, upath, spath) in enumerate(pairs):
            view_id = int(vid_code)
            both = both_lst[_idx]

            # 读取 GT 相机
            # 手脚视角与身体视角共享相机：按 BODY_VIEWS 周期取模配对
            base_view_id = int(vid_code) % cycle_size
            group_idx = int(vid_code) // cycle_size  # 0:body, 1:LHand, 2:RHand, 3:LFoot, 4:RFoot
            part_map = {1: 'left_hand', 2: 'right_hand', 3: 'left_foot', 4: 'right_foot'}
            cam_base_name = f"{subject:04d}_{base_view_id:02d}"
            cam_npz_path = os.path.join(self.CAMERAS_GT_PATH, f"{cam_base_name}.npz")
            if not os.path.exists(cam_npz_path):
                raise FileNotFoundError(f"未找到相机文件: {cam_npz_path}")

            cam_npz = np.load(cam_npz_path)
            R_np = cam_npz["R"].astype(np.float32)
            T_np = cam_npz["T"].astype(np.float32) / 1000.0
            K_np = cam_npz["K"].astype(np.float32)

            # ---- 不对输入做任何缩放，保持分辨率 ----
            orig_h, orig_w = both.shape[:2]
            # 获取原图尺寸（用于 bbox 归一化 与 cameras_orig 构建）
            image_ori_dir = os.path.join(os.path.dirname(self.CAMERAS_GT_PATH), 'image_ori')
            _ori_path = None
            for _ext in ['.png', '.jpg', '.jpeg']:
                _cand = os.path.join(image_ori_dir, f"{cam_base_name}{_ext}")
                if os.path.exists(_cand):
                    _ori_path = _cand
                    break
            if _ori_path is not None:
                _img_ori = imageio.v2.imread(_ori_path)
                if _img_ori is not None:
                    orig_h, orig_w = _img_ori.shape[:2]
            # 依据 test_script 的规则按 bbox 仿射修正 K（仅内参）
            # 标准化坐标（x,y 按原始宽度归一化）
            gt_root = os.path.join(os.path.dirname(self.CAMERAS_GT_PATH))
            body_bbox_json = os.path.join(gt_root, 'bbox', f'{cam_base_name}.json')

            tlwh = None
            if group_idx == 0:
                # 身体视角：直接读取 body bbox
                with open(body_bbox_json, 'r') as _f:
                    _jd = json.load(_f)
                tlwh = _jd['tlwh']
            else:
                # 手部视角：优先 bbox_hand_foot/<part>/*.json；若无则跳过该视角
                part = part_map.get(group_idx)
                # 1) 尝试手部检测结果 json
                handfoot_root = os.path.join(self.base_path, 'bbox_hand_foot')
                hand_json_path = os.path.join(handfoot_root, part, f'{subject:04d}_{base_view_id:02d}.json')
                if os.path.exists(hand_json_path):
                    with open(hand_json_path, 'r') as _f:
                        _jd = json.load(_f)
                    tlwh = _jd.get('tlwh')
                    if tlwh is None:
                        # json 文件存在但无 tlwh 字段，跳过
                        print(f"[Prepare] 主体 {subject} 视角 {vid_code} bbox 文件无 tlwh 字段，跳过")
                        continue
                else:
                    # bbox 文件不存在（手部未检测到），跳过该视角
                    print(f"[Prepare] 主体 {subject} 视角 {vid_code} 缺少 bbox 文件 ({part})，跳过")
                    continue

            t, l, w, h = tlwh
            x_min, y_min, x_max, y_max = t, l, t + w, l + h
            img_w = float(orig_w)
            bbox_norm = (
                x_min / img_w,
                y_min / img_w,
                x_max / img_w,
                y_max / img_w,
            )

            # 当前输入图尺寸（不缩放）
            resized_h, resized_w = both.shape[:2]

            if group_idx in (1, 2):
                # 手部视角：bbox_hand_foot json 已含完整扩张，按方形逻辑但 ratio=1.0（不再额外放大）
                K_scaled = self._update_intrinsic_by_bbox(
                    torch.from_numpy(K_np),
                    bbox_norm,
                    h_ori=orig_h,
                    w_ori=orig_w,
                    h_tgt=resized_h,
                    w_tgt=resized_w,
                    default_ratio=1.0,
                ).numpy().astype(np.float32)
            else:
                # 身体视角：保留原逻辑（按 DEFAULT_BBOX_RATIO 扩张为方框）
                K_scaled = self._update_intrinsic_by_bbox(
                    torch.from_numpy(K_np),
                    bbox_norm,
                    h_ori=orig_h,
                    w_ori=orig_w,
                    h_tgt=resized_h,
                    w_tgt=resized_w,
                    default_ratio=DEFAULT_BBOX_RATIO,
                ).numpy().astype(np.float32)

            # 构建 Pytorch3D 相机
            R_tensor = torch.from_numpy(R_np).to(self.device)
            T_tensor = torch.from_numpy(T_np).to(self.device)
            K_tensor = torch.from_numpy(K_scaled).to(self.device)

            cameras_p3d = cameras_from_opencv_projection(
                R_tensor[None],
                T_tensor[None],
                K_tensor[None],
                torch.tensor([[resized_h, resized_w]], dtype=torch.float32, device=self.device),
            )

            R_tensor = cameras_p3d.R[0]
            T_tensor = cameras_p3d.T[0]

            focal_x, focal_y, cx, cy = K_scaled[0, 0], K_scaled[1, 1], K_scaled[0, 2], K_scaled[1, 2]

            # original resolution camera
            cameras_orig = cameras_from_opencv_projection(
                torch.from_numpy(R_np).to(self.device)[None],
                torch.from_numpy(T_np).to(self.device)[None],
                torch.from_numpy(K_np).to(self.device)[None],
                torch.tensor([[orig_h, orig_w]], dtype=torch.float32, device=self.device),
            )

            # mask & uv tensors
            mask = (both[..., -1] > 0).astype(np.uint8)
            mask_tensor = torch.from_numpy(mask).to(self.device)

            uv_body_tensor = torch.from_numpy(both[..., :2]).to(self.device)

            # pixel grid
            y_grid, x_grid = np.mgrid[0:resized_h, 0:resized_w]
            cam_xy_tensor = torch.from_numpy(np.stack([x_grid, y_grid], axis=-1)).float().to(self.device)

            # UV -> SMPL lookup
            pts_face_all = torch.full((resized_h, resized_w), -1, dtype=torch.long, device=self.device)
            pts_bw_all = torch.zeros((resized_h, resized_w, 3), dtype=torch.float32, device=self.device)

            # 依据视角类型，仅使用对应语义颜色子集进行最近颜色匹配
            seg_np = both[..., -3:].copy()
            seg_tensor = torch.from_numpy(seg_np).to(self.device)
            if group_idx in (1, 2):
                # 手部视角：背景(0) + 左右手(12~23) + 灰手(24)
                allowed_ids = (0,) + LEFT_HAND_SEG_IDS + RIGHT_HAND_SEG_IDS + (GRAY_HAND_SEG_ID,)
            else:
                # 身体视角：背景(0) + 身体(1~11)
                allowed_ids = (0,) + BODY_SEG_IDS
            seg_mask_list = [m.to(self.device) for m in generate_class_masks_subset(seg_tensor, allowed_ids=allowed_ids)]
            # 边缘腐蚀：与 uncertainty 互斥（uncertainty=True 则禁用腐蚀；否则启用腐蚀）
            do_erode = (not self.cfg.USE_UNCERTAINTY_WEIGHT)
            if do_erode:
                kernel = torch.from_numpy(np.ones((3, 3), dtype=np.uint8)).to(self.device)
                eroded_masks = []
                for mk in seg_mask_list:
                    mk_np = mk.detach().cpu().numpy().astype(np.uint8)
                    mk_er = cv2.erode(mk_np, np.ones((3, 3), np.uint8), iterations=1)
                    eroded_masks.append(torch.from_numpy(mk_er).int().to(self.device))
                seg_mask_list = eroded_masks

            current_uv_idx = (uv_body_tensor.float() / 255.0 * self.uv_size).long()
            current_uv_idx[..., 0].clamp_(0, self.uv_size - 1)
            current_uv_idx[..., 1].clamp_(0, self.uv_size - 1)

            for gid, id_mask in enumerate(seg_mask_list):
                if gid == 0:
                    continue
                mask_g = id_mask.to(self.device).bool()
                if mask_g.sum() == 0:
                    continue
                g_info = self.group_uv_lookup.get(gid)
                if g_info is None:
                    continue
                g_pf = g_info["pixel_face"].to(self.device)
                g_bw = g_info["pixel_bary"].to(self.device)
                (x_min, y_min), (x_max, y_max) = g_info["bbox"]

                y_idx = ((current_uv_idx[..., 1].reshape(-1)[mask_g.reshape(-1)] / self.uv_size * (y_max - y_min) + y_min) * self.uv_size).long()
                x_idx = ((current_uv_idx[..., 0].reshape(-1)[mask_g.reshape(-1)] / self.uv_size * (x_max - x_min) + x_min) * self.uv_size).long()

                pf_now = g_pf[-y_idx, x_idx]
                bw_now = g_bw[-y_idx, x_idx]

                pts_face_all.reshape(-1)[mask_g.reshape(-1)] = pf_now
                pts_bw_all.reshape(-1, 3)[mask_g.reshape(-1)] = bw_now

            # ---------- 置信度权重 (1-uv_uncertainty)*(1-seg_uncertainty) ----------
            def _read_uncertainty_float01(path: str) -> np.ndarray:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"未找到 uncertainty 文件: {path}")
                arr = imageio.v2.imread(path).astype(np.float32)
                if arr.ndim == 3:
                    # 若是多通道，取均值为单通道 mask
                    if arr.shape[2] == 1:
                        arr = arr[..., 0]
                    else:
                        arr = arr.mean(axis=2)
                if arr.max() > 1.0:
                    arr = arr / 255.0
                arr = np.clip(arr, 0.0, 1.0)
                return arr

            uv_unc_np = None
            seg_unc_np = None
            if self.cfg.USE_UNCERTAINTY_WEIGHT:
                uv_unc_path = os.path.join(os.path.dirname(upath), os.path.basename(upath).replace('uv_', 'uv_uncertainty_'))
                seg_unc_path = os.path.join(os.path.dirname(spath), os.path.basename(spath).replace('seg_', 'seg_uncertainty_'))
                uv_unc = _read_uncertainty_float01(uv_unc_path)
                seg_unc = _read_uncertainty_float01(seg_unc_path)
                if uv_unc.shape[:2] != both.shape[:2]:
                    raise ValueError(f"UV uncertainty 尺寸与输入不一致: {uv_unc.shape} vs {both.shape[:2]}")
                if seg_unc.shape[:2] != both.shape[:2]:
                    raise ValueError(f"SEG uncertainty 尺寸与输入不一致: {seg_unc.shape} vs {both.shape[:2]}")
                conf_uv = 1.0 - uv_unc
                conf_seg = 1.0 - seg_unc
                uv_unc_np = uv_unc
                seg_unc_np = seg_unc
                pixel_weight_np = (conf_uv * conf_seg).astype(np.float32)
            else:
                pixel_weight_np = np.ones(both.shape[:2], dtype=np.float32)
                # 若仅用于筛选，则尽量读取不确定性用于统计
                if self.cfg.FILTER_VIEWS:
                    uv_unc_path = os.path.join(os.path.dirname(upath), os.path.basename(upath).replace('uv_', 'uv_uncertainty_'))
                    seg_unc_path = os.path.join(os.path.dirname(spath), os.path.basename(spath).replace('seg_', 'seg_uncertainty_'))
                    uv_unc_tmp = _read_uncertainty_float01(uv_unc_path)
                    seg_unc_tmp = _read_uncertainty_float01(seg_unc_path)
                    if uv_unc_tmp.shape[:2] == both.shape[:2] and seg_unc_tmp.shape[:2] == both.shape[:2]:
                        uv_unc_np = uv_unc_tmp
                        seg_unc_np = seg_unc_tmp


            pixel_weight_tensor = torch.from_numpy(pixel_weight_np).to(self.device)
            # ---------- 视角质量筛选：有效面片比例与平均不确定性 ----------
            if self.cfg.FILTER_VIEWS:
                # 构造允许的语义掩码（按视角类型）
                allow_ids = BODY_SEG_IDS if group_idx == 0 else (LEFT_HAND_SEG_IDS + RIGHT_HAND_SEG_IDS)
                mask_allowed = torch.zeros_like(mask_tensor, dtype=torch.bool)
                for sid_allow in allow_ids:
                    if 0 <= sid_allow < len(seg_mask_list):
                        mask_allowed |= seg_mask_list[sid_allow].bool()

                # 有效UV查找像素（face!=-1 且 bary>0）
                valid_face_mask = (pts_face_all >= 0) & (pts_bw_all.abs().sum(dim=-1) > 0)
                eff_mask = (mask_allowed & valid_face_mask)

                # 汇总视角中出现的全局 face id
                faces_now = pts_face_all[eff_mask]
                if faces_now.numel() > 0:
                    faces_now_cpu = faces_now.detach().flatten().cpu().numpy()
                    # 手部仅统计手部faces，排除灰手（通过全局手部面掩码过滤）
                    if group_idx in (1, 2) and (self.hand_face_mask_global is not None):
                        hand_mask_np = self.hand_face_mask_global.detach().cpu().numpy().astype(bool)
                        faces_now_cpu = faces_now_cpu[hand_mask_np[faces_now_cpu]]
                    unique_faces = np.unique(faces_now_cpu)
                    num_unique = int(unique_faces.size)
                else:
                    num_unique = 0

                # 计算分母：身体总faces / 手部总faces
                if group_idx == 0:
                    fg_np = face_group_tensor.detach().cpu().numpy()
                    denom = int(np.isin(fg_np, np.array(BODY_SEG_IDS)).sum())
                    min_ratio = self.cfg.MIN_VALID_FACE_RATIO_BODY
                    max_uncertain = 0.25
                else:
                    denom = int(self.hand_face_mask_global.sum().item()) if (self.hand_face_mask_global is not None) else 1
                    min_ratio = self.cfg.MIN_VALID_FACE_RATIO_HAND
                    max_uncertain = self.cfg.MAX_AVG_UNCERTAINTY
                denom = max(denom, 1)
                valid_face_ratio = num_unique / float(denom)

                # 平均不确定性（仅在非灰非黑的内容区域）：组合不确定性 = 1 - (1-uv_unc)*(1-seg_unc)
                avg_uncertainty = None
                if (uv_unc_np is not None) and (seg_unc_np is not None):
                    comb_unc = 1.0 - (1.0 - uv_unc_np) * (1.0 - seg_unc_np)

                    # 构造“非黑(0)且（手部视角时）非灰(24)”的内容掩码
                    content_mask = torch.zeros_like(mask_tensor, dtype=torch.bool)
                    # 允许的语义已体现在 mask_allowed，但需要去除黑与灰
                    content_mask |= mask_allowed
                    # 去除黑色背景 seg=0（若存在）
                    if 0 < len(seg_mask_list):
                        content_mask &= ~seg_mask_list[0].bool()
                    # 手部视角去除灰手 seg=24（若存在）
                    if group_idx in (1, 2) and (GRAY_HAND_SEG_ID < len(seg_mask_list)):
                        content_mask &= ~seg_mask_list[GRAY_HAND_SEG_ID].bool()

                    # 与有效面片相交作为首选统计区域
                    eff_mask_content = content_mask & eff_mask
                    eff_mask_content_np = eff_mask_content.detach().cpu().numpy().astype(bool)
                    content_mask_np = content_mask.detach().cpu().numpy().astype(bool)

                    if eff_mask_content_np.any():
                        avg_uncertainty = float(comb_unc[eff_mask_content_np].mean())
                    elif content_mask_np.any():
                        avg_uncertainty = float(comb_unc[content_mask_np].mean())
                    else:
                        avg_uncertainty = float(comb_unc.mean())

                # 判定与筛除
                # print(avg_uncertainty)
                drop_by_ratio = valid_face_ratio < min_ratio
                drop_by_unc = (avg_uncertainty is not None) and (avg_uncertainty > max_uncertain)
                if drop_by_ratio or drop_by_unc:
                    reason = []
                    if drop_by_ratio:
                        reason.append(f"ratio={valid_face_ratio:.4f}<min={min_ratio:.4f}")
                    if drop_by_unc:
                        reason.append(f"avg_unc={avg_uncertainty:.4f}>max={max_uncertain:.4f}")
                    print(f"[Prepare] 视角 {view_id} 被筛除: {', '.join(reason)}")
                    continue
            view_data[view_id] = {
                "R": R_tensor,
                "T": T_tensor,
                "focal_x": focal_x,
                "focal_y": focal_y,
                "cx": cx,
                "cy": cy,
                "cameras_p3d": cameras_p3d,
                "mask": mask_tensor.bool(),
                "cam_xy": cam_xy_tensor,
                "pts_face": pts_face_all,
                "pts_bw": pts_bw_all,
                "seg_mask_list": seg_mask_list,
                "seg_gt": build_seg_id_map(seg_mask_list).to(self.device),
                "img_size": (resized_h, resized_w),
                "orig_size": (orig_h, orig_w),
                "cameras_orig": cameras_orig,
                "cam_base_name": f"{subject:04d}_{view_id:02d}",
                "base_cam": base_view_id,
                "group_idx": group_idx,
                "cycle_size": cycle_size,
                # 附加：保存输入UV，供 GT-UV 可视化（必须存在）
                "uv_input": both[..., :2].astype(np.uint8),
                # 像素权重
                "pixel_weight": pixel_weight_tensor,
            }

        # 若筛选后无有效视角，返回 None 以便上层跳过
        if len(view_data) == 0:
            print(f"[Prepare] 主体 {subject} 所有视角被筛除，跳过")
            return None

        return view_data

    # ----------------- 优化 -----------------
    def optimize_subject(self, subject_id: int, view_data):
        """迁移自旧脚本的优化流程，返回优化后顶点和视图数据。"""

        cfg = self.cfg
        # 身体使用 VPoser 先验（当 BODY_PRIOR="vposer" 时）
        USE_VPOSER = (self._body_prior == "vposer")

        # ---------- 相机可学习参数（但脚本设计为固定） ----------
        camera_params = {}
        for vid, vdata in view_data.items():
            rot_aa = matrix_to_axis_angle(vdata["R"].unsqueeze(0))[0]
            camera_params[vid] = {
                "rot": rot_aa.clone().detach(),
                "trans": vdata["T"].clone().detach(),
            }
            vdata["rot_param"] = camera_params[vid]["rot"]
            vdata["trans_param"] = camera_params[vid]["trans"]

        # ---------------- 可学习主体参数 ----------------
        device = self.device
        # 手部参数维度：使用 PCA 时为 HAND_PCA_COMPS，否则为 45 维轴角
        hand_pca_dim = self._hand_pca_comps

        pose_params = {
            # 直接使用 axis-angle 作为可学习的 global_orient，避免 6D → axis-angle 映射导致的大角度跳变
            "global_orient": torch.nn.Parameter(torch.zeros(3, device=device)),
            "pelvis": torch.nn.Parameter(torch.zeros(3, device=device)),   #6
            "body1": torch.nn.Parameter(torch.zeros(15, device=device)),  #21
            "foot0": torch.nn.Parameter(torch.zeros(6, device=device)),  #27
            "body12": torch.nn.Parameter(torch.zeros(3, device=device)),  #30
            "foot": torch.nn.Parameter(torch.zeros(6, device=device)),  #36
            "body2": torch.nn.Parameter(torch.zeros(24, device=device)),  #60
            "hand0": torch.nn.Parameter(torch.zeros(6, device=device)), #66
            "jaw": torch.nn.Parameter(torch.zeros(3, device=device)),
            "leye": torch.nn.Parameter(torch.zeros(3, device=device)),
            "reye": torch.nn.Parameter(torch.zeros(3, device=device)),
            "left_hand": torch.nn.Parameter(torch.zeros(hand_pca_dim, device=device)),
            "right_hand": torch.nn.Parameter(torch.zeros(hand_pca_dim, device=device)),
        }
        betas = torch.nn.Parameter(torch.zeros(10, device=device))
        trans = torch.nn.Parameter(torch.zeros(3, device=device))
        scale = torch.nn.Parameter(torch.tensor([0.541324854612918], device=device))

        # ---------- [Step 0-H · EXP-6] 注入 SMPLest-X 回归 betas + body_pose 作为初值 ----------
        # 仅注入 shape + body pose，不动 global_orient/trans/scale（坐标系不一致风险）。
        # body_pose flat 顺序与 flat_poses 对齐：[pelvis(3), body1(15), foot0(6), body12(3), foot(6), body2(24), hand0(6)] = 63
        if getattr(self.cfg, "INIT_FROM_SMPLESTX", False):
            init_dir = getattr(self.cfg, "SMPLESTX_INIT_DIR", None)
            if not init_dir:
                # SMPLest-X v4 inference 直接保存的回归参数（非反算）
                init_dir = os.path.join(
                    DATASET_ROOT, "evaluate", self.cfg.data, "smplestx", "params"
                )
            init_view = int(getattr(self.cfg, "SMPLESTX_INIT_VIEW", 0))
            init_path = os.path.join(init_dir, f"{subject_id:04d}_{init_view:02d}.npz")
            if not os.path.exists(init_path):
                print(f"[Init][SMPLest-X] 找不到 {init_path}，回退 zero-init")
            else:
                with np.load(init_path) as nz:
                    init_betas = torch.from_numpy(nz["betas"].astype(np.float32)).to(device)
                    init_body = torch.from_numpy(nz["body_pose"].astype(np.float32)).to(device)
                    inject_hands = bool(getattr(self.cfg, "SMPLESTX_INIT_INJECT_HANDS", False))
                    init_lhand = torch.from_numpy(nz["left_hand_pose"].astype(np.float32)).to(device) if inject_hands and "left_hand_pose" in nz else None
                    init_rhand = torch.from_numpy(nz["right_hand_pose"].astype(np.float32)).to(device) if inject_hands and "right_hand_pose" in nz else None
                body_mode = str(getattr(self.cfg, "SMPLESTX_INIT_BODY_POSE_MODE", "warmstart")).lower()
                # betas 始终注入（shape 不依赖视角）
                with torch.no_grad():
                    betas.data.copy_(init_betas)
                    if body_mode in ("warmstart", "full"):
                        # body_pose 切片注入到 pose_params dict
                        pose_params["pelvis"].data.copy_(init_body[0:3])
                        pose_params["body1"].data.copy_(init_body[3:18])
                        pose_params["foot0"].data.copy_(init_body[18:24])
                        pose_params["body12"].data.copy_(init_body[24:27])
                        pose_params["foot"].data.copy_(init_body[27:33])
                        pose_params["body2"].data.copy_(init_body[33:57])
                        pose_params["hand0"].data.copy_(init_body[57:63])
                    elif inject_hands:
                        pose_params["hand0"].data.copy_(init_body[57:63])
                    if inject_hands and init_lhand is not None and init_rhand is not None:
                        lh_dim = pose_params["left_hand"].numel()
                        rh_dim = pose_params["right_hand"].numel()
                        if lh_dim == 45 and rh_dim == 45:
                            pose_params["left_hand"].data.copy_(init_lhand)
                            pose_params["right_hand"].data.copy_(init_rhand)
                        else:
                            print(f"[Init][SMPLest-X] hand dim mismatch (lhand={lh_dim} rhand={rh_dim}) → 跳过 hand 注入")
                if body_mode == "off":
                    inj_str = "betas-only"
                elif body_mode == "warmstart":
                    inj_str = "betas + body_pose(warmstart)"
                elif body_mode == "full":
                    inj_str = "betas + body_pose(full)"
                else:
                    inj_str = f"betas (unknown body_pose mode={body_mode})"
                if inject_hands and init_lhand is not None:
                    inj_str += " + hands"
                print(f"[Init][SMPLest-X] subject={subject_id:04d} view={init_view:02d} 注入 {inj_str}")

        # 手部独立拟合：单独的 betas / scale / trans，并在顶点级与身体 mesh 合并
        # 注意：在真正进入 Hands-* 阶段前，不启用分割前向（保持单一人体 mesh）。
        use_split_hand = bool(getattr(self.cfg, "SPLIT_HAND_FIT", False))
        if use_split_hand:
            betas_hand = torch.nn.Parameter(torch.zeros(10, device=device))
            trans_hand = torch.nn.Parameter(torch.zeros(3, device=device))
            # 初始时与身体 scale 相同，后续在 Hands-* 阶段单独更新
            scale_hand = torch.nn.Parameter(scale.detach().clone())
        else:
            # 不开启拆分时，手部与身体共用同一组参数（不额外引入可学习变量）
            betas_hand = betas
            trans_hand = trans
            scale_hand = scale

        # 是否在前向中实际启用“手部/身体分割”（仅在进入 Hands-* 阶段后置 True）
        split_hand_active = False

        # VPoser latent（可选）
        if USE_VPOSER and getattr(self, "vp_model", None) is not None:
            vp_latent = torch.nn.Parameter(torch.zeros(self.vp_model.latentD, device=device))
        else:
            vp_latent = None

        def flat_poses(d):
            keys = (
                # global_orient 直接为 axis-angle
                "pelvis",
                "body1",
                "foot0",
                "body12",
                "foot",
                "body2",
                "hand0",
                "jaw",
                "leye",
                "reye",
                "left_hand",
                "right_hand",
            )
            # 直接拼接 global_orient（axis-angle）与其余 body/hand poses
            pose_list = [d["global_orient"]] + [d[k] for k in keys]
            return torch.cat(pose_list)

        # ---------------- 优化阶段设置 ----------------
        view_ids = list(view_data.keys())
        body_views = sorted([vid for vid, v in view_data.items() if v.get('group_idx', int(vid)//max(1, int(v.get('cycle_size', 4)))) == 0])

        hand_views = [vid for vid, v in view_data.items() if v.get('group_idx', int(vid)//max(1, int(v.get('cycle_size', 4)))) in (1, 2)]

        # ---------------- 优化阶段设置 ----------------
        stages = [
            {
                "name": "global pose",
                "active_segments": [1, 2, 3, 8],
                "active_views": body_views,
                "keys": ["trans", "global_orient"], "iters": 50, "lr": 0.1, "optimizer":"adam",
                "repeat_max": 20, "repeat_improve": 0.03, "pixel_sample_ratio": 0.5,
                # 是否在该 stage 启用先验 / 关节角度约束
                "use_dposer": False,
                "use_joint_limit": False,
            },
            {
                "name": "scale",
                "active_segments": [1, 2],
                "active_views": body_views if len(body_views) >= 3 else body_views, "optimizer":"adam",
                "keys": ["scale", "trans", "global_orient"],
                # "keys": ["trans", "global_orient"],
                "iters": 50, "lr": 0.1,
                "repeat_max": 20, "repeat_improve": 0.1, "pixel_sample_ratio": 0.5,
                "use_dposer": False,
                "use_joint_limit": False,
            },
            {
                "name": "full pose",
                "active_segments": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "active_views": body_views,
                # 启用脚部优化：在除 global pose/scale/Hands-* 之外的阶段，将 foot0/foot 加入可学习参数
                "keys": ["scale", "trans", "global_orient", "pelvis", "body1", "foot0", "body12", "foot", "body2", "vp_latent"],
                # "keys": ["trans", "global_orient", "pelvis", "body1", "body12", "body2", "vp_latent"],
                "iters": 50, "lr": 0.01,
                "repeat_max": 20, "repeat_improve": 0.10, "pixel_sample_ratio": 1.0,
                "use_dposer": True,
                "use_joint_limit": True,
            },
            {
                "name": "Fine-tune All",
                "active_segments": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "active_views": body_views,
                # 启用脚部优化：在除 global pose/scale/Hands-* 之外的阶段，将 foot0/foot 加入可学习参数
                "keys": ["scale", "trans", "global_orient", "pelvis", "body1", "foot0", "body12", "foot", "body2", "betas", "hand0", "vp_latent"],
                # "keys": ["trans", "global_orient", "pelvis", "body1", "body12", "body2", "betas", "hand0", "vp_latent"],
                "iters": 50, "lr": 0.005,
                # [Step 0-G] revert thr 0.03→0.01 — Codex finding: v0f stops here at 2-3% drop, missing 2-3 polish repeats → body PA-MPJPE +5%
                "repeat_max": 20, "repeat_improve": 0.01, "pixel_sample_ratio": 1.0,
                "use_dposer": True,
                "use_joint_limit": True,
            },
            {
                "name": "Hands-A: palms optimize hand0+vp",
                "active_segments": [1, 2, 3, 4, 6, 7, 8, 10, 12, 18],
                "active_views": hand_views,
                "keys": ["hand0", "scale","betas"],
                "iters": 50, "lr": 0.01,
                "repeat_max": 20, "repeat_improve": 0.1, "pixel_sample_ratio": 1.0,
                "use_dposer": True,
                "use_joint_limit": True,
            },
            {
                "name": "Hands-C: fingers refine",
                "active_segments": list(LEFT_HAND_SEG_IDS + RIGHT_HAND_SEG_IDS),
                "active_views": hand_views,
                "keys": ["hand0", "left_hand", "right_hand", "scale", "trans", "global_orient", "betas"],
                # "keys": ["hand0", "left_hand", "right_hand", "trans", "global_orient", "betas"],
                "iters": 50, "lr": 0.005,
                "repeat_max": 20, "repeat_improve": 0.05, "pixel_sample_ratio": 1.0,
                "use_dposer": True,
                "use_joint_limit": True,
            },
        ]

        # 配置跳过手部拟合阶段（Hands-A / Hands-C）
        if getattr(self.cfg, 'SKIP_HAND_FIT', False) or getattr(self.cfg, "SMPLESTX_FIX_HANDS", False):
            stages = [st for st in stages if 'Hand' not in st.get('name', '')]
        # 仅跳过 Hands-C: fingers refine
        elif getattr(self.cfg, "SKIP_HANDS_C", False):
            stages = [st for st in stages if st.get("name", "") != "Hands-C: fingers refine"]

        # 若启用 VPoser，则不再直接优化 body pose 的 axis-angle，只优化 vp_latent；
        # 若未启用 VPoser，则从 keys 中移除 vp_latent。
        body_pose_keys = ["pelvis", "body1", "foot0", "body12", "foot", "body2", "hand0"]
        for stg in stages:
            keys = stg["keys"]
            if USE_VPOSER:
                keys = [k for k in keys if (k not in body_pose_keys)]
            else:
                keys = [k for k in keys if k != "vp_latent"]
            stg["keys"] = keys

        # 若启用了 split hand fit，则在（Hands-* 之后，或跳过手部阶段时）追加一个
        # “Fine-tune Body After” 阶段：此阶段仅优化身体参数，并固定手部。
        #
        # 注意：SKIP_HAND_FIT 只应跳过 Hands-* 阶段本身，不应连带跳过 Fine-tune Body After。
        if use_split_hand:
            if getattr(self.cfg, 'SKIP_HAND_FIT', False):
                stages.append(
                            {
                                "name": "Fine-tune Body After",
                                "active_segments": [1, 2, 3, 4, 6, 7, 8, 10],
                                "active_views": body_views,
                                # 不包含 left_hand / right_hand（手指），避免破坏手部细化结果；保留 hand0（手根/手腕）用于整体对齐微调
                                # 启用脚部优化：在除 global pose/scale/Hands-* 之外的阶段，将 foot0/foot 加入可学习参数
                                "keys": ["scale", "trans", "global_orient", "pelvis", "body1", "foot0", "body12", "foot", "body2", "betas", "vp_latent"],
                                # "keys": ["hand0", "trans", "global_orient", "pelvis", "body1", "body12", "body2", "betas", "vp_latent"],
                                "iters": 50,
                                "lr": 0.01,
                                "repeat_max": 20,
                                # [Step 0-G] revert thr 0.03→0.01 (final body settle stage)
                                "repeat_improve": 0.01,
                                "pixel_sample_ratio": 1.0,
                                "use_dposer": True,
                                "use_joint_limit": True,
                            }
                        )
            else:
                stages.append(
                    {
                        "name": "Fine-tune Body After",
                        "active_segments": [1, 2, 3, 4, 6, 7, 8, 10]+list(LEFT_HAND_SEG_IDS + RIGHT_HAND_SEG_IDS),
                        "active_views": body_views+hand_views,
                        # 不包含 left_hand / right_hand（手指），避免破坏手部细化结果；保留 hand0（手根/手腕）用于整体对齐微调
                        # 启用脚部优化：在除 global pose/scale/Hands-* 之外的阶段，将 foot0/foot 加入可学习参数
                        "keys": ["hand0", "scale", "trans", "global_orient", "pelvis", "body1", "foot0", "body12", "foot", "body2", "betas", "vp_latent"],
                        # "keys": ["hand0", "trans", "global_orient", "pelvis", "body1", "body12", "body2", "betas", "vp_latent"],
                        "iters": 50,
                        "lr": 0.01,
                        "repeat_max": 20,
                        # [Step 0-G] revert thr 0.03→0.01 (final body settle stage)
                        "repeat_improve": 0.01,
                        "pixel_sample_ratio": 1.0,
                        "use_dposer": True,
                        "use_joint_limit": True,
                    }
                )

        # 若使用 Adam，则只对 full pose 和 Fine-tune All 做调整
        global_opt_name = str(getattr(self.cfg, "OPTIMIZER", "lbfgs")).lower()
        # for st in stages:
        #     name_l = str(st.get("name", "")).lower()
        #     if name_l not in ("full pose", "fine-tune all"):#,"Hands-A: palms optimize hand0+vp","Hands-C: fingers refine","Fine-tune Body After"):
        #         continue  # 跳过其他所有阶段
        #     stage_opt = str(st.get("optimizer", global_opt_name)).lower()
        #     if stage_opt == "adam":
        #         base_iters = int(st.get("iters", 10))
        #         base_lr = float(st.get("lr", 0.1))
        #         st["iters"] = base_iters * 10
        #         st["lr"] = base_lr / 10.0
        # print(stages)
        # 若固定 SMPLest-X hands，则手腕/手指动作参数都不再优化。
        if getattr(self.cfg, "SMPLESTX_FIX_HANDS", False):
            fixed_hand_keys = {"hand0", "left_hand", "right_hand"}
            for stg in stages:
                stg["keys"] = [k for k in stg["keys"] if k not in fixed_hand_keys]

        # 若关闭 scale 优化，则从所有阶段的 keys 中移除 "scale"
        if not getattr(self.cfg, 'OPTIMIZE_SCALE', True):
            for stg in stages:
                stg["keys"] = [k for k in stg["keys"] if k != "scale"]

        smplx_neutral = self.smplx_model_neutral
        smplx_faces_tensor = self.smplx_faces_tensor

        print("[Optimize] 开始主体", subject_id)

        joint_limit_weight = float(getattr(self.cfg, "JOINT_LIMIT_WEIGHT", 0.0))
        dposer_weight = float(getattr(self.cfg, "DPOSER_WEIGHT", 0.0))

        def _joint_limit_penalty(poses_tensor: torch.Tensor) -> torch.Tensor:
            """
            基于 pose_flat 的关节角度限制（使用上面 JOINT_ANGLE_LIMITS）。
            实际是否启用，以及权重大小，由外层的 joint_limit_weight 和
            每个 stage 的 use_joint_limit 一起控制。
            """
            if joint_limit_weight <= 0:
                return poses_tensor.new_tensor(0.0)
            return joint_limit_weight * compute_pose_joint_limit_loss(poses_tensor)

        # ---- learning rate 相关工具函数（参考 difflr 版本，多组件可调 LR） ----
        def _extract_default_lr(lr_setting):
            """
            从 stage["lr"] 中抽取一个“默认 lr”：
            - 若是标量，直接返回；
            - 若是 dict，则优先找 "default" / "*" / "__all__" 这类 key，
              否则拿第一个非 None 的值；
            - 兜底为 1e-2。
            """
            if isinstance(lr_setting, dict):
                for alias in ("default", "*", "__all__"):
                    if alias in lr_setting and lr_setting[alias] is not None:
                        return float(lr_setting[alias])
                for val in lr_setting.values():
                    if val is not None:
                        return float(val)
                return 1e-2
            if isinstance(lr_setting, (int, float)):
                return float(lr_setting)
            return 1e-2

        def _resolve_param_lr(key, stage_lr_setting, default_lr, per_param_override, cfg_mult, stage_mult):
            """
            计算某个参数 key 在当前 stage 下最终使用的 lr：
            1) 以 default_lr 为基准
            2) 若 stage["lr"] 是 dict，优先使用里面针对该 key 的值
            3) 再看 stage["lr_by_param"] 中是否有覆盖
            4) 最后乘以 Config.PARAM_LR_MULTIPLIERS 和 stage["lr_multipliers"] 中的倍率
            """
            lr_value = default_lr
            # stage 级别的 lr dict （例如 {"default": 1e-2, "betas": 5e-3}）
            if isinstance(stage_lr_setting, dict):
                if key in stage_lr_setting and stage_lr_setting[key] is not None:
                    lr_value = float(stage_lr_setting[key])
                elif "*" in stage_lr_setting and stage_lr_setting["*"] is not None:
                    lr_value = float(stage_lr_setting["*"])
            # stage 内针对 key 的显式覆盖
            if isinstance(per_param_override, dict):
                if key in per_param_override and per_param_override[key] is not None:
                    lr_value = float(per_param_override[key])
                elif "*" in per_param_override and per_param_override["*"] is not None:
                    lr_value = float(per_param_override["*"])
            # 全局 Config 中的 multiplier
            lr_value *= cfg_mult.get(key, 1.0)
            # 当前 stage 内的 multiplier
            lr_value *= stage_mult.get(key, 1.0)
            return max(lr_value, 1e-12)

        def _build_param_groups(stage_cfg, param_entries):
            """
            根据当前 stage 的配置与参数列表，构造 optimizer 所需的 param_groups：
            - 支持 stage["lr"] 为标量或 dict
            - 支持 stage["lr_by_param"] 指定某些 key 的专属 lr
            - 支持 Config.PARAM_LR_MULTIPLIERS 与 stage["lr_multipliers"]
            """
            stage_lr_setting = stage_cfg.get("lr", 1e-2)
            default_lr = _extract_default_lr(stage_lr_setting)
            per_param_override = stage_cfg.get("lr_by_param", {})
            stage_mult = stage_cfg.get("lr_multipliers", {})
            cfg_mult = getattr(self.cfg, "PARAM_LR_MULTIPLIERS", {}) or {}

            if not isinstance(per_param_override, dict):
                per_param_override = {}
            if not isinstance(stage_mult, dict):
                stage_mult = {}

            # 若是手部相关 stage（名字中包含 "hand"），自动对手部参数乘以 Config.HAND_LR_MULTIPLIER
            name_l = str(stage_cfg.get("name", "")).lower()
            if "hand" in name_l:
                hand_stage_mult = float(getattr(self.cfg, "HAND_LR_MULTIPLIER", 1.0))
                if hand_stage_mult != 1.0:
                    for hand_key in ("left_hand", "right_hand", "hand0"):
                        stage_mult[hand_key] = stage_mult.get(hand_key, 1.0) * hand_stage_mult

            param_groups = []
            lr_snapshot = {}
            for param_tensor, key in param_entries:
                lr_value = _resolve_param_lr(
                    key,
                    stage_lr_setting,
                    default_lr,
                    per_param_override,
                    cfg_mult,
                    stage_mult,
                )
                param_groups.append({"params": [param_tensor], "lr": lr_value})
                lr_snapshot.setdefault(key, lr_value)

            if stage_cfg.get("log_lr", False):
                print(f"[{stage_cfg.get('name', 'stage')}] lr map: {lr_snapshot}")
            return param_groups, default_lr

        # ---- 像素子采样工具：根据比例在有效像素中采样一部分 ----
        def _subsample_mask(mask_bool: torch.Tensor,
                            ratio: float,
                            weight_map: torch.Tensor = None) -> torch.Tensor:
            """
            在 True 的位置中按比例采样，返回新的 bool mask。
            - 默认使用 weight_map 中的权重从高到低选取前 k 个像素（k = ratio * 有效像素数）；
            - 若未提供 weight_map，则退化为旧逻辑：在有效像素中随机采样。
            """
            if ratio is None or ratio >= 1.0:
                return mask_bool
            flat = mask_bool.view(-1)
            idx = flat.nonzero(as_tuple=False).squeeze(1)
            n = int(idx.numel())
            if n <= 1:
                return mask_bool

            k = max(1, int(n * ratio))

            if weight_map is None:
                # 兼容旧逻辑：无权重信息时仍随机采样
                perm = torch.randperm(n, device=mask_bool.device)[:k]
                chosen = idx[perm]
            else:
                # 使用当前像素的最终权重（例如 (1-uv_uncertainty)*(1-seg_uncertainty)）
                w_flat = weight_map.view(-1)[idx]
                # 权重从高到低排序，取前 k 个
                _, order = torch.sort(w_flat, descending=True)
                chosen = idx[order[:k]]

            new_flat = torch.zeros_like(flat, dtype=torch.bool)
            new_flat[chosen] = True
            return new_flat.view_as(mask_bool)

        # ---- 统一检查当前优化参数是否为有限值（NaN/Inf 调试用） ----
        def _assert_finite_params(tag: str):
            to_check = {
                "betas": betas,
                "trans": trans,
                "scale": scale,
            }
            for k, v in pose_params.items():
                to_check[k] = v
            if USE_VPOSER and (vp_latent is not None):
                to_check["vp_latent"] = vp_latent
            if use_split_hand:
                to_check["betas_hand"] = betas_hand
                to_check["trans_hand"] = trans_hand
                to_check["scale_hand"] = scale_hand
            for name, tensor in to_check.items():
                if not torch.isfinite(tensor).all():
                    raise RuntimeError(f"[NaN] {tag}: 参数 {name} 出现 NaN/Inf")

        # ---- 参数快照 / 恢复：供 LBFGS NaN 回退与 global pose 组合搜索共用 ----
        def _snapshot_params():
            pose_snap = {k: p.detach().clone() for k, p in pose_params.items()}
            snap = {
                "pose": pose_snap,
                "betas": betas.detach().clone(),
                "trans": trans.detach().clone(),
                "scale": scale.detach().clone(),
            }
            if USE_VPOSER and (vp_latent is not None):
                snap["vp_latent"] = vp_latent.detach().clone()
            if use_split_hand:
                snap["betas_hand"] = betas_hand.detach().clone()
                snap["trans_hand"] = trans_hand.detach().clone()
                snap["scale_hand"] = scale_hand.detach().clone()
            return snap

        def _restore_params(snap):
            for k, t in snap["pose"].items():
                pose_params[k].data.copy_(t)
            betas.data.copy_(snap["betas"])
            trans.data.copy_(snap["trans"])
            scale.data.copy_(snap["scale"])
            if USE_VPOSER and (vp_latent is not None) and ("vp_latent" in snap):
                vp_latent.data.copy_(snap["vp_latent"])
            if use_split_hand and ("betas_hand" in snap):
                betas_hand.data.copy_(snap["betas_hand"])
                trans_hand.data.copy_(snap["trans_hand"])
                scale_hand.data.copy_(snap["scale_hand"])

        _cached_body_verts = [None]  # mutable container for closure

        def _invalidate_body_cache():
            _cached_body_verts[0] = None

        # ---- 统一的 verts 前向：根据当前 pose / betas / trans / scale 计算顶点
        # 在开启 SPLIT_HAND_FIT 时，会分别前向身体与手部，再在顶点级做合并。
        def _forward_verts_from_pose(poses_flat: torch.Tensor) -> torch.Tensor:
            use_cache = (
                self.cfg.CACHE_BODY_VERTS_IN_HAND
                and split_hand_active
                and _cached_body_verts[0] is not None
            )
            if use_cache:
                verts_body_now = _cached_body_verts[0]
            else:
                verts_body_now = get_smplx_vertices(
                    poses_flat,
                    betas,
                    trans,
                    gender="neutral",
                    scale_param=(F.softplus(scale) + 1e-6),
                    smplx_model_neutral=smplx_neutral,
                    smplx_model_male=self.smplx_model_male,
                    smplx_model_female=self.smplx_model_female,
                    use_hand_pca=self._use_hand_pca,
                    hand_pca_comps=self._hand_pca_comps,
                )

            if (not use_split_hand) or (not split_hand_active):
                return verts_body_now

            # 以“掌心点”为中心做局部缩放：
            # - 先用身体的物理 scale 得到一份 hand verts（保证与身体同尺度基准）
            # - 再按 ratio = scale_hand/scale_body 对左右手顶点分别绕掌心中心缩放
            # 这样 scale_hand 的效果是“局部放大/缩小手”，而不是绕世界原点整体缩放。
            scale_body_phys = (F.softplus(scale) + 1e-6)
            scale_hand_phys = (F.softplus(scale_hand) + 1e-6)
            ratio = (scale_hand_phys / scale_body_phys).clamp(min=1e-6, max=1e6)

            verts_hand_base = get_smplx_vertices(
                poses_flat,
                betas_hand,
                trans_hand,
                gender="neutral",
                scale_param=scale_body_phys,
                smplx_model_neutral=smplx_neutral,
                smplx_model_male=self.smplx_model_male,
                smplx_model_female=self.smplx_model_female,
                use_hand_pca=self._use_hand_pca,
                hand_pca_comps=self._hand_pca_comps,
            )

            verts_hand_now = verts_hand_base
            # 若有左右手顶点掩码，则分别绕掌心中心缩放；否则退回到原来的全局缩放行为
            if hasattr(self, "left_hand_vertex_mask_global") and hasattr(self, "right_hand_vertex_mask_global"):
                vhn = verts_hand_base
                verts_hand_local = vhn.clone()
                l_mask = self.left_hand_vertex_mask_global.to(vhn.device)
                r_mask = self.right_hand_vertex_mask_global.to(vhn.device)
                lp_mask = getattr(self, "left_palm_vertex_mask_global", None)
                rp_mask = getattr(self, "right_palm_vertex_mask_global", None)
                lp_mask = lp_mask.to(vhn.device) if torch.is_tensor(lp_mask) else None
                rp_mask = rp_mask.to(vhn.device) if torch.is_tensor(rp_mask) else None

                def _center_from_masks(hand_mask: torch.Tensor, palm_mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
                    if palm_mask is not None and palm_mask.any():
                        return vhn[palm_mask].mean(dim=0)
                    if hand_mask is not None and hand_mask.any():
                        return vhn[hand_mask].mean(dim=0)
                    return None

                c_l = _center_from_masks(l_mask, lp_mask)
                c_r = _center_from_masks(r_mask, rp_mask)

                if c_l is not None and l_mask.any():
                    verts_hand_local[l_mask] = c_l + (vhn[l_mask] - c_l) * ratio
                if c_r is not None and r_mask.any():
                    verts_hand_local[r_mask] = c_r + (vhn[r_mask] - c_r) * ratio

                verts_hand_now = verts_hand_local
            else:
                # 兼容：若缺少左右手掩码，仍按旧方式全局缩放 hand verts
                verts_hand_now = get_smplx_vertices(
                    poses_flat,
                    betas_hand,
                    trans_hand,
                    gender="neutral",
                    scale_param=scale_hand_phys,
                    smplx_model_neutral=smplx_neutral,
                    smplx_model_male=self.smplx_model_male,
                    smplx_model_female=self.smplx_model_female,
                    use_hand_pca=self._use_hand_pca,
                    hand_pca_comps=self._hand_pca_comps,
                )

            # 若未构建手部顶点掩码，则退化为纯身体 mesh
            if not hasattr(self, "_merge_verts_hand_body"):
                return verts_body_now
            return self._merge_verts_hand_body(verts_body_now, verts_hand_now)

        # ---- 2D Procrustes 配准（仅平移+缩放，不旋转） ----
        def _procrustes_align_2d_no_rotation(
            pred: torch.Tensor,
            gt: torch.Tensor,
            weights: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            对 2D 预测点做平移+缩放配准（不旋转），使其尽可能对齐目标点。
            
            Args:
                pred: (N, 2) 预测的 2D 点
                gt: (N, 2) 目标的 2D 点
                weights: (N,) 可选的每点权重
                
            Returns:
                aligned: (N, 2) 配准后的预测点
            """
            if pred.shape[0] < 3:
                # 点数过少，无法可靠配准，直接返回原始预测
                return pred
            
            if weights is not None:
                # 加权质心
                w = weights.unsqueeze(-1)  # (N, 1)
                w_sum = w.sum().clamp(min=1e-8)
                mu_pred = (pred * w).sum(dim=0) / w_sum
                mu_gt = (gt * w).sum(dim=0) / w_sum
            else:
                mu_pred = pred.mean(dim=0)
                mu_gt = gt.mean(dim=0)
            
            # 中心化
            pred_c = pred - mu_pred
            gt_c = gt - mu_gt
            
            # 最优缩放：s = sum(gt_c · pred_c) / sum(pred_c · pred_c)
            if weights is not None:
                numerator = (gt_c * pred_c * w).sum()
                denominator = (pred_c * pred_c * w).sum()
            else:
                numerator = (gt_c * pred_c).sum()
                denominator = (pred_c * pred_c).sum()
            
            # 防止除零
            if denominator.abs() < 1e-8:
                return pred
            
            s = numerator / denominator
            # 限制缩放范围，避免极端值
            s = s.clamp(min=0.1, max=10.0)
            
            # 配准：先缩放再平移
            aligned = s * pred_c + mu_gt
            return aligned

        # ====================== 公共辅助函数（去重） ======================

        def _resolve_camera_rt(data, view_data_ref):
            """统一计算相机 R/T（含 cam_anchor 逻辑）。返回 (R_mat[3,3], T_vec[3])。"""
            group_idx_v = data.get('group_idx', int(data.get('_vid', 0)) // max(1, int(data.get('cycle_size', 4))))
            if group_idx_v in (1, 2) and ('cam_anchor' in data):
                anc_id = int(data['cam_anchor']['anchor_id'])
                anc = view_data_ref[anc_id]
                R_anc = axis_angle_to_matrix(anc['rot_param'])
                T_anc = anc['trans_param']
                R_mat = torch.matmul(data['cam_anchor']['R_rel'], R_anc)
                T_vec = torch.matmul(data['cam_anchor']['R_rel'], T_anc) + data['cam_anchor']['t_rel']
            else:
                R_mat = axis_angle_to_matrix(data['rot_param'])
                T_vec = data['trans_param']
            return R_mat, T_vec

        def _build_active_mask(data, active_segments, group_idx_v):
            """统一构造 seg mask & valid mask。返回 mask_smpl (H,W) bool。"""
            mask_smpl = torch.zeros_like(data["mask"], dtype=torch.bool)
            if group_idx_v in (1, 2):
                allowed = set(LEFT_HAND_SEG_IDS + RIGHT_HAND_SEG_IDS + (GRAY_HAND_SEG_ID,))
            else:
                allowed = set(BODY_SEG_IDS)
            for sid in active_segments:
                if sid in allowed and 0 <= sid < len(data["seg_mask_list"]):
                    mask_smpl |= data["seg_mask_list"][sid].bool()
            mask_valid = (data["pts_face"] >= 0) & (data["pts_bw"].abs().sum(dim=-1) > 0)
            mask_smpl &= mask_valid
            return mask_smpl

        def _compute_view_reprojection(faces_verts, data, mask_smpl, R_mat, T_vec,
                                       use_uncertainty_weight, use_procrustes=False):
            """
            统一的单视角投影 + loss 计算。
            返回 (loss_smpl, n_valid)。若无有效像素返回 (None, 0)。
            """
            faces_idx_raw = data["pts_face"]
            faces_idx = torch.clamp(faces_idx_raw, 0, smplx_faces_tensor.shape[0] - 1)

            faces_idx_valid = faces_idx[mask_smpl]
            bw_valid = data["pts_bw"][mask_smpl]
            verts_pix_valid = faces_verts[faces_idx_valid]
            smpl_points_valid = torch.einsum("pvc,pv->pc", verts_pix_valid, bw_valid)

            pts_cam_smpl = torch.matmul(smpl_points_valid, R_mat) + T_vec
            z_valid = pts_cam_smpl[..., 2] > 1e-6
            if z_valid.sum() == 0:
                return None, 0
            pts_cam_smpl = pts_cam_smpl[z_valid]
            z_smpl = pts_cam_smpl[..., 2]
            proj_u_smpl = data["cx"] - data["focal_x"] * pts_cam_smpl[..., 0] / z_smpl
            proj_v_smpl = data["cy"] - data["focal_y"] * pts_cam_smpl[..., 1] / z_smpl
            pred_xy_smpl = torch.stack([proj_u_smpl, proj_v_smpl], dim=-1)
            gt_xy_smpl = data["cam_xy"][mask_smpl][z_valid]

            if use_procrustes:
                w_proc = data["pixel_weight"][mask_smpl][z_valid] if (use_uncertainty_weight and "pixel_weight" in data) else None
                pred_xy_smpl = _procrustes_align_2d_no_rotation(pred_xy_smpl, gt_xy_smpl, weights=w_proc)

            if use_uncertainty_weight and ("pixel_weight" in data):
                w_pix = data["pixel_weight"][mask_smpl][z_valid].unsqueeze(-1)
                huber = F.smooth_l1_loss(pred_xy_smpl, gt_xy_smpl, reduction='none')
                denom = (w_pix.sum() * huber.shape[-1]).clamp(min=1e-8)
                loss_smpl = (huber * w_pix).sum() / denom
            else:
                loss_smpl = F.smooth_l1_loss(pred_xy_smpl, gt_xy_smpl)
            return loss_smpl, int(z_valid.sum().item())

        def _prepack_views(active_views, active_segments, pixel_sample_ratio):
            """
            预打包所有活跃视角的静态数据（mask, face_idx, bw, gt_xy, weights, camera intrinsics）。
            R/T 不在此打包（因为 cam_anchor 参数可能参与优化，每步会变）。
            返回 dict 或 None（若无有效像素）。
            """
            all_face_idx = []
            all_bw = []
            all_gt_xy = []
            all_view_id = []
            all_pixel_weight = []
            view_intrinsics = []
            view_id_map = {}
            has_uncertainty = self.cfg.USE_UNCERTAINTY_WEIGHT
            vid_list = list(active_views)

            for local_idx, vid in enumerate(vid_list):
                data = view_data[vid]
                group_idx_v = data.get('group_idx', int(vid) // max(1, int(data.get('cycle_size', 4))))
                mask_smpl = _build_active_mask(data, active_segments, group_idx_v)
                mask_smpl = _subsample_mask(mask_smpl, pixel_sample_ratio, data["pixel_weight"])
                n_pix = int(mask_smpl.sum().item())
                if n_pix == 0:
                    continue
                # 使用连续 packed 索引，避免中间视角被跳过后出现 view_id 越界。
                packed_view_idx = len(view_intrinsics)
                faces_idx_raw = data["pts_face"]
                faces_idx = torch.clamp(faces_idx_raw, 0, smplx_faces_tensor.shape[0] - 1)
                all_face_idx.append(faces_idx[mask_smpl])
                all_bw.append(data["pts_bw"][mask_smpl])
                all_gt_xy.append(data["cam_xy"][mask_smpl])
                all_view_id.append(torch.full((n_pix,), packed_view_idx, dtype=torch.long, device=device))
                if has_uncertainty and "pixel_weight" in data:
                    all_pixel_weight.append(data["pixel_weight"][mask_smpl])
                else:
                    all_pixel_weight.append(torch.ones(n_pix, device=device))
                view_intrinsics.append({
                    'vid': vid,
                    'group_idx_v': group_idx_v,
                    'focal_x': data["focal_x"], 'focal_y': data["focal_y"],
                    'cx': data["cx"], 'cy': data["cy"],
                    'view_w': self.cfg.HAND_VIEW_WEIGHT if (group_idx_v in (1, 2)) else 1.0,
                })
                view_id_map[packed_view_idx] = vid

            if not all_face_idx:
                return None
            return {
                'face_idx': torch.cat(all_face_idx, dim=0),
                'bw': torch.cat(all_bw, dim=0),
                'gt_xy': torch.cat(all_gt_xy, dim=0),
                'view_id': torch.cat(all_view_id, dim=0),
                'pixel_weight': torch.cat(all_pixel_weight, dim=0),
                'intrinsics': view_intrinsics,
                'vid_list': vid_list,
                'view_id_map': view_id_map,
                'has_uncertainty': has_uncertainty,
            }

        def _batch_view_projection(faces_verts, packed, view_data_ref):
            """
            batch 计算所有视角的投影 + loss。
            返回 (total_weighted_loss, weight_sum, view_smpl_losses_dict)。
            faces_verts: (F, 3, 3)
            """
            face_idx = packed['face_idx']
            bw = packed['bw']
            gt_xy = packed['gt_xy']
            view_id_t = packed['view_id']
            pixel_w = packed['pixel_weight']
            intrinsics = packed['intrinsics']
            has_unc = packed['has_uncertainty']

            if face_idx.numel() > 0:
                max_face_idx = int(face_idx.max().item())
                min_face_idx = int(face_idx.min().item())
                n_faces = int(faces_verts.shape[0])
                if (min_face_idx < 0) or (max_face_idx >= n_faces):
                    raise RuntimeError(
                        f"[batch_view_projection] face_idx 越界: min={min_face_idx}, max={max_face_idx}, n_faces={n_faces}"
                    )
            verts_pix = faces_verts[face_idx]
            smpl_pts = torch.einsum("pvc,pv->pc", verts_pix, bw)

            n_views = len(intrinsics)
            if n_views == 0:
                return torch.tensor(0.0, device=device), 0.0, {}
            if view_id_t.numel() > 0:
                max_view_id = int(view_id_t.max().item())
                min_view_id = int(view_id_t.min().item())
                if (min_view_id < 0) or (max_view_id >= n_views):
                    raise RuntimeError(
                        f"[batch_view_projection] view_id 越界: min={min_view_id}, max={max_view_id}, n_views={n_views}"
                    )
            R_list = []
            T_list = []
            for info in intrinsics:
                data_v = view_data_ref[info['vid']]
                R_mat, T_vec = _resolve_camera_rt(data_v, view_data_ref)
                R_list.append(R_mat)
                T_list.append(T_vec)
            R_stack = torch.stack(R_list, dim=0)
            T_stack = torch.stack(T_list, dim=0)

            R_per_pix = R_stack[view_id_t]
            T_per_pix = T_stack[view_id_t]
            pts_cam = torch.bmm(smpl_pts.unsqueeze(1), R_per_pix).squeeze(1) + T_per_pix

            z_vals = pts_cam[..., 2]
            z_ok = z_vals > 1e-6

            focal_x_all = torch.tensor([info['focal_x'] for info in intrinsics], device=device)
            focal_y_all = torch.tensor([info['focal_y'] for info in intrinsics], device=device)
            cx_all = torch.tensor([info['cx'] for info in intrinsics], device=device)
            cy_all = torch.tensor([info['cy'] for info in intrinsics], device=device)
            view_w_all = torch.tensor([info['view_w'] for info in intrinsics], device=device, dtype=torch.float32)

            fx_per_pix = focal_x_all[view_id_t]
            fy_per_pix = focal_y_all[view_id_t]
            cx_per_pix = cx_all[view_id_t]
            cy_per_pix = cy_all[view_id_t]
            vw_per_pix = view_w_all[view_id_t]

            pts_cam_ok = pts_cam[z_ok]
            z_ok_vals = pts_cam_ok[..., 2]
            proj_u = cx_per_pix[z_ok] - fx_per_pix[z_ok] * pts_cam_ok[..., 0] / z_ok_vals
            proj_v = cy_per_pix[z_ok] - fy_per_pix[z_ok] * pts_cam_ok[..., 1] / z_ok_vals
            pred_xy = torch.stack([proj_u, proj_v], dim=-1)
            gt_xy_ok = gt_xy[z_ok]
            vw_ok = vw_per_pix[z_ok]
            pw_ok = pixel_w[z_ok]
            vid_ok = view_id_t[z_ok]

            if pred_xy.shape[0] == 0:
                return torch.tensor(0.0, device=device), 0.0, {}

            view_smpl_losses = {}
            total_loss = torch.tensor(0.0, device=device)
            weight_sum = 0.0

            for local_idx, info in enumerate(intrinsics):
                pix_mask = (vid_ok == local_idx)
                if pix_mask.sum() == 0:
                    view_smpl_losses[info['vid']] = float('nan')
                    continue
                pred_v = pred_xy[pix_mask]
                gt_v = gt_xy_ok[pix_mask]
                w_v = info['view_w']
                if has_unc:
                    pw_v = pw_ok[pix_mask].unsqueeze(-1)
                    huber = F.smooth_l1_loss(pred_v, gt_v, reduction='none')
                    denom = (pw_v.sum() * huber.shape[-1]).clamp(min=1e-8)
                    loss_v = (huber * pw_v).sum() / denom
                else:
                    loss_v = F.smooth_l1_loss(pred_v, gt_v)
                total_loss = total_loss + loss_v * w_v
                weight_sum += w_v
                view_smpl_losses[info['vid']] = float(loss_v.detach().item())

            return total_loss, weight_sum, view_smpl_losses

        def _render_single_view_vis(verts_vis, data_v, view_data_ref, fg_dev):
            """
            统一的单视角可视化：光栅化 -> seg pred/gt + uv pred/gt。
            返回 (seg_pair, uv_pair) 两个 np.ndarray (H, 2W, 3)。
            """
            R_mat_vis, T_vec_vis = _resolve_camera_rt(data_v, view_data_ref)
            R_cur = R_mat_vis.unsqueeze(0)
            T_cur = T_vec_vis.unsqueeze(0)
            fx, fy, cx_v, cy_v = data_v['focal_x'], data_v['focal_y'], data_v['cx'], data_v['cy']
            h_img_vis, w_img_vis = data_v['img_size']
            cameras_vis = PerspectiveCameras(
                R=R_cur, T=T_cur,
                focal_length=torch.tensor([[fx, fy]], dtype=torch.float32, device=device),
                principal_point=torch.tensor([[cx_v, cy_v]], dtype=torch.float32, device=device),
                image_size=torch.tensor([[h_img_vis, w_img_vis]], dtype=torch.float32, device=device),
                in_ndc=False, device=device,
            )
            rasterizer_vis = MeshRasterizer(
                cameras=cameras_vis,
                raster_settings=RasterizationSettings(
                    image_size=data_v["img_size"], blur_radius=1e-10, bin_size=0,
                ),
            )
            tex_dummy = TexturesVertex(verts_features=torch.zeros_like(verts_vis)[None])
            mesh_tmp = Meshes(verts=[verts_vis], faces=[smplx_faces_tensor], textures=tex_dummy)
            frags = rasterizer_vis(mesh_tmp)
            pix_to_face = frags.pix_to_face[0, ..., 0]
            pred_seg = torch.zeros_like(pix_to_face, dtype=torch.long)
            valid_pix = pix_to_face >= 0
            group_idx_v = data_v.get('group_idx', 0)
            if valid_pix.any():
                if group_idx_v in (1, 2) and self.hand_face_group_global is not None and self.hand_face_mask_global is not None:
                    seg_map_dev = self.hand_face_group_global.to(device)
                    hand_mask_dev = self.hand_face_mask_global.to(device)
                    face_idx_valid = pix_to_face[valid_pix].long()
                    is_hand_valid = hand_mask_dev[face_idx_valid]
                    pred_vals = torch.where(
                        is_hand_valid,
                        seg_map_dev[face_idx_valid],
                        torch.full_like(face_idx_valid, GRAY_HAND_SEG_ID, dtype=torch.long)
                    )
                    pred_seg[valid_pix] = pred_vals
                else:
                    pred_seg[valid_pix] = fg_dev[pix_to_face[valid_pix]]
            pred_col = colorize_seg(pred_seg.cpu().numpy())
            gt_col = colorize_seg(data_v["seg_gt"].cpu().numpy())
            if (faces_uvs_tensor is None) or (uvs_new_tensor is None):
                raise RuntimeError("缺少 faces_uvs_tensor 或 uvs_new_tensor，无法可视化预测UV")
            if frags.bary_coords is None:
                raise RuntimeError("rasterizer 未返回 bary_coords，无法可视化预测UV")
            pix_bary = frags.bary_coords[0, ..., 0, :]
            h_img, w_img = pix_to_face.shape
            valid = valid_pix
            face_idx = pix_to_face[valid]
            if group_idx_v in (1, 2) and (self.hand_tri_uv_idx_global is not None) and (self.hand_uvs_new_tensor is not None):
                faces_uvs_dev = self.hand_tri_uv_idx_global.to(device)
                uvs_new_dev = self.hand_uvs_new_tensor.to(device)
            else:
                faces_uvs_dev = faces_uvs_tensor.to(device)
                uvs_new_dev = uvs_new_tensor.to(device)
            tri_uv_idx = faces_uvs_dev[face_idx]
            uv0 = uvs_new_dev[tri_uv_idx[:, 0]]
            uv1 = uvs_new_dev[tri_uv_idx[:, 1]]
            uv2 = uvs_new_dev[tri_uv_idx[:, 2]]
            w0 = pix_bary[..., 0][valid].unsqueeze(1)
            w1 = pix_bary[..., 1][valid].unsqueeze(1)
            w2 = pix_bary[..., 2][valid].unsqueeze(1)
            uv_interp = uv0 * w0 + uv1 * w1 + uv2 * w2
            uv_interp_uint8 = torch.clamp((uv_interp * 255.0).round(), 0, 255).to(torch.uint8)
            uv_pred_np = torch.zeros((h_img, w_img, 2), dtype=torch.uint8, device=device)
            uv_pred_np[valid] = uv_interp_uint8
            uv_pred_np = uv_pred_np.cpu().numpy()
            pred_uv_vis = colorize_uv(uv_pred_np)
            if "uv_input" not in data_v:
                raise RuntimeError("view_data 缺少 'uv_input'，无法可视化 GT UV")
            gt_uv_vis = colorize_uv(data_v["uv_input"])
            for arr in (pred_col, gt_col):
                arr[0, :, :] = 255; arr[-1, :, :] = 255; arr[:, 0, :] = 255; arr[:, -1, :] = 255
            if pred_uv_vis.shape[:2] != pred_col.shape[:2]:
                pred_uv_vis = cv2.resize(pred_uv_vis, (pred_col.shape[1], pred_col.shape[0]), interpolation=cv2.INTER_NEAREST)
            if gt_uv_vis.shape[:2] != pred_col.shape[:2]:
                gt_uv_vis = cv2.resize(gt_uv_vis, (pred_col.shape[1], pred_col.shape[0]), interpolation=cv2.INTER_NEAREST)
            pred_uv_vis_b = pred_uv_vis.copy(); gt_uv_vis_b = gt_uv_vis.copy()
            for arr in (pred_uv_vis_b, gt_uv_vis_b):
                arr[0, :, :] = 255; arr[-1, :, :] = 255; arr[:, 0, :] = 255; arr[:, -1, :] = 255
            uv_pair = np.hstack([pred_uv_vis_b, gt_uv_vis_b])
            seg_pair = np.hstack([pred_col, gt_col])
            return seg_pair, uv_pair

        def _build_tile_grid(tiles):
            """将多个 tile 图像拼接成网格。"""
            if not tiles:
                return None
            n = len(tiles)
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))
            tile_h = max(img.shape[0] for img in tiles)
            tile_w = max(img.shape[1] for img in tiles)
            grid_img = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
            for idx_t, img_arr in enumerate(tiles):
                r = idx_t // cols
                c = idx_t % cols
                h_img, w_img = img_arr.shape[:2]
                y0 = r * tile_h + (tile_h - h_img) // 2
                x0 = c * tile_w + (tile_w - w_img) // 2
                grid_img[y0:y0 + h_img, x0:x0 + w_img] = img_arr
            return grid_img

        def _run_vis_block(verts_vis, active_views_stage_sorted, view_data_ref, fg_dev_vis):
            """统一可视化流程：对所有活跃视角渲染 seg/uv，拼 grid 并 imshow。"""
            uv_tiles = []
            seg_tiles = []
            for v in active_views_stage_sorted:
                data_v = view_data_ref[v]
                seg_pair, uv_pair = _render_single_view_vis(verts_vis, data_v, view_data_ref, fg_dev_vis)
                for pair, bucket in ((uv_pair, uv_tiles), (seg_pair, seg_tiles)):
                    h_vis, w_vis = pair.shape[:2]
                    scale_vis = self.cfg.TARGET_SIZE / max(h_vis, w_vis)
                    new_w = int(round(w_vis * scale_vis))
                    new_h = int(round(h_vis * scale_vis))
                    bucket.append(cv2.resize(pair, (new_w, new_h), interpolation=cv2.INTER_NEAREST))
            grid_uv = _build_tile_grid(uv_tiles)
            if grid_uv is not None:
                cv2.imshow("uv_all", grid_uv)
            grid_seg = _build_tile_grid(seg_tiles)
            if grid_seg is not None:
                cv2.imshow("seg_all", grid_seg[..., [2, 1, 0]])
                cv2.waitKey(1)

        # ====================== 结束：公共辅助函数 ======================

        # ---- 评估函数：按当前参数与给定 active_segments/views 评估平均 loss ----
        def _compute_stage_avg_loss(active_segments_eval, active_views_eval, pixel_sample_ratio: float = 1.0, use_procrustes: bool = False):
            with torch.no_grad():
                poses_flat = flat_poses(pose_params)
                if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                    decoded = self.vp_model.decode(vp_latent.unsqueeze(0))  # type: ignore
                    poses_flat[3:66] = decoded["pose_body"].reshape(-1)
                verts_eval = _forward_verts_from_pose(poses_flat)
                faces_verts_eval = verts_eval[smplx_faces_tensor]
                total_loss = torch.tensor(0.0, device=device)
                weight_sum = 0.0

                for vid in active_views_eval:
                    data = view_data[vid]
                    group_idx_v = data.get('group_idx', int(vid) // max(1, int(data.get('cycle_size', 4))))
                    R_mat, T_vec = _resolve_camera_rt(data, view_data)
                    mask_smpl = _build_active_mask(data, active_segments_eval, group_idx_v)
                    mask_smpl = _subsample_mask(mask_smpl, pixel_sample_ratio, data["pixel_weight"])
                    if mask_smpl.sum() == 0:
                        continue
                    loss_smpl, n_valid = _compute_view_reprojection(
                        faces_verts_eval, data, mask_smpl, R_mat, T_vec,
                        self.cfg.USE_UNCERTAINTY_WEIGHT, use_procrustes=use_procrustes,
                    )
                    if loss_smpl is None:
                        continue
                    view_w = self.cfg.HAND_VIEW_WEIGHT if (group_idx_v in (1, 2)) else 1.0
                    total_loss += loss_smpl * view_w
                    weight_sum += view_w

                if weight_sum > 0:
                    return float((total_loss / weight_sum).item())
                return float('inf')

        def _filter_hand_views_by_residual(candidate_views, active_segments_eval):
            """Drop catastrophic hand views using the fitted body's reprojection residual."""
            if not getattr(self.cfg, "FILTER_HAND_VIEW_RESIDUALS", True):
                return list(candidate_views), []

            ratio = float(getattr(self.cfg, "HAND_VIEW_RESIDUAL_MAX_RATIO", 4.0))
            abs_floor = float(getattr(self.cfg, "HAND_VIEW_RESIDUAL_ABS_FLOOR", 50.0))
            min_keep = max(1, int(getattr(self.cfg, "HAND_VIEW_RESIDUAL_MIN_KEEP", 2)))
            losses_by_group = defaultdict(list)

            for vid in candidate_views:
                data = view_data[vid]
                group_idx_v = data.get(
                    'group_idx', int(vid) // max(1, int(data.get('cycle_size', 4)))
                )
                loss_v = _compute_stage_avg_loss(
                    active_segments_eval=active_segments_eval,
                    active_views_eval=[vid],
                    pixel_sample_ratio=1.0,
                )
                losses_by_group[int(group_idx_v)].append((vid, float(loss_v)))

            dropped = []
            for group_idx_v, group_losses in sorted(losses_by_group.items()):
                finite_losses = [loss for _, loss in group_losses if np.isfinite(loss)]
                if len(group_losses) <= min_keep or len(finite_losses) < min_keep:
                    print(f"[HandViewFilter] group={group_idx_v} 保留全部: {group_losses}")
                    continue

                median_loss = float(np.median(finite_losses))
                threshold = max(abs_floor, median_loss * ratio)
                outliers = sorted(
                    ((vid, loss) for vid, loss in group_losses
                     if (not np.isfinite(loss)) or loss > threshold),
                    key=lambda item: item[1],
                    reverse=True,
                )
                max_drop = max(0, len(group_losses) - min_keep)
                dropped.extend(vid for vid, _ in outliers[:max_drop])
                print(
                    f"[HandViewFilter] group={group_idx_v} losses={group_losses} "
                    f"median={median_loss:.5f} threshold={threshold:.5f} "
                    f"drop={[vid for vid, _ in outliers[:max_drop]]}"
                )

            dropped_set = set(dropped)
            kept = [vid for vid in candidate_views if vid not in dropped_set]
            return kept, dropped

        # ---- 单次执行指定 stage（提取自原有代码） ----
        def _run_stage_once(stage, active_segments, active_views_stage, optimizer_state=None):
            nonlocal split_hand_active
            iters = stage["iters"]
            pixel_sample_ratio = float(stage.get("pixel_sample_ratio", 1.0))

            # （param_tensor, key_name）列表，用于构造带独立 lr 的 param_groups
            param_entries: List[Tuple[torch.nn.Parameter, str]] = []

            def _register_param(key_name: str, param_tensor: torch.nn.Parameter):
                if param_tensor is None:
                    return
                param_tensor.requires_grad_(True)
                param_entries.append((param_tensor, key_name))

            # 当前 stage 名字（小写）
            name_l = str(stage.get("name", "")).lower()
            is_hands_stage = ("hands-" in name_l)
            freeze_body_in_hands = bool(getattr(self.cfg, "FREEZE_BODY_IN_HAND_STAGES", True))
            is_fine_tune_body_after = name_l.startswith("fine-tune body after")

            # 若在开启 split hand fit 时首次进入 Hands-* 阶段，用当前身体参数初始化手部参数，
            # 并开启 split_hand_active，使后续前向在手部使用独立 betas/scale/trans。
            if use_split_hand and is_hands_stage and (not split_hand_active):
                with torch.no_grad():
                    betas_hand.data.copy_(betas.data)
                    trans_hand.data.copy_(trans.data)
                    scale_hand.data.copy_(scale.data)
                split_hand_active = True

            if (self.cfg.CACHE_BODY_VERTS_IN_HAND
                    and is_hands_stage and freeze_body_in_hands
                    and use_split_hand and split_hand_active):
                with torch.no_grad():
                    poses_flat_cache = flat_poses(pose_params)
                    if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                        poses_flat_cache[3:66] = self.vp_model.decode(vp_latent.unsqueeze(0))["pose_body"].reshape(-1)
                    _cached_body_verts[0] = get_smplx_vertices(
                        poses_flat_cache, betas, trans,
                        gender="neutral",
                        scale_param=(F.softplus(scale) + 1e-6),
                        smplx_model_neutral=smplx_neutral,
                        smplx_model_male=self.smplx_model_male,
                        smplx_model_female=self.smplx_model_female,
                        use_hand_pca=self._use_hand_pca,
                        hand_pca_comps=self._hand_pca_comps,
                    ).detach()
                    print(f"[CACHE] body verts cached for hand stage: {stage.get('name', '')}")
            else:
                _invalidate_body_cache()

            for p in pose_params.values():
                p.requires_grad_(False)
            betas.requires_grad_(False)
            trans.requires_grad_(False)
            scale.requires_grad_(False)
            for k in stage["keys"]:
                # Hands-* 阶段：禁止优化会影响“身体整体”的共享参数（尤其是 global_orient）
                if is_hands_stage and freeze_body_in_hands:
                    if k == "global_orient":
                        continue
                    # 若未启用 split hand fit，则手部阶段也不允许动 body 的 trans/scale/betas
                    if (not use_split_hand) and (k in ("trans", "scale", "betas")):
                        continue
                # Fine-tune Body After：保持 Fine-tune All 的 body 形体/尺度不变
                if use_split_hand and is_fine_tune_body_after and (k in ("scale", "betas")):
                    continue
                if k in pose_params:
                    _register_param(k, pose_params[k])
                elif k == "betas":
                    # 在 Hands-* 阶段优先只优化手部 betas，身体 betas 保持上一阶段结果
                    if use_split_hand and is_hands_stage:
                        _register_param("betas_hand", betas_hand)
                    else:
                        _register_param(k, betas)
                elif k == "trans":
                    if use_split_hand and is_hands_stage:
                        _register_param("trans_hand", trans_hand)
                    else:
                        _register_param(k, trans)
                elif k == "scale":
                    if use_split_hand and is_hands_stage:
                        _register_param("scale_hand", scale_hand)
                    else:
                        _register_param(k, scale)
                elif k == "vp_latent" and vp_latent is not None:
                    _register_param("vp_latent", vp_latent)  # type: ignore

            optimize_camera = ('Hands-B' in stage['name'])
            if optimize_camera:
                def _compute_palm_loss(verts_now, data, palm_seg_id):
                    faces_verts_now = verts_now[smplx_faces_tensor]
                    R_mat = axis_angle_to_matrix(data["rot_param"]).detach()
                    T_vec = data["trans_param"].detach()
                    gidx = data.get('group_idx', 1)
                    mask_smpl = _build_active_mask(data, [palm_seg_id], gidx)
                    if mask_smpl.sum() == 0:
                        return float('inf')
                    loss_smpl, _ = _compute_view_reprojection(
                        faces_verts_now, data, mask_smpl, R_mat, T_vec,
                        use_uncertainty_weight=False,
                    )
                    if loss_smpl is None:
                        return float('inf')
                    return float(loss_smpl.detach().item())

                with torch.no_grad():
                    poses_flat_now = flat_poses(pose_params)
                verts_now = _forward_verts_from_pose(poses_flat_now)
                left_views_local = [vid for vid in hand_views if view_data[vid].get('group_idx', int(vid)//max(1, int(view_data[vid].get('cycle_size', 4)))) == 1]
                right_views_local = [vid for vid in hand_views if view_data[vid].get('group_idx', int(vid)//max(1, int(view_data[vid].get('cycle_size', 4)))) == 2]
                if left_views_local:
                    losses = [(vid, _compute_palm_loss(verts_now, view_data[vid], 12)) for vid in left_views_local]
                    best_left_anchor_local = min(losses, key=lambda x: x[1])[0]
                else:
                    best_left_anchor_local = None
                if right_views_local:
                    losses = [(vid, _compute_palm_loss(verts_now, view_data[vid], 18)) for vid in right_views_local]
                    best_right_anchor_local = min(losses, key=lambda x: x[1])[0]
                else:
                    best_right_anchor_local = None
                print(f"[Hands-B] anchors left={best_left_anchor_local} right={best_right_anchor_local}")

                def _setup_relations(anchor_vid, vids):
                    if anchor_vid is None:
                        return
                    R_a0 = axis_angle_to_matrix(view_data[anchor_vid]['rot_param'].detach())
                    T_a0 = view_data[anchor_vid]['trans_param'].detach()
                    for _vid in vids:
                        if _vid == anchor_vid:
                            view_data[_vid].pop('cam_anchor', None)
                            continue
                        vdat = view_data[_vid]
                        R_i0 = axis_angle_to_matrix(vdat['rot_param'].detach())
                        T_i0 = vdat['trans_param'].detach()
                        R_rel = torch.matmul(R_i0, R_a0.transpose(0, 1))
                        t_rel = T_i0 - torch.matmul(R_rel, T_a0)
                        vdat['cam_anchor'] = {
                            'anchor_id': anchor_vid,
                            'R_rel': R_rel,
                            't_rel': t_rel,
                        }

                _setup_relations(best_left_anchor_local, left_views_local)
                _setup_relations(best_right_anchor_local, right_views_local)

                for vid, v in view_data.items():
                    is_hand = v.get('group_idx', int(vid)//max(1, int(v.get('cycle_size', 4)))) in (1, 2)
                    if not is_hand:
                        v['rot_param'].requires_grad_(False)
                        v['trans_param'].requires_grad_(False)
                        continue
                    is_left_anchor = (best_left_anchor_local is not None and vid == best_left_anchor_local)
                    is_right_anchor = (best_right_anchor_local is not None and vid == best_right_anchor_local)
                    if is_left_anchor or is_right_anchor:
                        if not isinstance(v['rot_param'], torch.nn.Parameter):
                            v['rot_param'] = torch.nn.Parameter(v['rot_param'])
                        if not isinstance(v['trans_param'], torch.nn.Parameter):
                            v['trans_param'] = torch.nn.Parameter(v['trans_param'])
                        _register_param("camera_rot", v['rot_param'])
                        _register_param("camera_trans", v['trans_param'])
                    else:
                        v['rot_param'].requires_grad_(False)
                        v['trans_param'].requires_grad_(False)
            else:
                for v in view_data.values():
                    v['rot_param'].requires_grad_(False)
                    v['trans_param'].requires_grad_(False)

            if not param_entries:
                return

            # 基于 param_entries 构造带独立 lr 的 param_groups
            param_groups, default_lr = _build_param_groups(stage, param_entries)

            optimizer_name = str(stage.get('optimizer', getattr(self.cfg, 'OPTIMIZER', 'adam'))).lower()
            use_lbfgs = (optimizer_name == 'lbfgs')
            if use_lbfgs:
                optim_stage = torch.optim.LBFGS(
                    [pg["params"][0] for pg in param_groups],
                    lr=min({float(pg.get("lr", default_lr)) for pg in param_groups} or {default_lr}),
                    max_iter=5,
                    history_size=10,
                    line_search_fn='strong_wolfe',
                )
            else:
                optim_stage = torch.optim.Adam(param_groups, lr=default_lr)
            # 若提供历史状态，则尝试恢复以保留动量/自适应统计
            if optimizer_state is not None:
                try:
                    optim_stage.load_state_dict(optimizer_state)
                except Exception:
                    pass

            # 记录最近一次 step 的各项 loss 分量，便于打印
            last_smpl_loss = None
            last_dposer_prior = None
            last_mano_prior = None
            last_joint_limit = None
            last_weight_sum = 0.0
            last_view_smpl_losses = {}

            _use_batch_view = self.cfg.BATCH_VIEW_PROJECTION and len(active_views_stage) > 1
            _packed_views = None
            if _use_batch_view:
                _packed_views = _prepack_views(active_views_stage, active_segments, pixel_sample_ratio)
                if _packed_views is None:
                    _use_batch_view = False

            if use_lbfgs:
                backup_state = _snapshot_params()

                def _closure():
                    nonlocal last_smpl_loss, last_dposer_prior, last_joint_limit, last_weight_sum, last_view_smpl_losses, last_mano_prior
                    _assert_finite_params(f"{stage.get('name', '')} [LBFGS] before closure")
                    optim_stage.zero_grad()
                    poses_flat = flat_poses(pose_params)
                    if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                        decoded = self.vp_model.decode(vp_latent.unsqueeze(0))  # type: ignore
                        body_pose_aa = decoded["pose_body"].reshape(-1)
                        poses_flat[3:66] = body_pose_aa
                    verts = _forward_verts_from_pose(poses_flat)

                    # 预先按面片聚合顶点，避免逐像素重复 gather
                    faces_verts = verts[smplx_faces_tensor]
                    loss = torch.tensor(0.0, device=device)

                    # -------- DPoser-X whole-body 先验（仅在需要的阶段启用） --------
                    dposer_val = None
                    if use_dposer_this_stage and dposer_weight > 0.0:
                        dposer_loss = self._compute_dposer_loss(poses_flat, betas)
                        if dposer_loss is not None:
                            loss = loss + dposer_weight * dposer_loss
                            # 记录加权后的先验项，便于 log 直接看到"实际进入 loss 的量级"
                            dposer_val = float((dposer_weight * dposer_loss).detach().item())

                    # -------- MANO PCA L2 正则化（仅在 HAND_PRIOR="mano" 且手部阶段时启用） --------
                    mano_prior_val = None
                    if is_hand_stage and self._hand_prior == "mano" and hand_mano_weight > 0.0:
                        l_coeff = pose_params["left_hand"]
                        r_coeff = pose_params["right_hand"]
                        mano_loss = (l_coeff.pow(2).sum() + r_coeff.pow(2).sum()) * hand_mano_weight
                        loss = loss + mano_loss
                        mano_prior_val = float(mano_loss.detach().item())
                        last_mano_prior = mano_prior_val

                    if _use_batch_view and _packed_views is not None:
                        total_reproj, weight_sum, view_smpl_losses = _batch_view_projection(faces_verts, _packed_views, view_data)
                        smpl_loss_total = total_reproj.detach().clone()
                        if weight_sum > 0:
                            loss = loss + total_reproj
                    else:
                        smpl_loss_total = torch.tensor(0.0, device=device)
                        weight_sum = 0.0
                        view_smpl_losses = {}
                        for vid in active_views_stage:
                            data = view_data[vid]
                            group_idx_v = data.get('group_idx', int(vid) // max(1, int(data.get('cycle_size', 4))))
                            R_mat, T_vec = _resolve_camera_rt(data, view_data)
                            mask_smpl = _build_active_mask(data, active_segments, group_idx_v)
                            mask_smpl = _subsample_mask(mask_smpl, pixel_sample_ratio, data["pixel_weight"])
                            if mask_smpl.sum() == 0:
                                view_smpl_losses[int(vid)] = float('nan')
                                continue
                            loss_smpl, _ = _compute_view_reprojection(
                                faces_verts, data, mask_smpl, R_mat, T_vec,
                                self.cfg.USE_UNCERTAINTY_WEIGHT,
                            )
                            if loss_smpl is None:
                                view_smpl_losses[int(vid)] = float('nan')
                                continue
                            view_w = self.cfg.HAND_VIEW_WEIGHT if (group_idx_v in (1, 2)) else 1.0
                            loss = loss + loss_smpl * view_w
                            weight_sum += view_w
                            smpl_loss_total += (loss_smpl.detach() * view_w)
                            view_smpl_losses[int(vid)] = float(loss_smpl.detach().item())

                    if weight_sum > 0:
                        loss = loss / weight_sum
                        last_weight_sum = float(weight_sum)
                        last_smpl_loss = float((smpl_loss_total / weight_sum).detach().item())
                        last_view_smpl_losses = dict(view_smpl_losses)
                        if dposer_val is not None:
                            last_dposer_prior = dposer_val

                    joint_limit_val = None
                    if use_joint_limit_this_stage and joint_limit_weight > 0:
                        jl = _joint_limit_penalty(poses_flat)
                        loss = loss + jl
                        joint_limit_val = float(jl.detach().item())
                        last_joint_limit = joint_limit_val
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"[NaN] {stage.get('name', '')} [LBFGS] loss 非有限")
                    # 如果所有视图都被跳过且没有其他损失项，loss 不会有梯度追踪
                    if loss.requires_grad:
                        loss.backward()
                    return loss

                for it in range(iters):
                    try:
                        loss_val = optim_stage.step(_closure)
                    except RuntimeError as e:
                        # 若检测到 NaN/Inf（由 _assert_finite_params 或 DPoser 抛出），回退并用 Adam 兜底当前 repeat
                        if "[NaN]" in str(e):
                            print(f"[LBFGS] 检测到 NaN，回退并用 Adam 重跑当前 repeat，stage={stage.get('name','')}, iter={it}")
                            # 回退到进入本 stage 之前的安全状态
                            _restore_params(backup_state)

                            # 构造一个 Adam 版本的同名 stage，作为当前 repeat 的兜底
                            stage_adam = dict(stage)
                            stage_adam["optimizer"] = "adam"
                            # Adam 学习率适当减小，避免再次爆炸
                            stage_adam["lr"] = stage.get("lr", 0.01) * 0.1
                            # 作为兜底步，迭代次数减少
                            stage_adam["iters"] = max(10, int(stage.get("iters", 50) // 5))
                            # 避免在兜底 stage 内部再触发 repeat 逻辑
                            stage_adam["repeat_max"] = 0

                            _run_stage_once(stage_adam, active_segments, active_views_stage, optimizer_state=None)

                            # 本次 LBFGS stage 结束，外层 repeat 继续，下一轮仍使用 LBFGS
                            return
                        else:
                            # 其它错误保持 fail-fast
                            raise

                    if it % 1 == 0:
                        msg_parts = [f"lbfgs_loss={loss_val.item():.5f}"]
                        if last_smpl_loss is not None and last_weight_sum > 0:
                            msg_parts.append(f"w_smpl={last_smpl_loss:.5f}")
                        if last_dposer_prior is not None and last_weight_sum > 0:
                            # 这里的 w_dposer_prior 已包含 DPOSER_WEIGHT
                            msg_parts.append(f"w_dposer_prior={(last_dposer_prior/last_weight_sum):.5f}")
                        if last_mano_prior is not None and last_weight_sum > 0:
                            msg_parts.append(f"w_mano_prior={(last_mano_prior/last_weight_sum):.5f}")
                        if last_joint_limit is not None and last_weight_sum > 0:
                            msg_parts.append(f"w_joint_limit={(last_joint_limit/last_weight_sum):.5f}")
                        if last_view_smpl_losses:
                            view_str = ", ".join(
                                f"{vid}: smpl={last_view_smpl_losses.get(vid, float('nan')):.4f}"
                                for vid in active_views_stage
                            )
                            msg_parts.append(f"views=[{view_str}]")
                        print(f"  [{stage['name']}] {it}/{iters}  " + " ".join(msg_parts))
                    if it % 1 == 0:
                        if (self.cfg.ENABLE_VIS or self.cfg.ENABLE_IMG_VIS) and face_group_tensor is not None:
                            with torch.no_grad():
                                poses_flat_vis = flat_poses(pose_params)
                                if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                                    poses_flat_vis[3:66] = self.vp_model.decode(vp_latent.unsqueeze(0))["pose_body"].reshape(-1)
                                verts_vis = _forward_verts_from_pose(poses_flat_vis)
                                if self.cfg.ENABLE_VIS:
                                    test_utils.visualize_non_blocking_updates(
                                        verts=verts_vis, faces=smplx_faces_tensor,
                                        pts=None, pts_r=None, plot_coord=False,
                                    )
                                if self.cfg.ENABLE_IMG_VIS and face_group_tensor is not None:
                                    _run_vis_block(verts_vis, sorted(list(active_views_stage)), view_data, face_group_tensor.to(self.device))
            else:
                for it in range(iters):
                    _assert_finite_params(f"{stage.get('name', '')} iter={it} before step")
                    optim_stage.zero_grad()
                    poses_flat = flat_poses(pose_params)
                    if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                        decoded = self.vp_model.decode(vp_latent.unsqueeze(0))  # type: ignore
                        body_pose_aa = decoded["pose_body"].reshape(-1)
                        poses_flat[3:66] = body_pose_aa
                    verts = _forward_verts_from_pose(poses_flat)
                    # 预先按面片聚合顶点，避免逐像素重复 gather
                    faces_verts = verts[smplx_faces_tensor]

                    loss = torch.tensor(0.0, device=device)

                    # -------- DPoser-X whole-body 先验（仅在需要的阶段启用） --------
                    dposer_val = None
                    if use_dposer_this_stage and dposer_weight > 0.0:
                        dposer_loss = self._compute_dposer_loss(poses_flat, betas)
                        if dposer_loss is not None:
                            loss = loss + dposer_weight * dposer_loss
                            dposer_val = float((dposer_weight * dposer_loss).detach().item())

                    # -------- MANO PCA L2 正则化（仅在 HAND_PRIOR="mano" 且手部阶段时启用） --------
                    mano_prior_val = None
                    if is_hand_stage and self._hand_prior == "mano" and hand_mano_weight > 0.0:
                        l_coeff = pose_params["left_hand"]
                        r_coeff = pose_params["right_hand"]
                        mano_loss = (l_coeff.pow(2).sum() + r_coeff.pow(2).sum()) * hand_mano_weight
                        loss = loss + mano_loss
                        mano_prior_val = float(mano_loss.detach().item())
                        last_mano_prior = mano_prior_val

                    mask_loss_total = torch.tensor(0.0, device=device)
                    view_mask_losses = {}
                    for vid in active_views_stage:
                        view_mask_losses[vid] = 0.0

                    if _use_batch_view and _packed_views is not None:
                        total_reproj, weight_sum, view_smpl_losses = _batch_view_projection(faces_verts, _packed_views, view_data)
                        smpl_loss_total = total_reproj.detach().clone()
                        if weight_sum > 0:
                            loss = loss + total_reproj
                    else:
                        smpl_loss_total = torch.tensor(0.0, device=device)
                        weight_sum = 0.0
                        view_smpl_losses = {}
                        for vid in active_views_stage:
                            data = view_data[vid]
                            group_idx_v = data.get('group_idx', int(vid) // max(1, int(data.get('cycle_size', 4))))
                            R_mat, T_vec = _resolve_camera_rt(data, view_data)
                            mask_smpl = _build_active_mask(data, active_segments, group_idx_v)
                            mask_smpl = _subsample_mask(mask_smpl, pixel_sample_ratio, data["pixel_weight"])
                            if mask_smpl.sum() == 0:
                                view_smpl_losses[vid] = float('nan')
                                continue
                            loss_smpl, _ = _compute_view_reprojection(
                                faces_verts, data, mask_smpl, R_mat, T_vec,
                                self.cfg.USE_UNCERTAINTY_WEIGHT,
                            )
                            if loss_smpl is None:
                                view_smpl_losses[vid] = float('nan')
                                continue
                            view_w = self.cfg.HAND_VIEW_WEIGHT if (group_idx_v in (1, 2)) else 1.0
                            loss += loss_smpl * view_w
                            smpl_loss_total += (loss_smpl.detach() * view_w)
                            view_smpl_losses[vid] = loss_smpl.detach().item()
                            weight_sum += view_w

                    if weight_sum > 0:
                        loss = loss / weight_sum
                        last_weight_sum = float(weight_sum)
                        last_smpl_loss = float((smpl_loss_total / weight_sum).detach().item())
                        last_view_smpl_losses = dict(view_smpl_losses)
                        if dposer_val is not None:
                            last_dposer_prior = dposer_val

                    if use_joint_limit_this_stage and joint_limit_weight > 0:
                        jl = _joint_limit_penalty(poses_flat)
                        loss = loss + jl
                        last_joint_limit = float(jl.detach().item())
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"[NaN] {stage.get('name', '')} iter={it} loss 非有限")
                    if not loss.requires_grad:
                        if it % 10 == 0:
                            print(f"  [{stage['name']}] {it}/{iters} 无有效像素，跳过梯度更新")
                        continue
                    loss.backward()
                    optim_stage.step()
                    _assert_finite_params(f"{stage.get('name', '')} iter={it} after step")

                    if it % 10 == 0:
                        view_str = ", ".join(
                            [f"{vid}: smpl={view_smpl_losses.get(vid, float('nan')):.4f}, mask={view_mask_losses.get(vid, 0.0):.4f}"
                             for vid in active_views_stage]
                        )
                        denom = max(weight_sum, 1e-8)
                        msg_parts = [
                            f"loss={loss.item():.5f}",
                            f"smpl={(smpl_loss_total/denom):.5f}",
                            f"mask={(mask_loss_total/denom):.5f}",
                        ]
                        if dposer_val is not None and denom > 0:
                            msg_parts.append(f"w_dposer_prior={(dposer_val/denom):.5f}")
                        if mano_prior_val is not None and denom > 0:
                            msg_parts.append(f"w_mano_prior={(mano_prior_val/denom):.5f}")
                        if joint_limit_weight > 0 and last_joint_limit is not None and denom > 0:
                            msg_parts.append(f"w_joint_limit={(last_joint_limit/denom):.5f}")
                        msg_parts.append(f"views=[{view_str}]")
                        print(f"  [{stage['name']}] {it}/{iters}  " + " ".join(msg_parts))

                    if (self.cfg.ENABLE_VIS or self.cfg.ENABLE_IMG_VIS) and it % 10 == 0:
                        with torch.no_grad():
                            verts_vis = verts.detach()
                            if self.cfg.ENABLE_VIS:
                                test_utils.visualize_non_blocking_updates(
                                    verts=verts_vis, faces=smplx_faces_tensor,
                                    pts=None, pts_r=None, plot_coord=False,
                                )
                            if self.cfg.ENABLE_IMG_VIS and face_group_tensor is not None:
                                _run_vis_block(verts_vis, sorted(list(active_views_stage)), view_data, face_group_tensor.to(self.device))

            _invalidate_body_cache()
            state = optim_stage.state_dict()
            del optim_stage
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return state

        # ---------------- 按 stage 执行 + 可重复 ----------------
        # [Step 0-H · EXP-6] flag for warmstart body_pose reset (one-shot before "full pose")
        _smplxinit_body_reset_done = False
        _smplxinit_warmstart_active = (
            getattr(self.cfg, "INIT_FROM_SMPLESTX", False)
            and str(getattr(self.cfg, "SMPLESTX_INIT_BODY_POSE_MODE", "warmstart")).lower() == "warmstart"
        )
        _hand_residual_filter_done = False

        for stage in stages:
            # [Step 0-H · EXP-6 warm-start] 在 "full pose" stage 启动前 reset body_pose 到 zero
            # 这样 global pose / scale 享受了 SMPLest-X body_pose 的 warm-start 加速，
            # 而 full pose / Fine-tune All / Hands-* 不会被单视角 SMPLest-X body_pose 拖慢
            if (_smplxinit_warmstart_active and not _smplxinit_body_reset_done
                    and str(stage.get("name", "")).lower().startswith("full pose")):
                with torch.no_grad():
                    for k in ("pelvis", "body1", "foot0", "body12", "foot", "body2", "hand0"):
                        if k in pose_params:
                            if k == "hand0" and getattr(self.cfg, "SMPLESTX_FIX_HANDS", False):
                                continue
                            pose_params[k].data.zero_()
                _smplxinit_body_reset_done = True
                print(f"[Init][SMPLest-X] 'full pose' 启动前 reset body_pose 到 zero (warmstart 模式)")

            stage_name_lower = str(stage.get("name", "")).lower()
            if (not _hand_residual_filter_done) and stage_name_lower.startswith("hands-a"):
                filtered_hand_views, dropped_hand_views = _filter_hand_views_by_residual(
                    hand_views, stage["active_segments"]
                )
                hand_views[:] = filtered_hand_views
                for later_stage in stages:
                    later_name = str(later_stage.get("name", "")).lower()
                    if later_name.startswith("hands-"):
                        later_stage["active_views"] = hand_views
                    elif later_name.startswith("fine-tune body after"):
                        later_stage["active_views"] = body_views + hand_views
                _hand_residual_filter_done = True
                print(f"[HandViewFilter] kept={hand_views} dropped={dropped_hand_views}")

            lr = stage["lr"]
            iters = stage["iters"]
            active_segments = stage["active_segments"]
            active_views_stage = stage["active_views"]

            # 在“Fine-tune Body After”阶段开始前，将 hand 的非 betas 参数
            # 注入到主体（trans/scale），然后彻底关闭 split hand：后续仅保留一个人体模型。
            if use_split_hand and str(stage.get("name", "")).lower().startswith("fine-tune body after"):
                # if split_hand_active:
                #     with torch.no_grad():
                #         trans.data.copy_(trans_hand.data)
                #         scale.data.copy_(scale_hand.data)
                # 之后的 stage 不再进行手/身拆分前向，也不再保存/检查 hand 参数
                split_hand_active = False
                use_split_hand = False
            pixel_sample_ratio_stage = float(stage.get("pixel_sample_ratio", 1.0))

            # 每个 stage 自己决定是否启用 DPoser 先验 / 关节角度约束
            use_dposer_this_stage = bool(stage.get("use_dposer", False)) and (dposer_weight > 0.0)
            use_joint_limit_this_stage = bool(stage.get("use_joint_limit", True)) and (joint_limit_weight > 0.0)
            
            # MANO PCA L2 正则化（仅在手部阶段且 HAND_PRIOR="mano" 时启用）
            hand_mano_weight = float(getattr(self.cfg, "HAND_MANO_WEIGHT", 0.1))
            is_hand_stage = "Hand" in stage.get("name", "")

            # 按 stage 设置 DPoser 的时间策略；若该 stage 不启用 DPoser，则清空调度
            if use_dposer_this_stage:
                self._init_dposer_schedule(stage.get("name", ""))
            else:
                self._init_dposer_schedule("")  # 清空调度状态

            # 仅在 global pose 的正式优化前：
            # 1）先用当前 global_orient 粗略对齐 translation；
            # 2）在固定 translation 的情况下，对 global_orient 做一次 3D 粗粒度搜索；
            # 3）再进入正常的 global pose 联合优化（trans + global_orient）。
            if stage.get("name") == "global pose":
                if len(body_views) > 0:
                    # ---------- 1. translation 粗对齐（仅优化 trans，使用当前 global_orient） ----------
                    trans_stage = copy.deepcopy(stage)
                    trans_stage["name"] = stage.get("name", "") + " (pre-align trans)"
                    trans_stage["keys"] = ["trans"]
                    # 适当减小迭代次数，避免太耗时
                    trans_stage["iters"] = min(int(stage.get("iters", 50)), 30)
                    print("[global pose] 先进行 translation 粗对齐 ...")
                    _run_stage_once(trans_stage, active_segments, body_views, optimizer_state=None)

                    # ---------- 2. 在 3D axis-angle 空间上对 global_orient 做 coarse search ----------
                    print("[global pose] 进行粗粒度 global_orient 3D 搜索 ...")
                    angle_candidates = [-180, -120.0, -60.0, 0.0, 60.0, 120.0]

                    if self.cfg.BATCH_COARSE_SEARCH:
                        all_orient = []
                        all_deg = []
                        for ax_deg in angle_candidates:
                            for ay_deg in angle_candidates:
                                for az_deg in angle_candidates:
                                    all_orient.append(torch.tensor([math.radians(ax_deg), math.radians(ay_deg), math.radians(az_deg)], device=device))
                                    all_deg.append((ax_deg, ay_deg, az_deg))
                        all_orient_t = torch.stack(all_orient, dim=0)
                        batch_sz = self.cfg.COARSE_SEARCH_BATCH_SIZE
                        best_loss = float("inf")
                        best_angles = None

                        with torch.no_grad():
                            poses_flat_base = flat_poses(pose_params)
                            if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                                poses_flat_base[3:66] = self.vp_model.decode(vp_latent.unsqueeze(0))["pose_body"].reshape(-1)
                            scale_phys = F.softplus(scale) + 1e-6

                            for b_start in range(0, len(all_orient), batch_sz):
                                orient_batch = all_orient_t[b_start:b_start + batch_sz]
                                verts_batch = get_smplx_vertices_batch(
                                    poses_flat_base, betas, trans,
                                    global_orient_batch=orient_batch,
                                    gender="neutral", scale_param=scale_phys,
                                    smplx_model_neutral=smplx_neutral,
                                    smplx_model_male=self.smplx_model_male,
                                    smplx_model_female=self.smplx_model_female,
                                    use_hand_pca=self._use_hand_pca,
                                    hand_pca_comps=self._hand_pca_comps,
                                )
                                cur_B = verts_batch.shape[0]
                                for bi in range(cur_B):
                                    verts_i = verts_batch[bi]
                                    faces_verts_i = verts_i[smplx_faces_tensor]
                                    total_loss_i = torch.tensor(0.0, device=device)
                                    w_sum_i = 0.0
                                    for vid in body_views:
                                        data = view_data[vid]
                                        group_idx_v = data.get('group_idx', int(vid) // max(1, int(data.get('cycle_size', 4))))
                                        R_mat, T_vec = _resolve_camera_rt(data, view_data)
                                        mask_smpl = _build_active_mask(data, active_segments, group_idx_v)
                                        mask_smpl = _subsample_mask(mask_smpl, pixel_sample_ratio_stage, data["pixel_weight"])
                                        if mask_smpl.sum() == 0:
                                            continue
                                        loss_v, _ = _compute_view_reprojection(
                                            faces_verts_i, data, mask_smpl, R_mat, T_vec,
                                            self.cfg.USE_UNCERTAINTY_WEIGHT,
                                            use_procrustes=self.cfg.USE_PROCRUSTES_IN_COARSE_SEARCH,
                                        )
                                        if loss_v is None:
                                            continue
                                        vw = self.cfg.HAND_VIEW_WEIGHT if (group_idx_v in (1, 2)) else 1.0
                                        total_loss_i += loss_v * vw
                                        w_sum_i += vw
                                    loss_val = float((total_loss_i / w_sum_i).item()) if w_sum_i > 0 else float('inf')
                                    if loss_val < best_loss:
                                        best_loss = loss_val
                                        best_angles = all_deg[b_start + bi]
                    else:
                        best_loss = float("inf")
                        best_angles = None
                        for ax_deg in angle_candidates:
                            for ay_deg in angle_candidates:
                                for az_deg in angle_candidates:
                                    aa = torch.tensor([math.radians(ax_deg), math.radians(ay_deg), math.radians(az_deg)], device=device)
                                    with torch.no_grad():
                                        pose_params["global_orient"].data.copy_(aa)
                                    loss_val = _compute_stage_avg_loss(
                                        active_segments_eval=active_segments,
                                        active_views_eval=body_views,
                                        pixel_sample_ratio=pixel_sample_ratio_stage,
                                        use_procrustes=self.cfg.USE_PROCRUSTES_IN_COARSE_SEARCH,
                                    )
                                    if loss_val < best_loss:
                                        best_loss = loss_val
                                        best_angles = (ax_deg, ay_deg, az_deg)

                    if best_angles is not None:
                        ax_deg, ay_deg, az_deg = best_angles
                        print(f"[global pose] 选中初始 axis-angle≈({ax_deg:.1f}, {ay_deg:.1f}, {az_deg:.1f})° (avg loss={best_loss:.5f})")
                        aa = torch.tensor([math.radians(ax_deg), math.radians(ay_deg), math.radians(az_deg)], device=device)
                        with torch.no_grad():
                            pose_params["global_orient"].data.copy_(aa)

            # 在 global pose 阶段：采用“最佳→次优→两两→三三组合”重试策略，失败则跳过该样本
            stage_executed_in_selection = False
            if stage.get("name") == "global pose":
                candidates = list(body_views)
                if len(candidates) > 0:
                    # 有效像素检查
                    def _has_valid_pixels(v_id, seg_ids):
                        d = view_data[v_id]
                        gidx = d.get('group_idx', int(v_id) // max(1, int(d.get('cycle_size', 4))))
                        return bool(_build_active_mask(d, seg_ids, gidx).any().item())

                    # 过滤掉没有有效像素的单视角候选
                    valid_candidates = [v for v in candidates if _has_valid_pixels(v, active_segments)]

                    # 参数快照/恢复
                    def _snapshot_params():
                        pose_snap = {k: p.detach().clone() for k, p in pose_params.items()}
                        snap = {
                            "pose": pose_snap,
                            "betas": betas.detach().clone(),
                            "trans": trans.detach().clone(),
                            "scale": scale.detach().clone(),
                        }
                        return snap

                    def _restore_params(snap):
                        for k, t in snap["pose"].items():
                            pose_params[k].data.copy_(t)
                        betas.data.copy_(snap["betas"])
                        trans.data.copy_(snap["trans"])
                        scale.data.copy_(snap["scale"])

                    # 失败判定
                    rel_inc_thr = float(getattr(self.cfg, "GLOBAL_POSE_FAIL_REL_INCREASE", 0.3))
                    def _is_failure(prev_v, cur_v):
                        if not math.isfinite(cur_v):
                            return True
                        if math.isfinite(prev_v) and prev_v > 1e-6:
                            return (cur_v - prev_v) / max(prev_v, 1e-6) > rel_inc_thr
                        # prev 很小，则用绝对阈值
                        return (cur_v - prev_v) > 1e-3

                    # 组合列表：从双视角开始，再到三视角……（按当前配置上限）
                    max_comb = int(getattr(self.cfg, "GLOBAL_POSE_MAX_COMB_SIZE", 3))
                    combination_buckets: List[List[Tuple[float, Tuple[int, ...]]]] = []
                    # 多视角（基于有效候选）
                    for comb_size in range(1, max(2, min(max_comb, len(valid_candidates))) + 1):
                        combs = []
                        for comb in itertools.combinations(valid_candidates, comb_size):
                            lv = _compute_stage_avg_loss(active_segments, list(comb), pixel_sample_ratio_stage)
                            if math.isfinite(lv):
                                combs.append((lv, comb))
                        combs.sort(key=lambda x: x[0])
                        combination_buckets.append(combs)

                    # 逐组合尝试：每次从同一快照开始，失败则还原，成功则保留并采用该组合
                    stage_start_snap = _snapshot_params()
                    chosen_views: Optional[Tuple[int, ...]] = None
                    chosen_prev_loss: Optional[float] = None
                    chosen_opt_state = None
                    for bucket in combination_buckets:
                        if not bucket:
                            continue
                        for init_loss, comb in bucket:
                            _restore_params(stage_start_snap)
                            prev_loss_trial = init_loss
                            # 首次执行该组合
                            opt_state_trial = _run_stage_once(stage, active_segments, list(comb), optimizer_state=None)
                            cur_loss_trial = _compute_stage_avg_loss(active_segments, list(comb), pixel_sample_ratio_stage)
                            print(f"[global pose] 组合{comb} 试跑：prev={prev_loss_trial:.5f} cur={cur_loss_trial:.5f}")
                            if _is_failure(prev_loss_trial, cur_loss_trial):
                                print(f"[global pose] 组合{comb} 判定失败（上升超阈值），回滚")
                                continue
                            # 成功：保留当前参数，选用该组合
                            chosen_views = tuple(comb)
                            chosen_prev_loss = cur_loss_trial
                            chosen_opt_state = opt_state_trial
                            break
                        if chosen_views is not None:
                            break

                    if chosen_views is None:
                        print("[global pose] 所有双/三及以上组合均失败，跳过该样本")
                        return None

                    # 固定本阶段使用的视角为成功组合，并标记已执行过一次
                    active_views_stage = list(chosen_views)
                    prev_loss = chosen_prev_loss  # 为重复循环做基线
                    opt_state = chosen_opt_state
                    stage_executed_in_selection = True
                    print(f"[global pose] 选用组合 {active_views_stage}，初始loss={float(prev_loss) if prev_loss is not None else float('nan'):.5f}")
                    # 方案A：在进入重复判断前先执行一次当前 stage，使首轮比较产生实际下降
                    opt_state = _run_stage_once(stage, active_segments, active_views_stage, optimizer_state=opt_state)
                else:
                    print("[global pose] 无身体视角，跳过该样本")
                    return None


            rep_max = int(stage['repeat_max'])
            rep_thr = float(stage['repeat_improve'])

            if not stage_executed_in_selection:
                if rep_max > 0:
                    prev_loss = _compute_stage_avg_loss(active_segments, active_views_stage, pixel_sample_ratio_stage)
                else:
                    prev_loss = None
                # 先执行一次并保存优化器状态
                opt_state = _run_stage_once(stage, active_segments, active_views_stage, optimizer_state=None)

            if rep_max <= 0:
                continue

            repeats = 0
            while True:
                cur_loss = _compute_stage_avg_loss(active_segments, active_views_stage, pixel_sample_ratio_stage)
                rel_impr = (prev_loss - cur_loss) / max(prev_loss, 1e-8) if prev_loss is not None else 0.0
                if (rel_impr > rep_thr) and (repeats < rep_max):
                    repeats += 1
                    print(f"[{stage['name']}] 相对下降 {rel_impr:.2%} > 阈值 {rep_thr:.2%}，继续重复第 {repeats} 次")
                    prev_loss = cur_loss
                    opt_state = _run_stage_once(stage, active_segments, active_views_stage, optimizer_state=opt_state)
                    continue
                else:
                    if not (rel_impr > rep_thr):
                        print(f"[{stage['name']}] 相对下降 {rel_impr:.2%} ≤ 阈值 {rep_thr:.2%}，停止重复")
                    else:
                        print(f"[{stage['name']}] 达到最大重复次数 {rep_max}，停止重复")
                    break


        # ---- 获取最终 verts ----
        with torch.no_grad():
            poses_flat = flat_poses(pose_params)
            if USE_VPOSER and (vp_latent is not None) and (self.vp_model is not None):
                body_pose_final = self.vp_model.decode(vp_latent.unsqueeze(0))["pose_body"].reshape(-1)  # type: ignore
                poses_flat[3:66] = body_pose_final
            verts_final = _forward_verts_from_pose(poses_flat)

        # 将优化参数打包返回
        betas_opt = betas.detach().clone()
        scale_opt = (F.softplus(scale) + 1e-6).detach().clone()
        trans_opt = trans.detach().clone()
        poses_opt = poses_flat.detach().clone()
        result = {
            "verts_final": verts_final,
            "view_data": view_data,
            "betas": betas_opt,
            "scale": scale_opt,
            "trans": trans_opt,
            "poses": poses_opt,
        }
        # 若进行了手部拆分拟合，则额外返回手部的 betas / scale / trans，便于后续分析或复现
        if use_split_hand:
            result["betas_hand"] = betas_hand.detach().clone()
            result["scale_hand"] = (F.softplus(scale_hand) + 1e-6).detach().clone()
            result["trans_hand"] = trans_hand.detach().clone()
        return result

    # ----------------- 渲染 -----------------
    def render_subject(self, subject_id: int, result):
        """渲染并保存结果（移植自旧脚本）。"""
        verts_final = result["verts_final"]
        view_data = result["view_data"]
        # ----- 保存优化参数到 pred_params -----
        prefix = self._effective_output_prefix()
        params_dir = os.path.join(self.base_path, f"{prefix}_params")
        os.makedirs(params_dir, exist_ok=True)
        params_path = os.path.join(params_dir, f"{subject_id:04d}.npz")
        # 基础参数（与旧版保持兼容）
        npz_kwargs = dict(
            betas=result["betas"].cpu().numpy(),
            scale=result["scale"].cpu().numpy(),
            trans=result["trans"].cpu().numpy(),
            poses=result["poses"].cpu().numpy(),
        )
        # 若开启了手部拆分拟合，则额外写入手部的 betas/scale/trans
        if "betas_hand" in result:
            npz_kwargs["betas_hand"] = result["betas_hand"].cpu().numpy()
            npz_kwargs["scale_hand"] = result["scale_hand"].cpu().numpy()
            npz_kwargs["trans_hand"] = result["trans_hand"].cpu().numpy()
        np.savez(params_path, **npz_kwargs)
        print(f"[Render] 参数已保存到 {params_path}")
        # 创建输出目录
        smplx_dir = os.path.join(self.base_path, f"{prefix}_smplx")
        os.makedirs(smplx_dir, exist_ok=True)
        output_obj_path = os.path.join(smplx_dir, f"{subject_id:04d}.obj")
        save_obj(verts_final.detach().cpu(), self.smplx_faces_tensor.cpu(), output_obj_path)
        print(f"[Render] 模型已保存到 {output_obj_path}")

        # 新增：保存 A-pose mesh 到 pred_smplx_apose
        apose_dir = os.path.join(self.base_path, f"{prefix}_smplx_apose")
        os.makedirs(apose_dir, exist_ok=True)
        with torch.no_grad():
            # 根据参考生成 A-pose 参数
            rad = np.pi / 180.0
            base_len = 75
            hand_len = self._hand_pca_comps  # 根据 PCA 配置确定手部参数维度
            apose_len = base_len + 2 * hand_len
            apose_pose = torch.zeros(apose_len, device=self.device)
            # 左肩绕 Z 轴 -20°, 右肩 +20°
            apose_pose[3 + 15*3 + 2] = -45 * rad
            apose_pose[3 + 16*3 + 2] = 45 * rad
            # 使用优化后 betas，但忽略 trans 和 scale
            verts_apose = get_smplx_vertices(
                apose_pose,
                result["betas"],
                torch.zeros_like(result["trans"]),
                gender="neutral",
                smplx_model_neutral=self.smplx_model_neutral,
                smplx_model_male=self.smplx_model_male,
                smplx_model_female=self.smplx_model_female,
                use_hand_pca=self._use_hand_pca,
                hand_pca_comps=self._hand_pca_comps,
            )
        apose_obj_path = os.path.join(apose_dir, f"{subject_id:04d}_apose.obj")
        save_obj(verts_apose.detach().cpu(), self.smplx_faces_tensor.cpu(), apose_obj_path)
        print(f"[Render] A-pose 模型已保存到 {apose_obj_path}")

        # joints
        joints_3d = self.J_regressor_smplx @ verts_final

        joints_dir = os.path.join(self.base_path, f"{prefix}_smplx_2d_joints")
        os.makedirs(joints_dir, exist_ok=True)
        crop_list = []  # store each view's cropped rgba
        # 渲染已准备的全部视角（包含身体与手部），按 view_id 升序
        render_view_ids = sorted(list(view_data.keys()))
        for vid in render_view_ids:
            if vid in view_data:
                data = view_data[vid]
                # joints 投影使用现有（固定）相机参数
                R_mat_final = axis_angle_to_matrix(data["rot_param"]).detach()
                T_vec_final = data["trans_param"].detach()
                focal_x, focal_y = data["focal_x"], data["focal_y"]
                cx, cy = data["cx"], data["cy"]
                cameras = data.get("cameras_orig", data["cameras_p3d"])  # 原始分辨率优先
                orig_h, orig_w = data.get("orig_size", data["img_size"])
            else:
                # 从 GT 相机重建渲染所需的信息
                cam_base_name = f"{subject_id:04d}_{vid:02d}"
                cam_npz_path = os.path.join(self.CAMERAS_GT_PATH, f"{cam_base_name}.npz")
                cam_npz = np.load(cam_npz_path)
                R_np = cam_npz["R"].astype(np.float32)
                T_np = (cam_npz["T"].astype(np.float32) / 1000.0)
                K_np = cam_npz["K"].astype(np.float32)
                focal_x, focal_y, cx, cy = K_np[0, 0], K_np[1, 1], K_np[0, 2], K_np[1, 2]
                # 原图尺寸
                image_ori_dir = os.path.join(os.path.dirname(self.CAMERAS_GT_PATH), 'image_ori')
                _ori_path = None
                for _ext in ['.png', '.jpg', '.jpeg']:
                    _cand = os.path.join(image_ori_dir, f"{cam_base_name}{_ext}")
                    if os.path.exists(_cand):
                        _ori_path = _cand
                        break
                if _ori_path is not None:
                    _img_ori = imageio.v2.imread(_ori_path)
                    orig_h, orig_w = _img_ori.shape[:2]
                else:
                    # 若找不到原图，退回到一个合理的默认尺寸（不影响裁剪）
                    orig_h, orig_w = 1080, 1920
                cameras = cameras_from_opencv_projection(
                    torch.from_numpy(R_np).to(self.device)[None],
                    torch.from_numpy(T_np).to(self.device)[None],
                    torch.from_numpy(K_np).to(self.device)[None],
                    torch.tensor([[orig_h, orig_w]], dtype=torch.float32, device=self.device),
                )
                R_mat_final = torch.from_numpy(R_np).to(self.device)
                T_vec_final = torch.from_numpy(T_np).to(self.device)

            # joints 2D 保存
            joints_cam = torch.matmul(joints_3d, R_mat_final) + T_vec_final
            joints_2d = torch.zeros((joints_cam.shape[0], 2), device=self.device)
            z_joint = joints_cam[:, 2].clamp(min=1e-6)
            joints_2d[:, 0] = cx - focal_x * joints_cam[:, 0] / z_joint
            joints_2d[:, 1] = cy - focal_y * joints_cam[:, 1] / z_joint
            output_joints_path = os.path.join(joints_dir, f"{subject_id:04d}_{vid}.json")
            save_joints_2d(joints_2d, output_joints_path)

            # 渲染原始分辨率下的 mesh
            rgb, mask = render_mesh(
                verts=verts_final,
                faces=self.smplx_faces_tensor,
                cameras=cameras,
                image_size=(orig_h, orig_w),
                device=self.device,
            )

            rgba_np = np.concatenate(
                [
                    (rgb * 255).detach().cpu().numpy().astype(np.uint8),
                    (mask * 255).detach().cpu().numpy()[..., None].astype(np.uint8),
                ],
                axis=-1,
            )
            # 红色边框
            rgba_np[0, :, :3] = [255, 0, 0]
            rgba_np[-1, :, :3] = [255, 0, 0]
            rgba_np[:, 0, :3] = [255, 0, 0]
            rgba_np[:, -1, :3] = [255, 0, 0]
            rgba_dir = os.path.join(self.base_path, f"{prefix}_rgba")
            os.makedirs(rgba_dir, exist_ok=True)
            output_rgba_path = os.path.join(rgba_dir, f"{subject_id:04d}_{vid}.png")

            # 裁剪：手部视角使用 hand bbox，身体视角使用 body bbox
            bbox_dir = os.path.join(os.path.dirname(self.CAMERAS_GT_PATH), 'bbox')
            group_idx = 0
            if vid in view_data:
                group_idx = int(view_data[vid].get('group_idx', int(vid) // max(1, int(view_data[vid].get('cycle_size', getattr(self.cfg, 'BODY_VIEWS', 4))))))
                base_cam_idx = int(view_data[vid].get('base_cam', int(vid) % max(1, int(view_data[vid].get('cycle_size', getattr(self.cfg, 'BODY_VIEWS', 4))))))
            else:
                base_cam_idx = int(vid)

            bbox_ratio = DEFAULT_BBOX_RATIO
            if group_idx in (1, 2):
                part_map = {1: 'left_hand', 2: 'right_hand'}
                part = part_map.get(group_idx)
                handfoot_root = os.path.join(self.base_path, 'bbox_hand_foot')
                hand_json_path = os.path.join(handfoot_root, part, f'{subject_id:04d}_{base_cam_idx:02d}.json')
                if not os.path.exists(hand_json_path):
                    print(f"[Render] 视角 {vid}: 缺少手部 bbox，跳过 ({hand_json_path})")
                    continue
                with open(hand_json_path, 'r') as _f:
                    _jd = json.load(_f)
                tlwh = _jd.get('tlwh')
                if tlwh is None:
                    print(f"[Render] 视角 {vid}: 手部 bbox 无 tlwh，跳过 ({hand_json_path})")
                    continue
                t, l, w, h = tlwh
                bbox_ratio = 1.0
            else:
                base_cam_name = f'{subject_id:04d}_{base_cam_idx:02d}'
                bbox_json_path = os.path.join(bbox_dir, f'{base_cam_name}.json')
                with open(bbox_json_path, 'r') as _f:
                    _jd = json.load(_f)
                t, l, w, h = _jd['tlwh']

            bbox = [t, l, t + w, l + h]
            crop_rgba = crop_scale_center_tensor(
                torch.from_numpy(rgba_np).permute(2, 0, 1),
                bbox,
                self.cfg.TARGET_SIZE,
                self.cfg.TARGET_SIZE,
                bbox_ratio,
            ).permute(1, 2, 0).cpu().numpy().astype(np.uint8)

            imageio.imwrite(output_rgba_path, crop_rgba)
            print(f"[Render] 视角 {vid}: crop → {output_rgba_path}")
            print()
            crop_list.append(crop_rgba)
        # ------- concat per subject -------
        if crop_list:
            max_h = max(im.shape[0] for im in crop_list)
            resized = [cv2.resize(im, (int(im.shape[1]*max_h/im.shape[0]), max_h), interpolation=cv2.INTER_NEAREST) if im.shape[0]!=max_h else im for im in crop_list]
            concat_img = np.hstack(resized)
            cat_dir = os.path.join(self.base_path, f'{prefix}_rgba_cat')
            os.makedirs(cat_dir, exist_ok=True)
            cat_path = os.path.join(cat_dir, f'{subject_id:04d}.png')
            imageio.imwrite(cat_path, concat_img)
            print(f'[Render] concat image → {cat_path}')

    # ----------------- 检查是否已完成 -----------------
    def _is_subject_completed(self, subject_id: int) -> bool:
        """检查主体是否已经完成处理，通过检查关键输出文件是否存在"""
        prefix = self._effective_output_prefix()
        
        # 检查关键文件是否存在
        key_files = [
            os.path.join(self.base_path, f"{prefix}_params", f"{subject_id:04d}.npz"),
            os.path.join(self.base_path, f"{prefix}_smplx", f"{subject_id:04d}.obj"),
            os.path.join(self.base_path, f"{prefix}_rgba_cat", f"{subject_id:04d}.png"),
        ]
        
        # 如果所有关键文件都存在，则认为已完成
        all_exist = all(os.path.exists(f) for f in key_files)
        
        if all_exist:
            print(f"[Skip] 主体 {subject_id:04d} 已完成处理，跳过")
            return True
        else:
            # 显示缺失的文件（用于调试）
            missing = [f for f in key_files if not os.path.exists(f)]
            print(f"[Process] 主体 {subject_id:04d} 需要处理，缺失文件: {len(missing)}")
            return False

    def _effective_output_prefix(self) -> str:
        """根据 tts_count 返回最终使用的输出前缀。"""
        if self.cfg.tts_count is None:
            return self.cfg.output_prefix
        return f"{self.cfg.output_prefix}-tts{self.cfg.tts_count}"

    # ----------------- 主流程 -----------------
    def run(self):
        self.prepare_dataset()
        
        # 统计已完成和需要处理的数量
        total_subjects = len(self.subjects)
        completed_count = 0
        processing_count = 0
        
        print(f"[Info] 总共 {total_subjects} 个主体需要检查")
        
        for sid in self.subjects:
            # 检查是否已完成
            if self._is_subject_completed(sid):
                completed_count += 1
                continue
                
            processing_count += 1
            print(f"[Process] 开始处理主体 {sid:04d} ({processing_count}/{total_subjects - completed_count})")
            
            view_data = self._prepare_subject_views(sid)
            if view_data is None:
                continue
            # 将 view_data 存入全局，以便 optimize / render 使用
            fit_start_time = time.perf_counter()
            result = self.optimize_subject(sid, view_data)
            fit_elapsed_s = time.perf_counter() - fit_start_time
            print(f"[GPU{self.cfg.device_id}]")
            print(f"[Timing] Subject {sid:04d} 拟合耗时: {fit_elapsed_s:.3f}s")
            if result is None:
                print(f"[Process] 主体 {sid:04d} global pose 阶段所有组合失败，跳过")
                continue
            self.render_subject(sid, result)
            
        print(f"[Summary] 处理完成！已跳过: {completed_count}, 新处理: {processing_count}, 总计: {total_subjects}")

# --------------------------- 新入口函数 ---------------------------

def main(cfg: Config):
    """顶层入口：实例化 Reconstructor 并执行 run()"""
    reconstructor = DenseBodyReconstructor(cfg)
    reconstructor.run()


# --------------------------- 简易多任务调度 (按subject分配) ---------------------------
import multiprocessing as _mp
from queue import Empty as _QueueEmpty

# 顶层 GPU worker，需可 picklable (Windows)
def _gpu_worker(subject_task_q: "_mp.Queue", gid: int, path_config: dict):
    """
    子进程 worker，接收路径配置并初始化全局变量。
    path_config: 包含所有路径配置的字典
    """
    # 在子进程中重新初始化全局变量
    _init_worker_globals(path_config)
    
    torch.cuda.set_device(gid)
    reconstructor_cache = {}

    while True:
        try:
            subject_task = subject_task_q.get_nowait()
        except _QueueEmpty:
            break
        task_key = subject_task["task_key"]
        try:
            if task_key not in reconstructor_cache:
                task_cfg = subject_task["task_cfg"]
                cfg = build_config(task_cfg, gid)
                print(f"[GPU{gid}] 初始化任务实例 exp={cfg.exp} data={cfg.data}")
                reconstructor = DenseBodyReconstructor(cfg)
                reconstructor.prepare_dataset()
                reconstructor_cache[task_key] = reconstructor
            _worker_run_subject(subject_task, gid, reconstructor_cache[task_key])
        except Exception as e:
            subject_id = int(subject_task.get("subject_id", -1))
            print(f"[GPU{gid}][Error] subject={subject_id:04d} 失败，跳过并继续。{type(e).__name__}: {e}")
            print(traceback.format_exc())
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            continue

    reconstructor_cache.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _init_worker_globals(path_config: dict):
    """在子进程中初始化全局变量"""
    global DPOSER_ROOT, HBP_SRC_PATH, MODEL_FOLDER
    global DATASET_ROOT, UV_TEMPLATE_DIR, CACHE_DIR, DPOSER_CACHE_DIR, SMPLX_CACHE_DIR
    global TEMPLATE_OBJ_PATH, HAND_SPLIT_OBJ_PATH
    
    DPOSER_ROOT = path_config['DPOSER_ROOT']
    HBP_SRC_PATH = path_config['HBP_SRC_PATH']
    MODEL_FOLDER = path_config['MODEL_FOLDER']
    DATASET_ROOT = path_config['DATASET_ROOT']
    UV_TEMPLATE_DIR = path_config['UV_TEMPLATE_DIR']
    CACHE_DIR = path_config['CACHE_DIR']
    DPOSER_CACHE_DIR = path_config.get('DPOSER_CACHE_DIR')
    SMPLX_CACHE_DIR = path_config.get('SMPLX_CACHE_DIR')
    TEMPLATE_OBJ_PATH = path_config['TEMPLATE_OBJ_PATH']
    HAND_SPLIT_OBJ_PATH = path_config['HAND_SPLIT_OBJ_PATH']
    
    # 更新 DEFAULT_CONFIG
    DEFAULT_CONFIG['MODEL_FOLDER'] = MODEL_FOLDER
    DEFAULT_CONFIG['VPOSER_MODEL_DIR'] = path_config['VPOSER_MODEL_DIR']
    
    # 设置模块路径并导入依赖
    _setup_module_paths()
    
    # 初始化 UV 模板张量
    _init_uv_templates()


def _worker_run_subject(subject_task: dict, gpu_id: int, reconstructor: "DenseBodyReconstructor"):
    """子进程函数：为指定 GPU 处理单个 subject"""
    subject_id = subject_task["subject_id"]
    cfg = reconstructor.cfg
    print(f"[GPU{gpu_id}] 开始处理 exp={cfg.exp} data={cfg.data} subject={subject_id:04d}")
    
    # 检查该subject是否已完成
    if reconstructor._is_subject_completed(subject_id):
        print(f"[GPU{gpu_id}] Subject {subject_id:04d} 已完成，跳过")
        return
    
    # 处理单个subject
    view_data = reconstructor._prepare_subject_views(subject_id)
    if view_data is None:
        print(f"[GPU{gpu_id}] Subject {subject_id:04d} 无法准备视图数据，跳过")
        return
        
    fit_start_time = time.perf_counter()
    result = reconstructor.optimize_subject(subject_id, view_data)
    fit_elapsed_s = time.perf_counter() - fit_start_time
    print(f"[GPU{gpu_id}]")
    print(f"[Timing][GPU{gpu_id}] Subject {subject_id:04d} 拟合耗时: {fit_elapsed_s:.3f}s")
    if result is None:
        print(f"[GPU{gpu_id}] Subject {subject_id:04d} global pose 阶段所有组合失败，跳过")
        return
    reconstructor.render_subject(subject_id, result)
    print(f"[GPU{gpu_id}] Subject {subject_id:04d} 处理完成")
    del view_data
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _get_subjects_for_task(task_cfg: dict) -> list:
    """获取指定任务配置下的所有subject列表，不加载模型"""
    # 构建路径（使用全局 DATASET_ROOT）
    base_path = os.path.join(
        DATASET_ROOT,
        "evaluate",
        task_cfg["data"],
        f"ours{task_cfg.get('ours_suffix', '_bg')}",
        task_cfg["exp"],
    )
    # 根据 tts_count 选择输入目录
    tts_count_cfg = task_cfg.get('tts_count', None)
    if tts_count_cfg is not None:
        input_path = os.path.join(base_path, f"tempt_tts{tts_count_cfg}_single")
    else:
        input_path = os.path.join(base_path, "tempt_single")
    

    def _collect_subjects(prefix: str):
        if not os.path.isdir(input_path):
            print(f"[GetSubjects] 输入目录不存在: {input_path}")
            return set()
        # 仅匹配 uv_####_##_pred.png，排除 uv_uncertainty_* 等文件
        pat = re.compile(rf'^uv_(?P<sid>\d{{4}})_\d{{2}}_pred\.png$')
        c_files = [f for f in os.listdir(input_path) if pat.match(f)]
        return {int(pat.match(f).group('sid')) for f in c_files}
    
    subjects = sorted(_collect_subjects("uv_"))

    
    print(f"[GetSubjects] Task {task_cfg['exp']} on {task_cfg['data']}: 找到 {len(subjects)} 个subjects")
    return subjects


def run_scheduler_by_subject(tasks: list, gpu_ids=(0, 1), path_config: dict = None):
    """按subject分配任务的调度器：每个 GPU 一个进程，空闲即取下一个subject
    
    Args:
        tasks: 任务配置列表
        gpu_ids: GPU ID 元组
        path_config: 路径配置字典，用于传递给子进程初始化全局变量
    """
    if path_config is None:
        raise ValueError("path_config 是必需的，请从 main 块传入路径配置")
    
    # 使用 spawn 方法创建进程，避免 CUDA fork 问题
    ctx = _mp.get_context('spawn')
    
    # 生成所有subject级别的任务
    subject_tasks = []
    for task_cfg in tasks:
        subjects = _get_subjects_for_task(task_cfg)
        # Process every subject selected by the explicit filter or range.
        for subject_id in subjects:
            subject_tasks.append({
                "task_cfg": task_cfg,
                "task_key": json.dumps(task_cfg, sort_keys=True, ensure_ascii=True),
                "subject_id": subject_id
            })
    
    print(f"[Scheduler] 总共生成 {len(subject_tasks)} 个subject任务")
    if torch.cuda.device_count()==2:
        subject_tasks = subject_tasks[::-1]
    
    # 使用 spawn context 的 Queue
    subject_task_q = ctx.Queue()
    for st in subject_tasks:
        subject_task_q.put(st)
 
    procs = []
    for gid in gpu_ids:
        # 传递 path_config 给子进程
        p = ctx.Process(target=_gpu_worker, args=(subject_task_q, gid, path_config))
        p.start()
        procs.append(p)

    for p in procs:
        p.join()
    
    print(f"[Scheduler] 所有subject任务处理完成")


# --------------------------- 单线程顺序执行（便于调试） ---------------------------
def run_tasks_single_thread(tasks: list, gpu_id: int = 0):
    """在单个进程/线程中顺序执行所有任务与subject，便于debug。"""
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_id)

    for task_cfg in tasks:
        cfg = build_config(task_cfg, gpu_id)

        print(f"[SingleThread][GPU{gpu_id}] 开始处理 exp={cfg.exp} data={cfg.data}")

        reconstructor = DenseBodyReconstructor(cfg)
        # 只加载与准备一次数据集，然后顺序处理该任务下的所有subjects
        reconstructor.prepare_dataset()

        subjects = getattr(reconstructor, 'subjects', [])
        print(f"[SingleThread] 共 {len(subjects)} 个subjects 需要处理: {subjects}")

        # [Step 0-A claude] 用 task_cfg.max_subjects 限定（小样本快速迭代）
        max_subj = int(task_cfg.get('max_subjects', len(subjects))) if isinstance(task_cfg, dict) else len(subjects)
        # [Step 0-F claude] subject_filter (list[int]) — 仅跑指定 subject_id；用于多 GPU 切片
        # subject_start / subject_end (int) — 仅跑 [start, end) 区间；与 subject_filter 二选一
        _subj_filter = task_cfg.get('subject_filter') if isinstance(task_cfg, dict) else None
        _subj_start = task_cfg.get('subject_start') if isinstance(task_cfg, dict) else None
        _subj_end = task_cfg.get('subject_end') if isinstance(task_cfg, dict) else None
        if _subj_filter is not None:
            _filter_set = set(int(x) for x in _subj_filter)
            iter_subjects = [s for s in subjects if int(s) in _filter_set]
        elif _subj_start is not None or _subj_end is not None:
            _s = int(_subj_start) if _subj_start is not None else 0
            _e = int(_subj_end) if _subj_end is not None else len(subjects)
            iter_subjects = [s for s in subjects if _s <= int(s) < _e]
        else:
            iter_subjects = subjects[:max_subj]
        print(f"[SingleThread][GPU{gpu_id}] 实际处理 {len(iter_subjects)} 个 subjects")
        for subject_id in iter_subjects:
        # for subject_id in [subjects[10],subjects[11],subjects[19],subjects[21],subjects[25],subjects[30],subjects[35],subjects[38],
        #                    subjects[42],subjects[52],subjects[55],subjects[66],subjects[67],subjects[70],subjects[76],subjects[88],]:#[21:]:#[77:78]: #[2:]:#[5:]:

            # 跳过已完成
            if reconstructor._is_subject_completed(subject_id):
                print(f"[SingleThread][GPU{gpu_id}] Subject {subject_id:04d} 已完成，跳过")
                continue

            if cfg.RANDOM_SEED is not None:
                subject_seed = int(cfg.RANDOM_SEED) + int(subject_id)
                random.seed(subject_seed)
                np.random.seed(subject_seed % (2 ** 32))
                torch.manual_seed(subject_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(subject_seed)
                print(f"[Seed] subject={subject_id:04d} seed={subject_seed}")

            print(f"[SingleThread][GPU{gpu_id}] 开始 Subject {subject_id:04d}")
            view_data = reconstructor._prepare_subject_views(subject_id)
            if view_data is None:
                print(f"[SingleThread][GPU{gpu_id}] Subject {subject_id:04d} 无法准备视图数据，跳过")
                continue

            fit_start_time = time.perf_counter()
            result = reconstructor.optimize_subject(subject_id, view_data)
            fit_elapsed_s = time.perf_counter() - fit_start_time
            print(f"[GPU{gpu_id}]")
            print(f"[Timing][SingleThread][GPU{gpu_id}] Subject {subject_id:04d} 拟合耗时: {fit_elapsed_s:.3f}s")
            if result is None:
                print(f"[SingleThread][GPU{gpu_id}] Subject {subject_id:04d} global pose 阶段所有组合失败，跳过")
                continue
            reconstructor.render_subject(subject_id, result)

        print(f"[SingleThread][GPU{gpu_id}] 任务完成 exp={cfg.exp} data={cfg.data}")


# --------------------------- 默认配置与构建器 ---------------------------
# 路径配置将在 __main__ 块中设置
DEFAULT_CONFIG = {
    'MODEL_FOLDER': None,  # 在 main 中设置
    'VPOSER_MODEL_DIR': None,  # 在 main 中设置
    'ours_suffix': '_bg',
    'TARGET_SIZE': 400,
    'BODY_VIEWS': 4,
    'RANDOM_SEED': None,
    # 身体/手部先验类型
    'BODY_PRIOR': 'dposerx',  # "dposerx" 或 "vposer"
    'HAND_PRIOR': 'dposerx',  # "dposerx" 或 "mano"
    'HAND_PCA_COMPS': 12,  # MANO PCA 主成分数（仅 HAND_PRIOR="mano" 时生效）
    'HAND_MANO_WEIGHT': 0.1,  # MANO PCA L2 正则化权重（仅 HAND_PRIOR="mano" 时生效）
    'DPOSER_WEIGHT': 0.1,  # DPoser-X 先验权重
    'DPOSER_BODY_MULT': 1.0,  # DPoser-X 身体倍率
    'DPOSER_HAND_MULT': 0.1,  # [Queue B] reset to baseline; per-task override via 'dposer_hand_mult'
    'ENABLE_VIS': False,
    # 'ENABLE_VIS': True,
    'ENABLE_IMG_VIS': False, #True,#False, #True,#
    # 'ENABLE_IMG_VIS': True, #True,#False, #True,#
    'output_prefix': 'pred',
    'USE_UNCERTAINTY_WEIGHT': True, #False, #True,False,#
    'OPTIMIZER': 'adam',
    # 视角质量筛选默认关闭与阈值
    'FILTER_VIEWS': True,
    'MIN_VALID_FACE_RATIO_BODY': 0.01,
    'MIN_VALID_FACE_RATIO_HAND': 0.0001, #
    'MAX_AVG_UNCERTAINTY': 0.7, #0.39,# 0.7,
    'FILTER_HAND_VIEW_RESIDUALS': True,
    'HAND_VIEW_RESIDUAL_MAX_RATIO': 4.0,
    'HAND_VIEW_RESIDUAL_ABS_FLOOR': 50.0,
    'HAND_VIEW_RESIDUAL_MIN_KEEP': 2,
    'HAND_VIEW_WEIGHT': 0.1,  # [Queue B] reset to baseline; per-task override via 'HAND_VIEW_WEIGHT'
    'JOINT_LIMIT_WEIGHT': 0., #10., #100.0,
    # 手部阶段冻结身体根部/整体参数（默认开启）
    'FREEZE_BODY_IN_HAND_STAGES': True,
    'PARAM_LR_MULTIPLIERS': {
        # 参考 smplify-x 经验设置的默认倍率
        'global_orient': 1.0,
        'trans': 1.0,
        'scale': 1.0,
        'betas': 1.0,
        'left_hand': 1.0,
        'right_hand': 1.0,
    },
    'HAND_LR_MULTIPLIER': 10.0,  # [Queue B] reset to baseline; per-task override via 'hand_lr_multiplier'
    'TTS_COUNT': None,
    'SKIP_HAND_FIT': False,
    'SKIP_HANDS_C': False,
    # 手部分开拟合（独立 hand betas/scale/trans，并在顶点级合并），默认关闭
    'SPLIT_HAND_FIT': True,
    # scale 参数是否参与优化（默认开启）
    'OPTIMIZE_SCALE': True,
    # global pose 组合重试参数
    'GLOBAL_POSE_FAIL_REL_INCREASE': 0.1,
    'GLOBAL_POSE_MAX_COMB_SIZE': 10,
    # 性能优化开关
    'BATCH_COARSE_SEARCH': True,
    'CACHE_BODY_VERTS_IN_HAND': True,
    'BATCH_VIEW_PROJECTION': True,
    'COARSE_SEARCH_BATCH_SIZE': 64,
    # [Step 0-H · EXP-6] SMPLest-X init 默认关闭
    'INIT_FROM_SMPLESTX': False,
    'SMPLESTX_INIT_DIR': None,
    'SMPLESTX_INIT_VIEW': 0,
    'SMPLESTX_INIT_INJECT_HANDS': False,
    'SMPLESTX_FIX_HANDS': False,
    'SMPLESTX_INIT_BODY_POSE_MODE': 'warmstart',
}

def build_config(task_cfg: dict, gpu_id: int) -> Config:
    """根据默认配置与任务覆盖项构建 Config。task_cfg 中的键覆盖默认值。
    支持的覆盖键：exp, data, ours_suffix, output_prefix, use_uncertainty_weight, use_seg_edge_erosion, target_size, optimizer。
    """
    merged = dict(DEFAULT_CONFIG)
    # 必填项
    merged['exp'] = task_cfg['exp']
    merged['data'] = task_cfg['data']
    # 覆盖可选项（小写键）
    if 'ours_suffix' in task_cfg:
        merged['ours_suffix'] = task_cfg['ours_suffix']
    if 'output_prefix' in task_cfg:
        merged['output_prefix'] = task_cfg['output_prefix']
    if 'use_uncertainty_weight' in task_cfg:
        merged['USE_UNCERTAINTY_WEIGHT'] = task_cfg['use_uncertainty_weight']
    if 'target_size' in task_cfg:
        merged['TARGET_SIZE'] = task_cfg['target_size']
    if 'optimizer' in task_cfg:
        merged['OPTIMIZER'] = task_cfg['optimizer']
    if 'body_views' in task_cfg:
        merged['BODY_VIEWS'] = int(task_cfg['body_views'])
    if 'random_seed' in task_cfg:
        merged['RANDOM_SEED'] = task_cfg['random_seed']
    # TTS 支持
    if 'tts_count' in task_cfg:
        merged['TTS_COUNT'] = task_cfg['tts_count']
    # 跳过手部阶段
    if 'skip_hand_fit' in task_cfg:
        merged['SKIP_HAND_FIT'] = task_cfg['skip_hand_fit']
    # 仅跳过 Hands-C: fingers refine（任务中用 skip_hands_c）
    if 'skip_hands_c' in task_cfg:
        merged['SKIP_HANDS_C'] = task_cfg['skip_hands_c']
    # 手部分开拟合（task 中用小写 key）
    if 'split_hand_fit' in task_cfg:
        merged['SPLIT_HAND_FIT'] = task_cfg['split_hand_fit']
    # VPoser 模型目录（当 body_prior="vposer" 时需要）
    if 'vposer_model_dir' in task_cfg:
        merged['VPOSER_MODEL_DIR'] = task_cfg['vposer_model_dir']
    # 身体/手部先验类型
    if 'body_prior' in task_cfg:
        merged['BODY_PRIOR'] = str(task_cfg['body_prior']).lower()
    if 'hand_prior' in task_cfg:
        merged['HAND_PRIOR'] = str(task_cfg['hand_prior']).lower()
    if 'hand_pca_comps' in task_cfg:
        merged['HAND_PCA_COMPS'] = int(task_cfg['hand_pca_comps'])
    if 'hand_mano_weight' in task_cfg:
        merged['HAND_MANO_WEIGHT'] = float(task_cfg['hand_mano_weight'])
    if 'dposer_weight' in task_cfg:
        merged['DPOSER_WEIGHT'] = float(task_cfg['dposer_weight'])
    if 'dposer_body_mult' in task_cfg:
        merged['DPOSER_BODY_MULT'] = float(task_cfg['dposer_body_mult'])
    if 'dposer_hand_mult' in task_cfg:
        merged['DPOSER_HAND_MULT'] = float(task_cfg['dposer_hand_mult'])
    # 允许任务覆盖筛选参数
    if 'filter_views' in task_cfg:
        merged['FILTER_VIEWS'] = task_cfg['filter_views']
    if 'min_valid_face_ratio_body' in task_cfg:
        merged['MIN_VALID_FACE_RATIO_BODY'] = task_cfg['min_valid_face_ratio_body']
    if 'min_valid_face_ratio_hand' in task_cfg:
        merged['MIN_VALID_FACE_RATIO_HAND'] = task_cfg['min_valid_face_ratio_hand']
    if 'max_avg_uncertainty' in task_cfg:
        merged['MAX_AVG_UNCERTAINTY'] = task_cfg['max_avg_uncertainty']
    if 'filter_hand_view_residuals' in task_cfg:
        merged['FILTER_HAND_VIEW_RESIDUALS'] = bool(task_cfg['filter_hand_view_residuals'])
    if 'hand_view_residual_max_ratio' in task_cfg:
        merged['HAND_VIEW_RESIDUAL_MAX_RATIO'] = float(task_cfg['hand_view_residual_max_ratio'])
    if 'hand_view_residual_abs_floor' in task_cfg:
        merged['HAND_VIEW_RESIDUAL_ABS_FLOOR'] = float(task_cfg['hand_view_residual_abs_floor'])
    if 'hand_view_residual_min_keep' in task_cfg:
        merged['HAND_VIEW_RESIDUAL_MIN_KEEP'] = int(task_cfg['hand_view_residual_min_keep'])
    if 'HAND_VIEW_WEIGHT' in task_cfg:
        merged['HAND_VIEW_WEIGHT'] = task_cfg['HAND_VIEW_WEIGHT']
    if 'joint_limit_weight' in task_cfg:
        merged['JOINT_LIMIT_WEIGHT'] = task_cfg['joint_limit_weight']
    if 'param_lr_multipliers' in task_cfg:
        merged['PARAM_LR_MULTIPLIERS'] = task_cfg['param_lr_multipliers']
    if 'hand_lr_multiplier' in task_cfg:
        merged['HAND_LR_MULTIPLIER'] = task_cfg['hand_lr_multiplier']
    # global pose 组合重试参数覆盖
    if 'global_pose_fail_rel_increase' in task_cfg:
        merged['GLOBAL_POSE_FAIL_REL_INCREASE'] = task_cfg['global_pose_fail_rel_increase']
    if 'global_pose_max_comb_size' in task_cfg:
        merged['GLOBAL_POSE_MAX_COMB_SIZE'] = int(task_cfg['global_pose_max_comb_size'])

    # 手部阶段是否冻结身体整体参数
    if 'freeze_body_in_hand_stages' in task_cfg:
        merged['FREEZE_BODY_IN_HAND_STAGES'] = bool(task_cfg['freeze_body_in_hand_stages'])
    # scale 参数是否参与优化
    if 'optimize_scale' in task_cfg:
        merged['OPTIMIZE_SCALE'] = bool(task_cfg['optimize_scale'])
    if 'batch_coarse_search' in task_cfg:
        merged['BATCH_COARSE_SEARCH'] = bool(task_cfg['batch_coarse_search'])
    if 'cache_body_verts_in_hand' in task_cfg:
        merged['CACHE_BODY_VERTS_IN_HAND'] = bool(task_cfg['cache_body_verts_in_hand'])
    if 'batch_view_projection' in task_cfg:
        merged['BATCH_VIEW_PROJECTION'] = bool(task_cfg['batch_view_projection'])
    if 'coarse_search_batch_size' in task_cfg:
        merged['COARSE_SEARCH_BATCH_SIZE'] = int(task_cfg['coarse_search_batch_size'])
    # [Step 0-H] SMPLest-X init 开关
    if 'init_from_smplestx' in task_cfg:
        merged['INIT_FROM_SMPLESTX'] = bool(task_cfg['init_from_smplestx'])
    if 'smplestx_init_dir' in task_cfg:
        merged['SMPLESTX_INIT_DIR'] = task_cfg['smplestx_init_dir']
    if 'smplestx_init_view' in task_cfg:
        merged['SMPLESTX_INIT_VIEW'] = int(task_cfg['smplestx_init_view'])
    if 'smplestx_init_inject_hands' in task_cfg:
        merged['SMPLESTX_INIT_INJECT_HANDS'] = bool(task_cfg['smplestx_init_inject_hands'])
    if 'smplestx_fix_hands' in task_cfg:
        merged['SMPLESTX_FIX_HANDS'] = bool(task_cfg['smplestx_fix_hands'])
    if 'smplestx_init_body_pose_mode' in task_cfg:
        merged['SMPLESTX_INIT_BODY_POSE_MODE'] = str(task_cfg['smplestx_init_body_pose_mode']).lower()

    return Config(
        MODEL_FOLDER=merged['MODEL_FOLDER'],
        VPOSER_MODEL_DIR=merged['VPOSER_MODEL_DIR'],
        exp=merged['exp'],
        data=merged['data'],
        ours_suffix=merged['ours_suffix'],
        TARGET_SIZE=merged['TARGET_SIZE'],
        device_id=gpu_id,
        RANDOM_SEED=merged.get('RANDOM_SEED', None),
        BODY_PRIOR=merged.get('BODY_PRIOR', 'dposerx'),
        HAND_PRIOR=merged.get('HAND_PRIOR', 'dposerx'),
        HAND_PCA_COMPS=merged.get('HAND_PCA_COMPS', 12),
        HAND_MANO_WEIGHT=merged.get('HAND_MANO_WEIGHT', 0.1),
        DPOSER_WEIGHT=merged.get('DPOSER_WEIGHT', 0.1),
        DPOSER_BODY_MULT=merged.get('DPOSER_BODY_MULT', 1.0),
        DPOSER_HAND_MULT=merged.get('DPOSER_HAND_MULT', 0.1),
        ENABLE_VIS=merged['ENABLE_VIS'],
        ENABLE_IMG_VIS=merged['ENABLE_IMG_VIS'],
        output_prefix=merged['output_prefix'],
        OPTIMIZER=merged['OPTIMIZER'],
        tts_count=merged['TTS_COUNT'],
        SKIP_HAND_FIT=merged['SKIP_HAND_FIT'],
        SKIP_HANDS_C=merged['SKIP_HANDS_C'],
        SPLIT_HAND_FIT=merged['SPLIT_HAND_FIT'],
        USE_UNCERTAINTY_WEIGHT=merged['USE_UNCERTAINTY_WEIGHT'],
        FILTER_VIEWS=merged['FILTER_VIEWS'],
        MIN_VALID_FACE_RATIO_BODY=merged['MIN_VALID_FACE_RATIO_BODY'],
        MIN_VALID_FACE_RATIO_HAND=merged['MIN_VALID_FACE_RATIO_HAND'],
        MAX_AVG_UNCERTAINTY=merged['MAX_AVG_UNCERTAINTY'],
        FILTER_HAND_VIEW_RESIDUALS=merged['FILTER_HAND_VIEW_RESIDUALS'],
        HAND_VIEW_RESIDUAL_MAX_RATIO=merged['HAND_VIEW_RESIDUAL_MAX_RATIO'],
        HAND_VIEW_RESIDUAL_ABS_FLOOR=merged['HAND_VIEW_RESIDUAL_ABS_FLOOR'],
        HAND_VIEW_RESIDUAL_MIN_KEEP=merged['HAND_VIEW_RESIDUAL_MIN_KEEP'],
        HAND_VIEW_WEIGHT=merged['HAND_VIEW_WEIGHT'],
        JOINT_LIMIT_WEIGHT=merged['JOINT_LIMIT_WEIGHT'],
        PARAM_LR_MULTIPLIERS=merged.get('PARAM_LR_MULTIPLIERS', {}),
        HAND_LR_MULTIPLIER=merged.get('HAND_LR_MULTIPLIER', 1.0),
        BODY_VIEWS=merged['BODY_VIEWS'],
        GLOBAL_POSE_FAIL_REL_INCREASE=merged['GLOBAL_POSE_FAIL_REL_INCREASE'],
        GLOBAL_POSE_MAX_COMB_SIZE=merged['GLOBAL_POSE_MAX_COMB_SIZE'],
        FREEZE_BODY_IN_HAND_STAGES=merged.get('FREEZE_BODY_IN_HAND_STAGES', True),
        OPTIMIZE_SCALE=merged.get('OPTIMIZE_SCALE', True),
        BATCH_COARSE_SEARCH=merged.get('BATCH_COARSE_SEARCH', True),
        CACHE_BODY_VERTS_IN_HAND=merged.get('CACHE_BODY_VERTS_IN_HAND', True),
        BATCH_VIEW_PROJECTION=merged.get('BATCH_VIEW_PROJECTION', True),
        COARSE_SEARCH_BATCH_SIZE=merged.get('COARSE_SEARCH_BATCH_SIZE', 64),
        INIT_FROM_SMPLESTX=merged.get('INIT_FROM_SMPLESTX', False),
        SMPLESTX_INIT_DIR=merged.get('SMPLESTX_INIT_DIR', None),
        SMPLESTX_INIT_VIEW=int(merged.get('SMPLESTX_INIT_VIEW', 0)),
        SMPLESTX_INIT_INJECT_HANDS=bool(merged.get('SMPLESTX_INIT_INJECT_HANDS', False)),
        SMPLESTX_FIX_HANDS=bool(merged.get('SMPLESTX_FIX_HANDS', False)),
        SMPLESTX_INIT_BODY_POSE_MODE=str(merged.get('SMPLESTX_INIT_BODY_POSE_MODE', 'warmstart')).lower(),
    )

if __name__ == "__main__":
    # ==================== CLI ====================
    import argparse
    _ap = argparse.ArgumentParser(
        description="DiffProxy stage-2: fit SMPL-X to generated uv/seg proxies (multi-view).")
    _ap.add_argument("--dataset_root", required=True,
                     help="root that contains evaluate/<dataset>/ (inputs & outputs live under it)")
    _ap.add_argument("--dataset", required=True, help="dataset name, e.g. data_rich_4v_interval2")
    _ap.add_argument("--exp", default="sd21-ep30", help="proxy experiment name (matches stage-1 output dir)")
    _ap.add_argument("--ours_suffix", default="_4v", help="ours dir suffix -> ours<suffix>/<exp>/")
    _ap.add_argument("--output_prefix", default="pred", help="output name prefix")
    _ap.add_argument("--tts_count", default="03_v00",
                     help="proxy sampling tag: reads tempt_tts<tag>_single/, writes <prefix>-tts<tag>_* (empty = tempt_single/)")
    _ap.add_argument("--body_views", type=int, default=4)
    _ap.add_argument("--body_prior", default="dposerx", choices=["dposerx", "vposer", "none"])
    _ap.add_argument("--hand_prior", default="dposerx")
    _ap.add_argument("--optimizer", default="adam")
    _ap.add_argument("--hand_lr_multiplier", type=float, default=10.0,
                     help="hand-stage LR multiplier (release default: 10.0)")
    _ap.add_argument("--dposer_hand_mult", type=float, default=0.1,
                     help="DPoser hand-prior weight multiplier (release default: 0.1)")
    _ap.add_argument("--no_hand_residual_filter", action="store_true",
                     help="disable robust hand-view filtering based on fitted reprojection residuals")
    _ap.add_argument("--hand_residual_ratio", type=float, default=4.0,
                     help="drop a hand view above this multiple of its hand-group median")
    _ap.add_argument("--hand_residual_floor", type=float, default=50.0,
                     help="absolute residual floor below which a view is never dropped")
    _ap.add_argument("--hand_residual_min_keep", type=int, default=2,
                     help="minimum views retained for each hand group")
    _ap.add_argument("--init_from_smplestx", action="store_true",
                     help="warm-start betas/body_pose from SMPLest-X params (stage 0-H)")
    _ap.add_argument("--smplestx_init_dir", default=None,
                     help="dir with {sid:04d}_{view:02d}.npz init params; default <dataset>/smplestx/params")
    _ap.add_argument("--smplestx_init_view", type=int, default=0)
    _ap.add_argument("--smplestx_inject_hands", action="store_true",
                     help="also initialize wrist roots and 45D left/right hand poses from SMPLest-X")
    _ap.add_argument("--smplestx_fix_hands", action="store_true",
                     help="freeze SMPLest-X wrist/finger initialization and skip hand fitting stages")
    _ap.add_argument("--smplestx_body_pose_mode", choices=("off", "warmstart", "full"),
                     default="warmstart",
                     help="SMPLest-X body-pose initialization mode (hand-only ablations use off)")
    _ap.add_argument("--skip_hands_c", action="store_true",
                     help="skip the final finger-refinement stage while retaining Hands-A wrist/palm fitting")
    _ap.add_argument("--gpu", type=int, default=0, help="torch CUDA device index")
    _ap.add_argument("--seed", type=int, default=None,
                     help="base random seed; each subject uses seed + subject_id")
    _ap.add_argument("--subject_start", type=int, default=None,
                     help="only fit subjects in [start, end) — multi-GPU: give each card a disjoint range")
    _ap.add_argument("--subject_end", type=int, default=None, help="exclusive end of the subject range")
    _ap.add_argument("--subjects", default=None,
                     help="comma-separated subject IDs; mutually exclusive with --subject_start/--subject_end")
    _args = _ap.parse_args()
    if _args.subjects and (_args.subject_start is not None or _args.subject_end is not None):
        _ap.error("--subjects is mutually exclusive with --subject_start/--subject_end")
    _subject_ids = [int(subject_id) for subject_id in parse_subject_ids(_args.subjects)]

    # ==================== 路径配置 ====================
    # DPoser-X 根目录
    DPOSER_ROOT = os.path.join(PROJECT_ROOT, "dependents/DPoser-X")
    
    # human_body_prior（VPoser）源码路径
    HBP_SRC_PATH = os.path.join(PROJECT_ROOT, "dependents/human_body_prior/src")
    
    # SMPL-X 模型文件夹路径
    MODEL_FOLDER = os.path.join(PROJECT_ROOT, "data")
    
    # VPoser 模型目录
    VPOSER_MODEL_DIR = os.path.join(PROJECT_ROOT, "dependents/human_body_prior/support_data/dowloads/vposer_v2_05")
    
    # ---------- 数据集相关路径 ----------
    # 数据集根目录（包含 evaluate 等子目录，存放输入输出文件）
    DATASET_ROOT = _args.dataset_root
    
    # UV 模板文件目录（包含 obj 模板文件）
    UV_TEMPLATE_DIR = os.path.join(PROJECT_ROOT, "data")
    
    # 缓存目录（存放 UV group 预处理缓存）
    CACHE_DIR = os.path.join(PROJECT_ROOT, "cache")
    
    # Optional model cache. Leave unset to read licensed assets in place.
    # Point DIFFPROXY_CACHE_DIR to a data disk when local caching is desired.
    _cache_root = os.environ.get("DIFFPROXY_CACHE_DIR")
    DPOSER_CACHE_DIR = os.path.join(_cache_root, "dposer_x") if _cache_root else None
    SMPLX_CACHE_DIR = os.path.join(_cache_root, "smplx") if _cache_root else None
    
    # 身体模板 obj 文件路径
    TEMPLATE_OBJ_PATH = os.path.join(UV_TEMPLATE_DIR, "diffproxy_uv_template.obj")

    # Hand-split UV topology template (provided separately).
    HAND_SPLIT_OBJ_PATH = os.path.join(UV_TEMPLATE_DIR, "diffproxy_hand_uv_template.obj")
    
    # ==================== 更新全局变量和配置 ====================
    # 构建路径配置字典（用于传递给多进程子进程）
    PATH_CONFIG = {
        'DPOSER_ROOT': DPOSER_ROOT,
        'HBP_SRC_PATH': HBP_SRC_PATH,
        'MODEL_FOLDER': MODEL_FOLDER,
        'VPOSER_MODEL_DIR': VPOSER_MODEL_DIR,
        'DATASET_ROOT': DATASET_ROOT,
        'UV_TEMPLATE_DIR': UV_TEMPLATE_DIR,
        'CACHE_DIR': CACHE_DIR,
        'DPOSER_CACHE_DIR': DPOSER_CACHE_DIR,
        'SMPLX_CACHE_DIR': SMPLX_CACHE_DIR,
        'TEMPLATE_OBJ_PATH': TEMPLATE_OBJ_PATH,
        'HAND_SPLIT_OBJ_PATH': HAND_SPLIT_OBJ_PATH,
    }
    
    # 更新模块级全局变量（本模块自身；旧版误 import 兄弟文件 mesh_fitting.py，已修正）
    _self_module = sys.modules[__name__]
    _self_module.DPOSER_ROOT = DPOSER_ROOT
    _self_module.HBP_SRC_PATH = HBP_SRC_PATH
    _self_module.MODEL_FOLDER = MODEL_FOLDER
    _self_module.DATASET_ROOT = DATASET_ROOT
    _self_module.UV_TEMPLATE_DIR = UV_TEMPLATE_DIR
    _self_module.CACHE_DIR = CACHE_DIR
    _self_module.DPOSER_CACHE_DIR = DPOSER_CACHE_DIR
    _self_module.SMPLX_CACHE_DIR = SMPLX_CACHE_DIR
    _self_module.TEMPLATE_OBJ_PATH = TEMPLATE_OBJ_PATH
    _self_module.HAND_SPLIT_OBJ_PATH = HAND_SPLIT_OBJ_PATH
    
    # 更新 DEFAULT_CONFIG 中的路径
    DEFAULT_CONFIG['MODEL_FOLDER'] = MODEL_FOLDER
    DEFAULT_CONFIG['VPOSER_MODEL_DIR'] = VPOSER_MODEL_DIR
    
    # 设置模块搜索路径并导入依赖
    _setup_module_paths()
    
    # 缓存文件到本地快速存储（首次运行会复制，之后直接使用缓存）
    _cache_dposer_files()
    _cache_smplx_files()
    
    # 初始化 UV 模板张量
    _init_uv_templates()
    
    # ------------ 任务（由 CLI 构建）------------
    # smplxinit（stage 0-H）说明：--init_from_smplestx 时用 SMPLest-X 回归参数 warm-start
    # （只注入 betas + body_pose，不含 global_orient/trans/scale；初值缺失时自动回退 zero-init）。
    _task = {"exp": _args.exp, "data": _args.dataset, "ours_suffix": _args.ours_suffix,
             "output_prefix": _args.output_prefix, "tts_count": _args.tts_count,
             "body_views": _args.body_views, "body_prior": _args.body_prior,
             "hand_prior": _args.hand_prior, "optimizer": _args.optimizer,
             "random_seed": _args.seed,
             "skip_hands_c": _args.skip_hands_c,
             "hand_lr_multiplier": _args.hand_lr_multiplier,
             "dposer_hand_mult": _args.dposer_hand_mult,
             "filter_hand_view_residuals": not _args.no_hand_residual_filter,
             "hand_view_residual_max_ratio": _args.hand_residual_ratio,
             "hand_view_residual_abs_floor": _args.hand_residual_floor,
             "hand_view_residual_min_keep": _args.hand_residual_min_keep}
    if _args.init_from_smplestx:
        _task.update({"init_from_smplestx": True,
                      "smplestx_init_view": _args.smplestx_init_view,
                      "smplestx_init_inject_hands": _args.smplestx_inject_hands,
                      "smplestx_fix_hands": _args.smplestx_fix_hands,
                      "smplestx_init_body_pose_mode": _args.smplestx_body_pose_mode})
        if _args.smplestx_init_dir:
            _task["smplestx_init_dir"] = _args.smplestx_init_dir
    # subject range -> the existing task-level slicing consumed by run_tasks_single_thread
    if _args.subject_start is not None:
        _task["subject_start"] = _args.subject_start
    if _args.subject_end is not None:
        _task["subject_end"] = _args.subject_end
    if _subject_ids:
        _task["subject_filter"] = _subject_ids
    TASKS = [_task]
    run_tasks_single_thread(TASKS, gpu_id=_args.gpu)

