"""
MV-Adapter测试脚本 - 支持单线程和多线程切换

主要功能：
1. 支持单线程和多线程模式切换，方便debug
2. 支持多GPU并行处理（多线程模式）
3. 支持自定义VAE decoder

使用方法：
- 将 USE_MULTIPROCESSING 设置为 False 启用单线程模式（适合debug）
- 将 USE_MULTIPROCESSING 设置为 True 启用多进程多GPU并行模式
"""
DEFAULT_SAMPLE_SEED_OFFSET = 0
import os
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
import sys
import torch
import gc
import torchvision.utils as vutils
import numpy as np
from PIL import Image
import glob
import json
import math
import torch.nn.functional as F
import random
from sklearn.feature_extraction.image import reconstruct_from_patches_2d
import pytorch_lightning as pl
import multiprocessing as mp
from queue import Empty as QueueEmpty
import subprocess
from mvadapter.utils.config import load_config
from mvadapter.utils.core import find
from mvadapter.utils.geometry import get_plucker_embeds_from_cameras
from typing import Optional, List, Tuple, Set, Dict
import yaml
import tempfile
import shutil
import cv2
from tqdm import tqdm
import re
import itertools
import os.path as osp

# Reload the merged proxy VAE after restoring the main checkpoint.
from proxy_vae_loader import reload_proxy_vae
from demo_utils import filter_subject_items, parse_subject_ids



  
# 与 multiview_v3 保持一致的 bbox 扩张比例
DEFAULT_BBOX_RATIO = 1.25

# 手脚等局部部位的 bbox 扩张比例（独立于身体）
HAND_FOOT_BBOX_RATIO = 2.0

# 当 bbox 边长过小时视为无效（直接回退到不裁剪/身体裁剪）
MIN_BBOX_SIDE_RATIO = 0.02

# TTS/多次前向相关配置（用于第二步聚合，可在调用时覆盖）
BASE_SEED = 1314
SEED_STEP = 1000

# 每个部位拥有的基础视角数（如 1、4、8 等），用于第二步聚合与回贴
NUM_BASE_CAMS = 4


def ensure_valid_bbox_norm(bbox_norm: Optional[List[float]],
                           min_side_ratio: float = MIN_BBOX_SIDE_RATIO) -> Optional[List[float]]:
    if bbox_norm is None:
        return None

    x_min, y_min, x_max, y_max = bbox_norm
    bw = max(0.0, x_max - x_min)
    bh = max(0.0, y_max - y_min)

    if bw <= 0.0 or bh <= 0.0:
        return None

    if max(bw, bh) < float(min_side_ratio):
        return None

    return [float(x_min), float(y_min), float(x_max), float(y_max)]





def crop_scale_center_tensor(tensor: torch.Tensor, bbox_norm, target_height: int, target_width: int,
                             default_ratio: float = DEFAULT_BBOX_RATIO):
    """
    依据 bbox_v2 坐标裁剪并缩放 tensor：
    1. 将 bbox 四边以 default_ratio 做等比扩张；
    2. 若扩张后任意边触及图像边界，则返回 None（不做处理）；
    3. 取扩张后区域，保持长边与目标尺寸一致等比缩放；
    4. 将缩放后的 patch 置于黑色背景中心。
    返回处理后的 tensor，若返回 None 表示跳过。
    
    与 multiview_v3.py 完全一致的实现
    """
    C, h_ori, w_ori = tensor.shape
    h_tgt, w_tgt = target_height, target_width
    x_min, y_min, x_max, y_max = bbox_norm

    # 计算 bbox 中心与宽高
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    bw = x_max - x_min
    bh = y_max - y_min
    side = max(bw, bh) * default_ratio

    new_xmin_tmp = cx - side / 2.0
    new_xmax_tmp = cx + side / 2.0
    new_ymin_tmp = cy - side / 2.0
    new_ymax_tmp = cy + side / 2.0

    new_xmin = max(new_xmin_tmp, 0.)
    new_xmax = min(new_xmax_tmp, 1.)
    new_ymin = max(new_ymin_tmp, 0.)
    new_ymax = min(new_ymax_tmp, h_ori/w_ori)

    # if side <= 1e-8:
    #     return K_orig.clone()

    tgt_xmin = round((new_xmin - new_xmin_tmp) * w_tgt / side)
    tgt_xmax = round((new_xmax - new_xmin_tmp) * w_tgt / side)
    tgt_ymin = round((new_ymin - new_ymin_tmp) * h_tgt / side)
    tgt_ymax = round((new_ymax - new_ymin_tmp) * h_tgt / side)

    # 裁剪 patch
    crop_x_start = round(new_xmin * w_ori)
    crop_x_end = round(new_xmax * w_ori)
    crop_y_start = round(new_ymin * w_ori)
    crop_y_end = round(new_ymax * w_ori)

    # 如果裁剪区域过小（或为空），直接回退为不裁剪
    if crop_x_end - crop_x_start <= 1 or crop_y_end - crop_y_start <= 1:
        return None

    patch = tensor[:, crop_y_start:crop_y_end, crop_x_start:crop_x_end]

    # patch_scaled = F.interpolate(patch.unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False).squeeze(0)
    target_patch_h = tgt_ymax - tgt_ymin
    target_patch_w = tgt_xmax - tgt_xmin
    if target_patch_w <= 1 or target_patch_h <= 1:
        return None

    patch_scaled = F.interpolate(patch.unsqueeze(0), size=(target_patch_h, target_patch_w), mode='nearest-exact').squeeze(0)

    # 创建黑色背景并粘贴到中心
    result = torch.zeros(tensor.shape[0], h_tgt, w_tgt)
    result[:, tgt_ymin:tgt_ymax, tgt_xmin:tgt_xmax] = patch_scaled
    return result


def load_bbox(path_like: str, image_width: int, image_height: int,
              min_side_ratio: float = MIN_BBOX_SIDE_RATIO,
              allow_zero_ratio: bool = False):
    """
    给定任意同主体同视角的路径（image/camera/kpts），在同一组的 bbox 目录中查找同名的 .json 文件，
    并从中提取 "tlwh" 格式的 bbox，转换为标准化坐标 [xmin, ymin, xmax, ymax]。
    """
    # 允许传入 camera/kpts 路径：仅使用上级上级目录和同名基名
    parent_dir = os.path.dirname(os.path.dirname(path_like))
    bbox_dir = os.path.join(parent_dir, "bbox")
    bbox_path = os.path.join(bbox_dir, os.path.splitext(os.path.basename(path_like))[0] + ".json")


    with open(bbox_path, 'r') as f:
        data = json.load(f)
    tlwh = data.get("tlwh")
    t, l, w, h = tlwh
    # 转换为标准化坐标
    x_min = t / image_width
    y_min = l / image_width
    x_max = (t + w) / image_width
    y_max = (l + h) / image_width
    bbox_norm = [x_min, y_min, x_max, y_max]

    # 在某些标注中，(0,0,0,0) 表示无检测 → 允许返回
    if allow_zero_ratio and (w == 0 or h == 0):
        return None

    return ensure_valid_bbox_norm(bbox_norm, min_side_ratio=min_side_ratio)



def update_intrinsic_matrix(K_orig: torch.Tensor, 
                           bbox_norm, 
                           h_ori: int, 
                           w_ori: int, 
                           h_tgt: int, 
                           w_tgt: int, 
                           default_ratio: float = DEFAULT_BBOX_RATIO) -> torch.Tensor:
    """
    根据bbox裁剪和缩放更新相机内参矩阵
    
    与 multiview_v3.py 完全一致的实现
    
    Args:
        K_orig: 原始内参矩阵 (3x3)
        bbox_norm: 标准化bbox坐标 (x_min, y_min, x_max, y_max)
        h_ori, w_ori: 原始图像尺寸
        h_tgt, w_tgt: 目标图像尺寸  
        default_ratio: bbox扩张比例
        
    Returns:
        更新后的内参矩阵
    """
    x_min_norm, y_min_norm, x_max_norm, y_max_norm = bbox_norm

    # 1. 使用标准化坐标计算bbox中心和扩张
    cx = (x_min_norm + x_max_norm) / 2.0
    cy = (y_min_norm + y_max_norm) / 2.0
    bw = x_max_norm - x_min_norm
    bh = y_max_norm - y_min_norm
    side = max(bw, bh) * default_ratio
    
    # 2. 计算扩张后的区域（标准化坐标）
    new_xmin_tmp = cx - side / 2.0
    new_xmax_tmp = cx + side / 2.0
    new_ymin_tmp = cy - side / 2.0
    new_ymax_tmp = cy + side / 2.0
    
    # 3. 边界检查和调整
    new_xmin = max(new_xmin_tmp, 0.)
    new_xmax = min(new_xmax_tmp, 1.)
    new_ymin = max(new_ymin_tmp, 0.)
    new_ymax = min(new_ymax_tmp, h_ori/w_ori)
    
    # 4. 计算目标区域坐标
    tgt_xmin = round((new_xmin - new_xmin_tmp) * w_tgt / side)
    tgt_xmax = round((new_xmax - new_xmin_tmp) * w_tgt / side)
    tgt_ymin = round((new_ymin - new_ymin_tmp) * h_tgt / side)
    tgt_ymax = round((new_ymax - new_ymin_tmp) * h_tgt / side)
    
    # 5. 构建变换矩阵A
    # 计算实际的裁剪和缩放参数（与图像处理完全一致）
    # 实际裁剪区域的像素坐标
    actual_crop_x_start = round(new_xmin * w_ori)
    actual_crop_x_end = round(new_xmax * w_ori)
    actual_crop_y_start = round(new_ymin * w_ori)
    actual_crop_y_end = round(new_ymax * w_ori)
    
    # 实际patch尺寸
    actual_patch_w = actual_crop_x_end - actual_crop_x_start
    actual_patch_h = actual_crop_y_end - actual_crop_y_start
    
    # 目标区域尺寸
    target_w = tgt_xmax - tgt_xmin
    target_h = tgt_ymax - tgt_ymin
    
    # 实际缩放比例（与F.interpolate一致）
    if actual_patch_w <= 0 or actual_patch_h <= 0 or target_w <= 0 or target_h <= 0:
        return K_orig.clone()

    scale_x = target_w / actual_patch_w
    scale_y = target_h / actual_patch_h
    
    # 变换矩阵：先缩放后平移
    # x_new = scale_x * (x - crop_start_x) + tgt_xmin
    #       = scale_x * x + (tgt_xmin - scale_x * crop_start_x)
    offset_x = tgt_xmin - scale_x * actual_crop_x_start
    offset_y = tgt_ymin - scale_y * actual_crop_y_start
    
    A = torch.tensor([
        [scale_x, 0.0, offset_x],
        [0.0, scale_y, offset_y],
        [0.0, 0.0, 1.0]
    ], dtype=K_orig.dtype, device=K_orig.device)
    
    return A @ K_orig


def load_obj(path: str):
    """
    与 `multiview_v3.SingleViewDataset.load_obj` 保持一致：
    - 仅解析 'v ' 顶点与 'f ' 面，faces 返回 list[list[int]]，vertices 返回 FloatTensor
    - faces 可能包含非三角形，保持 list 形式
    """
    vertices = []
    faces = []

    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                vertices.append([float(v) for v in line.strip().split()[1:]])
            elif line.startswith('f '):
                face_v_indices = []
                for v_def in line.strip().split()[1:]:
                    face_v_indices.append(int(v_def.split('/')[0]) - 1)
                faces.append(face_v_indices)

    return torch.tensor(vertices, dtype=torch.float32), faces



def load_camera(cam_path: str, use_bbox: bool, target_height: int, target_width: int, original_height, original_width, bbox_override_norm: Optional[List[float]] = None):
    """
    加载相机参数并返回字典：{"extrinsic": c2w_4x4, "intrinsic": K_3x3}
    - 优先读取与图像同名的 .npz（包含 K,R,T）。若为世界到相机 (w2c) 则取逆得到 c2w。
    - 若无 .npz，则回退读取 .txt 外参与 info.json 的 fov 来构造 K。
    - 若 use_bbox=True，则对 K 按 bbox 仿射修正（与 multiview_v3 保持一致）。
    """
    base = os.path.splitext(cam_path)[0]
    npz_path = base + ".npz"
    # extrinsic = None
    # K = None

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"缺少相机参数: {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    K_np = np.array(data['K']).astype(np.float32)
    R = np.array(data['R']).astype(np.float32)
    T = np.array(data['T']).astype(np.float32).reshape(3, 1)/1000.

    # 构造 w2c，再求 c2w
    w2c4 = np.eye(4, dtype=np.float32)
    w2c4[:3, :3] = R
    w2c4[:3, 3:4] = T
    c2w4 = np.linalg.inv(w2c4)
    extrinsic = torch.from_numpy(c2w4)
    K = torch.from_numpy(K_np)

    # 如果启用 bbox，需要对 K 做缩放和平移修正
    if use_bbox:
        h_ori, w_ori = original_height, original_width
        h_tgt, w_tgt = target_height, target_width
        
        # 获取标准化bbox坐标
        if bbox_override_norm is not None:
            hand_bbox_norm = ensure_valid_bbox_norm(bbox_override_norm)
        else:
            hand_bbox_norm = None

        if hand_bbox_norm is not None:
            bbox_norm = hand_bbox_norm
            ratio_for_this = 1.0
        else:
            base_bbox = load_bbox(cam_path, w_ori, h_ori)
            if base_bbox is None:
                bbox_norm = [0.0, 0.0, 1.0, h_ori / w_ori]
            else:
                bbox_norm = base_bbox
            ratio_for_this = DEFAULT_BBOX_RATIO

        K = update_intrinsic_matrix(K, bbox_norm, h_ori, w_ori, h_tgt, w_tgt, ratio_for_this)


    return {"extrinsic": extrinsic, "intrinsic": K}


def load_image(img_path: str, use_bbox: bool, target_height: int, target_width: int, bbox_override_norm: Optional[List[float]] = None):
    """
    从路径加载图像，可选地裁剪和缩放，并返回 tensor.
    """
    img = Image.open(img_path).convert("RGBA")
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    rgb = img_tensor[:, :, :3]
    rgb = rgb.permute(2, 0, 1)  # [3, H, W]

    if use_bbox:
        # 获取原始图像尺寸
        h_ori, w_ori = img_tensor.shape[:2]
        if bbox_override_norm is not None:
            bbox_norm = ensure_valid_bbox_norm(bbox_override_norm)
        else:
            bbox_norm = load_bbox(img_path, w_ori, h_ori)

        if bbox_norm is None:
            # bbox 无效时，直接缩放到目标尺寸
            rgb = F.interpolate(rgb.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)
            return rgb.unsqueeze(0)
        # 手部 override 已含完整扩张（_expand_square 1.5x），不再额外放大；身体使用默认比例
        ratio_for_this = 1.0 if bbox_override_norm is not None else DEFAULT_BBOX_RATIO
        processed = crop_scale_center_tensor(rgb, bbox_norm, target_height, target_width, ratio_for_this)
        if processed is not None:
            rgb = processed
        else:
            # crop 失败时，回退到直接缩放到目标尺寸
            rgb = F.interpolate(rgb.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)
    else:
        # 不使用 bbox 时，直接缩放到目标分辨率
        rgb = F.interpolate(rgb.unsqueeze(0), size=(target_height, target_width), mode='bilinear', align_corners=False).squeeze(0)
            
    return rgb.unsqueeze(0)


# ===== 手脚 bbox 检测与辅助工具（复用 get_pred_bbox_foot_hand 的逻辑） =====
_SEG_PALETTE = np.array([
    [0, 0, 0],
    [0, 0, 255],
    [0, 255, 0],
    [255, 0, 0],
    [255, 255, 0],
    [255, 0, 255],  # left (coarse hand)
    [128, 128, 0],
    [255, 255, 255],
    [0, 255, 255],
    [0, 128, 128],  # right (coarse hand)
    [128, 0, 128],
    [128, 128, 255],

    [64, 255, 128],  # 12
    [255, 64, 128],
    [192, 255, 128],
    [0, 0, 128],
    [0, 128, 0],
    [0, 128, 255],

    [128, 0, 0],    # 18
    [128, 0, 255],
    [128, 255, 0],
    [128, 255, 255],
    [255, 128, 0],
    [255, 128, 255]
], dtype=np.uint8)

PART_LABELS = {
    "left_hand": 5,
    "left_foot": 7,
    "right_hand": 9,
    "right_foot": 10,
}

def _quantize_seg_by_palette(seg_rgb: np.ndarray) -> np.ndarray:
    h, w, _ = seg_rgb.shape
    seg_flat = seg_rgb.reshape(-1, 1, 3).astype(np.int32)
    pal = _SEG_PALETTE.reshape(1, -1, 3).astype(np.int32)
    dist2 = np.sum((seg_flat - pal) ** 2, axis=2)
    idx = np.argmin(dist2, axis=1).astype(np.uint8)
    return idx.reshape(h, w)

def _quantize_seg_by_palette_subset(seg_rgb: np.ndarray, allowed_ids: List[int]) -> np.ndarray:
    """
    将 seg 颜色图按调色板最近颜色量化，但仅在 allowed_ids 子集中匹配。
    返回的值是与全局调色板对齐的 seg id（未被允许的 id 不会出现）。
    """
    if seg_rgb.dtype != np.uint8:
        seg_rgb = np.clip(seg_rgb, 0, 255).astype(np.uint8)
    h, w, _ = seg_rgb.shape
    if len(allowed_ids) == 0:
        return np.zeros((h, w), dtype=np.uint8)
    seg_flat = seg_rgb.reshape(-1, 1, 3).astype(np.int32)
    palette_full = _SEG_PALETTE.astype(np.int32)
    allowed_ids_arr = np.asarray(allowed_ids, dtype=np.int64)
    pal_sub = palette_full[allowed_ids_arr].reshape(1, -1, 3)
    dist2 = np.sum((seg_flat - pal_sub) ** 2, axis=2)  # (N,K)
    nearest_sub_idx = np.argmin(dist2, axis=1)  # (N,)
    nearest_global = allowed_ids_arr[nearest_sub_idx]
    return nearest_global.reshape(h, w).astype(np.uint8)

def _largest_cc(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.sum() == 0:
        return None
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=np.uint8)
    best = []
    neighbors = [(-1,0),(1,0),(0,-1),(0,1)]
    for y in range(h):
        for x in range(w):
            if mask[y, x] and visited[y, x] == 0:
                stack = [(y, x)]
                visited[y, x] = 1
                comp = []
                while stack:
                    cy, cx = stack.pop()
                    comp.append((cy, cx))
                    for dy, dx in neighbors:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < h and 0 <= nx < w and visited[ny, nx] == 0 and mask[ny, nx]:
                            visited[ny, nx] = 1
                            stack.append((ny, nx))
                if len(comp) > len(best):
                    best = comp
    if not best:
        return None
    out = np.zeros_like(mask)
    for yy, xx in best:
        out[yy, xx] = True
    return out

def _bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int,int,int,int]]:
    ys, xs = np.where(mask)
    if ys.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1

def _expand_square(xmin:int, ymin:int, xmax:int, ymax:int, w:int, h:int, scale:float=DEFAULT_BBOX_RATIO) -> Tuple[int,int,int,int]:
    cx = (xmin + xmax) / 2.0
    cy = (ymin + ymax) / 2.0
    side = int(round(max(xmax - xmin, ymax - ymin) * scale))
    nx0 = int(np.clip(round(cx - side/2.0), 0, w))
    nx1 = int(np.clip(round(cx + side/2.0), 0, w))
    ny0 = int(np.clip(round(cy - side/2.0), 0, h))
    ny1 = int(np.clip(round(cy + side/2.0), 0, h))
    return nx0, ny0, nx1, ny1

def _split_tiles_h(img: np.ndarray, num_tiles: int) -> List[np.ndarray]:
    h, w = img.shape[:2]
    assert w % num_tiles == 0
    tile_w = w // num_tiles
    return [img[:, i*tile_w:(i+1)*tile_w] for i in range(num_tiles)]


def get_plucker(cameras, target_height: int, target_width: int):
    """
    基于已加载的相机参数列表，计算与 multiview_v3 一致的 Plücker 条件：
    1) 使用传入的 cameras（含 bbox 修正后的 K 与外参）；
    2) 将 c2w 从 GL 转为 OpenCV；
    3) 由最终 K 推导 fov（弧度）；
    4) 调用 get_plucker_embeds_from_cameras 得到 (V, 6, H, W)。
    返回：Tensor，形状 (V, 6, H, W)，位于 CPU；后续与 batch 一起迁移到 GPU。
    """
    P = torch.diag(torch.tensor([1.0, -1.0, -1.0, 1.0], dtype=torch.float32))

    c2w_opencv_list = []
    fov_rad_list = []
    for cam in cameras:
        c2w_gl = cam["extrinsic"].to(dtype=torch.float32)
        c2w_opencv = c2w_gl @ P
        c2w_opencv_list.append(c2w_opencv)

        K_final = cam["intrinsic"].to(dtype=torch.float32)
        fx = K_final[0, 0]
        fy = K_final[1, 1]
        focal_avg = (fx + fy) / 2.0
        fov_rad = 2 * torch.atan(target_height / (2.0 * focal_avg))
        fov_rad_list.append(float(fov_rad))

    plucker_embeds = get_plucker_embeds_from_cameras(
        [t.cpu() for t in c2w_opencv_list],
        fov_rad_list,
        target_height,
    )
    return plucker_embeds

def prepare_rgb(img: Image.Image, img_path: str, background_color: torch.Tensor,
                use_bbox: bool, target_height: int, target_width: int, default_ratio: float = 1.):
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0
    rgb = img_tensor[:, :, :3] 
    rgb = rgb.permute(2, 0, 1)  # [3, H, W]

    processed = None
    if use_bbox:
        # 获取原始图像尺寸
        h_ori, w_ori = img_tensor.shape[:2]
        bbox_norm = load_bbox(img_path, w_ori, h_ori)
        if bbox_norm is None:
            return rgb.unsqueeze(0), processed
        processed = crop_scale_center_tensor(rgb, bbox_norm, target_height, target_width, default_ratio)
        if processed is not None:
            rgb = processed
    return rgb.unsqueeze(0), processed




# ===== 不确定性多次推理与选择工具 =====
def generate_unique_seeds_results(
    system,
    batch,
    all_seeds: List[int],
    sample_cache_dir: Optional[str] = None,
    sample_cache_prefix: Optional[str] = None,
) -> Dict[int, torch.Tensor]:
    """
    对给定的 seed 列表做去重推理，返回 {seed: tensor} 字典。
    相同 seed 只推理一次（先查磁盘缓存，再查内存池）。
    """
    use_sample_cache = (sample_cache_dir is not None) and (sample_cache_prefix is not None)
    if use_sample_cache:
        os.makedirs(sample_cache_dir, exist_ok=True)

    original_eval_seed = getattr(system.cfg, 'eval_seed', all_seeds[0] if all_seeds else 1314)
    seed_results: Dict[int, torch.Tensor] = {}

    # [SDEdit 实验] 保存 batch 供后续实验使用
    if os.environ.get("SAVE_BATCH_FOR_SDEDIT") == "1" and sample_cache_dir is not None and sample_cache_prefix is not None:
        os.makedirs(sample_cache_dir, exist_ok=True)
        batch_save = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        batch_save_path = os.path.join(sample_cache_dir, f"{sample_cache_prefix}_batch.pt")
        torch.save(batch_save, batch_save_path)
        print(f"[SDEdit] Saved batch to {batch_save_path}")

    unique_seeds = list(dict.fromkeys(all_seeds))
    cache_hit = 0
    cache_miss = 0

    for seed in unique_seeds:
        cache_path = None
        if use_sample_cache:
            cache_path = os.path.join(sample_cache_dir, f"{sample_cache_prefix}_seed{seed}.pt")
            if os.path.exists(cache_path):
                cached_out = torch.load(cache_path, map_location="cpu")
                seed_results[seed] = cached_out.detach().float().cpu().clone()
                cache_hit += 1
                continue

        system.cfg.eval_seed = seed
        pl.seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()
        with torch.no_grad():
            if hasattr(system, "pipeline"):
                system.pipeline.to(dtype=torch.float32)
            out = system.generate_images(batch)

            # [SDEdit 实验] 保存 batch 供后续实验使用
            if os.environ.get("SAVE_BATCH_FOR_SDEDIT") == "1":
                batch_save = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
                batch_save_path = os.path.join(sample_cache_dir, f"{sample_cache_prefix}_batch.pt")
                torch.save(batch_save, batch_save_path)
                print(f"[SDEdit] Saved batch to {batch_save_path}")

        out_cpu = out.detach().float().cpu().clone()
        if cache_path is not None:
            torch.save(out_cpu, cache_path)
        seed_results[seed] = out_cpu
        cache_miss += 1

    system.cfg.eval_seed = original_eval_seed
    print(f"[seed-pool] {sample_cache_prefix}: {len(unique_seeds)} unique seeds, {cache_hit} hit, {cache_miss} inferred")
    return seed_results


def collect_results_for_offset(
    seed_results: Dict[int, torch.Tensor],
    num_samples: int,
    sample_index_offset: int,
    base_seed: int = 1314,
) -> List[torch.Tensor]:
    """从全局 seed 结果池中，按指定 offset 取出有序结果列表。"""
    results = []
    for i in range(int(num_samples)):
        seed = int(base_seed) + (int(sample_index_offset) + i) * 1000
        results.append(seed_results[seed])
    return results


def calculate_uncertainty_maps(results: list, uv_views_len: int, view_types: Optional[List[str]] = None):
    """
    基于多次推理结果计算不确定性，并按视角选择最稳定（与均值差异最小）的样本。
    Args:
        results: List[Tensor]，每个张量形状为 [uv_views_len + uv_views_len, C, H, W]
        uv_views_len: UV 部分的视角数量（等于前半段长度）
    Returns:
        dict，包含 uv_best、seg_best、不确定性 mask 及所选样本索引
    """
    stacked = torch.stack(results, dim=0)  # [S, 2*V, C, H, W]（放在 CPU 上）

    uv_results = stacked[:, :uv_views_len]       # [S, V, C, H, W]
    seg_results = stacked[:, uv_views_len:]      # [S, V, C, H, W]

    # 均值/方差
    uv_mean = uv_results.mean(dim=0)             # [V, C, H, W]
    uv_var = uv_results.var(dim=0, unbiased=False)
    seg_mean = seg_results.mean(dim=0)
    seg_var = seg_results.var(dim=0, unbiased=False)

    # 不确定性 mask（通道均值的方差）
    uv_uncertainty_map = uv_var.mean(dim=1, keepdim=True)   # [V,1,H,W]
    seg_uncertainty_map = seg_var.mean(dim=1, keepdim=True) # [V,1,H,W]

    # 与均值的 L2 差异（逐像素计算）
    uv_pixel_diff = ((uv_results - uv_mean.unsqueeze(0)) ** 2)  # [S, V, C, H, W]
    seg_pixel_diff = ((seg_results - seg_mean.unsqueeze(0)) ** 2)  # [S, V, C, H, W]
    
    # 在通道维度求平均，得到逐像素的uncertainty
    uv_pixel_uncertainty = uv_pixel_diff.mean(dim=2)  # [S, V, H, W]
    seg_pixel_uncertainty = seg_pixel_diff.mean(dim=2)  # [S, V, H, W]
    
    # 联合选择方案：逐像素相乘UV和Seg的uncertainty，然后对所有像素求和
    joint_diffs = (uv_pixel_uncertainty * seg_pixel_uncertainty).flatten(2).sum(dim=2)  # [S, V]
    
    # 保留原始的全图差异用于单独分析（可选）
    uv_diffs = uv_pixel_diff.flatten(2).sum(dim=2)  # [S, V]
    seg_diffs = seg_pixel_diff.flatten(2).sum(dim=2)  # [S, V]
    # --- 改为“按组”联合选择：同一组（身体/当前手）使用同一次采样 ---
    V = int(uv_views_len)
    indices_all = list(range(V))
    groups = []
    if view_types is not None and len(view_types) >= V:
        body_idx = [i for i in indices_all if str(view_types[i]).lower().startswith("body")]
        hand_idx = [i for i in indices_all if "hand" in str(view_types[i]).lower()]
        if len(body_idx) > 0:
            groups.append(body_idx)
        if len(hand_idx) > 0:
            groups.append(hand_idx)
        covered = set().union(*groups) if len(groups) > 0 else set()
        remain = [i for i in indices_all if i not in covered]
        if len(remain) > 0:
            groups.append(remain)
    else:
        groups = [indices_all]
    joint_best_indices = [0] * V
    for g in groups:
        group_scores = joint_diffs[:, g].sum(dim=1)  # [S]
        best_s = int(torch.argmin(group_scores).item())
        for i in g:
            joint_best_indices[i] = best_s

    # 使用同一组索引选出 uv_best 与 seg_best（确保两者来自同一次采样）
    uv_best = torch.stack([uv_results[joint_best_indices[i], i] for i in range(V)], dim=0)
    seg_best = torch.stack([seg_results[joint_best_indices[i], i] for i in range(V)], dim=0)

    # 仍保留单独最优索引以便调试
    uv_best_indices = torch.argmin(uv_diffs, dim=0).tolist()   # [V]
    seg_best_indices = torch.argmin(seg_diffs, dim=0).tolist() # [V]

    # === 新的不确定性计算（Seg 按语义投票；UV 保持原有方差方案） ===
    # 视角类型：'body' 或 'hand'，用于决定允许的调色板子集
    if view_types is None:
        view_types = ["body"] * uv_views_len
    # 构造 body 与 hand 的允许颜色 id 集合
    palette_len = int(_SEG_PALETTE.shape[0])
    if palette_len > 12:
        body_ids = list(range(12))
        hand_ids = list(range(12, palette_len))
    else:
        body_ids = list(range(palette_len))
        hand_ids = list(range(palette_len))

    S = seg_results.shape[0]
    V, C, H, W = seg_results.shape[1:]
    seg_uncert_list = []
    for vi in range(V):
        vt = str(view_types[vi]).lower() if vi < len(view_types) else "body"
        allowed = hand_ids if ("hand" in vt) else body_ids
        # 将每次结果量化为 seg id（按子集）
        ids_stack = []
        for si in range(S):
            seg_now = seg_results[si, vi]  # [C,H,W]
            if seg_now.dim() == 3 and seg_now.size(0) >= 3:
                seg_rgb = (seg_now[:3].detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
            else:
                # 非标准形状，回退为全零
                seg_rgb = np.zeros((H, W, 3), dtype=np.float32)
            ids = _quantize_seg_by_palette_subset(seg_rgb, allowed)
            ids_stack.append(ids)
        if len(ids_stack) == 0:
            seg_uncert_list.append(torch.ones(1, H, W))
            continue
        ids_arr = np.stack(ids_stack, axis=0)  # (S,H,W)
        # 统计每个像素在 allowed 子集的最大出现次数
        counts_max = np.zeros((H, W), dtype=np.int32)
        for aid in allowed:
            counts_max = np.maximum(counts_max, np.sum(ids_arr == aid, axis=0))
        half = max(S / 2.0, 1e-8)
        # 若最大次数 <= S/2 → 不确定性为 1；否则按 1 - (max_count - S/2)/(S/2)
        uncert = np.where(
            counts_max <= half,
            1.0,
            1.0 - (counts_max - half) / half
        ).astype(np.float32)
        uncert = np.clip(uncert, 0.0, 1.0)
        seg_uncert_list.append(torch.from_numpy(uncert).unsqueeze(0))  # [1,H,W]
    seg_uncertainty_map = torch.stack(seg_uncert_list, dim=0)  # [V,1,H,W]

    return {
        "uv_best": uv_best,
        "seg_best": seg_best,
        "uv_uncertainty_mask": uv_uncertainty_map,
        "seg_uncertainty_mask": seg_uncertainty_map,
        "uv_best_indices": uv_best_indices,
        "seg_best_indices": seg_best_indices,
        # 新增联合最优索引与联合差值
        "joint_best_indices": joint_best_indices,
        "uv_diffs": uv_diffs,
        "seg_diffs": seg_diffs,
        "joint_diffs": joint_diffs,
    }



def ensure_dir(path: str):
    """创建目录（存在则忽略）。"""
    os.makedirs(path, exist_ok=True)


def should_keep_hand_view_by_seg_overlap(
    subject: str,
    view_idx: int,
    group_idx: int,
    base_cam: int,
    hand_bbox_tlwh,
    img_ori_shape,
    image_crop_dir: str,
    bbox_body_dir: str,
    overlap_thresh: float = 0.10,
) -> bool:
    """
    对单个手部视角做过滤：
      1. 使用手部 bbox（以原图坐标）在身体 seg 视角中找到对应区域；
      2. 判断该区域内是否存在当前手（左/右）的 seg 像素；
      3. 这些像素与当前手部视角 seg 像素的空间重合比例是否 > overlap_thresh。

    返回：
      True  -> 通过过滤，可以使用该手部视角；
      False -> 不通过过滤，应丢弃该手部视角。
    """
    H_ori, W_ori = img_ori_shape
    x_h, y_h, w_h, h_h = hand_bbox_tlwh
    if w_h <= 1 or h_h <= 1:
        return False

    # 身体 bbox：与第二步回贴逻辑保持一致
    file_key_body = f"{subject}_{base_cam:02d}"
    bbox_body_json = osp.join(bbox_body_dir, file_key_body + ".json")
    if not osp.exists(bbox_body_json):
        return False
    with open(bbox_body_json, "r") as f:
        bbox_body_dict = json.load(f)
    bbox_body_tlwh = bbox_body_dict.get("tlwh", None)
    if bbox_body_tlwh is None:
        return False
    x_b, y_b, w_b, h_b = bbox_body_tlwh
    if w_b <= 1 or h_b <= 1:
        return False

    # 读取身体 seg 视角图
    seg_body_path = osp.join(image_crop_dir, f"seg_{subject}_{base_cam:02d}_pred.png")
    if not osp.exists(seg_body_path):
        return False
    seg_body_bgr = cv2.imread(
        seg_body_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if seg_body_bgr is None:
        return False
    # 其它脚本按 RGB `_SEG_PALETTE` 生成；这里转回 RGB 再做颜色匹配
    seg_body = cv2.cvtColor(seg_body_bgr, cv2.COLOR_BGR2RGB)

    # 读取当前手部 seg 视角图
    seg_hand_path = osp.join(image_crop_dir, f"seg_{subject}_{view_idx:02d}_pred.png")
    if not osp.exists(seg_hand_path):
        return False
    seg_hand_bgr = cv2.imread(
        seg_hand_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
    )
    if seg_hand_bgr is None:
        return False
    seg_hand = cv2.cvtColor(seg_hand_bgr, cv2.COLOR_BGR2RGB)

    H_body, W_body = seg_body.shape[:2]
    H_hand, W_hand = seg_hand.shape[:2]
    if H_body <= 1 or W_body <= 1 or H_hand <= 1 or W_hand <= 1:
        return False

    # 身体视角的裁剪参数（与第二步 body 分支逻辑保持一致）
    x_min_b, y_min_b, x_max_b, y_max_b = x_b, y_b, x_b + w_b, y_b + h_b
    cx_b = (x_min_b + x_max_b) / 2.0
    cy_b = (y_min_b + y_max_b) / 2.0
    bw = (x_max_b - x_min_b) * DEFAULT_BBOX_RATIO
    bh = (y_max_b - y_min_b) * DEFAULT_BBOX_RATIO
    side = max(bw, bh)
    if side <= 1e-6:
        return False
    new_xmin_b = cx_b - side / 2.0
    new_ymin_b = cy_b - side / 2.0

    # 采样手部 bbox 区域内的一致网格点（原图坐标系）
    sample_h = int(max(1, min(h_h, 128)))
    sample_w = int(max(1, min(w_h, 128)))
    ys_norm = (np.arange(sample_h, dtype=np.float32) + 0.5) / float(sample_h)
    xs_norm = (np.arange(sample_w, dtype=np.float32) + 0.5) / float(sample_w)
    ys = y_h + ys_norm * float(h_h)
    xs = x_h + xs_norm * float(w_h)
    xs_grid, ys_grid = np.meshgrid(xs, ys)  # [H,W]

    # 映射到身体 seg 的坐标
    u_body = (xs_grid - new_xmin_b) / side
    v_body = (ys_grid - new_ymin_b) / side
    valid_body = (u_body >= 0.0) & (u_body <= 1.0) & (v_body >= 0.0) & (v_body <= 1.0)
    ix_body = (u_body * float(W_body)).astype(np.int32)
    iy_body = (v_body * float(H_body)).astype(np.int32)
    ix_body = np.clip(ix_body, 0, W_body - 1)
    iy_body = np.clip(iy_body, 0, H_body - 1)

    # 映射到手部 seg 的坐标（手视角是直接缩放到 bbox 矩形）
    u_hand = (xs_grid - float(x_h)) / float(w_h)
    v_hand = (ys_grid - float(y_h)) / float(h_h)
    valid_hand = (u_hand >= 0.0) & (u_hand <= 1.0) & (v_hand >= 0.0) & (v_hand <= 1.0)
    ix_hand = (u_hand * float(W_hand)).astype(np.int32)
    iy_hand = (v_hand * float(H_hand)).astype(np.int32)
    ix_hand = np.clip(ix_hand, 0, W_hand - 1)
    iy_hand = np.clip(iy_hand, 0, H_hand - 1)

    valid = valid_body & valid_hand
    if not np.any(valid):
        return False

    iy_body_v = iy_body[valid]
    ix_body_v = ix_body[valid]
    iy_hand_v = iy_hand[valid]
    ix_hand_v = ix_hand[valid]

    body_samples = seg_body[iy_body_v, ix_body_v]  # [N,3]
    hand_samples = seg_hand[iy_hand_v, ix_hand_v]  # [N,3]

    # 当前手在身体 seg 中对应的调色板索引（coarse hand）
    if group_idx == 1:
        hand_body_id = 5  # left (coarse hand)
    elif group_idx == 2:
        hand_body_id = 9  # right (coarse hand)
    else:
        # 非手视角不应走到这里
        return True

    if hand_body_id < 0 or hand_body_id >= _SEG_PALETTE.shape[0]:
        return False

    body_hand_color = _SEG_PALETTE[hand_body_id]
    background_color = _SEG_PALETTE[0]
    gray_color = np.array([128, 128, 128], dtype=np.uint8)

    body_is_hand = np.all(body_samples == body_hand_color, axis=1)
    hand_not_bg = ~np.all(hand_samples == background_color, axis=1)
    hand_not_gray = ~np.all(hand_samples == gray_color, axis=1)
    hand_is_hand = hand_not_bg & hand_not_gray

    if not np.any(hand_is_hand):
        return False

    overlap_mask = body_is_hand & hand_is_hand
    overlap_pixels = int(overlap_mask.sum())
    total_hand_pixels = int(hand_is_hand.sum())
    if total_hand_pixels == 0:
        return False

    overlap_ratio = overlap_pixels / float(total_hand_pixels)
    return overlap_ratio > overlap_thresh


def detect_num_base_cams_from_tempt(src_dir: str) -> int:
    """
    从多次推理的拼图 (uv/seg_subject_pred_left/right/both_hand(s)[_sXX].png) 中自动推断 NUM_BASE_CAMS。
    - left_hand/right_hand 模式：2N 视角（前 N 为 body，后 N 为对应手部）
    - both_hands 模式：3N 视角（前 N 为 body，中间 N 为左手，后 N 为右手）
    若检测失败则回退为 4。
    """
    pat_grid = re.compile(
        r"^(?P<type>uv|seg)_(?P<subject>\d+)_pred_(?P<step>left_hand|right_hand|both_hands)(?:_s\d{2})?\.png$"
    )
    files = [f for f in os.listdir(src_dir) if f.endswith(".png")]
    for fname in files:
        m = pat_grid.match(fname)
        if not m:
            continue
        step = m.group("step")
        path_img = osp.join(src_dir, fname)
        img = cv2.imread(path_img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue
        h, w = img.shape[:2]
        if h <= 0 or w % h != 0:
            continue
        n_cols = w // h
        
        if step == "both_hands":
            # both_hands 模式：期望为 3N 视角
            if n_cols >= 3 and n_cols % 3 == 0:
                return n_cols // 3
        else:
            # left_hand/right_hand 模式：期望为 2N 视角
            if n_cols >= 2 and n_cols % 2 == 0:
                return n_cols // 2
    return 4


def organize_images(
    src_dir: str,
    cat_dir: str,
    single_dir: str,
    tts_count: int = 1,
    base_seed: int = BASE_SEED,
    seed_step: int = SEED_STEP,
    sample_index_offset: int = DEFAULT_SAMPLE_SEED_OFFSET,
    allowed_s_indices: Optional[Set[int]] = None,
    only_subjects: Optional[Set[str]] = None,
):
    """从 `tempt` 目录读取多次推理结果（按 sXX 标记），
    在允许的样本集合内按不确定性策略选择最优样本，
    并拆分为 3 * NUM_BASE_CAMS 视角的单图保存到 `single_dir`，不确定性可视化保存到 `cat_dir`。

    期望文件命名（与 test_script 保存一致）：
      - uv_{subject}_pred_{left_hand|right_hand}_sXX.png
      - seg_{subject}_pred_{left_hand|right_hand}_sXX.png
    若 tts_count == 1 且无 sXX 后缀，则默认视作 s00。
    每张图为 2 * NUM_BASE_CAMS 视角横向拼接：前 N 为 body，后 N 为对应手部。
    """

    ensure_dir(cat_dir)
    ensure_dir(single_dir)

    # 依据 src_dir 推导手部 bbox 根目录：与 tempt 同级的 bbox_hand_foot
    exp_dir = osp.dirname(src_dir)
    ours_dir = osp.dirname(exp_dir)
    data_dir = osp.dirname(ours_dir)
    bbox_hand_foot_root = osp.join(exp_dir, "bbox_hand_foot")
    use_handfoot_filter = os.path.isdir(bbox_hand_foot_root)
    # 身体 bbox 目录：数据集根目录下的 gt/bbox
    bbox_body_dir = osp.join(data_dir, "gt", "bbox")

    part_map = {1: "left_hand", 2: "right_hand"}

    def can_save_single_for_view(subject_str: str, view_idx: int) -> bool:
        # 永久丢弃 3 * NUM_BASE_CAMS 以后的脚部/其他视角
        total_valid_views = 3 * NUM_BASE_CAMS
        if view_idx >= total_valid_views:
            return False
        if not use_handfoot_filter:
            return True
        group_idx = view_idx // NUM_BASE_CAMS  # 0:body, 1:LHand, 2:RHand
        if group_idx == 0:
            return True
        base_cam = view_idx % NUM_BASE_CAMS
        part = part_map.get(group_idx)
        if part is None:
            return True
        file_key_body = f"{subject_str}_{base_cam:02d}"
        bbox_json = osp.join(bbox_hand_foot_root, part, file_key_body + ".json")
        return osp.exists(bbox_json)

    # 允许的样本索引集合
    if allowed_s_indices is None:
        sample_index_offset = int(sample_index_offset)
        allowed_s_indices = set(
            range(sample_index_offset, sample_index_offset + max(int(tts_count), 1))
        )
    else:
        allowed_s_indices = set(int(i) for i in allowed_s_indices)

    # 支持的文件命名：type(subject) + step + 可选 sXX
    # 支持 left_hand, right_hand, both_hands 三种模式
    pat_grid8 = re.compile(
        r"^(?P<type>uv|seg)_(?P<subject>\d+)_pred_(?P<step>left_hand|right_hand|both_hands)(?:_s(?P<s>\d{2}))?\.png$"
    )

    # 收集候选：按 (type, subject) 聚合
    from collections import defaultdict

    candidates_body = defaultdict(list)  # key: (type, subject, body_idx) -> list of (sample_id, tile_img)
    candidates_hand = defaultdict(list)  # key: (type, subject, part, hv_idx) -> list of (sample_id, tile_img)

    # 预扫描：记录哪些 (t, subj, step) 存在 _sXX 文件，并缓存匹配文件
    files_info = []
    has_sxx = set()  # set of (t, subj, step)
    for fname in os.listdir(src_dir):
        if not fname.endswith(".png"):
            continue
        m = pat_grid8.match(fname)
        if not m:
            continue
        t = m.group("type")
        subj = m.group("subject")
        step = m.group("step")
        s_str = m.group("s")
        if only_subjects is not None and subj not in only_subjects:
            continue
        if s_str is not None:
            if s_str.isdigit():
                sidx_tmp = int(s_str)
                if sidx_tmp in allowed_s_indices:
                    has_sxx.add((t, subj, step))
        files_info.append((fname, t, subj, step, s_str))

    # 二次遍历：依据 has_sxx 决定是否纳入无后缀文件
    for fname, t, subj, step, s_str in files_info:
        # 样本过滤
        if s_str is None:
            # 如果该组已有 sXX，忽略无后缀；否则仅 TTS=1 时回退使用
            if (t, subj, step) in has_sxx:
                continue
            if int(tts_count) != 1:
                continue
            sidx = 0
        else:
            if not s_str.isdigit():
                continue
            sidx = int(s_str)
        if sidx not in allowed_s_indices:
            continue
        sample_key = f"s{sidx:02d}"

        path_img = osp.join(src_dir, fname)
        img = cv2.imread(path_img, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if img is None:
            continue
        # seg_* 图像按 RGB `_SEG_PALETTE` 生成，这里统一转为 RGB，便于后续量化
        if t == "seg":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        if h <= 0 or w % h != 0:
            continue
        n = w // h
        
        # 根据 step 类型判断期望的视角数
        if step == "both_hands":
            # both_hands 模式：3 * NUM_BASE_CAMS 视角（身体 + 左手 + 右手）
            expected_views = 3 * NUM_BASE_CAMS
            if n < expected_views:
                continue
            
            # 提取 body 前 NUM_BASE_CAMS 块
            for bi in range(NUM_BASE_CAMS):
                tile = img[:, bi * h : (bi + 1) * h]
                candidates_body[(t, subj, bi)].append((sample_key, tile))
            
            # 提取左手 NUM_BASE_CAMS 块（位置：NUM_BASE_CAMS 到 2*NUM_BASE_CAMS）
            for hi in range(NUM_BASE_CAMS):
                tile = img[:, (NUM_BASE_CAMS + hi) * h : (NUM_BASE_CAMS + hi + 1) * h]
                candidates_hand[(t, subj, "left_hand", hi)].append((sample_key, tile))
            
            # 提取右手 NUM_BASE_CAMS 块（位置：2*NUM_BASE_CAMS 到 3*NUM_BASE_CAMS）
            for hi in range(NUM_BASE_CAMS):
                tile = img[:, (2 * NUM_BASE_CAMS + hi) * h : (2 * NUM_BASE_CAMS + hi + 1) * h]
                candidates_hand[(t, subj, "right_hand", hi)].append((sample_key, tile))
                
        else:
            # left_hand 或 right_hand 模式：2 * NUM_BASE_CAMS 视角（身体 + 指定手部）
            expected_views = 2 * NUM_BASE_CAMS
            if n < expected_views:
                continue
            
            # 提取 body 前 NUM_BASE_CAMS 块（来自该 step 的前 N 图块）
            # 仅从 left_hand 步提取身体候选
            if step == "left_hand":
                for bi in range(NUM_BASE_CAMS):
                    tile = img[:, bi * h : (bi + 1) * h]
                    candidates_body[(t, subj, bi)].append((sample_key, tile))

            # 提取该 step 的后 NUM_BASE_CAMS 块为对应手部视角 0..(N-1)
            for hi in range(NUM_BASE_CAMS):
                tile = img[:, (NUM_BASE_CAMS + hi) * h : (NUM_BASE_CAMS + hi + 1) * h]
                candidates_hand[(t, subj, step, hi)].append((sample_key, tile))

    # 新：像素级聚合
    def _uv_aggregate_and_unc(tiles: list):
        # tiles: list of np.ndarray (H,W,3), BGR uint8
        if len(tiles) == 0:
            return None, None
        arr = np.stack([t.astype(np.float32) / 255.0 for t in tiles], axis=0)  # [S,H,W,3]
        median = np.median(arr, axis=0)  # [H,W,3] in [0,1]
        var_map = arr.var(axis=0).mean(axis=2)  # [H,W]
        vmax = float(var_map.max()) if float(var_map.max()) > 1e-12 else 1.0
        unc_gray = (np.clip(var_map / vmax, 0.0, 1.0) * 255.0).astype(np.uint8)
        unc_bgr = cv2.cvtColor(unc_gray, cv2.COLOR_GRAY2BGR)
        out = (np.clip(median, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        return out, unc_bgr

    def _seg_aggregate_and_unc(tiles: list, view_type: str):
        # tiles: list of np.ndarray (H,W,3), RGB uint8
        if len(tiles) == 0:
            return None, None
        palette_len = int(_SEG_PALETTE.shape[0])
        if palette_len > 12:
            if view_type == "hand":
                # 手部：允许 背景(0) + 细分手部(12..)
                allowed_ids = [0] + list(range(12, palette_len))
            else:
                # 身体：仅 0..11
                allowed_ids = list(range(12))
        else:
            allowed_ids = list(range(palette_len))
        # 量化到 id
        ids_stack = []
        for t_img in tiles:
            # 这里 tiles 已经是 RGB；直接在 RGB 空间与 `_SEG_PALETTE` 做最近邻匹配
            seg_rgb = t_img
            if view_type == "hand":
                # 手部：直接在 黑色(0) + 灰色(128,128,128) + 12.. 之间最近邻分类
                h, w = seg_rgb.shape[:2]
                seg_flat = seg_rgb.reshape(-1, 1, 3).astype(np.int32)  # (N,1,3)
                palette_full = _SEG_PALETTE.astype(np.int32)
                allowed_ids_arr = np.asarray(allowed_ids, dtype=np.int64)
                pal_sub = palette_full[allowed_ids_arr].reshape(1, -1, 3)  # (1,K,3)
                dist2_pal = np.sum((seg_flat - pal_sub) ** 2, axis=2)  # (N,K)
                nearest_sub_idx = np.argmin(dist2_pal, axis=1)  # (N,)
                min_pal_dist2 = dist2_pal.min(axis=1)  # (N,)
                nearest_global = allowed_ids_arr[nearest_sub_idx]  # (N,)
                # 灰色距离
                gray_color = np.array([128, 128, 128], dtype=np.int32).reshape(1, 1, 3)
                dist2_gray = np.sum((seg_flat - gray_color) ** 2, axis=2).reshape(-1)  # (N,)
                choose_gray = dist2_gray < min_pal_dist2  # (N,)
                id_flat = nearest_global.astype(np.int32)
                # 使用 palette_len 作为灰色的“额外类别”索引
                id_flat[choose_gray] = int(palette_len)
                ids = id_flat.reshape(h, w).astype(np.int32)
            else:
                # 身体：仅在 0..11 之间量化
                ids = _quantize_seg_by_palette_subset(seg_rgb, allowed_ids).astype(np.int32)
            ids_stack.append(ids)
        ids_arr = np.stack(ids_stack, axis=0)  # [S,H,W]
        S, H, W = ids_arr.shape
        # 多数投票（扩展一个“灰色”通道）
        extra_gray_id = palette_len  # 扩展维度的最后一类
        counts_ext = np.zeros((palette_len + 1, H, W), dtype=np.int32)
        for aid in allowed_ids:
            counts_ext[aid] = (ids_arr == aid).sum(axis=0)
        # 手部：统计“灰色”类别票数（使用 extra_gray_id）
        if view_type == "hand":
            counts_ext[extra_gray_id] = (ids_arr == extra_gray_id).sum(axis=0).astype(np.int32)
        best_ext = counts_ext.argmax(axis=0).astype(np.int32)  # [H,W]
        # 生成颜色：palette 内按调色板；灰色类输出 (128,128,128)
        best_ids = np.clip(best_ext, 0, palette_len - 1).astype(np.uint8)
        seg_color = _SEG_PALETTE[best_ids]  # [H,W,3] BGR
        if view_type == "hand":
            seg_color[best_ext == extra_gray_id] = np.array([128, 128, 128], dtype=np.uint8)
        # 不确定性
        counts_max = counts_ext.max(axis=0)  # [H,W]
        half = max(S / 2.0, 1e-8)
        uncert = np.where(
            counts_max <= half,
            1.0,
            1.0 - (counts_max - half) / half,
        ).astype(np.float32)
        uncert = np.clip(uncert, 0.0, 1.0)
        seg_unc_gray = (uncert * 255.0 + 0.5).astype(np.uint8)
        seg_unc_bgr = cv2.cvtColor(seg_unc_gray, cv2.COLOR_GRAY2BGR)
        return seg_color, seg_unc_bgr

    # —— 联合选择缓存：确保 UV/SEG 来自同一采样 ——
    # subj -> {'uv_blocks': [3 * NUM_BASE_CAMS], 'seg_blocks':[...], 'uv_uncs': [...], 'seg_uncs': [...]}
    joint_cache = {}

    def ensure_joint_for_subject(subj: str):
        if subj in joint_cache:
            return
        total_views = 3 * NUM_BASE_CAMS
        uv_blocks = [None] * total_views
        seg_blocks = [None] * total_views
        uv_uncs = [None] * total_views
        seg_uncs = [None] * total_views
        # body 0..(N-1)：逐视角像素级聚合
        for bi in range(NUM_BASE_CAMS):
            uv_pairs = candidates_body.get(("uv", subj, bi), [])
            seg_pairs = candidates_body.get(("seg", subj, bi), [])
            uv_tiles = [p[1] for p in uv_pairs]
            seg_tiles = [p[1] for p in seg_pairs]
            uv_best, uv_unc = _uv_aggregate_and_unc(uv_tiles)
            seg_best, seg_unc = _seg_aggregate_and_unc(seg_tiles, view_type="body")
            uv_blocks[bi] = uv_best
            seg_blocks[bi] = seg_best
            uv_uncs[bi] = uv_unc
            seg_uncs[bi] = seg_unc
        # left hand N..(2N-1)：逐视角像素级聚合
        for hi in range(NUM_BASE_CAMS):
            idx = NUM_BASE_CAMS + hi
            uv_pairs = candidates_hand.get(("uv", subj, "left_hand", hi), [])
            seg_pairs = candidates_hand.get(("seg", subj, "left_hand", hi), [])
            uv_tiles = [p[1] for p in uv_pairs]
            seg_tiles = [p[1] for p in seg_pairs]
            uv_best, uv_unc = _uv_aggregate_and_unc(uv_tiles)
            seg_best, seg_unc = _seg_aggregate_and_unc(seg_tiles, view_type="hand")
            uv_blocks[idx] = uv_best
            seg_blocks[idx] = seg_best
            uv_uncs[idx] = uv_unc
            seg_uncs[idx] = seg_unc
        # right hand 2N..(3N-1)：逐视角像素级聚合
        for hi in range(NUM_BASE_CAMS):
            idx = 2 * NUM_BASE_CAMS + hi
            uv_pairs = candidates_hand.get(("uv", subj, "right_hand", hi), [])
            seg_pairs = candidates_hand.get(("seg", subj, "right_hand", hi), [])
            uv_tiles = [p[1] for p in uv_pairs]
            seg_tiles = [p[1] for p in seg_pairs]
            uv_best, uv_unc = _uv_aggregate_and_unc(uv_tiles)
            seg_best, seg_unc = _seg_aggregate_and_unc(seg_tiles, view_type="hand")
            uv_blocks[idx] = uv_best
            seg_blocks[idx] = seg_best
            uv_uncs[idx] = uv_unc
            seg_uncs[idx] = seg_unc
        joint_cache[subj] = {
            "uv_blocks": uv_blocks,
            "seg_blocks": seg_blocks,
            "uv_uncs": uv_uncs,
            "seg_uncs": seg_uncs,
        }

    # 遍历 subject，输出 3 * NUM_BASE_CAMS 视角（UV 与 SEG 同步采样）
    subjects = sorted(
        set([k[1] for k in candidates_body.keys()])
        | set([k[1] for k in candidates_hand.keys()])
    )
    for subj in subjects:
        ensure_joint_for_subject(subj)
        uv_blocks = joint_cache[subj]["uv_blocks"]
        seg_blocks = joint_cache[subj]["seg_blocks"]
        uv_uncs = joint_cache[subj]["uv_uncs"]
        seg_uncs = joint_cache[subj]["seg_uncs"]

        total_views = 3 * NUM_BASE_CAMS

        # 写入 single（仅保存真实存在的视角；手视角需存在对应 bbox；
        # 手视角进一步依据 seg 与身体 seg 的重合比例过滤）
        for i in range(total_views):
            # body 总是允许；手部需先通过 bbox 存在性校验
            allow = can_save_single_for_view(subj, i)
            if not allow:
                continue

            group_idx = i // NUM_BASE_CAMS  # 0:body, 1:LHand, 2:RHand
            base_cam = i % NUM_BASE_CAMS

            uv_name = f"uv_{subj}_{i:02d}_pred.png"
            seg_name = f"seg_{subj}_{i:02d}_pred.png"
            uv_unc_name = f"uv_uncertainty_{subj}_{i:02d}_pred.png"
            seg_unc_name = f"seg_uncertainty_{subj}_{i:02d}_pred.png"

            uv_dst = osp.join(single_dir, uv_name)
            seg_dst = osp.join(single_dir, seg_name)
            uv_unc_dst = osp.join(single_dir, uv_unc_name)
            seg_unc_dst = osp.join(single_dir, seg_unc_name)

            # 先写入该视角的 uv / seg / 不确定性
            if uv_blocks[i] is not None and not osp.exists(uv_dst):
                cv2.imwrite(uv_dst, uv_blocks[i])
            if seg_blocks[i] is not None and not osp.exists(seg_dst):
                # seg_blocks 内部按照 RGB `_SEG_PALETTE` 存储，写入 PNG 前需转为 BGR
                seg_bgr = cv2.cvtColor(seg_blocks[i], cv2.COLOR_RGB2BGR)
                cv2.imwrite(seg_dst, seg_bgr)
            if uv_uncs[i] is not None and not osp.exists(uv_unc_dst):
                cv2.imwrite(uv_unc_dst, uv_uncs[i])
            if seg_uncs[i] is not None and not osp.exists(seg_unc_dst):
                cv2.imwrite(seg_unc_dst, seg_uncs[i])

            # 手部视角：根据 seg 与身体 seg 的重叠比例进一步过滤
            if group_idx in (1, 2):
                # 读取该手的 bbox（原图坐标）
                part = part_map.get(group_idx)
                file_key_body = f"{subj}_{base_cam:02d}"
                bbox_hand_json = osp.join(
                    bbox_hand_foot_root, part, file_key_body + ".json"
                )
                if not osp.exists(bbox_hand_json):
                    # 没有对应手 bbox，直接移除该手视角
                    for pth in [uv_dst, seg_dst, uv_unc_dst, seg_unc_dst]:
                        if osp.exists(pth):
                            os.remove(pth)
                    continue
                with open(bbox_hand_json, "r") as f:
                    bbox_dict = json.load(f)
                bbox_hand_tlwh = bbox_dict.get("tlwh", None)

                if bbox_hand_tlwh is None:
                    # bbox 内容异常，也直接清理该视角
                    for pth in [uv_dst, seg_dst, uv_unc_dst, seg_unc_dst]:
                        if osp.exists(pth):
                            os.remove(pth)
                    continue

                # 在 single_dir 中使用 seg_* 图像进行重叠检测；
                # img_ori_shape 在当前实现中未实际使用，这里传入占位 (0,0)
                keep = should_keep_hand_view_by_seg_overlap(
                    subject=subj,
                    view_idx=i,
                    group_idx=group_idx,
                    base_cam=base_cam,
                    hand_bbox_tlwh=bbox_hand_tlwh,
                    img_ori_shape=(0, 0),
                    image_crop_dir=single_dir,
                    bbox_body_dir=bbox_body_dir,
                    overlap_thresh=0.10,
                )

                if not keep:
                    # 重叠比例不足，删除该手视角在 single 中的所有文件
                    for pth in [uv_dst, seg_dst, uv_unc_dst, seg_unc_dst]:
                        if osp.exists(pth):
                            os.remove(pth)

        # 额外：不确定性与选择结果拼接保存到 cat_dir（UV 与 SEG 各一张）
        def _save_cat(prefix: str, blocks: list):
            blocks_valid = [b for b in blocks if b is not None]
            if len(blocks_valid) == 0:
                return
            refb = blocks_valid[0]
            hb, wb = refb.shape[:2]
            blankb = np.zeros((hb, wb, 3), dtype=np.uint8)
            rowb = []
            for i in range(total_views):
                allow = can_save_single_for_view(subj, i)
                rowb.append(blocks[i] if (allow and blocks[i] is not None) else blankb)
            cat_sel = np.concatenate(rowb, axis=1)
            out_sel_path = osp.join(cat_dir, f"{prefix}_{subj}_pred.png")
            # seg cat 亦是 RGB，写盘前统一转为 BGR
            if prefix == "seg":
                cat_sel_bgr = cv2.cvtColor(cat_sel, cv2.COLOR_RGB2BGR)
                cv2.imwrite(out_sel_path, cat_sel_bgr)
            else:
                cv2.imwrite(out_sel_path, cat_sel)

        def _save_cat_unc(prefix: str, uncs: list):
            uncs_valid = [u for u in uncs if u is not None]
            if len(uncs_valid) == 0:
                return
            ref = uncs_valid[0]
            h, w = ref.shape[:2]
            blank = np.zeros((h, w, 3), dtype=np.uint8)
            row = []
            for i in range(total_views):
                allow = can_save_single_for_view(subj, i)
                row.append(uncs[i] if (allow and uncs[i] is not None) else blank)
            cat_unc = np.concatenate(row, axis=1)
            out_unc_path = osp.join(cat_dir, f"{prefix}_uncertainty_{subj}_pred.png")
            cv2.imwrite(out_unc_path, cat_unc)

        _save_cat("uv", uv_blocks)
        _save_cat("seg", seg_blocks)
        _save_cat_unc("uv", uv_uncs)
        _save_cat_unc("seg", seg_uncs)

    print(f"Images organized with selection for {int(tts_count)} TTS -> single: {single_dir}")


def run_step2_postprocess(
    image_crop_raw_dir: str,
    tts_count: int = 1,
    total_samples: Optional[int] = None,
    variant_count: int = 1,
    sample_index_offset: int = DEFAULT_SAMPLE_SEED_OFFSET,
    shuffle_variants: bool = False,
    shuffle_seed: int = 1314,
    only_subjects: Optional[Set[str]] = None,
    base_cams: Optional[int] = None,
):
    """
    在第一步推理（当前脚本）完成后，对 `tempt` 目录中的 uv/seg 结果执行第二步：
      1) 按 TTS 采样结果做像素级聚合与不确定性可视化；
      2) 将 2 * NUM_BASE_CAMS 视角拼接图拆分为 3 * NUM_BASE_CAMS 视角单图（body + 左/右手）；
      3) 利用 bbox 与 seg 将裁剪图回贴到原始分辨率，输出到 *_fixed 目录。
    """
    if total_samples is None:
        total_samples = tts_count

    assert tts_count >= 1 and total_samples >= 1, "tts_count 与 total_samples 必须为正整数"
    assert tts_count <= total_samples, "tts_count 不能大于 total_samples"

    tts_count = int(tts_count)
    total_samples = int(total_samples)
    variant_count = int(max(1, variant_count))
    sample_index_offset = int(sample_index_offset)

    tts_suffix = f"tts{tts_count:02d}"

    # 根据 `tempt` 路径推导数据目录与原图路径
    exp_dir = osp.dirname(image_crop_raw_dir)  # .../{exp}
    ours_dir = osp.dirname(exp_dir)  # .../ours_xv
    data_dir = osp.dirname(ours_dir)  # .../{dataset}
    root_dir = osp.join(data_dir, "gt")
    image_ori_dir = osp.join(root_dir, "image_ori")
    bbox_body_dir = osp.join(root_dir, "bbox")
    bbox_hand_foot_root = osp.join(exp_dir, "bbox_hand_foot")

    # 共享第一步的视角数：优先使用调用方传入的 body_views / num_views，
    # 若未提供则自动从拼接图中检测 NUM_BASE_CAMS（例如 1 / 4 / 8）
    global NUM_BASE_CAMS
    if base_cams is not None:
        NUM_BASE_CAMS = int(base_cams)
    else:
        detected_base = detect_num_base_cams_from_tempt(image_crop_raw_dir)
        NUM_BASE_CAMS = detected_base

    # 生成组合：从 TOTAL_SAMPLES 中取大小为 TTS_COUNT 的组合，取前 VARIANT_COUNT 组
    sample_indices = list(range(sample_index_offset, sample_index_offset + total_samples))
    comb_iter = itertools.combinations(sample_indices, tts_count)
    variant_groups: List[List[int]] = []
    if shuffle_variants:
        rng = random.Random(int(shuffle_seed))
        for i, comb in enumerate(comb_iter):
            if i < variant_count:
                variant_groups.append(list(comb))
            else:
                j = rng.randint(0, i)
                if j < variant_count:
                    variant_groups[j] = list(comb)
    else:
        for idx, comb in enumerate(comb_iter):
            if idx >= variant_count:
                break
            variant_groups.append(list(comb))

    # 若仅生成 1 组且等同于旧逻辑（[0..tts_count-1]），仍兼容
    if len(variant_groups) == 0:
        variant_groups = [list(range(tts_count))]

    for v_idx, s_indices in enumerate(variant_groups):
        if variant_count == 1:
            var_suffix = f"v{sample_index_offset:02d}"
        else:
            var_suffix = f"v{sample_index_offset + v_idx:02d}"

        # 处理 cat & single 目录（带变体后缀）
        cat_dir = osp.join(exp_dir, f"tempt_{tts_suffix}_{var_suffix}_cat")
        single_dir = osp.join(exp_dir, f"tempt_{tts_suffix}_{var_suffix}_single")
        ensure_dir(cat_dir)
        ensure_dir(single_dir)

        organize_images(
            image_crop_raw_dir,
            cat_dir,
            single_dir,
            tts_count=tts_count,
            base_seed=BASE_SEED,
            seed_step=SEED_STEP,
            sample_index_offset=sample_index_offset,
            allowed_s_indices=set(int(i) for i in s_indices),
            only_subjects=only_subjects,
        )

        # 后续流程均使用 single_dir 中的独立视角图片
        image_crop_dir = single_dir

        output_dir = osp.join(exp_dir, f"tempt_{tts_suffix}_{var_suffix}_fixed")
        ensure_dir(output_dir)

        # 遍历 single 目录下的视角切片，按 body/左右手分别取 bbox 源并回贴到原图
        pat_single = re.compile(
            r"^(?P<type>.+?)_(?P<subject>\d+)_(?P<view>\d+)_pred\.png$"
        )
        files = [f for f in os.listdir(image_crop_dir) if f.endswith(".png")]
        files.sort()
        for fname in tqdm(files, desc=f"Fixing views [{var_suffix}]"):
            m = pat_single.match(fname)
            if not m:
                continue
            t = m.group("type")
            subject = m.group("subject")
            if only_subjects is not None and subject not in only_subjects:
                continue
            view_idx = int(m.group("view"))
            total_valid_views = 3 * NUM_BASE_CAMS
            if view_idx >= total_valid_views:
                # 丢弃脚部等 3N 之后的视角
                continue
            base_cam = view_idx % NUM_BASE_CAMS
            group_idx = view_idx // NUM_BASE_CAMS  # 0:body, 1:LHand, 2:RHand

            # 原图路径（按 base_cam 对应 4 视角）
            file_key_body = f"{subject}_{base_cam:02d}"
            ori_path = None
            for ext in [".png", ".jpg", ".jpeg"]:
                pth = osp.join(image_ori_dir, file_key_body + ext)
                if osp.exists(pth):
                    ori_path = pth
                    break
            if ori_path is None:
                # 原图缺失，跳过
                continue

            img_ori = cv2.imread(
                ori_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            if img_ori is None:
                continue
            H, W = img_ori.shape[:2]

            # 加载对应视角的裁剪图
            crop_path = osp.join(image_crop_dir, fname)
            crop_img = cv2.imread(
                crop_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION
            )
            if crop_img is None:
                continue
            H_crop, W_crop = crop_img.shape[:2]

            # 选择 bbox 源
            bbox_tlwh = None
            if group_idx == 0:
                bbox_json = osp.join(bbox_body_dir, file_key_body + ".json")
                if osp.exists(bbox_json):
                    with open(bbox_json, "r") as f:
                        bbox_dict = json.load(f)
                    bbox_tlwh = bbox_dict.get("tlwh", None)
            else:
                part_map_fix = {1: "left_hand", 2: "right_hand"}
                part = part_map_fix.get(group_idx, None)
                if part is None:
                    continue
                bbox_json = osp.join(
                    bbox_hand_foot_root, part, file_key_body + ".json"
                )
                if osp.exists(bbox_json):
                    with open(bbox_json, "r") as f:
                        bbox_dict = json.load(f)
                    bbox_tlwh = bbox_dict.get("tlwh", None)

            if bbox_tlwh is None:
                # 无 bbox，跳过
                continue

            # bbox 在原图中的坐标
            x, y, w_box, h_box = bbox_tlwh
            canvas = np.zeros((H, W, 3), dtype=np.uint8)

            if group_idx == 0:
                # 身体：按比例扩张成正方形区域
                x_min, y_min, x_max, y_max = x, y, x + w_box, y + h_box
                cx = (x_min + x_max) / 2.0
                cy = (y_min + y_max) / 2.0
                bw = (x_max - x_min) * DEFAULT_BBOX_RATIO
                bh = (y_max - y_min) * DEFAULT_BBOX_RATIO
                side = max(bw, bh)

                # 目标放置区域（原图坐标）
                new_xmin = int(round(cx - side / 2.0))
                new_ymin = int(round(cy - side / 2.0))
                new_xmax = int(round(cx + side / 2.0))
                new_ymax = int(round(cy + side / 2.0))

                # 与原图的交集区域
                new_xmin_clip = max(0, new_xmin)
                new_ymin_clip = max(0, new_ymin)
                new_xmax_clip = min(W, new_xmax)
                new_ymax_clip = min(H, new_ymax)

                # 原图在 crop 坐标中的对应区域（crop 为正方形，使用 H_crop/side 比例）
                x_min_crop = (-new_xmin if new_xmin < 0 else 0) * H_crop / side
                y_min_crop = (-new_ymin if new_ymin < 0 else 0) * H_crop / side
                x_max_crop = (
                    side - (new_xmax - W) if new_xmax > W else side
                ) * H_crop / side
                y_max_crop = (
                    side - (new_ymax - H) if new_ymax > H else side
                ) * H_crop / side

                crop_resized = cv2.resize(
                    crop_img[int(y_min_crop) : int(y_max_crop), int(x_min_crop) : int(x_max_crop)],
                    (new_xmax_clip - new_xmin_clip, new_ymax_clip - new_ymin_clip),
                    interpolation=cv2.INTER_LINEAR,
                )
                canvas[new_ymin_clip:new_ymax_clip, new_xmin_clip:new_xmax_clip] = crop_resized
            else:
                # 手部：直接使用矩形 bbox（不扩张、不正方形化），越界部分以黑色填充
                rect_w = w_box
                rect_h = h_box
                if rect_w <= 0 or rect_h <= 0:
                    continue
                new_xmin = int(round(x))
                new_ymin = int(round(y))
                new_xmax = int(round(x + rect_w))
                new_ymax = int(round(y + rect_h))

                # 与原图的交集区域
                new_xmin_clip = max(0, new_xmin)
                new_ymin_clip = max(0, new_ymin)
                new_xmax_clip = min(W, new_xmax)
                new_ymax_clip = min(H, new_ymax)

                target_w = new_xmax_clip - new_xmin_clip
                target_h = new_ymax_clip - new_ymin_clip
                if target_w <= 0 or target_h <= 0:
                    continue

                rect_w_int = max(1, int(round(rect_w)))
                rect_h_int = max(1, int(round(rect_h)))
                hand_full = cv2.resize(
                    crop_img, (rect_w_int, rect_h_int), interpolation=cv2.INTER_LINEAR
                )

                # 在 hand_full 坐标中的切片起止（对应于目标与原图交集在未裁剪bbox中的位置）
                src_x0 = max(0, new_xmin_clip - new_xmin)
                src_y0 = max(0, new_ymin_clip - new_ymin)
                src_x1 = min(rect_w_int, src_x0 + target_w)
                src_y1 = min(rect_h_int, src_y0 + target_h)

                src_w_act = src_x1 - src_x0
                src_h_act = src_y1 - src_y0
                if src_w_act <= 0 or src_h_act <= 0:
                    continue

                dst_x1 = new_xmin_clip + src_w_act
                dst_y1 = new_ymin_clip + src_h_act

                canvas[new_ymin_clip:dst_y1, new_xmin_clip:dst_x1] = hand_full[
                    src_y0:src_y1, src_x0:src_x1
                ]

            # 保存，命名：type_subject_view_pred.png，其中 view 为 00..11
            fixed_path = osp.join(output_dir, f"{t}_{subject}_{view_idx:02d}_pred.png")
            cv2.imwrite(fixed_path, canvas)

        print("All fixed images saved to", output_dir)


def get_available_gpus():
    """检测可用的GPU设备"""
    # 使用nvidia-smi获取GPU信息（没有回退逻辑）
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.free,memory.total', '--format=csv,nounits,noheader'], 
                          capture_output=True, text=True, check=True)
    gpu_info = []
    for line in result.stdout.strip().split('\n'):
        parts = line.split(', ')
        gpu_id = int(parts[0])
        free_mem = int(parts[1])
        total_mem = int(parts[2])
        # 只选择有足够内存的GPU (至少8GB空闲内存)
        if free_mem > 8000:
            gpu_info.append({
                'id': gpu_id,
                'free_memory': free_mem,
                'total_memory': total_mem
            })
    # 按空闲内存排序
    gpu_info.sort(key=lambda x: x['free_memory'], reverse=True)
    return [gpu['id'] for gpu in gpu_info]


def test_model(config_path, output_dir, input_folder=None, proxy_vae_path=None,
               enable_uncertainty: bool = False,
               num_uncertainty_samples_list: Optional[List[int]] = None,
               body_views: Optional[int] = None,
               body_num_uncertainty_samples: Optional[int] = None,
               hand_num_uncertainty_samples_list: Optional[List[int]] = None,
               inference_both_hands_once: bool = False,
               sample_seed_offsets: Optional[List[int]] = None,
               subject_queue=None,
               subject_ids: Optional[List[str]] = None):
    """
    加载模型并测试到generate_images步骤，然后立即从显存卸载模型。
    支持多个 sample_seed_offset 和多个 TTS (num_uncertainty_samples)，
    在同一次模型加载中处理所有 (tts, offset) 组合，相同 seed 只推理一次。

    Args:
        config_path: 配置文件路径
        output_dir: 输出目录
        input_folder: 输入图像文件夹路径
        proxy_vae_path: 合并后的 proxy VAE 目录（在主 checkpoint 后重新加载）
        num_uncertainty_samples_list: TTS 值列表，例如 [1,3,5]
        hand_num_uncertainty_samples_list: 手部 TTS 值列表（若为 None 则同 num_uncertainty_samples_list）
        sample_seed_offsets: 采样偏移量列表，例如 [0,1,2]
    """
    if sample_seed_offsets is None:
        sample_seed_offsets = [DEFAULT_SAMPLE_SEED_OFFSET]
    sample_seed_offsets = [int(o) for o in sample_seed_offsets]
    if num_uncertainty_samples_list is None:
        num_uncertainty_samples_list = [5]
    num_uncertainty_samples_list = [int(t) for t in num_uncertainty_samples_list]

    # 设置随机种子
    pl.seed_everything(42, workers=True) #42

    # imports moved to file top

    # 加载配置（先覆写 num_views 到临时 YAML，再加载）
    print(f"加载配置文件: {config_path}")
    with open(config_path, 'r') as f:
        raw_cfg = yaml.safe_load(f)

    TARGET_VIEWS = 8
    def set_keys(d, keys, value):
        cur = d
        for k in keys[:-1]:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                return
        if isinstance(cur, dict):
            cur[keys[-1]] = value

    set_keys(raw_cfg, ['data', 'num_views'], TARGET_VIEWS)
    set_keys(raw_cfg, ['data', 'num_ref_views'], TARGET_VIEWS)
    set_keys(raw_cfg, ['system', 'init_adapter_kwargs', 'num_views'], TARGET_VIEWS)

    # SD2.1 is an external prerequisite; explicit local overrides remain available.
    sd21_base = os.environ.get("SD_BASE_MODEL", "pretrained/sd21_base")
    set_keys(raw_cfg, ['system', 'pretrained_model_name_or_path'], sd21_base)
    set_keys(raw_cfg, ['system', 'pretrained_unet_name_or_path'], sd21_base)
    print(f"[SD-BASE] pretrained_model/unet -> {sd21_base}  (override via env SD_BASE_MODEL)")

    tmp_dir = tempfile.mkdtemp(prefix='i2mv_cfg_')
    tmp_cfg_path = os.path.join(tmp_dir, os.path.basename(config_path))
    with open(tmp_cfg_path, 'w') as f:
        yaml.safe_dump(raw_cfg, f, sort_keys=False)

    cfg = load_config(tmp_cfg_path, cli_args=[], n_gpus=1)
    # cfg = load_config(config_path, cli_args=[], n_gpus=1)

    # 创建模型
    print("创建模型...")
    # 关闭自动混合精度，确保使用完整精度
    cfg.trainer["precision"] = 32

    # 使用Lightning的方式加载模型和检查点
    SystemClass = find(cfg.system_cls)

    print(f"从检查点加载模型: {cfg.resume}")
    system = SystemClass.load_from_checkpoint(
        cfg.resume,
        cfg=cfg.system,
        strict=False,
        weights_only=False
    )

 
    # 将模型移动到GPU并设置为评估模式
    if torch.cuda.is_available():
        system = system.to("cuda")
        print(f"模型已移动到GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")
    else:
        print("CUDA不可用，使用CPU")
    system.eval()
    
    # 确保vit模型使用float32精度
    if hasattr(system, "pipeline") and hasattr(system.pipeline, "vit"):
        system.pipeline.vit = system.pipeline.vit.float()

    # 确保cond_encoder使用与输入相同的精度
    if hasattr(system, "pipeline") and hasattr(system.pipeline, "cond_encoder") and len(cfg.data.condition_image_modality)>0:
        system.pipeline.cond_encoder = system.pipeline.cond_encoder.float()

    # 关闭推理阶段的随机 drop，确保 reference 与 condition 始终生效
    if hasattr(system, "pipeline"):
        if hasattr(system.pipeline, "did_drop_infer"):
            system.pipeline.did_drop_infer = False

    # Optional inference overrides without rewriting the release config.
    # CLI --cfg/--steps populate these environment variables before model loading.
    _env_cfg = os.environ.get("CFG_VALUE", None)
    if _env_cfg is not None and hasattr(system, 'cfg') and hasattr(system.cfg, 'eval_guidance_scale'):
        old_cfg = float(system.cfg.eval_guidance_scale)
        new_cfg = float(_env_cfg)
        system.cfg.eval_guidance_scale = new_cfg
        print(f"[CFG-OVERRIDE] eval_guidance_scale: {old_cfg} -> {new_cfg}  (env CFG_VALUE)")
    elif _env_cfg is not None:
        print("[CFG-OVERRIDE] WARNING: system.cfg.eval_guidance_scale not found; CFG override skipped.")

    _env_steps = os.environ.get("NUM_INFERENCE_STEPS", None)
    if _env_steps is not None and hasattr(system, 'cfg') and hasattr(system.cfg, 'eval_num_inference_steps'):
        old_steps = int(system.cfg.eval_num_inference_steps)
        new_steps = int(_env_steps)
        system.cfg.eval_num_inference_steps = new_steps
        print(f"[STEPS-OVERRIDE] eval_num_inference_steps: {old_steps} -> {new_steps}  (env NUM_INFERENCE_STEPS)")


    # The main checkpoint may restore its historical VAE; reapply the canonical
    # merged proxy VAE so the final inference state is deterministic.
    print(f"正在重新加载合并后的 proxy VAE: {proxy_vae_path}")
    if proxy_vae_path is not None:
        system = reload_proxy_vae(system, proxy_vae_path)
        print("✓ 成功加载合并后的 proxy VAE")

    # 创建输出目录
    print(f"结果将保存到: {output_dir}")

    # body_samples 固定（通常为 1），hand_samples_list 为所有需要的 TTS 值
    body_samples = int(body_num_uncertainty_samples) if body_num_uncertainty_samples is not None else int(num_uncertainty_samples_list[0])
    if hand_num_uncertainty_samples_list is not None:
        hand_samples_list = sorted(set(int(h) for h in hand_num_uncertainty_samples_list))
    else:
        hand_samples_list = sorted(set(int(t) for t in num_uncertainty_samples_list))
    max_hand_samples = max(hand_samples_list)

    # 直接从文件夹加载图像
    if input_folder is None:
        print("错误: 未指定输入图像文件夹")
        return []

    print(f"从文件夹加载图像: {input_folder}")
    image_files = sorted(glob.glob(os.path.join(input_folder, "*.jpg")) + 
                         glob.glob(os.path.join(input_folder, "*.png")))
    
    if len(image_files) == 0:
        print(f"错误: 在 {input_folder} 中未找到图像文件")
        return []

    # 配置信息
    h_ori, w_ori = np.array(Image.open(image_files[0]).convert("RGBA")).shape[:2]
    h_tgt = cfg.data.height
    w_tgt = cfg.data.width
    # 若指定 body_views，则覆盖使用的视角数量（以及 cfg 中对应字段），方便做 num_views 消融
    num_views = int(body_views) if (body_views is not None) else cfg.data.num_views
    num_ref_views = getattr(cfg.data, 'num_ref_views', 1)  # 从配置中读取 num_ref_views
    if body_views is not None:
        cfg.data.num_views = int(body_views)
        if hasattr(cfg.data, 'num_ref_views'):
            cfg.data.num_ref_views = int(body_views)
        else:
            setattr(cfg.data, 'num_ref_views', int(body_views))
    background_color = torch.tensor([0.0, 0.0, 0.0]) if cfg.data.background_color == "black" else torch.tensor([0.5, 0.5, 0.5])
    
    # demo 可视化目录：与 output_dir 同级的 tempt_demo
    demo_dir = os.path.join(os.path.dirname(output_dir), "tempt_demo")
    os.makedirs(demo_dir, exist_ok=True)

    # === 根据文件名前缀（下划线之前）对图片进行分组，支持同一主体多张 reference 图像 ===
    files_by_subject = {}
    for f in image_files:
        subject_id = os.path.basename(f).split("_")[0]
        files_by_subject.setdefault(subject_id, []).append(f)

    # 按 subject_id 排序，保证输出结果可复现
    # Do not truncate the subject list implicitly.
    sorted_subject_items = sorted(files_by_subject.items(), key=lambda x: x[0])
    sorted_subject_items = filter_subject_items(sorted_subject_items, subject_ids)
    if subject_ids:
        print(f"[SUBJECTS] explicit selection: {[sid for sid, _ in sorted_subject_items]}")

    def _iter_subjects():
        """根据 subject_queue 是否存在选择迭代模式：队列动态抢占 or 顺序遍历"""
        if subject_queue is not None:
            subject_lookup = dict(sorted_subject_items)
            idx = 0
            while True:
                try:
                    sid = subject_queue.get(timeout=5)
                except QueueEmpty:
                    break
                if sid in subject_lookup:
                    yield idx, (sid, subject_lookup[sid])
                    idx += 1
        else:
            yield from enumerate(sorted_subject_items)

    # 多视角模式：一次性处理所有视角
    for subj_idx, (subject_id, view_files) in _iter_subjects():
        # ===== Resume：若该 subject 的所有 (tts, offset) 输出已存在则跳过 =====
        # 检查 output_dir 下每个 (tts, offset) 组合的 uv/seg both_hands png 是否齐全
        _resume_done = True
        for _ts in hand_samples_list:
            for _ofs in sample_seed_offsets:
                _combo = f"tts{int(_ts):02d}_v{int(_ofs):02d}"
                _uv = os.path.join(output_dir, f"uv_{subject_id}_pred_both_hands_{_combo}.png")
                _seg = os.path.join(output_dir, f"seg_{subject_id}_pred_both_hands_{_combo}.png")
                if not (os.path.exists(_uv) and os.path.exists(_seg)):
                    _resume_done = False
                    break
            if not _resume_done:
                break
        if _resume_done:
            print(f"[{subject_id}] [Resume] 已检测到所有 (tts, offset) 输出，跳过整 subject")
            continue
        # 确定有效视角数，并按同一置换随机打乱视角顺序（同步作用于 condition/reference/camera/bbox 等）
        view_files_sorted = sorted(view_files)
        num_views_eff = min(num_views, len(view_files_sorted))
        selected_views = view_files_sorted[:num_views_eff]
        perm = list(range(num_views_eff))
        # random.shuffle(perm)
        permuted_views = [selected_views[i] for i in perm]
        # TODO
        # permuted_views = [selected_views[2],selected_views[2],selected_views[2],selected_views[2]]
        print(f"[{subject_id}] 视角置换: {perm}")

        # === 第一步：对当前 subject 做"整体身体 Seg"预测，用于自动推理手部 bbox ===
        # 视角 ID（文件名后缀）
        view_ids_local = [int(os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1]) for p in permuted_views]

        # 手部部位列表（左右手）
        part_names_order = ["left_hand", "right_hand"]
        # 预先构造 overrides 容器
        part_to_overrides = {pn: [None] * num_views_eff for pn in part_names_order}

        # 手部 bbox 保存目录（与原逻辑保持一致）
        bbox_hf_root = os.path.join(os.path.dirname(output_dir), "bbox_hand_foot")

        # ===== 检测当前 subject 是否已经有完整的手部 bbox，如有则跳过身体 Seg 推理 =====
        has_existing_hand_bboxes = False
        for pn in part_names_order:
            part_dir = os.path.join(bbox_hf_root, pn)
            for vi, vid in enumerate(view_ids_local):
                bbox_json_path = os.path.join(part_dir, f"{subject_id}_{vid:02d}.json")
                if os.path.exists(bbox_json_path):
                    has_existing_hand_bboxes = True
                    break
            if has_existing_hand_bboxes:
                break

        if has_existing_hand_bboxes:
            print(f"[{subject_id}] 检测到现有手部 bbox，跳过身体 Seg 推理，直接加载手部 bbox")
            # 从已有的 bbox JSON 中恢复 hand overrides
            for vi, p in enumerate(permuted_views):
                img_temp_rgba = Image.open(p).convert("RGBA")
                h_img, w_img = np.array(img_temp_rgba).shape[:2]
                view_id_i = view_ids_local[vi]
                for pn in part_names_order:
                    bbox_json_path = os.path.join(bbox_hf_root, pn, f"{subject_id}_{view_id_i:02d}.json")
                    if not os.path.exists(bbox_json_path):
                        continue
                    with open(bbox_json_path, "r") as f:
                        data = json.load(f)
                    tlwh = data.get("tlwh", None)
                    if tlwh is None or len(tlwh) != 4:
                        continue
                    t, l, w_box, h_box = tlwh
                    x_min = t / w_img
                    y_min = l / w_img  # 与身体 bbox 归一化方式保持一致（用宽做归一化）
                    x_max = (t + w_box) / w_img
                    y_max = (l + h_box) / w_img
                    override_norm = ensure_valid_bbox_norm([x_min, y_min, x_max, y_max])
                    part_to_overrides[pn][vi] = override_norm

        else:
            print(f"[{subject_id}] 未检测到完整手部 bbox，将执行身体 Seg 推理并自动生成手部 bbox")

            # 仅在需要生成新 bbox 时，才运行身体 Seg 推理和自动 hand bbox 推理

            def _part_label(name: str):
                return "full body" if name == "body" else name.replace("_", " ")

            # 构造仅包含身体的 UV / Seg prompts
            uv_prompts_body = []
            seg_prompts_body = []
            for i in range(num_views_eff):
                view_id_i = view_ids_local[i]
                label_i = _part_label("body")
                uv_prompts_body.append(f"view {view_id_i:02d}: {label_i} uv map")
                seg_prompts_body.append(f"view {view_id_i:02d}: {label_i} segmentation map")
            batch_body = {
                "prompts": uv_prompts_body + seg_prompts_body,
            }

            # reference：使用身体 bbox 裁剪
            references_body = [
                load_image(
                    permuted_views[i],
                    True,
                    h_tgt,
                    w_tgt,
                    None,
                ) for i in range(num_views_eff)
            ]
            reference_rgb_body = torch.cat(references_body, dim=0)  # [V,3,H,W]
            batch_body["reference_rgb"] = reference_rgb_body[None, None].repeat(1, 2, 1, 1, 1, 1).flatten(0, 2)

            # 相机参数（同样使用身体 bbox 修正）
            camera_dir_path_body = [p.replace("image_ori", "camera") for p in permuted_views]
            cameras_body = [
                load_camera(p, True, h_tgt, w_tgt, h_ori, w_ori, None)
                for p in camera_dir_path_body
            ]
            all_extrinsics_body = torch.stack([cam["extrinsic"] for cam in cameras_body], dim=0)
            all_intrinsics_body = torch.stack([cam["intrinsic"] for cam in cameras_body], dim=0)
            batch_body["all_cameras"] = all_extrinsics_body.unsqueeze(0)
            batch_body["dataset_extrinsics"] = all_extrinsics_body[None]
            batch_body["dataset_intrinsics"] = all_intrinsics_body[None]

            root_dir = os.path.dirname(os.path.dirname(view_files[0]))  # .../gt
            cam_base_body = os.path.splitext(camera_dir_path_body[0])[0]
            cam_base_body = cam_base_body.rsplit("_", 1)[0]
            batch_body["camera_dir"] = cam_base_body
            batch_body["num_views"] = num_views_eff
            cam_files_body = [f"{os.path.basename(cam_base_body)}_{vid:02d}.txt" for vid in view_ids_local]
            batch_body["cam_files"] = [cam_files_body]

            cache_dir = os.path.join(root_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            batch_body["cache_dir"] = [cache_dir]

            # 条件图像：与正式推理保持一致，但仅使用身体视角（无手部 overrides）
            condition_image_modality = cfg.data.condition_image_modality
            if len(condition_image_modality) > 0:
                if "image" in condition_image_modality:
                    condition_images_body = [
                        load_image(permuted_views[i], True, h_tgt, w_tgt, None)
                        for i in range(num_views_eff)
                    ]
                    condition_rgb_body = torch.cat(condition_images_body, dim=0)
                    batch_body["condition_rgb"] = condition_rgb_body[None, None].repeat(1, 2, 1, 1, 1, 1).flatten(0, 2)
                if "pose" in condition_image_modality:
                    condition_pose_paths_body = [
                        os.path.splitext(p.replace("image_ori", "kpts_rgb"))[0] + ".png"
                        for p in permuted_views
                    ]
                    condition_poses_body = [
                        load_image(p, False, h_tgt, w_tgt) for p in condition_pose_paths_body
                    ]
                    condition_poses_body = torch.cat(condition_poses_body, dim=0)
                    pose_tensor_body = condition_poses_body
                    pose_tensor_body = pose_tensor_body[None, None].repeat(1, 2, 1, 1, 1, 1).flatten(0, 2)
                    if "condition_rgb" in batch_body:
                        batch_body["condition_rgb"] = torch.cat([batch_body["condition_rgb"], pose_tensor_body], dim=1)
                    else:
                        batch_body["condition_rgb"] = pose_tensor_body
                if "plucker" in condition_image_modality:
                    plucker_embeds_body = get_plucker(cameras_body, h_tgt, w_tgt)
                    plucker_flat_body = torch.cat([plucker_embeds_body, plucker_embeds_body], dim=0)
                    if "condition_rgb" in batch_body:
                        batch_body["condition_rgb"] = torch.cat(
                            [batch_body["condition_rgb"] * 2 - 1, plucker_flat_body], dim=1
                        )
                    else:
                        batch_body["condition_rgb"] = plucker_flat_body

            # all_bboxes / view_ids 元信息（身体 bbox）
            all_bboxes_body = []
            all_view_ids_body = []
            for i, p in enumerate(permuted_views):
                img_temp = Image.open(p).convert("RGBA")
                h_img, w_img = np.array(img_temp).shape[:2]
                bbox_norm = load_bbox(p, w_img, h_img)
                # 若 bbox 为 None，使用整个图像作为默认 bbox
                if bbox_norm is None:
                    bbox_norm = [0.0, 0.0, 1.0, h_img / w_img]
                all_bboxes_body.append(torch.tensor(bbox_norm, dtype=torch.float32))
                all_view_ids_body.append(view_ids_local[i])
            batch_body["all_bboxes"] = torch.stack(all_bboxes_body, dim=0).unsqueeze(0)
            batch_body["all_view_ids"] = torch.tensor(all_view_ids_body, dtype=torch.long).unsqueeze(0)

            # SDXL 额外信息
            batch_body.update({
                "original_size": (h_ori, w_ori),
                "target_size": (h_tgt, w_tgt),
                "crops_coords_top_left": (0, 0),
            })

            # 迁移至 GPU
            for k in batch_body:
                if isinstance(batch_body[k], torch.Tensor):
                    batch_body[k] = batch_body[k].to(device=system.device, dtype=torch.float32)

            # 推理整体身体 Seg
            with torch.no_grad():
                if hasattr(system, "pipeline"):
                    system.pipeline.to(dtype=torch.float32)

                if enable_uncertainty:
                    sample_cache_dir_body = os.path.join(os.path.dirname(output_dir), "sample_cache")
                    sample_cache_prefix_body = f"{subject_id}_body_v{num_views_eff}"
                    body_first_offset = sample_seed_offsets[0]
                    body_seeds = [int(BASE_SEED) + (body_first_offset + i) * 1000 for i in range(body_samples)]
                    body_seed_results = generate_unique_seeds_results(
                        system, batch_body, all_seeds=body_seeds,
                        sample_cache_dir=sample_cache_dir_body,
                        sample_cache_prefix=sample_cache_prefix_body,
                    )
                    results_body = collect_results_for_offset(
                        body_seed_results, body_samples, body_first_offset, BASE_SEED,
                    )
                    view_types_body = ["body"] * int(num_views_eff)
                    unc_body = calculate_uncertainty_maps(results_body, num_views_eff, view_types=view_types_body)
                    seg_body = unc_body["seg_best"]  # [V,C,H,W]
                else:
                    out_body = system.generate_images(batch_body)
                    seg_body = out_body[num_views_eff:]  # [V,C,H,W]

            # 可选：保存整体 seg 可视化，方便 debug
            seg_body_path = os.path.join(output_dir, f"seg_{subject_id}_pred_body.png")
            vutils.save_image(
                seg_body.clamp(0, 1),
                seg_body_path,
                normalize=False,
                nrow=num_views_eff,
                padding=0,
            )
            print(f"保存整体身体 Seg 到: {seg_body_path}")

            # === 基于整体 Seg 自动推理手部 bbox（与 v3.1 脚本逻辑一致） ===
            for vi, p in enumerate(permuted_views):
                img_temp_rgba = Image.open(p).convert("RGBA")
                h_img, w_img = np.array(img_temp_rgba).shape[:2]
                # 先根据身体 bbox 预计算仿射参数（原图 -> 裁剪图）
                body_bbox_norm = load_bbox(p, w_img, h_img)
                if body_bbox_norm is None:
                    # 没有身体 bbox 时，无法推理手部，直接跳过
                    continue
                cx = (body_bbox_norm[0] + body_bbox_norm[2]) / 2.0
                cy = (body_bbox_norm[1] + body_bbox_norm[3]) / 2.0
                bw = body_bbox_norm[2] - body_bbox_norm[0]
                bh = body_bbox_norm[3] - body_bbox_norm[1]
                side = max(bw, bh) * DEFAULT_BBOX_RATIO
                new_xmin_tmp = cx - side/2.0; new_xmax_tmp = cx + side/2.0
                new_ymin_tmp = cy - side/2.0; new_ymax_tmp = cy + side/2.0
                new_xmin = max(new_xmin_tmp, 0.0); new_xmax = min(new_xmax_tmp, 1.0)
                new_ymin = max(new_ymin_tmp, 0.0); new_ymax = min(new_ymax_tmp, h_img / w_img)
                tgt_xmin = round((new_xmin - new_xmin_tmp) * w_tgt / side)
                tgt_xmax = round((new_xmax - new_xmin_tmp) * w_tgt / side)
                tgt_ymin = round((new_ymin - new_ymin_tmp) * h_tgt / side)
                tgt_ymax = round((new_ymax - new_ymin_tmp) * h_tgt / side)
                crop_x0 = round(new_xmin * w_img); crop_x1 = round(new_xmax * w_img)
                crop_y0 = round(new_ymin * w_img); crop_y1 = round(new_ymax * w_img)
                patch_w = max(1, crop_x1 - crop_x0); patch_h = max(1, crop_y1 - crop_y0)
                tgt_w_eff = max(1, tgt_xmax - tgt_xmin); tgt_h_eff = max(1, tgt_ymax - tgt_ymin)
                sx = tgt_w_eff / patch_w; sy = tgt_h_eff / patch_h
                ox = tgt_xmin - sx * crop_x0; oy = tgt_ymin - sy * crop_y0

                # 从整体 Seg 结果中取出对应视角的分割，并量化为 label id
                seg_view = seg_body[vi]  # [C,H,W]
                if seg_view.dim() < 3 or seg_view.size(0) < 3:
                    continue
                seg_rgb = (seg_view[:3].detach().cpu().float().clamp(0, 1).permute(1, 2, 0).numpy() * 255.0)
                seg_idx = _quantize_seg_by_palette(seg_rgb)
                th, tw = seg_idx.shape

                for part_name in part_names_order:
                    class_id = PART_LABELS[part_name]
                    comp = _largest_cc(seg_idx == class_id)
                    if comp is None:
                        continue
                    ys, xs = np.where(comp)
                    if ys.size == 0:
                        continue
                    x0, y0, x1, y1 = int(xs.min()), int(ys.min()), int(xs.max())+1, int(ys.max())+1
                    # 手部区域的 bbox 在 Seg 图中做一次方形扩张
                    x0e, y0e, x1e, y1e = _expand_square(x0, y0, x1, y1, tw, th, HAND_FOOT_BBOX_RATIO)
                    # 将 Seg 坐标反投到原图坐标
                    X0 = int(round((x0e - ox) / max(sx, 1e-8)))
                    Y0 = int(round((y0e - oy) / max(sy, 1e-8)))
                    X1 = int(round((x1e - ox) / max(sx, 1e-8)))
                    Y1 = int(round((y1e - oy) / max(sy, 1e-8)))
                    X0 = max(0, min(X0, w_img)); X1 = max(0, min(X1, w_img))
                    Y0 = max(0, min(Y0, h_img)); Y1 = max(0, min(Y1, h_img))
                    if X1 - X0 < 4 or Y1 - Y0 < 4:
                        continue
                    x_min = X0 / w_img
                    y_min = Y0 / w_img  # 与身体 bbox 归一化方式保持一致
                    x_max = X1 / w_img
                    y_max = Y1 / w_img
                    override_norm = ensure_valid_bbox_norm([x_min, y_min, x_max, y_max])
                    part_to_overrides[part_name][vi] = override_norm

            # 将手部 bbox 保存到 exp 目录的 bbox_hand_foot/<part>/subject_view.json
            # 要求：保存"平方化+scale 后"的最终 bbox（允许超出边界）
            for pn in part_names_order:
                os.makedirs(os.path.join(bbox_hf_root, pn), exist_ok=True)

            # 文件名中的视角 ID（与输入文件名后缀一致）
            for vi, p in enumerate(permuted_views):
                img_rgba = Image.open(p).convert("RGBA")
                h_img_i, w_img_i = np.array(img_rgba).shape[:2]
                view_id_i = view_ids_local[vi]
                for pn in part_names_order:
                    coords_norm = part_to_overrides.get(pn, [None] * num_views_eff)[vi]
                    if coords_norm is None:
                        continue
                    x_min_n, y_min_n, x_max_n, y_max_n = coords_norm
                    # 直接存 override_norm 对应的像素 bbox（已含 _expand_square 的 1.5x 扩张，不再额外放大）
                    t_px = int(round(x_min_n * w_img_i))
                    l_px = int(round(y_min_n * w_img_i))
                    w_px = int(round((x_max_n - x_min_n) * w_img_i))
                    h_px = int(round((y_max_n - y_min_n) * w_img_i))
                    out_json = {"tlwh": [t_px, l_px, w_px, h_px]}
                    out_path = os.path.join(bbox_hf_root, pn, f"{subject_id}_{view_id_i:02d}.json")
                    with open(out_path, "w") as f:
                        json.dump(out_json, f)
            print(f"保存手部 bbox 到: {bbox_hf_root}")


        # === 两种推理模式：分步推理 vs 一次性推理 ===
        if inference_both_hands_once:
            # 一次性推理模式：12 视角（4 身体 + 4 左手 + 4 右手）
            inference_steps = [("both_hands", part_names_order)]
        else:
            # 分步推理模式：两次推理，每次 8 视角（4 身体 + 4 指定部位）
            inference_steps = [(part_name, [part_name]) for part_name in part_names_order]
        
        for step_tag, parts_in_step in inference_steps:
            views_all = []
            overrides_all = []
            view_ids_all = []
            
            # 前 N：身体视角
            for vi, p in enumerate(permuted_views):
                views_all.append(p)
                overrides_all.append(None)
                vid_token = os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1]
                view_ids_all.append(int(vid_token))
            
            # 后续：各个部位视角
            for part_name in parts_in_step:
                for vi, p in enumerate(permuted_views):
                    views_all.append(p)
                    override_valid = part_to_overrides.get(part_name, [None]*num_views_eff)[vi]
                    if override_valid is None:
                        overrides_all.append(None)
                    else:
                        overrides_all.append(override_valid)
                    vid_token = os.path.splitext(os.path.basename(p))[0].rsplit("_", 1)[-1]
                    view_ids_all.append(int(vid_token))
            
            # 重命名以保持后续代码兼容
            views8 = views_all
            overrides8 = overrides_all
            view_ids8 = view_ids_all

            total_views = len(views8)

            # 按视角与部位生成 prompts（后半手部视角若回退为全身，则使用身体prompt）
            def _part_label(name: str):
                return "full body" if name == "body" else name.replace("_", " ")
            uv_prompts = []
            seg_prompts = []
            for i in range(total_views):
                # 判断当前视角属于哪个部位
                if i < num_views_eff:
                    # 身体视角
                    part_name_i = "body"
                else:
                    # 计算是第几个部位的第几个视角
                    offset = i - num_views_eff
                    part_idx = offset // num_views_eff
                    if part_idx < len(parts_in_step):
                        part_name_i = parts_in_step[part_idx]
                        # 如果该视角的 override 为 None，则回退为身体
                        if overrides8[i] is None:
                            part_name_i = "body"
                    else:
                        part_name_i = "body"
                
                label_i = _part_label(part_name_i)
                view_id_i = view_ids8[i]
                uv_prompts.append(f"view {view_id_i:02d}: {label_i} uv map")
                seg_prompts.append(f"view {view_id_i:02d}: {label_i} segmentation map")
            batch = {
                "prompts": uv_prompts + seg_prompts,
            }
            # load reference images（与 condition 完全一致：逐视角传入对应 overrides8）
            references = [load_image(
                views8[i],
                True,
                h_tgt,
                w_tgt,
                overrides8[i],
            ) for i in range(total_views)]
            reference_rgb = torch.cat(references, dim=0)  # [n_ref, 3, H, W]
            batch["reference_rgb"] = reference_rgb[None,None].repeat(1, 2, 1, 1, 1, 1).flatten(0, 2)  
            # load cameras
            camera_dir_path = [p.replace("image_ori", "camera") for p in views8]
            cameras = [load_camera(p, True, h_tgt, w_tgt, h_ori, w_ori, overrides8[i]) for i, p in enumerate(camera_dir_path)]
            all_extrinsics = torch.stack([cam["extrinsic"] for cam in cameras], dim=0)  # (V,4,4)
            batch["all_cameras"] = all_extrinsics.unsqueeze(0)
            all_intrinsics = torch.stack([cam["intrinsic"] for cam in cameras], dim=0)  # (V,3,3)
            batch["dataset_extrinsics"] = all_extrinsics[None]
            batch["dataset_intrinsics"] = all_intrinsics[None]

            root_dir = os.path.dirname(os.path.dirname(view_files[0]))  # .../gt

            cam_base = os.path.splitext(camera_dir_path[0])[0]
            cam_base = cam_base.rsplit("_", 1)[0]
            batch["camera_dir"] = cam_base
            batch["num_views"] = total_views

            cam_files = [f"{os.path.basename(cam_base)}_{vid:02d}.txt" for vid in view_ids8]
            batch["cam_files"] = [cam_files]

            cache_dir = os.path.join(root_dir, "cache")
            os.makedirs(cache_dir, exist_ok=True)
            batch["cache_dir"] = [cache_dir]

            # 条件
            condition_image_modality = cfg.data.condition_image_modality
            if len(condition_image_modality) > 0:
                if "image" in condition_image_modality:
                    condition_images = [load_image(views8[i], True, h_tgt, w_tgt, overrides8[i]) for i in range(total_views)]
                    condition_rgb = torch.cat(condition_images, dim=0)
                    batch["condition_rgb"] = condition_rgb[None,None].repeat(1, 2, 1, 1, 1, 1).flatten(0, 2)
                if "pose" in condition_image_modality:
                    condition_pose_paths = [os.path.splitext(p.replace("image_ori", "kpts_rgb"))[0] + ".png" for p in views8]
                    condition_poses = [load_image(p, False, h_tgt, w_tgt) for p in condition_pose_paths]
                    condition_poses = torch.cat(condition_poses, dim=0)
                    pose_tensor = condition_poses
                    pose_tensor = pose_tensor[None,None].repeat(1, 2, 1, 1, 1, 1).flatten(0, 2)
                    if "condition_rgb" in batch:
                        batch["condition_rgb"] = torch.cat([batch["condition_rgb"], pose_tensor], dim=1)
                    else:
                        batch["condition_rgb"] = pose_tensor
                if "plucker" in condition_image_modality:
                    plucker_embeds = get_plucker(cameras, h_tgt, w_tgt)
                    plucker_flat = torch.cat([plucker_embeds, plucker_embeds], dim=0)
                    if "condition_rgb" in batch:
                        batch["condition_rgb"] = torch.cat([batch["condition_rgb"] * 2 - 1, plucker_flat], dim=1)
                    else:
                        batch["condition_rgb"] = plucker_flat

            # all_bboxes / view_ids 元信息
            all_bboxes = []
            all_view_ids = []
            for i, p in enumerate(views8):
                img_temp = Image.open(p).convert("RGBA")
                h_img, w_img = np.array(img_temp).shape[:2]
                if overrides8[i] is not None:
                    bbox_norm = overrides8[i]
                else:
                    bbox_norm = load_bbox(p, w_img, h_img)
                # 若 bbox 为 None，使用整个图像作为默认 bbox
                if bbox_norm is None:
                    bbox_norm = [0.0, 0.0, 1.0, h_img / w_img]
                all_bboxes.append(torch.tensor(bbox_norm, dtype=torch.float32))
                all_view_ids.append(view_ids8[i])
            batch["all_bboxes"] = torch.stack(all_bboxes, dim=0).unsqueeze(0)
            batch["all_view_ids"] = torch.tensor(all_view_ids, dtype=torch.long).unsqueeze(0)

            # SDXL 额外信息
            batch.update({
                "original_size": (h_ori, w_ori),
                "target_size": (h_tgt, w_tgt),
                "crops_coords_top_left": (0, 0),
            })

            # 迁移至 GPU
            for k in batch:
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(device=system.device, dtype=torch.float32)
            if "mesh_vertices" in batch and isinstance(batch["mesh_vertices"], list) and len(batch["mesh_vertices"])>0 and isinstance(batch["mesh_vertices"][0], torch.Tensor):
                batch["mesh_vertices"][0] = batch["mesh_vertices"][0].to(device=system.device, dtype=torch.float32)

            # 推理
            with torch.no_grad():
                if hasattr(system, "pipeline"):
                    system.pipeline.to(dtype=torch.float32)

                if enable_uncertainty:
                    # 汇总所有 (tts, offset) 组合需要的 seed 并集，一次性推理
                    sample_cache_dir = os.path.join(os.path.dirname(output_dir), "sample_cache")
                    sample_cache_prefix = f"{subject_id}_{step_tag}_v{total_views}"
                    all_hand_seeds = []
                    for hs in hand_samples_list:
                        for ofs in sample_seed_offsets:
                            for i in range(hs):
                                all_hand_seeds.append(int(BASE_SEED) + (ofs + i) * 1000)

                    # PNG → .pt 迁移：将旧代码产出的 _sXX.png 转为 .pt 缓存
                    os.makedirs(sample_cache_dir, exist_ok=True)
                    unique_seeds_for_migration = list(dict.fromkeys(all_hand_seeds))
                    migrated_count = 0
                    for _seed in unique_seeds_for_migration:
                        pt_path = os.path.join(sample_cache_dir, f"{sample_cache_prefix}_seed{_seed}.pt")
                        if os.path.exists(pt_path):
                            continue
                        _sidx = (_seed - int(BASE_SEED)) // 1000
                        uv_png = os.path.join(output_dir, f"uv_{subject_id}_pred_{step_tag}_s{_sidx:02d}.png")
                        seg_png = os.path.join(output_dir, f"seg_{subject_id}_pred_{step_tag}_s{_sidx:02d}.png")
                        if os.path.exists(uv_png) and os.path.exists(seg_png):
                            uv_arr = np.array(Image.open(uv_png).convert("RGB")).astype(np.float32) / 255.0
                            seg_arr = np.array(Image.open(seg_png).convert("RGB")).astype(np.float32) / 255.0
                            h_tile = uv_arr.shape[0]
                            w_tile = uv_arr.shape[1] // total_views
                            uv_tiles = torch.stack([torch.from_numpy(uv_arr[:, j*w_tile:(j+1)*w_tile, :]).permute(2, 0, 1) for j in range(total_views)])
                            seg_tiles = torch.stack([torch.from_numpy(seg_arr[:, j*w_tile:(j+1)*w_tile, :]).permute(2, 0, 1) for j in range(total_views)])
                            combined = torch.cat([uv_tiles, seg_tiles], dim=0)
                            torch.save(combined, pt_path)
                            migrated_count += 1
                    if migrated_count > 0:
                        print(f"[png→pt] {subject_id}/{step_tag}: 从旧 PNG 迁移了 {migrated_count} 个样本到 .pt 缓存")

                    hand_seed_results = generate_unique_seeds_results(
                        system, batch, all_seeds=all_hand_seeds,
                        sample_cache_dir=sample_cache_dir,
                        sample_cache_prefix=sample_cache_prefix,
                    )

                    view_types = []
                    for i in range(total_views):
                        if i < num_views_eff:
                            view_types.append("body")
                        else:
                            voffset = i - num_views_eff
                            part_idx = voffset // num_views_eff
                            if part_idx < len(parts_in_step) and overrides8[i] is not None:
                                view_types.append("hand")
                            else:
                                view_types.append("body")

                    # 保存所有唯一 seed 的逐样本 PNG（只保存一次，不重复写）
                    saved_sample_pngs = set()
                    for seed_val, res in hand_seed_results.items():
                        sample_idx = (seed_val - int(BASE_SEED)) // 1000
                        if sample_idx in saved_sample_pngs:
                            continue
                        saved_sample_pngs.add(sample_idx)
                        uv_each = res[:total_views]
                        seg_each = res[total_views:]
                        uv_each_path = os.path.join(output_dir, f"uv_{subject_id}_pred_{step_tag}_s{sample_idx:02d}.png")
                        seg_each_path = os.path.join(output_dir, f"seg_{subject_id}_pred_{step_tag}_s{sample_idx:02d}.png")
                        vutils.save_image(uv_each.clamp(0, 1), uv_each_path, normalize=False, nrow=total_views, padding=0)
                        vutils.save_image(seg_each.clamp(0, 1), seg_each_path, normalize=False, nrow=total_views, padding=0)

                    # 按每个 (tts, offset) 对做 uncertainty 选择 + 保存 best
                    for cur_hand_samples in hand_samples_list:
                        for current_offset in sample_seed_offsets:
                            results = collect_results_for_offset(
                                hand_seed_results, cur_hand_samples, current_offset, BASE_SEED,
                            )
                            unc = calculate_uncertainty_maps(results, total_views, view_types=view_types)
                            uv_output = unc['uv_best']
                            seg_output = unc['seg_best']

                            combo_tag = f"tts{cur_hand_samples:02d}_v{current_offset:02d}"
                            uv_output_path = os.path.join(output_dir, f"uv_{subject_id}_pred_{step_tag}_{combo_tag}.png")
                            vutils.save_image(uv_output.clamp(0, 1), uv_output_path, normalize=False, nrow=total_views, padding=0)
                            print(f"保存 UV 结果到: {uv_output_path}")

                            seg_output_path = os.path.join(output_dir, f"seg_{subject_id}_pred_{step_tag}_{combo_tag}.png")
                            vutils.save_image(seg_output.clamp(0, 1), seg_output_path, normalize=False, nrow=total_views, padding=0)
                            print(f"保存 Seg 结果到: {seg_output_path}")

                            if inference_both_hands_once:
                                print(f"  → 一次性推理模式 [{combo_tag}]: {total_views} 视角 = {num_views_eff} 身体 + {num_views_eff} 左手 + {num_views_eff} 右手")

                            uv_unc_path = os.path.join(output_dir, f"uv_uncertainty_{subject_id}_pred_{step_tag}_{combo_tag}.png")
                            seg_unc_path = os.path.join(output_dir, f"seg_uncertainty_{subject_id}_pred_{step_tag}_{combo_tag}.png")
                            uv_unc_viz = unc['uv_uncertainty_mask']
                            seg_unc_viz = unc['seg_uncertainty_mask']
                            uv_max = uv_unc_viz.max().item() if hasattr(uv_unc_viz, 'max') else 0.0
                            seg_max = seg_unc_viz.max().item() if hasattr(seg_unc_viz, 'max') else 0.0
                            if uv_max > 0:
                                uv_unc_viz = uv_unc_viz / uv_max
                            if seg_max > 0:
                                seg_unc_viz = seg_unc_viz / seg_max
                            vutils.save_image(uv_unc_viz.repeat(1, 3, 1, 1).clamp(0, 1), uv_unc_path, normalize=False, nrow=total_views, padding=0)
                            vutils.save_image(seg_unc_viz.repeat(1, 3, 1, 1).clamp(0, 1), seg_unc_path, normalize=False, nrow=total_views, padding=0)

                    # demo 可视化只保存一次（不依赖 offset）
                    ref_demo_path = os.path.join(demo_dir, f"ref_{subject_id}_pred_{step_tag}.png")
                    vutils.save_image(reference_rgb.clamp(0, 1), ref_demo_path, normalize=False, nrow=total_views, padding=0)
                    print(f"保存 Reference 到: {ref_demo_path}")

                    if "image" in condition_image_modality:
                        cond_demo_path = os.path.join(demo_dir, f"cond_{subject_id}_pred_{step_tag}.png")
                        vutils.save_image(condition_rgb.clamp(0, 1), cond_demo_path, normalize=False, nrow=total_views, padding=0)
                        print(f"保存 Condition 到: {cond_demo_path}")

                    if "pose" in condition_image_modality:
                        pose_demo_path = os.path.join(demo_dir, f"pose_{subject_id}_pred_{step_tag}.png")
                        vutils.save_image(condition_poses.clamp(0, 1), pose_demo_path, normalize=False, nrow=total_views, padding=0)
                        print(f"保存 Pose 到: {pose_demo_path}")

                    if "prompts" in batch and isinstance(batch["prompts"], (list, tuple)) and len(batch["prompts"]) >= total_views:
                        uv_prompts = list(batch["prompts"][:total_views])
                        seg_prompts = list(batch["prompts"][total_views:total_views*2]) if len(batch["prompts"]) >= total_views*2 else ["N/A"] * total_views
                        prompts_txt_path = os.path.join(demo_dir, f"prompts_{subject_id}_pred_{step_tag}.txt")
                        with open(prompts_txt_path, "w", encoding="utf-8") as f:
                            for i in range(total_views):
                                fname = os.path.basename(views8[i]) if i < len(views8) else str(i)
                                uv_p = uv_prompts[i] if i < len(uv_prompts) else "N/A"
                                seg_p = seg_prompts[i] if i < len(seg_prompts) else "N/A"
                                f.write(f"{i:02d}\t{fname}\tuv: {uv_p}\tseg: {seg_p}\n")
                        print(f"保存 Prompts 到: {prompts_txt_path}")

                else:
                    out = system.generate_images(batch)
                    uv_output = out[:total_views]
                    seg_output = out[total_views:]

                    uv_output_path = os.path.join(output_dir, f"uv_{subject_id}_pred_{step_tag}.png")
                    vutils.save_image(uv_output.clamp(0, 1), uv_output_path, normalize=False, nrow=total_views, padding=0)
                    print(f"保存 UV 结果到: {uv_output_path}")

                    seg_output_path = os.path.join(output_dir, f"seg_{subject_id}_pred_{step_tag}.png")
                    vutils.save_image(seg_output.clamp(0, 1), seg_output_path, normalize=False, nrow=total_views, padding=0)
                    print(f"保存 Seg 结果到: {seg_output_path}")

                    if inference_both_hands_once:
                        print(f"  → 一次性推理模式: {total_views} 视角 = {num_views_eff} 身体 + {num_views_eff} 左手 + {num_views_eff} 右手")

            torch.cuda.empty_cache()

        # === 每个 subject 推理完成后，对每个 (tts, offset) 对分别执行第二步整理与回贴 ===
        base_cams_for_post = int(num_views_eff)

        if enable_uncertainty:
            for cur_hand_samples in hand_samples_list:
                for current_offset in sample_seed_offsets:
                    print(
                        f"[{subject_id}] 后处理 tts{cur_hand_samples:02d}_v{current_offset:02d}：NUM_BASE_CAMS={base_cams_for_post}，源目录: {output_dir}"
                    )
                    run_step2_postprocess(
                        image_crop_raw_dir=output_dir,
                        tts_count=int(cur_hand_samples),
                        total_samples=int(cur_hand_samples),
                        variant_count=1,
                        sample_index_offset=current_offset,
                        shuffle_variants=False,
                        shuffle_seed=1314,
                        only_subjects={subject_id},
                        base_cams=base_cams_for_post,
                    )
        else:
            run_step2_postprocess(
                image_crop_raw_dir=output_dir,
                tts_count=1,
                total_samples=1,
                variant_count=1,
                sample_index_offset=0,
                shuffle_variants=False,
                shuffle_seed=1314,
                only_subjects={subject_id},
                base_cams=base_cams_for_post,
            )

    # 卸载模型
    print("卸载模型...")
    del system
    gc.collect()
    torch.cuda.empty_cache()
    print("测试完成")
    # 清理临时配置目录
    if 'tmp_dir' in locals() and os.path.isdir(tmp_dir):
        shutil.rmtree(tmp_dir, ignore_errors=True)
    
    # 返回所有批次的输出，用于下一个模型




def discover_subjects(task):
    """扫描 task 对应 input_folder 中的所有 subject_id（不加载模型）"""
    exp = task["exp"]
    dataset = task.get("dataset", None)
    base_data_dir = task.get("base_data_dir", "")
    input_folder = task.get("input_folder", None)
    if input_folder is None:
        input_folder = f"{base_data_dir}/{dataset}/gt/image_ori"
    image_files = sorted(
        glob.glob(os.path.join(input_folder, "*.jpg")) +
        glob.glob(os.path.join(input_folder, "*.png"))
    )
    subjects = set()
    for f in image_files:
        subjects.add(os.path.basename(f).split("_")[0])
    subjects_sorted = sorted(subjects)
    requested_subjects = task.get("subjects")
    if requested_subjects:
        selected = filter_subject_items(
            [(subject_id, None) for subject_id in subjects_sorted],
            requested_subjects,
        )
        subjects_sorted = [subject_id for subject_id, _ in selected]
    # Optional debug-only subject limit.
    _max = os.environ.get("MAX_SUBJECTS", None)
    if _max is not None:
        try:
            n = int(_max)
            if n > 0 and n < len(subjects_sorted):
                print(f"[MAX_SUBJECTS] capping subjects: {len(subjects_sorted)} -> {n}")
                subjects_sorted = subjects_sorted[:n]
        except ValueError:
            pass
    return subjects_sorted


# 多进程GPU任务调度器
def gpu_worker(gpu_id, task, subject_queue):
    """GPU工作进程：加载模型后从共享队列动态抢占 subject"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    print(f"[GPU {gpu_id}] 工作进程启动, PID={os.getpid()}, "
          f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
          f"torch.cuda.device_count()={torch.cuda.device_count()}")
    run_single_task(task, subject_queue=subject_queue)
    print(f"[GPU {gpu_id}] 工作进程结束")
        

def run_single_task(task, subject_queue=None):
    """在指定GPU上运行单个任务"""
    exp = task["exp"]
    dataset = task.get("dataset", None)
    proxy_vae_path = task.get("proxy_vae_path", None)
    config_path = task.get("config_path", f"configs/view-guidance/{exp}.yaml")
    enable_uncertainty = task.get("enable_uncertainty", False)
    num_uncertainty_samples_list = task.get("num_uncertainty_samples_list", [5])
    if isinstance(num_uncertainty_samples_list, int):
        num_uncertainty_samples_list = [num_uncertainty_samples_list]
    body_views = task.get("body_views", None)
    body_num_uncertainty_samples = task.get("body_num_uncertainty_samples", None)
    hand_num_uncertainty_samples_list = task.get("hand_num_uncertainty_samples_list", None)
    inference_both_hands_once = task.get("inference_both_hands_once", False)
    sample_seed_offsets = task.get("sample_seed_offsets", [DEFAULT_SAMPLE_SEED_OFFSET])
    subject_ids = task.get("subjects")
    if isinstance(sample_seed_offsets, int):
        sample_seed_offsets = [sample_seed_offsets]
    sample_seed_offsets = [int(o) for o in sample_seed_offsets]
    
    # 从 task 中获取输入输出路径（支持自定义或使用默认模板）
    input_folder = task.get("input_folder", None)
    output_dir = task.get("output_dir", None)
    base_data_dir = task.get("base_data_dir", "")
    
    # 如果未直接指定路径，则使用模板生成
    if input_folder is None:
        input_folder = f"{base_data_dir}/{dataset}/gt/image_ori"
    if output_dir is None:
        dir_tag = f"ours_{int(body_views)}v"
        output_dir = f"{base_data_dir}/{dataset}/{dir_tag}/{exp}/tempt"
    
    os.makedirs(output_dir, exist_ok=True)

    # MV-Adapter 推理与后处理在 test_model 内按样本完成
    test_model(
        config_path,
        output_dir,
        input_folder=input_folder,
        proxy_vae_path=proxy_vae_path,
        enable_uncertainty=enable_uncertainty,
        num_uncertainty_samples_list=num_uncertainty_samples_list,
        body_views=body_views,
        body_num_uncertainty_samples=body_num_uncertainty_samples,
        hand_num_uncertainty_samples_list=hand_num_uncertainty_samples_list,
        inference_both_hands_once=inference_both_hands_once,
        sample_seed_offsets=sample_seed_offsets,
        subject_queue=subject_queue,
        subject_ids=subject_ids,
    )


def run_tasks_with_single_thread(tasks):
    for i, task in enumerate(tasks):
        print(f"\n[单线程] 开始执行任务 {i+1}/{len(tasks)}: {task.get('exp', 'Unknown')}")
        run_single_task(task)
        print(f"[单线程] 完成任务 {i+1}/{len(tasks)}")
    
    print("\n[单线程] 所有任务已完成")


def run_tasks_with_multiprocessing(tasks, gpu_ids=None):
    """多进程多GPU并行执行任务（按 subject 级别的进程池）

    逐 task 串行执行。每个 task 内，发现所有 subject 放入共享队列，
    每个 GPU worker 各自加载模型后从队列动态抢占 subject 处理。

    Args:
        tasks: 任务列表
        gpu_ids: 手动指定的GPU ID列表，为None时自动检测可用GPU
    """
    if gpu_ids is not None:
        available_gpus = list(gpu_ids)
        print(f"[多进程] 使用手动指定的GPU: {available_gpus}")
    else:
        available_gpus = get_available_gpus()
        if not available_gpus:
            print("[多进程] 没有可用的GPU（需要至少8GB空闲显存），回退到单线程模式")
            run_tasks_with_single_thread(tasks)
            return
        print(f"[多进程] 自动检测到 {len(available_gpus)} 个可用GPU: {available_gpus}")

    for task_idx, task in enumerate(tasks):
        task_desc = f"{task.get('exp', '?')} / {task.get('dataset', '?')}"
        subjects = discover_subjects(task)
        if not subjects:
            print(f"[多进程] task {task_idx+1}/{len(tasks)} ({task_desc}): 未发现 subject，跳过")
            continue

        num_workers = min(len(available_gpus), len(subjects))
        use_gpus = available_gpus[:num_workers]
        print(f"\n[多进程] task {task_idx+1}/{len(tasks)} ({task_desc}): "
              f"{len(subjects)} 个 subject，使用 {num_workers} 个 GPU {use_gpus}")

        subject_queue = mp.Queue()
        for sid in subjects:
            subject_queue.put(sid)

        # spawn 子进程会先 import 整个模块（含 torch/pytorch_lightning），
        # 这些 import 可能触发 CUDA 初始化。因此必须在 parent 环境中
        # 预设 CUDA_VISIBLE_DEVICES，让子进程在 import 阶段就看到正确的 GPU。
        original_cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        processes = []
        for gpu_id in use_gpus:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
            p = mp.Process(target=gpu_worker, args=(gpu_id, task, subject_queue))
            p.start()
            processes.append(p)
            print(f"[多进程] 启动工作进程 PID={p.pid}，GPU={gpu_id}")
        # 恢复 parent 的 CUDA_VISIBLE_DEVICES
        if original_cvd is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_cvd
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        for p in processes:
            p.join()

        print(f"[多进程] task {task_idx+1}/{len(tasks)} 完成")

    print("\n[多进程] 所有任务已完成")


if __name__ == "__main__":
    # 设置多进程启动方式（对Windows兼容性更好）
    mp.set_start_method('spawn', force=True)
    
    os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    
    # 显示GPU设备信息
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'None')}")
    if torch.cuda.is_available():
        print(f"可用GPU数量: {torch.cuda.device_count()}")
        print(f"当前GPU设备: {torch.cuda.current_device()}")
        print(f"GPU名称: {torch.cuda.get_device_name()}")
    else:
        print("CUDA不可用")
    
    # ==================== CLI ====================
    import argparse
    _ap = argparse.ArgumentParser(
        description="DiffProxy stage-1: generate dense uv/seg proxies from multi-view images (SD2.1).")
    _ap.add_argument("--base_data_dir", required=True,
                     help="dir that contains <dataset>/gt/image_ori/ (outputs go to <dataset>/ours_<Nv>v/<exp>/)")
    _ap.add_argument("--dataset", required=True, help="dataset name, e.g. data_rich_4v_interval2")
    _ap.add_argument("--exp", default="sd21-ep30",
                     help="experiment name; also selects configs/view-guidance/<exp_base>.yaml")
    _ap.add_argument("--exp_suffix", default="",
                     help="optional suffix appended to the output exp dir (e.g. 'cfg3' -> <exp>-cfg3)")
    _ap.add_argument("--config", default=None, help="config yaml (default configs/view-guidance/<exp>.yaml)")
    _ap.add_argument("--proxy_vae", default="pretrained/proxy_vae",
                     help="merged proxy VAE directory reapplied after checkpoint restore")
    _ap.add_argument("--tts_list", default="1,3",
                     help="comma-separated test-time-sampling K list (e.g. '3'); K samples share the seed cache")
    _ap.add_argument("--cfg", type=float, default=None, help="override eval guidance scale (default: from yaml)")
    _ap.add_argument("--steps", type=int, default=None, help="override num inference steps (default: from yaml)")
    _ap.add_argument("--body_views", type=int, default=4)
    _ap.add_argument("--max_subjects", type=int, default=None, help="truncate subject list (debug)")
    _ap.add_argument("--subjects", default=None,
                     help="comma-separated subject IDs in explicit processing order (e.g. 0000,0003)")
    _ap.add_argument("--multi_gpu", default=None,
                     help="comma-separated CUDA ids for the built-in multi-process mode (e.g. '0,1'); "
                          "default single-process (set CUDA_DEVICE_ORDER=PCI_BUS_ID and CUDA_VISIBLE_DEVICES outside)")
    _args = _ap.parse_args()
    if _args.subjects and _args.max_subjects is not None:
        _ap.error("--subjects and --max_subjects are mutually exclusive")
    _subject_ids = parse_subject_ids(_args.subjects)

    # the deep inference code reads these via os.environ (kept for behavior parity)
    if _args.cfg is not None:
        os.environ["CFG_VALUE"] = str(_args.cfg)
    if _args.steps is not None:
        os.environ["NUM_INFERENCE_STEPS"] = str(_args.steps)
    if _args.max_subjects is not None:
        os.environ["MAX_SUBJECTS"] = str(_args.max_subjects)

    EXP_NAME = _args.exp + (f"-{_args.exp_suffix}" if _args.exp_suffix else "")
    _config_path = _args.config or f"configs/view-guidance/{_args.exp}.yaml"
    TTS_LIST = [int(x.strip()) for x in _args.tts_list.split(",") if x.strip()]
    print(f"[DATASET] {_args.dataset}  [EXP] {EXP_NAME}  [CONFIG] {_config_path}  [TTS_LIST] {TTS_LIST}")

    tasks = [
        {
            "exp": EXP_NAME,
            "config_path": _config_path,
            "dataset": _args.dataset,
            "proxy_vae_path": _args.proxy_vae,
            "enable_uncertainty": True,
            # one model load handles the whole TTS_LIST; K values share the seed cache
            "num_uncertainty_samples_list": TTS_LIST,
            "body_views": _args.body_views,
            "body_num_uncertainty_samples": 1,
            "inference_both_hands_once": True,
            "sample_seed_offsets": [0],
            "base_data_dir": _args.base_data_dir,
            "subjects": _subject_ids or None,
        }
    ]

    if _args.multi_gpu:
        _gpu_ids = [int(x) for x in _args.multi_gpu.split(",") if x.strip()]
        print(f"[模式] 多进程多GPU并行模式 GPUs={_gpu_ids}")
        run_tasks_with_multiprocessing(tasks, gpu_ids=_gpu_ids)
    else:
        print("[模式] 单进程模式（CUDA_DEVICE_ORDER=PCI_BUS_ID；用 CUDA_VISIBLE_DEVICES 选卡）")
        run_tasks_with_single_thread(tasks)
