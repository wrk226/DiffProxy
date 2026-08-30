import numpy as np
import torch
import pickle
import os
from typing import Dict, Tuple, Optional, Any, List
from einops import rearrange
from torch.nn import functional as F

# PyTorch3D
from pytorch3d.renderer import NDCMultinomialRaysampler, ray_bundle_to_ray_points
from pytorch3d.utils import cameras_from_opencv_projection

# 缓存配置
_DEFAULT_CACHE_DIR = None  # 将在运行时设置为数据目录的cache文件夹

def set_cache_dir(cache_dir: str):
    """
    设置缓存目录
    """
    global _DEFAULT_CACHE_DIR
    _DEFAULT_CACHE_DIR = cache_dir
    os.makedirs(cache_dir, exist_ok=True)

def _generate_epipolar_cache_key(
    src_cam_file: str,
    tgt_cam_file: str,
    image_size: int,
    dialate_mask: bool
) -> str:
    """
    基于文件名生成极线mask的缓存key
    """
    # 提取文件名（去掉路径，保留扩展名用于唯一性）
    src_name = os.path.basename(src_cam_file)
    tgt_name = os.path.basename(tgt_cam_file)
    
    
    key_parts = [
        src_name,
        tgt_name,
        str(image_size),
        str(dialate_mask)
    ]
    return "_".join(key_parts)
 
def _load_from_disk_cache(cache_key: str, cache_dir: str) -> Optional[torch.Tensor]:
    """
    从磁盘缓存加载数据
    """
    cache_file = os.path.join(cache_dir, f"epipolar_{cache_key}.pkl")
    if os.path.exists(cache_file):
        try:
            # print("load cache")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                return torch.from_numpy(data['tensor']).bool()
        except Exception as e:
            print(f"警告：加载缓存文件失败 {cache_file}: {e}，正在删除损坏的缓存文件")
            os.remove(cache_file)
    return None

def _save_to_disk_cache(cache_key: str, tensor: torch.Tensor, cache_dir: str):
    """
    保存数据到磁盘缓存
    """
    cache_file = os.path.join(cache_dir, f"epipolar_{cache_key}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        data = {
            'tensor': tensor.cpu().numpy().astype(bool),  # 只存储bool类型以节省空间
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"警告：保存缓存文件失败 {cache_file}: {e}")

@torch.no_grad()
def compute_epipolar_mask_p3d_cached(src_cam, tgt_cam, H: int, W: int, 
                                   dialate_mask: bool = True,
                                   use_cache: bool = True,
                                   src_cam_file: str = None,
                                   tgt_cam_file: str = None,
                                   cache_dir: str = None) -> torch.Tensor:
    """
    带缓存的极线mask计算，基于文件名的硬盘缓存
    
    Args:
        src_cam, tgt_cam: PyTorch3D相机对象
        H, W: 图像尺寸
        dialate_mask: 是否膨胀mask
        use_cache: 是否使用缓存
        src_cam_file, tgt_cam_file: 相机文件名（用作缓存key）
        cache_dir: 缓存目录
    """
    if True or not use_cache or src_cam_file is None or tgt_cam_file is None:
        # 如果不使用缓存或缺少文件名，直接计算
        from .epi_spad_optimized import compute_epipolar_mask_p3d_optimized
        return compute_epipolar_mask_p3d_optimized(src_cam, tgt_cam, H, W, dialate_mask)
    
    # 设置缓存目录
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    if cache_dir is None:
        # 如果没有设置缓存目录，直接计算
        from .epi_spad_optimized import compute_epipolar_mask_p3d_optimized
        return compute_epipolar_mask_p3d_optimized(src_cam, tgt_cam, H, W, dialate_mask)
    
    # 生成缓存key
    cache_key = _generate_epipolar_cache_key(src_cam_file, tgt_cam_file, H, dialate_mask)
    
    # 检查磁盘缓存
    cached_result = _load_from_disk_cache(cache_key, cache_dir)
    if cached_result is not None:
        return cached_result.to(src_cam.device)
    
    # 计算新结果
    from .epi_spad_optimized import compute_epipolar_mask_p3d_optimized
    result = compute_epipolar_mask_p3d_optimized(src_cam, tgt_cam, H, W, dialate_mask)
    
    # 只保存到磁盘缓存
    _save_to_disk_cache(cache_key, result, cache_dir)
    
    return result

@torch.no_grad()
def compute_plucker_no_cache(cam, H: int, W: int) -> torch.Tensor:
    """
    直接计算Plucker嵌入（不使用缓存）
    """
    raysampler = NDCMultinomialRaysampler(
        image_width=W, image_height=H, n_pts_per_ray=1,
        min_depth=1.0, max_depth=1.0,
    )
    ray_bundle = raysampler(cam)
    dirs = F.normalize(ray_bundle.directions, dim=-1)
    center = cam.get_camera_center()
    cross = torch.cross(center[:, None, None, :], dirs, dim=-1)
    return torch.cat([dirs, cross], dim=-1).permute(0, 3, 1, 2).squeeze(0)

@torch.no_grad()
def build_pytorch3d_cameras(
    R: torch.Tensor,
    T: torch.Tensor, 
    K: torch.Tensor,
    image_size: int,
):
    """从原始epi_spad.py复制的相机构建函数"""
    B, V = R.shape[:2]
    device = R.device
    size = torch.tensor([image_size, image_size], device=device).float().unsqueeze(0)

    cameras = []
    for b in range(B):
        cams_b = []
        for v in range(V):
            cam = cameras_from_opencv_projection(
                R[b, v:v + 1],
                T[b, v:v + 1], 
                K[b, v:v + 1],
                size,
            )
            cams_b.append(cam)
        cameras.append(cams_b)
    return cameras

@torch.no_grad()
def build_block_epipolar_mask_and_plucker_cached(
    intrinsics: torch.Tensor,  # (B, V, 3, 3)
    extrinsics_w2c: torch.Tensor,  # (B, V, 3, 4) world->cam
    image_size: int,
    dialate_mask: bool = True,
    use_cache: bool = True,
    cam_files: List[List[str]] = None,  # (B, V) 每个batch每个view的相机文件名
    cache_dir: str = None,  # 缓存目录
        scale: int = 8,
    is_child_views: Optional[torch.Tensor] = None,  # (B, V) bool，True表示子视角
    parent_indices: Optional[torch.Tensor] = None,  # (B, V) long，子视角对应父主视角索引，主视角为-1
):
    """
    带缓存的SPAD风格批量极线mask和Plucker嵌入构建
    
    Args:
        intrinsics: 内参矩阵
        extrinsics_w2c: 外参矩阵
        image_size: 图像尺寸
        dialate_mask: 是否膨胀mask
        use_cache: 是否使用缓存机制
        cam_files: 相机文件名列表，用作缓存key
        cache_dir: 缓存目录
    """
    B, V = intrinsics.shape[:2]
    H = W = image_size // scale  # 在潜在分辨率上工作

    # 构建PyTorch3D相机
    R = extrinsics_w2c[..., :3, :3]
    T = extrinsics_w2c[..., :3, 3]
    cameras = build_pytorch3d_cameras(R, T, intrinsics, image_size)

    # 预分配结果张量
    block_mask = torch.zeros((B, V * H * W, V * H * W), dtype=torch.bool, device=intrinsics.device)
    plucker = torch.zeros((B, V, 6, H, W), dtype=intrinsics.dtype, device=intrinsics.device)

    for b in range(B):
        # 计算Plücker嵌入（不使用缓存）
        for i in range(V):
            cam = cameras[b][i]
            plucker[b, i] = compute_plucker_no_cache(cam, H, W)

        # 计算极线mask（使用基于文件名的缓存）
        for i in range(V):
            for j in range(V):
                if i == j:
                    # 自注意力（允许自身token）
                    mask_ij = torch.eye(H * W, dtype=torch.bool, device=intrinsics.device)
                else:
                    # 基于主/子视角关系的条件构建
                    allow_pair = True
                    if is_child_views is not None and parent_indices is not None:
                        is_i_child = bool(is_child_views[b, i].item())
                        is_j_child = bool(is_child_views[b, j].item())
                        if not is_i_child and not is_j_child:
                            # 主-主：允许
                            allow_pair = True
                        elif is_i_child:
                            # 子-?：仅允许 子->父
                            parent_j = int(parent_indices[b, i].item())
                            allow_pair = (j == parent_j and parent_j >= 0)
                        else:
                            # 主->子：不允许（节省显存）
                            allow_pair = False

                    if allow_pair:
                        # 获取文件名
                        src_cam_file = cam_files[b][i] if cam_files is not None else None
                        tgt_cam_file = cam_files[b][j] if cam_files is not None else None
                        # 使用缓存的极线mask
                        mask_ij = compute_epipolar_mask_p3d_cached(
                            cameras[b][i], cameras[b][j], H, W, dialate_mask, use_cache,
                            src_cam_file, tgt_cam_file, cache_dir
                        )
                    else:
                        mask_ij = torch.zeros(H * W, H * W, dtype=torch.bool, device=intrinsics.device)

                # 放置到块矩阵中
                q_start = i * (H * W)
                k_start = j * (H * W)
                block_mask[b, q_start : q_start + H * W, k_start : k_start + H * W] = mask_ij

    return block_mask, plucker

def clear_disk_cache(cache_dir: str = None):
    """清空磁盘缓存"""
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    if cache_dir and os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.startswith('epipolar_') and file.endswith('.pkl'):
                try:
                    os.remove(os.path.join(cache_dir, file))
                except Exception as e:
                    print(f"警告：删除缓存文件失败 {file}: {e}")

def get_cache_stats(cache_dir: str = None):
    """获取缓存统计信息"""
    if cache_dir is None:
        cache_dir = _DEFAULT_CACHE_DIR
    
    if not cache_dir or not os.path.exists(cache_dir):
        return {
            'cache_dir': cache_dir,
            'disk_cache_files': 0
        }
    
    cache_files = [f for f in os.listdir(cache_dir) if f.startswith('epipolar_') and f.endswith('.pkl')]
    return {
        'cache_dir': cache_dir,
        'disk_cache_files': len(cache_files)
    }
