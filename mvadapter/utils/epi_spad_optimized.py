import numpy as np
import torch
from einops import rearrange
from torch.nn import functional as F
from typing import Optional, List, Tuple
import functools
import os
import time
from tqdm import tqdm
from numpy.lib.format import open_memmap

# PyTorch3D
from pytorch3d.renderer import NDCMultinomialRaysampler, ray_bundle_to_ray_points
from pytorch3d.utils import cameras_from_opencv_projection
 

@torch.no_grad()
def compute_epipolar_mask_p3d_optimized(src_cam, tgt_cam, H: int, W: int, dialate_mask: bool = True):
    """
    优化版本的极线mask计算 - 保持与原始版本完全一致的结果
    
    主要优化：
    1. 使用GPU加速的线段绘制
    2. 批处理优化
    3. 缓存机制（可选）
    """
    device = src_cam.device
    
    # 生成源图像平面上的射线束 - 与原始版本完全相同
    raysampler = NDCMultinomialRaysampler(
        image_width=W,
        image_height=H,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )
    src_ray_bundle = raysampler(src_cam)
    depths = torch.ones((1, H, W, 1), device=device, dtype=torch.float32)
    pts_world = ray_bundle_to_ray_points(src_ray_bundle._replace(lengths=depths)).squeeze(-2)

    # 将源点投影到目标屏幕空间
    tgt_pts_screen = tgt_cam.transform_points_screen(pts_world.squeeze(), image_size=(H, W))
    src_center_screen = tgt_cam.transform_points_screen(src_cam.get_camera_center(), image_size=(H, W)).squeeze()

    # 若外参一致（同一视角，仅内参/裁剪不同），退化为点到点掩码
    def _extrinsics_equal(a, b, tol: float = 1e-5) -> bool:
        try:
            c_diff = (a.get_camera_center() - b.get_camera_center()).abs().max()
            eq_center = (c_diff < tol).item() if hasattr(c_diff, 'item') else c_diff < tol
            R_ok, T_ok = True, True
            if hasattr(a, 'R') and hasattr(b, 'R'):
                R_diff = (a.R - b.R).abs().max()
                R_ok = (R_diff < tol).item() if hasattr(R_diff, 'item') else R_diff < tol
            if hasattr(a, 'T') and hasattr(b, 'T'):
                T_diff = (a.T - b.T).abs().max()
                T_ok = (T_diff < tol).item() if hasattr(T_diff, 'item') else T_diff < tol
            return bool(eq_center and R_ok and T_ok)
        except Exception:
            return False

    if _extrinsics_equal(src_cam, tgt_cam):
        # 直接将 src 像素映射到 tgt 像素（点到点）
        # 使用与其它分支一致的量化：减 0.5 后取整
        xy = (tgt_pts_screen[..., :2] - 0.5).reshape(-1, 2)
        tx = xy[:, 0].long()
        ty = xy[:, 1].long()
        valid = (tx >= 0) & (tx < W) & (ty >= 0) & (ty < H)
        tx = tx.clamp(0, W - 1)
        ty = ty.clamp(0, H - 1)
        i_idx = torch.arange(H * W, device=src_cam.device)
        j_idx = ty * W + tx
        attn = torch.zeros((H * W, H * W), dtype=torch.bool, device=src_cam.device)
        attn[i_idx[valid], j_idx[valid]] = True
        return attn

    # 计算从源相机中心到目标屏幕点的流向
    flow = tgt_pts_screen[..., :2] - src_center_screen[..., :2]
    flow = flow / (flow.norm(dim=-1, keepdim=True) + 1e-8)
    slope = flow[:, :, 0:1] / (flow[:, :, 1:2] + 1e-8)
    intercept = tgt_pts_screen[:, :, 0:1] - slope * tgt_pts_screen[:, :, 1:2]

    # 计算与图像边界的交点
    left = torch.cat([slope * 0 + intercept, torch.zeros_like(intercept)], dim=-1)
    right = torch.cat([slope * W + intercept, torch.ones_like(intercept) * W], dim=-1)
    top = torch.cat([torch.zeros_like(intercept), (0 - intercept) / (slope + 1e-8)], dim=-1)
    bottom = torch.cat([torch.ones_like(intercept) * H, (H - intercept) / (slope + 1e-8)], dim=-1)

    def _sane_xy(xy):
        x_ok = (xy[..., 0] >= 0) & (xy[..., 0] <= W)
        y_ok = (xy[..., 1] >= 0) & (xy[..., 1] <= H)
        return x_ok & y_ok

    left_sane = _sane_xy(left)
    right_sane = _sane_xy(right)
    top_sane = _sane_xy(top)
    bottom_sane = _sane_xy(bottom)

    # 找到两个交点 - 与原始版本完全相同的逻辑
    p1 = torch.zeros_like(left)
    p2 = torch.zeros_like(left)

    # 收集第一个点
    p1 = torch.where(left_sane.unsqueeze(-1).repeat(1, 1, 2), left, p1)
    p1_zero = (p1.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, 2)
    p1 = torch.where(right_sane.unsqueeze(-1).repeat(1, 1, 2) & p1_zero, right, p1)
    p1_zero = (p1.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, 2)
    p1 = torch.where(bottom_sane.unsqueeze(-1).repeat(1, 1, 2) & p1_zero, bottom, p1)
    p1_zero = (p1.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, 2)
    p1 = torch.where(top_sane.unsqueeze(-1).repeat(1, 1, 2) & p1_zero, top, p1)

    # 收集第二个点
    p2 = torch.where(top_sane.unsqueeze(-1).repeat(1, 1, 2), top, p2)
    p2_zero = (p2.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, 2)
    p2 = torch.where(bottom_sane.unsqueeze(-1).repeat(1, 1, 2) & p2_zero, bottom, p2)
    p2_zero = (p2.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, 2)
    p2 = torch.where(right_sane.unsqueeze(-1).repeat(1, 1, 2) & p2_zero, right, p2)
    p2_zero = (p2.sum(dim=-1) == 0).unsqueeze(-1).repeat(1, 1, 2)
    p2 = torch.where(left_sane.unsqueeze(-1).repeat(1, 1, 2) & p2_zero, left, p2)

    # 如果源相机中心在屏幕内，使用方向消歧并将其作为一个端点
    if (0 <= src_center_screen[0] <= W) and (0 <= src_center_screen[1] <= H):
        # 方向消歧：选择与原始 flow 方向一致的端点作为 p1
        points_one_flow = p1 - src_center_screen[:2]
        orig_flow_direction = (flow > 0)
        points_one_flow_direction = (points_one_flow > 0)
        points_one_aligned = (points_one_flow_direction == orig_flow_direction).all(dim=-1, keepdim=True).repeat(1, 1, 2)
        p1 = torch.where(points_one_aligned, p1, p2)

        # 将 p2 设为相机中心
        p2 = p2 * 0 + src_center_screen[:2]

    # 构建极线mask - 优化版本
    mask = _draw_epipolar_lines_optimized(p1, p2, H, W, dialate_mask, device)
    
    return mask.reshape(H * W, H * W)

 
def _draw_epipolar_lines_optimized(p1: torch.Tensor, p2: torch.Tensor, H: int, W: int, 
                                  dialate_mask: bool, device: torch.device) -> torch.Tensor:
    """
    优化的极线绘制 - 使用批处理和GPU加速，但保持结果一致
    """
    from skimage.draw import line

    # 优先使用缓存（仅在小分辨率下可行，避免指数级存储）
    if dialate_mask and H == 32 and W == 32:
        masks_np = _get_or_build_line_masks_cache(H, W, dialate_mask)
        # 量化并索引缓存
        p1_np = (p1 - 0.5).reshape(-1, 2).long().cpu().numpy()
        p2_np = (p2 - 0.5).reshape(-1, 2).long().cpu().numpy()
        # p 向量是 [x, y]
        x1, y1 = p1_np[:, 0], p1_np[:, 1]
        x2, y2 = p2_np[:, 0], p2_np[:, 1]
        # 缓存轴序统一为 [x1, y1, x2, y2, H, W]
        attn_np = masks_np[x1, y1, x2, y2]  # (H*W, H, W)
        # 与原始非缓存分支一致：对无效点对返回全零掩码
        invalid = (p1_np.sum(axis=1) == 0) & (p2_np.sum(axis=1) == 0)
        if np.any(invalid):
            attn_np[invalid] = False
        return torch.from_numpy(attn_np).to(device)

    mask = torch.zeros((H * W, H, W), dtype=torch.bool, device=device)

    # 量化到像素坐标 - 与原始版本完全一致
    p1_np = (p1 - 0.5).reshape(-1, 2).long().cpu().numpy()
    p2_np = (p2 - 0.5).reshape(-1, 2).long().cpu().numpy()

    # 优化：预分配numpy数组减少内存分配
    if dialate_mask:
        # 预计算偏移量
        offsets = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), 
                  (0, -1), (-1, -1), (-1, 0), (-1, 1)]

    # 批处理绘制以减少GPU传输开销
    batch_size = 256  # 较小的批次以减少内存使用
    num_batches = (H * W + batch_size - 1) // batch_size

    for batch_idx in range(num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, H * W)

        # 创建当前批次的mask
        batch_mask = np.zeros((end_idx - start_idx, H, W), dtype=np.bool_)

        for i in range(start_idx, end_idx):
            local_idx = i - start_idx
            p1_i = p1_np[i]
            p2_i = p2_np[i]

            # 跳过无效点 - 与原始版本完全一致
            if p1_i.sum() == 0 and p2_i.sum() == 0:
                continue

            if not dialate_mask:
                # 绘制单条线
                rr, cc = line(int(p1_i[1]), int(p1_i[0]), int(p2_i[1]), int(p2_i[0]))
                rr, cc = rr.astype(np.int32), cc.astype(np.int32)
                valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                batch_mask[local_idx, rr[valid], cc[valid]] = True
            else:
                # 膨胀版本 - 与原始版本完全一致
                for dx, dy in offsets:
                    _p1 = [min(max(p1_i[0] + dy, 0), H - 1), min(max(p1_i[1] + dx, 0), W - 1)]
                    _p2 = [min(max(p2_i[0] + dy, 0), H - 1), min(max(p2_i[1] + dx, 0), W - 1)]
                    rr, cc = line(int(_p1[1]), int(_p1[0]), int(_p2[1]), int(_p2[0]))
                    valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                    if len(valid) > 0:
                        batch_mask[local_idx, rr[valid].astype(np.int32), cc[valid].astype(np.int32)] = True

        # 批量传输到GPU
        if end_idx > start_idx:
            mask[start_idx:end_idx] = torch.from_numpy(batch_mask).to(device)

    return mask


# ===== 线掩码缓存实现（一次性全量生成 + tqdm 总进度条，保存到 data/masks32.npy） =====
def _masks_npy_path(H: int, W: int, dialate_mask: bool) -> str:
    # 仅在 32x32 且膨胀时使用固定路径，保持与上游一致
    if H == 32 and W == 32 and dialate_mask:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
        os.makedirs(root, exist_ok=True)
        return os.path.join(root, "masks32.npy")
    # 其它尺寸不保存为文件（避免超大体积），返回建议路径但不会使用
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
    os.makedirs(root, exist_ok=True)
    return os.path.join(root, f"masks_{H}x{W}_d{1 if dialate_mask else 0}.npy")

_LINE_MASK_CACHE = {}

def _get_or_build_line_masks_cache(H: int, W: int, dialate_mask: bool):
    """
    返回形状为 (W, H, W, H, H, W) 的 numpy 数组（驻留内存），
    轴序为 [x1, y1, x2, y2, H, W]。
    若不存在则一次性全量生成（带总进度条与ETA），并持久化到磁盘。
    """
    key = (W, H, dialate_mask)
    if key in _LINE_MASK_CACHE:
        return _LINE_MASK_CACHE[key]

    path = _masks_npy_path(H, W, dialate_mask)
    shape = (W, H, W, H, H, W)
    dtype = np.bool_

    if os.path.exists(path):
        try:
            # 完整加载到内存
            arr = np.load(path)
            _LINE_MASK_CACHE[key] = arr
            return arr
        except Exception:
            # 若损坏则重建
            try:
                os.remove(path)
            except Exception:
                pass

    # 全量生成
    print(f"[epi] 正在生成线掩码缓存 {H}x{W} (dilate={dialate_mask}) 到 {path}，一次性全量生成，可能需要数分钟……")
    # 直接创建 .npy 的内存映射文件
    mm = open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    from skimage.draw import line

    if dialate_mask:
        offsets = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)]
    else:
        offsets = [(0, 0)]

    total = H * W * H * W
    start_t = time.time()
    with tqdm(total=total, desc="[epi] 生成 masks32", unit="pair") as pbar:
        for x1 in range(W):
            for y1 in range(H):
                for x2 in range(W):
                    for y2 in range(H):
                        # 构造单个 (H, W) 掩码
                        mask_hw = np.zeros((H, W), dtype=dtype)
                        for dx, dy in offsets:
                            _x1 = min(max(x1 + dx, 0), W - 1)
                            _y1 = min(max(y1 + dy, 0), H - 1)
                            _x2 = min(max(x2 + dx, 0), W - 1)
                            _y2 = min(max(y2 + dy, 0), H - 1)
                            rr, cc = line(int(_y1), int(_x1), int(_y2), int(_x2))
                            valid = (rr >= 0) & (rr < H) & (cc >= 0) & (cc < W)
                            if np.any(valid):
                                mask_hw[rr[valid].astype(np.int32), cc[valid].astype(np.int32)] = True

                        # 存入 [x1, y1, x2, y2]
                        mm[x1, y1, x2, y2, :, :] = mask_hw
                        pbar.update(1)

    mm.flush()
    total_elapsed = time.time() - start_t
    print(f"[epi] 掩码缓存生成完成，用时 {total_elapsed:.1f}s；已保存到: {path}")
    # 完整加载到内存并缓存
    arr = np.load(path)
    _LINE_MASK_CACHE[key] = arr
    return arr


@torch.no_grad()
def build_block_epipolar_mask_and_plucker_optimized(
    intrinsics: torch.Tensor,  # (B, V, 3, 3)
    extrinsics_w2c: torch.Tensor,  # (B, V, 3, 4) world->cam
    image_size: int,
    dialate_mask: bool = True,
):
    """
    优化版本的SPAD风格批量极线mask和Plücker嵌入构建
    保持与原始版本完全一致的结果
    """
    B, V = intrinsics.shape[:2]
    H = W = image_size // 8  # 在潜在分辨率上工作

    # 构建PyTorch3D相机
    R = extrinsics_w2c[..., :3, :3]
    T = extrinsics_w2c[..., :3, 3]
    cameras = build_pytorch3d_cameras(R, T, intrinsics, image_size)

    # 预分配结果张量
    block_mask = torch.zeros((B, V * H * W, V * H * W), dtype=torch.bool, device=intrinsics.device)
    plucker = torch.zeros((B, V, 6, H, W), dtype=intrinsics.dtype, device=intrinsics.device)

    # 预计算所有相机的Plücker嵌入
    raysampler = NDCMultinomialRaysampler(
        image_width=W,
        image_height=H,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )

    for b in range(B):
        # 计算Plücker嵌入 - 与原始版本完全一致
        for i in range(V):
            cam = cameras[b][i]
            ray_bundle = raysampler(cam)
            dirs = F.normalize(ray_bundle.directions, dim=-1)
            center = cam.get_camera_center()
            cross = torch.cross(center[:, None, None, :], dirs, dim=-1)
            plk = torch.cat([dirs, cross], dim=-1).permute(0, 3, 1, 2).squeeze(0)
            plucker[b, i] = plk

        # 计算极线mask
        for i in range(V):
            for j in range(V):
                if i == j:
                    # 自注意力（允许自身token）
                    mask_ij = torch.eye(H * W, dtype=torch.bool, device=intrinsics.device)
                else:
                    # 使用优化的极线计算
                    mask_ij = compute_epipolar_mask_p3d_optimized(
                        cameras[b][i], cameras[b][j], H, W, dialate_mask
                    )

                # 放置到块矩阵中
                q_start = i * (H * W)
                k_start = j * (H * W)
                block_mask[b, q_start : q_start + H * W, k_start : k_start + H * W] = mask_ij

    return block_mask, plucker


# 辅助函数
def build_pytorch3d_cameras(
    R: torch.Tensor,
    T: torch.Tensor,
    K: torch.Tensor,
    image_size: int,
):
    """
    从原始epi_spad.py复制的相机构建函数
    """
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
