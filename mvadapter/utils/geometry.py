from typing import List, Optional, Tuple, Dict

import numpy as np
import torch
from torch.nn import functional as F
from pytorch3d.renderer import (
    NDCMultinomialRaysampler,
    ray_bundle_to_ray_points,
)
from pytorch3d.utils import cameras_from_opencv_projection


def get_position_map_from_depth(depth, mask, intrinsics, extrinsics, image_wh=None):
    """Compute the position map from the depth map and the camera parameters for a batch of views.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        intrinsics (torch.Tensor): The camera intrinsics matrices with the shape (B, 3, 3).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        image_wh (Tuple[int, int]): The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    u_coord, v_coord = torch.meshgrid(
        torch.arange(image_wh[0]), torch.arange(image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map by back-projecting depth pixels to 3D space
    x = (
        (u_coord - intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    )
    y = (
        (v_coord - intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1))
        * depth
        / intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    )
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_position_map_from_depth_ortho(
    depth, mask, extrinsics, ortho_scale, image_wh=None
):
    """Compute the position map from the depth map and the camera parameters for a batch of views
    using orthographic projection with a given ortho_scale.

    Args:
        depth (torch.Tensor): The depth maps with the shape (B, H, W, 1).
        mask (torch.Tensor): The masks with the shape (B, H, W, 1).
        extrinsics (torch.Tensor): The camera extrinsics matrices with the shape (B, 4, 4).
        ortho_scale (torch.Tensor): The scaling factor for the orthographic projection with the shape (B, 1, 1, 1).
        image_wh (Tuple[int, int]): Optional. The image width and height.

    Returns:
        torch.Tensor: The position maps with the shape (B, H, W, 3).
    """
    if image_wh is None:
        image_wh = depth.shape[2], depth.shape[1]

    B, H, W, _ = depth.shape
    depth = depth.squeeze(-1)

    # Generating grid of coordinates in the image space
    u_coord, v_coord = torch.meshgrid(
        torch.arange(0, image_wh[0]), torch.arange(0, image_wh[1]), indexing="xy"
    )
    u_coord = u_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)
    v_coord = v_coord.type_as(depth).unsqueeze(0).expand(B, -1, -1)

    # Compute the position map using orthographic projection with ortho_scale
    x = (u_coord - image_wh[0] / 2) * ortho_scale / image_wh[0]
    y = (v_coord - image_wh[1] / 2) * ortho_scale / image_wh[1]
    z = depth

    # Concatenate to form the 3D coordinates in the camera frame
    camera_coords = torch.stack([x, y, z], dim=-1)

    # Apply the extrinsic matrix to get coordinates in the world frame
    coords_homogeneous = torch.nn.functional.pad(
        camera_coords, (0, 1), "constant", 1.0
    )  # Add a homogeneous coordinate
    world_coords = torch.matmul(
        coords_homogeneous.view(B, -1, 4), extrinsics.transpose(1, 2)
    ).view(B, H, W, 4)

    # Apply the mask to the position map
    position_map = world_coords[..., :3] * mask

    return position_map


def get_opencv_from_blender(matrix_world, fov=None, image_size=None):
    # convert matrix_world to opencv format extrinsics
    opencv_world_to_cam = matrix_world.inverse()
    opencv_world_to_cam[1, :] *= -1
    opencv_world_to_cam[2, :] *= -1
    R, T = opencv_world_to_cam[:3, :3], opencv_world_to_cam[:3, 3]

    if fov is None:  # orthographic camera
        return R, T

    R, T = R.unsqueeze(0), T.unsqueeze(0)
    # convert fov to opencv format intrinsics
    focal = 1 / np.tan(fov / 2)
    intrinsics = np.diag(np.array([focal, focal, 1])).astype(np.float32)
    opencv_cam_matrix = (
        torch.from_numpy(intrinsics).unsqueeze(0).float().to(matrix_world.device)
    )
    opencv_cam_matrix[:, :2, -1] += torch.tensor([image_size / 2, image_size / 2]).to(
        matrix_world.device
    )
    opencv_cam_matrix[:, [0, 1], [0, 1]] *= image_size / 2

    return R, T, opencv_cam_matrix


def get_ray_directions(
    H: int,
    W: int,
    focal: float,
    principal: Optional[Tuple[float, float]] = None,
    use_pixel_centers: bool = True,
) -> torch.Tensor:
    """
    Get ray directions for all pixels in camera coordinate.
    Args:
        H, W, focal, principal, use_pixel_centers: image height, width, focal length, principal point and whether use pixel centers
    Outputs:
        directions: (H, W, 3), the direction of the rays in camera coordinate
    """
    pixel_center = 0.5 if use_pixel_centers else 0
    cx, cy = W / 2, H / 2 if principal is None else principal
    i, j = torch.meshgrid(
        torch.arange(W, dtype=torch.float32) + pixel_center,
        torch.arange(H, dtype=torch.float32) + pixel_center,
        indexing="xy",
    )
    directions = torch.stack(
        [(i - cx) / focal, -(j - cy) / focal, -torch.ones_like(i)], -1
    )
    return F.normalize(directions, dim=-1)


def get_rays(
    directions: torch.Tensor, c2w: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Get ray origins and directions from camera coordinates to world coordinates
    Args:
        directions: (H, W, 3) ray directions in camera coordinates
        c2w: (4, 4) camera-to-world transformation matrix
    Outputs:
        rays_o, rays_d: (H, W, 3) ray origins and directions in world coordinates
    """
    # Rotate ray directions from camera coordinate to the world coordinate
    rays_d = directions @ c2w[:3, :3].T
    rays_o = c2w[:3, 3].expand(rays_d.shape)
    return rays_o, rays_d


def compute_plucker_embed(
    c2w: torch.Tensor, image_width: int, image_height: int, focal: float
) -> torch.Tensor:
    """
    Computes Plucker coordinates for a camera.
    Args:
        c2w: (4, 4) camera-to-world transformation matrix
        image_width: Image width
        image_height: Image height
        focal: Focal length of the camera
    Returns:
        plucker: (6, H, W) Plucker embedding
    """
    directions = get_ray_directions(image_height, image_width, focal)
    rays_o, rays_d = get_rays(directions, c2w)
    # Cross product to get Plucker coordinates
    cross = torch.linalg.cross(rays_o, rays_d, dim=-1)
    plucker = torch.cat((rays_d, cross), dim=-1)
    return plucker.permute(2, 0, 1)


def get_plucker_embeds_from_cameras(
    c2w: List[torch.Tensor], fov: List[float], image_size: int) -> torch.Tensor:
    """
    Given lists of camera transformations and fov, returns the batched plucker embeddings.
    Args:
        c2w: list of camera-to-world transformation matrices
        fov: list of field of view values
        image_size: size of the image
    Returns:
        plucker_embeds: (B, 6, H, W) batched plucker embeddings
    """
    plucker_embeds = []
    for cam_matrix, cam_fov in zip(c2w, fov):
        focal = 0.5 * image_size / np.tan(0.5 * cam_fov)
        plucker = compute_plucker_embed(cam_matrix, image_size, image_size, focal)
        plucker_embeds.append(plucker)
    return torch.stack(plucker_embeds)


def get_plucker_embeds_from_cameras_ortho(
    c2w: List[torch.Tensor], ortho_scale: List[float], image_size: int, rot_only=False
):
    """
    Given lists of camera transformations and fov, returns the batched plucker embeddings.

    Parameters:
        c2w: list of camera-to-world transformation matrices
        fov: list of field of view values
        image_size: size of the image

    Returns:
        plucker_embeds: plucker embeddings (B, 6, H, W)
    """
    plucker_embeds = []
    # compute pairwise mask and plucker embeddings
    for cam_matrix, scale in zip(c2w, ortho_scale):
        # blender to opencv to pytorch3d
        R, T = get_opencv_from_blender(cam_matrix)
        view_dir = R.T @ torch.tensor([0, 0, 1]).float().to(cam_matrix.device)
        # normalize camera position
        if not rot_only:
            cam_pos = -R.T @ T
            cam_pos = F.normalize(cam_pos, dim=0)
            plucker = torch.concat([view_dir, cam_pos])
        else:
            plucker = view_dir
        plucker = plucker.unsqueeze(-1).unsqueeze(-1).repeat(1, image_size, image_size)
        plucker_embeds.append(plucker)

    plucker_embeds = torch.stack(plucker_embeds)

    return plucker_embeds


def sample_along_rays(
    origins: torch.Tensor, 
    directions: torch.Tensor, 
    n_samples: int, 
    near: float, 
    far: float,
    perturb: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample points along rays for NeRF-style rendering.
    
    Args:
        origins: Ray origins with shape (*, 3)
        directions: Ray directions with shape (*, 3) 
        n_samples: Number of samples along each ray
        near: Near plane distance (float or tensor with shape (*,))
        far: Far plane distance  (float or tensor with shape (*,))
        perturb: Whether to add noise to sample positions
        
    Returns:
        points: Sampled points with shape (*, n_samples, 3)
        z_vals: Sample distances along rays with shape (*, n_samples)
    """
    device = origins.device
    ray_shape = origins.shape[:-1]  # e.g. (B, S, H, W)

    # Ensure near/far are tensors broadcastable to ray_shape
    if not torch.is_tensor(near):
        near = torch.tensor(near, device=device, dtype=origins.dtype)
    if not torch.is_tensor(far):
        far = torch.tensor(far, device=device, dtype=origins.dtype)

    # If near/far are scalar tensors, expand to ray_shape
    if near.dim() == 0:
        near = near.expand(ray_shape)
    if far.dim() == 0:
        far = far.expand(ray_shape)

    # 对 near 和 far 进行阈值截断（clamp 到 [0.5, 3.0] 区间）
    near = near.clamp(min=0., max=9.9)
    far = far.clamp(min=0.1, max=10.0)

    # Sampling interpolation factors
    t_vals = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=origins.dtype)
    # Reshape for broadcasting: (1, 1, ..., n_samples)
    for _ in range(len(ray_shape)):
        t_vals = t_vals.unsqueeze(0)

    # Compute z values: (*, n_samples)
    z_vals = near.unsqueeze(-1) * (1.0 - t_vals) + far.unsqueeze(-1) * t_vals

    if perturb:
        # Add stratified sampling noise
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], dim=-1)
        lower = torch.cat([z_vals[..., :1], mids], dim=-1)
        t_rand = torch.rand(z_vals.shape, device=device, dtype=origins.dtype)
        z_vals = lower + (upper - lower) * t_rand

    # Sample points along rays
    points = origins.unsqueeze(-2) + directions.unsqueeze(-2) * z_vals.unsqueeze(-1)

    return points, z_vals


def project_points_to_cameras(
    points: torch.Tensor,
    c2w: torch.Tensor,
    K: torch.Tensor,
    image_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用 PyTorch3D 将 3D 点投影到 2D 像素坐标。

    Args:
        points: (B, S, H, W, n_samples, 3)
        c2w: (B, S, 3, 4) Camera-to-world
        K:   (B, S, 3, 3)
        image_size: 若提供，则输出为屏幕像素坐标；否则输出为归一化屏幕坐标。

    Returns:
        pixel_coords: (B, S, H, W, n_samples, 2)
        depths: (B, S, H, W, n_samples)
    """
    B, S, H, W, n_samples, _ = points.shape
    device = points.device

    # 构建 PyTorch3D 相机（OpenCV 投影），逐视角
    R_list, T_list, K_list, size_list = [], [], [], []
    for b in range(B):
        R_b, T_b, K_b = [], [], []
        for s in range(S):
            c2w_ = c2w[b, s]
            # to 4x4 and invert to w2c
            w2c_4x4 = torch.eye(4, device=device, dtype=c2w_.dtype)
            w2c_4x4[:3, :4] = torch.linalg.inv(
                torch.vstack([c2w_, torch.tensor([0, 0, 0, 1], device=device, dtype=c2w_.dtype)])
            )[:3, :4]
            R = w2c_4x4[:3, :3].unsqueeze(0)
            T = w2c_4x4[:3, 3].unsqueeze(0)
            R_b.append(R)
            T_b.append(T)
            K_b.append(K[b, s].unsqueeze(0))
        R_list.append(torch.cat(R_b, dim=0))
        T_list.append(torch.cat(T_b, dim=0))
        K_list.append(torch.cat(K_b, dim=0))
        size_list.append(torch.tensor([image_size, image_size], device=device).float().unsqueeze(0) if image_size is not None else None)

    # 逐 batch 构建摄像机并投影
    pixel_coords_list, depths_list = [], []
    for b in range(B):
        cams = cameras_from_opencv_projection(
            R_list[b], T_list[b], K_list[b],
            size_list[b] if image_size is not None else torch.tensor([1, 1], device=device).float().unsqueeze(0)
        )
        pts = points[b].reshape(S, H * W * n_samples, 3)
        if image_size is not None:
            screen = cams.transform_points_screen(pts, image_size=(image_size, image_size))
            pix = screen[..., :2]
            depth = screen[..., 2]
        else:
            ndc = cams.transform_points(pts)
            pix = ndc[..., :2]
            depth = ndc[..., 2]
        pixel_coords_list.append(pix.view(S, H, W, n_samples, 2))
        depths_list.append(depth.view(S, H, W, n_samples))

    pixel_coords = torch.stack(pixel_coords_list, dim=0)
    depths = torch.stack(depths_list, dim=0)
    return pixel_coords, depths


def extract_vggt_predictions(
    vggt_model, 
    images: torch.Tensor, 
    image_size: int = 518
) -> dict[str, torch.Tensor]:
    """
    Extract camera parameters and point cloud predictions from VGGT model.
    
    Args:
        vggt_model: VGGT model instance
        images: Input images with shape (B, S, 3, H, W) in range [0, 1]
        image_size: Target image size for VGGT
        
    Returns:
        predictions: Dictionary containing:
            - pose_enc: Camera pose encoding (B, S, 9)
            - world_points: 3D world coordinates (B, S, H, W, 3)
            - world_points_conf: Confidence scores (B, S, H, W)
            - extrinsics: Camera extrinsic matrices (B, S, 3, 4)
            - intrinsics: Camera intrinsic matrices (B, S, 3, 3)
    """
    from dependents.vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
    
    # Forward pass through VGGT
    with torch.no_grad():
        predictions = vggt_model(images)
    
    # Extract pose encoding and convert to camera matrices
    pose_enc = predictions["pose_enc"]  # (B, S, 9)
    world_points = predictions["world_points"]  # (B, S, H, W, 3)
    world_points_conf = predictions["world_points_conf"]  # (B, S, H, W)
    depth = predictions["depth"][...,0]
    depth_conf = predictions["depth_conf"]
    
    # Convert pose encoding to camera matrices
    B, S = pose_enc.shape[:2]
    extrinsics_list = []
    intrinsics_list = []
    
    for b in range(B):
        for s in range(S):
            pose_enc_single = pose_enc[b:b+1, s:s+1]  # (1, 1, 9)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(
                pose_enc_single, 
                image_size_hw=(image_size, image_size),
                pose_encoding_type="absT_quaR_FoV"
            )
            extrinsics_list.append(extrinsics.squeeze(0))  # (3, 4)
            intrinsics_list.append(intrinsics.squeeze(0))   # (3, 3)
    
    # Stack into batch tensors
    extrinsics = torch.stack(extrinsics_list).view(B, S, 3, 4)
    intrinsics = torch.stack(intrinsics_list).view(B, S, 3, 3)
    
    return {
        "pose_enc": pose_enc,
        "world_points": world_points,
        "world_points_conf": world_points_conf,
        "extrinsics": extrinsics,
        "intrinsics": intrinsics,
        "depth": depth,
        "depth_conf": depth_conf,
    }


def generate_rays_from_camera(
    c2w: torch.Tensor,
    K: torch.Tensor,
    image_height: int,
    image_width: int,
    bbox: Optional[Tuple[int, int, int, int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate rays for a single camera.
    
    Args:
        c2w: Camera to world transformation matrix (3, 4) or (4, 4)
        K: Camera intrinsic matrix (3, 3)
        image_height: Image height
        image_width: Image width
        bbox: Optional bounding box (xmin, ymin, xmax, ymax) to restrict ray generation to.
        
    Returns:
        rays_o: Ray origins in world coordinates (H, W, 3)
        rays_d: Ray directions in world coordinates (H, W, 3)
    """
    device = c2w.device
    
    # Generate ray directions for pixels inside bbox (if provided)
    xmin, ymin, xmax, ymax = bbox
    # 将可能的 tensor/float 转为 python int 方便后续 torch.arange
    xmin = int(xmin.item() if torch.is_tensor(xmin) else xmin)
    ymin = int(ymin.item() if torch.is_tensor(ymin) else ymin)
    xmax = int(xmax.item() if torch.is_tensor(xmax) else xmax)
    ymax = int(ymax.item() if torch.is_tensor(ymax) else ymax)

    # Clamp to valid image range
    xmin = max(0, xmin)
    ymin = max(0, ymin)
    xmax = min(image_width, xmax)
    ymax = min(image_height, ymax)

    i_range = torch.arange(xmin, xmax, dtype=torch.float32, device=device)
    j_range = torch.arange(ymin, ymax, dtype=torch.float32, device=device)


    W = i_range.shape[0]
    H = j_range.shape[0]

    i, j = torch.meshgrid(
        i_range,
        j_range,
        indexing='xy'
    )
    
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    # Back-project to camera space (OpenCV convention: x right, y down, z forward)
    dirs_x = (i - cx) / fx
    dirs_y = (j - cy) / fy
    dirs_z = torch.ones_like(i)
    directions_cam = torch.stack([dirs_x, dirs_y, dirs_z], dim=-1)

    # Transform ray directions to world space
    R = c2w[:3, :3]
    directions_world = directions_cam @ R.T
    directions_world = F.normalize(directions_world, dim=-1)
    
    # Get ray origins (camera position in world space)
    origins = c2w[:3, 3].expand(W, H, 3)
    
    return origins, directions_world


def generate_ray_samples_from_cameras(
    c2w: torch.Tensor,
    K: torch.Tensor, 
    image_size: int,
    n_samples: int = 64,
    near: float = 0.1,
    far: float = 3.0,
        pts_debug=None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generate ray samples from camera parameters for NeRF-style rendering.
    
    Args:
        c2w: Camera to world transformation matrices (B, S, 3, 4)
        K: Camera intrinsic matrices (B, S, 3, 3)
        image_size: Image size (assumed square)
        n_samples: Number of samples along each ray
        near: Near plane distance
        far: Far plane distance
        
    Returns:
        points: Sampled 3D points (B, S, H, W, n_samples, 3)
        z_vals: Sample distances (B, S, H, W, n_samples)
        pixel_coords: 2D pixel coordinates (B, S, H, W, n_samples, 2)
    """
    B, S, _, _ = c2w.shape
    device = c2w.device
    
    H = W = image_size

    # 使用 PyTorch3D Raysampler 构建射线
    raysampler = NDCMultinomialRaysampler(
        image_width=W,
        image_height=H,
        n_pts_per_ray=1,
        min_depth=1.0,
        max_depth=1.0,
    )

    # 为每个相机构建 Ray bundle，并提取 rays 原点与方向（世界坐标）
    origins, directions = [], []
    for b in range(B):
        # 构建 PyTorch3D 相机（OpenCV 投影）
        R_b = []
        T_b = []
        K_b = []
        for s in range(S):
            c2w_ = c2w[b, s]
            w2c_4x4 = torch.eye(4, device=device, dtype=c2w_.dtype)
            w2c_4x4[:3, :4] = torch.linalg.inv(
                torch.vstack([c2w_, torch.tensor([0, 0, 0, 1], device=device, dtype=c2w_.dtype)])
            )[:3, :4]
            R_b.append(w2c_4x4[:3, :3].unsqueeze(0))
            T_b.append(w2c_4x4[:3, 3].unsqueeze(0))
            K_b.append(K[b, s].unsqueeze(0))
        cams = cameras_from_opencv_projection(
            torch.cat(R_b, dim=0), torch.cat(T_b, dim=0), torch.cat(K_b, dim=0),
            torch.tensor([H, W], device=device).float().unsqueeze(0)
        )
        ray_bundle = raysampler(cams)
        origins.append(ray_bundle.origins)
        directions.append(F.normalize(ray_bundle.directions, dim=-1))

    origins = torch.stack(origins, dim=0)          # (B, S, H, W, 3)
    directions = torch.stack(directions, dim=0)    # (B, S, H, W, 3)

    # import test_utils
    # test_utils.visualize_optional(pts_r=origins.reshape(-1, 3),
    #                               pts_g=(directions_world*2 + origins).reshape(-1, 3),
    #                               pts_b=pts_debug.reshape(-1, 3))



    # Sample along rays
    points, z_vals = sample_along_rays(origins, directions, n_samples, near, far)
    # 用 PyTorch3D 相机投影回 2D 屏幕
    pixel_coords, _ = project_points_to_cameras(points, c2w, K, image_size=image_size)
    
    return points, z_vals, pixel_coords

