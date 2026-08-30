import torch
from einops import rearrange, repeat
import torch.nn.functional as F
import copy
from torch import nn

class HarmonicEmbedding(torch.nn.Module):
    def __init__(self, n_harmonic_functions=6, append_input=True):
        super().__init__()
        self.register_buffer(
            "frequencies",
            2.0
            * torch.pi
            * torch.arange(n_harmonic_functions),
        )
        self.append_input = append_input

    @staticmethod
    def get_output_dim_static(input_dim, n_harmonic_functions, append_input):
        return (n_harmonic_functions * 2 + int(append_input)) * input_dim

    def get_output_dim(self, input_dim):
        return self.get_output_dim_static(
            input_dim, len(self.frequencies), self.append_input
        )

    def forward(self, x):
        embed = (x[..., None] * self.frequencies.to(x.device)).view(*x.shape[:-1], -1)
        if self.append_input:
            return torch.cat((x, embed.sin(), embed.cos()), dim=-1)
        return torch.cat((embed.sin(), embed.cos()), dim=-1)

def compute_rel_transform(c2w, ray_origins, ray_directions):
    upv = c2w[:, :3, 1]
    upv = rearrange(upv, "bf i -> bf 1 i")
    rdotup = (ray_directions * upv).sum(-1, keepdims=True)
    orthoup = upv - rdotup * ray_directions
    orthoup = F.normalize(orthoup, dim=-1)
    vec0 = torch.cross(orthoup, ray_directions)
    vec0 = F.normalize(vec0, dim=-1)
    r_relatives = torch.stack((vec0, orthoup, ray_directions), dim=-1)
    r_relatives_T = r_relatives.transpose(-1, -2)
    translation_relative = torch.einsum("... i j, ... j -> ... i", -r_relatives_T, ray_origins).unsqueeze(-1)
    canonical_w2c = torch.cat((r_relatives_T, translation_relative), dim=-1)
    return canonical_w2c

def ray_sample_2(cam2world, K, H, W):
    device, dtype = cam2world.device, cam2world.dtype
    N = cam2world.shape[0]
    # 1. 像素网格（像素单位）
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    xs = xs + 0.5  # 像素中心
    ys = ys + 0.5

    # 展平成 (H*W)
    pixels = torch.stack((xs, ys, torch.ones_like(xs)), dim=-1)   # H×W×3
    pixels = pixels.view(-1, 3).T                                 # 3×(HW)

    # 2. 相机→世界
    Kinv = torch.inverse(K)                                       # N×3×3
    rays_o = cam2world[:, :3, 3]                                  # N×3

    # 批量求方向
    dirs = (Kinv @ pixels).permute(0, 2, 1)                       # N×(HW)×3
    dirs = torch.cat([dirs, torch.zeros_like(dirs[..., :1])], -1) # 齐次
    dirs = (cam2world @ dirs.permute(0, 2, 1)).permute(0, 2, 1)[..., :3]
    dirs = torch.nn.functional.normalize(dirs - rays_o[:, None, :], dim=-1)

    rays_o = rays_o[:, None, :].expand_as(dirs)
    return rays_o, dirs


def ray_sample(cam2world_matrix, intrinsics, resolution, dtype=torch.float32):
    """
    Create batches of rays and return origins and directions.

    cam2world_matrix: (N, 4, 4)
    intrinsics: (N, 3, 3)
    resolution: int

    ray_origins: (N, M, 3)
    ray_dirs: (N, M, 3)
    """

    N, M = cam2world_matrix.shape[0], resolution**2
    cam_locs_world = cam2world_matrix[:, :3, 3]
    fx = intrinsics[:, 0, 0]
    fy = intrinsics[:, 1, 1]
    cx = intrinsics[:, 0, 2]
    cy = intrinsics[:, 1, 2]
    sk = intrinsics[:, 0, 1]
    uv = torch.stack(
        torch.meshgrid(
            torch.arange(
                resolution, dtype=dtype, device=cam2world_matrix.device
            ),
            torch.arange(
                resolution, dtype=dtype, device=cam2world_matrix.device
            ),
            indexing="ij",
        )
    ) * (1.0 / resolution) + (0.5 / resolution)
    uv = repeat(uv, "c h w -> b (h w) c", b=N)
    x_cam = uv[:, :, 0].view(N, -1)
    y_cam = uv[:, :, 1].view(N, -1)
    z_cam = torch.ones((N, M), device=cam2world_matrix.device, dtype=dtype)

    x_lift = (
        (
            x_cam
            - cx.unsqueeze(-1)
            + cy.unsqueeze(-1) * sk.unsqueeze(-1) / fy.unsqueeze(-1)
            - sk.unsqueeze(-1) * y_cam / fy.unsqueeze(-1)
        )
        / fx.unsqueeze(-1)
        * z_cam
    )
    y_lift = (y_cam - cy.unsqueeze(-1)) / fy.unsqueeze(-1) * z_cam

    cam_rel_points = torch.stack(
        (x_lift, y_lift, z_cam, torch.ones_like(z_cam)), dim=-1
    )

    # world_rel_points = torch.bmm(
    #     cam2world_matrix.double(), cam_rel_points.permute(0, 2, 1).double()
    # ).permute(0, 2, 1)[:, :, :3].float()
    world_rel_points = torch.bmm(
        cam2world_matrix, cam_rel_points.permute(0, 2, 1)
    ).permute(0, 2, 1)[:, :, :3]

    ray_dirs = world_rel_points - cam_locs_world[:, None, :]
    ray_dirs = torch.nn.functional.normalize(ray_dirs, dim=2)

    ray_origins = cam_locs_world.unsqueeze(1).repeat(1, ray_dirs.shape[1], 1)
    # ray_o,ray_d = ray_sample_2(cam2world_matrix,intrinsics,resolution,resolution)
    return ray_origins, ray_dirs

def sample_stratified(ray_origins, ray_start, ray_end, depth_resolution, disparity_space_sampling=False, return_depth=False, dtype=torch.float32):
    """
    Stratified sampling along rays, supporting per-ray near/far as tensors.
    ray_origins: (N, M, 3)
    ray_start, ray_end: float or tensor (N,)
    returns depths_coarse: (N, M, depth_resolution, 1)
    """
    N, M, _ = ray_origins.shape
    device = ray_origins.device
    # prepare near/far tensors of shape (N,)
    ray_start = ray_start.to(device=device, dtype=dtype).reshape(N)
    ray_end = ray_end.to(device=device, dtype=dtype).reshape(N)
    # reshape to (N,1,1,1)
    ray_start = ray_start.view(N, 1, 1, 1)
    ray_end = ray_end.view(N, 1, 1, 1)
    # sampling weights
    t = torch.linspace(0.0, 1.0, depth_resolution, device=device, dtype=ray_start.dtype)
    t = t.view(1, 1, depth_resolution, 1)
    # compute depths
    if disparity_space_sampling:
        d = t
        depths_coarse = 1.0 / (1.0 / ray_start * (1 - d) + 1.0 / ray_end * d)
    else:
        depths_coarse = ray_start * (1 - t) + ray_end * t
    # broadcast to (N, M, depth_resolution, 1)
    depths_coarse = depths_coarse.expand(N, M, depth_resolution, 1)
    return depths_coarse

def get_ray_limits_box(rays_o: torch.Tensor, rays_d: torch.Tensor, box_side_length):
    o_shape = rays_o.shape
    rays_o = rays_o.detach().reshape(-1, 3)
    rays_d = rays_d.detach().reshape(-1, 3)
    bb_min = [-1 * (box_side_length / 2), -1 * (box_side_length / 2), -1 * (box_side_length / 2)]
    bb_max = [1 * (box_side_length / 2), 1 * (box_side_length / 2), 1 * (box_side_length / 2)]
    bounds = torch.tensor([bb_min, bb_max], dtype=rays_o.dtype, device=rays_o.device)
    is_valid = torch.ones(rays_o.shape[:-1], dtype=bool, device=rays_o.device)
    invdir = 1 / rays_d
    sign = (invdir < 0).long()
    tmin = (bounds.index_select(0, sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tmax = (bounds.index_select(0, 1 - sign[..., 0])[..., 0] - rays_o[..., 0]) * invdir[..., 0]
    tymin = (bounds.index_select(0, sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    tymax = (bounds.index_select(0, 1 - sign[..., 1])[..., 1] - rays_o[..., 1]) * invdir[..., 1]
    is_valid[torch.logical_or(tmin > tymax, tymin > tmax)] = False
    tmin = torch.max(tmin, tymin)
    tmax = torch.min(tmax, tymax)
    tzmin = (bounds.index_select(0, sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    tzmax = (bounds.index_select(0, 1 - sign[..., 2])[..., 2] - rays_o[..., 2]) * invdir[..., 2]
    is_valid[torch.logical_or(tmin > tzmax, tzmin > tmax)] = False
    tmin = torch.max(tmin, tzmin)
    tmax = torch.min(tmax, tzmax)
    is_valid[tmax < 0] = False
    tmin[is_valid == 0] = -1
    tmax[is_valid == 0] = -1
    tmin[tmin < 0] = 0
    return tmin.reshape(*o_shape[:-1], 1), tmax.reshape(*o_shape[:-1], 1)

def sample_points(ray_origins, ray_directions, rendering_options, return_depth=False, dtype=torch.float32):
    depths_coarse = sample_stratified(ray_origins, rendering_options["ray_start"], rendering_options["ray_end"], rendering_options["n_samples"], rendering_options["disparity_space_sampling"], return_depth=return_depth, dtype=dtype)
    
    batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape
    sample_coordinates = ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)
    sample_directions = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1)
    return sample_coordinates, sample_directions, depths_coarse

def compute_projections(points, k_nearest_cameras):
    b, f, k, _ = k_nearest_cameras.shape
    intrinsics = k_nearest_cameras[..., :16].reshape(b, f, k, 4, 4)
    c2w = k_nearest_cameras[..., 16:].reshape(b, f, k, 4, 4)
    points_h = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    points_h = repeat(points_h, "b f n s c -> b f k n s c", k=k)
    # w2p = torch.matmul(intrinsics.double(), torch.inverse(c2w.double()))
    # projections = torch.einsum("b f k i j, b f k n s j -> b f k n s i", w2p, points_h.double()).float()
    w2p = torch.matmul(intrinsics, torch.inverse(c2w))
    projections = torch.einsum("b f k i j, b f k n s j -> b f k n s i", w2p, points_h)
    pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)
    pixel_locations = torch.clamp(pixel_locations, min=-10, max=10)
    mask = projections[..., 2] > 0
    inbound = (pixel_locations[..., 0] <= 1.0) & (pixel_locations[..., 0] >= 0) & \
              (pixel_locations[..., 1] <= 1.0) & (pixel_locations[..., 1] >= 0)
    mask = mask & inbound
    return pixel_locations, mask

def encode_plucker(ray_origins, ray_dirs, harmonic_embedding=None):
    plucker = torch.cat((ray_dirs, torch.cross(ray_origins, ray_dirs, dim=-1)), dim=-1)
    if harmonic_embedding is not None:
        plucker = harmonic_embedding(plucker)
    return plucker

def process_cameras(cameras, select_ids=None, harmonic_embedding=None, rendering_options=None, dtype=torch.float32):
    _rendering_options = copy.deepcopy(rendering_options)
    resolution = _rendering_options["resolution"]
    n_samples = _rendering_options["n_samples"]

    cameras = cameras.to(dtype=dtype)

    b, f, c = cameras.shape
    if select_ids is None:
        select_ids = torch.arange(f, device=cameras.device).repeat(b, f, 1)
    k_views = select_ids.shape[2]

    full_cameras = repeat(cameras, "b k c -> b f k c", f=f) # 4 near view --> 4 near view * 4 view
    b_ids = torch.arange(b)[:, None, None].expand(-1, f, k_views).to(cameras.device)
    f_ids = torch.arange(f)[None, :, None].expand(b, -1, k_views).to(cameras.device)
    k_nearest_cameras = full_cameras[b_ids, f_ids, select_ids]# [b,nv,n_near,c_cam]

    intrinsics, c2w = cameras[:, :, :16], cameras[:, :, 16:]
    c2w = rearrange(c2w, "b f (i j) -> (b f) i j", i=4, j=4)
    intrinsics = rearrange(intrinsics, "b f (i j) -> (b f) i j", i=4, j=4)[:, :3, :3]
    # ray_origins: [nv,res*res,axis]
    ray_origins, ray_directions = ray_sample(c2w, intrinsics, resolution, dtype=dtype)
    # sample_coordinates: [bs,nv,res*res,n_sample,c]
    sample_coordinates, sample_directions, sample_depths = sample_points(ray_origins, ray_directions, _rendering_options, return_depth=True, dtype=dtype)
    sample_coordinates = rearrange(sample_coordinates, "(b f) n s c -> b f n s c", f=f, n=resolution**2)
    sample_depths = rearrange(sample_depths, "(b f) n s c -> b f n s c", f=f, n=resolution**2)
    sample_depths = (sample_depths - _rendering_options["ray_start"]) / (_rendering_options["ray_end"] - _rendering_options["ray_start"])
    # projected_points: [bs,nv,n_near,res*res,n_sample,c]
    projected_points, mask = compute_projections(sample_coordinates, k_nearest_cameras)

    query_plucker = encode_plucker(ray_origins, ray_directions, harmonic_embedding)
    query_depth = harmonic_embedding(sample_depths)
    query_plucker = repeat(query_plucker, "(b f) n d -> b f k n s d", f=f, k=k_views, s=n_samples)
    query_depth = repeat(query_depth, "b f n s d -> b f k n s d", k=k_views)
    query_plucker = torch.cat((query_plucker, query_depth), dim=-1)

    k_nearest_c2w = rearrange(k_nearest_cameras[..., 16:], "b f k (i j) -> b f k i j", i=4, j=4)
    origins_cam = repeat(k_nearest_c2w[..., :3, 3], "b f k i -> b f k n s i", n=resolution**2, s=n_samples)
    sample_coordinates_rep = repeat(sample_coordinates, "b f n s c -> b f k n s c", k=k_views)
    input_dirs = F.normalize(sample_coordinates_rep - origins_cam, dim=-1)

    canonical_w2c = compute_rel_transform(c2w, ray_origins, ray_directions)
    canonical_w2c = repeat(canonical_w2c, "(b f) n i j -> b f k n s i j", f=f, s=n_samples, k=k_views)
    origins_cam_h = torch.cat((origins_cam, torch.ones_like(origins_cam[..., :1])), dim=-1)
    canonical_dirs = torch.einsum("... i j, ... j-> ... i", canonical_w2c[..., :3, :3], input_dirs)
    canonical_cam = torch.einsum("... i j, ... j -> ... i", canonical_w2c, origins_cam_h)

    reference_plucker = encode_plucker(canonical_cam, canonical_dirs, harmonic_embedding)
    reference_depth = harmonic_embedding(sample_depths)
    full_depth = repeat(reference_depth, "b k ... -> b f k ...", f=f)
    k_nearest_depth = full_depth[b_ids, f_ids, select_ids]
    reference_plucker = torch.cat((reference_plucker, k_nearest_depth), dim=-1)
    
    projected_points = projected_points * 2 - 1
    return projected_points, mask, query_plucker, reference_plucker, sample_depths, sample_coordinates

def get_epi_projection(cameras: torch.Tensor,  k_near_indices: torch.Tensor, rendering_opts: dict, num_tasks: int = 1, dtype=torch.float32):
    """Compute per-pixel epipolar projections."""
    proj, mask, *_, sample_coords = process_cameras(
        cameras,
        select_ids=k_near_indices,
        harmonic_embedding=HarmonicEmbedding(),
        rendering_options=rendering_opts,
        dtype=dtype,
    )
    # Replicate for num_tasks
    if num_tasks > 1:
        # proj: (b, v, k, n, s, 2) -> (b, t, v, k, n, s, 2)
        proj = proj.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1, 1)
        # mask: (b, v, k, n, s) -> (b, t, v, k, n, s)
        mask = mask.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1)
        # sample_coords: (b, v, n, s, 3) -> (b, t, v, n, s, 3)
        sample_coords = sample_coords.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1)

    return proj, mask, sample_coords

def _process_cameras_and_analyze_error(cameras, select_ids, harmonic_embedding, rendering_options, dtype, precision_str):
    """Helper function to run process_cameras and print intermediate values for debugging."""
    print(f"\n--- Running Calculations with {precision_str} ---")
    
    _rendering_options = copy.deepcopy(rendering_options)
    resolution = _rendering_options["resolution"]
    n_samples = _rendering_options["n_samples"]

    cameras = cameras.to(dtype=dtype)
    b, f, c = cameras.shape
    if select_ids is None:
        select_ids = torch.arange(f, device=cameras.device).repeat(b, f, 1)
    k_views = select_ids.shape[2]

    full_cameras = repeat(cameras, "b k c -> b f k c", f=f)
    b_ids = torch.arange(b)[:, None, None].expand(-1, f, k_views).to(cameras.device)
    f_ids = torch.arange(f)[None, :, None].expand(b, -1, k_views).to(cameras.device)
    k_nearest_cameras = full_cameras[b_ids, f_ids, select_ids]

    intrinsics, c2w = cameras[:, :, :16], cameras[:, :, 16:]
    c2w = rearrange(c2w, "b f (i j) -> (b f) i j", i=4, j=4)
    intrinsics = rearrange(intrinsics, "b f (i j) -> (b f) i j", i=4, j=4)[:, :3, :3]
    
    ray_origins, ray_directions = ray_sample(c2w, intrinsics, resolution, dtype=dtype)
    
    sample_coordinates, sample_directions, sample_depths = sample_points(ray_origins, ray_directions, _rendering_options, return_depth=True, dtype=dtype)
    sample_coordinates = rearrange(sample_coordinates, "(b f) n s c -> b f n s c", f=f, n=resolution**2)
    sample_depths_raw = sample_depths
    sample_depths = rearrange(sample_depths, "(b f) n s c -> b f n s c", f=f, n=resolution**2)
    sample_depths = (sample_depths - _rendering_options["ray_start"]) / (_rendering_options["ray_end"] - _rendering_options["ray_start"])

    projected_points, mask = compute_projections(sample_coordinates, k_nearest_cameras)

    return {
        "ray_origins": ray_origins, "ray_directions": ray_directions,
        "sample_coordinates": sample_coordinates, "sample_depths_raw": sample_depths_raw,
        "projected_points": projected_points, "mask": mask
    }

def get_epi_projection_with_error_analysis(cameras: torch.Tensor, k_near_indices: torch.Tensor, rendering_opts: dict, num_tasks: int = 1):
    """
    Computes per-pixel epipolar projections and analyzes the precision error at each step
    by comparing float32 and float64 calculations.
    """
    print("--- Starting EPI projection with error analysis ---")
    
    harmonic_embedding = HarmonicEmbedding()

    # --- Run calculations for float32 and float64 ---
    results_float = _process_cameras_and_analyze_error(
        cameras, k_near_indices, harmonic_embedding, rendering_opts, torch.float32, "float32"
    )
    results_double = _process_cameras_and_analyze_error(
        cameras, k_near_indices, harmonic_embedding, rendering_opts, torch.float64, "float64"
    )

    # --- Compare results and print errors ---
    print("\n--- STAGE-WISE ERROR ANALYSIS (L2 Norm of Difference) ---")
    
    # Error after ray sampling
    error_ray_origins = torch.norm(results_float["ray_origins"].double() - results_double["ray_origins"])
    error_ray_directions = torch.norm(results_float["ray_directions"].double() - results_double["ray_directions"])
    print(f"Error from ray_sample (origins): {error_ray_origins}")
    print(f"Error from ray_sample (directions): {error_ray_directions}")

    # Error after point sampling
    error_sample_coords = torch.norm(results_float["sample_coordinates"].double() - results_double["sample_coordinates"])
    error_sample_depths = torch.norm(results_float["sample_depths_raw"].double() - results_double["sample_depths_raw"])
    print(f"Error from sample_points (coordinates): {error_sample_coords}")
    print(f"Error from sample_points (depths): {error_sample_depths}")

    # Error after projection
    error_proj_points = torch.norm(results_float["projected_points"].double() - results_double["projected_points"])
    print(f"Error from compute_projections (pixel_locations): {error_proj_points}")
    
    proj = results_float["projected_points"]
    mask = results_float["mask"]
    sample_coords = results_float["sample_coordinates"]

    # Replicate for num_tasks if needed
    if num_tasks > 1:
        proj = proj.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1, 1)
        mask = mask.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1)
        sample_coords = sample_coords.unsqueeze(1).repeat(1, num_tasks, 1, 1, 1, 1)

    print("\n--- Error analysis complete. Returning float32 results. ---")
    return proj, mask, sample_coords