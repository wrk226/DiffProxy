import numpy as np
import json
import matplotlib.pyplot as plt
import torch
from math import sqrt, exp

# from lib.utils import base_utils
import cv2
import trimesh
import os
try:
    import open3d as o3d
except ImportError:  # Optional: only required by interactive visualization helpers.
    o3d = None
# from pytorch3d import _C
import time


def _require_open3d():
    if o3d is None:
        raise RuntimeError(
            "Open3D is required only for interactive visualization; "
            "install the optional open3d dependency or disable ENABLE_VIS."
        )

def save_obj(verts, faces, filename):
    """
    Saves the mesh represented by verts and faces to an .obj file.

    Parameters:
    - verts: Tensor of shape (N, 3), where N is the number of vertices.
    - faces: Tensor of shape (M, 3), where M is the number of faces.
    - filename: The name of the file to save the mesh to.
    """
    faces = torch.tensor(faces)

    with open(filename, 'w') as f:
        for v in verts:
            f.write(f"v {v[0].item()} {v[1].item()} {v[2].item()}\n")

        for face in faces:
            f.write(f"f {face[0].item()+1} {face[1].item()+1} {face[2].item()+1}\n")

# Code modified from function closest_point in Trimesh: https://github.com/mikedh/trimesh/blob/main/trimesh/triangles.py#L544
def _parse_nearest_projection(triangles, points, eps=1e-12):
    time_init = time.time()
    # store the location of the closest point
    result = torch.zeros_like(points, device=points.device)
    remain = torch.ones(len(points), dtype=torch.bool, device=points.device)
    # get the three points of each triangle
    # use the same notation as RTCD to avoid confusion
    a = triangles[:, 0, :]
    b = triangles[:, 1, :]
    c = triangles[:, 2, :]
    # check if P is in vertex region outside A
    ab = b - a
    ac = c - a
    ap = points - a
    # this is a faster equivalent of:
    # diagonal_dot(ab, ap)
    d1 = torch.sum(ab * ap, dim=-1)
    d2 = torch.sum(ac * ap, dim=-1)
    # is the point at A #即角pab, pac都是锐角
    # is_a = torch.logical_and(d1 < eps, d2 < eps)
    is_a = (d1 < eps) & (d2 < eps)
    result[is_a] = a[is_a]#将pab,pac都是锐角的a固定住
    remain[is_a] = False
    # check if P in vertex region outside B
    bp = points - b
    d3 = torch.sum(ab * bp, dim=-1)
    d4 = torch.sum(ac * bp, dim=-1)
    # do the logic check
    is_b = (d3 > -eps) & (d4 <= d3) & remain
    result[is_b] = b[is_b]# 固定b
    remain[is_b] = False
    # check if P in edge region of AB, if so return projection of P onto A
    vc = (d1 * d4) - (d3 * d2)
    is_ab = ((vc < eps) &
             (d1 > -eps) &
             (d3 < eps) & remain)
    v = (d1[is_ab] / (d1[is_ab] - d3[is_ab])).view((-1, 1))
    result[is_ab] = a[is_ab] + (v * ab[is_ab])
    remain[is_ab] = False
    # check if P in vertex region outside C
    cp = points - c
    d5 = torch.sum(ab * cp, dim=-1)
    d6 = torch.sum(ac * cp, dim=-1)
    is_c = (d6 > -eps) & (d5 <= d6) & remain
    result[is_c] = c[is_c]
    remain[is_c] = False
    # check if P in edge region of AC, if so return projection of P onto AC
    vb = (d5 * d2) - (d1 * d6)
    is_ac = (vb < eps) & (d2 > -eps) & (d6 < eps) & remain
    w = (d2[is_ac] / (d2[is_ac] - d6[is_ac])).view((-1, 1))
    result[is_ac] = a[is_ac] + w * ac[is_ac]
    remain[is_ac] = False
    # check if P in edge region of BC, if so return projection of P onto BC
    va = (d3 * d6) - (d5 * d4)
    is_bc = ((va < eps) &
             ((d4 - d3) > - eps) &
             ((d5 - d6) > -eps) & remain)
    d43 = d4[is_bc] - d3[is_bc]
    w = (d43 / (d43 + (d5[is_bc] - d6[is_bc]))).view((-1, 1))
    result[is_bc] = b[is_bc] + w * (c[is_bc] - b[is_bc])
    remain[is_bc] = False
    # any remaining points must be inside face region
    # point is inside face region
    denom = 1.0 / (va[remain] + vb[remain] + vc[remain])
    v = (vb[remain] * denom).reshape((-1, 1))
    w = (vc[remain] * denom).reshape((-1, 1))
    # compute Q through its barycentric coordinates
    result[remain] = a[remain] + (ab[remain] * v) + (ac[remain] * w)
    stats = {
        'is_a': is_a,
        'is_b': is_b,
        'is_c': is_c,
        'is_bc': is_bc,
        'is_ac': is_ac,
        'is_ab': is_ab,
        'remain': remain
    }

    return result, stats


# precise inside/outside computation based on ray-tracing
def _calculate_points_inside_meshes(points, verts, faces_idx):

    verts_trimesh = verts.to('cpu').detach().numpy().copy()
    faces_trimesh = faces_idx.squeeze(0).to('cpu').detach().numpy().copy()
    mesh = trimesh.Trimesh(vertices=verts_trimesh, faces=faces_trimesh, process=False)
    trimesh_intersector = trimesh.ray.ray_pyembree.RayMeshIntersector(mesh)
    points_trimesh = points.to('cpu').detach().numpy().copy()

    contains = trimesh_intersector.contains_points(points_trimesh)  # [n, ] bool
    contains = torch.tensor(contains, device=points.device)
    contains = 1 - 2*contains  # {-1, 1}, -1 for inside, +1 for outside

    return contains


# approximate inside/outside computation using vertex normals
def _calculate_points_inside_meshes_normals(points, nearest, triangles, normals_triangles):
    barycentric = points_to_barycentric(triangles, nearest)
    normal_at_s = barycentric_to_points(normals_triangles, barycentric)
    contains = ((points - nearest) * normal_at_s).sum(1) < 0.0
    contains = 1 - 2*contains
    return contains


def barycentric_to_points(triangles, barycentric):
    return (triangles * barycentric.view((-1, 3, 1))).sum(dim=1)


def points_to_barycentric(triangles, points):

    edge_vectors = triangles[:, 1:] - triangles[:, :1]
    w = points - triangles[:, 0].view((-1, 3))

    dot00 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 0])
    dot01 = diagonal_dot(edge_vectors[:, 0], edge_vectors[:, 1])
    dot02 = diagonal_dot(edge_vectors[:, 0], w)
    dot11 = diagonal_dot(edge_vectors[:, 1], edge_vectors[:, 1])
    dot12 = diagonal_dot(edge_vectors[:, 1], w)

    inverse_denominator = 1.0 / (dot00 * dot11 - dot01 * dot01 + 1e-12)

    barycentric = torch.zeros(len(triangles), 3, device=points.device)
    barycentric[:, 2] = (dot00 * dot12 - dot01 *
                         dot02) * inverse_denominator
    barycentric[:, 1] = (dot11 * dot02 - dot01 *
                         dot12) * inverse_denominator
    barycentric[:, 0] = 1 - barycentric[:, 1] - barycentric[:, 2]

    return barycentric

def diagonal_dot(a, b):
    return torch.matmul(a * b, torch.ones(a.shape[1], device=a.device))



def point_mesh_face_distance(meshes, pcls):

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    # max_tris = meshes.num_faces_per_mesh().max().item()

    point_to_face, idxs = _C.point_face_dist_forward(
            points, points_first_idx, tris, tris_first_idx, max_points,5e-3
    )

    return point_to_face, idxs


def get_bound_corners(bounds):
    min_x, min_y, min_z = bounds[0]
    max_x, max_y, max_z = bounds[1]
    corners_3d = np.array([
        [min_x, min_y, min_z],
        [min_x, min_y, max_z],
        [min_x, max_y, min_z],
        [min_x, max_y, max_z],
        [max_x, min_y, min_z],
        [max_x, min_y, max_z],
        [max_x, max_y, min_z],
        [max_x, max_y, max_z],
    ])
    return corners_3d


def get_bound_2d_mask(bounds, K, pose, H, W):
    corners_3d = get_bound_corners(bounds)
    corners_2d = base_utils.project(corners_3d, K, pose)
    corners_2d = np.round(corners_2d).astype(int)
    mask = np.zeros((H, W), dtype=np.uint8)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 3, 2, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[4, 5, 7, 6, 5]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 1, 5, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[2, 3, 7, 6, 2]]], 1)
    cv2.fillPoly(mask, [corners_2d[[0, 2, 6, 4, 0]]], 1)
    cv2.fillPoly(mask, [corners_2d[[1, 3, 7, 5, 1]]], 1)
    return mask

def get_rays(H, W, K, R, T):
    # calculate the camera origin
    rays_o = -np.dot(R.T, T).ravel()
    # calculate the world coodinates of pixels
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32),
                       indexing='xy')
    xy1 = np.stack([i, j, np.ones_like(i)], axis=2)
    pixel_camera = np.dot(xy1, np.linalg.inv(K).T)
    pixel_world = np.dot(pixel_camera - T.ravel(), R)
    # calculate the ray direction
    rays_d = pixel_world - rays_o[None, None]
    rays_d = rays_d / np.linalg.norm(rays_d, axis=2, keepdims=True)
    rays_o = np.broadcast_to(rays_o, rays_d.shape)
    # todo：是为了避免后面的除零错误，需要修改
    return rays_o+1e-5, rays_d+1e-5
    # return rays_o, rays_d


def get_near_far(bounds, ray_o, ray_d, use_full_patch=False):
    """calculate intersections with 3d bounding box"""
    bounds = bounds + np.array([-0.01, 0.01])[:, None]
    nominator = bounds[None] - ray_o[:, None]
    # calculate the step of intersections at six planes of the 3d bounding box
    d_intersect = (nominator / ray_d[:, None]).reshape(-1, 6)
    # calculate the six interections
    p_intersect = d_intersect[..., None] * ray_d[:, None] + ray_o[:, None]
    # calculate the intersections located at the 3d bounding box
    min_x, min_y, min_z, max_x, max_y, max_z = bounds.ravel()
    eps = 1e-6
    p_mask_at_box = (p_intersect[..., 0] >= (min_x - eps)) * \
                    (p_intersect[..., 0] <= (max_x + eps)) * \
                    (p_intersect[..., 1] >= (min_y - eps)) * \
                    (p_intersect[..., 1] <= (max_y + eps)) * \
                    (p_intersect[..., 2] >= (min_z - eps)) * \
                    (p_intersect[..., 2] <= (max_z + eps))
    # obtain the intersections of rays which intersect exactly twice
    mask_at_box = p_mask_at_box.sum(-1) == 2
    p_intervals = p_intersect[mask_at_box][p_mask_at_box[mask_at_box]].reshape(
        -1, 2, 3)

    # calculate the step of intersections

    ray_o_intersect = ray_o[mask_at_box]
    ray_d_intersect = ray_d[mask_at_box]
    norm_ray = np.linalg.norm(ray_d_intersect, axis=1)
    d0 = np.linalg.norm(p_intervals[:, 0] - ray_o_intersect, axis=1) / norm_ray
    d1 = np.linalg.norm(p_intervals[:, 1] - ray_o_intersect, axis=1) / norm_ray
    if use_full_patch:
        distance = np.concatenate([d0, d1]).reshape(-1)
        dist_min, dist_max = np.min(distance), np.max(distance)
        near = np.ones_like(ray_o[:, 0]) * dist_min
        far = np.ones_like(ray_o[:, 0]) * dist_max
    else:
        near = np.minimum(d0, d1)
        far = np.maximum(d0, d1)
    #      512,   512,  512
    return near, far, mask_at_box

def get_sampling_points(ray_o, ray_d, near, far, perturb=True):
    # calculate the steps for each ray
    t_vals = torch.linspace(0., 1., steps=64, device=near.device)#.to(near)
    z_vals = near[..., None] * (1. - t_vals) + far[..., None] * t_vals

    if perturb:
        # get intervals between samples
        mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat([mids, z_vals[..., -1:]], -1)
        lower = torch.cat([z_vals[..., :1], mids], -1)
        # stratified samples in those intervals
        t_rand = torch.rand(z_vals.shape, device=upper.device)#.to(upper)
        z_vals = lower + (upper - lower) * t_rand

    pts = ray_o[:, :, None] + ray_d[:, :, None] * z_vals[..., None]


    return pts, z_vals

def sample_ray_h36m_bbox(K, R, T, bounds, nrays):
    H, W = 512,512

    res = np.sqrt(nrays).astype(np.int32)
    scale = H//res

    K = K/scale
    K[2, 2] = 1
    ray_o, ray_d = get_rays(res, res, K, R, T)
    ray_o = ray_o.reshape(-1, 3).astype(np.float32)
    ray_d = ray_d.reshape(-1, 3).astype(np.float32)
    near, far, mask_at_box = get_near_far(bounds, ray_o, ray_d)

    ray_o = ray_o[mask_at_box]
    ray_d = ray_d[mask_at_box]
    near = near.astype(np.float32)
    far = far.astype(np.float32)

    return ray_o, ray_d, near, far


def load_obj(obj_path):
        with open(obj_path) as f:
            lines = f.readlines()
        verts = []
        faces = []
        for line in lines:
            if line.startswith('v '):
                verts.append([float(v) for v in line.split()[1:4]])
            elif line.startswith('f '):
                faces.append([int(v.split('/')[0]) - 1 for v in line.split()[1:4]])
        verts = np.array(verts, dtype=np.float32)
        faces = np.array(faces, dtype=np.int32)
        return verts, faces

def load_trans(offset_path):
    with open(offset_path, 'r') as file:
        offset = json.load(file)
    trans = np.array([offset['offset_x'], offset['offset_y'], 0.]).astype(np.float32)
    return trans


def visualize_mesh(verts, faces):
    _require_open3d()
    # verts = np.random.rand(100, 3)
    # faces = np.random.randint(0, 100, size=(200, 3))
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy().squeeze()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy().squeeze()

    # 创建一个TriangleMesh对象
    mesh = o3d.geometry.TriangleMesh()

    # 将verts和faces数据转换为Open3D的Vector3dVector和Vector3iVector，并设置给TriangleMesh对象
    mesh.vertices = o3d.utility.Vector3dVector(verts)
    mesh.triangles = o3d.utility.Vector3iVector(faces)

    # 计算顶点法线
    mesh.compute_vertex_normals()

    # 创建坐标轴对象
    coord_frame = create_coordinate_frame()
    # 创建一个可视化窗口并添加mesh对象
    o3d.visualization.draw_geometries([mesh, coord_frame])


def create_cylinder_between_points(p1, p2, radius, color=[1, 0, 0]):
    _require_open3d()
    # 计算圆柱体的高度和方向
    vec = p2 - p1
    height = max(np.linalg.norm(vec), 1e-5)
    vec = vec / (height+1e-5)


    # 创建圆柱体
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=height)
    cylinder.paint_uniform_color(color)

    # 将圆柱体移动到原点
    cylinder.translate(-cylinder.get_center())

    # 计算旋转矩阵
    up_vector = np.array([0, 0, 1])  # 圆柱体默认方向
    if np.linalg.norm(vec - up_vector) < 1e-6 or np.linalg.norm(vec + up_vector) < 1e-6:
        # 处理向量和Z轴平行的特殊情况
        rotation_matrix = np.eye(3) if np.dot(vec, up_vector) > 0 else np.diag([-1, -1, 1])
    else:
        axis = np.cross(up_vector, vec)
        axis = axis / np.linalg.norm(axis)
        angle = np.arccos(np.dot(up_vector, vec))
        rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(axis * angle)

    # 应用旋转和平移
    cylinder.rotate(rotation_matrix, center=np.array([0, 0, 0]))
    cylinder.translate((p1 + p2) / 2)

    return cylinder

# import numpy as np
import torch
# import matplotlib.pyplot as plt
import math

def visualize_images(images, figsize=(10.2, 15), title=None):
    """
    Visualize a list of images in a grid layout.

    Args:
        images: List of images. Each image can be:
            - 2D array (H x W)
            - 3D array (H x W x C) or (C x H x W)
            - 4D array (B x H x W x C) or (B x C x H x W)
            Format can be numpy array, torch tensor, or list
        figsize: Tuple of (width, height) for the figure
        title: Optional title for the figure
    """
    def preprocess_image(img):
        # Convert to numpy if needed
        if isinstance(img, torch.Tensor):
            if img.requires_grad:
                img = img.detach()
            if img.is_cuda:
                img = img.cpu()
            img = img.numpy()
        elif isinstance(img, list):
            img = np.array(img)

        # Ensure float type and proper range
        if img.dtype != np.uint8:
            img = img.astype(float)
            if img.max() <= 1.0 and img.min() >= 0:
                img = (img * 255).astype(np.uint8)
            else:
                img = np.clip(img, 0, 255).astype(np.uint8)

        # Handle different dimensions
        if img.ndim == 2:  # H x W
            return img

        elif img.ndim == 3:  # Could be H x W x C or C x H x W or B x H x W
            if img.shape[-1] in [1, 3, 4]:  # H x W x C format
                return img
            elif img.shape[0] in [1, 3, 4]:  # C x H x W format
                return np.transpose(img, (1, 2, 0))
            else:  # B x H x W format (grayscale batch)
                return [img[i] for i in range(img.shape[0])]

        elif img.ndim == 4:  # Could be B x H x W x C or B x C x H x W
            if img.shape[-1] in [1, 3, 4]:  # B x H x W x C format
                return [img[i] for i in range(img.shape[0])]
            else:  # B x C x H x W format
                return [np.transpose(img[i], (1, 2, 0)) for i in range(img.shape[0])]

        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Process all images
    processed_images = []
    for img in images:
        result = preprocess_image(img)
        if isinstance(result, list):
            processed_images.extend(result)
        else:
            processed_images.append(result)

    # Calculate grid size
    n_images = len(processed_images)
    grid_size = math.ceil(math.sqrt(n_images))

    # Create figure
    fig = plt.figure(figsize=figsize)
    if title:
        fig.suptitle(title)

    # Plot each image
    for idx, img in enumerate(processed_images):
        ax = fig.add_subplot(grid_size, grid_size, idx + 1)
        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            ax.imshow(img, cmap='gray')
        else:
            ax.imshow(img)
        ax.axis('off')
    plt.subplots_adjust(wspace=0.01, hspace=0.01)
    # plt.tight_layout()
    plt.show(block=True)
    return fig

def visualize_optional(verts=None, faces=None, pts=None, bounds=None, pts_color=None, plot_coord=True,
                       camera_verts1=None, camera_color1=None, camera_verts2=None, camera_color2=None,
                       joints=None, joints_color=None, joints_size=None, joints_line=None, joints_line_size=None, mesh_width=2., both_side=True,
                       pts_r=None,pts_g=None,pts_b=None, ortho=False):
    _require_open3d()
    # verts = np.random.rand(100, 3)
    # faces = np.random.randint(0, 100, size=(200, 3))
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy().squeeze()
    if verts is not None and verts.shape[1] == 2:
        verts = np.concatenate([verts, np.zeros_like(verts[:, :1])], axis=1)
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy().squeeze()
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy().squeeze()
    if isinstance(pts_r, torch.Tensor):
        pts_r = pts_r.detach().cpu().numpy().squeeze()
    if isinstance(pts_g, torch.Tensor):
        pts_g = pts_g.detach().cpu().numpy().squeeze()
    if isinstance(pts_b, torch.Tensor):
        pts_b = pts_b.detach().cpu().numpy().squeeze()
    if pts is not None:
        if pts.ndim == 1:
            pts = np.expand_dims(pts, 0)
        if pts.shape[1] == 2:
            pts = np.concatenate([pts, np.zeros_like(pts[:, :1])], axis=1)
    if pts_r is not None:
        if pts_r.ndim == 1:
            pts_r = np.expand_dims(pts_r, 0)
        if pts_r.shape[1] == 2:
            pts_r = np.concatenate([pts_r, np.zeros_like(pts_r[:, :1])], axis=1)
    if pts_g is not None:
        if pts_g.ndim == 1:
            pts_g = np.expand_dims(pts_g, 0)
        if pts_g.shape[1] == 2:
            pts_g = np.concatenate([pts_g, np.zeros_like(pts_g[:, :1])], axis=1)
    if pts_b is not None:
        if pts_b.ndim == 1:
            pts_b = np.expand_dims(pts_b, 0)
        if pts_b.shape[1] == 2:
            pts_b = np.concatenate([pts_b, np.zeros_like(pts_b[:, :1])], axis=1)
    if isinstance(bounds, torch.Tensor):
        bounds = bounds.detach().cpu().numpy().squeeze()
    if isinstance(pts_color, torch.Tensor):
        pts_color = pts_color.detach().cpu().numpy().squeeze()
    if isinstance(camera_verts1, torch.Tensor):
        camera_verts1 = camera_verts1.detach().cpu().numpy().squeeze()
    if isinstance(camera_color1, torch.Tensor):
        camera_color1 = camera_color1.detach().cpu().numpy().squeeze()
    if isinstance(camera_verts2, torch.Tensor):
        camera_verts2 = camera_verts2.detach().cpu().numpy().squeeze()
    if isinstance(camera_color2, torch.Tensor):
        camera_color2 = camera_color2.detach().cpu().numpy().squeeze()
    if isinstance(joints, torch.Tensor):
        joints = joints.detach().cpu().numpy().squeeze()
    if joints is not None and joints.shape[1] == 2:
        joints = np.concatenate([joints, np.zeros_like(joints[:, :1])], axis=1)
    if isinstance(joints_color, torch.Tensor):
        joints_color = joints_color.detach().cpu().numpy().squeeze()

    visualize_lst = []
    if verts is not None:
        assert faces is not None

        # 创建一个TriangleMesh对象
        mesh = o3d.geometry.TriangleMesh()
        # 将verts和faces数据转换为Open3D的Vector3dVector和Vector3iVector，并设置给TriangleMesh对象
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        # 计算顶点法线
        mesh.compute_vertex_normals()
        visualize_lst.append(mesh)
    if pts is not None:
        # 将NumPy数组转换为Open3D的PointCloud对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        if pts_color is not None:
            pcd.colors = o3d.utility.Vector3dVector(pts_color)
        visualize_lst.append(pcd)
    if pts_r is not None:
        # 将NumPy数组转换为Open3D的PointCloud对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_r)
        # red color
        pts_color = np.zeros_like(pts_r)
        pts_color[:, 0] = 1
        pcd.colors = o3d.utility.Vector3dVector(pts_color)
        visualize_lst.append(pcd)
    if pts_g is not None:
        # 将NumPy数组转换为Open3D的PointCloud对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_g)
        # red color
        pts_color = np.zeros_like(pts_g)
        pts_color[:, 1] = 1
        pcd.colors = o3d.utility.Vector3dVector(pts_color)
        visualize_lst.append(pcd)
    if pts_b is not None:
        # 将NumPy数组转换为Open3D的PointCloud对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_b)
        # red color
        pts_color = np.zeros_like(pts_b)
        pts_color[:, 2] = 1
        pcd.colors = o3d.utility.Vector3dVector(pts_color)
        visualize_lst.append(pcd)
    if bounds is not None:
        bound_vis = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounds[0, :], max_bound=bounds[1, :])
        # 创建LineSet对象，并设置颜色为黑色
        bound_vis.color = (0,0,0)
        visualize_lst.append(bound_vis)
    if plot_coord:
        coord_frame = create_coordinate_frame()
        visualize_lst.append(coord_frame)
    if camera_verts1 is not None:
        # Define the edges (as lines) or faces (as triangles)
        lines = [
            [0, 4], [1, 4], [2, 4], [3, 4],  # Lines from corners to apex
            [0, 1], [1, 3], [3, 2], [2, 0]   # Lines around the base
            , [4, 5]
        ]
        # 创建圆柱体来代替线
        for line in lines:
            p1, p2 = camera_verts1[line[0]], camera_verts1[line[1]]
            visualize_lst.append(create_cylinder_between_points(p1, p2, radius=0.02, color=camera_color1))  # radius 控制圆柱体粗细
    if camera_verts2 is not None:
        # Define the edges (as lines) or faces (as triangles)
        lines = [
            [0, 4], [1, 4], [2, 4], [3, 4],  # Lines from corners to apex
            [0, 1], [1, 3], [3, 2], [2, 0]   # Lines around the base
            , [4, 5]
        ]
        # 创建圆柱体来代替线
        for line in lines:
            p1, p2 = camera_verts2[line[0]], camera_verts2[line[1]]
            visualize_lst.append(create_cylinder_between_points(p1, p2, radius=0.02, color=camera_color2))  # radius 控制圆柱体粗细
    if joints is not None:
        for idx, joint in enumerate(joints):
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=joints_size or 0.05)  # 可以调整radius设置球体的大小
            if joints_color is not None:
                sphere.paint_uniform_color(joints_color[idx])
            sphere.translate(joint)  # 将球体移动到顶点的位置
            visualize_lst.append(sphere)
        if joints_line is not None:
            for line in joints_line:
                p1, p2 = joints[line[0]], joints[line[1]]
                visualize_lst.append(create_cylinder_between_points(p1, p2, radius=joints_line_size or 0.01 , color=[0,0,0]))  # radius 控制圆柱体粗细
    # 创建一个可视化窗口并添加mesh对象
    # o3d.visualization.draw_geometries(visualize_lst)
    # Create a Visualizer window with key callback functionality
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    for geom in visualize_lst:
        vis.add_geometry(geom)

    # 设置绘制选项以显示网格线
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True  # 显示网格线
    if both_side:
        opt.mesh_show_back_face = True  # 渲染网格的背面
    opt.line_width = mesh_width  # 设置网格线宽度

    # Register a key callback for taking a screenshot with a transparent background
    # vis.register_key_callback(ord("K"), capture_screen)  # Press 'K' to capture the screen
    # 获取 ViewControl 对象
    ctr = vis.get_view_control()

    # 如果请求正交视图，但经典 Visualizer 实际不支持正交投影，
    # 通过把视场角 (FOV) 调得足够小来"近似"正交。
    if ortho:
        try:
            target_fov = 5.0  # 目标视场角，单位：度
            cur_fov = ctr.get_field_of_view()
            # change_field_of_view(step) 中 step < 0 用于减小 FOV
            step = -5.0
            while cur_fov > target_fov + 1e-3:
                ctr.change_field_of_view(step)
                cur_fov = ctr.get_field_of_view()
        except AttributeError:
            # 某些版本可能没有 get_field_of_view，可一次性缩小 FOV
            try:
                ctr.change_field_of_view(-90.0)
            except Exception as e:
                print(f'[visualize_optional] pseudo-ortho failed: {e}')

    vis.run()  # This will show the window and start the event loop
    vis.destroy_window()


def create_coordinate_frame(size=1, origin=[0, 0, 0]):
    _require_open3d()
    """创建一个Open3D坐标系对象"""
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=size, origin=origin)
    return coord_frame

def visualize_pointcloud(pts):
    _require_open3d()

    # check if pts is numpy or tensor
    if isinstance(pts, torch.Tensor):
        point_cloud = pts.detach().cpu().numpy().squeeze()
    else:
        point_cloud = pts
    # 将NumPy数组转换为Open3D的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 创建坐标轴对象
    coord_frame = create_coordinate_frame()

    # 将坐标轴对象添加到可视化窗口
    o3d.visualization.draw_geometries([pcd, coord_frame])

def get_bounds(xyz, box_padding=0.05):
    min_xyz = np.min(xyz, axis=0)
    max_xyz = np.max(xyz, axis=0)
    min_xyz -= box_padding
    max_xyz += box_padding
    bounds = np.stack([min_xyz, max_xyz], axis=0)
    bounds = bounds.astype(np.float32)#[2,3]
    # visualize_pointcloud(xyz, bounds)
    return bounds

def visualize_pointcloud_with_bounds(pts, bounds):
    _require_open3d()
    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy().squeeze()
    else:
        pts = pts
    if isinstance(bounds, torch.Tensor):
        bounds = bounds.detach().cpu().numpy().squeeze()
    else:
        bounds = bounds

    point_cloud = pts
    # 将NumPy数组转换为Open3D的PointCloud对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)#.cpu().numpy())

    # 创建坐标轴对象
    coord_frame = create_coordinate_frame()

    # 创建bounding box
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=bounds[0, :], max_bound=bounds[1, :])

    # 创建LineSet对象，并设置颜色为红色
    aabb.color = (0,0,0)

    # 将坐标轴对象添加到可视化窗口
    o3d.visualization.draw_geometries([pcd, coord_frame, aabb])

def capture_screen(vis):
    opt = vis.get_render_option()
    opt.background_color = np.array([0, 0, 0, 0])  # Set the background to be transparent
    image_path = "capture.png"  # Set the path and filename for your screenshot
    vis.capture_screen_image(image_path, do_render=True)
    print(f"Saved screenshot to {image_path}")
    return False  # Return False to keep the window open

# visualize_optional(pts=torch.rand(100,3), pts_color=torch.rand(100,3), plot_coord=False)
# import test_utils
# proj_face_idx = inp['proj_face_idx']
# body_proj_info = inp['body_proj_info']
# spts = inp['spts']
# visiable_faces = inp['visiable_faces']
# front_msk = torch.tensor([i in visiable_faces  for i in proj_face_idx], device='cuda')
# pts_color = torch.zeros(front_msk.shape[0],3,device='cuda')
# pts_color[front_msk] =  0.7
#
# pts_mask = ret['raw'][0,:,-1]>0.1
#
# body_pts = spts[pts_mask]
# body_color = pts_color[pts_mask]
#
# # 设置每个维度上的点的数量
# num_points_y = 128
# num_points_z = 128
# # 生成每个维度的坐标
# y = torch.linspace(-0.9, 0.9, num_points_y)
# z = torch.linspace(-0.1, 1.7, num_points_z)
# # 生成三维网格点
# Y, Z = torch.meshgrid(y, z, indexing='ij')
# X = torch.full_like(Y, -1)
# X2 = torch.full_like(Y, 1)
# # 将网格点组合成点云
# points = torch.stack((X, Y, Z), dim=-1).cuda().reshape(-1,3)
# points2 = torch.stack((X2, Y, Z), dim=-1).cuda().reshape(-1,3)
# # points_color = torch.ones_like(points,device='cuda').reshape(-1,3)
# points_color = torch.rand_like(points)
# points_color2 = torch.rand_like(points)
# vis_pts = torch.cat([body_pts,points,points2],dim=0)
# vis_color = torch.cat([body_color, points_color, points_color2], dim=0)
# camera_verts1 = torch.tensor([[-1., -0.9, -0.1],[-1., -0.9, 1.7], [-1., 0.9, -0.1],[-1., 0.9, 1.7],[-2., 0., 0.8],[0., 0., 0.]],device='cuda')
# camera_verts1[-1] = body_pts[4000]
# camera_color1 = [0.,0.,0.]
# camera_verts2 = torch.tensor([[1., -0.9, -0.1],[1., -0.9, 1.7], [1., 0.9, -0.1],[1., 0.9, 1.7],[2., 0., 0.8],[1., -0.9, -0.1]],device='cuda')
# camera_color2 = [0.7,0.7,0.7]
# # x:+-1    y -0.9 0.9           z-0.1,1.7
# #    +-                +
# # x:red, y:green, z:blue
# test_utils.visualize_optional(pts=vis_pts,pts_color=vis_color, plot_coord=False, camera_verts1=camera_verts1 ,camera_color1=camera_color1, camera_verts2=camera_verts2, camera_color2=camera_color2)

# Global variables for non-blocking visualization
_non_blocking_vis = None
_non_blocking_opt = None

def visualize_non_blocking_updates(
    pts=None,
    pts_r=None,
    verts=None,
    faces=None,
    plot_coord=True,
    mesh_width=2.0, # Corresponds to line_width in RenderOption
    both_side=True,  # Corresponds to mesh_show_back_face
    pts_g=None,
    camera_frustums=None,
    camera_line_radius=0.01,
):
    _require_open3d()
    """
    Updates the non-blocking Open3D visualizer with new geometries.

    Args:
        pts: Predicted points (e.g., torch.Tensor or np.ndarray).
        pts_r: Reference/ground truth points (e.g., torch.Tensor or np.ndarray).
        verts: Mesh vertices (e.g., torch.Tensor or np.ndarray).
        faces: Mesh faces (e.g., torch.Tensor or np.ndarray).
        plot_coord: Whether to plot a coordinate frame.
        mesh_width: Width of lines for wireframe rendering.
        both_side: Whether to render back faces of the mesh.
    """
    global _non_blocking_vis, _non_blocking_opt

    # Convert inputs to NumPy arrays on CPU
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy().squeeze()
    if isinstance(faces, torch.Tensor):
        # Ensure faces are int32 for Open3D
        faces = faces.detach().cpu().numpy().squeeze().astype(np.int32)
    elif faces is not None and not isinstance(faces, np.ndarray):
        faces = np.array(faces).squeeze().astype(np.int32)
    elif faces is not None: # Already numpy array
        faces = faces.squeeze().astype(np.int32)


    if isinstance(pts, torch.Tensor):
        pts = pts.detach().cpu().numpy().squeeze()
    elif pts is not None and not isinstance(pts, np.ndarray):
        pts = np.array(pts).squeeze()
    
    if isinstance(pts_r, torch.Tensor):
        pts_r = pts_r.detach().cpu().numpy().squeeze()
    elif pts_r is not None and not isinstance(pts_r, np.ndarray):
        pts_r = np.array(pts_r).squeeze()

    if isinstance(pts_g, torch.Tensor):
        pts_g = pts_g.detach().cpu().numpy().squeeze()
    elif pts_g is not None and not isinstance(pts_g, np.ndarray):
        pts_g = np.array(pts_g).squeeze()

    if isinstance(camera_frustums, torch.Tensor):
        camera_frustums = camera_frustums.detach().cpu().numpy()
    elif camera_frustums is not None and not isinstance(camera_frustums, np.ndarray):
        camera_frustums = np.array(camera_frustums)
    if camera_frustums is not None and camera_frustums.ndim == 2:
        camera_frustums = camera_frustums[None, ...]


    if _non_blocking_vis is None:
        _non_blocking_vis = o3d.visualization.Visualizer()
        _non_blocking_vis.create_window()
        _non_blocking_opt = _non_blocking_vis.get_render_option()
        _non_blocking_opt.mesh_show_wireframe = True
        _non_blocking_opt.mesh_show_back_face = both_side
        _non_blocking_opt.line_width = mesh_width # Set line width for wireframes/lines
        # _non_blocking_opt.background_color = np.asarray([0.1, 0.1, 0.1]) # Optional: Dark background

    _non_blocking_vis.clear_geometries()
    
    geometries_to_add = []

    if verts is not None and faces is not None:
        if verts.ndim == 2 and verts.shape[0] > 0 and \
           faces.ndim == 2 and faces.shape[0] > 0:
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(verts)
            mesh.triangles = o3d.utility.Vector3iVector(faces)
            mesh.compute_vertex_normals()
            geometries_to_add.append(mesh)
        else:
            print("Warning: Vertices or faces are not valid for mesh creation.")


    if pts is not None and pts.ndim == 2 and pts.shape[0] > 0:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        # Predicted points color: blue
        pts_colors_np = np.zeros_like(pts)
        pts_colors_np[:, 2] = 0.8  # Blue
        pcd.colors = o3d.utility.Vector3dVector(pts_colors_np)
        geometries_to_add.append(pcd)

    if pts_r is not None and pts_r.ndim == 2 and pts_r.shape[0] > 0:
        pcd_r = o3d.geometry.PointCloud()
        pcd_r.points = o3d.utility.Vector3dVector(pts_r)
        # Reference points color: red
        pts_r_colors_np = np.zeros_like(pts_r)
        pts_r_colors_np[:, 0] = 0.8  # Red
        pcd_r.colors = o3d.utility.Vector3dVector(pts_r_colors_np)
        geometries_to_add.append(pcd_r)

    if pts_g is not None and pts_g.ndim == 2 and pts_g.shape[0] > 0:
        pcd_g = o3d.geometry.PointCloud()
        pcd_g.points = o3d.utility.Vector3dVector(pts_g)
        # Green points
        pts_g_colors_np = np.zeros_like(pts_g)
        pts_g_colors_np[:, 1] = 0.8  # Green
        pcd_g.colors = o3d.utility.Vector3dVector(pts_g_colors_np)
        geometries_to_add.append(pcd_g)

    if plot_coord:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0]) # Smaller coord frame
        geometries_to_add.append(coord_frame)

    if camera_frustums is not None and camera_frustums.ndim == 3 and camera_frustums.shape[1:] == (6, 3):
        cam_lines = [[0, 4], [1, 4], [2, 4], [3, 4], [0, 1], [1, 3], [3, 2], [2, 0], [4, 5]]
        for i in range(camera_frustums.shape[0]):
            cam_v = camera_frustums[i]
            color = [1.0, 0.85, 0.0]
            for line in cam_lines:
                p1, p2 = cam_v[line[0]], cam_v[line[1]]
                geometries_to_add.append(
                    create_cylinder_between_points(p1, p2, radius=camera_line_radius, color=color)
                )

    for geom in geometries_to_add:
        _non_blocking_vis.add_geometry(geom, reset_bounding_box=True) # reset_bounding_box=True should be fine after clear

    if not _non_blocking_vis.poll_events():
        # This might happen if the window is closed by the user.
        # We might want to recreate it or handle it. For now, just proceed.
        pass
    _non_blocking_vis.update_renderer()


def visualize_stage_end_interactive(
    verts=None,
    faces=None,
    camera_frustums=None,
    plot_coord=True,
    mesh_width=2.0,
    both_side=True,
    camera_line_radius=0.01,
):
    _require_open3d()
    """
    阻塞式可交互可视化：用于每个 stage 结束后的人工检查。
    """
    if isinstance(verts, torch.Tensor):
        verts = verts.detach().cpu().numpy().squeeze()
    if isinstance(faces, torch.Tensor):
        faces = faces.detach().cpu().numpy().squeeze().astype(np.int32)
    elif faces is not None and not isinstance(faces, np.ndarray):
        faces = np.array(faces).squeeze().astype(np.int32)
    elif faces is not None:
        faces = faces.squeeze().astype(np.int32)

    if isinstance(camera_frustums, torch.Tensor):
        camera_frustums = camera_frustums.detach().cpu().numpy()
    elif camera_frustums is not None and not isinstance(camera_frustums, np.ndarray):
        camera_frustums = np.array(camera_frustums)
    if camera_frustums is not None and camera_frustums.ndim == 2:
        camera_frustums = camera_frustums[None, ...]

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    opt = vis.get_render_option()
    opt.mesh_show_wireframe = True
    opt.mesh_show_back_face = both_side
    opt.line_width = mesh_width

    if verts is not None and faces is not None and verts.ndim == 2 and faces.ndim == 2:
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(verts)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        vis.add_geometry(mesh)

    if plot_coord:
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
        vis.add_geometry(coord_frame)

    if camera_frustums is not None and camera_frustums.ndim == 3 and camera_frustums.shape[1:] == (6, 3):
        cam_lines = [[0, 4], [1, 4], [2, 4], [3, 4], [0, 1], [1, 3], [3, 2], [2, 0], [4, 5]]
        for i in range(camera_frustums.shape[0]):
            cam_v = camera_frustums[i]
            color = [1.0, 0.85, 0.0]
            for line in cam_lines:
                p1, p2 = cam_v[line[0]], cam_v[line[1]]
                vis.add_geometry(create_cylinder_between_points(p1, p2, radius=camera_line_radius, color=color))

    vis.run()
    vis.destroy_window()

def close_non_blocking_visualizer():
    """Closes the non-blocking Open3D visualizer window."""
    global _non_blocking_vis, _non_blocking_opt
    if _non_blocking_vis is not None:
        _non_blocking_vis.destroy_window()
        _non_blocking_vis = None
        _non_blocking_opt = None