import torch
import numpy as np
import imageio
import os
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    OrthographicCameras,
    RasterizationSettings,
    MeshRasterizer,
    TexturesUV
)
# from pytorch3d.ops import interpolate_face_attributes # Not directly used for final image if using texture map
import time
from collections import defaultdict, deque
import copy
import sys
import concurrent.futures
from tqdm import tqdm
import cv2  # 新增：用于在UV空间光栅化三角形

# --- Global Configuration ---
# 选择一个GPU设备ID，所有工作进程将尝试使用此ID指定的GPU
# 如果你的机器有多个GPU，你可以改为0, 1, 2等
# 如果只有一个GPU，通常是0
TARGET_GPU_ID = 0 # 你代码中使用的 device_id
IMAGE_SIZE = (512, 512) # (Height, Width)
BLENDER_ORTHOGRAPHIC_SCALE = 2.5 # 默认的Blender orthographic scale
# 开源版：默认输出到仓库内的 `outputs/dense_body`，也允许用环境变量覆盖。
# 说明：该脚本主要用于生成/调试 UV 分组相关的中间文件，不影响主 pipeline 的最小依赖导入。
SAVE_DIR = os.getenv(
    "DENSE_BODY_SAVE_DIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "outputs", "dense_body"),
)
SAVE_DIR = os.path.abspath(SAVE_DIR)

# --- Helper Functions ---
def B2P(B: np.ndarray):
    B2P_R1 = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64)
    B2P_R2 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]], dtype=np.float64)
    B2P_T = np.array([[-1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)
    R = B2P_R1 @ B[:3, :3] @ B2P_R2
    T = B2P_T @ B[:3, 3] @ R
    return R, T

# --- Texture Preload ---
# 纹理图像只需要创建一次，并可被所有渲染任务共享
# (假设所有模型共用此程序化生成的UV纹理图)
def create_programmatic_texture_image(resolution=512, device=None, to_numpy=False):
    """
    生成程序化UV纹理图像。
    - 如果 to_numpy=True，返回 numpy 格式 (H, W, 3)，uint8。
    - 如果 to_numpy=False，返回 torch 格式 (3, H, W)，float32，范围0~1。
    """
    if to_numpy:
        tex_img = np.zeros((resolution, resolution, 3), dtype=np.uint8)
        for i in range(resolution):
            for j in range(resolution):
                u = j / (resolution - 1)
                v = 1 - i / (resolution - 1)
                tex_img[i, j, 0] = int(u * 255)
                tex_img[i, j, 1] = int(v * 255)
                tex_img[i, j, 2] = 127
        return tex_img
    else:
        tex_img = torch.zeros(3, resolution, resolution, device=device)
        tex_img[0] = torch.linspace(0, 1, resolution, device=device).repeat(resolution, 1)  # R: 0 to 1
        tex_img[1] = torch.linspace(1, 0, resolution, device=device).repeat(resolution, 1).T # G: 1 to 0
        tex_img[2] = 0.5 # B: constant 1.0
        return tex_img.permute(1, 2, 0) # (H, W, 3)

# 全局预加载纹理图像 (将在工作进程中按需移动到各自的device)
# 为了避免在主进程过早占用GPU，这里先不指定device，工作进程会处理
# 如果texture_image非常大，并且你希望只在主进程加载一次并通过IPC共享，那会更复杂。
# 对于这种程序化生成的小纹理，每个进程创建一次是可接受的。
# 如果是加载自文件的巨大纹理，则需要更高级的共享策略或确保每个进程都能高效加载。
# 这里我们假设它在每个worker中创建。
# global_texture_image = create_programmatic_texture_image(IMAGE_SIZE[0])


# --- Core Rendering Function (for a single view given a loaded mesh) ---
def render_single_view_pytorch3d(
    loaded_meshes, # 预加载的 Pytorch3D Meshes 对象
    camera_matrix_path,
    blender_orthographic_scale,
    output_image_path,
    image_size,
    device,
    texture_image_on_device # 已经在正确设备上的纹理图像
):
    try:
        # --- 1. 加载相机参数 ---
        matrix_world = np.loadtxt(camera_matrix_path)
        R_np, T_np = B2P(matrix_world)
        R = torch.from_numpy(R_np).float().to(device)
        T = torch.from_numpy(T_np).float().to(device)

        # --- 2. 设置 PyTorch3D 正交相机 ---
        img_h, img_w = image_size
        img_size_tensor = torch.tensor([[img_h, img_w]], dtype=torch.float32, device=device)

        focal_y = img_h / blender_orthographic_scale
        focal_x = img_w / (blender_orthographic_scale * (img_w / img_h))
        focal_length = torch.tensor([[focal_x, focal_y]], dtype=torch.float32, device=device)
        principal_point = torch.tensor([[img_w / 2, img_h / 2]], dtype=torch.float32, device=device)

        cameras = OrthographicCameras(
            focal_length=focal_length,
            principal_point=principal_point,
            R=R[None],
            T=T[None],
            image_size=img_size_tensor,
            in_ndc=False, # 通常Blender导出的相机参数更适合 in_ndc=False
            device=device
        )

        # --- 3. 配置 PyTorch3D 渲染器 ---
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-10,
            faces_per_pixel=1,
            # perspective_correct=False, # 对于正交投影可能不需要，但通常开启也无妨
        )
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        # --- 4. 准备纹理 ---
        # OBJ文件应包含UV坐标。loaded_meshes.textures 应该已经包含了它们。
        # 我们将用全局纹理图像替换模型自带的纹理图（如果有的话）
        if loaded_meshes.textures is None or not isinstance(loaded_meshes.textures, TexturesUV):
             # 如果OBJ没有UV或纹理信息，我们需要创建它
             # 这通常意味着OBJ本身缺少UVs，渲染出来的可能是纯色
             # 或者你需要确保OBJ文件本身包含有效的 verts_uvs 和 faces_uvs
            print(f"Warning: Mesh for {output_image_path} might be missing UVs. Attempting to use TexturesUV.")
            verts_uvs = loaded_meshes.textures.verts_uvs_padded() if loaded_meshes.textures else torch.rand(loaded_meshes.verts_padded().shape[0], 2, device=device) # Placeholder
            faces_uvs = loaded_meshes.textures.faces_uvs_padded() if loaded_meshes.textures else loaded_meshes.faces_padded() # Placeholder
        else:
            verts_uvs = loaded_meshes.textures.verts_uvs_list()[0] # Assume single mesh in batch
            faces_uvs = loaded_meshes.textures.faces_uvs_list()[0] # Assume single mesh in batch

        # 创建新的纹理对象，使用全局纹理图和模型自己的UV布局
        final_texture = TexturesUV(
            maps=[texture_image_on_device], # 使用传入的、已在设备上的纹理图像
            verts_uvs=[verts_uvs.to(device)],
            faces_uvs=[faces_uvs.to(device)]
        )
        
        # 创建一个新的Meshes对象或克隆以避免修改原始缓存的meshes对象（如果它被多个进程共享且修改有害）
        # 在当前ProcessPoolExecutor模型下，每个worker加载自己的mesh，所以可以直接修改
        current_meshes = loaded_meshes.clone() # 克隆以确保线程安全或避免意外修改
        current_meshes.textures = final_texture


        # --- 5. 渲染 ---
        fragments = rasterizer(current_meshes)
        
        # 使用纹理颜色
        # sample_textures 会根据UV坐标从纹理图中采样颜色
        textured_render = current_meshes.textures.sample_textures(fragments) # (N, H, W, K, C) K是faces_per_pixel

        # 提取颜色和mask
        image_render = textured_render.squeeze(3)[0]  # (H, W, C)
        alpha_mask = (fragments.pix_to_face[0, ..., 0] >= 0).float().unsqueeze(-1) # (H, W, 1)

        # 合成RGBA图像
        final_image_render = torch.cat([image_render, alpha_mask], dim=-1) # (H, W, 4)
        
        # 转换为Numpy数组并保存
        output_image_np = (final_image_render * 255.0).cpu().numpy().astype(np.uint8)
        imageio.imwrite(output_image_path, output_image_np)
        return True, None

    except Exception as e:
        # print(f"Error rendering {output_image_path}: {e}")
        return False, f"Error rendering {output_image_path}: {e}"


# --- Worker Function for ProcessPoolExecutor ---
def process_obj_batch(args):
    """
    工作函数：加载一个OBJ文件，并为其渲染所有关联的相机视角。
    """
    obj_path, camera_jobs, blender_orthographic_scale, image_size, target_gpu_id = args
    
    # --- 1. 设置设备 (每个工作进程一次) ---
    device = None
    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{target_gpu_id}")
            torch.cuda.set_device(device) # 确保后续的cuda操作在此设备上
            # print(f"Worker {os.getpid()} using device: {torch.cuda.get_device_name(device)}")
        except Exception as e:
            print(f"Worker {os.getpid()} failed to set CUDA device cuda:{target_gpu_id} ({e}), falling back to CPU.")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")
        # print(f"Worker {os.getpid()} using device: CPU (CUDA not available)")

    # --- 2. 为当前worker创建/加载纹理图像到其设备 ---
    # 如果纹理图像很大，且从文件加载，这里可以进行加载
    # 对于程序化生成的小纹理，直接在设备上创建
    texture_image_on_device = create_programmatic_texture_image(image_size[0], device=device)

    # --- 3. 加载OBJ模型 (每个OBJ文件一次) ---
    try:
        # PyTorch3D期望obj文件中的mtl和纹理图路径相对于obj文件本身。
        # 如果纹理文件不在预期位置，load_objs_as_meshes可能无法正确加载纹理，
        # 但由于我们后面会用自己的 TexturesUV 覆盖它，所以主要关注几何和UV坐标的加载。
        loaded_meshes = load_objs_as_meshes([obj_path], device=device, load_textures=True)
    except Exception as e:
        error_msg = f"Failed to load OBJ {obj_path} in worker {os.getpid()}: {e}"
        # print(error_msg)
        # 为此OBJ的所有相机任务返回失败
        return [(False, job['output_image_path'], error_msg) for job in camera_jobs]

    results = []
    for job in camera_jobs:
        cam_matrix_path = job['camera_matrix_path']
        output_img_path = job['output_image_path']
        
        success, error = render_single_view_pytorch3d(
            loaded_meshes=loaded_meshes, # 使用已加载的模型
            camera_matrix_path=cam_matrix_path,
            blender_orthographic_scale=blender_orthographic_scale,
            output_image_path=output_img_path,
            image_size=image_size,
            device=device,
            texture_image_on_device=texture_image_on_device
        )
        results.append((success, output_img_path, error))
    
    return results

def group_faces_and_normalize_uvs(obj_path):
    """
    读取OBJ，基于UV连通性对面分组，并对每组UV归一化，返回新UV、面组信息和新OBJ保存路径。
    """
    # 1. 解析OBJ文件，获取v/vt/f/ft
    verts = []
    uvs = []
    faces = []
    faces_uvs = []
    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.strip().split()[1:4]])
            elif line.startswith('vt '):
                uvs.append([float(x) for x in line.strip().split()[1:3]])
            elif line.startswith('f '):
                v_idx = []
                vt_idx = []
                for part in line.strip().split()[1:]:
                    vals = part.split('/')
                    v_idx.append(int(vals[0]) - 1)
                    vt_idx.append(int(vals[1]) - 1)
                faces.append(v_idx)
                faces_uvs.append(vt_idx)
    verts = np.array(verts)
    uvs = np.array(uvs)
    faces = np.array(faces)
    faces_uvs = np.array(faces_uvs)

    # 2. 构建UV连通性邻接表
    uv_face_map = defaultdict(list)  # uv_idx -> [face_idx, ...]
    for face_idx, uv_idxs in enumerate(faces_uvs):
        for uv_idx in uv_idxs:
            uv_face_map[uv_idx].append(face_idx)
    face_neighbors = defaultdict(set)
    for face_idx, uv_idxs in enumerate(faces_uvs):
        neighbors = set()
        for uv_idx in uv_idxs:
            neighbors.update(uv_face_map[uv_idx])
        neighbors.discard(face_idx)
        face_neighbors[face_idx] = neighbors

    # 3. BFS分组
    n_faces = len(faces_uvs)
    face_group = [-1] * n_faces
    group_id = 1
    for i in range(n_faces):
        if face_group[i] != -1:
            continue
        queue = deque([i])
        face_group[i] = group_id
        while queue:
            cur = queue.popleft()
            for nb in face_neighbors[cur]:
                if face_group[nb] == -1:
                    face_group[nb] = group_id
                    queue.append(nb)
        group_id += 1

    # 4. 每组归一化UV（直接覆盖原uvs）
    group_bbox = []
    uvs_new = uvs.copy()
    for gid in range(1, group_id):
        # 找到本组所有面涉及的uv idx
        group_face_indices = [i for i, g in enumerate(face_group) if g == gid]
        group_uv_indices = set()
        for fi in group_face_indices:
            group_uv_indices.update(faces_uvs[fi])
        group_uv_indices = sorted(list(group_uv_indices))
        group_uvs = uvs[list(group_uv_indices)]
        # 归一化
        min_uv = group_uvs.min(axis=0)
        max_uv = group_uvs.max(axis=0)
        group_bbox.append((min_uv, max_uv))
        norm_uvs = (group_uvs - min_uv) / (max_uv - min_uv)
        # 覆盖原uvs
        for idx, uv_idx in enumerate(group_uv_indices):
            uvs_new[uv_idx] = norm_uvs[idx]

    # 5. 新OBJ/MTL/PNG保存
    obj_base = os.path.splitext(os.path.basename(obj_path))[0]
    obj_name = obj_base + "_uvgroup"
    texture_img_np = create_programmatic_texture_image(512, to_numpy=True)
    
    # Ensure SAVE_DIR exists
    os.makedirs(SAVE_DIR, exist_ok=True)

    obj_path_new, mtl_path, tex_path = save_obj_with_mtl_and_texture(
        obj_name, verts, faces, uvs_new, faces_uvs, texture_img_np
    )
    return obj_path_new, mtl_path, tex_path, face_group, uvs_new, faces_uvs, group_bbox

def render_uv_and_semantic(
    obj_path, camera_matrix_path, output_uv_path, output_semantic_path,
    face_group, image_size, device, blender_orthographic_scale,
    global_verts_uvs=None
):
    loaded_meshes = load_objs_as_meshes([obj_path], device=device, load_textures=True)
    # 2. 生成程序化UV纹理
    tex_img = create_programmatic_texture_image(image_size[0], device=device, to_numpy=False)
    # 3. 相机参数
    matrix_world = np.loadtxt(camera_matrix_path)
    R_np, T_np = B2P(matrix_world)
    R = torch.from_numpy(R_np).float().to(device)
    T = torch.from_numpy(T_np).float().to(device)
    img_h, img_w = image_size
    img_size_tensor = torch.tensor([[img_h, img_w]], dtype=torch.float32, device=device)
    focal_y = img_h / blender_orthographic_scale
    focal_x = img_w / (blender_orthographic_scale * (img_w / img_h))
    focal_length = torch.tensor([[focal_x, focal_y]], dtype=torch.float32, device=device)
    principal_point = torch.tensor([[img_w / 2, img_h / 2]], dtype=torch.float32, device=device)
    cameras = OrthographicCameras(
        focal_length=focal_length,
        principal_point=principal_point,
        R=R[None],
        T=T[None],
        image_size=img_size_tensor,
        in_ndc=False,
        device=device
    )
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=1e-10,
        faces_per_pixel=1,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    # 4. UV渲染
    if global_verts_uvs is not None:
        verts_uvs = torch.from_numpy(global_verts_uvs).float().to(device) if isinstance(global_verts_uvs, np.ndarray) else torch.tensor(global_verts_uvs, dtype=torch.float32, device=device)
        faces_uvs = loaded_meshes.textures.faces_uvs_list()[0]
    else:
        verts_uvs = loaded_meshes.textures.verts_uvs_list()[0]
        faces_uvs = loaded_meshes.textures.faces_uvs_list()[0]
    final_texture = TexturesUV(
        maps=[tex_img],
        verts_uvs=[verts_uvs],
        faces_uvs=[faces_uvs]
    )
    current_meshes = loaded_meshes.clone()
    current_meshes.textures = final_texture
    fragments = rasterizer(current_meshes)
    textured_render = current_meshes.textures.sample_textures(fragments)
    image_render = textured_render.squeeze(3)[0]
    alpha_mask = (fragments.pix_to_face[0, ..., 0] >= 0).float().unsqueeze(-1)
    final_image_render = torch.cat([image_render, alpha_mask], dim=-1)
    
    # Save UV map (RGBA)
    output_uv_image_np = (final_image_render * 255.0).cpu().numpy().astype(np.uint8)
    os.makedirs(os.path.dirname(output_uv_path), exist_ok=True)
    imageio.imwrite(output_uv_path, output_uv_image_np)

    # 5. 语义分割渲染
    face_idx_map = fragments.pix_to_face[0, ..., 0].cpu().numpy()  # (H, W)
    face_idx_map_flat = face_idx_map.reshape(-1)
    valid_mask        = face_idx_map_flat >= 0
    semantic_flat     = np.zeros_like(face_idx_map_flat, dtype=np.uint16)
    semantic_flat[valid_mask] = np.take(face_group, face_idx_map_flat[valid_mask])
    semantic_img = semantic_flat.reshape(img_h, img_w)
    os.makedirs(os.path.dirname(output_semantic_path), exist_ok=True)
    imageio.imwrite(output_semantic_path, semantic_img)

def save_obj_with_mtl_and_texture(obj_name, verts, faces, uvs, faces_uvs, texture_img_np):
    """
    保存OBJ、MTL和PNG纹理文件，并在OBJ顶部写入mtl引用。
    obj_name: 不带扩展名的文件名（如 'xxx_uvgroup'）
    """
    obj_path = os.path.join(SAVE_DIR, obj_name + ".obj")
    mtl_path = os.path.join(SAVE_DIR, obj_name + ".mtl")
    tex_path = os.path.join(SAVE_DIR, obj_name + ".png")
    mtl_name = obj_name + ".mtl"
    tex_name = obj_name + ".png"
    # 1. 写PNG纹理
    imageio.imwrite(tex_path, texture_img_np)
    # 2. 写MTL
    with open(mtl_path, "w") as f:
        f.write(f"newmtl mat0\n")
        f.write(f"Ka 1.0 1.0 1.0\n")
        f.write(f"Kd 1.0 1.0 1.0\n")
        f.write(f"Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {tex_name}\n")
    # 3. 写OBJ
    with open(obj_path, "w") as f:
        f.write(f"mtllib {mtl_name}\n")
        for v in verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        f.write(f"usemtl mat0\n")
        for i, face in enumerate(faces):
            uv_idx = faces_uvs[i]
            f.write("f " + " ".join([f"{v+1}/{uv+1}" for v, uv in zip(face, uv_idx)]) + "\n")
    return obj_path, mtl_path, tex_path

# --- New Worker Function for UV and Semantic Rendering ---
def process_obj_batch_uv_semantic(args):
    (
        original_obj_path,
        camera_jobs,
        image_size,
        target_gpu_id,
        blender_orthographic_scale_arg,
        face_group_global,
        uvs_new_global,
    ) = args
    
    # -------------------------------------------------------------
    # 1) 设备初始化
    # -------------------------------------------------------------
    if torch.cuda.is_available():
        try:
            device = torch.device(f"cuda:{target_gpu_id}")
            torch.cuda.set_device(device)
        except Exception as _:
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    results = []

    # -------------------------------------------------------------
    # 2) 预加载静态资源（只做一次）
    #    • Mesh
    #    • 统一纹理 & UV
    #    • raster_settings / 其它常量
    # -------------------------------------------------------------
    with torch.no_grad():
        # 2.1 载入 OBJ
        loaded_meshes = load_objs_as_meshes([original_obj_path], device=device, load_textures=True)

        # 2.2 覆盖 UV（使用主进程预计算好的归一化 UV）并设置程序化纹理
        verts_uvs_global = torch.tensor(uvs_new_global, dtype=torch.float32, device=device)
        faces_uvs       = loaded_meshes.textures.faces_uvs_list()[0]
        tex_img         = create_programmatic_texture_image(image_size[0], device=device, to_numpy=False)
        loaded_meshes.textures = TexturesUV(
            maps=[tex_img],
            verts_uvs=[verts_uvs_global],
            faces_uvs=[faces_uvs]
        )

        # 2.3 语义 face_group 张量（放 GPU，后续直接索引）
        face_group_tensor = torch.tensor(face_group_global, dtype=torch.int16, device=device)

        # 2.4 公共 raster 设置（正交相机每次仍要变，但 raster_settings 可复用）
        raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=1e-10,
            faces_per_pixel=1,
        )

        img_h, img_w = image_size
        img_size_tensor  = torch.tensor([[img_h, img_w]], dtype=torch.float32, device=device)
        focal_base_y = img_h / blender_orthographic_scale_arg
        focal_base_x = img_w / (blender_orthographic_scale_arg * (img_w / img_h))
        focal_length_base    = torch.tensor([[focal_base_x, focal_base_y]], dtype=torch.float32, device=device)
        principal_point_base = torch.tensor([[img_w / 2, img_h / 2]], dtype=torch.float32, device=device)

        # ---------------------------------------------------------
        # 3) 针对每个视角渲染
        # ---------------------------------------------------------
        for job in camera_jobs:
            output_uv_p       = job['output_uv_path']
            output_semantic_p = job['output_semantic_path']
            cam_matrix_p      = job['camera_matrix_path']

            try:
                # 3.1 相机姿态
                matrix_world = np.loadtxt(cam_matrix_p)
                R_np, T_np   = B2P(matrix_world)
                R = torch.from_numpy(R_np).float().to(device)
                T = torch.from_numpy(T_np).float().to(device)

                cameras = OrthographicCameras(
                    focal_length=focal_length_base,
                    principal_point=principal_point_base,
                    R=R[None],
                    T=T[None],
                    image_size=img_size_tensor,
                    in_ndc=False,
                    device=device,
                )

                # 3.2 渲染
                rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
                fragments  = rasterizer(loaded_meshes)
                image_rgb  = loaded_meshes.textures.sample_textures(fragments).squeeze(3)[0]  # (H, W, 3)
                alpha_mask = (fragments.pix_to_face[0, ..., 0] >= 0).float().unsqueeze(-1)
                final_rgba = torch.cat([image_rgb, alpha_mask], dim=-1)  # (H, W, 4)

                # 3.3 保存 UV rgba
                os.makedirs(os.path.dirname(output_uv_p), exist_ok=True)
                imageio.imwrite(output_uv_p, (final_rgba * 255.0).byte().cpu().numpy())

                # 3.4 GPU 向量化生成语义图
                face_idx_map_gpu = fragments.pix_to_face[0, ..., 0]  # (H, W)
                semantic_gpu = torch.zeros_like(face_idx_map_gpu, dtype=torch.int16)
                valid_mask   = face_idx_map_gpu >= 0
                semantic_gpu[valid_mask] = face_group_tensor[face_idx_map_gpu[valid_mask]]

                os.makedirs(os.path.dirname(output_semantic_p), exist_ok=True)
                imageio.imwrite(output_semantic_p, semantic_gpu.cpu().numpy().astype(np.uint16))

                results.append({
                    'uv_path': output_uv_p,
                    'semantic_path': output_semantic_p,
                    'success': True,
                    'error': None,
                })
            except Exception as e:
                results.append({
                    'uv_path': output_uv_p,
                    'semantic_path': output_semantic_p,
                    'success': False,
                    'error': f"Error rendering {original_obj_path} cam {cam_matrix_p}: {e}",
                })

    return results

# === 新增函数: 将每个UV分组保存为单独的纹理PNG ===
def save_uv_group_texture_images(uvs, faces_uvs, face_group, resolution=512, output_dir=None):
    """
    根据归一化后的 UV、三角形到UV索引映射以及 face_group，
    为每个分组生成一张独立的纹理 PNG（RGBA）。
    颜色规则与 create_programmatic_texture_image 一致；非本组区域透明。
    """
    if output_dir is None:
        output_dir = os.path.join(SAVE_DIR, "uv_group_textures")
    os.makedirs(output_dir, exist_ok=True)

    # 基础渐变纹理 (H, W, 3) uint8
    base_tex = create_programmatic_texture_image(resolution, to_numpy=True)

    face_group = np.asarray(face_group)
    unique_groups = np.unique(face_group)

    for gid in unique_groups:
        # 创建遮罩
        mask = np.zeros((resolution, resolution), dtype=np.uint8)
        face_indices = np.where(face_group == gid)[0]
        for fi in face_indices:
            uv_idx = faces_uvs[fi]  # (3,)
            pts_uv = uvs[uv_idx]    # (3,2)
            # 转为像素坐标
            pts_px = np.stack([
                pts_uv[:, 0] * (resolution - 1),              # x = u * (W-1)
                (1.0 - pts_uv[:, 1]) * (resolution - 1)        # y = (1-v)*(H-1)
            ], axis=1).astype(np.int32)
            cv2.fillConvexPoly(mask, pts_px, 255)

        # 组合 RGBA 输出
        rgba = np.zeros((resolution, resolution, 4), dtype=np.uint8)
        rgba[..., 3] = mask  # Alpha
        inside = mask == 255
        rgba[inside, :3] = base_tex[inside]

        out_path = os.path.join(output_dir, f"group_{gid}.png")
        imageio.imwrite(out_path, rgba)

    print(f"已生成 {len(unique_groups)} 张分组纹理，保存在 {output_dir}")

