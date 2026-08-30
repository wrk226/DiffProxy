import math
import torch


def expand_bbox_norm(bbox_norm, crop_ratio: float):
    """
    将标准化 bbox (x_min, y_min, x_max, y_max) 按照 crop_ratio 等比扩张，裁剪到 [0,1]。
    返回扩张后的标准化坐标 (nx0, ny0, nx1, ny1)。
    """
    x0, y0, x1, y1 = [float(v) for v in bbox_norm]
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bw = (x1 - x0)
    bh = (y1 - y0)
    side = max(bw, bh) * float(crop_ratio)
    nx0 = max(0.0, cx - side * 0.5)
    nx1 = min(1.0, cx + side * 0.5)
    ny0 = max(0.0, cy - side * 0.5)
    ny1 = min(1.0, cy + side * 0.5)
    return nx0, ny0, nx1, ny1


def compute_affine_from_main_bbox(main_bbox_norm,
                                  main_crop_ratio: float,
                                  source_height: int,
                                  source_width: int,
                                  target_height: int,
                                  target_width: int,
                                  dtype: torch.dtype,
                                  device: torch.device) -> torch.Tensor:
    """
    构建从原图像素坐标 (source) 到主视角目标画布像素坐标 (target) 的 3x3 仿射矩阵 A。
    该 A 与数据集中的裁剪/缩放逻辑一致：先对 main bbox 做等比扩张，再裁剪、等比缩放并放置到目标画布中心。
    """
    mx0, my0, mx1, my1 = [float(v) for v in main_bbox_norm]
    mcx = (mx0 + mx1) * 0.5
    mcy = (my0 + my1) * 0.5
    mbw = (mx1 - mx0)
    mbh = (my1 - my0)
    mside = max(mbw, mbh) * float(main_crop_ratio)

    m_nx0_tmp = mcx - mside * 0.5
    m_nx1_tmp = mcx + mside * 0.5
    m_ny0_tmp = mcy - mside * 0.5
    m_ny1_tmp = mcy + mside * 0.5

    m_nx0 = max(m_nx0_tmp, 0.0)
    m_nx1 = min(m_nx1_tmp, 1.0)
    m_ny0 = max(m_ny0_tmp, 0.0)
    m_ny1 = min(m_ny1_tmp, 1.0)

    crop_x0 = round(m_nx0 * source_width)
    crop_x1 = round(m_nx1 * source_width)
    crop_y0 = round(m_ny0 * source_height)
    crop_y1 = round(m_ny1 * source_height)

    actual_w = max(1, crop_x1 - crop_x0)
    actual_h = max(1, crop_y1 - crop_y0)

    tgt_w = round((m_nx1 - m_nx0_tmp) * target_width / (mside if mside != 0 else 1))
    tgt_h = round((m_ny1 - m_ny0_tmp) * target_height / (mside if mside != 0 else 1))
    tgt_w = max(1, tgt_w)
    tgt_h = max(1, tgt_h)

    scale_x = tgt_w / float(actual_w)
    scale_y = tgt_h / float(actual_h)
    off_x = int(round((m_nx0 - m_nx0_tmp) * target_width / (mside if mside != 0 else 1))) - scale_x * crop_x0
    off_y = int(round((m_ny0 - m_ny0_tmp) * target_height / (mside if mside != 0 else 1))) - scale_y * crop_y0

    A = torch.tensor([
        [scale_x, 0.0, off_x],
        [0.0, scale_y, off_y],
        [0.0, 0.0, 1.0],
    ], dtype=dtype, device=device)
    return A


def map_child_bbox_to_main_canvas(child_bbox_norm,
                                  child_crop_ratio: float,
                                  A_main: torch.Tensor,
                                  source_height: int,
                                  source_width: int,
                                  clip_height: int,
                                  clip_width: int) -> tuple:
    """
    将子视角 bbox 从原图坐标映射到主视角目标画布坐标系，返回裁剪后的矩形 (xmin, ymin, xmax, ymax)。
    """
    cx0, cy0, cx1, cy1 = expand_bbox_norm(child_bbox_norm, child_crop_ratio)
    x0p = cx0 * float(source_width)
    x1p = cx1 * float(source_width)
    y0p = cy0 * float(source_height)
    y1p = cy1 * float(source_height)

    def _apply_A(xp, yp):
        v = torch.tensor([xp, yp, 1.0], dtype=A_main.dtype, device=A_main.device)
        out = A_main @ v
        return float(out[0].item()), float(out[1].item())

    X0, Y0 = _apply_A(x0p, y0p)
    X1, Y1 = _apply_A(x1p, y1p)
    xmin = max(0.0, min(X0, X1))
    xmax = min(float(clip_width), max(X0, X1))
    ymin = max(0.0, min(Y0, Y1))
    ymax = min(float(clip_height), max(Y0, Y1))
    return xmin, ymin, xmax, ymax


def rect_to_mask_on_grid(xmin: float,
                         ymin: float,
                         xmax: float,
                         ymax: float,
                         canvas_height: int,
                         canvas_width: int,
                         grid_height: int,
                         grid_width: int,
                         device: torch.device,
                         add_global_tokens: int = 0,
                         dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """
    将画布像素坐标的矩形映射到特征网格 (grid_height x grid_width)，返回扁平化掩码。
    可选在前部拼接 add_global_tokens 个 1（用于 ViT 等 global tokens）。
    """
    px0 = int(round(max(0.0, min(float(canvas_width),  xmin)) / max(1.0, float(canvas_width))  * grid_width))
    px1 = int(round(max(0.0, min(float(canvas_width),  xmax)) / max(1.0, float(canvas_width))  * grid_width))
    py0 = int(round(max(0.0, min(float(canvas_height), ymin)) / max(1.0, float(canvas_height)) * grid_height))
    py1 = int(round(max(0.0, min(float(canvas_height), ymax)) / max(1.0, float(canvas_height)) * grid_height))

    px0, py0 = max(0, px0), max(0, py0)
    px1, py1 = min(grid_width, px1), min(grid_height, py1)

    m = torch.zeros(grid_height, grid_width, dtype=dtype, device=device)
    if px1 > px0 and py1 > py0:
        m[py0:py1, px0:px1] = 1.0
    flat = m.flatten()
    if add_global_tokens > 0:
        flat = torch.cat([torch.ones(add_global_tokens, dtype=dtype, device=device), flat], dim=0)
    return flat


def estimate_patch_grid_dims(num_tokens_total: int, num_global_tokens: int) -> tuple:
    """
    根据总 token 数与 global token 数估计 patch 网格尺寸 (H_feat, W_feat)。
    优先取接近正方形的网格；若无法整除，退化为 (1, S_patch)。
    """
    S_patch = max(1, int(num_tokens_total) - int(num_global_tokens))
    H_feat = int(math.isqrt(S_patch))
    if H_feat <= 0:
        H_feat = 1
    W_feat = max(1, S_patch // H_feat)
    if H_feat * W_feat != S_patch:
        H_feat = 1
        W_feat = S_patch
    return H_feat, W_feat


def map_rect_with_scale_pad(xmin: float,
                            ymin: float,
                            xmax: float,
                            ymax: float,
                            scale_x: float,
                            scale_y: float,
                            pad_x: float,
                            pad_y: float,
                            clip_height: int,
                            clip_width: int) -> tuple:
    """
    将矩形 (xmin,ymin,xmax,ymax) 先按 (scale_x,scale_y) 缩放，再加 (pad_x,pad_y) 平移，
    并裁剪到提供的画布范围，返回新矩形坐标。
    """
    x0 = xmin * scale_x + pad_x
    x1 = xmax * scale_x + pad_x
    y0 = ymin * scale_y + pad_y
    y1 = ymax * scale_y + pad_y
    xmin_o = max(0.0, min(float(clip_width),  min(x0, x1)))
    xmax_o = min(float(clip_width),  max(x0, x1))
    ymin_o = max(0.0, min(float(clip_height), min(y0, y1)))
    ymax_o = min(float(clip_height), max(y0, y1))
    return xmin_o, ymin_o, xmax_o, ymax_o


