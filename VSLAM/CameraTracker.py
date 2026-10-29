

import pypose as pp
import torch
import random
from einops import rearrange
import mast3r_slam_backends
import time
import cv2
from VSLAM.ImageFrame import ImageFrame
from VSLAM.mvslam.geometry import (
    act_Sim3,
    point_to_ray_dist,
    get_pixel_coords,
    constrain_points_to_ray,
    project_calib,
)
from VSLAM.mvslam.nonlinear_optimizer import check_convergence, huber
from VSLAM.utils import (
    inverse_normalize, 
    Matrix2Quan, Quan2Matrix, 
    Matrix2Quan_Torch, Quan2Matrix_Torch,
    Matrix2Quan_Kornia, Quan2Matrix_Kornia
)
from VSLAM.utils_uncertainty import local_diag_cov_from_X1
from VSLAM.utils_matching import match_pi3, match
from VSLAM.mvslam.global_opt import FactorGraph
from VSLAM.utils import load_retriever
import torch
from torch_scatter import scatter_max


def compute_valid_mask(
    local_points_f, local_points_k, 
    feats_f, feats_k, 
    idx_f2k, idx_mask,
    K, H, W
):
    device = local_points_k.device
    dtype = local_points_k.dtype
    B, L, C = feats_f.shape

    # --- 1. 提取匹配特征并计算点积 ---
    # 根据索引 idx_f2k 从 feats_f 中提取对应的特征描述子
    idx_feat_expand = idx_f2k.unsqueeze(-1).expand(-1, -1, C)
    D_f1 = torch.gather(feats_f, 1, idx_feat_expand) # [B, L, C]
    
    # 计算点积 (Dot Product) [B, L]
    feat_values = (feats_k * D_f1).sum(dim=-1)
    
    feat_map = feat_values[-1]
    combined_invalid_mask = ~idx_mask[-1]
    for i in range(2, len(feat_values) + 1):
        curr_feat = feat_values[-i]
        curr_valid_mask = idx_mask[-i]
        feat_map[combined_invalid_mask] = curr_feat[combined_invalid_mask]
        combined_invalid_mask &= ~curr_valid_mask
    feat_map[combined_invalid_mask] = feat_map[~combined_invalid_mask].mean()
    
    feat_values[-1] = feat_map

    # --- 2. 投影坐标准备 ---
    if K.dim() == 2:
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    else:
        fx, fy = K[:, 0, 0].view(B, 1), K[:, 1, 1].view(B, 1)
        cx, cy = K[:, 0, 2].view(B, 1), K[:, 1, 2].view(B, 1)

    z_kk = local_points_k[..., 2].clamp(min=1e-8)
    u_k = (local_points_k[..., 0] * fx / z_kk) + cx
    v_k = (local_points_k[..., 1] * fy / z_kk) + cy
    
    u_px = torch.round(u_k).long()
    v_px = torch.round(v_k).long()
    
    # 有效性检查：在图像范围内且深度大于 0
    in_bounds = (u_px >= 0) & (u_px < W) & (v_px >= 0) & (v_px < H) & (local_points_k[..., 2] > 1e-5)

    # --- 3. 渲染到图像平面 ---
    # 展平处理有效点
    b_idx = torch.arange(B, device=device).view(B, 1).expand(B, L)
    b_valid = b_idx[in_bounds]
    u_valid = u_px[in_bounds]
    v_valid = v_px[in_bounds]
    z_valid = local_points_k[..., 2][in_bounds]
    feat_valid = feat_values[in_bounds]

    # 处理遮挡：按深度降序排列，确保较近的点最后赋值（覆盖远点）
    sort_idx = torch.argsort(z_valid, descending=True)
    
    feat_map = torch.full((B, H, W), -1.0, dtype=dtype, device=device)

    # 2. 批量渲染有效点（利用 sort_idx 确保近处物体覆盖远处物体）
    # 使用 b_sorted 等变量使代码更整洁
    b_sorted = b_valid[sort_idx]
    v_sorted = v_valid[sort_idx]
    u_sorted = u_valid[sort_idx]
    feat_map[b_sorted, v_sorted, u_sorted] = feat_valid[sort_idx]

    # 3. 填充背景（无效区域）
    # 提取所有有效填充的值
    valid_mask = feat_map != -1.0
    
    if valid_mask.any():
        # 计算有效值的平均值，并填充到背景
        fill_value = feat_map[valid_mask].mean()
        feat_map[~valid_mask] = fill_value
    else:
        feat_map.fill_(1.0)

    return feat_map

def project_and_refine_batched(
    pts_w,          # (N, H, W, 3)
    desc_1,         # (N, H, W, C)
    desc_2,         # (N, H, W, C)
    c2w_j,          # (N, 4, 4)
    K_j,            # (N, 3, 3) or (3, 3)
    search_radius=2
):
    N, H, W, C = desc_1.shape
    M = H * W
    device = pts_w.device

    # 1. Data Flattening (keep batch dim)
    pts_w_flat = pts_w.reshape(N, M, 3)
    desc_1_flat = desc_1.reshape(N, M, C)
    desc_2_flat = desc_2.reshape(N, M, C)

    # 2. Geometric Projection (Batched)
    # w2c = inverse(c2w) -> (N, 4, 4)
    w2c_j = torch.inverse(c2w_j)
    R = w2c_j[:, :3, :3] # (N, 3, 3)
    t = w2c_j[:, :3, 3:4] # (N, 3, 1)

    # Transform pts to camera coords: (N, M, 3)
    pts_c = (R @ pts_w_flat.transpose(1, 2) + t).transpose(1, 2)
    
    z = pts_c[..., 2:] # (N, M, 1)
    z_safe = torch.clamp(z, min=1e-3)
    
    # Projection using batched intrinsics
    # If K_j is (3, 3), unsqueeze to (1, 3, 3) for broadcasting
    if K_j.dim() == 2: K_j = K_j.unsqueeze(0)
    
    # Project: (N, M, 2) = (N, M, 2) * (N, 1, 2) + (N, 1, 2)
    u_proj = (pts_c[..., 0] / z_safe.squeeze(-1)) * K_j[:, 0, 0] + K_j[:, 0, 2]
    v_proj = (pts_c[..., 1] / z_safe.squeeze(-1)) * K_j[:, 1, 1] + K_j[:, 1, 2]
    
    valid_mask_geo = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H) & (z.squeeze(-1) > 1e-3)

    # 3. 2D Local Feature Search
    if search_radius > 0:
        r = torch.arange(-search_radius, search_radius + 1, device=device)
        dy, dx = torch.meshgrid(r, r, indexing='ij')
        offsets = torch.stack([dx.flatten(), dy.flatten()], dim=1) # (K*K, 2)
        num_offsets = offsets.shape[0]

        u_init = torch.round(u_proj).long()
        v_init = torch.round(v_proj).long()

        # neighbor_coords: (N, M, K*K, 2)
        coords = torch.stack([u_init, v_init], dim=-1).unsqueeze(2) + offsets.view(1, 1, -1, 2)
        
        u_nb = torch.clamp(coords[..., 0], 0, W - 1)
        v_nb = torch.clamp(coords[..., 1], 0, H - 1)

        # Batch-aware indexing: (N, M, K*K)
        # We need to add the batch offset to use a flat view
        nb_indices_local = v_nb * W + u_nb 
        
        # Gather features from desc_2_flat: (N, M, C)
        # We use gather or clever reshaping. Reshaping to (N, M*C) is complex, 
        # so we expand desc_2_flat or use advanced indexing.
        
        # Create batch indices for advanced indexing: (N, M, K*K)
        batch_idx = torch.arange(N, device=device).view(N, 1, 1).expand(N, M, num_offsets)
        
        # desc_neighbors: (N, M, K*K, C)
        desc_neighbors = desc_2_flat[batch_idx, nb_indices_local]

        # Similarity: (N, M, C) x (N, M, K*K, C) -> (N, M, K*K)
        similarity = torch.einsum('nmc,nmkc->nmk', desc_1_flat, desc_neighbors)

        best_nb_local_idx = torch.argmax(similarity, dim=2) # (N, M)
        
        # Final indices relative to each batch: (N, M)
        # To get the absolute index in a flattened (N*H*W) array, you'd add batch_idx * (H*W)
        idx_i2j = torch.gather(nb_indices_local, 2, best_nb_local_idx.unsqueeze(-1)).squeeze(-1)
    else:
        u_idx = torch.round(torch.clamp(u_proj, 0, W - 1)).long()
        v_idx = torch.round(torch.clamp(v_proj, 0, H - 1)).long()
        idx_i2j = v_idx * W + u_idx

    return idx_i2j, valid_mask_geo


def update_points_single(local_points, camera_pose, K_old, K_new):
    """
    针对单帧数据更新点云坐标 (无 Batch 维度)
    
    Args:
        local_points: (H, W, 3) 相机坐标系下的点云
        camera_pose:  (4, 4) c2w 矩阵
        K_old:        (3, 3) 原始内参
        K_new:        (3, 3) 新内参
    
    Returns:
        pts_local_flat: (M, 3) 转换后并展平的相机坐标系点云 (M = H * W)
        pts_world:      (H, W, 3) 转换后的世界坐标系点云
    """
    
    def homogenize_points(points):
        """将点云 (xyz) 转换为齐次坐标 (xyz1)"""
        return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

    # 1. 提取深度 Z (H, W, 1)
    z = local_points[..., 2:3]
    
    # 2. 计算焦距缩放比例
    # 注意：如果内参改变是为了匹配图像缩放，sx/sy 通常用于调整相机坐标系的 X/Y 跨度
    fx_old, fy_old = K_old[0, 0], K_old[1, 1]
    fx_new, fy_new = K_new[0, 0], K_new[1, 1]
    
    sx = fx_new / fx_old
    sy = fy_new / fy_old
    
    # 3. 变换相机坐标系下的 X 和 Y
    # 根据你的原逻辑：缩放相机坐标以适应新的内参定义
    new_x = local_points[..., 0:1] * sx
    new_y = local_points[..., 1:2] * sy
    
    # 4. 组合新的相机坐标系点云 (H, W, 3)
    new_local_points = torch.cat([new_x, new_y, z], dim=-1)
    
    # 5. 转换到世界坐标系
    # einsum 逻辑：(4, 4) @ (H, W, 4) -> (H, W, 4)
    # 'ij, hwj -> hwi' 表示矩阵右乘每一像素点的齐次坐标
    new_world_points = torch.einsum('ij, hwj -> hwi', camera_pose, homogenize_points(new_local_points))[..., :3]
    
    # 6. 展平相机坐标系点云 (H*W, 3)，保持世界坐标系点云为 (H, W, 3)
    new_local_points_flat = rearrange(new_local_points, "h w c -> (h w) c")    
    
    return new_local_points_flat, new_world_points

def update_points(local_points, camera_poses, K_old, K_new):
    """
    针对 Batch 维度数据更新点云坐标
    
    Args:
        local_points: (N, H, W, 3) 相机坐标系下的点云
        camera_poses: (N, 4, 4) c2w 矩阵 (本函数中物理位姿通常保持不变)
        K_old: (3, 3) 原始内参
        K_new: (3, 3) 新内参
    
    Returns:
        pts_new: (N, H, W, 3) 转换后的点云
        camera_poses: (N, 4, 4) 保持不变的物理位姿
    """
    def homogenize_points(
        points,
    ):
        """Convert batched points (xyz) to (xyz1)."""
        return torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)

    # 提取深度 Z (N, H, W, 1)
    # 深度是物理距离，不随内参缩放而改变
    z = local_points[..., 2:3]
    
    # 获取焦距缩放比例
    # s = f_new / f_old
    fx_old, fy_old = K_old[0, 0], K_old[1, 1]
    fx_new, fy_new = K_new[0, 0], K_new[1, 1]
    
    sx = fx_new / fx_old
    sy = fy_new / fy_old
    
    # 1. 变换 X 和 Y 坐标
    # 原理：X_new = (u - cx_new) * Z / fx_new
    # 当我们从旧点转换时，实际上是按比例缩放相机坐标系的跨度
    new_x = local_points[..., 0:1] * sx
    new_y = local_points[..., 1:2] * sy
    
    # 2. 组合新的点云 (N, H, W, 3)
    new_local_points = torch.cat([new_x, new_y, z], dim=-1)
    new_points = torch.einsum('nij, nhwj -> nhwi', camera_poses, homogenize_points(new_local_points))[..., :3]
    new_local_points = rearrange(new_local_points, "b h w c -> b (h w) c")    
    
    return new_local_points, new_points

def project_and_refine(
    pts_w,          # (H, W, 3) 源点云 (世界坐标系)
    desc_1,         # (H, W, C) 源图像特征
    desc_2,         # (H, W, C) 目标图像特征
    c2w_j,          # (4, 4) 目标相机位姿
    K_j,            # (3, 3) 目标相机内参
    search_radius=2 # 2D 搜索半径
):
    H, W, C = desc_1.shape
    device = pts_w.device

    # 1. 数据扁平化 (Flatten)
    # 保持 (M, C) 格式，其中 M = H * W
    pts_w_flat = pts_w.reshape(-1, 3)
    desc_1_flat = desc_1.reshape(-1, C)
    desc_2_flat = desc_2.reshape(-1, C)

    # 2. 几何投影 (Coarse Stage)
    # 计算 World -> Camera 变换
    w2c_j = torch.inverse(c2w_j)
    pts_c = pts_w_flat @ w2c_j[:3, :3].T + w2c_j[:3, 3]
    
    z = pts_c[..., 2]
    z_safe = torch.where(z > 1e-3, z, torch.ones_like(z) * 1e-3)
    
    # 投影公式: u = fx * (X/Z) + cx, v = fy * (Y/Z) + cy
    u_proj = (pts_c[..., 0] / z_safe) * K_j[0, 0] + K_j[0, 2]
    v_proj = (pts_c[..., 1] / z_safe) * K_j[1, 1] + K_j[1, 2]
    
    # 
    # 几何有效性掩码
    valid_mask_geo = (u_proj >= 0) & (u_proj < W) & (v_proj >= 0) & (v_proj < H) & (z > 1e-3)

    # 3. 2D 局部特征搜索 (Fine Stage)
    if search_radius > 0:
        # 构建 2D 搜索偏移量网格
        from VSLAM.local_search_fused import local_search, supported
        u_init = torch.round(u_proj).long()
        v_init = torch.round(v_proj).long()
        if not getattr(project_and_refine, '_fused_path_logged', False):
            fast_path = supported(
                desc_1_flat, desc_2_flat, u_init, v_init, H, W, search_radius,
            )
            print(f'[fused_matching] fast_path={fast_path} '
                  f'tf32={torch.backends.cuda.matmul.allow_tf32} '
                  f'autocast={torch.is_autocast_enabled("cuda")} '
                  f'dtype={desc_1_flat.dtype} shape={tuple(desc_1_flat.shape)}', flush=True)
            project_and_refine._fused_path_logged = True
        idx_i2j = local_search(
            desc_1_flat, desc_2_flat, u_init, v_init, H, W, search_radius,
        )
    else:
        # 无搜索，直接返回投影索引
        u_idx = torch.round(torch.clamp(u_proj, 0, W - 1)).long()
        v_idx = torch.round(torch.clamp(v_proj, 0, H - 1)).long()
        idx_i2j = v_idx * W + u_idx

    return idx_i2j, valid_mask_geo

def estimate_intrinsic_ransac_guided(pts_w, idx_i2j, valid_mask, c2w_j, W_j, H_j, 
                                     K_initial=None, iters=500, threshold=1.0):
    """
    Args:
        K_initial: (3, 3) 初始内参矩阵，用于加速和引导
    """
    pts_w_v = pts_w[valid_mask]
    u_v = (idx_i2j[valid_mask] % W_j).float()
    v_v = (idx_i2j[valid_mask] // W_j).float()
    pix_v = torch.stack([u_v, v_v], dim=-1)

    w2c_j = torch.inverse(c2w_j)
    pts_c = pts_w_v @ w2c_j[:3, :3].T + w2c_j[:3, 3]
    z = pts_c[:, 2:3]
    xy_n = pts_c[:, :2] / (z + 1e-8)

    M = pts_w_v.shape[0]
    if M < 4: return None, None

    # --- 优化 1: 预计算初始 K 的效果 ---
    best_inliers_mask = None
    max_inliers = 0
    
    if K_initial is not None:
        with torch.no_grad():
            i_fx, i_fy = K_initial[0, 0], K_initial[1, 1]
            i_cx, i_cy = K_initial[0, 2], K_initial[1, 2]
            
            pred_u = xy_n[:, 0] * i_fx + i_cx
            pred_v = xy_n[:, 1] * i_fy + i_cy
            error = torch.norm(torch.stack([pred_u - u_v, pred_v - v_v], dim=-1), dim=-1)
            
            inliers_mask = error < threshold
            max_inliers = inliers_mask.sum()
            best_inliers_mask = inliers_mask
            
            # 如果初始 K 已经非常完美，直接精修并返回
            if max_inliers > M * 0.9: 
                return refine_k(xy_n, pix_v, best_inliers_mask, pts_w.device), best_inliers_mask

    # --- 优化 2: 引导采样权重 ---
    # 如果有初始误差，我们更倾向于采样误差小的点（PROSAC 思想）
    if K_initial is not None:
        # 权重与误差成反比，增加采样到 Inlier 的概率
        weights = 1.0 / (error + 0.1) 
    else:
        weights = torch.ones(M, device=pts_w.device)

    # RANSAC 循环
    for i in range(iters):
        # 使用权重进行多项式采样
        indices = torch.multinomial(weights, 4) 
        sample_xy_n = xy_n[indices]
        sample_pix = pix_v[indices]

        try:
            # 求解临时 K (逻辑同前)
            t_fx, t_cx = solve_lstsq(sample_xy_n[:, 0], sample_pix[:, 0:1])
            t_fy, t_cy = solve_lstsq(sample_xy_n[:, 1], sample_pix[:, 1:2])

            # 优化 3: 合理性检查 (解出来的焦距不能偏离初始值太离谱)
            if K_initial is not None:
                if torch.abs(t_fx - i_fx) > i_fx * 0.5: continue 

            # 计算 Inliers
            cur_u = xy_n[:, 0] * t_fx + t_cx
            cur_v = xy_n[:, 1] * t_fy + t_cy
            cur_inliers = (torch.sqrt((cur_u - u_v)**2 + (cur_v - v_v)**2) < threshold)
            num_inliers = cur_inliers.sum()

            if num_inliers > max_inliers:
                max_inliers = num_inliers
                best_inliers_mask = cur_inliers
                
                # 动态调整迭代次数：如果已经找到了足够好的模型，提前退出
                if max_inliers > M * 0.95: break
        except:
            continue

    if best_inliers_mask is not None:
        return refine_k(xy_n, pix_v, best_inliers_mask, pts_w.device), best_inliers_mask
    return None, None


def estimate_intrinsic_ransac_joint(pts_w, idx_i2j, valid_mask, c2w_j, W_j, H_j, 
                                    K_initial=None, iters=1000, threshold=1.0):
    """
    多视图联合 RANSAC 内参估计 (支持每个视角独立的 pts_w)
    
    Args:
        pts_w: (N, M, 3) 每个视角下的世界坐标点
        idx_i2j: (N, M) 像素索引
        valid_mask: (N, M) 布尔掩码
        c2w_j: (N, 4, 4) 外参
        ...
    """
    N, M = idx_i2j.shape
    device = pts_w.device

    # --- 1. 批量预处理 ---
    
    # A. 计算 w2c
    w2c_j = torch.inverse(c2w_j)
    R = w2c_j[:, :3, :3] # (N, 3, 3)
    t = w2c_j[:, :3, 3:] # (N, 3, 1) -> 转置后为 (N, 1, 3) 以便广播

    # B. 坐标转换: 从 pts_w (N, M, 3) 到 pts_c (N, M, 3)
    # 计算公式: P_c[i] = P_w[i] @ R[i].T + t[i].T
    # 使用 @ 运算符在 PyTorch 中会自动进行 Batch Matrix Multiplication
    pts_c = torch.bmm(pts_w, R.transpose(1, 2)) + t.transpose(1, 2)

    # C. 归一化平面坐标 (N, M, 2)
    z = pts_c[..., 2:3]
    xy_n_all = pts_c[..., :2] / (z + 1e-8) 

    # D. 提取像素坐标 (N, M, 2)
    # 假设 idx_i2j 存储的是扁平化索引 (v * W + u)
    u_all = (idx_i2j % W_j).float()
    v_all = (idx_i2j // W_j).float()
    pix_all = torch.stack([u_all, v_all], dim=-1)
    
    # --- 2. 数据池化 (Pooling) ---
    # 利用 valid_mask 提取所有有效的对齐点对
    pool_xy_n = xy_n_all[valid_mask] # (K, 2)
    pool_pix  = pix_all[valid_mask]  # (K, 2)

    num_points = pool_xy_n.shape[0]
    if num_points < 6:
        return None, None

    # --- 3. 初始引导 & RANSAC 逻辑 ---
    # (此部分逻辑与之前一致，因为 pool_xy_n 已经是扁平化的全量数据池)
    best_inliers_mask = None
    max_inliers = 0
    
    # 初始化内参参考值 (fx, fy, cx, cy)
    i_fx, i_fy = (K_initial[0,0], K_initial[1,1]) if K_initial is not None else (W_j, W_j)
    i_cx, i_cy = (K_initial[0,2], K_initial[1,2]) if K_initial is not None else (W_j/2, H_j/2)

    # 计算初始权重 (如果提供初始 K)
    if K_initial is not None:
        pred_u = pool_xy_n[:, 0] * i_fx + i_cx
        pred_v = pool_xy_n[:, 1] * i_fy + i_cy
        error = torch.sqrt((pred_u - pool_pix[:, 0])**2 + (pred_v - pool_pix[:, 1])**2)
        weights = 1.0 / (error + 0.1)
        
        # 初始检查
        best_inliers_mask = error < threshold
        max_inliers = best_inliers_mask.sum()
        if max_inliers > num_points * 0.95:
            return refine_k(pool_xy_n, pool_pix, best_inliers_mask, device), best_inliers_mask
    else:
        weights = torch.ones(num_points, device=device)

    # --- 4. 循环采样 ---
    for _ in range(iters):
        indices = torch.multinomial(weights, 4)
        s_xy_n = pool_xy_n[indices]
        s_pix  = pool_pix[indices]

        try:
            # 最小二乘求解 (Ax = B)
            t_fx, t_cx = solve_lstsq(s_xy_n[:, 0], s_pix[:, 0:1])
            t_fy, t_cy = solve_lstsq(s_xy_n[:, 1], s_pix[:, 1:2])
            
            # 计算全量误差
            err_sq = (pool_xy_n[:, 0] * t_fx + t_cx - pool_pix[:, 0])**2 + \
                     (pool_xy_n[:, 1] * t_fy + t_cy - pool_pix[:, 1])**2
            
            cur_mask = err_sq < (threshold ** 2)
            cur_inliers = cur_mask.sum()

            if cur_inliers > max_inliers:
                max_inliers = cur_inliers
                best_inliers_mask = cur_mask
                if max_inliers > num_points * 0.98: break
        except:
            continue

    if best_inliers_mask is not None:
        return refine_k(pool_xy_n, pool_pix, best_inliers_mask, device), best_inliers_mask
    
    return None, None

def solve_lstsq(a, b):
    A = torch.stack([a, torch.ones_like(a)], dim=1)
    return torch.linalg.lstsq(A, b).solution.flatten()

def refine_k(xy_n, pix_v, mask, device):
    final_fx, final_cx = solve_lstsq(xy_n[mask, 0], pix_v[mask, 0:1])
    final_fy, final_cy = solve_lstsq(xy_n[mask, 1], pix_v[mask, 1:2])
    K = torch.eye(3, device=device)
    K[0,0], K[0,2], K[1,1], K[1,2] = final_fx, final_cx, final_fy, final_cy
    return K

def compute_idx_i2j(pts_w, c2w_j, K_j, H_j, W_j):
    """
    计算从世界点云投影到图 j 的索引，越界点会被限制在边缘。
    
    Args:
        pts_w: (N, 3) 世界坐标系下的点云
        c2w_j: (4, 4) 图 j 的 Camera-to-World 矩阵
        K_j:   (3, 3) 图 j 的内参矩阵
        H_j, W_j: int 图 j 的分辨率
        
    Returns:
        idx_i2j: (N,) 映射索引，越界点被 clamp 到边缘 [0, Wj*Hj - 1]
        valid_mask: (N,) 布尔类型，True 表示点在图 j 的原始视野内且深度为正
    """
    # 1. 坐标系转换: World -> Camera_j
    # T_cw = T_wc^(-1)
    w2c_j = torch.inverse(c2w_j)
    R_j = w2c_j[:3, :3]
    t_j = w2c_j[:3, 3]
    
    # P_c = R * P_w + t
    pts_c_j = pts_w @ R_j.T + t_j
    
    # 2. 投影到像素平面
    z = pts_c_j[..., 2]
    # 防止除以 0，并在后续 Mask 中处理 z <= 0 的情况
    z_safe = torch.where(z > 1e-3, z, torch.ones_like(z) * 1e-3)
    
    fx, fy = K_j[0, 0], K_j[1, 1]
    cx, cy = K_j[0, 2], K_j[1, 2]
    
    u = (pts_c_j[..., 0] / z_safe) * fx + cx
    v = (pts_c_j[..., 1] / z_safe) * fy + cy
    
    # 3. 计算有效性掩码 (Original Validity)
    # 只有在图像范围内且深度为正的点才是真正有效的
    valid_mask = (u >= 0) & (u < W_j - 0.5) & \
                 (v >= 0) & (v < H_j - 0.5) & \
                 (z > 1e-3)
    
    # 4. 边界截断 (Clamping)
    # 将坐标限制在 [0, W-1] 和 [0, H-1] 之间
    u_clamped = torch.clamp(u, 0, W_j - 1)
    v_clamped = torch.clamp(v, 0, H_j - 1)
    
    # 5. 生成索引
    # 使用 round 确保映射到最近的像素中心
    u_idx = torch.round(u_clamped).long()
    v_idx = torch.round(v_clamped).long()
    
    idx_i2j = v_idx * W_j + u_idx
    
    return idx_i2j.view(-1), valid_mask.view(-1)

def compute_idx_i2j_batched(pts_w, c2ws, K, H, W):
    """
    计算世界点云投影到 N 个相机视角的索引 (支持 Per-view pts_w)。
    
    Args:
        pts_w: (N, M, 3) 每个视角各自对应的世界坐标点
        c2ws:  (N, 4, 4) N 个视角的 Camera-to-World 矩阵
        K:     (3, 3) 相机内参
        H, W:  int 图像分辨率
    """
    N = c2ws.shape[0] 
    M = pts_w.shape[1] # 注意这里从 dim 1 获取 M
    device = pts_w.device

    # 1. 批量坐标系转换: World -> Cameras
    w2cs = torch.inverse(c2ws)
    R = w2cs[:, :3, :3] # (N, 3, 3)
    t = w2cs[:, :3, 3]  # (N, 3)

    # 批量投影变动点：
    # pts_w: (N, M, 3)
    # R.transpose(1, 2): (N, 3, 3)
    # t.unsqueeze(1): (N, 1, 3)
    # 结果 pts_c: (N, M, 3)
    # 计算逻辑：对于每个 batch i, 执行 pts_w[i] @ R[i].T + t[i]
    pts_c = pts_w @ R.transpose(1, 2) + t.unsqueeze(1)
    
    # 2. 投影到像素平面 (逻辑保持不变，因为 pts_c 依然是 (N, M, 3))
    x_c = pts_c[..., 0]
    y_c = pts_c[..., 1]
    z_c = pts_c[..., 2]
    
    z_safe = torch.where(z_c > 1e-3, z_c, torch.ones_like(z_c) * 1e-3)
    
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    u = (x_c / z_safe) * fx + cx
    v = (y_c / z_safe) * fy + cy
    
    # 3. 计算有效性掩码 (N, M)
    valid_mask = (u >= 0) & (u < W - 0.5) & \
                 (v >= 0) & (v < H - 0.5) & \
                 (z_c > 1e-3) 
    
    # 4. 生成索引
    u_idx = torch.round(torch.clamp(u, 0, W - 1)).long()
    v_idx = torch.round(torch.clamp(v, 0, H - 1)).long()
    
    idx_batched = v_idx * W + u_idx
    
    return idx_batched, valid_mask

class CameraTracker:
    def __init__(self, args, config, model, H_slam, W_slam, K_slam, frames, states, device):

        self.config = config
        self.args = args
        self.cfg_track = config["tracking"]

        self.keyframes = frames
        self.model = model
        self.num_ref = self.args.num_ref
        self.num_GBA = self.args.num_GBA
        self.device = device
        self.H_slam = H_slam
        self.W_slam = W_slam
        self.K_slam = K_slam
        self.states = states
        self.thres_keyframe = self.args.thres_keyframe
        self.optimize_focal = args.optimize_focal
        self.covariance_filter = args.covariance_filter
        self.min_displacement = max(self.args.min_displacement * self.W_slam, 30)
        self.last_dist = 0

        self.retrieval_database = None
        self.factor_graph = FactorGraph(self.args, self.config, self.model, self.keyframes, self.device)
        self.temp_frames = []
        self.last_track_id = -1
        self.last_dynamic_mask = None

    def init_retrieval(self):
        self.retrieval_database = load_retriever(self.args, self.config, self.model, self.keyframes, None, device=self.device)

    # 追踪当前帧
    def track(self, new_frames, key_T_WC=None):
        # use last keyframes to act as reference frame for current frames's localization
        num_key = len(self.keyframes)
        num_prevkey = min(num_key, self.num_ref)
        if num_prevkey > 0:
            last_keyframe_id = num_key - 1
            last_keyframe = self.keyframes[last_keyframe_id]
            if last_keyframe_id != self.last_track_id:
                kf_idx = [last_keyframe_id]
                ###### 2. 从库中通过特征聚合找到与当前帧相关的k个帧，顺便加入到库中
                retrieval_inds = self.retrieval_database.update(
                    last_keyframe_id,
                    k=num_prevkey - 1,
                    # min_thresh=self.config["retrieval"]["min_thresh"],
                )
                kf_idx += retrieval_inds
                self.last_track_id = last_keyframe_id
                self.kf_idx = kf_idx
                self.temp_frames = [self.keyframes[idx].to(self.device) for idx in kf_idx]
            num_prevkey = len(self.temp_frames)
            cached_frames = self.temp_frames + new_frames
        else:
            cached_frames = new_frames
            last_keyframe_id = None
            last_keyframe = None
            self.kf_idx = None

        views = {}

        # inference dense points and initializal poses
        images = [inverse_normalize(frame.img).permute(2, 0, 1) for frame in cached_frames]
        images = torch.stack(images, dim=0)[None].contiguous()
        views["images"] = images
        if self.K_slam is not None and self.args.use_calib_for_inference:
            intrinsics = [self.K_slam] * len(cached_frames)
            intrinsics = torch.stack(intrinsics, dim=0)[None].contiguous()
            views["intrinsics"] = intrinsics

        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            res = self.model(views)

        if key_T_WC is not None:
            k2w = Quan2Matrix(key_T_WC.data[:, :7]).to(self.device)
        else:
            k2w = torch.eye(4).to(self.device)

        points, conf, local_points, camera_poses, feats, match_conf= self.model.post_process(res, k2w)
        est_N, est_C = points.shape[0], points.shape[3]
        ii_idxs = [0] * (len(points)-1)
        jj_idxs = range(1, len(points))
        points_ii, feats_ii, cams_ii = points[ii_idxs].contiguous(), feats[ii_idxs].contiguous(), camera_poses[ii_idxs].contiguous()
        points_jj, feats_jj, cams_jj = points[jj_idxs].contiguous(), feats[jj_idxs].contiguous(), camera_poses[jj_idxs].contiguous()
        idx_f2k_opt_calib, idx_f2k_valids = match(self.config, points_jj, points_ii, feats_jj, feats_ii, None)
    
        points_ii, points_jj = points_ii.view(len(points)-1, -1, est_C), points_jj.view(len(points)-1, -1, est_C)
        if self.K_slam is not None:
            idx_f2k_use_extr, idx_valid_mask = compute_idx_i2j_batched(points_ii, cams_jj, self.K_slam, self.H_slam, self.W_slam)
            est_K, _ = estimate_intrinsic_ransac_joint(points_ii, idx_f2k_opt_calib, idx_valid_mask, cams_jj, self.W_slam, self.H_slam, self.K_slam)
            local_points, points = update_points(local_points.reshape(est_N, self.H_slam, self.W_slam, est_C), camera_poses, self.K_slam, est_K)
        else:
            idx_valid_mask = (idx_f2k_opt_calib >= 0) & (idx_f2k_opt_calib < self.H_slam * self.W_slam)
            self.K_slam, _ = estimate_intrinsic_ransac_joint(points_ii, idx_f2k_opt_calib, idx_valid_mask, cams_jj, self.W_slam, self.H_slam)

        T_WCs_init = Matrix2Quan(camera_poses)
        T_WCs_init = torch.concatenate([T_WCs_init, torch.ones_like(T_WCs_init[:, :1])], dim=1)
        T_WCs_init = pp.Sim3(T_WCs_init).to(torch.float32)

        # 两帧之间推理，找出关键帧
        frames, frames_style = [], []
        idx_f2ks, idx_masks = [], []
        start_index = num_prevkey

        if num_prevkey == 0:
            cached_frames[0].update_pointmap(local_points[0], conf[0].unsqueeze(-1))
            T_WCf = T_WCs_init[0:1]
            T_WCf = pp.quat2unit(T_WCf)
            cached_frames[0].T_WC = T_WCf
            cached_frames[0].K = self.K_slam
            frames.append(cached_frames[0])
            frames_style.append(1)
            
            idx_f2ks_init = torch.arange(local_points[0].shape[0]).to(self.device)
            idx_masks_init = torch.ones_like(idx_f2ks_init).to(self.device)
            idx_f2ks.append(idx_f2ks_init)
            idx_masks.append(idx_masks_init)
            start_index += 1

        if self.args.use_dynamic:
            f_indexs = list(range(start_index, len(cached_frames)))
            k_indexs = [0] * len(f_indexs)
            idx_f2k_use_extr, idx_valid_mask = compute_idx_i2j_batched(
                points[k_indexs].contiguous().view(len(k_indexs), self.H_slam * self.W_slam, -1), 
                camera_poses[f_indexs].contiguous(), 
                self.K_slam, self.H_slam, self.W_slam
            )
            feat_map = compute_valid_mask(
                local_points[f_indexs].contiguous(),
                local_points[k_indexs].contiguous(),
                feats[f_indexs].contiguous().view(len(f_indexs), self.H_slam * self.W_slam, -1), 
                feats[k_indexs].contiguous().view(len(f_indexs), self.H_slam * self.W_slam, -1), 
                idx_f2k_use_extr, idx_valid_mask, 
                self.K_slam, self.H_slam, self.W_slam
            )
            if self.last_dynamic_mask is not None:
                dynamic_mask = self.last_dynamic_mask * feat_map[-1]
            else:
                dynamic_mask = feat_map[-1]
        else:
            dynamic_mask = None

        
        img_size = [self.H_slam, self.W_slam]
        k_index = 0
        if start_index < len(cached_frames):
            for f_index in range(start_index, len(cached_frames)):
                k_frame = cached_frames[k_index]
                f_frame = cached_frames[f_index]
                f_frame.update_pointmap(local_points[f_index], conf[f_index].unsqueeze(-1))
                cached_frames[f_index].K = self.K_slam
            
                points_k = points[k_index].contiguous()
                points_f = points[f_index].contiguous()
                feats_k = feats[k_index].contiguous()
                feats_f = feats[f_index].contiguous()
                idx_f2k_valid = idx_f2k_valids[f_index - 1]
                idx_f2k, idx_mask = project_and_refine(points_k, feats_k, feats_f, camera_poses[f_index], self.K_slam)
                
                idx_f2ks.append(idx_f2k)
                idx_masks.append(idx_mask)

                T_Wk = T_WCs_init[k_index:k_index+1] if key_T_WC is None else key_T_WC
                T_Wf = T_WCs_init[f_index:f_index+1]
                Cf = torch.gather(conf[f_index], 0, idx_f2k)
                Ck = conf[k_index]
                valid_Cf = Cf > self.cfg_track["C_conf"]
                valid_Ck = Ck > self.cfg_track["C_conf"]
                Qfk = torch.gather(match_conf[f_index], 0, idx_f2k)
                Qkk = match_conf[k_index]
                Qk = torch.sqrt(Qfk * Qkk)
                valid_Q = Qk > self.cfg_track["Q_conf"]
                valid_mask = idx_f2k_valid & valid_Cf & valid_Ck & valid_Q 
                match_frac = valid_mask.sum() / valid_mask.numel()
                if match_frac < self.cfg_track["min_match_frac"]:
                    frames.append(cached_frames[f_index])
                    frames_style.append(-1)
                    continue

                points_k_local = k_frame.X_canon
                points_f_local = f_frame.X_canon
                T_WCf, T_CkCf = self.inferenceSim3(
                    points_f_local, T_Wf, points_k_local, T_Wk,
                    ((Cf+Ck)/2).unsqueeze(-1), valid_mask.unsqueeze(-1),
                    idx_f2k, img_size, self.K_slam
                )

                if T_WCf is None:
                    frames.append(cached_frames[f_index])
                    frames_style.append(-1)
                    continue
                else:
                    T_WCs_init[f_index:f_index + 1] = T_WCf
                    T_WCf = pp.quat2unit(T_WCf)
                    T_CkCf = pp.quat2unit(T_CkCf)
                    cached_frames[f_index].T_WC = T_WCf
                    cached_frames[f_index].T_CkC = T_CkCf

                is_keyframe = self.check_keyframe(idx_f2k, valid_mask, idx_f2k_valid)
                if is_keyframe:
                    if self.args.use_dynamic:
                        f_indexs = [f_index]
                        k_indexs = [k_index] 
                        idx_k2f_use_extr, idx_valid_mask = compute_idx_i2j_batched(
                            points[f_indexs].contiguous().view(len(f_indexs), self.H_slam * self.W_slam, -1), 
                            camera_poses[k_indexs].contiguous(), 
                            self.K_slam, self.H_slam, self.W_slam
                        )
                        feat_map = compute_valid_mask(
                            local_points[k_indexs].contiguous(),
                            local_points[f_indexs].contiguous(),
                            feats[k_indexs].contiguous().view(len(k_indexs), self.H_slam * self.W_slam, -1), 
                            feats[f_indexs].contiguous().view(len(f_indexs), self.H_slam * self.W_slam, -1), 
                            idx_k2f_use_extr, idx_valid_mask, 
                            self.K_slam, self.H_slam, self.W_slam
                        )
                        self.last_dynamic_mask = feat_map[-1]
            
                    frames.append(cached_frames[f_index])
                    frames_style.append(1)
                    num_kf_idx = len(self.kf_idx) if self.kf_idx is not None else -1
                    if  num_kf_idx > 1:
                        for kf_idx, global_kf_idx in enumerate(self.kf_idx):
                            frame = self.keyframes[global_kf_idx].to(self.device)
                            frame.update_pointmap(local_points[kf_idx], conf[kf_idx].unsqueeze(-1))
                            self.keyframes[global_kf_idx] = frame
                        
                        self.factor_graph.add_factors(self.kf_idx, self.config["local_opt"]["min_match_frac"], self.K_slam,
                                                    points[:num_kf_idx], conf[:num_kf_idx], local_points[:num_kf_idx], camera_poses[:num_kf_idx], feats[:num_kf_idx], match_conf[:num_kf_idx])
                        with self.states.lock:
                            self.states.edges_ii[:] = self.factor_graph.ii.cpu().tolist()
                            self.states.edges_jj[:] = self.factor_graph.jj.cpu().tolist()

                        # 4. 全局优化
                        with torch.cuda.device(self.device):
                            self.factor_graph.solve_GN_calib()

                    self.last_dist = 0
                    break

                is_keyframe_map, dist = self.check_keyframe_map(idx_f2k, valid_mask)
                if is_keyframe_map:
                    frames.append(cached_frames[f_index])
                    frames_style.append(2)
                    self.last_dist = dist
                    continue

                frames.append(cached_frames[f_index])
                frames_style.append(0)
            
            cached_frames = cached_frames[f_index + 1:]
        else:
            cached_frames = []

        return frames, frames_style, idx_f2ks, idx_masks, dynamic_mask, last_keyframe_id, cached_frames
        
    def check_keyframe(self, idx_f2k, valid_mask, valid_match_k):
        n_valid = valid_mask.sum()
        match_frac_k = n_valid / valid_mask.numel()  # 关键帧中有效的匹配点占总数的百分比
        # 当前帧中与关键帧匹配的独一无二的点的数量占所有匹配的点的百分比，有可能会有一对多的情况
        unique_frac_f = (torch.unique(idx_f2k[valid_match_k]).shape[0] / valid_mask.numel())
        add_new_kf = min(match_frac_k, unique_frac_f) < self.cfg_track["match_frac_thresh"]
        return add_new_kf

    def check_keyframe_map(self, idx_f2k, valid_mask):
        uf = idx_f2k % self.W_slam
        vf = idx_f2k // self.W_slam
        uvf = torch.stack([uf, vf], dim=-1)

        uk, vk = torch.meshgrid(torch.arange(self.W_slam).to(self.device),
                                torch.arange(self.H_slam).to(self.device),
                                indexing="xy")
        uk = uk.reshape(-1)
        vk = vk.reshape(-1)
        uvk = torch.stack([uk, vk], dim=-1)
        dist = torch.norm((uvf - uvk).float(), dim=-1)[valid_mask.squeeze(-1)]

        dist_quantile = torch.quantile(dist, self.thres_keyframe).item()
        add_new_kf = (dist_quantile - self.last_dist) > self.min_displacement

        return add_new_kf, dist_quantile

    def inferenceSim3(self, Xf, T_WCf, Xk, T_WCk, conf_match, valid_match, idx_f2k, img_size, K):
        use_calib = self.config["use_calib"]
        Xf, Xf_cov, Xk, Xk_cov, T_WCf, T_WCk, meas_k, valid_meas_k = (
            self.get_points_poses(Xf, T_WCf, Xk, T_WCk, idx_f2k, img_size, K))
        try:
            T_WCf, T_CkCf, K = self.opt_pose_calib_sim3(
                Xf, Xf_cov, Xk, Xk_cov,
                T_WCf, T_WCk, conf_match, valid_match,
                meas_k,               # 关键帧的像素坐标及其对数深度
                valid_meas_k,         # 关键帧中深度大于阈值的像素mask
                idx_f2k,
                img_size,
                K
            )
        except Exception as e:
            print(f"Optimize failed: {e}")
            T_WCf = None
            T_CkCf = None
        return T_WCf, T_CkCf

    # 按照关键帧的顺序，返回匹配的点对，位姿，匹配点对的置信度，以及深度大于阈值的像素mask
    def get_points_poses(self, Xf, T_WCf, Xk, T_WCk, idx_f2k, img_size, K):
        meas_k = None             # 存储像素坐标及其深度
        valid_meas_k = None       # 深度大于阈值的像素mask

        # 根据内参重新获得相机坐标系点云，这里只用了预测点云的深度z坐标，其余xy坐标通过K的逆矩阵和像素网格获得
        Xf = constrain_points_to_ray(img_size, Xf[None], K).squeeze(0)
        Xk = constrain_points_to_ray(img_size, Xk[None], K).squeeze(0)

        Xf_cov = local_diag_cov_from_X1(Xf, self.H_slam, self.W_slam)
        Xk_cov = local_diag_cov_from_X1(Xk, self.H_slam, self.W_slam)

        # Setup pixel coordinates
        uv_k = get_pixel_coords(1, img_size, device=Xf.device, dtype=Xf.dtype)
        uv_k = uv_k.view(-1, 2)
        meas_k = torch.cat((uv_k, torch.log(Xk[..., 2:3])), dim=-1)
        # Avoid any bad calcs in log
        valid_meas_k = Xk[..., 2:3] > self.cfg_track["depth_eps"]
        meas_k[~valid_meas_k.repeat(1, 3)] = 0.0

        return Xf[idx_f2k], Xf_cov[idx_f2k], Xk, Xk_cov, T_WCf, T_WCk, meas_k, valid_meas_k
    
    # todo:这个地方的残差权重计算没搞清楚
    # 输入jacobian J， residual r， 以及权重，用高斯牛顿法计算更新量
    def solve(self, sqrt_info, r, J):
        whitened_r = sqrt_info * r
        robust_sqrt_info = sqrt_info * torch.sqrt(
            huber(whitened_r, k=self.cfg_track["huber"])
        )
        mdim = J.shape[-1]
        A = (robust_sqrt_info[..., None] * J).view(-1, mdim)  # dr_dX
        b = (robust_sqrt_info * r).view(-1, 1)  # z-h
        H = A.T @ A
        g = -A.T @ b
        cost = 0.5 * (b.T @ b).item()

        L = torch.linalg.cholesky(H, upper=False)
        tau_j = torch.cholesky_solve(g, L, upper=False).view(1, -1)

        return tau_j, cost

    # 输入一一对应的点对（不同坐标系），初始估计位姿，按照关键帧像素顺序的特征匹配权重，以及匹配mask
    # 用LM算法计算相对位姿改变
    def opt_pose_ray_dist_sim3(self, Xf, Xk, T_WCf, T_WCk, Qk, valid):
        # 计算LM中每个误差项的权重
        last_error = 0
        sqrt_info_ray = 1 / self.cfg_track["sigma_ray"] * valid * torch.sqrt(Qk)
        sqrt_info_dist = 1 / self.cfg_track["sigma_dist"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_ray.repeat(1, 3), sqrt_info_dist), dim=1)

        # Solving for relative pose without scale!
        T_CkCf = T_WCk.Inv().mul(T_WCf)

        # 在关键帧坐标系中，计算关键帧的所有归一化的点和模长，存储在rd_K中
        # Precalculate distance and ray for obs k
        rd_k = point_to_ray_dist(Xk, jacobian=False)

        # 迭代优化位姿
        old_cost = float("inf")
        for step in range(self.cfg_track["max_iters"]):
            # 将Xf变换到关键帧坐标系
            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            # 对关键帧坐标系中的Xf_Ck归一化，并且计算模长
            rd_f_Ck, drd_f_Ck_dXf_Ck = point_to_ray_dist(Xf_Ck, jacobian=True)
            # r = z-h(x)
            # 损失计算为方向损失和模损失
            r = rd_k - rd_f_Ck
            # 计算损失对相似变换的Jacobian
            J = -drd_f_Ck_dXf_Ck @ dXf_Ck_dT_CkCf

            tau_ij_sim3, new_cost = self.solve(sqrt_info, r, J)
            T_CkCf = T_CkCf.add(tau_ij_sim3)
            T_CkCf = pp.quat2unit(T_CkCf)
            if check_convergence(
                step,
                self.cfg_track["rel_error"],
                self.cfg_track["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3,
            ):
                break
            old_cost = new_cost

            if step == self.cfg_track["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk.mul(T_CkCf)

        return T_WCf, T_CkCf

    # 输入一一对应的点对（不同坐标系），初始估计位姿，按照关键帧像素顺序的特征匹配权重
    # 关键帧的像素坐标及其深度， 关键帧中深度大于阈值的像素mask
    # 内参矩阵和图像大小
    # 用LM算法计算相对位姿改变
    def opt_pose_calib_sim3(self, Xf, Xf_cov, Xk, Xk_cov,
                            T_WCf, T_WCk, Qk, valid, meas_k, valid_meas_k, idx_f2k, img_size, K):
        last_error = 0
        sqrt_info_pixel = 1 / self.cfg_track["sigma_pixel"] * valid * torch.sqrt(Qk)
        sqrt_info_depth = 1 / self.cfg_track["sigma_depth"] * valid * torch.sqrt(Qk)
        sqrt_info = torch.cat((sqrt_info_pixel.repeat(1, 2), sqrt_info_depth), dim=1)

        T_CkCf = T_WCk.Inv().mul(T_WCf)

        old_cost = float("inf")
        for step in range(self.cfg_track["max_iters"]):
            if self.optimize_focal:
                u = idx_f2k % self.W_slam
                v = idx_f2k // self.W_slam
                uv = torch.stack([u, v], dim=-1)

                dXox_f = -(uv[..., 0] - K[0, 2]) / (K[0, 0] ** 2) * Xf[..., 2]
                dXoy_f = -(uv[..., 1] - K[1, 2]) / (K[1, 1] ** 2) * Xf[..., 2]
                dXoz_f = torch.zeros_like(dXoy_f)
                dXf_f = torch.stack([dXox_f, dXoy_f, dXoz_f], dim=-1).unsqueeze(-1)

                Xf = backproject(uv, Xf[..., 2:3], K)
            else:
                dXf_f = torch.zeros((Xf.shape[0], 3, 1), dtype=torch.float32).to(self.device)

            Xf_Ck, dXf_Ck_dT_CkCf = act_Sim3(T_CkCf, Xf, jacobian=True)
            dXf_Ck_d_Xf = T_CkCf.matrix()[:, :3, :3].expand(Xf_Ck.shape[0], -1, -1)
            # todo：此处valid_proj要用吗
            pzf_Ck, dpzf_Ck_dXf_Ck, valid_proj = project_calib(
                Xf_Ck,
                K,
                img_size,
                jacobian=True,
                border=self.cfg_track["pixel_border"],
                z_eps=self.cfg_track["depth_eps"],
                dXf_Ck_d_f= (dXf_Ck_d_Xf @ dXf_f)
            )

            # 计算协方差
            if self.covariance_filter:
                R_CkCf = T_CkCf.matrix()[:, :3, :3]
                Xfk_cov = R_CkCf @ Xf_cov @ R_CkCf.transpose(-1, -2)
                X, Y, Z = Xf_Ck.unbind(-1)
                o = torch.zeros_like(X)
                JCOV = torch.stack([K[0,0]/Z, o, -K[0,0]*X/(Z**2),
                                 o, K[1,1]/Z, -K[1,1]*Y/(Z**2),
                                 o, o, 1/Z], dim=-1).reshape(-1, 3, 3)
                pixfk_cov = JCOV @ Xfk_cov @ JCOV.transpose(-1, -2)
                pixfk_cov_det = torch.det(pixfk_cov)
                valid_cov = pixfk_cov_det < max(torch.quantile(pixfk_cov_det, 0.9), 1)
                valid_cov = valid_cov[..., None]
            else:
                valid_cov = torch.ones_like(valid_meas_k)
            valid2 = valid_proj & valid_meas_k & valid_cov   # 在关键帧坐标系中，关键帧对应的点云和当前帧对应的点云深度均合理且范围也合理的mask
            sqrt_info2 = valid2 * sqrt_info

            r = meas_k - pzf_Ck
            # Jacobian
            J = -dpzf_Ck_dXf_Ck[..., :3] @ dXf_Ck_dT_CkCf
            if self.optimize_focal:
                J = torch.concatenate([J,-dpzf_Ck_dXf_Ck[..., 3:]], dim=-1)

            tau_ij_sim3, new_cost = self.solve(sqrt_info2, r, J)
            T_CkCf = pp.sim3(tau_ij_sim3[..., :7]).Exp().mul(T_CkCf)
            T_CkCf = pp.quat2unit(T_CkCf)

            if self.optimize_focal:
                K[0, 0] = K[0, 0] + tau_ij_sim3[0, -1]
                K[1, 1] = K[1, 1] + tau_ij_sim3[0, -1]

            if check_convergence(
                step,
                self.cfg_track["rel_error"],
                self.cfg_track["delta_norm"],
                old_cost,
                new_cost,
                tau_ij_sim3[..., :7],
            ):
                break
            old_cost = new_cost

            if step == self.cfg_track["max_iters"] - 1:
                print(f"max iters reached {last_error}")

        # Assign new pose based on relative pose
        T_WCf = T_WCk.mul(T_CkCf)
        return T_WCf, T_CkCf, K
