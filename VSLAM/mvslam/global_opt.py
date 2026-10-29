import pypose as pp
import torch
import torch.nn.functional as F
from VSLAM.SharedKeyframes import SharedKeyframes
from VSLAM.mvslam.geometry import constrain_points_to_ray
from VSLAM.utils import inverse_normalize
import mast3r_slam_backends
from VSLAM.component_ba import componentwise_calib
from VSLAM.utils import (
    inverse_normalize, 
    Matrix2Quan, Quan2Matrix, 
    Matrix2Quan_Torch, Quan2Matrix_Torch,
    Matrix2Quan_Kornia, Quan2Matrix_Kornia
)
from VSLAM.utils_matching import match_pi3, match
from einops import rearrange

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

def project_and_refine_batched(
    pts_w,          # (N, H, W, 3) 源点云
    desc_1,         # (N, H, W, C) 源图像特征
    desc_2,         # (N, H, W, C) 目标图像特征
    c2w_j,          # (N, 4, 4) 目标相机位姿
    K_j,            # (3, 3) 目标相机内参 (假设 Batch 内参共享)
    search_radius=2 # 2D 搜索半径
):
    """
    Batched version of Coarse-to-Fine matching.
    """
    N, H, W, C = desc_1.shape
    device = pts_w.device
    M = H * W # 每张图的像素数

    # 1. 数据重塑 (Reshape for Batch Processing)
    # pts: (N, M, 3)
    # desc: (N, M, C)
    pts_w_flat = pts_w.view(N, M, 3)
    desc_1_flat = desc_1.view(N, M, C)
    
    # 为了方便 Gather，我们将目标特征完全扁平化为 (N*M, C)
    # 这样我们可以通过 "Batch偏移 + 局部索引" 来定位
    desc_2_total_flat = desc_2.view(-1, C) 

    # 2. 几何投影 (Coarse Stage) - Batched
    # 计算 World -> Camera 变换 (N, 4, 4)
    w2c_j = torch.inverse(c2w_j)
    R = w2c_j[:, :3, :3] # (N, 3, 3)
    t = w2c_j[:, :3, 3].unsqueeze(1) # (N, 1, 3)

    # 批量投影: (N, M, 3) @ (N, 3, 3).T + (N, 1, 3) -> (N, M, 3)
    pts_c = pts_w_flat @ R.transpose(1, 2) + t
    
    z = pts_c[..., 2]
    # 防止除零 (N, M)
    z_safe = torch.where(z > 1e-3, z, torch.ones_like(z) * 1e-3)
    
    # 投影公式 (假设内参 K 共享，自动广播)
    # u, v: (N, M)
    u_proj = (pts_c[..., 0] / z_safe) * K_j[0, 0] + K_j[0, 2]
    v_proj = (pts_c[..., 1] / z_safe) * K_j[1, 1] + K_j[1, 2]
    
    # 几何有效性掩码 (N, M)
    valid_mask_geo = (u_proj >= 0) & (u_proj < W) & \
                     (v_proj >= 0) & (v_proj < H) & \
                     (z > 1e-3)

    # 3. 2D 局部特征搜索 (Fine Stage) - Batched
    if search_radius > 0:
        # --- A. 构建搜索网格 ---
        r = torch.arange(-search_radius, search_radius + 1, device=device)
        dy, dx = torch.meshgrid(r, r, indexing='ij')
        offsets = torch.stack([dx.flatten(), dy.flatten()], dim=1) # (K*K, 2)
        num_offsets = offsets.shape[0]

        # --- B. 获取搜索中心 ---
        u_init = torch.round(u_proj).long() # (N, M)
        v_init = torch.round(v_proj).long()

        # --- C. 生成邻居坐标 ---
        # 扩展维度以便广播:
        # center: (N, M, 1, 2)
        # offsets: (1, 1, K*K, 2)
        center_coords = torch.stack([u_init, v_init], dim=-1).unsqueeze(2)
        neighbor_coords = center_coords + offsets.view(1, 1, num_offsets, 2) # (N, M, K*K, 2)

        # 边界 Clamp (N, M, K*K)
        u_nb = torch.clamp(neighbor_coords[..., 0], 0, W - 1)
        v_nb = torch.clamp(neighbor_coords[..., 1], 0, H - 1)

        # --- D. 计算全局索引 (Global Indexing) ---
        # 这是 Batched 版本的核心：
        # 局部索引 (0~HW-1) 只能在当前图内索引
        # 全局索引 = Batch_ID * HW + 局部索引
        local_indices = v_nb * W + u_nb # (N, M, K*K)
        
        # 生成 Batch 偏移量: [0, HW, 2HW, ...] shape (N, 1, 1)
        batch_offsets = (torch.arange(N, device=device) * M).view(N, 1, 1)
        
        # global_indices 指向 desc_2_total_flat (N*M, C)
        global_indices = local_indices + batch_offsets 

        # --- E. 提取特征 ---
        # (N, M, K*K, C)
        desc_neighbors = desc_2_total_flat[global_indices.view(-1)].view(N, M, num_offsets, C)

        # --- F. 计算相似度 ---
        # desc_1: (N, M, C) -> (N, M, 1, C)
        # desc_nb: (N, M, K*K, C)
        # Output: (N, M, K*K)
        # 使用 einsum 更加清晰: n batch, m points, k neighbors, c channels
        similarity = torch.einsum('nmc,nmkc->nmk', desc_1_flat, desc_neighbors)

        # --- G. 选取最佳匹配 ---
        best_nb_idx = torch.argmax(similarity, dim=2) # (N, M)
        
        # 获取最佳匹配的局部索引 (N, M)
        # 使用 gather 从 local_indices 中取出最佳的那个索引
        # local_indices: (N, M, K*K), best_nb_idx: (N, M) -> 扩展为 (N, M, 1)
        final_indices_local = torch.gather(local_indices, 2, best_nb_idx.unsqueeze(2)).squeeze(2)

    else:
        # 无搜索，直接返回投影索引
        u_idx = torch.round(torch.clamp(u_proj, 0, W - 1)).long()
        v_idx = torch.round(torch.clamp(v_proj, 0, H - 1)).long()
        final_indices_local = v_idx * W + u_idx

    # 返回形状 (N, H*W) 的局部索引
    return final_indices_local, valid_mask_geo

def estimate_intrinsic_ransac_joint(pts_w, idx_i2j, valid_mask, c2w_j, W_j, H_j, 
                                    K_initial=None, iters=1000, threshold=1.0):
    """
    多视图联合 RANSAC 内参估计
    
    Args:
        pts_w: (M, 3) 世界坐标系下的共享点云
        idx_i2j: (N, M) N 个视角下的像素索引映射 (-1 表示无效)
        c2w_j: (N, 4, 4) N 个视角的相机外参
        W_j, H_j: 图像分辨率 (假设所有视图分辨率一致)
        K_initial: (3, 3) 初始内参矩阵
    """
    N, M = idx_i2j.shape
    device = pts_w.device

    # --- 1. 批量预处理 (Vectorized Pre-processing) ---
    
    # A. 计算所有视角的 w2c: (N, 4, 4)
    w2c_j = torch.inverse(c2w_j)
    R = w2c_j[:, :3, :3] # (N, 3, 3)
    t = w2c_j[:, :3, 3:] # (N, 3, 1)

    # B. 将点云扩展并转到所有相机坐标系下
    # pts_w: (M, 3) -> (1, M, 3)
    # R: (N, 3, 3) -> (N, 1, 3, 3) 用广播或者矩阵乘法
    # 这里使用 batch matrix multiplication 的技巧:
    # P_c = (R @ P_w.T + t).T
    
    # 也就是 P_c[i] = P_w @ R[i].T + t[i].T
    # 我们可以先把 P_w 扩充为 (N, M, 3)，但为了省显存，我们可以分步或者利用广播]
    pts_c = (pts_w.unsqueeze(0) @ R.transpose(1, 2)) + t.transpose(1, 2) # (N, M, 3)

    # C. 提取深度并计算归一化坐标 (N, M, 2)
    z = pts_c[..., 2:3]
    xy_n_all = pts_c[..., :2] / (z + 1e-8) 

    # D. 还原所有视角的像素坐标 (N, M, 2)
    u_all = (idx_i2j % W_j).float()
    v_all = (idx_i2j // W_j).float()
    pix_all = torch.stack([u_all, v_all], dim=-1)
    
    # 使用 mask 提取所有有效数据，合并成一个巨大的列表
    # Total_Valid_Points = K
    pool_xy_n = xy_n_all[valid_mask] # (K, 2)
    pool_pix  = pix_all[valid_mask]  # (K, 2)

    num_points = pool_xy_n.shape[0]
    if num_points < 6: # 至少需要几个点，多视角下通常点很多
        return None, None

    # --- 3. 初始引导 (Guided Setup) ---
    best_inliers_mask = None
    max_inliers = 0
    weights = torch.ones(num_points, device=device)

    if K_initial is not None:
        with torch.no_grad():
            i_fx, i_fy = K_initial[0, 0], K_initial[1, 1]
            i_cx, i_cy = K_initial[0, 2], K_initial[1, 2]
            
            # 计算所有点在初始 K 下的误差
            pred_u = pool_xy_n[:, 0] * i_fx + i_cx
            pred_v = pool_xy_n[:, 1] * i_fy + i_cy
            
            diff = torch.stack([pred_u - pool_pix[:, 0], pred_v - pool_pix[:, 1]], dim=-1)
            error = torch.norm(diff, dim=-1)
            
            inliers_mask = error < threshold
            max_inliers = inliers_mask.sum()
            best_inliers_mask = inliers_mask
            
            # 快速退出：如果初始值已经非常好 (>95% inliers)，直接微调
            if max_inliers > num_points * 0.95:
                return refine_k(pool_xy_n, pool_pix, best_inliers_mask, device), best_inliers_mask
            
            # 引导权重
            weights = 1.0 / (error + 0.1)

    # --- 4. 联合 RANSAC 循环 ---
    # 注意：这里的采样是在所有视角的所有点中随机采 4 个
    # 这意味着我们可能同时利用了 View 1 的左上角和 View 5 的右下角来解算 K
    
    for _ in range(iters):
        # 多项式采样
        indices = torch.multinomial(weights, 4)
        sample_xy_n = pool_xy_n[indices]
        sample_pix  = pool_pix[indices]

        try:
            # 解算 K (同前)
            t_fx, t_cx = solve_lstsq(sample_xy_n[:, 0], sample_pix[:, 0:1])
            t_fy, t_cy = solve_lstsq(sample_xy_n[:, 1], sample_pix[:, 1:2])
            
            # 合理性检查
            if K_initial is not None:
                if abs(t_fx - i_fx) > i_fx * 0.3: continue # 更严格的检查

            # 计算 Inliers (针对全量数据池)
            pred_u_iter = pool_xy_n[:, 0] * t_fx + t_cx
            pred_v_iter = pool_xy_n[:, 1] * t_fy + t_cy
            
            dist = (pred_u_iter - pool_pix[:, 0])**2 + (pred_v_iter - pool_pix[:, 1])**2
            cur_inliers_mask = dist < (threshold ** 2)
            num_cur_inliers = cur_inliers_mask.sum()

            if num_cur_inliers > max_inliers:
                max_inliers = num_cur_inliers
                best_inliers_mask = cur_inliers_mask
                
                # 动态提前退出
                if max_inliers > num_points * 0.98: break

        except RuntimeError:
            continue

    # --- 5. 最终精修 ---
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
def compute_idx_i2j_batched(pts_w, c2ws, K, H, W):
    """
    计算世界点云投影到 N 个相机视角的索引 (Batch Version)。
    支持每个视角有独立的内参。
    
    Args:
        pts_w: (M, 3) 世界坐标系下的点云 (M = Hi * Wi)
        c2ws:  (N, 4, 4) N 个视角的 Camera-to-World 矩阵
        K:     (N, 3, 3) 每个视角独立的相机内参矩阵
        H, W:  int 图像分辨率
        
    Returns:
        idx_batched: (N, M) 每个视角下的扁平化索引
        mask_batched: (N, M) 每个视角下的有效性掩码
    """
    # 获取维度信息
    N = c2ws.shape[0]  # batch size (views)
    M = pts_w.shape[0] # number of points
    device = pts_w.device

    # 1. 批量坐标系转换: World -> Cameras
    # 计算所有相机的 w2c 矩阵: (N, 4, 4)
    w2cs = torch.inverse(c2ws)
    
    # 提取旋转和平移
    R = w2cs[:, :3, :3] # (N, 3, 3)
    t = w2cs[:, :3, 3]  # (N, 3)

    # 批量投影: pts_c = pts_w @ R^T + t
    # (N, M, 3) = (1, M, 3) @ (N, 3, 3) + (N, 1, 3)
    pts_c = pts_w.unsqueeze(0) @ R.transpose(1, 2) + t.unsqueeze(1)
    
    # 2. 投影到像素平面 (Perspective Projection)
    x_c = pts_c[..., 0] # (N, M)
    y_c = pts_c[..., 1] # (N, M)
    z_c = pts_c[..., 2] # (N, M)
    
    # 防止除零
    z_safe = torch.where(z_c > 1e-3, z_c, torch.ones_like(z_c) * 1e-3)
    
    # --- 修改部分：处理 (N, 3, 3) 的内参 K ---
    # 提取每个视角的参数并增加维度以便广播: (N, 1)
    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)
    
    # 利用广播机制计算 u, v: (N, M) = (N, M) * (N, 1) + (N, 1)
    u = (x_c / z_safe) * fx + cx
    v = (y_c / z_safe) * fy + cy
    # ---------------------------------------
    
    # 3. 计算有效性掩码 (N, M)
    valid_mask = (u >= 0) & (u < W - 0.5) & \
                 (v >= 0) & (v < H - 0.5) & \
                 (z_c > 1e-3) 
    
    # 4. 边界截断 (Clamping)
    u_clamped = torch.clamp(u, 0, W - 1)
    v_clamped = torch.clamp(v, 0, H - 1)
    
    # 5. 生成索引 (N, M)
    u_idx = torch.round(u_clamped).long()
    v_idx = torch.round(v_clamped).long()
    
    idx_batched = v_idx * W + u_idx
    
    return idx_batched, valid_mask

class FactorGraph:
    def __init__(self, args, config, model, frames: SharedKeyframes, device="cuda"):
        self.args = args
        self.model = model
        self.frames = frames
        self.device = device
        self.config = config
        self.cfg = config["local_opt"]
        # 与jj中相连的keyframe id
        self.ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 按照顺序排列的关键帧id， 例如 1 2 2 3 3 3 4 4 4 4之类的，与ii中相应位置构成因子图
        self.jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 在i坐标系，照j的顺序，给出i的匹配位置
        self.idx_ii2jj = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 在j坐标系，照i的顺序，给出j的匹配位置
        self.idx_jj2ii = torch.as_tensor([], dtype=torch.long, device=self.device)
        # 在i坐标系，按照j的顺序的有效的匹配对数（match函数给出的最初的匹配mask：考虑了损失和距离的匹配mask）
        self.valid_match_j = torch.as_tensor([], dtype=torch.bool, device=self.device)
        # 在j坐标系，按照i的顺序的有效的匹配对数（match函数给出的最初的匹配mask：考虑了损失和距离的匹配mask）
        self.valid_match_i = torch.as_tensor([], dtype=torch.bool, device=self.device)
        # 在i坐标系，按照j的顺序的特征预测分数
        self.Q_ii2jj = torch.as_tensor([], dtype=torch.float32, device=self.device)
        # 在j坐标系，按照i的顺序的特征预测分数
        self.Q_jj2ii = torch.as_tensor([], dtype=torch.float32, device=self.device)

        self.window_size = self.cfg["window_size"]

    # 向因子图中增加因子，匹配数量低于一定阈值的因子不会加进去
    def add_factors(self, kf_ids, min_match_frac, K, points, conf, local_points, camera_poses, feats, match_conf, is_reloc=False):
        
        frame_info = {}
        views = {}
    
        i_indexs = list(range(1, len(kf_ids)))
        j_indexs = [0] * (len(kf_ids) - 1)
        pts_i_batch = points[i_indexs].contiguous()
        pts_j_batch = points[j_indexs].contiguous()
        fts_i_batch = feats[i_indexs].contiguous()
        fts_j_batch = feats[j_indexs].contiguous()
        cam_i_batch = camera_poses[i_indexs].contiguous()
        cam_j_batch = camera_poses[j_indexs].contiguous()
        
        # 假设 match 函数支持 Batch 维度
        _, valid_match_j = match(self.config, pts_i_batch, pts_j_batch, fts_i_batch, fts_j_batch, None)
        _, valid_match_i = match(self.config, pts_j_batch, pts_i_batch, fts_j_batch, fts_i_batch, None)
        idx_i2j, valid_i2j_mask = project_and_refine_batched(pts_j_batch, fts_j_batch, fts_i_batch, cam_i_batch, K)
        idx_j2i, valid_j2i_mask = project_and_refine_batched(pts_i_batch, fts_i_batch, fts_j_batch, cam_j_batch, K)

        # 2. 计算出来在i和j坐标系，各自的匹配百分比
        batch_inds = torch.arange(idx_i2j.shape[0], device=idx_i2j.device)[
            :, None
        ].repeat(1, idx_i2j.shape[1])
        Qii, Qji = match_conf[i_indexs], match_conf[j_indexs]
        Qjj, Qij = match_conf[j_indexs], match_conf[i_indexs]
        
        Qj = torch.sqrt(Qii[batch_inds, idx_i2j] * Qji).unsqueeze(-1)
        Qi = torch.sqrt(Qjj[batch_inds, idx_j2i] * Qij).unsqueeze(-1)
        valid_match_j = valid_match_j.unsqueeze(-1)
        valid_match_i = valid_match_i.unsqueeze(-1)
        valid_i2j_mask = valid_i2j_mask.unsqueeze(-1)
        valid_j2i_mask = valid_j2i_mask.unsqueeze(-1)
        valid_Qj = Qj > self.cfg["Q_conf"]
        valid_Qi = Qi > self.cfg["Q_conf"]
        valid_j = valid_match_j & valid_Qj & valid_i2j_mask
        valid_i = valid_match_i & valid_Qi & valid_j2i_mask

        nj = valid_j.shape[1] * valid_j.shape[2]
        ni = valid_i.shape[1] * valid_i.shape[2]
        match_frac_j = valid_j.sum(dim=(1, 2)) / nj # i坐标系下的匹配百分比
        match_frac_i = valid_i.sum(dim=(1, 2)) / ni # j坐标系下的匹配百分比
        
        # 3. 去除无效的因子匹配：匹配分数低的边 且 时间上（指的是keyframe的local id）不相连的边
        # 在重定位上，如果retrival找到了上一个关键帧，则上一个关键帧一定不会被去除
        ii = kf_ids[1:]
        jj = kf_ids[0:1] * len(ii)
        ii_tensor = torch.as_tensor(ii, device=self.device)
        jj_tensor = torch.as_tensor(jj, device=self.device)
        # NOTE: Saying we need both edge directions to be above thrhreshold to accept either
        invalid_edges = torch.minimum(match_frac_j, match_frac_i) < min_match_frac
        consecutive_edges = ii_tensor == (jj_tensor - 1)
        invalid_edges = (~consecutive_edges) & invalid_edges
        # todo:如果至少有一个没匹配且处于重定位模式，则退出 这个是不是有点太严格?
        if invalid_edges.any() and is_reloc:
            return False, frame_info

        # 4. 保存因子
        valid_edges = ~invalid_edges
        ii_tensor = ii_tensor[valid_edges]
        jj_tensor = jj_tensor[valid_edges]
        idx_i2j = idx_i2j[valid_edges]
        idx_j2i = idx_j2i[valid_edges]
        valid_match_j = valid_j[valid_edges]
        valid_match_i = valid_i[valid_edges]
        Qj = Qj[valid_edges]
        Qi = Qi[valid_edges]

        self.ii = torch.cat([self.ii, ii_tensor])
        self.jj = torch.cat([self.jj, jj_tensor])
        self.idx_ii2jj = torch.cat([self.idx_ii2jj, idx_i2j])
        self.idx_jj2ii = torch.cat([self.idx_jj2ii, idx_j2i])
        self.valid_match_j = torch.cat([self.valid_match_j, valid_match_j])
        self.valid_match_i = torch.cat([self.valid_match_i, valid_match_i])
        self.Q_ii2jj = torch.cat([self.Q_ii2jj, Qj])
        self.Q_jj2ii = torch.cat([self.Q_jj2ii, Qi])

        added_new_edges = valid_edges.sum() > 0

        return added_new_edges, frame_info

    # 得到因子图中所有独一无二的帧，升序
    def get_unique_kf_idx(self):
        return torch.unique(torch.cat([self.ii, self.jj]), sorted=True)

    # 得到所有帧对应的点云，位姿，置信度
    def get_poses_points(self, unique_kf_idx):
        kfs = [self.frames[idx].to(self.device) for idx in unique_kf_idx]
        Xs = torch.stack([kf.X_canon for kf in kfs])
        Cs = torch.stack([kf.get_average_conf() for kf in kfs])
        T_WCs = pp.quat2unit(pp.Sim3(torch.stack([kf.T_WC.data for kf in kfs])))       
        return Xs, T_WCs, Cs, kfs[0].K

    # todo:这样让计算压力加倍
    # 对于某一个因子i和j，将j到i的匹配和i到j的匹配看做两个因子，因为最初的因子是j到i，所以将i到j加到j到i的后面
    def prep_two_way_edges(self):
        ii = torch.cat((self.ii, self.jj), dim=0)
        jj = torch.cat((self.jj, self.ii), dim=0)
        idx_ii2jj = torch.cat((self.idx_ii2jj, self.idx_jj2ii), dim=0)
        valid_match = torch.cat((self.valid_match_j, self.valid_match_i), dim=0)
        Q_ii2jj = torch.cat((self.Q_ii2jj, self.Q_jj2ii), dim=0)
        return ii, jj, idx_ii2jj, valid_match, Q_ii2jj

    def solve_GN_calib(self):
        pin = self.cfg["pin"]
        unique_kf_idx = self.get_unique_kf_idx()
        n_unique_kf = unique_kf_idx.numel()
        if n_unique_kf and (unique_kf_idx.min() < 0 or unique_kf_idx.max() >= len(self.frames)):
            raise RuntimeError(f"BA references uncommitted frames: {unique_kf_idx.tolist()}, stored={len(self.frames)}")
        if n_unique_kf <= pin:
            return

        Xs, T_WCs, Cs, K = self.get_poses_points(unique_kf_idx)

        # Constrain points to ray
        img_size = self.frames[0].img.shape[-2:]
        Xs = constrain_points_to_ray(img_size, Xs, K)

        ii, jj, idx_ii2jj, valid_match, Q_ii2jj = self.prep_two_way_edges()

        C_thresh = self.cfg["C_conf"]
        Q_thresh = self.cfg["Q_conf"]
        pixel_border = self.cfg["pixel_border"]
        z_eps = self.cfg["depth_eps"]
        max_iter = self.cfg["max_iters"]
        sigma_pixel = self.cfg["sigma_pixel"]
        sigma_depth = self.cfg["sigma_depth"]
        delta_thresh = self.cfg["delta_norm"]

        pose_data = T_WCs.data[:, 0, :]
        height, width = img_size

        assert torch.isfinite(pose_data).all(), "Nonfinite BA input poses"
        assert torch.isfinite(Xs).all(), "Nonfinite BA input points"
        assert idx_ii2jj.shape == valid_match.shape[:2], "BA index/mask shape mismatch"
        bad = ((idx_ii2jj < 0) | (idx_ii2jj >= Xs.shape[1])) & valid_match.squeeze(-1)
        assert not bad.any(), "Valid BA factor has out-of-range pixel index"
        componentwise_calib(mast3r_slam_backends.gauss_newton_calib,
            pose_data,
            Xs,
            Cs,
            K,
            ii,
            jj,
            idx_ii2jj,
            valid_match,
            Q_ii2jj,
            height,
            width,
            pixel_border,
            z_eps,
            sigma_pixel,
            sigma_depth,
            C_thresh,
            Q_thresh,
            max_iter,
            delta_thresh,
        )

        self.frames.update_T_WCs(T_WCs[pin:], unique_kf_idx[pin:])
