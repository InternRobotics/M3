import torch
import torch.nn.functional as F

@torch.no_grad()
def local_diag_cov_from_X1(
    X1: torch.Tensor,     # (H*W, 3) 3D点云(任意坐标系)，展平顺序为行优先
    H: int, W: int,       # 图像尺寸
    win: int = 5,         # 邻域窗口(奇数)，如 3/5/7
    valid: torch.Tensor | None = None,  # (H*W,) 可选有效掩码；None则自动: finite & z>0
    var_floor: float = 1e-12            # 方差下限，避免数值为负/零
) -> torch.Tensor:
    """
    返回: Sigma (H*W, 3, 3) —— 仅包含对角项的局部协方差矩阵
          diag = [Var(X), Var(Y), Var(Z)]，其余为 0
    """
    device = X1.device
    assert X1.shape[0] == H * W and X1.shape[1] == 3, "X1应为(H*W, 3)"
    pad = win // 2

    # 重塑为 (H, W, 3)
    Xv = X1.view(H, W, 3)

    # 有效掩码处理
    if valid is None:
        valid_hw1 = (torch.isfinite(Xv).all(-1) & (Xv[..., 2] > 0)).float().unsqueeze(-1)
    else:
        assert valid.shape[0] == H * W
        valid_hw1 = valid.view(H, W, 1).float()

    # 盒滤波内部函数（用 avg_pool2d 实现）
    def boxfilter(img_hwC):
        # (H, W, C) -> (1, C, H, W)
        x = img_hwC.permute(2, 0, 1).unsqueeze(0)
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        y = F.avg_pool2d(x, kernel_size=win, stride=1)
        # (1, C, H, W) -> (H, W, C)
        return y.squeeze(0).permute(1, 2, 0)

    # 计算均值和二阶矩以获得方差
    denom = boxfilter(valid_hw1).clamp_min(1e-9)
    mean = boxfilter(Xv * valid_hw1) / denom             # E[X]
    ex2 = boxfilter((Xv * Xv) * valid_hw1) / denom       # E[X^2]
    var = (ex2 - mean * mean).clamp_min(var_floor)       # Var = E[X^2] - E[X]^2

    # 组装对角协方差 (H*W, 3, 3)
    N = H * W
    Sigma = torch.zeros(N, 3, 3, device=device)
    var_flat = var.view(N, 3)

    # 填充对角线项
    Sigma[:, 0, 0] = var_flat[:, 0]   # Var(X)
    Sigma[:, 1, 1] = var_flat[:, 1]   # Var(Y)
    Sigma[:, 2, 2] = var_flat[:, 2]   # Var(Z)

    return Sigma

@torch.no_grad()
def local_diag_cov_from_X1_batched(
    X1: torch.Tensor,     # (B, H*W, 3) 
    H: int, W: int,       
    win: int = 5,         
    valid: torch.Tensor | None = None,  # (B, H*W)
    var_floor: float = 1e-12            
) -> torch.Tensor:
    """
    支持 Batch 处理。
    返回: Sigma (B, H*W, 3, 3) 对角协方差矩阵
    """
    B, N, _ = X1.shape
    device = X1.device
    pad = win // 2

    # 1. 重塑为图像空间 (B, H, W, 3)
    Xv = X1.view(B, H, W, 3)

    # 2. 处理有效掩码 (B, H, W, 1)
    if valid is None:
        valid_hw1 = (torch.isfinite(Xv).all(-1) & (Xv[..., 2] > 0)).float().unsqueeze(-1)
    else:
        valid_hw1 = valid.view(B, H, W, 1).float()

    # 3. 盒滤波（适配 Batch）
    def boxfilter(img_bhwC):
        # img_bhwC: (B, H, W, C) -> (B, C, H, W)
        x = img_bhwC.permute(0, 3, 1, 2)
        # padding 模式针对 Batch 进行处理
        x = F.pad(x, (pad, pad, pad, pad), mode='reflect')
        # avg_pool2d 会自动处理第 0 维的 Batch
        y = F.avg_pool2d(x, kernel_size=win, stride=1)
        return y.permute(0, 2, 3, 1)  # (B, H, W, C)

    # 4. 计算均值和方差
    denom = boxfilter(valid_hw1).clamp_min(1e-9)
    mean  = boxfilter(Xv * valid_hw1) / denom               # E[X]
    ex2   = boxfilter((Xv * Xv) * valid_hw1) / denom        # E[X^2]
    var   = (ex2 - mean * mean).clamp_min(var_floor)        # Var (B, H, W, 3)

    # 5. 组装对角协方差 (B, N, 3, 3)
    # 使用 scatter 或 直接切片赋值
    Sigma = torch.zeros(B, N, 3, 3, device=device, dtype=X1.dtype)
    var_flat = var.view(B, N, 3)

    # 使用 diagonal 视图进行快速批量赋值
    # Sigma.diagonal(dim1=-2, dim2=-1) 返回的是 (B, N, 3) 的视图
    Sigma.diagonal(dim1=-2, dim2=-1).copy_(var_flat)

    return Sigma