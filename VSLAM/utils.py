import torch
import einops
import VSLAM.utils_matching as matching
# from VSLAM.mvslam.retrieval_database import RetrievalDatabase
from VSLAM.mvslam.retrieval_database import RetrievalDatabase
from salad.eval import load_model 

from scipy.spatial.transform import Rotation
import kornia
import numpy as np
import os

def Matrix2Quan_Kornia(Twcs_pre):
    translation = Twcs_pre[:, :3, 3]
    rotation_matrix = Twcs_pre[:, :3, :3]
    # Kornia returns (w, x, y, z)
    quat_wxyz = kornia.geometry.conversions.rotation_matrix_to_quaternion(rotation_matrix)
    # Roll to (x, y, z, w)
    quat_xyzw = torch.roll(quat_wxyz, shifts=-1, dims=-1)
    return torch.cat([translation, quat_xyzw], dim=-1)

def Quan2Matrix_Kornia(Twcs_tum):
    """
    使用 Kornia 将 TUM 格式转换为 4x4 矩阵
    """
    N = Twcs_tum.shape[0]
    t = Twcs_tum[:, :3]
    quat_xyzw = Twcs_tum[:, 3:]
    
    # 将 (x, y, z, w) 调整为 Kornia 要求的 (w, x, y, z)
    quat_wxyz = torch.roll(quat_xyzw, shifts=1, dims=-1)
    
    # 转换为旋转矩阵 (N, 3, 3)
    R = kornia.geometry.conversions.quaternion_to_rotation_matrix(quat_wxyz)
    
    # 组装 4x4
    Twcs_mat = torch.eye(4, device=Twcs_tum.device).repeat(N, 1, 1)
    Twcs_mat[:, :3, :3] = R
    Twcs_mat[:, :3, 3] = t
    
    return Twcs_mat

def Matrix2Quan_Torch(Twcs_pre):
    """
    Converts a batch of 4x4 or 3x4 matrices to TUM format [tx, ty, tz, qx, qy, qz, qw]
    Args:
        Twcs_pre: (N, 4, 4) or (N, 3, 4) tensor
    Returns:
        Twcs_tum: (N, 7) tensor
    """
    # 1. Extract Translation (N, 3)
    translation = Twcs_pre[:, :3, 3]

    # 2. Extract Rotation Matrix (N, 3, 3)
    R = Twcs_pre[:, :3, :3]
    
    # 3. Vectorized Rotation Matrix to Quaternion (x, y, z, w)
    # Using the robust method to handle numerical stability
    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]

    tr = m00 + m11 + m22

    # We compute four possible cases for the quaternion components
    # and then choose the one that is numerically most stable.
    q = torch.stack([
        torch.stack([1 + tr, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, 1 + m00 - m11 - m22, m01 + m10, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m01 + m10, 1 - m00 + m11 - m22, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m02 + m20, m12 + m21, 1 - m00 - m11 + m22], dim=-1)
    ], dim=1) # Shape: (N, 4, 4)

    # Find the max diagonal element to avoid square roots of small/negative numbers
    idxs = torch.stack([tr, m00, m11, m22], dim=-1).argmax(dim=-1)
    
    # Gather the best case for each matrix in the batch
    quat = q[torch.arange(q.shape[0]), idxs]
    quat /= 2 * torch.sqrt(quat.norm(dim=-1, keepdim=True)) # Normalize

    # Reorder from (w, x, y, z) to (x, y, z, w) for TUM format/SciPy convention
    quat_xyzw = torch.cat([quat[:, 1:], quat[:, :1]], dim=-1)

    # 4. Concatenate to [tx, ty, tz, qx, qy, qz, qw]
    Twcs_tum = torch.cat([translation, quat_xyzw], dim=-1)
    
    return Twcs_tum

def Quan2Matrix_Torch(Twcs_tum):
    """
    将 TUM 格式 [tx, ty, tz, qx, qy, qz, qw] 转换为 4x4 变换矩阵
    Args:
        Twcs_tum: (N, 7) 形状的 Tensor
    Returns:
        Twcs_mat: (N, 4, 4) 形状的变换矩阵
    """
    N = Twcs_tum.shape[0]
    device = Twcs_tum.device
    
    # 1. 提取平移和四元数
    t = Twcs_tum[:, :3]        # (N, 3)
    q = Twcs_tum[:, 3:]        # (N, 4) -> [x, y, z, w]

    # 2. 四元数归一化（增强鲁棒性）
    q = q / (torch.norm(q, dim=-1, keepdim=True) + 1e-9)
    
    x, y, z, w = q[:, 0], q[:, 1], q[:, 2], q[:, 3]

    # 3. 计算旋转矩阵的分量
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, xw = x * y, x * z, x * w
    yz, yw, zw = y * z, y * w, z * w

    # 4. 组装旋转矩阵 (N, 3, 3)
    R = torch.stack([
        1 - 2 * (y2 + z2),     2 * (xy - zw),     2 * (xz + yw),
        2 * (xy + zw),     1 - 2 * (x2 + z2),     2 * (yz - xw),
        2 * (xz - yw),         2 * (yz + xw), 1 - 2 * (x2 + y2)
    ], dim=-1).reshape(N, 3, 3)

    # 5. 组装 4x4 变换矩阵
    Twcs_mat = torch.eye(4, device=device).unsqueeze(0).repeat(N, 1, 1) # (N, 4, 4)
    Twcs_mat[:, :3, :3] = R
    Twcs_mat[:, :3, 3] = t

    return Twcs_mat

def Matrix2Quan(Twcs_pre):
    translation = Twcs_pre[:, :3, 3]  # N x 3
    rotation_matrix = Twcs_pre[:, :3, :3]  # N x 3 x 3
    
    quaternions_list = []
    # Move to CPU once if the data is on GPU for Scipy compatibility
    rot_np = rotation_matrix.detach().cpu().numpy()
    
    for i in range(rot_np.shape[0]):
        r = Rotation.from_matrix(rot_np[i])
        quat = r.as_quat()  # [x, y, z, w]
        quaternions_list.append(quat)
    
    # FIX: Convert list to a single numpy array first, then to torch tensor
    quaternions_np = np.stack(quaternions_list) 
    quaternions = torch.from_numpy(quaternions_np).to(Twcs_pre.device).to(Twcs_pre.dtype)
    
    # Concatenate into TUM format [tx, ty, tz, qx, qy, qz, qw]
    Twcs_tum = torch.cat([translation, quaternions], dim=1)  # N x 7
    return Twcs_tum

def Quan2Matrix(Twcs_tum):
    """
    将 TUM 格式 [tx, ty, tz, qx, qy, qz, qw] 转换为 4x4 变换矩阵
    参数:
        Twcs_tum: Tensor (N, 7)
    返回:
        Twcs_pre: Tensor (N, 4, 4)
    """
    device = Twcs_tum.device
    dtype = Twcs_tum.dtype
    N = Twcs_tum.shape[0]

    # 1. 拆分位移和四元数
    translation = Twcs_tum[:, :3].detach().cpu().numpy()  # N x 3
    quaternions = Twcs_tum[:, 3:].detach().cpu().numpy()  # N x 4

    # 2. 使用 Scipy 进行向量化转换（非常快）
    r = Rotation.from_quat(quaternions)
    rotation_matrices = r.as_matrix()  # 输出形状为 (N, 3, 3)

    # 3. 构造 4x4 矩阵
    # 先在 CPU 上构造，减少与 GPU 的频繁交互
    matrices = np.zeros((N, 4, 4), dtype=np.float32)
    matrices[:, :3, :3] = rotation_matrices
    matrices[:, :3, 3] = translation
    matrices[:, 3, 3] = 1.0

    # 4. 转回 Tensor
    Twcs_pre = torch.from_numpy(matrices).to(device).to(dtype)
    
    return Twcs_pre
# load mastr main model
def load_encoder(path=None, device="cuda:0"):
    # Legacy helper, not used by the Pi3/Pi3X pipeline; import only if called.
    from VSLAM.thirdparty.mast3r.mast3r.model import AsymmetricMASt3R
    weights_path = (
        "VSLAM/retrieval_models/encoder.pth"
        if path is None
        else path
    )
    model = AsymmetricMASt3R.from_pretrained(weights_path).to(device)
    return model

# # load mast3r retriever model
# def load_retriever(args, config, model, mast3r_model, keyframes, retriever_path=None, device="cuda:0"):
#     retriever_path = (
#         "VSLAM/retrieval_models/retrieval_trainingfree.pth"
#         if retriever_path is None
#         else retriever_path
#     )
#     retriever = RetrievalDatabase(args, config, retriever_path, keyframes, model, backbone=mast3r_model, device=device)
#     return retriever


def load_retriever(args, config, model_pi3, keyframes, salad_ckpt_path=None, device="cuda:0"):
    """
    加载基于 SALAD 的图像检索系统
    
    参数:
        args: 全局参数
        config: 配置文件
        model_pi3: 用于第二阶段精确几何匹配的模型 (原代码中的 model)
        keyframes: 关键帧列表引用
        salad_ckpt_path: SALAD 模型权重的路径
        device: 运行设备
    """
    # 1. 设置 SALAD 权重路径 (你可以根据实际情况修改默认路径)
    salad_ckpt_path = (
        "checkpoints/dino_salad.ckpt"
        if salad_ckpt_path is None
        else salad_ckpt_path
    )
    
    if not os.path.exists(salad_ckpt_path):
        raise FileNotFoundError(f"SALAD 权重文件未找到: {salad_ckpt_path}")

    # 2. 加载 SALAD 模型并设置为评估模式
    print(f"Loading SALAD model from {salad_ckpt_path}...")
    salad_model = load_model(salad_ckpt_path)
    salad_model = salad_model.to(device)
    salad_model.eval()  # 检索提取特征时必须开启 eval 模式

    # 3. 实例化我们刚刚写好的、基于 SALAD 的纯净版 RetrievalDatabase
    retriever = RetrievalDatabase(
        args=args, 
        config=config, 
        keyframes=keyframes, 
        model_pi3=model_pi3, 
        salad_model=salad_model, 
        device=device
    )
    
    print("SALAD Retriever loaded successfully.")
    return retriever

# input: 3 h w [-1,1]
# output h w 3 [0, 1]
def inverse_normalize(tensor):
    device = tensor.device
    mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    restored_img = tensor.clamp(0, 1)  # 确保值在 [0, 1] 范围内
    restored_img = restored_img.permute(1, 2, 0)
    return restored_img

