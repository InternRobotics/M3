import torch
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
import cv2

# 假设这些是你项目中已有的模块
from VSLAM.utils_matching import match

# ==========================================
# 工具函数
# ==========================================

def inverse_normalize(tensor):
    """用于精确匹配前的反归一化"""
    device = tensor.device
    mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
    std = torch.tensor([0.5, 0.5, 0.5]).to(device)
    tensor = tensor * std[:, None, None] + mean[:, None, None]
    restored_img = tensor.clamp(0, 1)  # 确保值在 [0, 1] 范围内
    restored_img = restored_img.permute(1, 2, 0)
    return restored_img

def preprocess_for_salad(image_tensor, target_size=224):
    """
    专门为 SALAD 准备的图像预处理和 Resize 函数。
    image_tensor: 形状为 [3, H, W] 的 torch.Tensor，取值范围假设在 [0, 1]
    返回: 形状为 [1, 3, target_size, target_size] 的 Tensor，并已做 ImageNet 归一化
    """
    # 1. Resize 处理
    # F.interpolate 需要 4D tensor [B, C, H, W]，所以先 unsqueeze(0) 加一个 Batch 维度
    image_4d = image_tensor.unsqueeze(0)
    resized = F.interpolate(
        image_4d,
        size=(target_size, target_size),
        mode='bilinear',
        align_corners=False
    )
    
    # 2. ImageNet 归一化
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # 提取第一帧并进行归一化，然后再加回 Batch 维度
    normalized_image = normalize(resized[0]).unsqueeze(0)
    
    return normalized_image

# ==========================================
# 核心检索类 (Hierarchical Retrieval)
# ==========================================

class RetrievalDatabase:
    def __init__(self, args, config, keyframes, model_pi3, salad_model, device="cuda"):
        """
        不再继承 Retriever，直接初始化为独立的数据库类。
        salad_model: 你外部加载好的 SALAD 模型实例
        """
        self.keyframes = keyframes
        self.model_pi3 = model_pi3
        self.salad_model = salad_model
        
        self.config = config
        self.args = args
        self.device = device

        # 匹配相关参数
        self.min_window_number = 12
        self.max_window_number = 24
        self.accurate_loop_closure_number = 12
        self.is_accurate_loop_closure = self.args.accurate_loop_closure

        self.kf_counter = 0   # 当前库中有几个关键帧
        self.global_descriptors = [] # 用于存放历史关键帧的 SALAD 特征向量 [1, D]

    def extract_salad_feature(self, keyframe_id):
        """使用 SALAD 提取单张图片的全局特征"""
        with torch.no_grad():
            frame = self.keyframes[keyframe_id].to(self.device)
            image_tensor = inverse_normalize(frame.img).permute(2, 0, 1)
            # 将 [3, H, W] 转为 SALAD 需要的 [1, 3, 224, 224] 归一化 Tensor
            processed_img = preprocess_for_salad(image_tensor, target_size=224).to(self.device)
            # 提取特征，假设输出形状为 [1, D]
            desc = self.salad_model(processed_img)
            return desc

    def update(self, keyframe_id, k):
        """
        增加当前帧特征，并查询候选回环帧
        current_image_tensor: 当前帧图像 [3, H, W]
        """
        # 1. 提取当前帧 SALAD 全局特征
        query_desc = self.extract_salad_feature(keyframe_id)

        database_size = len(self.global_descriptors)
        topk_image_inds = []

        if database_size > 0:
            if k > 0:
                # 2. 将历史所有特征拼接成 [N, D] 的矩阵
                db_tensor = torch.cat(self.global_descriptors, dim=0)
                
                # 3. 计算当前帧与历史帧的 L2 距离
                distances = torch.linalg.norm(db_tensor - query_desc, dim=1)

                # self.sim_Graph.plot_similarity_heatmap(False)
                if (database_size < self.min_window_number) or (not self.is_accurate_loop_closure):
                    topk_images = torch.topk(distances, min(k, database_size), largest=False)
                    topk_image_inds = (topk_images.indices).tolist()
                else:
                    topk_images = torch.topk(distances, min(k, database_size), largest=False)
                    topk_image_inds_retrival = (topk_images.indices).tolist()

                    need_accurate_loop_closure = (database_size - min(topk_image_inds_retrival)) > self.accurate_loop_closure_number
                    if need_accurate_loop_closure:
                        num_candidates = min(self.max_window_number - 1, database_size)
                        _, candidate_indices = torch.topk(distances, num_candidates, largest=False)
                        candidate_indices = candidate_indices.tolist()
                        topk_image_inds = self.accurate_loop_closure(keyframe_id, candidate_indices, k)
                    else:
                        topk_image_inds = topk_image_inds_retrival
            else:
                topk_image_inds = []
                
                # target_idx = keyframe_id - 1
                # if target_idx not in topk_image_inds:
                #     topk_image_inds = [target_idx] + topk_image_inds[:k-1]

        # 7. 将当前帧的特征加入数据库，供后续查询
        self.global_descriptors.append(query_desc)
        self.kf_counter += 1

        return topk_image_inds

    def accurate_loop_closure(self, keyframe_id, candidate_indices, k):
        """第二阶段：利用 pi3 模型进行基于点和描述子的几何校验"""
        idxs_all = [keyframe_id] + candidate_indices

        views = {}
        kfs = [self.keyframes[index].to(self.device) for index in idxs_all]
        
        # 准备图像输入
        images = [inverse_normalize(frame.img).permute(2, 0, 1) for frame in kfs]
        images = torch.stack(images, dim=0)[None].contiguous()
        views["images"] = images
        
        # 准备内参
        if self.args.use_calib_for_inference:
            intrinsics = [frame.K for frame in kfs]
            intrinsics = torch.stack(intrinsics, dim=0)[None].contiguous()
            views["intrinsics"] = intrinsics

        # pi3 推理
        dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
            res = self.model_pi3(views)
        
        points, feats = res['points'][0], res['desc']

        # i 对应所有候选帧，j 对应当前帧 (索引为0)
        i_indexs = list(range(1, len(idxs_all)))
        j_indexs = [0] * (len(idxs_all) - 1)

        try:
            pts_i_batch = points[i_indexs].contiguous()
            pts_j_batch = points[j_indexs].contiguous()
            fts_i_batch = feats[i_indexs].contiguous()
            fts_j_batch = feats[j_indexs].contiguous()
        except:
            print(i_indexs, points)
        
        
        # 执行稠密匹配打分
        _, valid_match_j = match(self.config, pts_i_batch, pts_j_batch, fts_i_batch, fts_j_batch, None)

        # 计算成功匹配点的百分比
        match_percentage = valid_match_j.sum(-1) / valid_match_j[0].numel()
        
        # 按百分比降序排列
        loop_idxs = torch.argsort(match_percentage, descending=True)
        
        # 映射回原始的候选帧 ID
        topk_image_inds = [candidate_indices[i] for i in loop_idxs.tolist()]

        return topk_image_inds[:k]