import pathlib
import os
import random
import time

import numpy as np
import pypose as pp
from tqdm import tqdm
import torch
import cv2
import json
from VSLAM.CameraTracker import CameraTracker
import VSLAM.mvslam.evaluate as eval
from VSLAM.ImageFrame import Mode, ImageFrame
import pathlib
import time
import json
import torch
import numpy as np
import pypose as pp
from tqdm import tqdm
from VSLAM.mvslam.geometry import constrain_points_to_ray

import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

def visualize_dense_correspondences(img_i, img_j, idx_i2j, id, num_draw=100):
    """
    Args:
        img_i: (3, Hi, Wi) Tensor
        img_j: (3, Hj, Wj) Tensor
        idx_i2j: (N,) Tensor, 存储的是 img_j 扁平化后的索引
    """
    # 1. 转换图像格式
    def to_numpy(img):
        if torch.is_tensor(img):
            img = img.permute(1, 2, 0).detach().cpu().numpy()
        # 归一化处理
        return img / 255.0 if img.max() > 1.0 else img

    im_i = to_numpy(img_i)
    im_j = to_numpy(img_j)
    
    # 获取尺寸
    hi, wi, _ = im_i.shape
    hj, wj, _ = im_j.shape

    # 2. 拼接画布
    # 创建一个能同时容纳两张图的画布
    max_h = max(hi, hj)
    total_w = wi + wj
    canvas = np.zeros((max_h, total_w, 3))
    canvas[:hi, :wi, :] = im_i
    canvas[:hj, wi:, :] = im_j

    # 3. 准备坐标数据
    # 生成图 i 的所有坐标 (y, x) -> (N, 2)
    # 注意：idx_i2j 也是展平的，所以我们需要图 i 的展平坐标
    grid_y, grid_x = torch.meshgrid(torch.arange(hi), torch.arange(wi), indexing='ij')
    pts_i_all = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(idx_i2j.device)

    # 4. 筛选有效匹配 (关键修改部分)
    # 过滤掉无效索引（比如 -1 或越界值）
    valid_mask = (idx_i2j >= 0) & (idx_i2j < hj * wj)
    valid_indices = torch.where(valid_mask)[0] # 得到的是图 i 中有效点的索引 (0 到 N-1)

    num_valid = len(valid_indices)
    
    if num_valid == 0:
        print(f"Warning: No valid correspondences found for id {id}")
        return

    if num_valid > num_draw:
        # 产生一个 0 到 num_valid-1 的随机排列，并取前 num_draw 个
        random_indices = torch.randperm(num_valid)[:num_draw]
        
        # 从有效索引中提取这些随机位置的索引
        final_selection_idx = valid_indices[random_indices]
    else:
        # 如果有效点数不足 num_draw，则全部绘制
        final_selection_idx = valid_indices

    # 5. 获取最终要画的坐标对
    # 图 i 的坐标 (x, y)
    p1 = pts_i_all[final_selection_idx].cpu().numpy()
    
    # 图 j 的坐标 (x, y) - 需要从 idx_i2j 中解码
    # 获取图 j 中对应的展平索引
    idx_j_flat = idx_i2j[final_selection_idx]
    
    # 解码为 (x, y)
    y_j = (idx_j_flat // wj).cpu().numpy()
    x_j = (idx_j_flat % wj).cpu().numpy()
    p2 = np.stack([x_j, y_j], axis=1)

    # 6. 绘图
    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    
    # 绘制连线
    for k in range(len(p1)):
        x1, y1 = p1[k]
        x2, y2 = p2[k]
        
        # 给图 j 的 x 坐标加上图 i 的宽度作为偏移
        x2_offset = x2 + wi 
        
        # 随机颜色方便区分
        color = np.random.rand(3,)
        
        plt.plot([x1, x2_offset], [y1, y2], c=color, lw=1.2, alpha=0.7)
        plt.scatter([x1, x2_offset], [y1, y2], c=[color], s=20, edgecolors='white', linewidths=0.5)

    plt.axis('off')
    plt.tight_layout()
    plt.title(f"Dense Correspondences (Uniform Sample): {len(p1)} points")
    plt.savefig(f"corrs/correspondences{id}.png", dpi=150)
    plt.close() # 关闭图形释放内存

def consume_dense_correspondence_rng(idx_i2j, height, width, num_draw=100):
    """Retain original CPU RNG progression, without image transfer or plotting."""
    idx_cpu = idx_i2j.detach().cpu()
    num_valid = int(((idx_cpu >= 0) & (idx_cpu < height * width)).sum())
    if num_valid > num_draw:
        torch.randperm(num_valid)[:num_draw]
    for _ in range(min(num_valid, num_draw)):
        np.random.rand(3,)


class Frontend:
    @torch.no_grad()
    def __init__(self, args, config, dataset, keyframes, states, model, device="cuda:0"):
        self.args = args
        self.config = config
        self.use_calib = config["use_calib"]
        self.device = device

        # args: camera intrinsics for slam
        self.dataset = dataset
        self.H_slam, self.W_slam = dataset.H_slam, dataset.W_slam
        self.K_slam = self.dataset.K_slam.to(device) if self.dataset.K_slam is not None else None

        # Shared variables
        self.keyframes = keyframes  # shared
        self.states = states  # shared
        self.model = model

        # initialize tracker and backend
        self.tracker = CameraTracker(self.args, self.config, self.model, self.H_slam, self.W_slam, self.K_slam,
                                  self.keyframes, self.states, self.device)
        # class domain variable
        self.frames_info = []
        self.frames_Twc_gt = []
        self.qframes = []
        self.is_test = {}
        
    def init_pose_per_frame(self, current_idx, Twc_gt=None):
        if current_idx == 0:
            T_WC = pp.identity_Sim3(1, device=self.device)
        else:
            T_WC = pp.Sim3(self.states.T_WC).to(self.device)
        if Twc_gt is not None:
            T_WC.data[:, :7] = Twc_gt.data[:, :7]
        T_WC = pp.quat2unit(T_WC)
        return T_WC

    @torch.no_grad()
    def run(self):
        seed = int(os.environ.get("ARTDECO_EXPERIMENT_SEED", "0"))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        print(f"Frontend experiment seed: {seed}", flush=True)
        self.lost_frame_ids = []
        frameID = 0
        is_end = False
        last_keyframe = None
        self.tracker.init_retrieval()
        bar = tqdm(total=len(self.dataset), desc=f"Frontend: ", position=0, leave=True)
        while True:
            # 1. read data
            try:
                original_image, info = self.dataset.getnext()
                # 2. 生成ImageFrame
                Twc_gt = None
                timestamp = info["timestamp"]
                if "Twc_gt" in info:
                    x, y, z, qx, qy, qz, qw = info["Twc_gt"]
                    self.frames_Twc_gt.append([timestamp, x, y, z, qx, qy, qz, qw])
                    if self.args.use_gt_pose:
                        Twc_gt = pp.Sim3(torch.tensor([[x, y, z, qx, qy, qz, qw, 1]]))
                        Twc_gt = pp.quat2unit(Twc_gt).to(self.device)
                T_WC = self.init_pose_per_frame(frameID, Twc_gt)
                img_slam = self.dataset.transform.to_slam(original_image, device=self.device)
                self.is_test[frameID] = info["is_test"]
                self.qframes.append(
                    ImageFrame(frameID, 0, float(timestamp), img_slam, T_WC, K=self.K_slam)
                )
                frameID += 1
            except Exception as e:
                if frameID < len(self.dataset):
                    raise RuntimeError(f"Dataset failed before end at frame {frameID}") from e
                is_end = True

            if is_end and len(self.qframes) == 0:
                break

            th_f = max(1, self.args.max_queue_size - min(len(self.keyframes), self.args.num_ref))
            if is_end or (len(self.qframes) >= th_f):

                # 3. Track
                if last_keyframe is not None:
                    frames, frames_style, idx_f2ks, idx_masks, dynamic_mask, last_keyframe_id, self.qframes = self.tracker.track(self.qframes, last_keyframe.T_WC)
                else:
                    frames, frames_style, idx_f2ks, idx_masks, dynamic_mask, last_keyframe_id, self.qframes = self.tracker.track(self.qframes)

                # 4. Add keyframe
                if frames is None:
                    continue
                for i in range(len(frames)):
                    current_frame = frames[i]
                    current_style = frames_style[i]
                    idx_f2k = idx_f2ks[i]
                    idx_mask = idx_masks[i]
                    current_test = self.is_test[current_frame.frame_id]

                    bar.update(1)
                    if current_style == -1:
                        self.lost_frame_ids.append(int(current_frame.frame_id))
                        with self.states.lock:
                            self.states.lost_number.value += 1
                        print('lost', current_frame.frame_id)
                        continue

                    # Test frame is at least mapping frame
                    if current_test and (current_style != 1):
                        current_style = 2

                    if current_style == 1: # keyframe
                        self.keyframes.append(current_frame)
                        last_keyframe = current_frame                        
                    else: # non keyframe
                        self.store_relative_pose(current_frame, len(self.keyframes) - 1,
                                            last_keyframe.T_WC.Inv().mul(current_frame.T_WC))
                    
                    self.states.set_frame(current_frame)
                    self.prepare_for_mapper(
                        current_frame, last_keyframe, 
                        len(self.keyframes) - 1, 
                        idx_f2k, idx_mask, dynamic_mask,
                        is_test=current_test, 
                        is_important=(current_style > 0), 
                        is_slam_keyframe=(current_style == 1),
                        K=current_frame.K
                    )

        self.states.set_mode(Mode.TERMINATED)
        self.sav_results(self.args.model_path)

    @torch.no_grad()
    def prepare_for_mapper(self, keyframe, last_keyframe, last_keyframe_index, idx_k2l, idx_mask, dynamic_mask,
                           is_test = False, is_important = False, is_slam_keyframe=False, 
                           K = None):
        if last_keyframe is None:
            X_init, C_init = keyframe.X_canon, keyframe.get_average_conf()
            point = (constrain_points_to_ray(keyframe.img.shape[-2:], X_init[None], K).squeeze(0).reshape(self.H_slam, self.W_slam, 3))
            point_valid = (keyframe.get_average_conf() > 0.5).squeeze(-1).reshape(self.H_slam, self.W_slam)
            T_CkC = None
        else:
            point, point_valid = self.compute_dense_point(keyframe, last_keyframe, idx_k2l, idx_mask, K.to(self.device))
            T_CkC = last_keyframe.T_WC.Inv().mul(keyframe.T_WC).cpu()
        densePoint = torch.cat([point, point_valid[..., None]], dim=-1)
        keyframe_map_dict = {
            "is_test": is_test,
            "is_important": is_important,
            "T_WC": keyframe.T_WC.data.cpu(),
            "frame_id": keyframe.frame_id,
            "densePoint": densePoint.cpu(),
            "is_slam_keyframe": is_slam_keyframe,
            # For map keyframe
            "dynamic_mask": dynamic_mask.cpu() if dynamic_mask is not None else None,
            "T_CkC": T_CkC,
            "last_keyframe_index": last_keyframe_index,
            "last_keyframe_frame_id": last_keyframe.frame_id if last_keyframe is not None else None,
            # camera
            "K": K.cpu() if K is not None else None,
        }
        self.states.msgPut(keyframe_map_dict)

    @torch.no_grad()
    def compute_dense_point(self, keyframe, last_keyframe, idx_k2l, idx_mask, K, valid_pixel=3):
        assert self.config["use_calib"]==True, "Only work for known calib"
        Xkk = keyframe.X_canon
        H, W = keyframe.img.shape[-2:]   

        consume_dense_correspondence_rng(idx_k2l, H, W)

        # get matched points on the last keyframe
        Twk = keyframe.T_WC
        Twl = last_keyframe.T_WC
        Tlk = Twl.Inv().mul(Twk)
        Xkk_calib = constrain_points_to_ray(keyframe.img.shape[-2:], Xkk[None], K).squeeze(0)
        Xkk_calib_match = Xkk_calib[idx_k2l]
        Xkl_calib_match = Tlk.Act(Xkk_calib_match).reshape(H, W, 3)  # H W 3

        # Compute Residual
        pkl = Xkl_calib_match / Xkl_calib_match[..., 2:]
        pkl_es = (K[None, None, ...] @ pkl[..., None]).squeeze(-1)[..., :2]  # H W 2
        uk, vk = torch.meshgrid(torch.arange(W).to(Xkk_calib.device),
                                torch.arange(H).to(Xkk_calib.device),
                                indexing="xy")
        pll = torch.stack([uk, vk], dim=-1)
        resi = torch.norm(pkl_es - pll, dim=-1).reshape(-1)
        conf_valid = torch.where(resi < valid_pixel, 1., 1 / (resi - valid_pixel + 1))

        # 找出有效的相机坐标系的点
        Xkw = Twk.Act(Xkk_calib_match)
        Tkw_se3 = pp.SE3(Twk.data[:, :7]).Inv()
        Xkk_calib_match_map = Tkw_se3.Act(Xkw)  # (H*W) 3

        # 恢复原本的相机坐标系的顺序坐标
        # 初始化输出数组
        Xkk_map_sorted = Tkw_se3.Act(Twk.Act(Xkk_calib))
        conf_valid_sorted = torch.zeros(H * W, device=conf_valid.device, dtype=conf_valid.dtype)
        # # 找到有效的索引（在0到H*W-1范围内）
        # valid_range = (idx_k2l >= 0) & (idx_k2l < H * W)
        # # 获取有效的索引和对应的数据
        # valid_idx = idx_k2l[valid_range]
        # valid_Xkk = Xkk_calib_match_map[valid_range]
        # valid_mask = conf_valid[valid_range]
        valid_idx = idx_k2l[idx_mask]
        valid_Xkk = Xkk_calib_match_map[idx_mask]
        valid_mask = conf_valid[idx_mask]
        # 根据idx_f2k进行排序赋值
        Xkk_map_sorted[valid_idx] = valid_Xkk
        conf_valid_sorted[valid_idx] = valid_mask

        return Xkk_map_sorted.reshape(H, W, 3), conf_valid_sorted.reshape(H, W)

    def store_relative_pose(self, frame, index_keyframe, Tckc):
        self.frames_info.append([frame.frame_id, frame.frame_time, index_keyframe, Tckc])

    def sav_results(self, path):
        print("Saving the poses and point clouds...")
        save_dir = pathlib.Path(path).joinpath("slam")
        save_dir.mkdir(exist_ok=True, parents=True)

        lost_percentage = self.states.lost_number.value / len(self.dataset)
        with open(save_dir / f"lost_percentage.txt", 'w') as f:
            f.write(str(lost_percentage))

        (save_dir / 'tracking_accounting.json').write_text(json.dumps({
            'input_frames': len(self.dataset), 'lost_frames': len(self.lost_frame_ids),
            'lost_frame_ids': self.lost_frame_ids, 'lost_fraction': lost_percentage,
            'reason_scope': 'tracker returned style -1; detailed rejection reasons not yet instrumented'
        }, indent=2))

        # save config
        configFile = save_dir / f"config.json"
        with open(configFile, 'w') as f:
            json.dump(self.config, f, indent=4)

        # save pose
        Twc_est, Twc_est_keyframe = eval.save_traj(save_dir, self.keyframes, self.frames_info)
        if len(self.frames_Twc_gt) > 0:
            self.frames_Twc_gt = np.array(self.frames_Twc_gt)
            eval.evaluate_trajectory(save_dir, "evaluate_frames.json", Twc_est, self.frames_Twc_gt)
            eval.evaluate_trajectory(save_dir, "evaluate_keyframes.json", Twc_est_keyframe, self.frames_Twc_gt)

            with open(save_dir.joinpath("gt_pose.txt"), "w") as f:
                for i in range(len(self.frames_Twc_gt)):
                    t, x, y, z, qx, qy, qz, qw = self.frames_Twc_gt[i]
                    f.write(f"{t} {x} {y} {z} {qx} {qy} {qz} {qw}\n")

        # save point cloud
        if self.args.save_point_could:
            eval.save_keyframe(save_dir, self.keyframes, 1.5, device=self.device, use_calib=self.use_calib)
        
        import yaml

        fx = self.tracker.K_slam[0, 0]
        fy = self.tracker.K_slam[1, 1]
        cx = self.tracker.K_slam[0, 2]
        cy = self.tracker.K_slam[1, 2]

        # Prepare the data dictionary
        data = {
            'width': int(self.tracker.W_slam),
            'height': int(self.tracker.H_slam),
            'calibration': [float(fx), float(fy), float(cx), float(cy)]
        }

        # Write to intr.yaml
        with open(save_dir.joinpath('intr.yaml'), 'w') as f:
            # We use default_flow_style=None to get the list on one line 
            # and sort_keys=False to keep the order you requested
            yaml.dump(data, f, default_flow_style=None, sort_keys=False)

        print("Saved!")