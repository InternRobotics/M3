#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import os
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from pathlib import Path
import logging
from argparse import Namespace

from dataloaders.DatasetBasic import BaseDataset
from dataloaders.read_write_model import read_model, qvec2rotmat
from Reconstruct.utils import get_image_names
from scipy.spatial.transform import Rotation as R


# 提取数字部分进行排序
def extract_number(filename):
    return int(filename.split('_DSC')[-1].replace('.JPG', ''))


class ColmapDataset(BaseDataset):
    """
    The main dataset class for loading images from disk in a multithreaded manner.
    It also supports loading masks and COLMAP poses if available.
    The next image can be fetched using the `getnext` method.
    """
    def __init__(self, args: Namespace):
        self.image_dir = os.path.join(args.source_path, args.images_dir)
        self.image_name_list = get_image_names(self.image_dir)
        self.image_name_list.sort()
        self.timestamp = []
        for i, name in enumerate(self.image_name_list):
            self.timestamp.append(float(extract_number(name)))
        self.Twc_gt = None

        # Load COLMAP data
        self.load_poses_and_intrinsic(args.source_path)

        super().__init__(args)

    def load_poses_and_intrinsic(self, source_path):
        """Load COLMAP camera intrinsics and extrinsics. Stores them in self.infos."""
        colmap_folder_path = os.path.join(source_path, "sparse/0")
        cameras, images, _ = read_model(colmap_folder_path)
        if len(cameras) != 1:
            logging.warning(" Only supports one camera")
        model = list(cameras.values())[0].model
        if model != "PINHOLE" and model != "SIMPLE_PINHOLE":
            logging.warning("Unexpected camera model: " + model)

        pose_data = []
        pose_name = []
        pose_intrinsic = []
        for image_id, image in images.items():
            camera = cameras[image.camera_id]

            # Intrinsics and projection matrix
            if camera.model == "PINHOLE":
                focal_x = camera.params[0]
                focal_y = camera.params[1]
                cx = camera.params[2]
                cy = camera.params[3]
            else:
                focal_y = focal_x = camera.params[0]
                cx = camera.params[1]
                cy = camera.params[2]

            intri = np.zeros((3, 3), dtype=np.float32)
            intri[0, 0] = focal_x
            intri[1, 1] = focal_y
            intri[0, 2] = cx
            intri[1, 2] = cy

            # Pose
            w2c = np.eye(4, dtype=np.float32)
            w2c[:3, :3] = qvec2rotmat(image.qvec)
            w2c[:3, 3] = image.tvec
            extri = np.linalg.inv(w2c)
            extri[:3, 1:3] *= -1

            # Store CameraInfo for each image
            name = os.path.basename(image.name)

            pose_name.append(name)
            pose_data.append(extri)
            pose_intrinsic.append(intri)

        sorted_indices = np.argsort([extract_number(x) for x in pose_name])
        pose_data = np.array(pose_data)[sorted_indices]
        pose_name = np.array(pose_name)[sorted_indices]
        pose_intrinsic = np.array(pose_intrinsic)[sorted_indices]

        # 每个时间戳下的位姿和内参
        frame_dict = {}
        for i in range(len(pose_name)):
            frame_dict[extract_number(pose_name[i])] = [pose_data[i], pose_intrinsic[i]]

        poses_list = []
        intrinsics_list = []

        Twc0 = pose_data[0]
        for i in range(len(self.timestamp)):
            current_timestamp = self.timestamp[i]

            c2w = frame_dict[current_timestamp][0]
            intrinsic = frame_dict[current_timestamp][1]

            # 检查是否有nan
            if np.isnan(c2w).any() or ~np.isfinite(c2w).all():
                pose_7d = np.full(7, np.nan)
                poses_list.append(pose_7d)
                intrinsic = np.full((3,3), np.nan)
                intrinsics_list.append(intrinsic)
                continue
            c2w = np.linalg.inv(Twc0) @ c2w

            # 提取平移向量
            translation = c2w[:3, 3]  # tx, ty, tz

            # 提取旋转矩阵并转换为四元数
            rotation_matrix = c2w[:3, :3]
            rotation = R.from_matrix(rotation_matrix)
            quaternion = rotation.as_quat()  # [qx, qy, qz, qw]

            # 组合为 [tx, ty, tz, qx, qy, qz, qw]
            pose_7d = np.concatenate([translation, quaternion])
            poses_list.append(pose_7d)
            intrinsics_list.append(intrinsic)

        # 返回 N×7 的numpy数组
        self.Twc_gt = np.array(poses_list)  # shape: (N, 7)
        intrinsics_list = np.array(intrinsics_list)
        self.frame_intrinsic = intrinsics_list