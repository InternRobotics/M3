import torch
import torch.nn.functional as F
import torch.nn as nn
from typing import *
import math

from ..utils.geometry import homogenize_points, se3_inverse, depth_edge, get_corr_from_pts
from ..utils.alignment import align_points_scale

# ---------------------------------------------------------------------------
# Some functions from MoGe
# ---------------------------------------------------------------------------

import numpy as np
from plyfile import PlyData, PlyElement
from train_utils.corr import CorrespondingLoss, MatchingLoss, InfoNCE

def storePly(path, xyz):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    normal = np.zeros_like(xyz)
    rgb = np.ones_like(xyz)
    attributes = np.concatenate((xyz, normal, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))


class VisibilityLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criteria_fn = torch.nn.BCEWithLogitsLoss()

    def forward(self, gt, pred):
        pred_vis = pred['visibility']
        gt_vis = gt['visibility']
        B, N, H, W = gt_vis.shape

        valid_masks = gt['valid_masks'].view(B * N, H, W)
        gt_vis = gt_vis.view(B * N, H, W)

        details = dict()
        final_loss = 0.0

        vis_loss = self.criteria_fn(pred_vis[valid_masks], gt_vis[valid_masks])
        
        final_loss += vis_loss.mean()
        details['vis_loss'] = vis_loss.mean()

        return final_loss, details


class PointCovarianceLoss(nn.Module):
    def __init__(self, local_align_res=4096):
        super().__init__()
        self.local_align_res = local_align_res

    def prepare_ROE(self, pts, mask, target_size=4096):
        B, N, H, W, C = pts.shape
        output = []
        
        for i in range(B):
            valid_pts = pts[i][mask[i]]

            if valid_pts.shape[0] > 0:
                valid_pts = valid_pts.permute(1, 0).unsqueeze(0)  # (1, 3, N1)
                # NOTE: Is is important to use nearest interpolate. Linear interpolate will lead to unstable result!
                valid_pts = F.interpolate(valid_pts, size=target_size, mode='nearest')  # (1, 3, target_size)
                valid_pts = valid_pts.squeeze(0).permute(1, 0)  # (target_size, 3)
            else:
                valid_pts = torch.ones((target_size, C), device=valid_pts.device)

            output.append(valid_pts)

        return torch.stack(output, dim=0)
    
    def forward(self, gt, pred, eps = 1e-7):
        pred_local_pts = pred['local_points']
        gt_local_pts = gt['local_points']
        valid_masks = gt['valid_masks']
        details = dict()
        final_loss = 0.0

        B, N, H, W, _ = pred_local_pts.shape

        weights_ = gt_local_pts[..., 2]
        weights_ = weights_.clamp_min(0.1 * weighted_mean(weights_, valid_masks, dim=(-2, -1), keepdim=True))
        weights_ = 1 / (weights_ + 1e-6)

        # alignment
        with torch.no_grad():
            xyz_pred_local = self.prepare_ROE(pred_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_gt_local = self.prepare_ROE(gt_local_pts.reshape(B, N, H, W, 3), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()
            xyz_weights_local = self.prepare_ROE((weights_[..., None]).reshape(B, N, H, W, 1), valid_masks.reshape(B, N, H, W), target_size=self.local_align_res).contiguous()[:, :, 0]

            S_opt_local = align_points_scale(xyz_pred_local, xyz_gt_local, xyz_weights_local)
            S_opt_local[S_opt_local <= 0] *= -1

        aligned_local_pts = S_opt_local.view(B, 1, 1, 1, 1) * pred_local_pts

        

        # local point loss
        local_pred = aligned_local_pts[valid_masks].float()
        local_gt = gt_local_pts[valid_masks].float()
        local_error = ((local_pred - local_gt)**2).detach()

        point_covariance = pred['covariance'].view(B, N, H, W, 3)
        exp_cov = point_covariance[valid_masks] + eps
        local_pts_loss = (local_error / exp_cov ) + torch.log(exp_cov)

        final_loss += local_pts_loss.mean()
        details['cov_local_pts_loss'] = local_pts_loss.mean()

        return final_loss, details


class CorrCovarianceLoss(nn.Module):
    def __init__(self):
        super().__init__()


# ---------------------------------------------------------------------------
# Final Loss
# ---------------------------------------------------------------------------

class Pi3Loss(nn.Module):
    def __init__(
        self,
        corr_loss_weight=0.,
        vis_loss_weight=0.,
        corr_cov_loss_weight=1.,
        point_cov_loss_weight=1.,
    ):
        super().__init__()
        self.vis_loss_weight = vis_loss_weight
        self.corr_loss_weight = corr_loss_weight
        self.corr_cov_loss_weight = corr_cov_loss_weight
        self.point_cov_loss_weight = point_cov_loss_weight

        if self.vis_loss_weight > 0:
            self.vis_loss = VisibilityLoss()
        
        if self.corr_loss_weight > 0:
            self.corr_loss = CorrespondingLoss(
                MatchingLoss(
                    InfoNCE(
                        mode='proper', temperature=0.05
                    ), negatives_padding=0, blocksize=8192
                ), alpha=10.0, confmode='mean'
            )

        if self.corr_cov_loss_weight > 0:
            self.ccov_loss = CorrCovarianceLoss()

        if self.point_cov_loss_weight > 0:
            self.pcov_loss = PointCovarianceLoss()

    def prepare_gt(self, gt):
        imgs = gt['images']
        gt_pts = gt['world_points']
        masks = gt['point_masks']
        poses = gt['camera_poses']
        visibility = gt['visibility']
        corres_from = gt['corres_from']
        corres_to = gt['corres_to']
        valid_corres = gt['valid_corres']

        B, N, H, W, _ = gt_pts.shape

        # transform to first frame camera coordinate
        w2c_target = se3_inverse(poses[:, 0])
        gt_pts = torch.einsum('bij, bnhwj -> bnhwi', w2c_target, homogenize_points(gt_pts))[..., :3]
        poses = torch.einsum('bij, bnjk -> bnik', w2c_target, poses)

        # normalize points
        valid_batch = masks.sum([-1, -2, -3]) > 0
        if valid_batch.sum() > 0:
            B_ = valid_batch.sum()
            all_pts = gt_pts[valid_batch].clone()
            all_pts[~masks[valid_batch]] = 0
            all_pts = all_pts.reshape(B_, N, -1, 3)
            all_dis = all_pts.norm(dim=-1)
            norm_factor = all_dis.sum(dim=[-1, -2]) / (masks[valid_batch].float().sum(dim=[-1, -2, -3]) + 1e-8)

            gt_pts[valid_batch] = gt_pts[valid_batch] / norm_factor[..., None, None, None, None]
            poses[valid_batch, ..., :3, 3] /= norm_factor[..., None, None]

        extrinsics = se3_inverse(poses)
        gt_local_pts = torch.einsum('bnij, bnhwj -> bnhwi', extrinsics, homogenize_points(gt_pts))[..., :3]
        
        return dict(
            imgs=imgs,
            global_points=gt_pts,
            local_points=gt_local_pts,
            valid_masks=masks,
            camera_poses=poses,
            visibility=visibility,
            corres_from=corres_from,
            corres_to=corres_to,
            valid_corres=valid_corres,
        )
    
    def normalize_pred(self, pred, gt):
        local_points = pred['local_points']
        B, N, H, W, _ = local_points.shape
        masks = gt['valid_masks']

        # normalize predict points
        all_pts = local_points.clone()
        all_pts[~masks] = 0
        all_pts = all_pts.reshape(B, N, -1, 3)
        all_dis = all_pts.norm(dim=-1)
        norm_factor = all_dis.sum(dim=[-1, -2]) / (masks.float().sum(dim=[-1, -2, -3]) + 1e-8)
        local_points  = local_points / norm_factor[..., None, None, None, None]
        pred['local_points'] = local_points

        return pred

    def forward(self, pred, gt_raw):
        with torch.amp.autocast(device_type='cuda', enabled=False): 
            gt = self.prepare_gt(gt_raw)
            if 'local_points' in pred:
                pred = self.normalize_pred(pred, gt)

            final_loss = 0.0
            loss_dict = dict()

            if self.vis_loss_weight > 0:
                # Visibility Loss
                vis_loss, vis_loss_details = self.vis_loss(gt, pred)
                final_loss += vis_loss * self.vis_loss_weight
                loss_dict.update(vis_loss_details)

            if self.corr_loss_weight > 0:
                # Correspondence Loss
                corr_loss, corr_loss_details = self.corr_loss(gt, pred)
                final_loss += corr_loss * self.corr_loss_weight
                loss_dict.update(corr_loss_details)

            if self.corr_cov_loss_weight:
                # Local Point Loss
                corr_cov_loss, corr_cov_loss_details = self.ccov_loss(gt, pred)
                final_loss += corr_cov_loss * self.corr_cov_loss_weight
                loss_dict.update(corr_cov_loss_details)

            if self.point_cov_loss_weight:
                # Local Point Loss
                point_cov_loss, point_cov_loss_details = self.pcov_loss(gt, pred)
                final_loss += point_cov_loss * self.point_cov_loss_weight
                loss_dict.update(point_cov_loss_details)

            loss_dict['objective'] = final_loss

        return loss_dict
