# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# MASt3R heads
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from itertools import repeat
import collections.abc

inf = float('inf')
    
def reg_cov(x, mode):
    """
    extract confidence from prediction head output
    """
    mode, vmin, vmax = mode
    if mode == 'exp':
        return vmin + x.exp().clip(max=vmax-vmin)
    if mode == 'sigmoid':
        return (vmax - vmin) * torch.sigmoid(x) + vmin
    raise ValueError(f'bad {mode=}')
    

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))
    return parse
to_2tuple = _ntuple(2)
    
class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, bias=True, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = to_2tuple(bias)
        drop_probs = to_2tuple(drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class CovarianceHead(nn.Module):
    """ Mixture between MLP and DPT head that outputs 3d points and local features (with MLP).
    The input for both heads is a concatenation of Encoder and Decoder outputs
    """

    def __init__(self, 
        dim=1024, 
        cov_dim=3, 
        patch_size=14,
        hidden_dim_factor=4,
        cov_mode=('exp', 0, inf),
    ):
        
        super().__init__()
        self.patch_size = patch_size
        self.cov_dim = cov_dim
        self.cov_mode = cov_mode

        self.head_local_features = Mlp(in_features=dim,
                                       hidden_features=int(hidden_dim_factor * dim),
                                       out_features=(self.cov_dim) * self.patch_size**2)

    def forward(self, feat, img_shape):
        H, W = img_shape
        BN = feat.shape[0]
        local_features = self.head_local_features(feat)  # B,S,D
        local_features = local_features.transpose(-1, -2).view(BN, -1, H // self.patch_size, W // self.patch_size)
        local_features = F.pixel_shuffle(local_features, self.patch_size)  # B,d,H,W
        local_features = local_features.permute(0, 2, 3, 1)
        point_cov = reg_cov(local_features * 2, mode=self.cov_mode)

        return point_cov
