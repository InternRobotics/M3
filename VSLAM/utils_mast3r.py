import torch
import einops
import mast3r.utils.path_to_dust3r  # noqa
import VSLAM.utils_matching as matching
from VSLAM.thirdparty.mast3r.mast3r.model import AsymmetricMASt3R

@torch.inference_mode
def decoder(model, feat1, feat2, pos1, pos2, shape1, shape2):
    dec1, dec2 = model._decoder(feat1, pos1, feat2, pos2)
    with torch.amp.autocast(enabled=False, device_type="cuda"):
        res1 = model._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = model._downstream_head(2, [tok.float() for tok in dec2], shape2)
    return res1, res2


# 循环对称批量求解两帧对应的点云
# NOTE: Assumes img shape the same
@torch.inference_mode
def mast3r_decode_symmetric_batch(model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    B = feat_i.shape[0]
    X, C, D, Q = [], [], [], []
    for b in range(B):
        feat1 = feat_i[b][None]
        feat2 = feat_j[b][None]
        pos1 = pos_i[b][None]
        pos2 = pos_j[b][None]
        res11, res21 = decoder(model, feat1, feat2, pos1, pos2, shape_i[b], shape_j[b])
        res22, res12 = decoder(model, feat2, feat1, pos2, pos1, shape_j[b], shape_i[b])
        res = [res11, res21, res22, res12]
        Xb, Cb, Db, Qb = zip(
            *[
                (r["pts3d"][0], r["conf"][0], r["desc"][0], r["desc_conf"][0])
                for r in res
            ]
        )
        X.append(torch.stack(Xb, dim=0))
        C.append(torch.stack(Cb, dim=0))
        D.append(torch.stack(Db, dim=0))
        Q.append(torch.stack(Qb, dim=0))

    X, C, D, Q = (
        torch.stack(X, dim=1),
        torch.stack(C, dim=1),
        torch.stack(D, dim=1),
        torch.stack(Q, dim=1),
    )
    # 4 b h w 3
    return X, C, D, Q

# 对称求解两帧对应的点云，并且分别在i和j坐标系进行匹配
def mast3r_match_symmetric(config, model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j):
    X, C, D, Q = mast3r_decode_symmetric_batch(
        model, feat_i, pos_i, feat_j, pos_j, shape_i, shape_j
    )

    # Ordering 4xbxhxwxc
    b = X.shape[1]

    Xii, Xji, Xjj, Xij = X[0], X[1], X[2], X[3] # b h w 3
    Dii, Dji, Djj, Dij = D[0], D[1], D[2], D[3]
    Qii, Qji, Qjj, Qij = Q[0], Q[1], Q[2], Q[3]

    # Always matching both
    X11 = torch.cat((Xii, Xjj), dim=0)
    X21 = torch.cat((Xji, Xij), dim=0)
    D11 = torch.cat((Dii, Djj), dim=0)
    D21 = torch.cat((Dji, Dij), dim=0)

    idx_1_to_2, valid_match_2 = matching.match(config, X11, X21, D11, D21)

    # TODO: Avoid this
    match_b = X11.shape[0] // 2
    idx_i2j = idx_1_to_2[:match_b]
    idx_j2i = idx_1_to_2[match_b:]
    valid_match_j = valid_match_2[:match_b]
    valid_match_i = valid_match_2[match_b:]

    return (
        idx_i2j, # 在i坐标系，照j的顺序，给出i的匹配位置
        idx_j2i, # 在j坐标系，按照i的顺序，给出j的匹配位置
        valid_match_j.unsqueeze(-1),
        valid_match_i.unsqueeze(-1),
        Qii.view(b, -1, 1),
        Qjj.view(b, -1, 1),
        Qji.view(b, -1, 1),
        Qij.view(b, -1, 1),
    )