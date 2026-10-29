"""Give each disconnected factor component its own fixed pose (gauge)."""
import torch


def componentwise_calib(native, poses, points, conf, K, ii, jj, indices, valid, Q, *params):
    ids=torch.unique(torch.cat((ii,jj)),sorted=True)
    parent={i:i for i in ids.tolist()}
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]]
            i=parent[i]
        return i
    for i,j in zip(ii.tolist(),jj.tolist()):
        parent[find(j)]=find(i)
    groups={}
    for i in parent:
        groups.setdefault(find(i),[]).append(i)
    if len(groups)<=1:
        return native(poses,points,conf,K,ii,jj,indices,valid,Q,*params)
    print('BA_COMPONENT_GAUGES', [len(g) for g in groups.values()],flush=True)
    for group in groups.values():
        group=torch.tensor(sorted(group),device=ii.device,dtype=ii.dtype)
        rows=torch.searchsorted(ids,group)
        edges=torch.isin(ii,group)
        assert torch.isin(jj[edges],group).all()
        component_poses=poses[rows].contiguous()
        native(component_poses,points[rows].contiguous(),conf[rows].contiguous(),K,
               ii[edges].contiguous(),jj[edges].contiguous(),indices[edges].contiguous(),
               valid[edges].contiguous(),Q[edges].contiguous(),*params)
        if not torch.isfinite(component_poses).all():
            raise RuntimeError('Nonfinite component BA result')
        poses.index_copy_(0,rows,component_poses)
