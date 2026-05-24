import torch
import numpy as np
import pyflow

def optical_flow(src, tgt):
    """Optical flow from the source frame to each target frame using pyflow (https://github.com/pathak22/pyflow)."""
    para = dict(alpha=0.012, ratio=0.75, minWidth=20, nOuterFPIterations=7, nInnerFPIterations=1, nSORIterations=30, colType=0)
    
    assert src.dtype == np.uint8 and len(src.shape) == 3, src.shape
    assert tgt.dtype == np.uint8 and len(tgt.shape) in [3, 4], tgt.shape
    assert tgt.shape[-3:] == src.shape, (src.shape, tgt.shape)
    
    src = src.astype(float) / 255
    tgt = tgt.astype(float) / 255
    
    if len(tgt.shape) == 3:
        *uv, _ = pyflow.coarse2fine_flow(src, tgt, **para)
        return np.stack(uv, axis=2)
    else:
        flow = []
        for im in tgt:
            *uv, _ = pyflow.coarse2fine_flow(src, im, **para)
            flow.append(np.stack(uv, axis=2))
        return np.stack(flow)

def get_raft_model(model_size="small", device=None):
    if model_size == "small":
        from torchvision.models.optical_flow import raft_small as raft
        from torchvision.models.optical_flow import Raft_Small_Weights as Weights
    else:
        from torchvision.models.optical_flow import raft_large as raft
        from torchvision.models.optical_flow import Raft_Large_Weights as Weights

    weights = Weights.DEFAULT
    transforms = weights.transforms()
    raft_model = raft(weights=weights, progress=False).eval()
    if device is not None:
        raft_model = raft_model.to(device)
    
    return raft_model, transforms

@torch.no_grad()
def optical_flow_raft(src, tgt, model, transforms, batch_size=1):
    assert src.dtype == np.uint8 and len(src.shape) == 3, src.shape
    assert tgt.dtype == np.uint8 and len(tgt.shape) in [3, 4], tgt.shape
    assert tgt.shape[-3:] == src.shape, (src.shape, tgt.shape)
    
    device = next(model.parameters()).device
    
    if len(tgt.shape) == 3:
        src = torch.from_numpy(src).unsqueeze(0).permute(0, 3, 1, 2)
        tgt = torch.from_numpy(tgt).unsqueeze(0).permute(0, 3, 1, 2)
        src, tgt = transforms(src, tgt)
        out = model(src.to(device), tgt.to(device))[-1]
        return out.permute(0, 2, 3, 1).cpu().numpy()
    else:
        src = torch.from_numpy(src).unsqueeze(0).permute(0, 3, 1, 2)
        tgt = torch.from_numpy(tgt).permute(0, 3, 1, 2)
        
        nb = int(np.ceil(tgt.shape[0] / batch_size))
        flow = []
        for i in range(nb):
            s = i * batch_size
            e = min(tgt.shape[0], s + batch_size)
            src_, tgt_ = transforms(src.repeat(e - s, 1, 1, 1), tgt[s:e])
            out = model(src_.to(device), tgt_.to(device))[-1]
            flow.append(out.cpu())
        return torch.cat(flow).permute(0, 2, 3, 1).numpy()