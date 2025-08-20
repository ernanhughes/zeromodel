# zeromodel/maestro/masking.py
import torch

def structured_mask(shape, *, mask_ratio=0.75, time_frac=0.3, group_frac=0.3, space_frac=0.4, device=None):
    """
    Returns a boolean mask of shape (T, Hp, Wp, G) with an exact global ratio of True values.
    """
    T, Hp, Wp, G = shape
    m = torch.zeros(shape, dtype=torch.bool, device=device)

    # 1) Time masking: hide whole time-bins for all patches/groups
    t_hide = int(round(T * time_frac))
    if t_hide > 0:
        tidx = torch.randperm(T, device=device)[:t_hide]
        m[tidx, :, :, :] = True

    # 2) Group masking: hide entire metric-groups across T×Hp×Wp
    g_hide = int(round(G * group_frac))
    if g_hide > 0:
        gidx = torch.randperm(G, device=device)[:g_hide]
        m[:, :, :, gidx] = True

    # 3) Spatial masking: hide random patches across T
    if space_frac > 0:
        space = (torch.rand((T, Hp, Wp), device=device) < space_frac)  # (T,Hp,Wp)
        # broadcast to groups and OR in-place
        m |= space[..., None]  # (T,Hp,Wp,1) -> (T,Hp,Wp,G)

    # 4) Adjust to exact global ratio
    total = T * Hp * Wp * G
    need  = int(round(mask_ratio * total))
    flat  = m.view(-1)
    cur   = int(flat.sum().item())

    if cur < need:
        # randomly add masks
        unmasked_idx = (~flat).nonzero(as_tuple=False).view(-1)
        add = need - cur
        sel = unmasked_idx[torch.randperm(unmasked_idx.numel(), device=device)[:add]]
        flat[sel] = True
    elif cur > need:
        # randomly remove masks
        masked_idx = flat.nonzero(as_tuple=False).view(-1)
        rem = cur - need
        sel = masked_idx[torch.randperm(masked_idx.numel(), device=device)[:rem]]
        flat[sel] = False

    return m.view(T, Hp, Wp, G)
