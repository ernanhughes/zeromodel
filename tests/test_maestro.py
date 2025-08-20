def test_mask_ratio_exact():
    from zeromodel.maestro.masking import structured_mask
    m = structured_mask((8, 8, 8, 4), mask_ratio=0.75)
    assert abs(m.float().mean().item() - 0.75) < 1e-3


def test_pgw_norm_groups_balanced():
    import torch
    from zeromodel.maestro.pgw_norm import pgw_normalize
    x = torch.randn(128,128,8)
    groups = [(0,2),(2,5),(5,8)]
    y = pgw_normalize(x, groups)
    for s,e in groups:
        g = y[..., s:e]
        m, sdev = g.mean().abs().item(), g.std().item()
        assert m < 1e-2 and 0.8 < sdev < 1.2


