import torch
import pdb

def sample_pdf(bins, weights, N_importance, perturb, eps=1e-5):
    device = weights.device
    weights = weights + eps
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1], device=device), cdf], -1)

    if perturb:
        u = torch.rand(list(cdf.shape[:-1]) + [N_importance], device=device)
    else:
        u = torch.linspace(0. + 0.5 / N_importance, 1. - 0.5 / N_importance, steps=N_importance, device=device)
        u = u.expand(list(cdf.shape[:-1]) + [N_importance])

    u = u.contiguous()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)

    bins_below = torch.gather(bins, 1, below)
    bins_above = torch.gather(bins, 1, above)
    cdf_below = torch.gather(cdf, 1, below)
    cdf_above = torch.gather(cdf, 1, above)

    denom = cdf_above - cdf_below
    denom = torch.where(denom < eps, torch.ones_like(denom, device=device), denom)
    t = (u - cdf_below) / denom
    samples = bins_below + t * (bins_above - bins_below)

    return samples