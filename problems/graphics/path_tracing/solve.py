"""The path tracer: follow every ray of a batch bounce by bounce, then average samples into images."""

import torch

from problems._common.progress import steps

from .equation import EPS, cosine_sample, direct_light, nearest_hit
from .tensors import camera_rays


def trace(o: torch.Tensor, d: torch.Tensor, scene: dict, gen: torch.Generator, bounces: int, roulette: bool):
    """Radiance carried back along each ray: global illumination and direct-only, both (R, 3).

    Only live paths are kept in the batch (escaped or terminated ones are dropped), and their contributions are
    scattered back to the original ray index with index_add_. With roulette, after `bounces` bounces a path survives
    with probability q = max_c β_c (≤ 0.95) and is re-weighted by 1/q, which keeps the estimate unbiased; otherwise
    it stops after exactly `bounces` bounces."""
    n_rays = o.shape[0]
    total = torch.zeros(n_rays, 3, device=o.device)
    direct = torch.zeros(n_rays, 3, device=o.device)
    idx = torch.arange(n_rays, device=o.device)  # original ray of every live path
    beta = torch.ones(n_rays, 3, device=o.device)  # path throughput β
    nee = scene["light"] is not None
    max_depth = 4 * bounces + 8 if roulette else bounces
    for k in range(max_depth + 1):
        t, prim, n = nearest_hit(o, d, scene)
        live = prim >= 0  # paths that escape through the open front are dropped
        idx, o, d, t, prim, n, beta = idx[live], o[live], d[live], t[live], prim[live], n[live], beta[live]
        if len(idx) == 0:
            break
        x = o + t[:, None] * d
        rho = scene["albedo"][prim]
        contrib = torch.zeros_like(beta)
        if k == 0 or not nee:  # with next-event estimation, light reached by bouncing is already counted
            contrib += beta * scene["emission"][prim]
        if nee:
            contrib += beta * direct_light(x, n, rho, scene, gen)
        total.index_add_(0, idx, contrib)
        if k == 0:
            direct.index_add_(0, idx, contrib)
        if k == max_depth:
            break
        beta = beta * rho  # f cos θ / p = (ρ/π) cos θ / (cos θ/π) = ρ
        if roulette and k + 1 >= bounces:
            q = beta.max(dim=1).values.clamp(max=0.95)
            survive = torch.rand(len(idx), generator=gen, device=o.device) < q
            beta = beta / q.clamp(min=1e-6)[:, None]
            idx, x, n, beta = idx[survive], x[survive], n[survive], beta[survive]
        o, d = x + EPS * n, cosine_sample(n, gen)
    return total, direct


def render(scene, cam, spp: int, gen, bounces: int, roulette: bool, chunk: int = 16):
    """Mean of spp samples per pixel: GI and direct images (res, res, 3), rendered chunk samples at a time."""
    res, dev = cam["res"], scene["n"].device
    gi = torch.zeros(res * res, 3, device=dev)
    di = torch.zeros(res * res, 3, device=dev)
    for s in steps((spp + chunk - 1) // chunk, "sample batches"):
        m = min(chunk, spp - s * chunk)
        o, d = camera_rays(cam, m, gen, dev)
        total, direct = trace(o, d, scene, gen, bounces, roulette)
        gi += total.reshape(m, -1, 3).sum(0)
        di += direct.reshape(m, -1, 3).sum(0)
    return (gi / spp).reshape(res, res, 3), (di / spp).reshape(res, res, 3)


def progressive(scene, cam, milestones: list[int], gen, bounces: int, roulette: bool, chunk: int = 16):
    """Keep adding samples and snapshot the running mean at each milestone: (M, res, res, 3) for GI and direct."""
    res, dev = cam["res"], scene["n"].device
    gi_sum = torch.zeros(res, res, 3, device=dev)
    di_sum = torch.zeros(res, res, 3, device=dev)
    gi_snaps, di_snaps, done = [], [], 0
    for m in steps(len(milestones), "milestones"):
        while done < milestones[m]:
            n = min(chunk, milestones[m] - done)
            gi, di = render(scene, cam, n, gen, bounces, roulette, chunk)
            gi_sum, di_sum, done = gi_sum + n * gi, di_sum + n * di, done + n
        gi_snaps.append(gi_sum / done)
        di_snaps.append(di_sum / done)
    return torch.stack(gi_snaps), torch.stack(di_snaps)


def rmse(images: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Root-mean-square error against the reference in the displayed range [0, 1]: (M, H, W, 3), (H, W, 3) → (M,).

    Clipping first keeps the few pixels on the edge of the bright light (radiance 0 or L_e at random) from
    dominating the average."""
    return ((images.clamp(0, 1) - reference.clamp(0, 1)) ** 2).mean(dim=(-3, -2, -1)).sqrt()


def noise_slope(spp: torch.Tensor, err: torch.Tensor) -> float:
    """Least-squares slope of log(error) against log(spp); Monte Carlo predicts −1/2."""
    x, y = spp.log(), err.log()
    x, y = x - x.mean(), y - y.mean()
    return float((x * y).sum() / (x * x).sum())
