"""Train a radiance field on random batches of rays (Adam on the pixel MSE) and render whole held-out images."""

import torch

from problems._common.progress import steps

from .equation import psnr, render_rays
from .tensors import BACKGROUND, FAR, NEAR, SCALE, parameters


def render_image(layers, n_freq, o, d, n_samples, chunk=2048):
    """Render every ray of one image (P rays) in chunks, with bin-centre samples: colour (P, 3), depth (P,)."""
    bg = torch.tensor(BACKGROUND, device=o.device)
    out_rgb, out_depth = [], []
    with torch.no_grad():
        for start in range(0, o.shape[0], chunk):
            rgb, depth, _, _ = render_rays(o[start : start + chunk], d[start : start + chunk], layers, n_freq, SCALE,
                                           n_samples, NEAR, FAR, bg)  # fmt: skip
            out_rgb.append(rgb)
            out_depth.append(depth)
    return torch.cat(out_rgb), torch.cat(out_depth)


def train(layers, n_freq, o, d, target, novel, n_steps, lr, batch, n_samples, n_snap, seed, label="Adam steps"):
    """Adam on the mean squared error of random ray batches drawn from all training pixels.

    o, d, target: (N_rays, 3) every training ray and its true colour; the first P of them are training view 0.
    novel = (o, d, rgb) of the held-out image, P rays. Returns the loss per step and, at n_snap checkpoints, the
    held-out render, its depth map and PSNR, the PSNR of training view 0, and the weights w_s along the held-out
    view's centre ray.
    """
    gen = torch.Generator().manual_seed(seed)
    bg = torch.tensor(BACKGROUND, device=o.device)
    opt = torch.optim.Adam(parameters(layers), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_steps, eta_min=lr / 10)
    snap_at = set(torch.linspace(0, n_steps, n_snap).round().long().tolist())
    o_n, d_n, rgb_n = novel
    n_pix = o_n.shape[0]
    centre = o_n.shape[0] // 2 + int(o_n.shape[0] ** 0.5) // 2  # the middle pixel of the image
    losses, snaps = [], []

    def snapshot(it: int):
        img, depth = render_image(layers, n_freq, o_n, d_n, n_samples)
        img_train, _ = render_image(layers, n_freq, o[:n_pix], d[:n_pix], n_samples)
        with torch.no_grad():
            _, _, w, t = render_rays(o_n[[centre]], d_n[[centre]], layers, n_freq, SCALE, n_samples, NEAR, FAR, bg)
        snaps.append({"it": it, "rgb": img.cpu(), "depth": depth.cpu(), "psnr": psnr(img, rgb_n).item(),
                      "psnr_train": psnr(img_train, target[:n_pix]).item(),
                      "w": w[0].cpu(), "t": t[0].cpu()})  # fmt: skip

    for it in steps(n_steps, label):
        if it in snap_at:
            snapshot(it)
        idx = torch.randint(0, o.shape[0], (batch,), generator=gen).to(o.device)  # random rays from all views
        rgb, _, _, _ = render_rays(o[idx], d[idx], layers, n_freq, SCALE, n_samples, NEAR, FAR, bg, gen)
        loss = ((rgb - target[idx]) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    snapshot(n_steps)
    return {"loss": torch.tensor(losses), "snaps": snaps}
