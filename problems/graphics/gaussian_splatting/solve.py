"""Fit the Gaussians to the training images: Adam on the photometric loss, one random view per step, with a simple
split-style densification that recycles transparent Gaussians."""

import math

import torch

from problems._common.progress import steps

from .equation import photometric_loss, psnr, quat_to_rotmat, render

# Per-parameter learning rates relative to `lr`, in the spirit of the paper (positions move slowly, opacity fast).
LR_SCALE = {"mean": 1.0, "log_scale": 2.0, "quat": 1.0, "colour_logit": 5.0, "opacity_logit": 5.0}
DENSIFY_EVERY = 50  # steps
DEAD = 0.05  # Gaussians fainter than this opacity are recycled
SPLIT = 1.6  # a split Gaussian's axes shrink by this factor (as in the paper)


def select(cams: dict, idx: torch.Tensor) -> dict:
    """The sub-batch of cameras idx (the shared intrinsics and pixel grid are kept)."""
    return {**cams, "W": cams["W"][idx], "t": cams["t"][idx], "centre": cams["centre"][idx]}


def densify(g: dict, opt: torch.optim.Adam, grad_acc: torch.Tensor, max_new: int, gen: torch.Generator) -> int:
    """Split the Gaussians whose centres were pushed hardest (largest accumulated |∂L/∂μ|) into two smaller
    copies (axes ÷ 1.6), reusing nearly transparent Gaussians as the new halves, so the total M stays fixed.

    The two halves sit at μ ± R s ε with ε ~ N(0, 1), i.e. a sample from the parent Gaussian. Returns how many
    Gaussians were split.
    """
    with torch.no_grad():
        opacity = torch.sigmoid(g["opacity_logit"])
        dead = (opacity < DEAD).nonzero().squeeze(1)
        alive = opacity >= DEAD
        k = min(len(dead), max_new, int(alive.sum()))
        if k == 0:
            return 0
        parent = torch.where(alive, grad_acc, -1.0).argsort(descending=True)[:k]
        child = dead[:k]
        R = quat_to_rotmat(g["quat"][parent])  # (k, 3, 3)
        eps = torch.randn(k, 3, generator=gen).to(R.device)
        offset = torch.einsum("kij,kj->ki", R, g["log_scale"][parent].exp() * eps)  # a sample from the parent
        for name in ("log_scale", "quat", "colour_logit", "opacity_logit"):
            g[name][child] = g[name][parent]
        g["mean"][child] = g["mean"][parent] + offset
        g["mean"][parent] -= offset
        g["log_scale"][parent] -= math.log(SPLIT)
        g["log_scale"][child] -= math.log(SPLIT)
        for p in g.values():  # forget Adam's running moments for the rows that changed
            state = opt.state.get(p, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state:
                    state[key][parent] = 0
                    state[key][child] = 0
    return k


def train(g: dict, cams: dict, gt: torch.Tensor, novel: dict, gt_novel: torch.Tensor, background: torch.Tensor,
          iterations: int, lr: float, loss_kind: str, densify_on: bool, n_snap: int, seed: int) -> dict:  # fmt: skip
    """Adam on all Gaussian parameters. gt (V, P, 3) are the training images, gt_novel (1, P, 3) the held-out view.

    Returns the loss per step, the number of splits, and at n_snap checkpoints: the renders of training view 0 and
    of the held-out view, the PSNR on four training views and on the held-out view, and the held-out 2D splats.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam([{"params": [g[k]], "lr": lr * s} for k, s in LR_SCALE.items()])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iterations, eta_min=lr / 10)
    snap_at = set(torch.linspace(0, iterations, n_snap).round().long().tolist())
    n_views = gt.shape[0]
    monitor = torch.arange(0, n_views, max(1, n_views // 4), device=gt.device)  # training views used for the PSNR
    grad_acc = torch.zeros(g["mean"].shape[0], device=gt.device)
    losses, snaps, n_split = [], [], 0

    def snapshot(it: int):
        with torch.no_grad():
            train_img, _ = render(g, select(cams, monitor), background)  # view 0 and a few others
            novel_img, splats = render(g, novel, background)
        snaps.append(
            {
                "it": it,
                "train": train_img[0].cpu(),
                "novel": novel_img[0].cpu(),
                "psnr_train": psnr(train_img, gt[monitor]).item(),
                "psnr_novel": psnr(novel_img, gt_novel).item(),
                **{k: splats[k][0].cpu() for k in ("mu2", "cov2", "visible")},  # held-out view
                "colour": splats["colour"].cpu(),
                "opacity": splats["opacity"].cpu(),
            }
        )

    for it in steps(iterations, "Adam steps"):
        if it in snap_at:
            snapshot(it)
        idx = torch.randint(0, n_views, (1,), generator=gen).to(gt.device)  # one random training view per step
        pred, _ = render(g, select(cams, idx), background)
        loss = photometric_loss(pred, gt[idx], loss_kind)
        opt.zero_grad()
        loss.backward()
        grad_acc += g["mean"].grad.norm(dim=1)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if densify_on and it % DENSIFY_EVERY == DENSIFY_EVERY - 1 and it < 0.7 * iterations:
            n_split += densify(g, opt, grad_acc, g["mean"].shape[0] // 5, gen)
            grad_acc.zero_()
    snapshot(iterations)
    return {"loss": torch.tensor(losses), "snaps": snaps, "n_split": n_split}
