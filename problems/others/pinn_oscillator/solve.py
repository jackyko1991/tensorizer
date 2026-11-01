"""Train the network with Adam on  loss = data term + λ_phys · physics term + initial-condition term."""

import math

import torch

from problems._common.progress import steps

from .equation import network, network_and_derivatives, residual


def loss_terms(weights, mu, t_d, x_d, t_c, m: float, k: float, physics_on: bool = True) -> dict[str, torch.Tensor]:
    """The three pieces of the loss (each a scalar).

    physics:  mean_c r_c²                                obey m x̂'' + μ x̂' + k x̂ = 0 at the collocation times
    data:     mean_d (x̂(t_d) - x_d)²                    fit the few measurements
    initial:  (x̂(0) - 1)² + (x̂'(0) / ω₀)²               start at rest at x = 1  (ω₀ = √(k/m) makes ẋ dimensionless)

    All times go through the network in ONE batch [t_c, t_d, 0]; then we slice the pieces back out.
    physics_on = False drops the collocation points and x̂'' (a plain network: cheaper, physics term = 0).
    """
    if not physics_on:
        t_c = t_c[:0]  # no collocation points
    C, D = len(t_c), len(t_d)
    t_all = torch.cat([t_c, t_d, torch.zeros(1, device=t_c.device)])  # (C + D + 1,)
    x, x_t, x_tt = network_and_derivatives(t_all, weights, second=physics_on)
    physics = torch.zeros((), device=t_c.device)
    if physics_on:
        r = residual(x[:C], x_t[:C], x_tt[:C], m, mu, k)  # (C,)
        physics = torch.mean(r**2)
    data = torch.mean((x[C : C + D] - x_d) ** 2)
    initial = (x[-1] - 1.0) ** 2 + (x_t[-1] / math.sqrt(k / m)) ** 2
    return {"data": data, "physics": physics, "initial": initial}


def train(
    weights,
    mu_start: float,
    learn_mu: bool,
    t_d: torch.Tensor,
    x_d: torch.Tensor,
    t_c: torch.Tensor,
    m: float,
    k: float,
    lam_phys: float,
    n_iter: int,
    lr: float,
    t_plot: torch.Tensor,
    n_snapshots: int = 50,
    label: str = "training",
) -> dict:
    """Run Adam for n_iter steps. Returns the loss history (n_iter, 3), μ history (n_iter,),
    and snapshots of x̂ on t_plot taken at the iterations in 'snap_iters'.

    learn_mu = True is the inverse problem: μ is one more trainable number, found from data + physics.
    lam_phys = 0 switches the physics off: a plain network fitted to the data only.
    """
    mu = torch.tensor(mu_start, device=t_d.device, requires_grad=learn_mu)
    groups = [{"params": [p for Wb in weights for p in Wb], "lr": lr}]
    if learn_mu:
        groups.append({"params": [mu], "lr": 10 * lr})  # one scalar: let it move faster than the weights
    opt = torch.optim.Adam(groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, n_iter, eta_min=lr / 20)  # smaller steps at the end

    snap_iters = sorted({round(s * (n_iter - 1) / max(1, n_snapshots - 1)) for s in range(n_snapshots)})
    snap_set = set(snap_iters)
    losses = torch.zeros(n_iter, 3)
    mus = torch.zeros(n_iter)
    snapshots = []
    for it in steps(n_iter, label):
        terms = loss_terms(weights, mu, t_d, x_d, t_c, m, k, physics_on=lam_phys > 0)
        loss = terms["data"] + lam_phys * terms["physics"] + terms["initial"]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()

        losses[it] = torch.stack([terms["data"], terms["physics"], terms["initial"]]).detach().cpu()
        mus[it] = mu.detach().cpu()
        if it in snap_set:
            with torch.no_grad():
                snapshots.append(network(t_plot, weights).cpu())
    return {
        "losses": losses,  # columns: data, physics (mean r²), initial
        "mu": mus,
        "snap_iters": snap_iters,
        "snapshots": torch.stack(snapshots),  # (S, n_plot)
        "mu_final": mu.item(),
    }
