"""Minimise -2 ln L (or χ²) with autograd: Adam for the animated iterations, then L-BFGS to polish."""

import torch

from problems._common.progress import steps

from .equation import chi2, neg2_log_likelihood


def objective(p: torch.Tensor, t, x, sigma, jitter: bool) -> torch.Tensor:
    """-2 ln L with the jitter s fitted, or plain χ² (s = 0) when jitter is off."""
    return neg2_log_likelihood(p, t, x, sigma) if jitter else chi2(p, t, x, sigma)


def fit(p0: torch.Tensor, t, x, sigma, lr: float, iterations: int, jitter: bool = True):
    """Return (p_best (6,), history (K, 6), loss (K,)): parameters after every Adam step, the last row after L-BFGS.

    With jitter off, ln s gets no gradient and stays where it started; it is simply unused.
    """
    p = p0.clone().requires_grad_(True)
    history, losses = [p.detach().clone()], [objective(p, t, x, sigma, jitter).item()]

    adam = torch.optim.Adam([p], lr=lr)
    for _ in steps(iterations, "Adam iterations"):
        adam.zero_grad()
        loss = objective(p, t, x, sigma, jitter)
        loss.backward()  # autograd: ∂loss/∂p_p for all six parameters at once
        adam.step()
        history.append(p.detach().clone())
        losses.append(objective(p, t, x, sigma, jitter).item())

    lbfgs = torch.optim.LBFGS(
        [p], max_iter=200, tolerance_grad=1e-10, tolerance_change=1e-14, line_search_fn="strong_wolfe"
    )

    def closure():
        lbfgs.zero_grad()
        loss = objective(p, t, x, sigma, jitter)
        loss.backward()
        return loss

    lbfgs.step(closure)
    history.append(p.detach().clone())
    losses.append(objective(p, t, x, sigma, jitter).item())
    return p.detach(), torch.stack(history), torch.tensor(losses)
