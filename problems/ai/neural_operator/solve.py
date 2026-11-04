"""The data solver (pseudo-spectral Burgers, a whole batch at once) and the FNO training loop (Adam)."""

import math

import torch

from problems._common.progress import steps

from .equation import burgers_nonlinear, fno, relative_l2
from .tensors import parameters


def solve_burgers(u0: torch.Tensor, nu: float, t_final: float, dx: float) -> torch.Tensor:
    r"""u(·, T) for every initial condition in the batch: u0 (B, N) → (B, N).

    Integrating-factor RK4 in Fourier space: the stiff diffusion -νk² û is solved exactly by the factor
    e^{-νk²Δt}, RK4 handles only the advection, so Δt is limited by Δx / max|u| alone.
    """
    n = u0.shape[-1]
    k = torch.fft.rfftfreq(n, d=dx / (2 * math.pi)).to(u0.device)  # wavenumbers 0, 1, …, N/2   (K,)
    dealias = (k < n / 3).to(u0.dtype)  # 2/3 rule
    u_max = u0.abs().max().item() + 1e-12
    n_steps = max(1, math.ceil(t_final / (0.5 * dx / u_max)))
    dt = t_final / n_steps
    E, E2 = torch.exp(-nu * k**2 * dt), torch.exp(-nu * k**2 * dt / 2)  # integrating factors (K,)

    def N(v):
        return burgers_nonlinear(v, k, dealias, n)

    u_hat = torch.fft.rfft(u0)  # (B, K)
    for _ in steps(n_steps, "solver steps"):
        a = dt * N(u_hat)
        b = dt * N(E2 * (u_hat + a / 2))
        c = dt * N(E2 * u_hat + b / 2)
        d = dt * N(E * u_hat + E2 * c)
        u_hat = E * u_hat + (E * a + 2 * E2 * (b + c) + d) / 6
    return torch.fft.irfft(u_hat, n=n)


def train(w: dict, data: dict, epochs: int, lr: float, batch: int, seed: int, label: str = "epochs") -> dict:
    """Adam on the relative L2 loss, in mini-batches of shuffled training samples. data holds float32 tensors
    x (N,), u0_train/uT_train (n_train, N), u0_test/uT_test (n_test, N) and, optionally, for the
    super-resolution panel, u0_fine (S, M) on the finer grid x_fine32 (M,).

    Returns the train/test error after every epoch (entry 0 = untrained) and, for the animation, the
    predictions on the whole test set (coarse grid) and on u0_fine (fine grid) after every epoch.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(parameters(w), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 20)
    x, u0, uT = data["x"], data["u0_train"], data["uT_train"]
    fine = "u0_fine" in data

    def evaluate(train_err: float) -> dict:
        with torch.no_grad():
            pred_test = fno(data["u0_test"], x, w)
            return {
                "train_err": train_err,
                "test_err": relative_l2(pred_test, data["uT_test"]).item(),
                "pred_test": pred_test.cpu(),  # (n_test, N)
                "pred_fine": fno(data["u0_fine"], data["x_fine32"], w).cpu() if fine else torch.zeros(0),
            }

    with torch.no_grad():
        history = [evaluate(relative_l2(fno(u0, x, w), uT).item())]  # entry 0: the untrained model
    for _ in steps(epochs, label):
        order = torch.randperm(len(u0), generator=gen).to(u0.device)  # shuffle the training set
        losses = []
        for start in range(0, len(u0), batch):
            idx = order[start : start + batch]
            loss = relative_l2(fno(u0[idx], x, w), uT[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history.append(evaluate(sum(losses) / len(losses)))  # train error = mean batch loss of the epoch

    stack = {"train_err": torch.tensor, "test_err": torch.tensor, "pred_test": torch.stack, "pred_fine": torch.stack}
    return {key: fn([h[key] for h in history]) for key, fn in stack.items()}  # each with leading dim epochs + 1
