"""Training loop (Adam on the mean squared error), shared by the GNN and the MLP baseline."""

import torch

from problems._common.progress import steps

from .equation import mse
from .tensors import parameters


def subset(data: dict, idx: torch.Tensor) -> dict:
    return {k: v[idx] for k, v in data.items()}


def train(model, w: dict, train_set: dict, tests: dict, epochs: int, lr: float, batch: int, seed: int, label: str):
    """Fit model(data, w) to train_set["y"]. Returns, per epoch (entry 0 = untrained), the training MSE and the
    predictions on every test set in `tests` (name → data dict with "y")."""
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(parameters(w), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs, eta_min=lr / 30)
    inputs = {k: v for k, v in train_set.items() if k != "y"}
    y = train_set["y"]

    def record(loss: float) -> dict:
        with torch.no_grad():
            return {"loss": loss, **{name: model(d, w).cpu() for name, d in tests.items()}}

    with torch.no_grad():
        history = [record(mse(model(inputs, w), y).item())]
    for _ in steps(epochs, label):
        order = torch.randperm(len(y), generator=gen).to(y.device)
        losses = []
        for start in range(0, len(y), batch):
            idx = order[start : start + batch]
            loss = mse(model(subset(inputs, idx), w), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        history.append(record(sum(losses) / len(losses)))
    out = {"loss": torch.tensor([h["loss"] for h in history])}
    for name in tests:
        out[name] = torch.stack([h[name] for h in history])  # (epochs + 1, M_test)
    return out
