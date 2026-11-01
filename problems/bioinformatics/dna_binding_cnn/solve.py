"""Train the convolutional network with mini-batch Adam, recording losses, accuracies and filter snapshots."""

import math

import torch

from problems._common.progress import steps

from .equation import accuracy, forward, loss_fn


def train(params, x_tr, y_tr, x_te, y_te, epochs: int, lr: float, batch: int, seed: int, n_snapshots: int = 40):
    """Adam on the binary cross-entropy, with a little L2 on the filters (weight decay). Returns a history dict:

    loss (n_steps,) per mini-batch;  train_acc, test_acc (n_evals,) at the evaluation steps eval_steps;
    snapshots of the filters (W, c) at snap_steps, for turning them into logos later.
    """
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(params.values(), lr=lr)
    n = x_tr.shape[0]
    per_epoch = (n + batch - 1) // batch
    n_steps = epochs * per_epoch
    # Snapshots on a log scale: the filter changes fastest at the start.
    snap_steps = sorted({int(s) - 1 for s in torch.logspace(0, math.log10(n_steps), n_snapshots).round().tolist()})
    hist = {"loss": torch.zeros(n_steps), "eval_steps": [], "train_acc": [], "test_acc": [], "test_loss": []}
    hist["snap_steps"], hist["snapshots"] = snap_steps, []
    step = 0
    for _ in steps(epochs, "epochs"):
        order = torch.randperm(n, generator=gen).to(x_tr.device)
        for b in range(per_epoch):
            idx = order[b * batch : (b + 1) * batch]
            loss = loss_fn(forward(x_tr[idx], params), y_tr[idx]) + 1e-3 * (params["W"] ** 2).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            hist["loss"][step] = loss.item()
            if step in snap_steps:
                with torch.no_grad():
                    hist["snapshots"].append((params["W"].detach().clone(), params["c"].detach().clone()))
                    _evaluate(hist, step, params, x_tr, y_tr, x_te, y_te)
            step += 1
    return hist


def _evaluate(hist, step, params, x_tr, y_tr, x_te, y_te) -> None:
    """Accuracy on (up to 2000 of) the training sequences and on the held-out test set."""
    logits_te = forward(x_te, params)
    hist["eval_steps"].append(step)
    hist["train_acc"].append(accuracy(forward(x_tr[:2000], params), y_tr[:2000]))
    hist["test_acc"].append(accuracy(logits_te, y_te))
    hist["test_loss"].append(loss_fn(logits_te, y_te).item())
