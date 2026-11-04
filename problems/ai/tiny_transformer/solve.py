"""Train the GPT on next-character prediction with Adam, and sample text from it."""

import math

import torch

from problems._common.progress import steps

from . import equation, tensors


def overfit(x: torch.Tensor, y: torch.Tensor, w: dict, n_steps: int, lr: float) -> list[float]:
    """Adam on one fixed batch: a correct model and gradient memorise it. Returns the loss per step."""
    opt = torch.optim.Adam(tensors.parameters(w), lr=lr)
    losses = []
    for _ in range(n_steps):
        loss = equation.cross_entropy(equation.forward(x, w)[0], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def sample(w: dict, prompt: torch.Tensor, n_new: int, ctx: int, temperature: float, seed: int) -> torch.Tensor:
    """Generate n_new characters one at a time: feed the last ctx tokens, draw the next from softmax(z / temperature).

    Autoregression is a loop over time by nature: every new character depends on the ones drawn before it.
    """
    gen = torch.Generator(device=prompt.device).manual_seed(seed)
    out = prompt.clone()
    for _ in range(n_new):
        logits, _ = equation.forward(out[None, -ctx:], w)
        p = torch.softmax(logits[0, -1] / temperature, dim=-1)
        out = torch.cat([out, torch.multinomial(p, 1, generator=gen)])
    return out[len(prompt) :]


def train(data, val, w, chars, n_steps, batch, ctx, lr, temperature, checkpoints, probe, n_sample=160, seed=0):
    """AdamW with a short warm-up and cosine decay. At each checkpoint: validation loss and a text sample.

    Returns the train loss per step and, at the checkpoints, the validation loss, the sampled text and the
    attention maps (layers, H, T, T) of the fixed `probe` line.
    """
    gen = torch.Generator(device=data.device).manual_seed(seed)
    val_gen = torch.Generator(device=data.device).manual_seed(seed + 1)
    xv, yv = tensors.batch(val, 64, ctx, val_gen)  # one fixed validation batch
    opt = torch.optim.AdamW(tensors.parameters(w), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / n_steps)))
    )
    prompt = probe[:1]  # every sample starts from the first letter of the probe line
    hist = {"train": [], "at": [], "val": [], "text": [], "maps": []}

    def checkpoint(t):
        with torch.no_grad():
            hist["val"].append(equation.cross_entropy(equation.forward(xv, w)[0], yv).item())
            hist["maps"].append(equation.forward(probe[None], w)[1][0].cpu())
        hist["at"].append(t)
        hist["text"].append(tensors.decode(sample(w, prompt, n_sample, ctx, temperature, seed=seed), chars))

    for t in steps(n_steps, "training steps"):
        if t in checkpoints:
            checkpoint(t)
        x, y = tensors.batch(data, batch, ctx, gen)
        loss = equation.cross_entropy(equation.forward(x, w)[0], y)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tensors.parameters(w), 1.0)
        opt.step()
        sched.step()
        hist["train"].append(loss.item())
    checkpoint(n_steps)
    return hist
