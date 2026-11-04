"""Train direct and chain-of-thought models in one batched run, then measure what each of them learned.

All models (length n × format × seed, plus the length-generalisation models) are stacked along the model axis
of one weight dict, see by `tensors.init_weights`; one AdamW step updates all of them at once, and gradient
clipping is done per model so the models never influence one another.
"""

import math
from dataclasses import dataclass, field

import torch

from problems._common.progress import steps as progress_steps

from . import equation, tensors


@dataclass
class Config:
    lengths: list[int] = field(default_factory=lambda: [4, 6, 8, 10])  # n of experiment (a)
    seeds: int = 2  # independent replicates (different weights and data)
    steps: int = 600
    d: int = 64
    heads: int = 4
    layers: int = 2
    batch: int = 32
    lr: float = 3e-3
    train_max: int = 8  # experiment (b): trained on 1 ≤ n ≤ train_max ...
    gen_lengths: list[int] = field(default_factory=lambda: [4, 8, 12, 16])  # ... and tested on these n
    test_size: int = 256
    curve_size: int = 64


def clip_per_model(params: list[torch.Tensor], max_norm: float) -> None:
    """Scale model m's gradient so that its own norm is at most max_norm (a global clip would couple the models)."""
    sq = sum(p.grad.pow(2).flatten(1).sum(1) for p in params)  # (M,)
    scale = (max_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for p in params:
        p.grad.mul_(scale.view(-1, *[1] * (p.dim() - 1)))


def overfit(x, y, mask, w, n_steps: int, lr: float) -> torch.Tensor:
    """Adam on one fixed batch per model (x, y, mask (M, N, T)): the per-model loss for every step (steps, M)."""
    opt = torch.optim.Adam(tensors.parameters(w), lr=lr)
    losses = []
    for _ in range(n_steps):
        loss = equation.masked_loss(equation.forward(x, w)[0], y, mask)
        opt.zero_grad()
        loss.sum().backward()
        opt.step()
        losses.append(loss.detach())
    return torch.stack(losses)


def _batch(bits, n, cot, ctx):
    """make_batch for stacked models: bits (M, N, n_max), n (M, N), cot (M,) → three tensors (M, N, T)."""
    m, rows = n.shape
    flat_cot = cot[:, None].expand(m, rows).reshape(-1)
    out = tensors.make_batch(bits.reshape(m * rows, -1), n.reshape(-1), flat_cot, ctx)
    return tuple(t.reshape(m, rows, ctx) for t in out)


def _at(logits: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """logits (M, N, T, V) at position pos (M, N) → (M, N, V)."""
    return logits.gather(2, pos[..., None, None].expand(*pos.shape, 1, logits.shape[-1]))[:, :, 0]


@torch.no_grad()
def greedy(w, bits, n, cot, ctx):
    """Let every model write its own scratchpad, then answer.  bits (M, N, n_max), n (M, N), cot (M,).

    CoT models fill s_1 … s_n one token at a time (step k reads the logits at position n + k − 1 and writes
    position n + k), so later cells are their own guesses, never the truth; the oracle cells that are still to
    the right cannot leak because of the causal mask. Returns the tokens (M, N, T) and the answers (M, N).
    """
    tok = _batch(bits, n, cot, ctx)[0]
    for k in range(1, int(n.max()) + 1) if bool(cot.any()) else []:  # autoregression: a loop over steps
        pos = (n + k - 1).clamp(max=ctx - 2)
        guess = equation.bit_choice(_at(equation.forward(tok, w)[0], pos))  # (M, N)
        write = cot[:, None] & (k <= n)
        old = tok.gather(2, (pos + 1)[..., None])[..., 0]
        tok = tok.scatter(2, (pos + 1)[..., None], torch.where(write, guess, old)[..., None])
    return tok, answer(w, tok, n, cot)


@torch.no_grad()
def answer(w, tok, n, cot):
    """The model's p: its bit choice at the "=" token (position 2n + 1 for CoT, n for direct).  → (M, N)"""
    pos = torch.where(cot[:, None], 2 * n + 1, n)
    return equation.bit_choice(_at(equation.forward(tok, w)[0], pos))


def _test_bits(split, ns: list[int], size: int, gen) -> torch.Tensor:
    """Test strings for a stack of models of lengths ns → (M, size, n_max)."""
    return torch.stack([split.sample_test(n, size, gen) for n in ns])


@torch.no_grad()
def chain_right(w, x, y, mask) -> torch.Tensor:
    """Teacher-forced check of every bit the model must write (s_1 … s_n and p, or p alone).  → (M, N) bool.

    If every next-bit guess is right given the true prefix, the model's own greedy chain is exactly the true
    chain, so this equals "the model writes the whole chain and the answer correctly" in a single forward pass.
    """
    due = mask & (y < 2)  # positions where a bit is due (not the "=" token)
    ok = (equation.bit_choice(equation.forward(x, w)[0]) == y) | ~due
    return ok.all(-1)


def run_experiments(cfg: Config, dev: torch.device, n_checkpoints: int = 24, length_gen: bool = True, seed: int = 0):
    """Experiments (a)–(d) from one training run. Returns a dict of CPU tensors (see main.py).

    Models come in groups that share a context length T: one group per n of experiment (a) holding the direct
    and the CoT models of every seed, (2S models stacked on the model axis, the direct strings padded with blanks),
    plus one group for experiment (b). Every step trains all groups and sums their losses: one AdamW step.
    """
    L, S = len(cfg.lengths), cfg.seeds
    n_max = max(cfg.lengths + cfg.gen_lengths + [cfg.train_max])
    V = len(tensors.VOCAB)
    fmt = torch.tensor([False] * S + [True] * S, device=dev)  # (2S,) model order inside a group: direct, then CoT
    split = tensors.StringSplit(sorted(set(cfg.lengths + cfg.gen_lengths)), seed=seed + 100, device=dev, n_max=n_max)
    groups = [{"n": n, "ctx": 2 * n + 3} for n in cfg.lengths]
    groups += [{"n": 0, "ctx": 2 * n_max + 3}] if length_gen else []
    for g_i, g in enumerate(groups):  # n = 0 marks the length-generalisation group (random n ≤ train_max)
        g["w"] = tensors.init_weights(2 * S, V, cfg.d, cfg.heads, cfg.layers, g["ctx"], seed=seed + g_i, device=dev)
    params = [p for g in groups for p in tensors.parameters(g["w"])]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / 30) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t / cfg.steps)))
    )
    gen = torch.Generator(device=dev).manual_seed(seed + 1)
    test_gen = torch.Generator(device=dev).manual_seed(seed + 2)
    curve = []  # fixed held-out strings per group of (a), in both formats
    for g in groups[:L]:
        bits = _test_bits(split, [g["n"]] * 2 * S, cfg.curve_size, test_gen)
        curve.append(_batch(bits, torch.full((2 * S, cfg.curve_size), g["n"], device=dev), fmt, g["ctx"]))
    checkpoints = sorted({round(k * cfg.steps / n_checkpoints) for k in range(n_checkpoints)} | {cfg.steps})
    hist = {"at": [], "acc": [], "loss": []}

    def checkpoint(t):
        hist["at"].append(t)
        hist["acc"].append(torch.stack([chain_right(g["w"], *c).float().mean(-1).cpu() for g, c in zip(groups, curve)]))

    for t in progress_steps(cfg.steps, "training steps"):
        if t in checkpoints:
            checkpoint(t)
        total, losses = 0.0, []
        for g in groups:
            if g["n"]:
                n = torch.full((2 * S, cfg.batch), g["n"], device=dev)
            else:
                n = torch.randint(1, cfg.train_max + 1, (2 * S, cfg.batch), generator=gen, device=dev)
            t_len = g["ctx"] if g["n"] else 2 * cfg.train_max + 3  # (b) trains on the short strings only
            x, y, mask = _batch(split.sample_train(n, gen), n, fmt, t_len)  # never a held-out string
            loss = equation.masked_loss(equation.forward(x, g["w"])[0], y, mask)  # (2S,)
            total = total + loss.sum()
            losses.append(loss.detach().cpu())
        opt.zero_grad()
        total.backward()  # each model's loss only reaches its own weights
        for g in groups:
            clip_per_model(tensors.parameters(g["w"]), 1.0)
        opt.step()
        sched.step()
        hist["loss"].append(torch.stack(losses[:L]))
    checkpoint(cfg.steps)

    res = {"cfg": cfg, "M": 2 * S * len(groups), "at": torch.tensor(hist["at"]),
           "curve_acc": torch.stack(hist["acc"]).reshape(-1, L, 2, S),  # (C, L, f, S): whole chain right
           "loss": torch.stack(hist["loss"]).reshape(-1, L, 2, S),  # (steps, L, f, S)
           "held_out": [split.held_out[n] for n in cfg.lengths],
           "params_per_model": [tensors.parameter_count(V, cfg.d, cfg.layers, g["ctx"]) for g in groups]}  # fmt: skip

    # (a) final accuracy, (c) hidden scratchpad steps, oracle chain: one group at a time
    acc = torch.zeros(L, 2, S)
    hidden = torch.zeros(L, S, max(cfg.lengths) + 1)
    oracle = torch.zeros(L, S)
    for li, g in enumerate(groups[:L]):
        n_val, ctx = g["n"], g["ctx"]
        bits = _test_bits(split, [n_val] * 2 * S, cfg.test_size, test_gen)
        nn = torch.full((2 * S, cfg.test_size), n_val, device=dev)
        parity = bits.sum(-1) % 2  # cells past n are 0
        tok, ans = greedy(g["w"], bits, nn, fmt, ctx)
        acc[li] = (ans == parity).float().mean(-1).reshape(2, S).cpu()
        w_cot = tensors.select(g["w"], list(range(S, 2 * S)))
        cot_tok, cot_n, cot_flag = tok[S:], nn[S:], fmt[S:]
        j = torch.arange(ctx, device=dev)
        for k in range(max(cfg.lengths) + 1):  # keep the first k model-written steps, blank out s_{k+1} … s_n
            hidden_tok = torch.where((j > n_val + k) & (j <= 2 * n_val), tensors.BLANK, cot_tok)
            hidden[li, :, k] = (answer(w_cot, hidden_tok, cot_n, cot_flag) == parity[S:]).float().mean(-1).cpu()
        oracle_tok = _batch(bits[S:], cot_n, cot_flag, ctx)[0]  # the true chain fed in
        s_n = equation.running_parity(bits[S:, :, :n_val])[..., -1]
        oracle[li] = (answer(w_cot, oracle_tok, cot_n, cot_flag) == s_n).float().mean(-1).cpu()
    res.update(acc=acc, hidden=hidden, oracle_answer_agrees=oracle)

    # (d) attention of the CoT model (seed 0) at one length: where the query that writes s_k looks
    n_show = 8 if 8 in cfg.lengths else max(cfg.lengths)
    g = groups[cfg.lengths.index(n_show)]
    w1 = tensors.select(g["w"], [S])
    bits = _test_bits(split, [n_show], cfg.test_size, test_gen)
    nn = torch.full((1, cfg.test_size), n_show, device=dev)
    tok, ans = greedy(w1, bits, nn, fmt[S:][:1], g["ctx"])
    with torch.no_grad():
        maps = equation.forward(tok, w1)[1][0]  # (N, layers, H, T, T)
    k = torch.arange(1, n_show + 1, device=dev)
    rows = n_show + k - 1  # the query that predicts s_k sits on s_{k-1} (on ":" for k = 1)
    on_b_k = maps[..., rows, k - 1].mean(0)  # (layers, H, n) weight on b_k at every step k
    on_b = on_b_k.mean(-1)  # (layers, H)
    on_prev = maps[..., rows, rows].mean((0, -1))  # weight on itself, s_{k-1}
    best = divmod(int(on_b.flatten().argmax()), cfg.heads)
    res.update(n_show=n_show, on_b=on_b.cpu(), on_prev=on_prev.cpu(), best_head=best,
               sample_tok=tok[0, 0].cpu(), sample_answer=int(ans[0, 0]),
               sample_parity=int(bits[0, 0].sum() % 2), sample_maps=maps[0].cpu(), on_b_k=on_b_k.cpu())  # fmt: skip

    # (b) length generalisation: models trained on n ≤ train_max, tested on every n of (a)
    if length_gen:
        g = groups[-1]
        gen_acc = torch.zeros(2, S, len(cfg.gen_lengths))
        for li, n_val in enumerate(cfg.gen_lengths):
            bits = _test_bits(split, [n_val] * 2 * S, cfg.test_size, test_gen)
            nn = torch.full((2 * S, cfg.test_size), n_val, device=dev)
            _, ans = greedy(g["w"], bits, nn, fmt, g["ctx"])
            gen_acc[:, :, li] = (ans == bits.sum(-1) % 2).float().mean(-1).reshape(2, S).cpu()
        res["gen_acc"] = gen_acc
    return res
