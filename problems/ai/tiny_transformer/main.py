"""Main loop: Shakespeare → character tokens → hand-written GPT → Adam on next-character loss → samples and plot."""

import math
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors

EXCERPT = "When shall we three meet again?"  # Macbeth, Act I, Scene I: the line whose attention is drawn


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    text = tensors.load_text()
    chars, data = tensors.encode_text(text, dev)
    split = int(0.9 * len(data))
    train_data, val_data = data[:split], data[split:]
    V = len(chars)
    print(f"text: {len(text)} characters of Macbeth and Romeo and Juliet, vocabulary V = {V}, ln V = {math.log(V):.3f}")

    d, heads, layers, ctx = params["d_model"], params["heads"], params["layers"], params["context"]
    w = tensors.init_weights(V, d, heads, layers, ctx, seed=0, device=dev)
    n_params = sum(t.numel() for t in tensors.parameters(w))
    print(f"GPT: {layers} layers × {heads} heads, d = {d}, context T = {ctx}: {n_params} weights "
          f"(formula {tensors.parameter_count(V, d, layers, ctx)})")  # fmt: skip

    n_steps = 20 if fast else params["steps"]
    n_ck = 3 if fast else 24
    checkpoints = sorted({round(k * n_steps / n_ck) for k in range(n_ck)})
    start = text.index(EXCERPT)
    probe = data[start : start + len(EXCERPT)]  # (L,) the line whose attention maps are drawn
    hist = solve.train(train_data, val_data, w, chars, n_steps, params["batch"], ctx, params["lr"],
                       params["temperature"], set(checkpoints), probe, n_sample=40 if fast else 200)  # fmt: skip
    print(
        f"loss: first batch {hist['train'][0]:.3f} (ln V = {math.log(V):.3f}), "
        f"validation at the end {hist['val'][-1]:.3f}"
    )
    for k in sorted({0, len(hist["at"]) // 3, len(hist["at"]) - 1}):
        first = hist["text"][k].replace("\n", " / ")[:90]
        print(f"  step {hist['at'][k]:4d}  val {hist['val'][k]:.3f}  sample: {first}")

    A = hist["maps"][-1]  # (layers, H, L, L) final model on the probe line
    prev = torch.diagonal(A, offset=-1, dim1=-2, dim2=-1).mean(-1)  # (layers, H) weight on the previous char
    heads_shown = prev.argmax(-1).tolist()
    print(f"attention rows sum to 1: max |Σ_j A_ij − 1| = {(A.sum(-1) - 1).abs().max():.1e}; "
          f"weight above the diagonal: {torch.triu(A, 1).abs().max():.1e}")  # fmt: skip
    for layer, h in enumerate(heads_shown):
        print(f"layer {layer + 1}: head {h + 1} puts {prev[layer, h]:.2f} of its weight on the previous character")
    all_maps = torch.stack(hist["maps"]).numpy()
    return plot.animate(hist, all_maps, EXCERPT, heads_shown, math.log(V), out_dir / "tiny_transformer.gif",
                        hold=1 if fast else 3)  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
