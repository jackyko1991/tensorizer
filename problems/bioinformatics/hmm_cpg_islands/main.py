"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import log_emissions


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_iter = 3 if fast_mode() else params["iterations"]
    truth = tensors.true_model(params["gc_island"], dev)  # (π, A, E) that generated the data
    x_codes, z = tensors.planted_sequences(params["n_seq"], params["length"], params["n_islands"], truth[2], 0)
    x = tensors.one_hot(x_codes)  # (B, T, 4)
    B, T = z.shape
    print(f"{B} sequences × {T} letters, {float(z.float().mean()):.1%} of positions inside planted islands")
    for k, name in enumerate(tensors.STATES):
        gc = x[..., 1:3].sum(-1)[z == k].mean()
        print(f"  GC content in {name:10s}: {float(gc):.3f}")

    pi, A, E = tensors.initial_guess(dev)
    models, log_lik, post0 = solve.baum_welch(x, pi, A, E, n_iter)
    for it in sorted({0, 1, 2, n_iter}):
        print(f"Baum–Welch iteration {it:2d}: log P(x) = {log_lik[it]:.1f}")
    pi, A, E = models[-1]
    print("learned emissions  A     C     G     T      (true)")
    for k, name in enumerate(tensors.STATES):
        row = "  ".join(f"{p:.3f}" for p in E[k].tolist())
        true_row = " ".join(f"{p:.2f}" for p in truth[2][k].tolist())
        print(f"  {name:10s}  {row}   ({true_row})")
    print(f"learned mean island length 1/(1 − A_11) = {1 / (1 - float(A[1, 1])):.0f} letters")

    _, _, log_px_true = solve.e_step(x, *truth)
    print(f"log P(x) under the true model: {float(log_px_true.sum()):.1f}")
    path = solve.viterbi(log_emissions(x, E), pi, A)  # (B, T)
    gamma, _, _ = solve.e_step(x, pi, A, E)
    acc_v = float((path == z).double().mean())
    acc_p = float(((gamma[..., 1] > 0.5).long() == z).double().mean())
    print(f"accuracy against the planted islands: Viterbi {acc_v:.1%}, posterior > 0.5 {acc_p:.1%}")

    paths0 = [solve.viterbi(log_emissions(x[:1], m[2]), m[0], m[1])[0] for m in models]  # sequence 0, every model
    return plot.animate(
        x_codes[0], z[0], post0, paths0, models, truth, log_lik, float(log_px_true.sum()), out_dir / "cpg_islands.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
