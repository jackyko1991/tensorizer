"""Main loop: planted motif → labelled sequences → train the 1D CNN → read the filters as logos → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors
from .equation import accuracy, conv, filter_pwms, information, motif_similarity

SEED = 0
TEST_FRACTION = 0.2
BATCH = 100


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    fast = fast_mode()
    t0 = time.perf_counter()

    # --- data: random DNA, half of it with one degenerate copy of the motif ---
    consensus = tensors.MOTIFS[params["motif"]]
    motif = tensors.motif_pwm(consensus, params["strength"], dev)  # (W, 4)
    n_seq = 600 if fast else params["n_seq"]
    x, y, _ = tensors.make_dataset(n_seq, params["length"], motif, SEED, dev)  # (N, ℓ, 4), (N,)
    n_test = int(TEST_FRACTION * n_seq)
    x_tr, y_tr, x_te, y_te = x[n_test:], y[n_test:], x[:n_test], y[:n_test]
    print(f"{n_seq} sequences of {params['length']} bp ({n_seq - n_test} train / {n_test} test), half with a "
          f"{params['motif']} site {consensus}, consensus base kept with p = {params['strength']}")  # fmt: skip

    # Reference: scan with the TRUE motif (log-odds vs uniform background), best threshold on the test set.
    true_score = conv(x_te, torch.log(motif / 0.25)[None], torch.zeros(1, device=dev))[:, 0].amax(1)  # (N_test,)
    oracle = max(accuracy(true_score - t, y_te) for t in torch.linspace(0, true_score.max().item(), 200).tolist())
    print(f"reference: scanning with the true PWM classifies {oracle:.1%} of the test set correctly")

    # --- train ---
    weights = tensors.init_weights(params["n_filters"], params["width"], SEED, dev)
    epochs = 2 if fast else params["epochs"]
    hist = solve.train(weights, x_tr, y_tr, x_te, y_te, epochs, params["lr"], BATCH, SEED)
    loss0, loss1 = hist["test_loss"][0], hist["test_loss"][-1]
    print(f"Adam, {epochs} epochs × {len(hist['loss']) // epochs} batches: test loss {loss0:.3f} → {loss1:.3f}, "
          f"accuracy train {hist['train_acc'][-1]:.1%}, test {hist['test_acc'][-1]:.1%}")  # fmt: skip

    # --- open the box: which filter matters, and what has it learned? ---
    with torch.no_grad():
        m = torch.relu(conv(x_te, weights["W"], weights["c"])).amax(2)  # (N_test, F)
        importance = weights["v"] * (m[y_te == 1].mean(0) - m[y_te == 0].mean(0))  # (F,)
        best = int(importance.argmax())
        # Read each snapshot of that filter back as a PWM: average its best-matching windows in the positive test set.
        pos = x_te[y_te == 1]
        pwm_snaps = torch.stack([filter_pwms(pos, W, c)[best] for W, c in hist["snapshots"]]).cpu()  # (S, w, 4)
        r_snaps = [motif_similarity(p, motif.cpu())[0] for p in pwm_snaps]
        shift = motif_similarity(pwm_snaps[-1], motif.cpu())[1]  # where the motif sits inside the filter
        r_all = [motif_similarity(p, motif)[0] for p in filter_pwms(pos, weights["W"], weights["c"])]
    bits = information(pwm_snaps[-1]).sum()
    print(f"most important filter: #{best + 1} (importance {importance[best]:.2f}): correlation with the planted "
          f"motif r = {r_snaps[0]:.2f} at step 1 → {r_snaps[-1]:.3f}, {bits:.1f} bits")  # fmt: skip
    print("final r of every filter: " + ", ".join(f"{r:.2f}" for r in r_all))
    print(f"total time {time.perf_counter() - t0:.1f} s")

    return plot.animate(
        motif.cpu(), consensus, pwm_snaps, r_snaps, hist, oracle, best, shift, out_dir / "dna_binding_cnn.gif"
    )


if __name__ == "__main__":
    cli(run, __file__)
