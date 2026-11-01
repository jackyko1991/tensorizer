"""Main loop: training sites → PWM → scan a long sequence (both strands) → empirical p-values → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(params["seed"])
    M = tensors.true_motif(params["strength"], dev)  # (m, 4) hidden motif
    q = tensors.background(dev)  # (4,)
    train = tensors.sample_sites(M, params["n_train"], gen)  # (N, m) aligned binding sites
    pwm = solve.build_pwm(train, params["pseudo"], q)

    seq, starts, strands = tensors.sequence_with_sites(M, params["length"], params["n_sites"], gen)
    scores = solve.scan_both_strands(seq, pwm["W"])  # (2, n)
    null = solve.scan_both_strands(tensors.shuffles(seq, params["shuffles"], gen), pwm["W"])  # (R, 2, n)
    p = equation.p_values(scores, null)
    p_max = 10 ** params["log10_p"]
    hits = solve.call_hits(p, p_max)
    found = solve.match(hits, starts, strands)

    m, n = M.shape[0], scores.shape[1]
    print(f"PWM from {params['n_train']} sites of length {m}   ·   pseudocount β = {params['pseudo']}")
    print("consensus      " + "".join(tensors.BASES[a] for a in pwm["P"].argmax(dim=1).tolist()))
    print("IC (bits)      " + " ".join(f"{v:.2f}" for v in pwm["IC"].tolist()) + f"   total {pwm['IC'].sum():.1f}")
    print(f"max score {pwm['W'].max(dim=1).values.sum():.2f} bits   ·   {2 * n:,} windows scanned (both strands)")
    s_thr = solve.score_threshold(null, p_max)
    print(
        f"null: {null.numel():,} scores from {params['shuffles']} shuffles "
        f"(smallest possible p = {1 / (1 + null.numel()):.1e})"
    )
    print(f"p ≤ {p_max:.0e} ⇔ score > {s_thr:.2f} bits: {len(hits)} hits, about {p_max * 2 * n:.1f} expected by chance")
    for i in range(len(starts)):
        s0, sd = int(starts[i]), int(strands[i])
        print(
            f"  planted {s0:>6} {'+' if sd > 0 else '−'}  score {scores[0 if sd > 0 else 1, s0]:6.2f}  "
            f"p = {p[0 if sd > 0 else 1, s0]:.1e}  {'found' if found[i] else 'missed'}"
        )
    false_hits = [h for h in hits if not any(h == (int(a), int(b)) for a, b in zip(starts, strands))]
    print(f"found {int(found.sum())}/{len(starts)} planted sites   ·   {len(false_hits)} other hits")
    return plot.figure(pwm, scores, null, s_thr, p_max, starts, strands, hits, out_dir / "pwm_motif_scan.png")


if __name__ == "__main__":
    cli(run, __file__)
