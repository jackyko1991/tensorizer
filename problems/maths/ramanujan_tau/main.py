"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n_max = 20000 if fast_mode() else params["n_max"]
    n_exact = 600

    # 1. exact τ(n) in int64 from Δ = q Π (1 − qⁿ)²⁴
    tau_e = solve.tau_exact(n_exact, dev)
    print("τ(1 … 12) =", ", ".join(f"{t:,}" for t in tau_e[1:13].tolist()))
    m, n, ok = equation.multiplicativity(tau_e)
    lhs, rhs = equation.hecke(tau_e, tensors.primes_up_to(n_exact, dev))
    print(
        f"multiplicative: τ(mn) = τ(m)τ(n) on {int(ok.sum())}/{len(ok)} coprime pairs with mn ≤ {n_exact}; "
        f"Hecke recurrence on {int((lhs == rhs).sum())}/{len(lhs)} prime powers"
    )
    print(
        f"Ramanujan noticed that τ(mn) = τ(m)τ(n) for coprime m, n — checked on {len(ok):,} pairs (Mordell proved it)"
    )

    # 2. large n: exact residues modulo a batch of primes, Chinese remainder theorem
    moduli = tensors.crt_moduli(n_max, dev)
    tau = solve.tau_float(n_max, dev)
    print(f"τ({n_max:,}) = {tau[-1]:.6e}  (residues modulo {len(moduli)} primes below {moduli[-1] + 1:,})")
    agree = torch.allclose(tau[1 : n_exact + 1], tau_e[1:].double(), rtol=1e-13, atol=0)
    print(f"  agrees with the int64 series for n ≤ {n_exact}: {agree}")
    t691 = solve.tau_mod(n_max, [691], dev)[0]
    sigma = equation.sigma_mod(n_max, 11, 691, dev)
    same = bool(torch.equal(t691[1:], sigma[1:]))
    print(f"Ramanujan noticed that τ(n) ≡ σ₁₁(n) (mod 691) — {'true' if same else 'FALSE'} for every n ≤ {n_max:,}")
    zeros = int((tau[1:] == 0).sum())
    print(f"Lehmer's conjecture (open): τ(n) ≠ 0 — {zeros} zeros for n ≤ {n_max:,}")

    # 3. Deligne's bound and the Sato–Tate law at the primes
    primes = tensors.primes_up_to(n_max, dev)
    x = equation.normalise(tau[primes], primes)  # (P,)
    worst = int(x.abs().argmax())
    print(
        f"Deligne: |τ(p)| ≤ 2p^(11/2) for all {len(primes):,} primes p ≤ {n_max:,}; "
        f"closest: p = {int(primes[worst]):,}, "
        f"|x_p| = {x.abs().max():.5f}"
    )
    target = equation.semicircle_moments(3)
    print(f"{'moment':>8} {'measured':>9} {'Sato–Tate':>10}")
    print(f"{'E[x]':>8} {x.mean():>9.4f} {0:>10.4f}")
    for k in range(3):
        print(f"{f'E[x^{2 * k + 2}]':>8} {(x ** (2 * k + 2)).mean():>9.4f} {target[k]:>10.4f}")

    data = {"p": primes, "x": x, "target": target}
    return plot.animate(data, out_dir / "ramanujan_tau.gif")


if __name__ == "__main__":
    cli(run, __file__)
