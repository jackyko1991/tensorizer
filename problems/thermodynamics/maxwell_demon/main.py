"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

import torch

from problems._common.plotting import frame_indices
from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    gen = torch.Generator().manual_seed(params["seed"])
    n, eps_sel, tau = params["n_engines"], params["eps"], params["tau"]
    ln2 = math.log(2.0)

    # 1. Szilard engines with a demon, for a batch of measurement error rates at once
    eps = torch.tensor(sorted({0.0, 0.05, 0.1, 0.2, 0.3, 0.5, round(eps_sel, 3)}), dtype=tensors.DTYPE)
    n_steps = math.ceil(tau / tensors.DT - 1e-9)
    rec_a = frame_indices(n_steps + 1, 36)
    sz = solve.szilard(n, eps, tau, gen, dev, record=rec_a)
    E = len(eps)
    W_ext = -sz["W_on"].view(E, n)
    info = sz["info"].view(E, n)
    I_th = equation.mutual_information(eps)
    su = torch.exp(-(sz["W_on"] - sz["dF_m"]) - sz["info"]).view(E, n).mean(1)
    no_info = torch.exp(-(sz["W_on"] - sz["dF_m"])).view(E, n).mean(1)
    print(f"Szilard engines: {n} per error rate, partition moved in τ = {tau} a²/D (kT = 1)")
    print("   ε    I = ln2−H(ε)   ⟨I⟩ sampled   ⟨W_ext⟩ ± se     ⟨e^{-β(W−ΔF)−I}⟩   ⟨e^{-β(W−ΔF)}⟩")
    for e in range(E):
        se = W_ext[e].std() / math.sqrt(n)
        print(f"  {eps[e]:.2f}   {I_th[e]:9.4f}     {info[e].mean():9.4f}    {W_ext[e].mean():7.4f} ± {se:.4f}"
              f"      {su[e]:8.4f}          {no_info[e]:8.4f}")  # fmt: skip
    print(f"Szilard limit kT ln 2 = {ln2:.4f}: every ⟨W_ext⟩ stays below kT·I (second law with information)")
    print(f"removing the partition returns ⟨{-sz['W_remove'].mean():.4f}⟩ kT of spring energy (soft walls)")

    # 2. Landauer erasure, a batch of protocol times
    taus = torch.tensor([0.25, 0.5, 1.0, 2.0, params["tau_erase"]], dtype=tensors.DTYPE).unique()
    rec_b = frame_indices(math.ceil(taus.max().item() / tensors.DT - 1e-9) + 1, 18)
    er = solve.erasure(max(200, n // 2), taus, gen, dev, record=rec_b)
    Wer = er["W"].view(len(taus), -1)
    mean_er, se_er = Wer.mean(1), Wer.std(1) / math.sqrt(Wer.shape[1])
    B_fit = (((mean_er - ln2) / taus).sum() / (1 / taus**2).sum()).item()  # least squares ⟨W⟩ = ln2 + B/τ
    print(f"Landauer erasure (ΔF = {er['dF']:.6f} = kT ln 2): fit ⟨W⟩ = kT ln 2 + B/τ with B = {B_fit:.3f} kT·a²/D")
    for t, m, s, ok in zip(taus, mean_er, se_er, er["success"]):
        print(f"  τ = {t:5.2f}: ⟨W⟩ = {m:.4f} ± {s:.4f} kT   (bit reset in {100 * ok:.1f}% of runs)")
    print("Bérut et al. (2012; 2015): ⟨Q⟩ = kT ln 2 + B/τ with B = 8.15 kT·s; free fit gives A = 0.72 kT (±0.15 kT)")

    # 3. Time reversal: Crooks and path probabilities
    cr = solve.crooks(20000, 0.1, gen, dev)
    slope, intercept = equation.crooks_line(cr["W_forward"], -cr["W_reverse"])
    sig_err = (cr["sigma_path"] - (cr["W_forward"] - cr["dF"])).abs().max().item()
    print(f"Crooks (piston 2a → a in τ = 0.1): ln P_F(W)/P_R(−W) = {slope:.3f} W − {-intercept:.3f};"
          f" ΔF = {cr['dF']:.4f}")  # fmt: skip
    print(f"path log ratio ln P_F[x]/P_R[x̃] = β(W − ΔF) on every trajectory: max error {sig_err:.1e};"
          f" ⟨e^(−σ)⟩ = {torch.exp(-cr['sigma_path']).mean():.4f}, ⟨σ⟩ = {cr['sigma_path'].mean():.3f}")  # fmt: skip

    sel = int((eps - eps_sel).abs().argmin())
    summary = {"eps": eps, "I": I_th, "W_ext": W_ext, "su": su, "sel": sel, "taus": taus, "mean_er": mean_er,
               "se_er": se_er, "B_fit": B_fit, "crooks": cr, "slope": slope, "dt": tensors.DT}  # fmt: skip
    return plot.animate(sz, er, summary, out_dir / "maxwell_demon.gif")


if __name__ == "__main__":
    cli(run, __file__)
