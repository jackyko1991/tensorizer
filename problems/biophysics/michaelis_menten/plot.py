"""Animate enzyme kinetics: progress curves (full vs QSSA), the rate law v(S) and a Lineweaver–Burk plot."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import frame_indices, save_gif

S_COLORS = ["#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a"]  # light → dark = low → high S0


def animate(
    times: torch.Tensor,
    full: torch.Tensor,
    P_qssa: torch.Tensor,
    S0: torch.Tensor,
    S_v: torch.Tensor,
    v0: torch.Tensor,
    K_m: torch.Tensor,
    V_max: torch.Tensor,
    K_fit: torch.Tensor,
    V_fit: torch.Tensor,
    names: list[str],
    e: int,
    path: Path,
) -> Path:
    stride = max(1, len(times) // 1500)  # RK4 takes small steps; ~1500 points are plenty for drawing
    t = times[::stride].cpu().numpy()
    S = full[::stride, 1].cpu().numpy()  # (T, B, C)
    ES = full[::stride, 2].cpu().numpy()
    P = full[::stride, 3].cpu().numpy()
    Pq = P_qssa[::stride].cpu().numpy()
    s0, s_v, v_init = S0.cpu().numpy(), S_v.cpu().numpy(), v0.cpu().numpy()
    Km, Vm, Kf, Vf = (x.cpu().numpy() for x in (K_m, V_max, K_fit, V_fit))
    k2 = Vm / full[0, 0, :, 0].cpu().numpy()  # k_cat = V_max / E0
    B, C = len(names), len(s0)
    colors = ["C0", "C1", "C2"]

    fig = plt.figure(figsize=(8.4, 4.4))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.15, 1])
    ax_p = fig.add_subplot(grid[:, 0])
    ax_v = fig.add_subplot(grid[0, 1])
    ax_lb = fig.add_subplot(grid[1, 1])

    # --- progress curves of one enzyme: fraction of substrate turned into product ---
    ax_p.set_xlim(0, t[-1])
    ax_p.set_ylim(0, 1.05)
    ax_p.set_xlabel("time (s)")
    ax_p.set_ylabel("product formed  P / S₀")
    ax_p.set_title(f"enzyme {names[e]}: solid = full kinetics, dashed = QSSA", fontsize=10)
    full_lines = [ax_p.plot([], [], lw=1.8, color=S_COLORS[c], label=f"S₀ = {s0[c]:g} µM")[0] for c in range(C)]
    qssa_lines = [ax_p.plot([], [], "--", lw=1.2, color="#f59e0b")[0] for c in range(C)]
    ax_p.legend(loc="upper left", fontsize=7.5)
    clock = ax_p.text(0.97, 0.04, "", transform=ax_p.transAxes, ha="right", fontsize=9)

    # --- rate law: v = V_max S / (K_m + S) (lines) and the current (S, k2 ES) of every test tube (dots) ---
    S_fine = np.linspace(0, s0.max() * 1.05, 200)
    ax_v.set_xlim(0, S_fine[-1])
    ax_v.set_ylim(0, Vm.max() * 1.1)
    ax_v.set_xlabel("substrate S (µM)", labelpad=1)
    ax_v.set_ylabel("rate v (µM/s)")
    ax_v.set_title("rate law; dots = current (S, k₂·ES)", fontsize=10)
    for b in range(B):
        ax_v.plot(
            S_fine,
            Vm[b] * S_fine / (Km[b] + S_fine),
            "--",
            lw=2 if b == e else 1,
            color=colors[b],
            label=f"{names[b]}: Kₘ = {Km[b]:g} µM",
        )
    ax_v.legend(loc="upper left", fontsize=7)
    dots = [ax_v.plot([], [], "o", ms=3.5, color=colors[b])[0] for b in range(B)]

    # --- Lineweaver–Burk: 1/v against 1/S0 is a straight line; intercepts give 1/V_max and −1/K_m ---
    x = np.linspace(-1.3 / Km.min(), 1.1 / s0.min(), 50)
    for b in range(B):
        ax_lb.plot(1 / s_v[b], 1 / v_init[b], "o", ms=3.5, color=colors[b])
        ax_lb.plot(x, (Kf[b] * x + 1) / Vf[b], "-", lw=1, color=colors[b])
    ax_lb.axhline(0, color="#374151", lw=0.8)
    ax_lb.axvline(0, color="#374151", lw=0.8)
    ax_lb.set_xlim(x[0], x[-1])
    ax_lb.set_ylim(-0.1 * (1 / v_init).max(), 1.1 * (1 / v_init).max())
    ax_lb.set_xlabel("1 / S (1/µM)", labelpad=1)
    ax_lb.set_ylabel("1 / v₀ (s/µM)")
    ax_lb.set_title("Lineweaver–Burk fit (x-intercept = −1/Kₘ)", fontsize=10)

    def update(step: int):
        for c in range(C):
            full_lines[c].set_data(t[: step + 1], P[: step + 1, e, c] / s0[c])
            qssa_lines[c].set_data(t[: step + 1], Pq[: step + 1, e, c] / s0[c])
        for b in range(B):
            dots[b].set_data(S[step, b], k2[b] * ES[step, b])
        clock.set_text(f"t = {t[step]:.1f} s")

    fig.tight_layout()
    return save_gif(fig, update, frame_indices(len(t), 90), path)
