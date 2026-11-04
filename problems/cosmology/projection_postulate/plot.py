"""Animate the shadow of a high-dimensional state, Beta concentration, the projection race, minimal D and the data."""

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation

M_SHOW = [1, 2, 4, 8]


def animate(con, rc, dm, sb, d, path: Path) -> Path:
    Ds = con["Ds"].tolist()
    nD = len(Ds)
    n_pts = min(400, con["dir3"].shape[0])
    dir3 = con["dir3"][:n_pts].cpu().numpy()
    sh3 = con["shadow3"][:n_pts].cpu().numpy()  # (P, n_D, 3)
    q = con["q"].cpu().numpy()  # (T, n_D)
    a, b = con["a"].cpu(), con["b"].cpu()
    m_max = len(rc["m"])
    m_show = [m for m in M_SHOW if m <= m_max]
    K = len(rc["w"])
    w = rc["w"].cpu().numpy()
    f_c, P_c = rc["f_c"].cpu().numpy(), rc["P_c"].cpu().numpy()

    fig = plt.figure(figsize=(11.5, 7.2))
    ax3 = fig.add_subplot(2, 3, 1, projection="3d")
    ax_b, ax_r = fig.add_subplot(2, 3, 2), fig.add_subplot(2, 3, 3)
    ax_t, ax_n, ax_k = fig.add_subplot(2, 3, 4), fig.add_subplot(2, 3, 5), fig.add_subplot(2, 3, 6)

    # --- 3D shadow: grey = normalised psi on the unit sphere, blue = P Psi itself ---
    ax3.scatter(*dir3.T, s=2, color="#9ca3af", alpha=0.5)
    cloud = ax3.scatter(*sh3[:, 0].T, s=5, color=ACCENT[0])
    for v in (ax3.set_xlim, ax3.set_ylim, ax3.set_zlim):
        v(-1, 1)
    ax3.set_box_aspect((1, 1, 1))
    ax3.set_xticks([-1, 0, 1])
    ax3.set_yticks([-1, 0, 1])
    ax3.set_zticks([-1, 0, 1])
    t3 = ax3.set_title("", fontsize=9)

    # --- Beta laws on a log axis (density of log10 q) ---
    edges = np.linspace(-5.5, 0, 70)
    centres = 0.5 * (edges[1:] + edges[:-1])
    xg = 10 ** np.linspace(-5.5, -1e-9, 400)
    import torch

    pdfs = []
    for i in range(nD):
        lp = equation.beta_log_pdf(torch.tensor(xg, dtype=torch.float64), a[i], b[i]).numpy()
        pdfs.append(np.exp(lp) * xg * math.log(10))  # density per unit log10 q
    ax_b.set_xlim(-5.5, 0)
    ax_b.set_ylim(0, max(p.max() for p in pdfs) * 1.15)
    ax_b.set_xlabel("$\\log_{10}\\|P\\Psi\\|^2$")
    ax_b.set_ylabel("density")
    ghosts = [ax_b.plot(np.log10(xg), pdfs[i], color="#d1d5db", lw=0.8, visible=False)[0] for i in range(nD)]
    hist_line = ax_b.step(centres, np.zeros_like(centres), where="mid", color=ACCENT[0], lw=1.2)[0]
    pdf_line = ax_b.plot([], [], color="#111827", lw=1.4, label="Beta$(d, D-d)$")[0]
    mean_line = ax_b.axvline(0, color="#f59e0b", lw=1.2, label="mean $d/D$")
    ax_b.legend(fontsize=7, loc="upper left")
    tb = ax_b.set_title("", fontsize=9)

    # --- race bars ---
    width = 0.8 / (len(m_show) + 1)
    x = np.arange(K)
    ax_r.bar(x - 0.4 + width / 2, w, width, color="white", edgecolor="#111827", label="Born $|c_k|^2$")
    bars, dots = [], []
    for j, m in enumerate(m_show):
        xs = x - 0.4 + (j + 1.5) * width
        bars.append(ax_r.bar(xs, f_c[m - 1], width, color=ACCENT[j % len(ACCENT)], label=f"m = {m}"))
        dots.append(ax_r.plot(xs, P_c[m - 1], "k.", ms=4)[0])
    ax_r.set_xticks(x, [f"k={k + 1}" for k in range(K)])
    ax_r.set_ylabel("probability")
    ax_r.set_ylim(0, max(f_c[m_show[-1] - 1].max(), w.max()) * 1.15)
    ax_r.legend(fontsize=7, ncol=2, loc="upper right")
    tr = ax_r.set_title("", fontsize=9)

    # --- total variation vs m ---
    ax_t.plot(
        rc["a_line"].cpu(), rc["tv_line"].cpu(), color=ACCENT[0], lw=1.6, label="complex (Gamma($m$)), closed form"
    )
    m = rc["m"].cpu().numpy()
    ax_t.plot(m, rc["tv_mc_c"].cpu(), "o", color=ACCENT[0], ms=4, label="Monte Carlo")
    ax_t.plot(m, rc["tv_r"].cpu(), "--", color=ACCENT[1], lw=1.2, label="real (Gamma($m/2$))")
    ax_t.plot(m, rc["tv_mc_r"].cpu(), "s", color=ACCENT[1], ms=3)
    ax_t.axvline(1, color="#16a34a", lw=0.8, ls=":")
    ax_t.text(1.05, 0.002, "Born", color="#166534", fontsize=8)
    ax_t.set_xlabel("hidden dimensions per outcome $m$")
    ax_t.set_ylabel("TV distance from Born")
    ax_t.legend(fontsize=7, loc="lower right")
    cursor = ax_t.axvline(1, color="#f59e0b", lw=1.5)
    ax_t.set_title("deviation grows with $m$", fontsize=9)

    # --- minimal dimension ---
    N = dm["N"].cpu().numpy()
    ax_n.plot(np.log10(N), np.log10(dm["log10_D"].cpu().numpy()), color=ACCENT[3], lw=2)
    for n_lab, txt in ((50.0, "50 qubits"), (1e23, "a mole"), (1e80, "observable\nuniverse")):
        ax_n.plot(math.log10(n_lab), math.log10(n_lab * math.log10(2)), "o", color="#111827", ms=4)
        ax_n.annotate(txt, (math.log10(n_lab), math.log10(n_lab * math.log10(2))), fontsize=7,
                      xytext=(4, -12), textcoords="offset points")  # fmt: skip
    ax_n.set_xlabel("$\\log_{10} N$ (entangled two-level systems)")
    ax_n.set_ylabel("$\\log_{10}\\log_{10} D_{min}$")
    ax_n.set_title(f"P5: $D \\geq 2^N$; Born race needs $D \\geq d + K$ = {dm['D_born']}", fontsize=9)
    n_dot = ax_n.plot([], [], "o", color="#f59e0b", ms=7)[0]

    # --- data: |kappa| vs |a - 1| ---
    av = sb["a"].cpu().numpy()
    kv = np.abs(sb["kappa"].cpu().numpy())
    up, lo = av > 1, av < 1
    ax_k.loglog(av[up] - 1, kv[up], color=ACCENT[0], lw=1.6, label="race, $a > 1$")
    ax_k.loglog(1 - av[lo], kv[lo], color=ACCENT[0], lw=1.0, ls="--", label="race, $a < 1$")
    colors = ["#6b7280", "#9333ea", "#ea580c", "#0891b2", "#dc2626"]
    for j, (r, lim) in enumerate(zip(sb["rows"], sb["limits"])):
        ax_k.axhline(lim, color=colors[j % 5], lw=0.9, ls=":")
        label = f"{r['source'].split(' ')[0]} {r['year']}" + (
            f" ({r['regime'].split(' ')[0]})" if "Kauten" in r["source"] else ""
        )
        ax_k.text(1.3e-6 * (12 ** (j % 3)) if j < 3 else 1.3e-6, lim * 1.15, label, fontsize=6, color=colors[j % 5])
    ax_k.axvspan(1e-6, min(sb["allowed"]), color="#bbf7d0", alpha=0.8, lw=0)
    kc, kr = np.abs(sb["kappa_complex"].cpu().numpy()), np.abs(sb["kappa_real"].cpu().numpy())
    mi = sb["m_int"].cpu().numpy()
    ax_k.plot(mi[1:] - 1, kc[1:], "o", color="#dc2626", ms=4, label="complex $m = 2, 3, \\ldots$ (excluded)")
    ax_k.plot([0.5], [kr[0]], "s", color="#dc2626", ms=4, label="real $m = 1$ (excluded)")
    ax_k.set_xlim(1e-6, 10)
    ax_k.set_ylim(1e-6, 30)
    ax_k.set_xlabel("$|a - 1|$  (complex $m = a$, real $m = 2a$)")
    ax_k.set_ylabel("Sorkin $|\\kappa|$")
    ax_k.legend(fontsize=6, loc="lower right")
    ax_k.set_title(f"data allow $|a-1| \\leq$ {min(sb['allowed']):.0e} (green): $m = 1$", fontsize=9)
    k_dot = ax_k.plot([], [], "o", color="#f59e0b", ms=8, mfc="none", mew=2)[0]

    n_race = 3 * len(m_show)
    hold = 4
    phases = list(range(nD)) + [nD + i for i in range(n_race)] + [nD + n_race - 1] * hold
    phases = [phases[i] for i in frame_indices(len(phases), 120)]

    def update(f: int):
        i = min(f, nD - 1)
        D = Ds[i]
        cloud._offsets3d = tuple(sh3[:, i].T)
        t3.set_text(f"shadow of $\\Psi\\in\\mathbb{{C}}^{{{D}}}$ on 3 real axes\n"
                    f"blue $P\\Psi$ (radius $\\approx\\sqrt{{3/2D}}$), grey $\\psi$: uniform")  # fmt: skip
        for g in ghosts[:i]:
            g.set_visible(True)
        hist, _ = np.histogram(np.log10(q[:, i]), bins=edges, density=True)
        hist_line.set_ydata(hist)
        pdf_line.set_data(np.log10(xg), pdfs[i])
        mean_line.set_xdata([math.log10(d / D)])
        tb.set_text(f"$\\|P\\Psi\\|^2\\sim$ Beta({d}, {D - d}), KS = {con['ks'][i]:.3f}")
        n_dot.set_data([math.log10(max(math.log2(D), 1.0))], [math.log10(max(math.log2(D), 1.0) * math.log10(2))])
        j = (f - nD) // 3 if f >= nD else 0
        mm = m_show[min(j, len(m_show) - 1)]
        for jj, (bb, dd) in enumerate(zip(bars, dots)):
            on = jj == min(j, len(m_show) - 1) or f < nD and jj == 0
            for patch in bb:
                patch.set_alpha(1.0 if on else 0.2)
            dd.set_alpha(1.0 if on else 0.2)
        tv = rc["tv_c"][mm - 1].item()
        tr.set_text(
            f"race $k^* = \\arg\\max\\, |c_k|^2/E_k$, m = {mm}, TV {tv:.3f}\nbars Monte Carlo, dots closed form"
        )
        cursor.set_xdata([mm])
        if mm > 1:
            k_dot.set_data([mm - 1], [kc[mm - 1]])
        else:
            k_dot.set_data([], [])

    fig.subplots_adjust(left=0.06, right=0.98, bottom=0.08, top=0.9, wspace=0.32, hspace=0.42)
    return save_gif(fig, update, phases, path, fps=6)
