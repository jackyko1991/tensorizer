"""Three acts: the batch of profiles trains and the residual funnels form; golden-section steps pin λ down; then the
found profile blows up, u(x, t) = τ^λ U(x/τ^{1+λ}), with max|u_x| ∝ 1/τ. The CCF panel shows the learned vorticity
profile and its λ against the published values."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation

LEVEL_NAMES = ["no constraint", r"$U'''(0)=0$", r"$U'''(0)=U^{(5)}(0)=0$"]
GRAY = "#9ca3af"


def animate(
    lams, funnels, paths, found, exact, taus, x, u, slope, clams, ccf_funnel, ccf, references, path: Path
) -> Path:
    lam_np = lams.cpu().numpy()
    fun = [f.cpu().numpy() for f in funnels]  # (S, B) each
    n_gn = fun[0].shape[0]
    golden = [f["hist"].cpu().numpy() for f in found]
    n_gold = len(golden[0])
    n_blow = len(taus)
    y = torch.linspace(0, 6, 241, dtype=lams.dtype, device=lams.device)

    # profiles of three members of the batch (λ on the grid nearest 1/2, 1/4, 1/6) after every Gauss–Newton step
    members = [int((lams - e).abs().argmin()) for e in exact]
    prof = np.stack(
        [
            np.stack(
                [
                    equation.burgers_profile(paths[lv][s, members[lv]], lams[members[lv]], y)[0].cpu().numpy()
                    for s in range(n_gn)
                ]
            )
            for lv in range(3)
        ]
    )  # (3, S, Y)
    batch_idx = list(range(0, len(lam_np), max(1, len(lam_np) // 8)))
    batch = np.stack(
        [
            np.stack([equation.burgers_profile(paths[0][s, b], lams[b], y)[0].cpu().numpy() for b in batch_idx])
            for s in range(n_gn)
        ]
    )  # (S, nb, Y)
    exact_prof = [equation.burgers_exact(y, i)[0].cpu().numpy() for i in range(3)]
    y_np = y.cpu().numpy()

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.8))
    (ax_p, ax_f, ax_l), (ax_u, ax_s, ax_c) = axes

    # --- profiles
    batch_lines = [ax_p.plot(y_np, batch[0, k], color=GRAY, lw=0.6, alpha=0.6)[0] for k in range(len(batch_idx))]
    for i in range(3):
        ax_p.plot(y_np, exact_prof[i], color=ACCENT[i], lw=1.0, ls="--")
    prof_lines = [
        ax_p.plot(
            y_np, prof[i, 0], color=ACCENT[i], lw=2, label=rf"$\lambda={lam_np[members[i]]:.3f}$, {LEVEL_NAMES[i]}"
        )[0]
        for i in range(3)
    ]
    ax_p.plot([], [], color="#374151", ls="--", lw=1, label=r"exact $y=-U-U^{2i+3}$")
    ax_p.set_xlim(0, 6)
    ax_p.set_ylim(-2.2, 0.3)
    ax_p.set_xlabel("$y$")
    ax_p.set_ylabel("$U(y)$")
    ax_p.legend(fontsize=6.5, loc="lower left")
    title_p = ax_p.set_title("", fontsize=9)

    # --- funnels
    fun_lines = [ax_f.semilogy(lam_np, fun[i][0], color=ACCENT[i], lw=1.6, label=LEVEL_NAMES[i])[0] for i in range(3)]
    for e, name in zip(exact, ["1/2", "1/4", "1/6"]):
        ax_f.axvline(e, color=GRAY, ls=":", lw=1)
        ax_f.text(e, 2e-2, name, fontsize=7, ha="center", color="#4b5563")
    lo = min(f[-1].min() for f in fun)
    ax_f.set_ylim(max(lo, 1e-17) / 3, max(f.max() for f in fun) * 3)
    ax_f.set_xlabel(r"$\lambda$ (fixed while $U$ trains)")
    ax_f.set_ylabel(r"best residual $\|R\|$")
    ax_f.legend(fontsize=7, loc="lower right")
    title_f = ax_f.set_title("", fontsize=9)
    marks = [ax_f.plot([], [], "o", color=ACCENT[i], ms=6, mec="white")[0] for i in range(3)]

    # --- λ search
    steps_g = np.arange(n_gold)
    err = [np.abs(golden[i] - exact[i]).clip(1e-16) for i in range(3)]
    err_lines = [
        ax_l.semilogy(
            [], [], "o-", color=ACCENT[i], ms=3, lw=1.3, label=rf"level {i}: $\lambda\to${found[i]['lam']:.5f}"
        )[0]
        for i in range(3)
    ]
    ax_l.set_xlim(0, n_gold - 1)
    ax_l.set_ylim(1e-8, 0.1)
    ax_l.set_xlabel("golden-section step")
    ax_l.set_ylabel(r"$|\lambda - 1/(2i+2)|$")
    ax_l.legend(fontsize=7, loc="lower left")
    ax_l.set_title(r"pinning $\lambda$ down (profile retrained at each trial $\lambda$)", fontsize=9)

    # --- blow-up
    x_np, u_np, tau_np, s_np = x.cpu().numpy(), u.cpu().numpy(), taus.cpu().numpy(), slope.cpu().numpy()
    ghosts = [ax_u.plot(x_np, u_np[0], color=ACCENT[0], lw=0.8, alpha=0.25)[0] for _ in range(3)]
    u_line = ax_u.plot(x_np, u_np[0], color=ACCENT[0], lw=2)[0]
    ax_u.set_xlim(-1, 1)
    ax_u.set_ylim(-1.1, 1.1)
    ax_u.set_xlabel("$x$")
    ax_u.set_ylabel("$u(x,t)$")
    title_u = ax_u.set_title("", fontsize=9)

    ax_s.loglog(tau_np, s_np, color=ACCENT[0], lw=2, label=r"$\max|u_x|$ from the learned $U$")
    ax_s.loglog(tau_np, 1 / tau_np, color="#374151", ls="--", lw=1, label=r"$1/(T-t)$")
    s_mark = ax_s.plot([tau_np[0]], [s_np[0]], "o", color=ACCENT[1], ms=7)[0]
    ax_s.invert_xaxis()
    ax_s.set_xlabel(r"$T-t$")
    ax_s.set_ylabel(r"$\max|u_x|$")
    ax_s.legend(fontsize=7, loc="upper left")
    ax_s.set_title(r"gradient blow-up: $\max|u_x| = |U'(0)|/(T-t)$", fontsize=9)

    # --- CCF
    yc = ccf["y"].cpu().numpy()
    om = ccf["omega"][:, 0].cpu().numpy()
    ax_c.semilogx(yc, om, color=ACCENT[3], lw=2, label=r"learned $\Omega(y)$, odd")
    ax_c.set_xlim(yc[1], yc[-1])
    ax_c.set_ylim(-0.75, 0.05)
    ax_c.set_xlabel("$y$ (log scale; $\\Omega(-y) = -\\Omega(y)$)")
    ax_c.set_ylabel(r"$\Omega$")
    lam_c = ccf["lam"][0].item()
    ax_c.set_title(rf"CCF stable profile: $\lambda$ = {lam_c:.4f}, $\|R\|$ = {ccf['res'][0]:.0e}", fontsize=9)
    ins = ax_c.inset_axes([0.56, 0.5, 0.41, 0.42])
    ins.semilogy(clams.cpu().numpy(), ccf_funnel.cpu().numpy(), color=ACCENT[3], lw=1.3)
    for k, (name, val) in enumerate(references.items()):
        ins.axvline(val, color=ACCENT[4 + k], ls=":", lw=1.2)
        ax_c.plot([], [], color=ACCENT[4 + k], ls=":", lw=1.2, label=name)
    ins.plot([lam_c], [ccf["res"][0].item()], "o", color=ACCENT[3], ms=5)
    ins.tick_params(labelsize=6)
    ins.set_xlabel(r"$\lambda$", fontsize=7, labelpad=0)
    ins.set_title("CCF funnel vs published", fontsize=7)
    ax_c.legend(fontsize=6.5, loc="lower right")

    def update(frame: int) -> None:
        s = min(frame, n_gn - 1)
        for k, line in enumerate(batch_lines):
            line.set_ydata(batch[s, k])
        for i in range(3):
            prof_lines[i].set_ydata(prof[i, s])
            fun_lines[i].set_ydata(fun[i][s])
        title_p.set_text(f"batch of profiles, Gauss–Newton step {s}/{n_gn - 1}")
        title_f.set_text(f"residual funnels, step {s}: smooth profiles only at $\\lambda = 1/(2i+2)$")
        g = frame - n_gn
        if g >= 0:
            gs = min(g, n_gold - 1)
            for i in range(3):
                err_lines[i].set_data(steps_g[: gs + 1], err[i][: gs + 1])
                j = int(np.abs(lam_np - golden[i][gs]).argmin())
                marks[i].set_data([golden[i][gs]], [fun[i][-1][j]])
        b = frame - n_gn - n_gold
        k = int(np.clip(b, 0, n_blow - 1))
        u_line.set_ydata(u_np[k])
        for m, ghost in enumerate(ghosts):
            ghost.set_ydata(u_np[max(0, k - 3 * (m + 1))])
        s_mark.set_data([tau_np[k]], [s_np[k]])
        title_u.set_text(rf"blow-up with the learned $\lambda$ = {found[0]['lam']:.4f}: $T-t$ = {tau_np[k]:.1e}")

    fig.tight_layout()
    n_total = n_gn + n_gold + n_blow
    return save_gif(fig, update, frame_indices(n_total, 40), path, fps=4)
