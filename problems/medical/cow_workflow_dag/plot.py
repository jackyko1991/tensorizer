"""The animation: six cases (MRA maximum-intensity projection, threshold segmentation, centreline graph coloured by
flow, inlets and outlets) beside the workflow panels (job DAG by level with the critical path, Gantt chart of Snakemake
against the tensor engine, wall-clock bars, the equivalence table, the threshold gradient and the collateral test).
Also the interactive 3D scenes: the anatomy (segmentation and centrelines by radius) and the flow sweep (wall
shear stress, idealised axial streamlines, particles, inlets and outlets)."""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.collections import LineCollection

from problems._common.plotting import ACCENT  # noqa: F401  (white theme and Agg backend)
from problems.medical.aneurysm_cfd.plot import boundary_glyphs, save_gif_shared_palette, write_vtp

from . import equation as eq
from . import tensors as T

MM = 1e3
GREEN, MAGENTA = "#40e65a", "#f24dd9"  # inlet / outlet colours of the aneurysm problem's glyphs
Q_MAX = 6.0  # flow colour scale (ml/s), fixed over cases, conditions and frames
RULE_COLOURS = {"segment": "#2563eb", "centreline": "#0891b2", "measure": "#16a34a", "boundaries": "#65a30d",
                "flow": "#ea580c", "report": "#9333ea", "cohort": "#db2777", "all": "#374151"}  # fmt: skip


def _case_panel(ax, img, mask, g, b, q, title, norm, cmap):
    """Axial MIP (x → patient left, y → posterior) with the segmentation and the centreline graph coloured by |Q|."""
    X, Y, _ = img.shape
    ext = [-0.5 * T.H * MM, (X - 0.5) * T.H * MM, (Y - 0.5) * T.H * MM, -0.5 * T.H * MM]
    ax.imshow(img.max(2).T, cmap="gray", vmin=0, vmax=1.2, extent=ext, interpolation="nearest")
    seg = np.ma.masked_where(~mask.any(2).T, np.ones((Y, X)))
    ax.imshow(seg, cmap=matplotlib.colors.ListedColormap(["#38bdf8"]), alpha=0.28, extent=ext, interpolation="nearest")
    ok = g["edge_valid"]
    poly = g["poly"][ok].numpy() * MM  # (E, P, 3)
    segs = [p[~np.isnan(p[:, 0])][:, :2] for p in poly]
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidths=1.0 + 1.6 * np.clip(b["r"], 0, 2.5), capstyle="round")
    lc.set_array(np.abs(q[ok.numpy()]))
    ax.add_collection(lc)
    pos = g["node_pos"].numpy() * MM
    for mask_n, col, mk in ((b["inlet"], GREEN, "^"), (b["outlet"], MAGENTA, "o")):
        p = pos[mask_n]
        ax.scatter(p[:, 0], p[:, 1], s=34, c=col, marker=mk, edgecolors="#111827", linewidths=0.6, zorder=5)
    ax.set_xlim(ext[0], ext[1]), ax.set_ylim(ext[2], ext[3])
    ax.set_xlabel(r"$x$ → patient left (mm)", fontsize=10), ax.set_ylabel(r"$y$ → posterior (mm)", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(False)
    t = ax.set_title(title, fontsize=11)
    return lc, t


def animate(store, dag, cases, timing, equiv, grad, path: Path):
    """store: the tensor engine's outputs; timing: per-job Snakemake times and the engine's rule times; equiv: rows of
    the equivalence table; grad: autograd and finite-difference derivatives per case."""
    B = len(cases)
    C = len(T.CONDITIONS)
    img = store["image"].numpy()
    mask = store["segment"]["mask"].numpy()
    g = store["centreline"]
    bnd = store["boundaries"]
    q = store["flow"]["flow"]["q"].detach().numpy().reshape(B, C, -1) * 1e6  # (B, C, E) ml/s
    r_e = store["measure"]["radius"].detach().numpy() * MM
    rep = {k: v.detach().numpy().reshape(B, C) for k, v in store["report"].items()}
    cmap = plt.get_cmap("plasma", 12)
    norm = matplotlib.colors.Normalize(0, Q_MAX)

    fig = plt.figure(figsize=(17.5, 12.6))
    gs = fig.add_gridspec(3, 5, width_ratios=[1, 1, 1, 0.045, 1.15], hspace=0.42, wspace=0.34)
    lcs, titles = [], []
    for k in range(min(B, 6)):
        ax = fig.add_subplot(gs[k // 3, k % 3])
        gk = {kk: v[k] for kk, v in g.items()}
        bk = {
            "inlet": bnd["inlet"][k].numpy(),
            "outlet": bnd["outlet"][k].numpy(),
            "r": r_e[k][gk["edge_valid"].numpy()],
        }
        lc, t = _case_panel(ax, img[k], mask[k], gk, bk, q[k, 0], "", norm, cmap)
        lcs.append(lc), titles.append(t)
    cax = fig.add_subplot(gs[0:2, 3])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cb.set_label(r"centreline flow $|Q_e|$ (ml/s)", fontsize=10)
    cax.tick_params(labelsize=10)

    # job DAG drawn by level, critical path (from Snakemake's 1-core job times) in red
    ax_d = fig.add_subplot(gs[0, 4])
    jobs, A = dag["jobs"], dag["A"].numpy()
    lev = timing["levels"]
    ypos = np.zeros(len(jobs))
    for L in range(lev.max() + 1):
        idx = np.nonzero(lev == L)[0]
        ypos[idx] = np.linspace(-1, 1, len(idx)) * min(1.0, len(idx) / 12) if len(idx) > 1 else 0.0
    cp = set(timing["critical"])
    for i, j in zip(*np.nonzero(A)):
        on = i in cp and j in cp
        ax_d.plot([lev[j], lev[i]], [ypos[j], ypos[i]], color="#dc2626" if on else "#d1d5db", lw=2.2 if on else 0.6,
                  zorder=2 if on else 1)  # fmt: skip
    cols = [RULE_COLOURS[j[0]] for j in jobs]
    nodes = ax_d.scatter(lev, ypos, s=46, c=cols, edgecolors="#111827", linewidths=0.5, zorder=3)
    for name, col in RULE_COLOURS.items():
        ax_d.scatter([], [], s=30, c=col, label=name)
    ax_d.legend(loc="upper center", fontsize=9, ncol=4, frameon=False, handletextpad=0.1, columnspacing=0.5)
    ax_d.set_xlim(-0.4, lev.max() + 0.4), ax_d.set_ylim(-1.2, 1.75)
    ax_d.set_xlabel(r"level $\ell$ (dimensionless)", fontsize=10)
    ax_d.set_ylabel("job within its level (dimensionless)", fontsize=10)
    ax_d.set_yticks([]), ax_d.tick_params(labelsize=10), ax_d.grid(False)
    ax_d.set_title(f"job DAG: {len(jobs)} jobs, {int(A.sum())} edges; red = critical path", fontsize=11)

    # Gantt chart: Snakemake (12 cores, measured) and the tensor engine (one batched call per rule, measured live)
    ax_g = fig.add_subplot(gs[1, 4])
    sm = timing["snakemake12"]
    span = max(r["end"] for r in sm)
    order = sorted(range(len(sm)), key=lambda k: sm[k]["start"])
    for row, k in enumerate(order):
        r = sm[k]
        ax_g.barh(row + 9, r["end"] - r["start"], left=r["start"], color=RULE_COLOURS[r["rule"]], height=0.8)
    for row, (name, (a, b)) in enumerate(timing["engine"].items()):
        if name in RULE_COLOURS:
            ax_g.barh(row % 8, max(b - a, span / 300), left=a, color=RULE_COLOURS[name], height=0.8)
    eng_end = max(b for _, b in timing["engine"].values())
    ax_g.axhline(8.0, color="#9ca3af", lw=0.8)
    ax_g.text(span * 0.99, 3.5, f"tensor engine: {eng_end:.1f} s", ha="right", va="center", fontsize=9)
    ax_g.text(span * 0.02, 9 + len(sm) * 0.85, f"Snakemake, 12 cores: {timing['wall12']:.1f} s", ha="left",
              va="center", fontsize=9)  # fmt: skip
    cursor = ax_g.axvline(0, color="#f59e0b", lw=2)
    ax_g.set_xlim(0, timing["wall12"] * 1.02), ax_g.set_ylim(-1, len(sm) + 10)
    ax_g.set_xlabel(r"wall-clock time $t$ (s)", fontsize=10), ax_g.set_ylabel("job (dimensionless)", fontsize=10)
    ax_g.set_yticks([]), ax_g.tick_params(labelsize=10)
    ax_g.set_title("Gantt: Snakemake jobs vs tensor rules", fontsize=11)

    # wall-clock bars (log scale)
    ax_t = fig.add_subplot(gs[2, 0])
    bars = timing["bars"]
    names, vals = list(bars), list(bars.values())
    colours = ["#6b7280", "#6b7280", "#6b7280", "#d1d5db", "#16a34a", "#2563eb"][: len(vals)]
    ax_t.barh(range(len(vals)), vals, color=colours)
    for k, v in enumerate(vals):
        ax_t.text(v * 1.08, k, f"{v:.1f} s", va="center", fontsize=9)
    ax_t.set_yticks(range(len(vals)), names, fontsize=9.5)
    ax_t.invert_yaxis()
    ax_t.set_xscale("log"), ax_t.set_xlim(0.5, max(vals) * 4)
    ax_t.set_xlabel("wall-clock time for the whole cohort (s)", fontsize=10), ax_t.tick_params(labelsize=10)
    ax_t.set_ylabel("how it was run", fontsize=10)
    ax_t.set_title("time: same work, different orchestration", fontsize=11)

    # equivalence table
    ax_e = fig.add_subplot(gs[2, 1])
    ax_e.axis("off")
    cells = [[r["check"], r["tensor"], r["snakemake"], "✓" if r["ok"] else "✗"] for r in equiv]
    tab = ax_e.table(cellText=cells, colLabels=["check", "tensors", "Snakemake", ""], loc="center",
                     colWidths=[0.40, 0.26, 0.30, 0.07], cellLoc="left")  # fmt: skip
    tab.auto_set_font_size(False), tab.set_fontsize(9.5), tab.scale(1, 1.55)
    for (rr, cc), cell in tab.get_celld().items():
        cell.set_edgecolor("#e5e7eb")
        if cc == 3 and rr > 0:
            cell.get_text().set_color("#16a34a" if equiv[rr - 1]["ok"] else "#dc2626")
    ax_e.set_title("tensor engine ≡ Snakemake", fontsize=11)

    # gradient of the left-MCA flow share with respect to the threshold, autograd vs finite differences
    ax_r = fig.add_subplot(gs[2, 2])
    xk = np.arange(B)
    ax_r.bar(xk, grad["autograd"], color="#2563eb", width=0.6, label="autograd (one backward pass)")
    ax_r.scatter(xk, grad["fd"], color="#dc2626", marker="_", s=260, linewidths=2.5, zorder=3,
                 label="finite differences (2 reruns)")  # fmt: skip
    ax_r.axhline(0, color="#9ca3af", lw=0.8)
    ax_r.set_xticks(xk, cases, rotation=30, fontsize=10)
    lim = 1.25 * max(np.abs(grad["autograd"]).max(), np.abs(grad["fd"]).max())
    ax_r.set_ylim(-lim, lim * 2.0)
    ax_r.set_xlabel("case", fontsize=10)
    ax_r.set_ylabel(r"$\partial s_{\mathrm{L\,MCA}}/\partial\theta$ (1/a.u.)", fontsize=10)
    ax_r.legend(loc="upper left", fontsize=9, frameon=False)
    ax_r.tick_params(labelsize=10)
    ax_r.set_title(r"gradient through the pipeline ($s$ = L-MCA share)", fontsize=11)

    # collateral test: left-hemisphere outflow with the left ICA closed, as % of baseline
    ax_c = fig.add_subplot(gs[2, 3:5])
    keys = [("Q_L-MCA", "L-MCA"), ("Q_L-ACA", "L-ACA"), ("Q_L-PCA", "L-PCA")]
    w = 0.26
    for j, (key, lab) in enumerate(keys):
        base, occl = rep[key][:, 0], rep[key][:, 1]
        pct = np.where(base > 1e-9, 100 * occl / np.maximum(base, 1e-12), np.nan)
        ax_c.bar(xk + (j - 1) * w, pct, width=w, color=ACCENT[j], label=lab)
    ax_c.axhline(100, color="#9ca3af", lw=0.8, ls="--")
    ax_c.set_ylim(0, 128)
    ax_c.set_xticks(xk, cases, rotation=30, fontsize=10)
    ax_c.set_xlabel("case", fontsize=10), ax_c.set_ylabel("outflow, L-ICA closed / open (%)", fontsize=10)
    ax_c.legend(loc="upper center", fontsize=9, ncol=3, frameon=False)
    ax_c.tick_params(labelsize=10)
    ax_c.set_title("collateral test: left ICA inlet closed", fontsize=11)

    n_frames = 24

    def update(f):
        c = 0 if f < n_frames // 2 else 1
        for k in range(len(lcs)):
            ok = g["edge_valid"][k].numpy()
            lcs[k].set_array(np.abs(q[k, c][ok]))
            cond = "baseline" if c == 0 else "L-ICA closed"
            titles[k].set_text(f"{cases[k]} · Dice {rep['dice'][k, 0]:.2f} · {cond}\n"
                               f"Δp {rep['dp_max_mmHg'][k, c]:.1f} mmHg, "
                               f"L-MCA {rep['Q_L-MCA'][k, c]:.2f} ml/s")  # fmt: skip
        t = timing["wall12"] * f / (n_frames - 1)
        cursor.set_xdata([t, t])
        done = np.array([0.0] * len(jobs))
        for k, j in enumerate(jobs):
            e = timing["end12"].get(j)
            done[k] = 1.0 if (e is not None and e <= t) or (e is None and f == n_frames - 1) else 0.25
        nodes.set_alpha(None)
        rgba = matplotlib.colors.to_rgba_array(cols)
        rgba[:, 3] = done
        nodes.set_facecolor(rgba)

    fig.suptitle("Circle of Willis cohort (TopCoW MRA, CC0) as a tensor DAG — the same Snakefile run by Snakemake "
                 "and by tensors; educational flow model, not for clinical use", fontsize=12)  # fmt: skip
    fig.subplots_adjust(left=0.095, right=0.985, top=0.925, bottom=0.07)
    return save_gif_shared_palette(fig, update, list(range(n_frames)), path, fps=3)


# ---------------------------------------------------------------- interactive 3D scene (VTK collection)
def _disc(centre, normal, r, m=16):
    """A flat fan of triangles (m, 3, 3) in the plane ⟂ normal, wound so its area vector points along normal."""
    n = normal / np.linalg.norm(normal)
    u = np.cross(n, [1.0, 0, 0] if abs(n[0]) < 0.9 else [0, 1.0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    t = np.linspace(0, 2 * np.pi, m + 1)
    rim = centre + r * (np.cos(t)[:, None] * u + np.sin(t)[:, None] * v)
    return np.stack([np.repeat(centre[None], m, 0), rim[:-1], rim[1:]], 1)


def _slices(img, offset):
    """Three orthogonal planes through the crop centre as triangle grids at 1 mm, with the MRA intensity per vertex."""
    pts, tris, vals = [], [], []
    X, Y, Z = img.shape
    for axis in range(3):
        k = img.shape[axis] // 2
        plane = np.take(img, k, axis=axis)[::2, ::2]
        a, b = plane.shape
        ia, ib = np.meshgrid(np.arange(a) * 2, np.arange(b) * 2, indexing="ij")
        coords = [None, None, None]
        rest = [d for d in range(3) if d != axis]
        coords[axis], coords[rest[0]], coords[rest[1]] = np.full_like(ia, k), ia, ib
        P = np.stack([c.ravel() for c in coords], 1) * T.H * MM + offset
        base = sum(len(p) for p in pts)
        idx = np.arange(a * b).reshape(a, b)
        q = np.stack([idx[:-1, :-1].ravel(), idx[1:, :-1].ravel(), idx[1:, 1:].ravel(), idx[:-1, 1:].ravel()], 1)
        tris.append(np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]]) + base)
        pts.append(P), vals.append(plane.ravel())
    return np.concatenate(pts), np.concatenate(tris), np.concatenate(vals)


def decimate(V: np.ndarray, F: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: vertices in the same cell (m) merge at their mean; degenerate and repeated triangles go."""
    _, inv = np.unique(np.floor(V / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    cnt = np.bincount(inv, minlength=n)
    Vc = np.stack([np.bincount(inv, V[:, i], n) / cnt for i in range(3)], 1)
    Fc = inv[F]
    Fc = Fc[(Fc[:, 0] != Fc[:, 1]) & (Fc[:, 1] != Fc[:, 2]) & (Fc[:, 0] != Fc[:, 2])]
    _, first = np.unique(np.sort(Fc, axis=1), axis=0, return_index=True)
    return Vc, Fc[np.sort(first)]


def _collection(path: Path, entries: list[str]) -> Path:
    head = '<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0" byte_order="LittleEndian"><Collection>\n'
    path.write_text(head + "\n".join(entries) + "\n</Collection></VTKFile>\n", encoding="ascii")
    return path


RINGS = ((0.0, 1), (0.35, 3), (0.7, 6))  # streamline seeds per vessel: (ρ = distance from the axis / r, angles)
DT = 4e-3  # pseudo-time step (s) the particles advance by between two sweep steps (each step is a steady state)
SEED_SPACING = 3.0  # one particle per this many mm of streamline
STREAM_STEP = 1.5  # streamline point spacing (mm)


def resample(pe: np.ndarray, step: float) -> np.ndarray:
    """A polyline (n, 3) resampled at equal arc length, about `step` apart, end points kept."""
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pe, axis=0), axis=1))]
    n = max(int(np.ceil(s[-1] / step)), 1) + 1
    t = np.linspace(0, s[-1], n)
    return np.stack([np.interp(t, s, pe[:, i]) for i in range(3)], 1)


def transport_frame(pe: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Orthonormal normals (n, 3), binormals (n, 3) along a polyline by parallel transport: each normal is the previous
    one with its component along the new tangent removed, so the frame does not twist around the axis."""
    t = np.gradient(pe, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1, keepdims=True), 1e-12)
    n = np.zeros_like(pe)
    a = np.cross(t[0], [1.0, 0, 0] if abs(t[0, 0]) < 0.9 else [0, 1.0, 0])
    n[0] = a / np.linalg.norm(a)
    for i in range(1, len(pe)):
        a = n[i - 1] - np.dot(n[i - 1], t[i]) * t[i]
        nrm = np.linalg.norm(a)
        n[i] = a / nrm if nrm > 1e-9 else n[i - 1]
    return n, np.cross(t, n)


def axial_streamlines(pe: np.ndarray, r: float, rings=RINGS) -> tuple[np.ndarray, np.ndarray]:
    """Streamlines of axial Poiseuille flow in a tube of radius r around the polyline pe (n, 3): curves parallel to
    the centreline at ρ r from it (ρ, angle from `rings`). Returns points (S, n, 3) and ρ (S,)."""
    nrm, bin_ = transport_frame(pe)
    lines, rho = [], []
    for rh, m in rings:
        for phi in 2 * np.pi * (np.arange(m) + 0.5 * (rh > 0.5)) / m:
            lines.append(pe + rh * r * (np.cos(phi) * nrm + np.sin(phi) * bin_))
            rho.append(rh)
    return np.stack(lines), np.array(rho)


def along(line: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Points at arc lengths s (wrapped into [0, length)) along a polyline (n, 3)."""
    c = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    s = np.mod(s, max(c[-1], 1e-9))
    return np.stack([np.interp(s, c, line[:, i]) for i in range(3)], 1)


def write_scene(folder: Path, store, cases, surfaces, sev, sweep) -> tuple[Path, Path]:
    """Two 3D results, per case tiled 3 × 2 (60 mm apart):
    centreline.pvd — the anatomy pipeline: the segmentation surface and the centreline graph, both coloured by the
      vessel radius (static);
    flow.pvd — the flow of the 1D Poiseuille network, one timestep per left-ICA stenosis severity s (0 open … 1
      closed): the vessel surface coloured by the wall shear stress 4μ|Q|/(πr³) of the nearest centreline edge (and
      its pressure), idealised axial streamlines per vessel (speed 2U(1 − ρ²)), particles moved along them by a
      fixed pseudo-time DT per step, and the inlet (green) and outlet (magenta) discs with arrows."""
    folder.mkdir(parents=True, exist_ok=True)
    B, C = len(cases), len(sev)
    g, bnd, m = store["centreline"], store["boundaries"], store["measure"]
    q = np.stack([fs["q"].detach().numpy() for fs in sweep], 1)  # (B, S, E) m³/s, u → v
    p = np.stack([fs["p"].detach().numpy() for fs in sweep], 1) / T.MMHG  # (B, S, N) mmHg
    steps = [f"{float(s):.2f}" for s in sev]
    r_m = m["radius"].detach().numpy()  # (B, E) m
    r_e = r_m * MM
    anatomy = []
    groups = {name: [] for name in ("inlet", "outlets")}
    per_t = {c: {"surface": [], "streamlines": [], "particles": []} for c in range(C)}
    for k in range(B):
        offset = np.array([60.0 * (k % 3), 55.0 * (k // 3), 0.0])
        ok = g["edge_valid"][k].numpy()
        poly = g["poly"][k].numpy() * MM
        E = int(ok.sum())
        lines_pts, lines_idx, eid, frac = [], [], [], []
        n0 = 0
        for e in range(E):
            pe = poly[e][~np.isnan(poly[e][:, 0])]
            s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pe, axis=0), axis=1))]
            lines_pts.append(pe + offset), lines_idx.append(np.arange(n0, n0 + len(pe)))
            eid.append(np.full(len(pe), e)), frac.append(s / max(s[-1], 1e-9))
            n0 += len(pe)
        LP, EI, FR = np.concatenate(lines_pts), np.concatenate(eid), np.concatenate(frac)
        u, v = g["edges"][k, :E, 0].numpy(), g["edges"][k, :E, 1].numpy()
        V, F = decimate(*surfaces[k], cell=1.0e-3)  # 1 mm clusters: 11 flow steps stay light
        Vmm = V * MM + offset
        Vt, Lt = torch.tensor(Vmm), torch.tensor(LP)
        near = torch.cat([torch.cdist(Vt[i : i + 8192], Lt).argmin(1) for i in range(0, len(Vt), 8192)]).numpy()
        # anatomy (static): segmentation surface and centrelines coloured by radius
        fa = write_vtp(folder / f"centreline_{cases[k]}.vtp", LP, lines=lines_idx,
                       point_data={"radius": (r_e[k][EI], "mm")})  # fmt: skip
        fs = write_vtp(folder / f"segmentation_{cases[k]}.vtp", Vmm, polys=F,
                       point_data={"radius": (r_e[k][EI][near], "mm")})  # fmt: skip
        anatomy += [f'<DataSet timestep="0" group="segmentation" part="1" file="{fs.name}"/>',
                    f'<DataSet timestep="0" group="centrelines" part="2" file="{fa.name}"/>']  # fmt: skip
        # streamline geometry (fixed) and particle seeds (fixed, deterministic) per vessel edge
        rng = np.random.default_rng(k)
        sl, sl_edge, sl_rho, seeds = [], [], [], []
        for e in range(E):
            pe = resample(poly[e][~np.isnan(poly[e][:, 0])], STREAM_STEP) + offset
            X, rho = axial_streamlines(pe, r_e[k, e])
            for line, rh in zip(X, rho):
                c_len = float(np.linalg.norm(np.diff(line, axis=0), axis=1).sum())
                n_p = max(int(c_len / SEED_SPACING), 1)
                sl.append(line), sl_edge.append(e), sl_rho.append(rh)
                seeds.append((np.arange(n_p) + rng.uniform()) * c_len / n_p)
        sl_edge, sl_rho = np.array(sl_edge), np.array(sl_rho)
        sl_pts = np.concatenate(sl)
        n_per = np.array([len(x) for x in sl])
        sl_idx = np.split(np.arange(len(sl_pts)), np.cumsum(n_per)[:-1])
        n_seed = np.array([len(x) for x in seeds])
        shift = np.zeros(len(sl))  # particle displacement along each streamline so far (mm)
        for c in range(C):
            pl = p[k, c][u][EI] * (1 - FR) + p[k, c][v][EI] * FR  # pressure linear along each edge
            wss = eq.wall_shear_stress(q[k, c, :E], r_m[k, :E], T.MU)  # (E,) Pa
            f1 = write_vtp(folder / f"surface_{cases[k]}_t{c}.vtp", Vmm, polys=F,
                           point_data={"WSS": (wss[EI][near], "Pa"), "pressure": (pl[near], "mmHg")})  # fmt: skip
            U = np.abs(q[k, c, :E]) / (np.pi * r_m[k, :E] ** 2)  # mean speed per edge (m/s)
            spd = eq.poiseuille_speed(sl_rho, U[sl_edge])  # (S,) constant along each streamline
            f2 = write_vtp(folder / f"streamlines_{cases[k]}_t{c}.vtp", sl_pts, lines=sl_idx,
                           point_data={"speed": (np.repeat(spd, n_per), "m/s")})  # fmt: skip
            if c > 0:  # a fixed pseudo-time step at this step's speed, in the direction of this step's flow
                shift += np.sign(q[k, c, :E])[sl_edge] * spd * DT * MM
            P = np.concatenate([along(line, s0 + d) for line, s0, d in zip(sl, seeds, shift)])
            f3 = write_vtp(folder / f"particles_{cases[k]}_t{c}.vtp", P, verts=True,
                           point_data={"speed": (np.repeat(spd, n_seed), "m/s")})  # fmt: skip
            for name, f in (("surface", f1), ("streamlines", f2), ("particles", f3)):
                per_t[c][name].append(f)
        # boundary glyphs exactly as in the aneurysm problem: a disc in the cut plane and an arrow along the flow
        pos = g["node_pos"][k].numpy() * MM + offset
        nrm = bnd["normal"][k].numpy()
        be = bnd["edge"][k].numpy()
        for kind, mask in (("inlet", bnd["inlet"][k].numpy()), ("outlets", bnd["outlet"][k].numpy())):
            for n in np.nonzero(mask & (be >= 0))[0]:
                disc = _disc(pos[n], nrm[n], max(r_e[k][be[n]], 0.4))
                tris = torch.tensor(disc / MM)
                ids = torch.full((len(disc),), 1 if kind == "inlet" else 2)
                gl = boundary_glyphs(tris, ids, LP)
                groups[kind].append(gl["inlet" if kind == "inlet" else "outlets"])
    entries = []
    for c in range(C):
        for part, (name, files) in enumerate(per_t[c].items()):
            for f in files:
                entries.append(f'<DataSet timestep="{steps[c]}" group="{name}" part="{part}" file="{f.name}"/>')
    for part, (name, colour) in enumerate((("inlet", GREEN), ("outlets", MAGENTA)), start=3):
        pts, tri, n0 = [], [], 0
        for P_, T_ in groups[name]:
            pts.append(P_), tri.append(T_ + n0)
            n0 += len(P_)
        f = write_vtp(folder / f"{name}.vtp", np.concatenate(pts), polys=np.concatenate(tri))
        for c in range(C):
            entries.append(
                f'<DataSet timestep="{steps[c]}" group="{name}" part="{part}" file="{f.name}" color="{colour}"/>'
            )
    return _collection(folder / "centreline.pvd", anatomy), _collection(folder / "flow.pvd", entries)
