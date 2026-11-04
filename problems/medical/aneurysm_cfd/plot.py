"""Eight panels over the last cardiac cycle: wall shear stress on the surface; streamlines and tracers inside the
semi-transparent vessel; a velocity slice; ours − OpenFOAM on that slice; the inflow waveform; ours − OpenFOAM TAWSS
on the wall; the pipe check against OpenFOAM; the centreline check against VMTK. The last frames show TAWSS and OSI.
Also writes coloured surfaces as binary PLY files for the interactive 3D view."""

import base64
import xml.etree.ElementTree as ET
import zlib
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from problems._common.plotting import export_plot_data
from problems._common.progress import report

N_SUMMARY = 4  # frames that hold the TAWSS / OSI maps at the end
MM = 1e3


def _box(ax, P):
    c, r = (P.max(0) + P.min(0)) / 2, (P.max(0) - P.min(0)).max() / 2
    ax.set_xlim(c[0] - r, c[0] + r), ax.set_ylim(c[1] - r, c[1] + r), ax.set_zlim(c[2] - r, c[2] + r)
    for lab, f in (("$x$ (mm)", ax.set_xlabel), ("$y$ (mm)", ax.set_ylabel), ("$z$ (mm)", ax.set_zlabel)):
        f(lab, fontsize=7, labelpad=-8)
    ax.tick_params(labelsize=6, pad=-3)
    ax.view_init(elev=14, azim=-62)
    ax.set_box_aspect(None, zoom=1.15)


def wall_mesh(tris: torch.Tensor, ids: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Wall vertices (V, 3) in the same order as tensors.vertices_and_normals, and triangles (F, 3) as indices."""
    wall = tris[ids == 0]
    V, inv = torch.unique((wall.reshape(-1, 3) / 1e-9).round(), dim=0, return_inverse=True)
    return (V * 1e-9).numpy(), inv.reshape(-1, 3).numpy()


def display_mesh(tris, ids, grid, cell: float = 0.7e-3):
    """Decimated wall mesh for display: vertices clustered on a `cell` grid. Returns cluster points (C, 3) in mm,
    triangles (F, 3) and a function mapping per-vertex values (…, V) to per-cluster values (…, C)."""
    _, tri_v = wall_mesh(tris, ids)
    _, cl = torch.unique((grid["V"] / cell).round().long(), dim=0, return_inverse=True)
    cl = cl.numpy()
    nc = cl.max() + 1
    cnt = np.bincount(cl, minlength=nc)
    Vc = np.stack([np.bincount(cl, grid["V"][:, i].numpy(), nc) / cnt for i in range(3)], 1) * MM
    ct = cl[tri_v]
    ct = ct[(ct[:, 0] != ct[:, 1]) & (ct[:, 1] != ct[:, 2]) & (ct[:, 0] != ct[:, 2])]
    return Vc, ct, lambda a: np.stack([np.bincount(cl, row, nc) / cnt for row in np.atleast_2d(a)])


def _b64(a: np.ndarray) -> str:
    """VTK binary DataArray payload: one zlib block, header [1, n, n, compressed size] (UInt32), both base64."""
    raw = np.ascontiguousarray(a).tobytes()
    c = zlib.compress(raw, 6)
    head = np.array([1, len(raw), len(raw), len(c)], dtype="<u4").tobytes()
    return base64.b64encode(head).decode() + base64.b64encode(c).decode()


def write_vtp(path: Path, points, polys=None, lines=None, verts=False, point_data=None) -> Path:
    """Minimal VTK XML PolyData writer (no vtk dependency): Float32 points, Int32 connectivity/offsets, named point
    arrays {name: (values, unit)}. polys: (M, 3) triangles; lines: list of index arrays; verts: one vertex per point."""
    pts = np.asarray(points, dtype="<f4").reshape(-1, 3)

    def arr(a, name, dtype, ncomp=1, unit=None):
        u = f' unit="{unit}"' if unit else ""
        return (f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{ncomp}" format="binary"{u}>'
                f"{_b64(np.asarray(a, dtype='<f4' if dtype == 'Float32' else '<i4'))}</DataArray>")  # fmt: skip

    def cells(conn, offs):
        return arr(conn, "connectivity", "Int32") + arr(offs, "offsets", "Int32")

    counts = {"Verts": 0, "Lines": 0, "Polys": 0}
    body = ""
    if verts:
        counts["Verts"] = len(pts)
        body += "<Verts>" + cells(np.arange(len(pts)), np.arange(1, len(pts) + 1)) + "</Verts>"
    if lines:
        conn = np.concatenate(lines) if lines else np.zeros(0)
        counts["Lines"] = len(lines)
        body += "<Lines>" + cells(conn, np.cumsum([len(x) for x in lines])) + "</Lines>"
    if polys is not None and len(polys):
        counts["Polys"] = len(polys)
        body += "<Polys>" + cells(np.asarray(polys).ravel(), 3 * np.arange(1, len(polys) + 1)) + "</Polys>"
    pd = "".join(arr(v, k, "Float32", unit=u) for k, (v, u) in (point_data or {}).items())
    xml = (
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" header_type="UInt32"'
        ' compressor="vtkZLibDataCompressor"><PolyData>'
        f'<Piece NumberOfPoints="{len(pts)}" NumberOfVerts="{counts["Verts"]}" NumberOfLines="{counts["Lines"]}"'
        f' NumberOfStrips="0" NumberOfPolys="{counts["Polys"]}"><PointData>{pd}</PointData>'
        f"<Points>{arr(pts, 'Points', 'Float32', 3, 'mm')}</Points>{body}</Piece></PolyData></VTKFile>\n"
    )
    Path(path).write_text(xml, encoding="ascii")
    return Path(path)


def read_vtp(path: Path) -> dict:
    """Parse a VTP written by write_vtp back: counts from the Piece and every DataArray decoded (zlib + base64)."""
    root = ET.parse(path).getroot()
    piece = root.find("PolyData/Piece")
    out = {k: int(piece.get(k)) for k in ("NumberOfPoints", "NumberOfVerts", "NumberOfLines", "NumberOfPolys")}
    for da in root.iter("DataArray"):
        text = da.text.strip()
        head = np.frombuffer(base64.b64decode(text[:24]), dtype="<u4")  # 16 bytes → 24 base64 characters
        raw = zlib.decompress(base64.b64decode(text[24:]))
        assert len(raw) == head[1]
        dt = "<f4" if da.get("type") == "Float32" else "<i4"
        a = np.frombuffer(raw, dtype=dt)
        parent = [e.tag for e in root.iter() if da in list(e)][0]
        out[f"{parent}/{da.get('Name')}"] = a.reshape(-1, int(da.get("NumberOfComponents", "1")))
    return out


def write_vtk_series(folder: Path, tris, ids, grid, rec, TA, OS) -> Path:
    """ParaView time series of the recorded cycle: per frame a decimated wall surface (WSS, TAWSS, OSI), the
    streamlines (speed) and the tracers (speed, age), listed in aneurysm.pvd with one group per layer."""
    folder.mkdir(parents=True, exist_ok=True)
    Vc, F, per = display_mesh(tris, ids, grid)
    wss = per(rec["wss"].norm(dim=-1).numpy())
    ta, osi = per(TA.numpy())[0], per(OS.numpy())[0]
    wss_t = np.array(rec["wss_t"])
    tf = (np.array(rec["t"]) - 1e-9) % rec["T"]
    X, S = rec["stream"].numpy() * MM, rec["stream_speed"].numpy()
    entries = []
    for k, t in enumerate(tf):
        j = int(np.abs(wss_t - rec["t"][k]).argmin())
        f1 = write_vtp(folder / f"surface_t{k:03d}.vtp", Vc, polys=F,
                       point_data={"WSS": (wss[j], "Pa"), "TAWSS": (ta, "Pa"), "OSI": (osi, "1")})  # fmt: skip
        pts, lines, spd, n0 = [], [], [], 0
        for s in range(X.shape[1]):
            ok = ~np.isnan(X[k, s, :, 0])
            n = int(ok.sum())
            if n >= 2:
                pts.append(X[k, s, ok]), spd.append(S[k, s, ok]), lines.append(np.arange(n0, n0 + n))
                n0 += n
        f2 = write_vtp(folder / f"streamlines_t{k:03d}.vtp", np.concatenate(pts), lines=lines,
                       point_data={"speed": (np.concatenate(spd), "m/s")})  # fmt: skip
        f3 = write_vtp(folder / f"particles_t{k:03d}.vtp", rec["P"][k].numpy() * MM, verts=True,
                       point_data={"speed": (rec["P_speed"][k].numpy(), "m/s"),
                                   "age": (rec["P_age"][k].numpy(), "s")})  # fmt: skip
        for part, (group, f) in enumerate((("surface", f1), ("streamlines", f2), ("particles", f3))):
            entries.append(f'<DataSet timestep="{t:.4f}" group="{group}" part="{part}" file="{f.name}"/>')
    # the openings (STL attribute 1 inlet, 2, 3 outlets): static layers in a solid colour, shown from the first step
    for name, sel, colour in (("inlet", ids == 1, "#16a34a"), ("outlets", ids >= 2, "#dc2626")):
        cap = tris[sel].reshape(-1, 3).numpy() * MM
        f = write_vtp(folder / f"{name}.vtp", cap, polys=np.arange(len(cap)).reshape(-1, 3))
        entries.append(f'<DataSet timestep="{tf[0]:.4f}" group="{name}" part="{3 + (name == "outlets")}" '
                       f'file="{f.name}" color="{colour}"/>')  # fmt: skip
    pvd = folder / "aneurysm.pvd"
    pvd.write_text('<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">'
                   "<Collection>\n" + "\n".join(entries) + "\n</Collection></VTKFile>\n", encoding="ascii")  # fmt: skip
    return pvd


def animate(tris, ids, grid, mesh, lines, vres, rec, TA, OS, sac, pres, ares, waveform, path: Path, title: str = ""):
    _, tri_v = wall_mesh(tris, ids)
    wss = rec["wss"].norm(dim=-1).numpy()  # (T, V) |τ| in Pa
    # lighter display mesh: cluster the wall vertices on a 0.7 mm grid (values averaged per cluster)
    _, cl = torch.unique((grid["V"] / 0.7e-3).round().long(), dim=0, return_inverse=True)
    cl = cl.numpy()
    nc = cl.max() + 1
    cnt = np.bincount(cl, minlength=nc)
    Vc = np.stack([np.bincount(cl, grid["V"][:, i].numpy(), nc) / cnt for i in range(3)], 1) * MM
    ct = cl[tri_v]
    tri_v = ct[(ct[:, 0] != ct[:, 1]) & (ct[:, 1] != ct[:, 2]) & (ct[:, 0] != ct[:, 2])]
    polys = Vc[tri_v]
    per_cluster = lambda a: np.stack([np.bincount(cl, row, nc) / cnt for row in np.atleast_2d(a)])  # noqa: E731
    wss = per_cluster(wss)
    TAc, OSc = per_cluster(TA.numpy())[0], per_cluster(OS.numpy())[0]
    face = lambda v: v[tri_v].mean(1)  # noqa: E731  vertex values → triangle values
    wss_t = np.array(rec["wss_t"])
    vmax = float(max(wss.max(), TAc.max()))  # full-data limit, fixed over all frames
    norm1 = matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=vmax)  # square-root scale: low sac WSS stays visible
    cmap = plt.get_cmap("inferno", 12)
    Pall = polys.reshape(-1, 3)

    fig = plt.figure(figsize=(15.6, 7.4))
    ax = [fig.add_subplot(2, 4, k + 1, projection="3d" if k in (0, 1, 5) else None) for k in range(8)]

    # 1. wall shear stress on the surface
    surf = Poly3DCollection(polys, linewidths=0)
    surf.set_facecolor(cmap(norm1(face(wss[0]))))
    ax[0].add_collection3d(surf)
    _box(ax[0], Pall)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm1)
    fig.colorbar(sm, ax=ax[0], shrink=0.6, pad=0.1).set_label(r"$|\boldsymbol{\tau}_w|$ (Pa)")
    t1 = ax[0].set_title("wall shear stress", fontsize=10)

    # 2. streamlines (coloured by speed) and tracers inside the semi-transparent vessel
    shell = Poly3DCollection(polys, linewidths=0, facecolor=(0.6, 0.65, 0.75, 0.07))
    ax[1].add_collection3d(shell)
    X, Sp = rec["stream"].numpy() * MM, rec["stream_speed"].numpy()  # (T, S, L, 3), (T, S, L)
    X, Sp = X[:, :, ::2], Sp[:, :, ::2]
    smax_s = float(np.nanmax(np.where(np.isnan(X[..., 0]), np.nan, Sp)))
    snorm = plt.Normalize(0, smax_s)

    def segs(k):
        a, b = X[k, :, :-1], X[k, :, 1:]
        ok = ~(np.isnan(a[..., 0]) | np.isnan(b[..., 0]))
        return np.stack([a[ok], b[ok]], 1), Sp[k, :, :-1][ok]

    s0, c0 = segs(0)
    stream = Line3DCollection(s0, cmap="viridis", norm=snorm, linewidths=0.9, alpha=0.6)
    stream.set_array(c0)
    ax[1].add_collection3d(stream)
    cb_s = fig.colorbar(stream, ax=ax[1], shrink=0.6, pad=0.1)
    cb_s.set_label(r"streamline speed $|\mathbf{u}|$ (m/s)")
    for ln in lines:
        x = ln["x"].numpy() * MM
        ax[1].plot(x[:, 0], x[:, 1], x[:, 2], color="#111827", lw=0.8, alpha=0.5)
    Pp = rec["P"][:, ::4].numpy() * MM  # every 4th tracer in the GIF (all of them in the VTK files)
    in_sac = _in_sac(rec["P"][:, ::4], mesh, sac)
    part = ax[1].scatter(*Pp[0].T, s=2.5, c=np.where(in_sac[0], "#dc2626", "#1e3a8a"), depthshade=False)
    osi_max = max(0.05, float(OSc.max()))
    osi_sm = plt.cm.ScalarMappable(cmap=plt.get_cmap("plasma", 10), norm=plt.Normalize(0, osi_max))
    osi_surf = Poly3DCollection(polys, linewidths=0, visible=False)
    osi_surf.set_facecolor(plt.get_cmap("plasma", 10)(np.clip(face(OSc) / osi_max, 0, 1)))
    ax[1].add_collection3d(osi_surf)
    _box(ax[1], Pall)
    t2 = ax[1].set_title("streamlines, tracers (red: in the sac)", fontsize=10)

    # 3. velocity-magnitude slice through the sac (x–z plane)
    sl = rec["slice"].numpy()  # (T, nx, nz)
    solid = ~grid["inside"].select(1, rec["slice_index"]).numpy()
    lo, h = grid["lo"].numpy() * MM, grid["h"] * MM
    ext = [lo[0] - h / 2, lo[0] + h * (sl.shape[1] - 0.5), lo[2] - h / 2, lo[2] + h * (sl.shape[2] - 0.5)]
    img = ax[2].imshow(np.ma.masked_array(sl[0], solid).T, origin="lower", extent=ext,
                       cmap=plt.get_cmap("viridis", 12),
                       norm=matplotlib.colors.PowerNorm(0.5, vmin=0, vmax=float(sl[:, ~solid].max())))  # fmt: skip
    for a in (ax[2], ax[3]):
        a.set_facecolor("#f3f4f6"), a.grid(False), a.set_xlabel("$x$ (mm)"), a.set_ylabel("$z$ (mm)")
    fig.colorbar(img, ax=ax[2], shrink=0.85).set_label(r"$|\mathbf{u}|$ (m/s)")
    ax[2].set_title(f"slice $y$ = {lo[1] + h * rec['slice_index']:.1f} mm", fontsize=10)

    # 4. ours − OpenFOAM on the slice (same voxel mesh), fixed symmetric scale
    tf = (np.array(rec["t"]) - 1e-9) % rec["T"]
    if ares is not None:
        du = ares["du"].numpy()  # (S, T)
        pts = ares["ref"]["points"].numpy() * MM  # OpenFOAM slice points (cell centres of its 0.4 mm mesh)
        tref = ares["ref"]["t"].numpy() % rec["T"]
        lim = float(np.abs(du).max())  # full-data symmetric limit
        dimg = ax[3].scatter(pts[:, 0], pts[:, 2], c=du[:, 0], cmap="RdBu_r", vmin=-lim, vmax=lim, marker="s", s=14)
        ax[3].set_xlim(ext[0], ext[1]), ax[3].set_ylim(ext[2], ext[3]), ax[3].set_aspect("equal")
        fig.colorbar(dimg, ax=ax[3], shrink=0.85).set_label(r"$\Delta|\mathbf{u}|$ ours − OpenFOAM (m/s)")
        ax[3].set_title(f"vs OpenFOAM: L2 {ares['u_L2'] * 100:.0f} %, L∞ {ares['u_Linf'] * 100:.0f} %", fontsize=10)
    else:
        dimg = None
        ax[3].text(0.5, 0.5, "no OpenFOAM reference\nfor the synthetic geometry", ha="center",
                   va="center", transform=ax[3].transAxes)  # fmt: skip
        ax[3].set_title("ours − OpenFOAM on the slice", fontsize=10)

    # 5. inflow waveform with a cursor
    t_w, Q = waveform
    T = rec["T"]
    ax[4].plot(np.r_[t_w.numpy(), T], np.r_[Q.numpy(), Q[0].item()] * 1e6, color="#111827", lw=1.5,
               label="prescribed $Q(t)$")  # fmt: skip
    ax[4].plot(tf, np.array(rec["Q_in"]) * 1e6, "o", ms=2.5, color="#2563eb", label="discrete inflow")
    ax[4].plot(tf, rec["Q_out"].sum(1).numpy() * 1e6, "x", ms=3, color="#dc2626", label="Σ outflow")
    cursor = ax[4].axvline(tf[0], color="#f59e0b", lw=2)
    ax[4].set_xlabel("time in cycle $t$ (s)"), ax[4].set_ylabel("flow rate $Q$ (ml/s)"), ax[4].set_xlim(0, T)
    ax[4].legend(loc="upper right", fontsize=7)
    vs = _wall_on_sac(grid, mesh, sac)
    ta, os_ = TA.numpy(), OS.numpy()
    note = f"TAWSS sac {np.median(ta[vs]):.2f} Pa, rest {np.median(ta[~vs]):.2f} Pa\n"
    note += f"OSI sac {np.median(os_[vs]):.3f} (max {os_[vs].max():.2f}), rest {np.median(os_[~vs]):.3f}"
    ax[4].text(0.98, 0.42, note, transform=ax[4].transAxes, ha="right", fontsize=7.5)
    ax[4].set_title(f"inflow (mean {Q.mean() * 1e6:.2f} ml/s)", fontsize=10)

    # 6. ours − OpenFOAM TAWSS on the wall
    if ares is not None:
        dta = per_cluster(ares["dtawss"].numpy())[0]
        lim_w = float(np.abs(dta).max())
        dsurf = Poly3DCollection(polys, linewidths=0)
        dsurf.set_facecolor(plt.get_cmap("RdBu_r")(plt.Normalize(-lim_w, lim_w)(face(dta))))
        ax[5].add_collection3d(dsurf)
        dsm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=plt.Normalize(-lim_w, lim_w))
        fig.colorbar(dsm, ax=ax[5], shrink=0.6, pad=0.1).set_label(r"$\Delta$TAWSS ours − OpenFOAM (Pa)")
        ax[5].set_title(f"TAWSS vs OpenFOAM: L2 {ares['tawss_L2'] * 100:.0f} %, L∞ {ares['tawss_Linf'] * 100:.0f} %",
                        fontsize=10)  # fmt: skip
    else:
        ax[5].text2D(
            0.5, 0.5, "no OpenFOAM reference\nfor the synthetic geometry", ha="center", transform=ax[5].transAxes
        )
        ax[5].set_title("TAWSS: ours − OpenFOAM", fontsize=10)
    _box(ax[5], Pall)

    # 7. pipe check: ours vs OpenFOAM and Womersley, residual on a fixed symmetric scale
    tp = pres["t"].numpy() - 1.0
    ua, uo = (x.numpy() for x in pres["u_axis"])
    ax[6].plot(tp, uo, color="#111827", lw=2.2, label="OpenFOAM pimpleFoam")
    ax[6].plot(tp, ua, color="#2563eb", lw=1.0, label="ours (tensors)")
    ax[6].plot(tp, pres["exact"].numpy(), color="#16a34a", lw=0.8, ls="--", label="Womersley exact")
    ax[6].set_xlabel("time in cycle $t$ (s)"), ax[6].set_ylabel(r"pipe axis velocity $u_{\mathrm{axis}}$ (m/s)")
    ax[6].set_xlim(0, 1)
    r7 = ax[6].twinx()
    res = (ua - uo) * 1e3
    lim7 = max(1.0, float(np.ceil(np.abs(res).max())))
    r7.scatter(tp, res, c=res, cmap="RdBu_r", vmin=-lim7, vmax=lim7, s=6)
    r7.axhline(0, color="#9ca3af", lw=0.8), r7.set_ylim(-4 * lim7, lim7 * 1.2), r7.grid(False)
    r7.set_ylabel(r"$\Delta u$ ours − OpenFOAM (mm/s)", fontsize=8)
    ax[6].legend(loc="lower left", fontsize=6.5)
    ax[6].set_title(f"pipe: L2 {pres['u_axis_L2'] * 100:.1f} % (u), {pres['wss_L2'] * 100:.1f} % (WSS)", fontsize=10)
    cursor7 = ax[6].axvline(tf[0], color="#f59e0b", lw=1.5)

    # 8. centreline vs VMTK: radius along the lines and the distance between the two centrelines
    hv = grid["h"]
    for r, col in zip(vres, ["#2563eb", "#9333ea", "#16a34a"]):
        ln = lines[r["k"]]
        ax[7].plot(ln["s"].numpy() * MM, ln["R"].numpy() * MM, color=col, lw=1.5, label=f"ours, outlet {r['k'] + 1}")
        ax[7].plot(r["s_ref"].numpy() * MM, r["R_ref"].numpy() * MM, color="#111827", lw=0.8, ls="--",
                   label="VMTK" if r is vres[0] else None)  # fmt: skip
    ax[7].set_xlabel("abscissa from the inlet $s$ (mm)"), ax[7].set_ylabel("inscribed radius $R$ (mm)")
    ax[7].set_ylim(0, None)
    r8 = ax[7].twinx()
    for r in vres:
        d = r["d"].numpy() / hv
        r8.scatter(r["s_ref"].numpy() * MM, d, c=d, cmap="Reds", vmin=0, vmax=2.5, s=4)
    r8.set_ylim(0, 8), r8.grid(False), r8.set_ylabel("distance to VMTK $d/h$ (voxels)", fontsize=8)
    ax[7].legend(loc="upper right", fontsize=6.5)
    dm = max(r["d"].mean().item() for r in vres) / hv
    ax[7].set_title(f"centreline vs VMTK: mean distance ≤ {dm:.2f} voxel", fontsize=10)
    fig.suptitle(f"Educational voxel CFD — {title} — not for clinical use", fontsize=10)
    n_cyc = len(rec["t"])

    def update(k: int):
        if k < n_cyc:
            j = int(np.abs(wss_t - rec["t"][k]).argmin())
            surf.set_facecolor(cmap(norm1(face(wss[j]))))
            t1.set_text(f"|WSS|, t = {tf[k]:.2f} s")
            s_k, c_k = segs(k)
            stream.set_segments(s_k), stream.set_array(c_k)
            part._offsets3d = tuple(Pp[k].T)
            part.set_color(np.where(in_sac[k], "#dc2626", "#1e3a8a"))
            img.set_data(np.ma.masked_array(sl[k], solid).T)
            if dimg is not None:
                jj = int(np.abs((tref - tf[k] + 0.5) % 1 - 0.5).argmin())  # nearest OpenFOAM sample (periodic)
                dimg.set_array(du[:, jj])
            for c in (cursor, cursor7):
                c.set_xdata([tf[k], tf[k]])
        else:
            surf.set_facecolor(cmap(norm1(face(TAc))))
            t1.set_text("TAWSS (time-averaged |WSS|)")
            part.set_visible(False), stream.set_visible(False), shell.set_visible(False)
            osi_surf.set_visible(True)
            cb_s.update_normal(osi_sm)
            cb_s.set_label("OSI (dimensionless)")
            t2.set_text(f"OSI on the wall (plasma, 0 … {osi_max:.2f}, dimensionless)")

    fig.tight_layout()
    frames = list(range(n_cyc)) + [n_cyc] * N_SUMMARY
    return save_gif_shared_palette(fig, update, frames, path, fps=10)


def _in_sac(P: torch.Tensor, mesh: dict, sac: torch.Tensor) -> np.ndarray:
    """(T, Np) bool: is each tracer in a sac voxel?"""
    g = ((P - mesh["lo"]) / mesh["h"]).round().long()
    shape = torch.tensor(mesh["shape"])
    ok = ((g >= 0) & (g < shape)).all(-1)
    g = g.clamp(min=0).minimum(shape - 1)
    cid = mesh["idx"][g[..., 0], g[..., 1], g[..., 2]]
    return (ok & (cid >= 0) & sac[cid.clamp(min=0)]).numpy()


def _wall_on_sac(grid: dict, mesh: dict, sac: torch.Tensor) -> np.ndarray:
    return (torch.cdist(grid["V"].float(), mesh["pos"][sac].float()).min(1).values < mesh["h"]).numpy()


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 10) -> Path:
    """Render every frame, then quantise all of them with ONE palette built from a strip of frames, so a given
    value keeps the same colour throughout the GIF (Pillow would otherwise pick a new palette per frame)."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)  # numbers behind the final frame (Results page)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)
