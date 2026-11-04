"""Animate four remnants being fed gas, in 3D (top row) and as curves: a brown dwarf crossing the hydrogen-burning
limit, nova cycles on a white dwarf, X-ray bursts on a neutron star and a black hole that only shines (up to
Eddington), plus the batched ignition-mass surveys; and the same four as interactive VTK scenes (export_vtk)."""

import base64
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from problems._common.plotting import ACCENT, frame_indices, save_gif

from . import equation as eq
from .tensors import T_IGNITE


def _np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def _small(ax, title):
    ax.set_title(title, fontsize=8)
    ax.tick_params(labelsize=6)


def _sphere(n_u: int = 22, n_v: int = 12):
    u, v = np.meshgrid(np.linspace(0, 2 * np.pi, n_u), np.linspace(0, np.pi, n_v))
    return np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)


def _stage(ax, title: str, half: float) -> None:
    """Clear a 3D panel and set an equal, axis-free box of half-width `half`."""
    ax.cla()
    ax.set_axis_off()
    ax.set_box_aspect((1, 1, 1), zoom=1.45)  # matplotlib leaves wide margins round 3D boxes
    for setter in (ax.set_xlim, ax.set_ylim, ax.set_zlim):
        setter(-half, half)
    ax.set_title(title, fontsize=7.5)


def clocks(d: dict, s: np.ndarray) -> dict:
    """Each panel's own clock at progress s ∈ [0, 1] (shared by the GIF and the 3D scenes): the brown dwarf's mass and
    central temperature (Myr), the nova shell / ΔM_ign (yr), the burst column / y_ign (hours) and the black hole's
    accretion-rate sweep with L/L_Edd."""
    M, T = _np(d["M_bd"]) / eq.M_SUN, _np(d["T_max"])
    m0, rate, m_h = d["bd_m0"], d["bd_rate"], d["m_h"]
    t_end = 1.3 * max(m_h - m0, 0.2 * m_h) / rate
    m_t = (m0 + rate * s * t_end) / eq.M_SUN
    mine, me = d["mine"], d["me"]
    t_rec = (me["dM"][0] / mine["Mdot"][0]).item() / eq.YEAR
    t_wd = s * 3.3 * t_rec
    t_b = (me["dM"][1] / mine["Mdot"][1]).item() / 3600
    t_ns = s * 4.3 * t_b
    M_bh, eta = d["M_bh"], d["eta"]
    ratio = np.logspace(-3, 1.5, len(s))
    mdot = torch.tensor(ratio, dtype=torch.float64) * eq.eddington_rate(M_bh, eta).item()
    return {
        "t_end": t_end, "t_bd": s * t_end / eq.YEAR / 1e6, "m_t": m_t, "T_t": np.interp(np.log(m_t), np.log(M), T),
        "t_rec": t_rec, "t_wd": t_wd, "shell": (t_wd % t_rec) / t_rec,
        "t_b": t_b, "t_ns": t_ns, "column": (t_ns % t_b) / t_b,
        "ratio": ratio, "lum": _np(eq.black_hole_luminosity(mdot, M_bh, eta)) / d["L_edd"].item(),
    }  # fmt: skip


def animate(d: dict, path: Path) -> Path:
    n = 48
    frames = frame_indices(n, 48)
    s = np.linspace(0, 1, n)  # progress through each panel's own clock

    fig = plt.figure(figsize=(11, 8.8))
    grid = fig.add_gridspec(3, 12, height_ratios=[1.1, 1, 1])
    ax3 = [fig.add_subplot(grid[0, 3 * i : 3 * i + 3], projection="3d") for i in range(4)]
    ax_bd, ax_wd, ax_ns = (fig.add_subplot(grid[1, 4 * i : 4 * i + 4]) for i in range(3))
    ax_bh, ax_wds, ax_nss = (fig.add_subplot(grid[2, 4 * i : 4 * i + 4]) for i in range(3))
    X, Y, Z = _sphere()
    shade = 0.55 + 0.45 * np.clip(0.6 * X - 0.5 * Y + 0.6 * Z, 0, 1)  # light from the upper front

    # --- brown dwarf: T_max(M) and an accreting brown dwarf walking up the curve ---
    M = _np(d["M_bd"]) / eq.M_SUN
    T = _np(d["T_max"])
    ax_bd.loglog(M, T, color=ACCENT[0], lw=1.5, label="max central T (one-zone)")
    for key, lab, col in (("D", "D burning", ACCENT[2]), ("H", "H burning", ACCENT[3])):
        ax_bd.axhline(T_IGNITE[key], color=col, ls="--", lw=0.8)
        ax_bd.text(M[0] * 1.1, T_IGNITE[key] * 1.12, lab, fontsize=6, color=col)
    ax_bd.axvline(d["m_h"] / eq.M_SUN, color=ACCENT[3], lw=0.8)
    ax_bd.axvline(d["m_d"] / eq.M_SUN, color=ACCENT[2], lw=0.8)
    ax_bd.set_xlabel("mass (M☉)", fontsize=7)
    ax_bd.set_ylabel("K", fontsize=7)
    ax_bd.set_ylim(T.max() * 1e-3, T.max() * 2)
    clk = clocks(d, s)
    m_h, t_end, m_t, T_t = d["m_h"], clk["t_end"], clk["m_t"], clk["T_t"]
    bd_dot = ax_bd.plot([], [], "o", color=ACCENT[1], ms=6)[0]
    bd_txt = ax_bd.text(0.97, 0.05, "", transform=ax_bd.transAxes, ha="right", fontsize=6.5)
    _small(ax_bd, f"brown dwarf fed gas: H ignites above {m_h / eq.M_SUN:.3f} M☉")

    # --- white dwarf: shell grows to ΔM_ign, nova, repeat ---
    mine, me = d["mine"], d["me"]
    dm_wd = me["dM"][0].item() / eq.M_SUN
    t_rec, t_wd, shell = clk["t_rec"], clk["t_wd"], clk["shell"]  # shell in units of the ignition mass
    ax_wd.set_xlim(0, t_wd[-1])
    ax_wd.set_ylim(0, 1.3)
    ax_wd.axhline(1, color=ACCENT[3], ls="--", lw=0.8)
    ax_wd.text(0.02 * t_wd[-1], 1.12, f"ignition: ΔM = {dm_wd:.1e} M☉", fontsize=6, color=ACCENT[3])
    wd_line = ax_wd.plot([], [], color=ACCENT[0], lw=1.5)[0]
    wd_flash = ax_wd.plot([], [], "*", color=ACCENT[1], ms=12)[0]
    ax_wd.set_xlabel("time (yr)", fontsize=7)
    ax_wd.set_ylabel("accreted shell / ΔM_ign", fontsize=7)
    m_wd = mine["M"][0].item() / eq.M_SUN
    _small(ax_wd, f"{m_wd:.2f} M☉ white dwarf: classical nova every {t_rec:.1e} yr")

    # --- neutron star: column grows to y_ign, X-ray burst, repeat ---
    y_ns = me["y"][1].item()
    t_b, t_ns, column = clk["t_b"], clk["t_ns"], clk["column"]  # column in units of the ignition column
    ax_ns.set_xlim(0, t_ns[-1])
    ax_ns.set_ylim(0, 1.3)
    ax_ns.axhline(1, color=ACCENT[3], ls="--", lw=0.8)
    ax_ns.text(0.02 * t_ns[-1], 1.12, f"ignition: y = {y_ns:.1e} g/cm²", fontsize=6, color=ACCENT[3])
    ns_line = ax_ns.plot([], [], color=ACCENT[0], lw=1.5)[0]
    ns_flash = ax_ns.plot([], [], "*", color=ACCENT[1], ms=12)[0]
    ax_ns.set_xlabel("time (hours)", fontsize=7)
    ax_ns.set_ylabel("accreted column / y_ign", fontsize=7)
    _small(ax_ns, f"neutron star ({d['comp']} fuel): X-ray burst every {t_b:.1f} h")

    # --- black hole: luminosity vs accretion rate, capped at Eddington ---
    eta, ratio, lum = d["eta"], clk["ratio"], clk["lum"]
    ax_bh.loglog(ratio, lum, color="#d1d5db", lw=1)
    ax_bh.axhline(1, color=ACCENT[3], ls="--", lw=0.8)
    ax_bh.text(1.2e-3, 1.25, "Eddington limit", fontsize=6, color=ACCENT[3])
    bh_line = ax_bh.plot([], [], color="black", lw=1.8)[0]
    bh_dot = ax_bh.plot([], [], "o", color="black", ms=6)[0]
    ax_bh.set_xlabel("accretion rate / Ṁ_Edd", fontsize=7)
    ax_bh.set_ylabel("L / L_Edd", fontsize=7)
    ax_bh.set_ylim(5e-4, 4)
    ax_bh.text(0.97, 0.05, f"no surface: nothing piles up,\nnothing ignites (η = {eta.item():.3f})",
               transform=ax_bh.transAxes, ha="right", fontsize=6.5)  # fmt: skip
    _small(ax_bh, "10 M☉ black hole: accretion light only")

    # --- survey: nova ignition mass vs WD mass, for each accretion rate (batched) ---
    sv, ign = d["survey"], d["ign"]
    wd = _np(sv["kind"]) == 0
    dm = (_np(ign["dM"])[wd] / eq.M_SUN).reshape(len(d["wd_masses"]), len(d["wd_rates"]))
    cmap = plt.get_cmap("viridis", len(d["wd_rates"]))
    for j, r in enumerate(d["wd_rates"]):
        ax_wds.semilogy(d["wd_masses"], dm[:, j], color=cmap(j), lw=1.3, label=f"$10^{{{np.log10(r):.0f}}}$ M☉/yr")
    ax_wds.axvline(d["m_ch"] / eq.M_SUN, color="black", lw=1)
    ax_wds.text(d["m_ch"] / eq.M_SUN - 0.02, dm.max(), "M_Ch → SN Ia", fontsize=6, ha="right", va="top")
    ax_wds.plot([m_wd], [dm_wd], "*", color=ACCENT[1], ms=10)
    ax_wds.set_xlabel("white-dwarf mass (M☉)", fontsize=7)
    ax_wds.set_ylabel("ignition mass (M☉)", fontsize=7)
    ax_wds.legend(fontsize=5, frameon=False, loc="lower left")
    _small(ax_wds, "nova ignition mass (one-zone, ±order of magnitude)")

    # --- survey: X-ray burst ignition column vs accretion rate ---
    ns = ~wd
    rates = np.array(d["ns_rates"])
    y_all = _np(ign["y"])[ns].reshape(2, len(rates))
    for k, (lab, col) in enumerate((("solar (H/He)", ACCENT[0]), ("pure He", ACCENT[2]))):
        ax_nss.loglog(rates, y_all[k], color=col, lw=1.3, label=lab)
    ax_nss.plot([mine["Mdot"][1].item() * eq.YEAR / eq.M_SUN], [y_ns], "*", color=ACCENT[1], ms=10)
    ax_nss.set_xlabel("accretion rate (M☉/yr)", fontsize=7)
    ax_nss.set_ylabel("ignition column (g/cm²)", fontsize=7)
    ax_nss.set_ylim(1e7, 1e10)
    ax_nss.legend(fontsize=6, frameon=False)
    _small(ax_nss, f"burst ignition (ΔM ≈ {me['dM'][1].item() / eq.M_SUN:.0e} M☉)")

    # --- 3D: geometry and colours for every frame ---
    spin = torch.tensor([float(d["bh_spin"])], dtype=torch.float64)
    r_in, r_h = eq.isco_radius(spin).item(), eq.horizon_radius(spin).item()
    r_disc = torch.linspace(r_in, 11, 18, dtype=torch.float64)
    t_disc = _np(eq.disc_temperature(r_disc, torch.tensor(r_in, dtype=torch.float64)))  # (18,) peak 1
    phi = np.linspace(0, 2 * np.pi, 48)
    R_d, P_d = np.meshgrid(_np(r_disc), phi)
    hot = plt.get_cmap("inferno")
    t_h = T_IGNITE["H"]
    flash_wd = np.r_[[-99], np.nonzero(np.diff(shell) < 0)[0] + 1]  # frames where a nova goes off
    flash_ns = np.r_[[-99], np.nonzero(np.diff(column) < 0)[0] + 1]  # frames where a burst ignites
    spiral = np.linspace(0, 1, 120)

    def draw_3d(k: int) -> None:
        # brown dwarf: colour from its central temperature, a gas stream spiralling in
        a = ax3[0]
        heat = np.clip(np.log10(T_t[k] / (t_h / 30)) / np.log10(30 * 1.4), 0.05, 1)  # 1 once T ≳ 1.4 T_H
        _stage(a, "brown dwarf + stream" + ("  → star!" if m_t[k] * eq.M_SUN > m_h else ""), 1.9)
        a.plot_surface(X, Y, Z, facecolors=hot(heat * shade), lw=0, antialiased=False, shade=False)
        r_s = 1.9 - 0.9 * spiral
        ang = 5 * np.pi * spiral + 0.5 * k
        a.plot(r_s * np.cos(ang), r_s * np.sin(ang), 0.15 * (1 - spiral), color="#60a5fa", lw=1.4)

        # white dwarf: a translucent accreted shell grows; a nova blows it off
        a = ax3[1]
        _stage(a, "white dwarf: shell → nova", 2.0)
        a.plot_surface(X, Y, Z, color="#e5e7eb", lw=0, alpha=1, shade=True)
        thick = 1 + 0.35 * shell[k]
        a.plot_surface(thick * X, thick * Y, thick * Z, color="#3b82f6", lw=0, alpha=0.18, shade=False)
        age = k - flash_wd[flash_wd <= k].max()
        if 0 <= age < 8:
            r_e = 1.35 + 0.08 * age
            a.plot_surface(r_e * X, r_e * Y, r_e * Z, color="#f97316", lw=0, alpha=0.35 * (1 - age / 8), shade=False)

        # neutron star: accretion column at the pole; a burst front spreads from an ignition spot
        a = ax3[2]
        _stage(a, "neutron star: column → X-ray burst", 1.6)
        age = k - flash_ns[flash_ns <= k].max()
        spot = np.arccos(np.clip(0.8 * X + 0.6 * Z, -1, 1))  # angle from the ignition point
        front = np.pi * min(1.0, (age + 1) / 4)  # the flame covers the star in ~4 frames
        glow = np.where(spot <= front, np.exp(-max(0, age - 3) / 3), 0.08) if age < 12 else np.full_like(spot, 0.08)
        a.plot_surface(X, Y, Z, facecolors=hot(np.clip(0.15 + 0.85 * glow, 0, 1) * shade), lw=0, shade=False)
        a.plot([0, 0], [0, 0], [1, 1.55], color="#60a5fa", lw=2.5 * (0.3 + column[k]))

        # black hole: horizon r₊ and a thin disc from the ISCO, T ∝ r^(−3/4), brightness ∝ L/L_Edd
        a = ax3[3]
        _stage(a, f"black hole a = {d['bh_spin']:g}: disc from r_ISCO = {r_in:.1f} GM/c²", 11)
        swirl = 0.85 + 0.15 * np.cos(3 * P_d - 2.0 * k * (R_d / r_in) ** -1.5)  # differential rotation
        bright = np.clip(lum[k], 0, 1) ** 0.35
        colour = hot(np.clip(np.interp(R_d, _np(r_disc), t_disc) * swirl * bright, 0, 1))
        a.plot_surface(
            R_d * np.cos(P_d), R_d * np.sin(P_d), 0 * R_d, facecolors=colour, lw=0, antialiased=False, shade=False
        )
        a.plot_surface(r_h * X, r_h * Y, r_h * Z, color="black", lw=0, shade=False)
        for ax_ in ax3:
            ax_.view_init(elev=30, azim=-60 + 0.8 * k)

    def update(k: int):
        draw_3d(k)
        bd_dot.set_data([m_t[k]], [T_t[k]])
        mk = m_t[k] * eq.M_SUN
        state = (
            "hydrogen burning: a red dwarf!" if mk > m_h else ("burning deuterium" if mk > d["m_d"] else "brown dwarf")
        )
        bd_txt.set_text(f"t = {s[k] * t_end / eq.YEAR / 1e6:.0f} Myr, M = {m_t[k]:.3f} M☉\n{state}")
        wd_line.set_data(t_wd[: k + 1], shell[: k + 1])
        flashes = np.nonzero(np.diff(shell[: k + 1]) < 0)[0]
        wd_flash.set_data(t_wd[flashes], np.ones(len(flashes)))
        ns_line.set_data(t_ns[: k + 1], column[: k + 1])
        bursts = np.nonzero(np.diff(column[: k + 1]) < 0)[0]
        ns_flash.set_data(t_ns[bursts], np.ones(len(bursts)))
        bh_line.set_data(ratio[: k + 1], lum[: k + 1])
        bh_dot.set_data([ratio[k]], [lum[k]])

    fig.tight_layout()
    return save_gif(fig, update, frames + [frames[-1]] * 6, path, fps=10)


# ---------- interactive 3D scenes (VTK PolyData + ParaView collections, for the web viewer and ParaView) ----------
def _vtk_array(values: np.ndarray, name: str, dtype: str, components: int = 1, unit: str | None = None) -> str:
    """One <DataArray> in VTK's binary XML encoding: base64 of the UInt32 header [blocks = 1, block size, last block
    size, compressed size], then base64 of the zlib block."""
    raw = np.ascontiguousarray(values, dtype={"Float32": "<f4", "Int32": "<i4"}[dtype]).tobytes()
    packed = zlib.compress(raw, 6)
    header = np.array([1, len(raw), len(raw), len(packed)], dtype="<u4").tobytes()
    data = base64.b64encode(header).decode() + base64.b64encode(packed).decode()
    extra = f' unit="{unit}"' if unit else ""
    return (f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="binary"{extra}>'
            f"{data}</DataArray>")  # fmt: skip


def write_vtp(path: Path, points: np.ndarray, cells: list | None = None, kind: str = "Polys",
              point_data: dict | None = None) -> Path:  # fmt: skip
    """Write VTK PolyData (.vtp, XML, zlib-compressed binary). points (P, 3); cells: list of index arrays (one per
    polygon / polyline / vertex), stored as Int32 connectivity + offsets under <Polys>, <Lines> or <Verts>;
    point_data: {name: (values (P,), unit)}. Empty `points` write an empty layer (something not there yet)."""
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    cells = cells if cells is not None else ([np.arange(len(points))] if len(points) else [])
    conn = np.concatenate([np.asarray(c) for c in cells]) if cells else np.zeros(0)
    offsets = np.cumsum([len(c) for c in cells]) if cells else np.zeros(0)
    counts = {k: 0 for k in ("Verts", "Lines", "Polys")}
    counts[kind] = len(cells)
    pd = "".join(_vtk_array(v, name, "Float32", 1, unit) for name, (v, unit) in (point_data or {}).items())
    xml = (
        '<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
        'header_type="UInt32" compressor="vtkZLibDataCompressor"><PolyData>'
        f'<Piece NumberOfPoints="{len(points)}" NumberOfVerts="{counts["Verts"]}" NumberOfLines="{counts["Lines"]}" '
        f'NumberOfStrips="0" NumberOfPolys="{counts["Polys"]}">'
        f"<PointData>{pd}</PointData><Points>{_vtk_array(points, 'Points', 'Float32', 3)}</Points>"
        f"<{kind}>{_vtk_array(conn, 'connectivity', 'Int32')}{_vtk_array(offsets, 'offsets', 'Int32')}</{kind}>"
        "</Piece></PolyData></VTKFile>\n"
    )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(xml, encoding="utf-8")
    return path


def write_pvd(path: Path, entries: list[tuple[float, str, str]]) -> Path:
    """ParaView collection: entries (time, layer name, file relative to the .pvd); the layer is the DataSet's group."""
    rows = "".join(
        f'<DataSet timestep="{t:.6g}" group="{name}" part="0" name="{name}" file="{file}"/>'
        for t, name, file in entries
    )
    path = Path(path)
    path.write_text(f'<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">'
                    f"<Collection>{rows}</Collection></VTKFile>\n", encoding="ascii")  # fmt: skip
    return path


def read_vtp(path: Path) -> dict:
    """Parse a .vtp written by write_vtp back (XML + base64 + zlib): counts, points, connectivity, offsets and the
    named point arrays with their units (for the tests)."""
    import xml.etree.ElementTree as ET

    def decode(node) -> np.ndarray:
        text = node.text.strip()
        head = np.frombuffer(base64.b64decode(text[:24]), dtype="<u4")  # 4 UInt32 = 16 bytes = 24 base64 chars
        raw = zlib.decompress(base64.b64decode(text[24:]))
        assert head[0] == 1 and len(raw) == head[1] == head[2]
        return np.frombuffer(raw, dtype={"Float32": "<f4", "Int32": "<i4"}[node.get("type")])

    piece = ET.parse(path).getroot().find("PolyData/Piece")
    out = {k: int(piece.get(f"NumberOf{k}")) for k in ("Points", "Verts", "Lines", "Polys")}
    out["points"] = decode(piece.find("Points/DataArray")).reshape(-1, 3)
    out["arrays"] = {a.get("Name"): decode(a) for a in piece.find("PointData")}
    out["units"] = {a.get("Name"): a.get("unit") for a in piece.find("PointData")}
    for kind in ("Verts", "Lines", "Polys"):
        node = piece.find(kind)
        if node is not None:
            arrays = {a.get("Name"): decode(a) for a in node}
            out["kind"], out["connectivity"], out["offsets"] = kind, arrays["connectivity"], arrays["offsets"]
    return out


def sphere_mesh(radius: float = 1.0, n_u: int = 24, n_v: int = 12) -> tuple[np.ndarray, list]:
    """Latitude–longitude sphere: points (n_u (n_v − 1) + 2, 3) with one point per pole; triangles at the poles and
    quads between the rings (n_u n_v cells)."""
    u = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v = np.linspace(0, np.pi, n_v + 1)[1:-1]
    U, V = np.meshgrid(u, v)  # (n_v − 1, n_u)
    ring = np.stack([np.cos(U) * np.sin(V), np.sin(U) * np.sin(V), np.cos(V)], -1).reshape(-1, 3)
    pts = radius * np.concatenate([[[0, 0, 1]], ring, [[0, 0, -1]]])
    south = len(pts) - 1
    idx = 1 + np.arange((n_v - 1) * n_u).reshape(n_v - 1, n_u)
    nxt = np.roll(idx, -1, axis=1)
    cells = [np.array([0, idx[0, j], nxt[0, j]]) for j in range(n_u)]
    cells += list(np.stack([idx[:-1], idx[1:], nxt[1:], nxt[:-1]], -1).reshape(-1, 4))
    cells += [np.array([south, nxt[-1, j], idx[-1, j]]) for j in range(n_u)]
    return pts, cells


def _ring(radius: float, n: int = 96) -> np.ndarray:
    """Closed circle in the z = 0 plane as polyline points (n + 1, 3), the first point repeated at the end."""
    a = np.linspace(0, 2 * np.pi, n + 1)
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.zeros_like(a)], -1)


def _const(pts: np.ndarray, value: float) -> np.ndarray:
    return np.full(len(pts), float(value))


class _Scene:
    """Collect the layers of one animated scene; write <name>_<layer>_<k>.vtp files and <name>.pvd in `folder`."""

    def __init__(self, folder: Path, name: str):
        self.folder, self.name, self.entries = Path(folder), name, []

    def add(self, k: int, t: float, layer: str, points, cells, kind: str, data: dict) -> None:
        file = f"{self.name}_{layer}_{k:03d}.vtp"
        write_vtp(self.folder / file, points, cells, kind, data)
        self.entries.append((float(t), layer, file))

    def empty(self, k: int, t: float, layer: str, kind: str, names: dict) -> None:
        """A layer that is not there at this step (no points), keeping its array names and units for the legend."""
        self.add(k, t, layer, np.zeros((0, 3)), [], kind, {n: (np.zeros(0), u) for n, u in names.items()})

    def write(self) -> Path:
        return write_pvd(self.folder / f"{self.name}.pvd", self.entries)


def brown_dwarf_scene(folder: Path, t_myr, T_c, m_t, m_h: float) -> Path:
    """Brown dwarf fed by a spiralling gas stream (lengths in brown-dwarf radii, time in Myr): the star carries its
    central temperature (K) and contracts by 8 % over the run; stream speed ∝ r^(−1/2) (Kepler); a glow shell
    appears once the mass passes the hydrogen-burning limit m_H (m_t in M☉, m_h in g)."""
    sc = _Scene(folder, "brown_dwarf")
    unit_sphere, cells = sphere_mesh()
    lam = np.linspace(0, 1, 160)  # along the stream, from its outer end to the surface
    for k, t in enumerate(t_myr):
        R = 1 - 0.08 * k / max(1, len(t_myr) - 1)
        sc.add(k, t, "star", R * unit_sphere, cells, "Polys", {"T_central": (_const(unit_sphere, T_c[k]), "K")})
        r = 3.2 - (3.2 - R) * lam
        ang = 4 * np.pi * lam + 0.6 * k
        stream = np.stack([r * np.cos(ang), r * np.sin(ang), 0.25 * (1 - lam)], -1)
        sc.add(k, t, "stream", stream, [np.arange(len(stream))], "Lines", {"speed": (r**-0.5, "a.u.")})
        if m_t[k] * eq.M_SUN > m_h:
            glow = 1.3 * R * unit_sphere
            sc.add(k, t, "ignition", glow, cells, "Polys", {"flag": (_const(glow, 1.0), "dimensionless")})
        else:
            sc.empty(k, t, "ignition", "Polys", {"flag": "dimensionless"})
    return sc.write()


def white_dwarf_scene(folder: Path, t_yr, shell, t_rec: float) -> Path:
    """White dwarf (lengths in white-dwarf radii, time in years): a static star, the accreted envelope whose thickness
    follows ΔM/ΔM_ign, and the nova shell thrown off at each ignition, expanding for 0.6 of a recurrence time."""
    sc = _Scene(folder, "white_dwarf")
    star, cells = sphere_mesh()
    for k, t in enumerate(t_yr):
        sc.add(k, t, "star", star, cells, "Polys", {"radius": (_const(star, 1.0), "dimensionless (R/R_WD)")})
        env = (1.02 + 0.33 * shell[k]) * star
        sc.add(k, t, "shell", env, cells, "Polys", {"dM": (_const(env, shell[k]), "dimensionless (ΔM/ΔM_ign)")})
        age = t % t_rec  # years since the last nova …
        if t >= t_rec and age < 0.6 * t_rec:  # … once there has been one
            r_e = 1.4 + 3.0 * age / t_rec
            data = {"radius": (_const(star, r_e), "dimensionless (R/R_WD)"), "age": (_const(star, age), "yr")}
            sc.add(k, t, "ejecta", r_e * star, cells, "Polys", data)
        else:
            sc.empty(k, t, "ejecta", "Polys", {"radius": "dimensionless (R/R_WD)", "age": "yr"})
    return sc.write()


def burst_burning(points: np.ndarray, phase: float, burst: bool) -> np.ndarray:
    """Burning level 0 … 1 on the neutron-star surface; `phase` = time since the last burst / recurrence time. The
    flame spreads from the ignition spot (unit vector (0.8, 0, 0.6)) over the whole star by phase 0.12, then the ashes
    cool (e-folding 0.08 of the recurrence time). Zero before the first burst."""
    if not burst:
        return np.zeros(len(points))
    spot = np.arccos(np.clip(points @ np.array([0.8, 0.0, 0.6]) / np.linalg.norm(points, axis=1), -1, 1))
    front = np.pi * min(1.0, phase / 0.12)
    return np.where(spot <= front + 1e-9, np.exp(-max(0.0, phase - 0.12) / 0.08), 0.0)


def neutron_star_scene(folder: Path, t_h, column, t_b: float) -> Path:
    """Neutron star (lengths in neutron-star radii, time in hours): the burning level on the surface after each X-ray
    burst, and the polar accretion column whose height and `fill` follow column / y_ign."""
    sc = _Scene(folder, "neutron_star")
    star, cells = sphere_mesh()
    for k, t in enumerate(t_h):
        sc.add(
            k, t, "star", star, cells, "Polys", {"burning": (burst_burning(star, column[k], t >= t_b), "dimensionless")}
        )
        z = 1 + np.linspace(0, 0.15 + 0.6 * column[k], 16)
        line = np.stack([np.zeros_like(z), np.zeros_like(z), z], -1)
        sc.add(k, t, "column", line, [np.arange(len(line))], "Lines",
               {"fill": (_const(line, column[k]), "dimensionless (y/y_ign)")})  # fmt: skip
    return sc.write()


def disc_mesh(r_in: float, r_out: float, n_r: int = 18, n_phi: int = 48) -> tuple[np.ndarray, np.ndarray, list]:
    """Annulus r_in … r_out: radii (n_r,), angles (n_phi,) and quads (ring-major point order, wrapping in φ)."""
    r = np.linspace(r_in, r_out, n_r)
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    idx = np.arange(n_r * n_phi).reshape(n_r, n_phi)
    nxt = np.roll(idx, -1, axis=1)
    return r, phi, list(np.stack([idx[:-1], nxt[:-1], nxt[1:], idx[1:]], -1).reshape(-1, 4))


def black_hole_scene(folder: Path, spin: float, ratio, lum, r_out: float = 14.0) -> Path:
    """Black hole (lengths in GM/c²; 'time' = accretion rate / Ṁ_Edd, which grows step by step as in the 2D panel):
    the horizon r₊, a thin disc from r_ISCO whose gas orbits at Ω = r^(−3/2) (each point advances in φ, so the three
    arcs painted on the gas wind up), carrying the thin-disc temperature (peak 1) and brightness T⁴ × L/L_Edd, and the
    prograde photon ring (1.5 r_s = 3 GM/c² for a = 0)."""
    sc = _Scene(folder, "black_hole")
    a = torch.tensor([float(spin)], dtype=torch.float64)
    r_in, r_h, r_ph = (f(a).item() for f in (eq.isco_radius, eq.horizon_radius, eq.photon_orbit_radius))
    horizon, h_cells = sphere_mesh(r_h)
    r, phi, d_cells = disc_mesh(r_in, r_out)
    T_r = _np(eq.disc_temperature(torch.tensor(r), torch.tensor(r_in, dtype=torch.float64)))  # (n_r,) peak 1
    R, P0 = np.meshgrid(r, phi, indexing="ij")  # (n_r, n_phi)
    T_disc = np.broadcast_to(T_r[:, None], R.shape).ravel()
    pattern = (0.8 + 0.2 * np.cos(3 * P0)).ravel()  # three bright arcs painted on the gas
    dt = (np.pi / 10) * r_in**1.5  # the inner edge turns 18° per step
    ring = _ring(r_ph)
    for k, t in enumerate(ratio):
        sc.add(k, t, "horizon", horizon, h_cells, "Polys", {"radius": (_const(horizon, r_h), "GM/c²")})
        P = P0 + dt * k * R**-1.5  # differential rotation, Keplerian Ω = r^(−3/2) in c³/GM
        pts = np.stack([R * np.cos(P), R * np.sin(P), np.zeros_like(R)], -1).reshape(-1, 3)
        bright = T_disc**4 * pattern * lum[k]  # flux ∝ T⁴, scaled by the luminosity L/L_Edd at this accretion rate
        data = {"T_disc": (T_disc, "dimensionless"), "brightness": (bright, "dimensionless (L/L_Edd)")}
        sc.add(k, t, "disc", pts, d_cells, "Polys", data)
        sc.add(k, t, "photon_ring", ring, [np.arange(len(ring))], "Lines", {"radius": (_const(ring, r_ph), "GM/c²")})
    return sc.write()


def export_vtk(d: dict, out_dir: Path, n_steps: int = 36) -> list[Path]:
    """The four interactive scenes in <out_dir>/results/ (a .pvd collection and its .vtp layers each), on the same
    clocks as the 2D panels."""
    res = Path(out_dir) / "results"
    res.mkdir(parents=True, exist_ok=True)
    for old in [*res.glob("*.vtp"), *res.glob("*.pvd")]:
        old.unlink()
    clk = clocks(d, np.linspace(0, 1, n_steps))
    return [
        brown_dwarf_scene(res, clk["t_bd"], clk["T_t"], clk["m_t"], d["m_h"]),
        white_dwarf_scene(res, clk["t_wd"], clk["shell"], clk["t_rec"]),
        neutron_star_scene(res, clk["t_ns"], clk["column"], clk["t_b"]),
        black_hole_scene(res, d["bh_spin"], clk["ratio"], clk["lum"]),
    ]
