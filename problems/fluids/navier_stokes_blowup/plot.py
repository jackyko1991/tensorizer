"""The Euler vorticity in 3D (strong-|ω| region and vortex lines), max_z |ω| for Euler and a viscous run, the growth of
‖ω‖_∞ and Ω, and E(k)."""

import base64
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from PIL import Image

from problems._common.plotting import ACCENT, export_plot_data, frame_indices
from problems._common.progress import report

from .equation import sample_periodic, vortex_lines


def vorticity_3d(
    w: torch.Tensor, n_seeds: int = 20, n_points: int = 2500
) -> tuple[list, np.ndarray, np.ndarray, np.ndarray]:
    """For every saved frame: the grid points with |ω| ≥ ½ the initial max (a fixed level), and vortex lines seeded
    among the strongest points. w (F, 3, N, N, N) → per-frame point clouds (positions, |ω|), lines (F, S, L, 3),
    |ω| along the lines (F, S, L) and max |ω| (F,)."""
    n = w.shape[-1]
    mag = w.norm(dim=1)  # (F, N, N, N)
    peak = mag.flatten(1).amax(dim=1)  # (F,)
    x = torch.arange(n, dtype=w.dtype, device=w.device) * (2 * np.pi / n)
    X = torch.stack(torch.meshgrid(x, x, x, indexing="ij"), dim=-1).reshape(-1, 3)  # (N³, 3)
    flat = (mag / peak[:, None, None, None]).flatten(1)  # (F, N³)
    clouds = []
    for f in range(w.shape[0]):
        keep = torch.nonzero(mag[f].flatten() >= 0.5 * peak[0]).squeeze(1)  # fixed level: ½ the initial max |ω|
        keep = keep[:: max(1, len(keep) // n_points)]
        clouds.append((X[keep].cpu().numpy(), (flat[f, keep] * peak[f]).cpu().numpy()))  # colour: absolute |ω|
    top = flat.topk(20 * n_seeds, dim=1).indices[:, ::20]  # (F, S): spread through the 20·S strongest points
    lines = vortex_lines(w, X[top], n_steps=40, h=0.8 * 2 * np.pi / n)  # (F, S, L, 3)
    f, s_, l_ = lines.shape[:3]
    along = sample_periodic(w, lines.reshape(f, s_ * l_, 3)).norm(dim=-1).reshape(f, s_, l_)  # absolute |ω|
    return clouds, lines.cpu().numpy(), along.cpu().numpy(), peak.cpu().numpy()


def line_segments(lines: np.ndarray, along: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Consecutive-point segments of every line (S, L, 3) with the mean |ω| of their ends, dropping the segments where
    a periodic line leaves one face of the box and re-enters at the opposite face."""
    seg = np.stack([lines[:, :-1], lines[:, 1:]], axis=2).reshape(-1, 2, 3)  # (S·(L−1), 2, 3)
    val = 0.5 * (along[:, :-1] + along[:, 1:]).reshape(-1)
    keep = np.abs(seg[:, 1] - seg[:, 0]).max(axis=1) < np.pi
    return seg[keep], val[keep]


def animate(out: dict, nu_list: list, bkm: torch.Tensor, delta: torch.Tensor, k_max: int, t_lost, path: Path) -> Path:
    t = out["t"].cpu().numpy()
    z, wmax = out["enstrophy"].cpu().numpy(), out["max_vorticity"].cpu().numpy()
    bkm = bkm.cpu().numpy()
    ts, sl, sp = out["t_save"].cpu().numpy(), out["slices"].cpu().numpy(), out["spectra"].cpu().numpy()
    dl = delta.cpu().numpy()
    names = ["Euler" if v == 0 else f"ν = {v:g}" for v in nu_list]

    clouds, vlines, valong, peak = vorticity_3d(out["w_first"])
    fig = plt.figure(figsize=(12.5, 6.8))
    grid = fig.add_gridspec(2, 3, width_ratios=[1.45, 1, 1.15])
    ax_3d = fig.add_subplot(grid[:, 0], projection="3d")
    # One colour scale for every frame and panel: the largest |ω| reached while the runs are resolved (later Euler
    # frames saturate, which is honest: that growth is a grid artefact).
    resolved = ts < t_lost if t_lost is not None else np.ones_like(ts, dtype=bool)
    w_ref = float(max(peak[resolved].max(), sl[resolved].max()))
    line_norm = plt.Normalize(0, w_ref)
    bar = fig.colorbar(plt.cm.ScalarMappable(norm=line_norm, cmap="viridis"), ax=ax_3d, shrink=0.4, pad=0.02)
    bar.set_label(r"$|\omega|$ on vortex lines (fixed scale)", fontsize=7)
    bar.ax.tick_params(labelsize=6)
    ax_e, ax_v = fig.add_subplot(grid[0, 1]), fig.add_subplot(grid[0, 2])
    ax_t, ax_s = fig.add_subplot(grid[1, 1]), fig.add_subplot(grid[1, 2])
    ims = []
    for ax, b in ((ax_e, 0), (ax_v, 1)):
        ims.append(ax.imshow(sl[0, b].T, origin="lower", extent=[0, 2 * np.pi] * 2, cmap="magma", vmin=0, vmax=w_ref))
        ax.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        ax.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        ax.grid(False)
        ax.set_xlabel("$x$")
    ax_e.set_ylabel("$y$")
    titles = [ax_e.set_title("", fontsize=9), ax_v.set_title("", fontsize=9)]

    for b, name in enumerate(names):
        ax_t.semilogy(t, wmax[:, b], color=ACCENT[b], lw=1.5, label=rf"$\|\omega\|_\infty$ {name}")
        ax_t.semilogy(t, z[:, b], color=ACCENT[b], lw=1.0, ls="--")
    ax_t.plot([], [], color="#6b7280", ls="--", lw=1, label=r"enstrophy $\Omega$ (dashed)")
    if t_lost is not None:
        ax_t.axvspan(t_lost, t[-1], color="#fde68a", alpha=0.5, lw=0)
        ax_t.text(t_lost + 0.05, 0.6 * np.nanmax(wmax), "Euler\nunder-resolved", fontsize=7, va="top", color="#92400e")
    cursor = ax_t.axvline(0, color="#9ca3af", lw=1)
    ax_t.set_xlabel("$t$")
    ax_t.legend(fontsize=6.5, loc="lower right")
    bkm_text = ax_t.set_title("", fontsize=9)

    kappa = np.arange(sp.shape[-1])
    lines = [
        ax_s.semilogy(kappa[1:], sp[0, b, 1:], color=ACCENT[b], lw=1.3, label=names[b])[0] for b in range(len(names))
    ]
    ax_s.axvline(k_max, color="#9ca3af", ls="--", lw=1)
    ax_s.text(k_max + 0.3, 1e-3, "2/3 cut", fontsize=7, color="#6b7280", rotation=90)
    ax_s.set_ylim(max(sp[:, :, 1:].min(), 1e-22), 1)
    ax_s.set_xlim(0, int(np.sqrt(3) * k_max) + 1)
    ax_s.set_xlabel(r"$|k|$")
    ax_s.legend(fontsize=7, loc="upper right")
    spec_title = ax_s.set_title("", fontsize=9)

    def draw_3d(f: int) -> None:
        ax_3d.cla()
        pts, val = clouds[f]
        ax_3d.scatter(*pts.T, c=val, cmap="inferno", vmin=0, vmax=w_ref, s=4, alpha=0.35, lw=0, depthshade=False)
        seg, val = line_segments(vlines[f], valong[f])
        ax_3d.add_collection3d(Line3DCollection(seg, cmap="viridis", norm=line_norm, array=val, lw=1.3))
        for setter in (ax_3d.set_xlim, ax_3d.set_ylim, ax_3d.set_zlim):
            setter(0, 2 * np.pi)
        ax_3d.set_xticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        ax_3d.set_yticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        ax_3d.set_zticks([0, np.pi, 2 * np.pi], ["0", "π", "2π"])
        ax_3d.set_xlabel("$x$")
        ax_3d.set_ylabel("$y$")
        ax_3d.set_zlabel("$z$")
        ax_3d.view_init(elev=22, azim=-60 + 1.5 * f)  # slow turn
        late = t_lost is not None and ts[f] >= t_lost
        ax_3d.set_title(
            r"Euler: $|\omega| \geq \frac{1}{2}|\omega|_{\max}(0)$ (dots) and vortex lines" + "\n"
            rf"$\max|\omega|$ = {peak[f]:.1f}, $t$ = {ts[f]:.2f}" + (" (under-resolved)" if late else ""),
            fontsize=9,
        )

    def update(f: int):
        draw_3d(f)
        for b in range(2):
            ims[b].set_data(sl[f, b].T)  # the same colour scale in every frame
            titles[b].set_text(rf"$\max_z|\omega|$ (peak {sl[f, b].max():.1f}), {names[b]}, $t$ = {ts[f]:.2f}")
        cursor.set_xdata([ts[f]])
        i = int(np.searchsorted(t, ts[f] - 1e-9))
        bkm_text.set_text(
            r"BKM $\int\|\omega\|_\infty dt$: " + " / ".join(f"{bkm[i, b]:.1f}" for b in range(len(names)))
        )
        for b, line in enumerate(lines):
            line.set_ydata(sp[f, b, 1:])
        d = dl[f, 0]
        if not np.isfinite(d):
            spec_title.set_text(r"energy spectrum $E(|k|)$")
        elif d > 0:
            spec_title.set_text(rf"$E(|k|)$, Euler strip $\delta = {d:.3f}$")
        else:
            spec_title.set_text(r"$E(|k|)$, Euler tail no longer decays")

    fig.tight_layout()
    return save_gif_shared_palette(fig, update, frame_indices(len(ts), len(ts)), path, fps=10)


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 10) -> Path:
    """Like plotting.save_gif, but every frame uses ONE palette built from all frames, so a given |ω| keeps exactly
    the same colour throughout (Pillow otherwise picks a new 256-colour palette per frame and colours drift)."""
    fig.set_dpi(80)
    shots = []
    for i, f in enumerate(frames):
        update(f)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)  # numbers behind the final frame (Results page)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]  # a strip of frames spanning the run sets the palette
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for k, im in enumerate(sample):
        strip.paste(im, (0, k * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)


def _vtk_array(values: np.ndarray, name: str, dtype: str, components: int = 1, unit: str | None = None) -> str:
    """One <DataArray> in VTK's binary XML encoding: base64 of (UInt32 byte count + zlib-compressed block)."""
    raw = np.ascontiguousarray(values, dtype={"Float32": "<f4", "Int32": "<i4"}[dtype]).tobytes()
    packed = zlib.compress(raw, 6)
    header = np.array([1, len(raw), len(raw), len(packed)], dtype="<u4").tobytes()  # 1 block, sizes
    data = base64.b64encode(header).decode() + base64.b64encode(packed).decode()
    extra = f' unit="{unit}"' if unit else ""
    return (f'<DataArray type="{dtype}" Name="{name}" NumberOfComponents="{components}" format="binary"{extra}>'
            f"{data}</DataArray>")  # fmt: skip


def write_vtp(path: Path, points: np.ndarray, cells: list | None = None, kind: str = "Polys",
              point_data: dict | None = None) -> Path:  # fmt: skip
    """Write VTK PolyData (.vtp, XML, zlib-compressed binary). points (P, 3); cells: list of index arrays (one per
    polygon / polyline / vertex), stored as Int32 connectivity + offsets under <Polys>, <Lines> or <Verts>;
    point_data: {name: (values (P,) or (P, k), unit)}."""
    cells = cells if cells is not None else [np.arange(len(points))]
    conn = np.concatenate([np.asarray(c) for c in cells]) if cells else np.zeros(0)
    offsets = np.cumsum([len(c) for c in cells]) if cells else np.zeros(0)
    counts = {k: 0 for k in ("Verts", "Lines", "Polys")}
    counts[kind] = len(cells)
    pd = "".join(
        _vtk_array(v, name, "Float32", 1 if np.ndim(v) == 1 else np.shape(v)[1], unit)
        for name, (v, unit) in (point_data or {}).items()
    )
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
    """ParaView collection: entries (time, layer name, file relative to the .pvd); each layer is one 'part'."""
    layers = list(dict.fromkeys(name for _, name, _ in entries))
    rows = "".join(
        f'<DataSet timestep="{t:.6g}" group="" part="{layers.index(name)}" name="{name}" file="{file}"/>'
        for t, name, file in entries
    )
    path = Path(path)
    path.write_text(f'<?xml version="1.0"?>\n<VTKFile type="Collection" version="1.0" byte_order="LittleEndian">'
                    f"<Collection>{rows}</Collection></VTKFile>\n", encoding="ascii")  # fmt: skip
    return path


def read_vtp(path: Path) -> dict:
    """Parse a .vtp written by write_vtp back: counts, point coordinates and named point arrays (for the tests)."""
    import xml.etree.ElementTree as ET

    def decode(node) -> np.ndarray:
        text = node.text.strip()
        head = np.frombuffer(base64.b64decode(text[:24]), dtype="<u4")  # 4 UInt32 = 16 bytes = 24 base64 chars
        raw = zlib.decompress(base64.b64decode(text[24:]))
        assert len(raw) == head[1]
        return np.frombuffer(raw, dtype={"Float32": "<f4", "Int32": "<i4"}[node.get("type")])

    piece = ET.parse(path).getroot().find("PolyData/Piece")
    out = {k: int(piece.get(f"NumberOf{k}")) for k in ("Points", "Verts", "Lines", "Polys")}
    out["points"] = decode(piece.find("Points/DataArray")).reshape(-1, 3)
    out["arrays"] = {a.get("Name"): decode(a) for a in piece.find("PointData")}
    for kind in ("Verts", "Lines", "Polys"):
        node = piece.find(kind)
        if node is not None:
            arrays = {a.get("Name"): decode(a) for a in node}
            out["connectivity"], out["offsets"] = arrays["connectivity"], arrays["offsets"]
    return out


def export_vtk(out: dict, t_lost, out_dir: Path, n_frames: int = 20) -> Path:
    """Animated VTK of the Euler run for the web viewer and ParaView, in results/: vorticity.pvd with one layer, iso:
    the surface |ω| = ½ of the initial max |ω| (surface nets on the periodic box) at about 20 saved times up to t_lost
    (while the run is resolved). Point array `vorticity` is |ω| one grid step inside the surface along ∇|ω|, i.e. how
    strong the vortex sheet is under the iso-surface (on the surface itself |ω| is the constant level)."""
    from .tensors import surface_nets

    res = Path(out_dir) / "results"
    w = out["w_first"]  # (S, 3, N, N, N), Euler run
    ts = out["t_save"].cpu().numpy()
    n = w.shape[-1]
    h = 2 * np.pi / n
    mag = w.norm(dim=1)  # (S, N, N, N)
    level = 0.5 * float(mag[0].max())
    resolved = np.nonzero(ts < t_lost)[0] if t_lost is not None else np.arange(len(ts))
    picks = resolved[frame_indices(len(resolved), n_frames)]
    entries = []
    for j, s in enumerate(picks):
        verts, faces = surface_nets(mag[s], level, periodic=True)  # grid units
        if len(faces) == 0:
            continue
        x = verts.to(w.dtype)[None] * h  # (1, V, 3) physical positions
        m3 = mag[s][None, None].expand(1, 3, n, n, n)
        grad = torch.stack([(torch.roll(mag[s], -1, a) - torch.roll(mag[s], 1, a)) / (2 * h) for a in range(3)])[None]
        g = sample_periodic(grad, x % (2 * np.pi))
        inward = x + h * g / g.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # towards stronger |ω|
        value = sample_periodic(m3.contiguous(), inward % (2 * np.pi))[0, :, 0].cpu().numpy()
        name = f"iso_{j:03d}.vtp"
        write_vtp(res / name, x[0].cpu().numpy(), list(faces.cpu().numpy()), "Polys",
                  {"vorticity": (value, "a.u.")})  # fmt: skip
        entries.append((float(ts[s]), "iso", name))
    return write_pvd(res / "vorticity.pvd", entries)
