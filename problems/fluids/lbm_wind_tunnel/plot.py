"""Left: the wind tunnel in 3D — the STL body as a shaded surface, streamlines from an upstream rake and tracer
particles, all coloured by speed, over a floor grid of the lattice. Right: speed in the mid-span plane (the wake) and
the force coefficients C_D(t), C_L(t); in between, the validation residual of this solver against OpenLB on the
sphere case, computed live in the same run."""

import base64
import zlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from PIL import Image

from problems._common.plotting import export_plot_data, frame_indices
from problems._common.progress import report

SPEED = Normalize(0, 1.3)  # |u| / U, shared by every speed-coloured element and every frame
CMAP = plt.get_cmap("turbo")
RESIDUAL_SCALE = 0.05  # residual maps: fixed symmetric scale ±5 % of U


def shaded_faces(tri: np.ndarray) -> np.ndarray:
    """RGBA per triangle: light blue, darker where the face turns away from a light above the inlet."""
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    light = np.array([-0.5, -0.4, 0.8]) / np.linalg.norm([-0.5, -0.4, 0.8])
    shade = 0.45 + 0.55 * np.clip(n @ light, 0, 1)
    return np.concatenate([np.array([0.58, 0.77, 0.99]) * shade[:, None], np.ones((len(tri), 1))], 1)


def save_gif_shared_palette(fig, update, frames: list[int], path: Path, fps: int = 10) -> Path:
    """Render every frame, then quantise them all with ONE palette built from a strip of frames, so a given speed
    keeps exactly the same colour throughout (Pillow otherwise picks a new palette per frame and colours drift)."""
    fig.set_dpi(80)
    shots = []
    for i, k in enumerate(frames):
        update(k)
        fig.canvas.draw()
        shots.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3]))
        report(i + 1, len(frames), "frames")
    export_plot_data(fig, path)  # numbers behind the final frame (Results page)
    plt.close(fig)
    sample = shots[:: max(1, len(shots) // 12)]
    strip = Image.new("RGB", (sample[0].width, sample[0].height * len(sample)))
    for j, im in enumerate(sample):
        strip.paste(im, (0, j * im.height))
    palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    colours = np.array(palette.getpalette()[: 3 * 255]).reshape(-1, 3)
    colours[np.argmin(((colours - 255) ** 2).sum(1))] = 255  # keep the background pure white
    palette.putpalette(colours.reshape(-1).tolist())
    frames_p = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in shots]
    tmp = path.with_name(f".{path.stem}.tmp{path.suffix}")
    frames_p[0].save(tmp, save_all=True, append_images=frames_p[1:], duration=int(1000 / fps), loop=0)
    return tmp.replace(path)


def segments(line: np.ndarray, speed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Streamlines (S, K, 3) → line segments (M, 2, 3) and their speeds (M,), dropping the stopped (NaN) parts."""
    seg = np.stack([line[:, :-1], line[:, 1:]], 2).reshape(-1, 2, 3)
    val = 0.5 * (speed[:, :-1] + speed[:, 1:]).reshape(-1)
    ok = np.isfinite(seg).all((1, 2)) & np.isfinite(val)
    return seg[ok], val[ok]


def body_length_ticks(set_ticks, set_labels, n: int, step: float, unit: float):
    """Ticks every `step` lattice cells, labelled in body lengths (unit = L in cells)."""
    ticks = np.arange(0, n, step)
    set_ticks(ticks)
    set_labels([f"{v / unit:g}" for v in ticks])


def residual_panel(ax, fig, cax_rect, residual: np.ndarray, extent, title: str, text: str):
    """Reusable validation panel: (ours − reference) / U as a diverging map with a fixed symmetric scale that covers
    every value (at least ±5 %), the error numbers printed on it, and a labelled colour bar."""
    scale = max(RESIDUAL_SCALE, 1.1 * float(np.nanmax(np.abs(residual))))
    image = ax.imshow(residual.T, origin="lower", cmap=plt.get_cmap("RdBu_r", 11), vmin=-scale, vmax=scale,
                      extent=extent, interpolation="nearest", aspect="equal")  # fmt: skip
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    ax.set_xlabel("mid-height plane $x$–$y$ of the sphere case (lattice nodes)", fontsize=7, labelpad=1)
    ax.text(0.02, 0.05, text, transform=ax.transAxes, fontsize=7, color="#111827",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "none", "pad": 1.5})  # fmt: skip
    bar = fig.colorbar(image, cax=fig.add_axes(cax_rect), format=lambda v, _: f"{v:+.0%}")
    bar.ax.tick_params(labelsize=6)
    bar.set_label("$\\Delta u_x/U$ (dimensionless)", fontsize=7)
    return image


def animate(result: dict, coeffs: torch.Tensor, tri: torch.Tensor, solid: torch.Tensor, u_in: float, body: str,
            length: float, path: Path, validation: dict | None = None) -> Path:  # fmt: skip
    tracks = result["tracks"].cpu().numpy()  # (F, P, 3)
    speeds = result["speeds"].cpu().numpy() / u_in  # (F, P)
    lines = result["lines"].cpu().numpy()  # (F, S, K, 3)
    line_speeds = result["line_speeds"].cpu().numpy() / u_in
    slices = result["slices"].cpu().numpy() / u_in  # (F, Nx, Nz), NaN inside the body
    saved = result["steps"].cpu().numpy()
    c = coeffs.cpu().numpy()
    tri = tri.cpu().numpy()
    nx, ny, nz = solid.shape

    fig = plt.figure(figsize=(9.6, 5.8), facecolor="white")
    ax3 = fig.add_axes((0.04, 0.02, 0.52, 0.86), projection="3d")
    ax_s = fig.add_axes((0.6, 0.68, 0.3, 0.22))
    ax_v = fig.add_axes((0.6, 0.38, 0.3, 0.22))
    ax_c = fig.add_axes((0.64, 0.08, 0.27, 0.2))

    # 3D tunnel: equal aspect over the whole lattice, grid panes, ticks in body lengths, floor grid every 2 cells
    ax3.set_xlim(0, nx - 1)
    ax3.set_ylim(0, ny - 1)
    ax3.set_zlim(0, nz - 1)
    ax3.set_box_aspect((nx - 1, ny - 1, nz - 1), zoom=0.95)
    for axis in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
        axis.pane.set_facecolor((0.96, 0.97, 0.98, 1.0))
        axis.pane.set_edgecolor("#d1d5db")
        axis._axinfo["grid"].update(color="#d1d5db", linewidth=0.5)
    body_length_ticks(ax3.set_xticks, ax3.set_xticklabels, nx, length, length)
    body_length_ticks(ax3.set_yticks, ax3.set_yticklabels, ny, length if ny > 3 * length else length / 2, length)
    body_length_ticks(ax3.set_zticks, ax3.set_zticklabels, nz, length if nz > 3 * length else length / 2, length)
    ax3.tick_params(labelsize=7, pad=0)
    ax3.set_xlabel("$x/L$ (dimensionless)", fontsize=7, labelpad=-2)
    ax3.set_ylabel("$y/L$ (dimensionless)", fontsize=7, labelpad=-2)
    ax3.set_zlabel("$z/L$ (dimensionless)", fontsize=7, labelpad=-2)
    floor = [[(x, 0, 0), (x, ny - 1, 0)] for x in range(0, nx, 2)]
    floor += [[(0, y, 0), (nx - 1, y, 0)] for y in range(0, ny, 2)]
    ax3.add_collection3d(Line3DCollection(floor, colors="#9ca3af", linewidths=0.3, alpha=0.6))
    ax3.add_collection3d(Poly3DCollection(tri, facecolors=shaded_faces(tri), edgecolors="none"))
    ax3.quiver(0, ny / 2, nz * 0.6, 8, 0, 0, color="#2563eb", linewidth=1.5, arrow_length_ratio=0.4)
    ax3.text(0, ny / 2, nz * 0.6 + 3, "inflow", color="#1e40af", fontsize=8)
    seg0, val0 = segments(lines[0], line_speeds[0])
    stream = Line3DCollection(seg0, cmap=CMAP, norm=SPEED, linewidths=0.6, alpha=0.4)  # see-through bundle
    stream.set_array(val0)
    ax3.add_collection3d(stream, autolim=False)
    dots = ax3.scatter(*tracks[0].T, c=speeds[0], cmap=CMAP, norm=SPEED, s=1.0, depthshade=False, linewidths=0,
                       alpha=0.7)  # fmt: skip
    title3 = fig.text(0.02, 0.95, " ", fontsize=10)
    fig.text(0.02, 0.915, "streamlines and tracers coloured by speed $|u|/U$ (dimensionless, same scale as the slice)",
             fontsize=7, color="#6b7280")  # fmt: skip

    # mid-span plane: speed |u| / U, lattice grid every 2 cells, equal aspect
    smap = plt.get_cmap("turbo", 13).with_extremes(bad="#9ca3af")
    image = ax_s.imshow(slices[0].T, origin="lower", cmap=smap, norm=SPEED, aspect="equal",
                        extent=(-0.5, nx - 0.5, -0.5, nz - 0.5), interpolation="nearest")  # fmt: skip
    ax_s.set_title("speed $|u|/U$, mid-span plane", fontsize=8)
    ax_s.grid(False)
    ax_s.set_xticks(np.arange(-0.5, nx, 2), minor=True)
    ax_s.set_yticks(np.arange(-0.5, nz, 2), minor=True)
    ax_s.grid(True, which="minor", color="white", linewidth=0.25, alpha=0.5)
    ax_s.tick_params(which="minor", length=0)
    body_length_ticks(ax_s.set_xticks, ax_s.set_xticklabels, nx, length, length)
    body_length_ticks(ax_s.set_yticks, ax_s.set_yticklabels, nz, length / 2, length)
    ax_s.tick_params(labelsize=6)
    ax_s.set_xlabel("$x/L$ (dimensionless; white grid = 2 lattice cells)", fontsize=7, labelpad=1)
    ax_s.set_ylabel("$z/L$", fontsize=7, labelpad=1)
    bar = fig.colorbar(image, cax=fig.add_axes((0.92, 0.68, 0.01, 0.22)))
    bar.ax.tick_params(labelsize=6)
    bar.set_label("speed $|u|/U$ (dimensionless)", fontsize=7)

    # validation: tensor solver − OpenLB on the sphere case, computed live in this run
    if validation is None:
        ax_v.set_axis_off()
    else:
        ux_ref = validation["ux_ref"].numpy()
        res = np.where(ux_ref != 0, (validation["ux_ours"].numpy() - ux_ref) / validation["u"], np.nan)
        text = (f"$\\Delta C_D$ = {validation['cd_rel']:+.1%}   rel. $L_2$ = {validation['l2']:.1%}   "
                f"$L_\\infty$ = {validation['linf']:.1%}")  # fmt: skip
        extent = (-0.5, res.shape[0] - 0.5, -0.5, res.shape[1] - 0.5)
        residual_panel(ax_v, fig, (0.92, 0.38, 0.01, 0.22), res, extent,
                       "tensor solver − OpenLB (sphere, Re = 20): $\\Delta u_x/U$", text)  # fmt: skip

    # force coefficients: fixed limits from the whole history (10 % margin), so nothing is clipped in any frame
    t = np.arange(len(c))
    curves = [c[:, 0], c[:, 2]]
    if body == "sphere_openlb" and validation is not None:
        curves.append(validation["cd_ref_t"].numpy()[1:])  # step 0 of OpenLB is its start-up spike
    lo, hi = min(float(v.min()) for v in curves), max(float(v.max()) for v in curves)
    pad = 0.1 * (hi - lo) if hi > lo else 0.1
    ax_c.set_xlim(0, len(c))
    ax_c.set_ylim(lo - pad, hi + pad)
    ax_c.set_xlabel("lattice time step (steps)", fontsize=7, labelpad=1)
    ax_c.set_ylabel("force coefficient\n$C_D$, $C_L$ (dimensionless)", fontsize=7)
    ax_c.tick_params(labelsize=6)
    drag = ax_c.plot([], [], color="#dc2626", lw=1.2, label="$C_D$")[0]
    lift = ax_c.plot([], [], color="#2563eb", lw=1.0, label="$C_L$")[0]
    if body == "sphere_openlb" and validation is not None:
        ax_c.plot(validation["cd_ref_steps"].numpy()[1:], validation["cd_ref_t"].numpy()[1:], color="#111827", lw=0.7,
                  ls="--", alpha=0.7, label="$C_D$ OpenLB")  # fmt: skip
    mean = ax_c.axhline(np.nan, color="#dc2626", lw=0.8, ls=":")
    ax_c.legend(loc="upper right", fontsize=6, ncol=3)

    frames = frame_indices(len(saved), 48)

    def update(k: int):
        frac = frames.index(k) / max(1, len(frames) - 1)
        ax3.view_init(elev=35, azim=-135 + 20 * (frac - 0.5))  # inlet bottom-left, wake towards top-right, ±10°
        seg, val = segments(lines[k], line_speeds[k])
        stream.set_segments(seg)
        stream.set_array(val)
        dots._offsets3d = tuple(tracks[k].T)
        dots.set_array(speeds[k])
        image.set_data(slices[k].T)
        n = int(saved[k])
        drag.set_data(t[: n + 1], c[: n + 1, 0])
        lift.set_data(t[: n + 1], c[: n + 1, 2])
        cd, cl = c[n // 2 : n + 1, 0].mean(), c[n // 2 : n + 1, 2].mean()
        mean.set_ydata([cd] * 2)
        title3.set_text(f"{body}   step {n}   $C_D \\approx$ {cd:.2f},  $C_L \\approx$ {cl:+.2f}")

    return save_gif_shared_palette(fig, update, frames, path, fps=10)


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


def export_vtk(result: dict, tri: torch.Tensor, solid: torch.Tensor, u_in: float, length: float, out_dir: Path,
               n_steps: int = 24) -> Path:  # fmt: skip
    """Animated VTK for the web viewer and ParaView, in results/ (the .pvd and its .vtp files): wind_tunnel.pvd with
    surface (STL triangles, point array Cp, static), streamlines (polylines, speed |u|/U), particles (vertices, speed
    and age) and wake (iso-surface |u|/U = 0.7 of the final field, vorticity |ω| L/U, static). Coordinates in body
    lengths, x downstream, z up; time in lattice steps."""
    from . import tensors
    from .solve import interpolate, macroscopic

    res = Path(out_dir) / "results"
    e, _, _ = tensors.lattice(result["f"].device, result["f"].dtype)
    rho, u = macroscopic(result["f"], e)
    entries = []

    # surface: C_p = (p − p∞)/(½ρU²), p = ρ/3, sampled one lattice cell outside every face (each face keeps 3 vertices)
    cp_field = (rho / 3 - rho[1].mean() / 3) / (0.5 * u_in**2)
    v = tri.reshape(-1, 3)
    normal = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    probe = v + (normal / normal.norm(dim=1, keepdim=True).clamp_min(1e-12)).repeat_interleave(3, 0)
    cp = interpolate(cp_field[None], probe)[:, 0].cpu().numpy()
    write_vtp(res / "surface.vtp", (v / length).cpu().numpy(), list(np.arange(len(v)).reshape(-1, 3)), "Polys",
              {"Cp": (cp, "dimensionless")})  # fmt: skip

    # wake: iso-surface of the final speed field, coloured by vorticity magnitude
    speed = (u.norm(dim=0) / u_in).masked_fill(solid, 0.0)
    grad = [torch.gradient(u[a], dim=(0, 1, 2)) for a in range(3)]  # grad[a][b] = ∂u_a/∂x_b
    omega = torch.stack([grad[2][1] - grad[1][2], grad[0][2] - grad[2][0], grad[1][0] - grad[0][1]])
    wmag = (omega.norm(dim=0) * length / u_in).masked_fill(solid, 0.0)
    verts, faces = tensors.surface_nets(-speed, -0.7)  # inside = slower than 0.7 U
    if len(faces):
        at = verts.to(u.dtype)
        wake_data = {
            "vorticity": (interpolate(wmag[None], at)[:, 0].cpu().numpy(), "dimensionless (|ω| L/U)"),
            "speed": (interpolate(speed[None], at)[:, 0].cpu().numpy(), "dimensionless (|u|/U)"),
        }
        write_vtp(res / "wake.vtp", (verts / length).cpu().numpy(), list(faces.cpu().numpy()), "Polys", wake_data)

    saved = result["steps"].cpu().numpy()
    frames = frame_indices(len(saved), n_steps)
    for j, k in enumerate(frames):
        t = float(saved[k])
        entries += [(t, "surface", "surface.vtp")]
        if len(faces):
            entries += [(t, "wake", "wake.vtp")]
        lines = result["lines"][k].cpu().numpy()  # (S, K, 3), NaN after a line stops
        line_speed = result["line_speeds"][k].cpu().numpy() / u_in
        pts, val, cells, start = [], [], [], 0
        for line, sp in zip(lines, line_speed):  # loop over the 45 rake lines to pack them, not over nodes
            ok = np.isfinite(line).all(1) & np.isfinite(sp)
            m = int(np.argmin(ok)) if not ok.all() else len(ok)  # keep the valid head of each line
            if m >= 2:
                pts.append(line[:m])
                val.append(sp[:m])
                cells.append(np.arange(start, start + m))
                start += m
        write_vtp(res / f"streamlines_{j:03d}.vtp", np.concatenate(pts) / length, cells, "Lines",
                  {"speed": (np.concatenate(val), "dimensionless (|u|/U)")})  # fmt: skip
        p = result["tracks"][k].cpu().numpy()
        write_vtp(res / f"particles_{j:03d}.vtp", p / length, list(np.arange(len(p))[:, None]), "Verts",
                  {"speed": (result["speeds"][k].cpu().numpy() / u_in, "dimensionless (|u|/U)"),
                   "age": (result["ages"][k].cpu().numpy(), "lattice steps")})  # fmt: skip
        entries += [(t, "streamlines", f"streamlines_{j:03d}.vtp"),
                    (t, "particles", f"particles_{j:03d}.vtp")]  # fmt: skip
    return write_pvd(res / "wind_tunnel.pvd", entries)
