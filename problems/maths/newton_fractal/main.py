"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    name, n_iter, tol = params["polynomial"], params["iterations"], params["tol"]
    n = 96 if fast_mode() else params["pixels"]
    c = tensors.coefficients(name, dev)  # (K,)
    roots = tensors.roots(c)  # (J,)
    print(f"p(z) = {tensors.LABELS[name]}, roots " + ", ".join(f"{r.real:+.4f}{r.imag:+.4f}i" for r in roots.tolist()))

    half = params["extent"]
    z_full = tensors.grid(0j, half, n, dev)  # (H, W) complex: one starting point per pixel
    labels, conv = solve.iterate(z_full, c, roots, n_iter, tol)
    final = labels[-1].long()
    for j in range(len(roots)):
        print(f"basin of root {j}: {(final == j).float().mean().item():6.1%} of the pixels")
    print(f"not converged after {n_iter} steps: {(final < 0).float().mean().item():.1%} of the pixels")
    print(f"mean steps to converge: {conv[final >= 0].float().mean().item():.1f}")

    center = solve.zoom_center(z_full, final, len(roots), half)
    z_zoom = tensors.grid(center, half / params["zoom"], n, dev)
    labels_zoom, conv_zoom = solve.iterate(z_zoom, c, roots, n_iter, tol)
    print(f"zoom ×{params['zoom']:.0f} around {center.real:+.4f}{center.imag:+.4f}i")
    for tag, lab in (("full view", final), ("zoom", labels_zoom[-1].long())):
        b, w = solve.boundary_stats(lab, len(roots))
        print(f"{tag:9s}: {b:5.1%} of pixels on a boundary, {w:5.1%} of those touch all {len(roots)} basins")

    errors, root = solve.error_trace(complex(roots[0].item()) + 0.45 + 0.3j, c)
    limit = solve.newton_constant(root, c)
    print(f"one start point → root {root.real.item():+.4f}{root.imag.item():+.4f}i, |p''/(2p')| = {limit:.4f}")
    for k in range(len(errors) - 1):
        if errors[k + 1] < 1e-14:
            break
        print(f"  step {k + 1}: error {errors[k + 1]:.3e},  e_(k+1) / e_k² = {errors[k + 1] / errors[k] ** 2:.4f}")

    views = [(labels, conv, half, 0j), (labels_zoom, conv_zoom, half / params["zoom"], center)]
    return plot.animate(views, len(roots), errors, limit, tensors.LABELS[name], out_dir / "newton_fractal.gif")


if __name__ == "__main__":
    cli(run, __file__)
