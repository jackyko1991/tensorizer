"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    k_max = 6 if fast_mode() else params["k_max"]
    n_max = 3 if fast_mode() else params["n_max"]
    h = tensors.HALF_WIDTH

    # 1. Perron trees: 2^k thin triangles, shifted sideways, overlapping more with every round
    trees = [solve.perron_tree(k, dev) for k in range(k_max + 1)]
    print(f"{'k':>3} {'triangles':>9} {'alpha':>6} {'area / T':>9} {'Perron bound':>13} {'area × k':>9}")
    for t in trees:
        k, a = t["k"], t["alpha"]
        bound = a ** (2 * k) + 2 * (1 - a)
        print(f"{k:>3} {2**k:>9} {a:>6.3f} {t['area'] / h:>9.4f} {min(bound, 1):>13.4f} {t['area'] / h * k:>9.3f}")
    print("area shrinks with k, but only like ln k / k: Córdoba (1977) showed it cannot beat c / log N")

    # 2. the δ-neighbourhood of the Kakeya set made of three turned trees (all directions)
    kakeya = []
    for t in trees[3 : min(k_max, 9) + 1]:
        delta = 4 * h / 2 ** t["k"]  # two base widths: the tubes of this tree are this thin
        cells = solve.kakeya_cells(t, delta)
        kakeya.append((delta, cells))
        print(
            f"Kakeya set from three trees, k = {t['k']}: δ = {delta:.4f}, N(δ) = {cells:,}, "
            f"area ≈ {cells * delta**2:.3f}"
        )

    # 3. one δ-tube per direction in ℝⁿ: bush (all through 0) and random centres
    scales = {2: [1 / 8, 1 / 16, 1 / 32, 1 / 64, 1 / 128, 1 / 256], 3: [1 / 4, 1 / 8, 1 / 16, 1 / 32, 1 / 64]}
    scales[4] = [1 / 4, 1 / 8, 1 / 16]
    if fast_mode():
        scales = {2: scales[2][:4], 3: scales[3][:3]}
    tubes = {}
    for n in range(2, n_max + 1):
        for arrangement in ("bush", "random"):
            counts = solve.tube_counts(n, scales[n], arrangement, dev)
            slope = solve.dimension_slope(scales[n], counts)
            tubes[(n, arrangement)] = (scales[n], counts, slope)
            vols = ", ".join(f"{c * d**n:.3f}" for c, d in zip(counts, scales[n]))
            print(f"ℝ{n} {arrangement:>6}: N(δ)δⁿ = {vols};  log N / log(1/δ) slope ≈ {slope:.2f} (n = {n})")
    print("every arrangement tried has slope → n; the Kakeya conjecture says no arrangement can do better")

    u3 = tensors.directions(3, 0.25, dev)
    c3 = tensors.centres("random", len(u3), 3, 1, dev) * 0.6
    return plot.animate(trees, kakeya, tubes, (c3 - u3 / 2, c3 + u3 / 2), out_dir / "kakeya_conjecture.gif")


if __name__ == "__main__":
    cli(run, __file__)
