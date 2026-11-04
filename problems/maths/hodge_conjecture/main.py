"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, fast_mode

from . import plot, solve


def run(params: dict, out_dir: Path) -> Path:
    # Pure combinatorics on small integer tensors: the CPU is the right device here.
    d_max = 6 if fast_mode() else params["d_max"]
    g_max, n_max = params["g_max"], params["n_max"]

    shown = solve.cases(d_max, g_max)
    for c in shown:
        h = c["h"]
        n = h.shape[0] - 1
        middle = ", ".join(str(int(h[n - q, q])) for q in range(n + 1))
        line = f"{c['label'].split(':')[0]:<38} middle row h^(n−q,q) = [{middle}], χ = {c['chi']}"
        if "chi_formula" in c:
            line += f" (formula {c['chi_formula']})"
        if c.get("hodge") is not None:
            line += f", Hodge classes in H^{n}: {c['hodge']}"
        print(line)

    check = solve.euler_check(n_max, d_max)
    print(
        f"Euler characteristic: diamond = formula for all {len(check)} Fermat hypersurfaces with n ≤ {n_max}, "
        f"d ≤ {d_max}: {bool((check[:, 2] == check[:, 3]).all())}"
    )

    surf = solve.surface_table(d_max)
    print(f"{'d':>3} {'h^(1,1)':>8} {'ρ (Shioda)':>11} {'lines':>6} {'rank(lines)':>12} {'signature':>10}")
    for d, h11, rho, rank, sig in zip(surf["d"], surf["h11"], surf["rho"], surf["lines"], surf["signature"]):
        print(f"{d:>3} {h11:>8} {rho:>11} {3 * d * d:>6} {rank:>12} {str(sig):>10}")
    print("Lefschetz (1,1): every Hodge class on a surface is algebraic, so ρ is the rank of the Néron–Severi group;")
    print("the lines alone span it (over ℚ) only when d ≤ 4 or gcd(d, 6) = 1 (Shioda; Schütt–Shioda–van Luijk 2010)")

    four = solve.fourfold_table(d_max)
    for d, h22, hc in zip(four["d"], four["h22"], four["hodge"]):
        print(f"Fermat fourfold d = {d}: h^(2,2) = {h22}, rational Hodge classes in H⁴: {hc}")
    print("whether all of these are algebraic is the Hodge conjecture: known in some degrees, open in general")
    return plot.animate(shown, surf, four, check, out_dir / "hodge_conjecture.gif")


if __name__ == "__main__":
    cli(run, __file__)
