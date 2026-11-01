r"""Physics → tensor translation.

The semi-empirical (Weizsäcker) mass formula treats the nucleus as a charged liquid drop. Its binding energy
for Z protons and N neutrons (A = Z + N) is

    B(Z, N) = a_V A - a_S A^{2/3} - a_C \frac{Z(Z-1)}{A^{1/3}} - a_A \frac{(N - Z)^2}{A} + δ(Z, N),
    δ = +a_P / \sqrt{A} (Z, N both even),   -a_P / \sqrt{A} (both odd),   0 (A odd).

On a grid Z_z (column vector) and N_n (row vector) every term broadcasts to a (Z, N) table B_{zn} —
there is no summed index, just one formula evaluated everywhere at once.

Setting ∂B/∂Z = 0 at fixed A (ignoring pairing and the −1 in Z(Z−1)) gives the valley of stability

    Z_0(A) = \frac{A}{2 + (a_C / 2 a_A) A^{2/3}}.
"""

import torch


def binding_energy(z: torch.Tensor, n: torch.Tensor, a: dict[str, float]) -> torch.Tensor:
    r"""B_{zn} in MeV.   z: (Z, 1), n: (1, N) → (Z, N) by broadcasting."""
    mass = z + n  # A_{zn}
    volume = a["a_V"] * mass
    surface = a["a_S"] * mass ** (2 / 3)
    coulomb = a["a_C"] * z * (z - 1) / mass ** (1 / 3)
    asymmetry = a["a_A"] * (n - z) ** 2 / mass
    return volume - surface - coulomb - asymmetry + pairing(z, n, a["a_P"])


def pairing(z: torch.Tensor, n: torch.Tensor, a_p: float) -> torch.Tensor:
    r"""δ_{zn} = ±a_P/\sqrt{A} for even-even / odd-odd nuclei, 0 for odd A.   → (Z, N)"""
    sign = ((z % 2 == 0) & (n % 2 == 0)).double() - ((z % 2 == 1) & (n % 2 == 1)).double()
    return sign * a_p / (z + n).sqrt()


def per_nucleon(b: torch.Tensor, z: torch.Tensor, n: torch.Tensor) -> torch.Tensor:
    r"""B/A for every nucleus.   (Z, N) → (Z, N)"""
    return b / (z + n)


def stability_line(mass: torch.Tensor, a: dict[str, float]) -> torch.Tensor:
    r"""Z_0(A) = A / (2 + (a_C / 2a_A) A^{2/3}).   (K,) → (K,)"""
    return mass / (2 + a["a_C"] / (2 * a["a_A"]) * mass ** (2 / 3))
