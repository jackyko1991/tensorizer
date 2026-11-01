"""Main loop: parameters → tensors → solve → plot."""

import math
from pathlib import Path

from problems._common.runtime import cli

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    e_in = tensors.linear_input(params["input_angle"])
    w = tensors.plate(params["plate"], params["plate_angle"])
    angles = tensors.analyzer_angles()
    wt = tensors.phases()

    result = solve.solve(e_in, w, angles, wt)

    ex, ey, phase = solve.describe(result["e_plate"])
    print(f"after the {params['plate']} plate: |Ex| = {ex:.3f}, |Ey| = {ey:.3f}, phase(Ey) − phase(Ex) = {phase:.1f}°")
    # classic check: unpolarized light through polarizers at 0°, 45°, 90° passes I₀/8 (I₀/2 after the first one)
    crossed = equation.chain(equation.polarizer(0.0), equation.polarizer(math.pi / 2))
    three = equation.chain(equation.polarizer(0.0), equation.polarizer(math.pi / 4), equation.polarizer(math.pi / 2))
    print(f"unpolarized I0 = 1 through 0°, 90° polarizers:      I = {equation.unpolarized_intensity(crossed):.4f}")
    print(f"unpolarized I0 = 1 through 0°, 45°, 90° polarizers: I = {equation.unpolarized_intensity(three):.4f}")
    return plot.animate(angles, result, params["plate"], out_dir / "polarization_jones.gif")


if __name__ == "__main__":
    cli(run, __file__)
