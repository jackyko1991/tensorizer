"""Main loop: masses → moles → limiting reagent and yields for the batch → extent bars and yield map."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    reaction = params["reaction"]
    nu, molar, species = tensors.stoichiometry(reaction, dev)  # (S,), (S,)
    m0, weighed = tensors.experiments(reaction, params["mass_1"], params["mass_2"], params["measured"], dev)
    res = solve.analyse(m0, weighed, nu, molar)
    print("molar masses: " + ", ".join(f"{s} {M:.3f} g/mol" for s, M in zip(species, molar.tolist())))

    for b in range(len(m0)):
        label = "your run" if b == len(m0) - 1 else f"run {b + 1}"
        lim = res["limiting"][b].item()
        excess = 1 - lim
        print(
            f"{label:<8} {m0[b, 0].item():7.2f} g {species[0]} + {m0[b, 1].item():7.2f} g {species[1]}: "
            f"limiting {species[lim]:<3} ξ_max = {res['xi_max'][b].item():.4g} mol   "
            f"theoretical {res['theory'][b].item():7.3f} g {species[2]}   "
            f"left over {res['leftover_mass'][b, excess].item():6.3f} g {species[excess]}   "
            f"yield {weighed[b].item():.2f} g = {res['percent'][b].item():5.1f} %"
        )
    if res["percent"][-1] > 100:
        print("your run is above 100 %: the product must be wet or impure (or a mass was misread)")

    xi, n_sweep = solve.extent_sweep(res["n0"], nu, res["xi_max"])  # (B, K), (B, K, S)
    ratio = (nu[1].abs() * molar[1]) / (nu[0].abs() * molar[0])  # g of reactant 2 per g of reactant 1
    print(f"stoichiometric mass ratio: {ratio.item():.4f} g {species[1]} per g {species[0]}")
    G = 41 if fast_mode() else 201
    x_max = max(2.0 * params["mass_1"], 2.0 * params["mass_2"] / ratio.item())  # map axes scaled so that the
    grid_1 = torch.linspace(0.0, x_max, G, dtype=torch.float64, device=dev)  # stoichiometric line is the diagonal
    grid_2 = torch.linspace(0.0, x_max * ratio.item(), G, dtype=torch.float64, device=dev)
    product_map = solve.yield_map(grid_1, grid_2, nu, molar)  # (G, G)

    return plot.figure(xi[-1], n_sweep[-1], species, res, weighed, grid_1, grid_2, product_map, ratio, m0, out_dir)


if __name__ == "__main__":
    cli(run, __file__)
