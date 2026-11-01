"""Main loop: formulas → composition matrices → SVD null space → whole-number coefficients → plot."""

from pathlib import Path

import torch

from problems._common.runtime import cli, device

from . import equation, plot, solve, tensors


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    names = list(tensors.REACTIONS)
    parts = [tensors.composition(*tensors.REACTIONS[n], dev) for n in names]  # (C, σ, elements, species) each
    M_list = [equation.signed_composition(C, sigma) for C, sigma, _, _ in parts]
    M, n_species = tensors.batch(M_list)  # (R, E_max + S_max, S_max)
    print(f"{len(names)} reactions in one batch: M has shape {tuple(M.shape)}")

    sv, V = solve.null_vectors(M)  # (R, S), (R, S, S)
    nu, ok = solve.to_integers(V[..., -1], params["max_scale"])  # smallest singular vector of each reaction
    dims = equation.null_dimension(sv)  # (R,)
    residual = equation.imbalance(M, nu)  # (R, E_max + S_max): must be all zero
    sv_real = [torch.linalg.svdvals(M[r, :, :s]) for r, s in enumerate(n_species)]  # without the padding columns

    balanced, plane = [], []
    for r, name in enumerate(names):
        species = parts[r][3]
        if dims[r] == 1 and ok[r]:
            text = _equation_text(nu[r, : n_species[r]], species, len(tensors.REACTIONS[name][0]))
            balanced.append((name, nu[r, : n_species[r]]))
            print(f"{name:<15} {text}   (max |Δ| = {residual[r].abs().max().item():.0f})")
        else:
            print(f"{name:<15} null space of dimension {dims[r].item()}: no unique balance")
            N = V[r, : n_species[r], -dims[r] :]  # (S, 2) basis of the null space
            cands = solve.reactions_in_plane(N)
            n_left = len(tensors.REACTIONS[name][0])
            for c in cands.unique(dim=0).flip(0):
                if solve.physical(c) and (c != 0).sum() > 1:
                    plane.append(c)
                    print(f"{'':<15}   basis reaction: {_equation_text(c, species, n_left)}")
            if len(plane) == 2:
                mix = plane[0] + plane[1]
                print(f"{'':<15}   any positive mix also balances, e.g. {_equation_text(mix, species, n_left)}")
    print("element and charge conservation M ν = 0 holds for every reaction:", bool((residual[dims == 1] == 0).all()))

    show = params["show"]
    r = names.index(show)
    C, sigma, elements, species = parts[r]
    nu_show = nu[r, : n_species[r]] if dims[r] == 1 else plane[0]
    return plot.figure(M_list[r], nu_show, elements, species, show, names, sv_real, balanced, plane, out_dir)


def _equation_text(nu, species: list[str], n_left: int) -> str:
    def side(idx):
        terms = [(f"{int(nu[i])} " if nu[i] != 1 else "") + tensors.pretty(species[i]) for i in idx if nu[i] != 0]
        return " + ".join(terms)

    return f"{side(range(n_left))} → {side(range(n_left, len(species)))}"


if __name__ == "__main__":
    cli(run, __file__)
