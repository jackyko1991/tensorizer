"""Build the concrete tensors: species table, molecules in a periodic box, TCSPC channels.

Units: lengths in µm, times in s (microtimes in ns), D in µm²/s, rates in photons/s.
"""

import torch

from . import equation

SEED = 1
W0 = 0.25  # lateral 1/e² radius of the detection volume (µm)
DT = 5e-6  # Brownian time step = photon time bin (s)
PERIOD = 25.0  # laser pulse period (ns), 40 MHz
N_CHANNELS = 256  # TCSPC microtime channels
IRF_T0, IRF_SIGMA = 1.5, 0.1  # Gaussian instrument response: centre and width (ns)

# name, colour, counts per molecule ε (photons/s); D, N and lifetime come from the parameters.
# The labelled protein is a little quenched: shorter lifetime, fewer photons.
SPECIES = {
    "dye": ("free dye", "#2563eb", 100e3),
    "protein": ("labelled protein", "#dc2626", 65e3),
}


def species_table(params: dict, device: torch.device) -> dict:
    """Per-species values as (S,) tensors, for the species chosen in the sample."""
    keys = ["dye", "protein"] if params["species"] == "both" else [params["species"]]
    return {
        "keys": keys,
        "names": [SPECIES[k][0] for k in keys],
        "colors": [SPECIES[k][1] for k in keys],
        "N": torch.tensor([params[f"N_{k}"] for k in keys], device=device),
        "D": torch.tensor([params[f"D_{k}"] for k in keys], device=device),
        "tau": torch.tensor([params[f"tau_{k}"] for k in keys], device=device, dtype=torch.float64),
        "eps": torch.tensor([SPECIES[k][2] for k in keys], device=device),
    }


def box_size(w0: float, z0: float, device: torch.device) -> torch.Tensor:
    """Periodic box (3,): 12 w_0 wide and 4.8 z_0 tall, so the detection weight is ~0 at its faces."""
    return torch.tensor([12 * w0, 12 * w0, 4.8 * z0], device=device)


def molecules(N: torch.Tensor, D: torch.Tensor, box: torch.Tensor, w0: float, z0: float, gen: torch.Generator):
    """Random start positions x (n, 3), membership S (n, s) and per-molecule D_n (n,).

    Species s has concentration c_s = N_s / V_eff, so the box holds M_s = c_s V_box of its molecules.
    """
    V_box = box.prod().item()
    counts = [max(1, round(n * V_box / equation.effective_volume(w0, z0))) for n in N.tolist()]
    species = torch.repeat_interleave(
        torch.arange(len(counts), device=box.device), torch.tensor(counts, device=box.device)
    )
    x = (torch.rand(len(species), 3, generator=gen, device=box.device) - 0.5) * box  # focus at the box centre
    S = torch.nn.functional.one_hot(species, len(counts)).float()  # (n, s)
    return x, S, D[species]


def channels(device: torch.device) -> torch.Tensor:
    """Centres of the TCSPC microtime channels (C,) in ns."""
    width = PERIOD / N_CHANNELS
    return (torch.arange(N_CHANNELS, device=device, dtype=torch.float64) + 0.5) * width
