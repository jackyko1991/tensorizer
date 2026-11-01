"""Build the concrete tensors: parse formulas, list the reactions and stack their composition matrices.

A formula is written the usual way, with an optional charge after a caret: 'H2O', 'Ca(OH)2', 'MnO4^-', 'Fe^3+'.
The charge is stored as one more "element" called 'q', so the same matrix product checks atoms AND charge.
"""

import re

import torch

ATOM = re.compile(r"([A-Z][a-z]?)(\d*)")  # an element symbol and an optional count: 'C', 'H12', 'Ca'
GROUP_END = re.compile(r"\)(\d*)")  # a closing bracket and its multiplier: ')2'
CHARGE = re.compile(r"(\d*)([+-])")  # '3+', '-', '2-'

# name: (reactants, products)
REACTIONS = {
    "methane": (["CH4", "O2"], ["CO2", "H2O"]),
    "propane": (["C3H8", "O2"], ["CO2", "H2O"]),
    "glucose": (["C6H12O6", "O2"], ["CO2", "H2O"]),
    "photosynthesis": (["CO2", "H2O"], ["C6H12O6", "O2"]),
    "permanganate": (["MnO4^-", "Fe^2+", "H^+"], ["Mn^2+", "Fe^3+", "H2O"]),
    "phosphate": (["Ca(OH)2", "H3PO4"], ["Ca3(PO4)2", "H2O"]),
    "peroxide": (["H2", "O2"], ["H2O", "H2O2"]),  # two independent reactions hide in here
}


def parse_formula(formula: str) -> dict[str, int]:
    """Count atoms: 'Ca(OH)2' → {'Ca': 1, 'O': 2, 'H': 2};  'MnO4^-' → {'Mn': 1, 'O': 4, 'q': -1}.

    Brackets are handled with a stack: '(' opens a new tally, ')n' multiplies it by n and adds it to the outer one.
    """
    body, _, charge = formula.partition("^")
    stack = [{}]  # one tally per open bracket; stack[0] is the whole formula
    i = 0
    while i < len(body):
        if body[i] == "(":
            stack.append({})
            i += 1
        elif body[i] == ")":
            m = GROUP_END.match(body, i)
            group, times = stack.pop(), int(m.group(1) or 1)
            for element, n in group.items():
                stack[-1][element] = stack[-1].get(element, 0) + n * times
            i = m.end()
        else:
            m = ATOM.match(body, i)
            if m is None:
                raise ValueError(f"cannot read {formula!r} at position {i}")
            stack[-1][m.group(1)] = stack[-1].get(m.group(1), 0) + int(m.group(2) or 1)
            i = m.end()
    counts = stack[0]
    if charge:
        m = CHARGE.fullmatch(charge)
        counts["q"] = int(m.group(1) or 1) * (1 if m.group(2) == "+" else -1)
    return counts


def composition(reactants: list[str], products: list[str], device: torch.device):
    """Composition matrix C_es (E, S), side signs σ_s (S,), element names and species names.

    C_es = number of atoms of element e in species s (the charge row 'q' holds the charge of s);
    σ_s = +1 for a reactant and -1 for a product.
    """
    species = reactants + products
    counts = [parse_formula(f) for f in species]
    elements = sorted({e for c in counts for e in c if e != "q"})
    if any("q" in c for c in counts):
        elements.append("q")  # charge balance as one more row
    C = torch.tensor([[c.get(e, 0) for c in counts] for e in elements], dtype=torch.float64, device=device)
    sigma = torch.tensor([1.0] * len(reactants) + [-1.0] * len(products), dtype=torch.float64, device=device)
    return C, sigma, elements, species


def batch(matrices: list[torch.Tensor]) -> tuple[torch.Tensor, list[int]]:
    """Stack signed matrices M_es of different sizes into one (R, E_max + S_max, S_max) tensor.

    Missing element rows are zero (a zero row changes nothing). A missing species column would add a fake
    solution ν_s = anything, so each unused column s gets a 1 in an extra row of its own: that row says ν_s = 0.
    The extra rows also make the matrix tall, so the SVD returns one singular value per species.
    """
    E = max(M.shape[0] for M in matrices)
    S = max(M.shape[1] for M in matrices)
    out = torch.zeros(len(matrices), E + S, S, dtype=matrices[0].dtype, device=matrices[0].device)
    for r, M in enumerate(matrices):
        e, s = M.shape
        out[r, :e, :s] = M
        for j in range(s, S):
            out[r, E + j, j] = 1.0  # pin the padding coefficient to zero
    return out, [M.shape[1] for M in matrices]


def pretty(formula: str) -> str:
    """'C6H12O6' → 'C₆H₁₂O₆', 'Fe^2+' → 'Fe²⁺' for the terminal."""
    body, _, charge = formula.partition("^")
    sub = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
    sup = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
    return body.translate(sub) + charge.translate(sup)


def latex(formula: str) -> str:
    """'Ca3(PO4)2' → 'Ca_{3}(PO_{4})_{2}', 'MnO4^-' → 'MnO_{4}^{-}' for matplotlib mathtext."""
    body, _, charge = formula.partition("^")
    body = re.sub(r"(\d+)", r"_{\1}", body)
    return rf"\mathrm{{{body}}}" + (f"^{{{charge}}}" if charge else "")
