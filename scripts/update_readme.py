"""Regenerate the README's problem library from every problem's meta.yaml.

    python scripts/update_readme.py

Rewrites the text between <!-- library:start --> and <!-- library:end --> as collapsible
sections: topic → difficulty level → table of problems with their key (index-form) equation.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
START, END = "<!-- library:start -->", "<!-- library:end -->"
CATEGORIES = {
    "mechanics": "Mechanics",
    "optics": "Optics",
    "waves": "Waves",
    "thermodynamics": "Thermodynamics",
    "em": "Electromagnetism",
    "fluids": "Fluids & Continuum",
    "quantum": "Quantum",
    "astro": "Astrophysics & Relativity",
    "nuclear": "Nuclear & Particle",
    "physchem": "Physical Chemistry",
    "chemistry": "Chemistry",
    "medical": "Medical Physics",
    "biophysics": "Biophysics",
    "bioinformatics": "Bioinformatics",
    "maths": "Mathematics",
    "graphics": "Gaming & Computer Vision",
    "others": "Others",
}
LEVELS = {"beginner": "🟢 Beginner", "intermediate": "🟠 Intermediate", "advanced": "🔴 Advanced"}

# Where the first relation of the index form is only a definition, show the governing law instead.
KEY_OVERRIDES = {
    "mechanics/n_body_gravity": r"\ddot x_{ai} = G\,m_b\,(r_{abj}r_{abj} + \varepsilon^2)^{-3/2}\,r_{abi}",
    "mechanics/three_body_chaos": (
        r"\ddot x_{kai} = G\,m_b\,(r_{kabj}r_{kabj})^{-3/2}\,r_{kabi},\;\; \lVert\Delta\rVert \propto e^{\lambda t}"
    ),
    "others/neural_operator": r"\hat w_{bok} = R_{iok}\,\hat v_{bik}",
    "mechanics/mass_spring_cloth": r"m\,\ddot x_{ai} = -k\,D_{ea}\,(\lVert d_e\rVert - \ell_0)\,\hat d_{ei} + m\,g_i",
    "optics/grin_mirage": r"\frac{d}{ds}\Big(n\,\frac{dr_i}{ds}\Big) = \partial_i n",
    "thermodynamics/carnot_cycle": r"W = \oint p\,dV,\;\; \eta = 1 - T_c/T_h",
    "thermodynamics/ideal_gas_processes": r"W = \int p\,dV,\;\; Q = \Delta U + W",
    "thermodynamics/heat_diffusion": r"\partial_t T = \alpha\,\partial_i\partial_i T",
    "thermodynamics/random_walk_diffusion": r"\langle x_i x_i\rangle = 2dDt",
    "quantum/harmonic_oscillator": r"H_{xy}\,\varphi_{yn} = E_n\,\varphi_{xn},\;\; V_x = \tfrac12\omega^2 x^2",
    "quantum/free_wave_packet": r"\tilde\psi_{tk} = e^{-ik_k^2 t_t/2}\,\tilde\psi_k",
    "quantum/hydrogen_radial": r"H_{lij}\,u_{ljn} = E_{ln}\,u_{lin}",
    "quantum/tunneling": r"\psi_x \leftarrow F^{-1}_{xk}\,K_k\,F_{ky}\,P_y\,\psi_y",
    "quantum/scattering_2d": r"\psi \leftarrow P\,\mathcal F^{-1} K\,\mathcal F\,P\,\psi",
}


def key_equation(index_form: str) -> str:
    """First relation of the index form (relations are separated by \\qquad), squashed to one line."""
    first = re.split(r"\\qquad(?![a-zA-Z])|,\s*\\quad(?![a-zA-Z])", index_form)[0]
    return " ".join(first.strip().rstrip(",").split()).replace("|", r"\vert ")


def load() -> dict[str, dict[str, list[dict]]]:
    library: dict[str, dict[str, list[dict]]] = {c: {lv: [] for lv in LEVELS} for c in CATEGORIES}
    for meta_file in sorted((ROOT / "problems").glob("*/*/meta.yaml")):
        meta = yaml.safe_load(meta_file.read_text(encoding="utf-8"))
        if meta.get("hidden"):  # kept in the repository, off the site
            continue
        cat, level = meta_file.parent.parent.name, meta.get("level")
        if cat in library and level in LEVELS:
            library[cat][level].append({**meta, "path": meta_file.parent.relative_to(ROOT).as_posix()})
    for levels in library.values():
        for problems in levels.values():
            problems.sort(key=lambda m: m.get("order", 99))
    return library


def render(library: dict[str, dict[str, list[dict]]]) -> str:
    total = sum(len(p) for levels in library.values() for p in levels.values())
    out = [f"{total} problems in {len(CATEGORIES)} topics by difficulty level. Click a topic to expand it.", ""]
    for cat, title in CATEGORIES.items():
        levels = library[cat]
        count = sum(len(p) for p in levels.values())
        out += ["<details>", f"<summary><b>{title}</b> ({count} problem{'s' if count != 1 else ''})</summary>", ""]
        for level, label in LEVELS.items():
            problems = levels[level]
            if not problems:
                continue
            out += ["<details>", f"<summary>{label} ({len(problems)})</summary>", ""]
            out += ["| Problem | Key equation |", "|---|---|"]
            for m in problems:
                key = KEY_OVERRIDES.get(m["path"].removeprefix("problems/")) or key_equation(m["index_form"])
                out.append(f"| [{m['title']}]({m['path']}) | ${key}$ |")
            out += ["", "</details>", ""]
        out += ["</details>", ""]
    return "\n".join(out).rstrip() + "\n"


def main() -> None:
    text = README.read_text(encoding="utf-8")
    if START not in text or END not in text:
        raise SystemExit(f"README.md needs {START} and {END} markers")
    head, rest = text.split(START, 1)
    _, tail = rest.split(END, 1)
    README.write_text(f"{head}{START}\n{render(load())}{END}{tail}", encoding="utf-8")
    print("README problem library updated")


if __name__ == "__main__":
    main()
