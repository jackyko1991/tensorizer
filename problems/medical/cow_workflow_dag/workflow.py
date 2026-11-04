"""One Snakemake job: ONE rule for ONE case (B = 1) with exactly the functions the tensor engine calls once per rule
over the whole cohort, reading and writing .npz files. Called from data/Snakefile as

    python -m problems.medical.cow_workflow_dag.workflow <rule> <output> <inputs…>

Only torch and numpy are imported (no plotting), so a job starts as fast as Python + torch allow.
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch

from . import solve
from . import tensors as T


def save(path: str, d: dict) -> None:
    """Nested dict of tensors → one .npz (keys joined with '/'); writes atomically next to the target."""
    flat = {}

    def walk(prefix, x):
        for k, v in x.items():
            if isinstance(v, dict):
                walk(f"{prefix}{k}/", v)
            else:
                flat[f"{prefix}{k}"] = v.detach().numpy()

    walk("", d)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        np.savez(f, **flat)


def load(path: str) -> dict:
    out: dict = {}
    with np.load(path) as z:
        for k in z.files:
            *head, last = k.split("/")
            node = out
            for h in head:
                node = node.setdefault(h, {})
            node[last] = torch.from_numpy(z[k])
    return out


def case_of(path: str) -> str:
    return next(p for p in Path(path).parts if p.startswith("sub-"))


def run(rule: str, out: str, ins: list[str]) -> None:
    if rule in ("segment", "centreline"):
        image, labels = T.load_case_file(ins[0])
        if rule == "segment":
            seg = solve.segment(image, labels, torch.full((1,), T.THETA, dtype=torch.float64))
            save(out, seg)
        else:
            save(out, solve.centreline(labels))
    elif rule == "measure":
        save(out, solve.measure(load(ins[0]), load(ins[1])))
    elif rule == "boundaries":
        g = load(ins[0])
        save(out, {"bnd": solve.boundaries(g), "g": g})  # the graph travels with its boundaries to the flow job
    elif rule == "flow":
        m, bg = load(ins[0]), load(ins[1])
        b, g = bg["bnd"], bg["g"]
        occluded = torch.tensor([Path(out).parent.name == "lica_occluded"])
        save(out, {"flow": solve.flow(m, b, g, occluded), "g": g, "b": b})
    elif rule == "report":
        f, m = load(ins[0]), load(ins[1])
        r = solve.report(m, f["b"], f["g"], f["flow"])
        row = {"case": case_of(out), "condition": Path(out).parent.name, **{k: float(v[0]) for k, v in r.items()}}
        Path(out).write_text(json.dumps(row), encoding="utf-8")
    elif rule == "cohort":
        rows = [json.loads(Path(p).read_text(encoding="utf-8")) for p in ins]
        Path(out).write_text(solve.cohort_csv(rows), encoding="utf-8")
    else:
        raise SystemExit(f"unknown rule {rule}")


if __name__ == "__main__":
    run(sys.argv[1], sys.argv[2], sys.argv[3:])
