"""Build the concrete tensors: the bundled Circle of Willis crops, the constants, and the Snakemake references.

data/cases/<case>.npz     TopCoW MRA_Lausanne (CC0) crops around the Circle of Willis, 0.5 mm isotropic, 96×80×48
                          voxels, axes i → patient left, j → posterior, k → superior: `image` uint8 = 170 × the
                          normalised TOF-MRA intensity (I − p50)/(p99 − p50) clipped to [0, 1.5], `labels` uint8 =
                          TopCoW vessel classes (see LABELS). data/SOURCES.md has the processing steps.
data/Snakefile            the workflow: the same rules, run offline by Snakemake to make the references below
data/snakemake/           what Snakemake printed and measured (test references, never re-run here)
"""

import csv
import re
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data"
SNAKEFILE = DATA / "Snakefile"
REF = DATA / "snakemake"

H = 0.5e-3  # voxel size (m)
MU = 1060.0 * 3.5e-6  # blood dynamic viscosity (Pa s), Newtonian: ρν with ρ = 1060 kg/m³, ν = 3.5e-6 m²/s
P_IN = 12.0e3  # mean arterial pressure at the inlets (Pa), about 90 mmHg; outlets drain to 0 through Murray loads
Q_TOT = 10e-6  # total cerebral inflow the outlet loads are tuned for (m³/s): ICAs ~4 ml/s each + basilar ~2 ml/s
MMHG = 133.322
THETA, WIDTH = 0.30, 0.02  # segmentation threshold on the normalised intensity, and the soft-threshold width (a.u.)

# TopCoW 2024 multiclass labels
LABELS = {1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom",
          10: "Acom", 11: "R-ACA", 12: "L-ACA", 15: "3rd-A2"}  # fmt: skip
INLETS = (4, 6, 1)  # ICA ends and the basilar end feed the circle (as in Venova: Pcom/Acom ends are no boundaries)
OUTLETS = (5, 7, 11, 12, 2, 3, 15)  # MCA, ACA, PCA (and a third A2) ends drain it
TERMINAL = INLETS + OUTLETS  # labels with a free end in the crop; 8, 9, 10 are connectors between two junctions
CONNECTIONS = (
    (1, 2),
    (1, 3),
    (2, 8),
    (3, 9),
    (4, 8),
    (6, 9),
    (4, 5),
    (4, 11),
    (5, 11),
    (6, 7),
    (6, 12),
    (7, 12),
    (10, 11),
    (10, 12),
    (10, 15),
    (11, 15),
    (12, 15),
)  # label pairs that join in the Circle of Willis
PAIRS = {"ICA": (4, 6), "MCA": (5, 7), "ACA": (11, 12), "PCA": (2, 3), "Pcom": (8, 9)}  # (right, left)
L_ICA = 6

CASES = ("sub-000", "sub-002", "sub-006", "sub-021", "sub-033", "sub-354")
STENOSIS_LENGTH = 10e-3  # length of the left-ICA stenosis in the severity sweep (m)
SWEEP_STEPS = 11  # severity 0, 0.1, …, 1 (fraction of the left-ICA lumen diameter lost)
CONDITIONS = ("baseline", "lica_occluded")  # both inlets open, or the left ICA inlet closed (a collateral test)


def load_case_file(path) -> tuple[torch.Tensor, torch.Tensor]:
    """One case file → image (1, 96, 80, 48) float64 (normalised intensity, a.u.) and labels (1, 96, 80, 48) int64."""
    with np.load(path) as d:
        image = torch.from_numpy(d["image"].astype(np.float64) / 170)
        labels = torch.from_numpy(d["labels"].astype(np.int64))
    return image[None], labels[None]


def load_cases(cases=CASES) -> tuple[torch.Tensor, torch.Tensor]:
    """Images and labels of several cases stacked on the batch axis: (B, 96, 80, 48) each."""
    pairs = [load_case_file(DATA / "cases" / f"{c}.npz") for c in cases]
    return torch.cat([p[0] for p in pairs]), torch.cat([p[1] for p in pairs])


# ---------------------------------------------------------------- Snakemake references (made offline, in WSL)
def ref_plan(name: str = "dryrun.txt") -> dict:
    """Jobs of a `snakemake -n` listing, by Snakemake job id: (rule, case, condition), inputs and outputs."""
    out = {}
    for block in re.split(r"\n(?=(?:local)?rule \w+:)", (REF / name).read_text(encoding="utf-8")):
        m = re.match(r"(?:local)?rule (\w+):", block.strip())
        if not m:
            continue
        field = {k: v.strip() for k, v in re.findall(r"^\s+(\w+): (.*)$", block, flags=re.M)}
        wc = dict(re.findall(r"(\w+)=([\w-]+)", field.get("wildcards", "")))
        job = (m.group(1), wc.get("case", ""), wc.get("condition", ""))
        ins, outs = ([f for f in field.get(k, "").split(", ") if f] for k in ("input", "output"))
        out[int(field["jobid"])] = {"job": job, "input": ins, "output": outs}
    return out


def ref_dag() -> tuple[set, set, set]:
    """Snakemake's job DAG: its jobs, the edges of `snakemake --dag` (dot, node ids = the dry run's job ids) and the
    edges implied by the dry run's input/output files, both as (consumer, input job)."""
    plan = ref_plan()
    text = (REF / "dag.dot").read_text(encoding="utf-8")
    for k, label in re.findall(r'^\s*(\d+)\[label = "([^"\\]*)', text, flags=re.M):
        assert plan[int(k)]["job"][0] == label  # the same job numbering
    dot = {(plan[int(b)]["job"], plan[int(a)]["job"]) for a, b in re.findall(r"^\s*(\d+) -> (\d+)", text, flags=re.M)}
    producer = {f: v["job"] for v in plan.values() for f in v["output"]}
    files = {(v["job"], producer[f]) for v in plan.values() for f in v["input"] if f in producer}
    return {v["job"] for v in plan.values()}, dot, files


def ref_jobs(name: str) -> list[tuple]:
    """Jobs a Snakemake run or dry run listed (csv: rule, case, condition), in its order."""
    with open(REF / name, encoding="utf-8") as f:
        rows = [r for r in csv.reader(line for line in f if not line.startswith("#"))]
    return [tuple(r) for r in rows[1:]]


def ref_timings(cores: int) -> list[dict]:
    """Per-job start/end (s, from the job's shell) of the Snakemake run with this many cores."""
    with open(REF / f"timings_cores{cores}.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(line for line in f if not line.startswith("#")))
    return [{**r, "start": float(r["start"]), "end": float(r["end"])} for r in rows]


def ref_runs() -> dict:
    """Whole-run wall-clock times of the Snakemake runs (s), keyed by cores, and the machine notes."""
    out = {}
    for line in (REF / "runs.csv").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith(("#", "cores")):
            c, wall = line.split(",")[:2]
            out[int(c)] = float(wall)
    return out


def ref_cohort() -> list[dict]:
    """results/cohort.csv as Snakemake's jobs wrote it."""
    with open(REF / "cohort.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))
