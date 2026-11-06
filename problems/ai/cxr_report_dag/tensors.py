"""Build the concrete tensors: the NIH chest X-ray subset, the template reports, the PMC caption pairs, stacked weights.

data/nih_cxr_subset.npz   64×64 NIH ChestX-ray14 frontal (PA) radiographs, one per patient, six findings (multi-hot)
data/pmc_cxr_subset.npz   64×64 chest radiograph figures from CC BY / CC BY-SA / CC0 PubMed Central articles
data/pmc_captions.csv     their captions (from the article XML); data/ATTRIBUTION.csv credits every figure
data/training.smk         the training process as a Snakefile-style rule graph, run by the tensor DAG engine

Every weight tensor has the model axis m first (model m = seed), so one optimiser step trains every seed at once,
each model only touching its own slice. Stacking, minibatch draws and weight-dict helpers are imported from
problems/ai/stitch_llm_cnn_dag.
"""

import csv
import itertools
import math
import re
from pathlib import Path

import numpy as np
import torch

from problems.ai.stitch_llm_cnn_dag.tensors import _stack, clone, count, draws, expand, leaves, select  # noqa: F401

DATA = Path(__file__).parent / "data"
TRAINING_GRAPH = DATA / "training.smk"
IMG_SIZE = 64

FINDINGS = ["cardiomegaly", "effusion", "infiltration", "atelectasis", "pneumothorax", "mass/nodule"]
CODES = "ceiapm"  # one header letter per finding, '-' when absent
NIH_NAMES = {"Cardiomegaly": 0, "Effusion": 1, "Infiltration": 2, "Atelectasis": 3, "Pneumothorax": 4, "Mass": 5,
             "Nodule": 5}  # fmt: skip
PHRASES = [  # two phrasings per finding, in this canonical order
    ["cardiomegaly.", "enlarged heart."],
    ["pleural effusion.", "effusion."],
    ["infiltrate.", "consolidation."],
    ["atelectasis.", "volume loss."],
    ["pneumothorax.", "small pneumothorax."],
    ["nodule.", "mass."],
]
NORMAL = ["normal chest.", "clear lungs."]
NEGATION = "no pneumothorax."
N_VARIANTS = 8  # bit 0: phrasing of sentence 1, bit 1: of sentence 2, bit 2: add "no pneumothorax." when it applies
KEYWORDS = [  # what the parser looks for (also in free text such as captions)
    r"cardiomegaly|enlarged heart|cardiac enlargement|enlarged cardiac",
    r"effusion",
    r"infiltrat|consolidation",
    r"atelecta|volume loss|collapse",
    r"(?<!no )pneumothora",
    r"nodul|mass",
]

K = 6  # header / soft-prefix length: one slot per finding
MAX_TXT = 50  # longest template report (characters)
CTX = K + 1 + MAX_TXT + 1  # header, BOS, text, EOS
MAX_CAP = 360  # longest stage-2 text: caption + the article's sentences citing the figure (characters)
CTX_CAP = K + 1 + MAX_CAP + 1  # the LM's position table covers both
PAD, IMG, BOS, EOS = "_", "#", ">", "|"
VOCAB = PAD + IMG + BOS + EOS + " .,-" + "0123456789" + "abcdefghijklmnopqrstuvwxyz"
STOI = {c: i for i, c in enumerate(VOCAB)}


# ---------------------------------------------------------------- template reports and their parser
def label_sets() -> list[tuple]:
    """Every label set the toy reports: no finding, each single finding, each pair (22 multi-hot tuples)."""
    out = [(0,) * 6]
    for k in (1, 2):
        for c in itertools.combinations(range(6), k):
            out.append(tuple(int(f in c) for f in range(6)))
    return out


SETS = label_sets()
SET_Y = torch.tensor(SETS, dtype=torch.float32)  # (22, 6)
SET_INDEX = {y: i for i, y in enumerate(SETS)}


def report(y: tuple, variant: int) -> str:
    """'enlarged heart. pleural effusion. no pneumothorax.' from a multi-hot label tuple and a phrasing variant."""
    found = [f for f in range(6) if y[f]]
    if not found:
        return NORMAL[variant & 1]
    parts = [PHRASES[f][(variant >> i) & 1] for i, f in enumerate(found)]
    if not y[4] and variant & 4:
        parts.append(NEGATION)
    return " ".join(parts)


TEMPLATES = {report(y, v): y for y in SETS for v in range(N_VARIANTS)}


def parse_report(text: str) -> tuple:
    """Labels back from any text by keyword (negated pneumothorax excluded): the multi-hot tuple (6,)."""
    t = text.lower()
    return tuple(int(re.search(k, t) is not None) for k in KEYWORDS)


def template_labels(text: str) -> tuple | None:
    """The labels if the text is exactly one of the template reports, else None (not well formed)."""
    return TEMPLATES.get(text)


def header(y: tuple) -> str:
    return "".join(c if v else "-" for c, v in zip(CODES, y))


def sequence(y: tuple, variant: int) -> str:
    """'ce----' + '>' + report + '|' padded with '_' to CTX characters."""
    return (header(y) + BOS + report(y, variant) + EOS).ljust(CTX, PAD)


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids)


def report_text(ids) -> str:
    """The text a sequence writes after BOS, up to EOS (or the end)."""
    s = decode(ids)[K + 1 :]
    return s.split(EOS)[0].replace(PAD, "").strip()


def target_mask(tokens: torch.Tensor) -> torch.Tensor:
    """Targets j ≥ K (the first letter is predicted at the BOS position) up to and including EOS: (…, T − 1)."""
    tgt = tokens[..., 1:]
    j = torch.arange(tgt.shape[-1], device=tokens.device)
    after_eos = (tgt == STOI[EOS]).long().cumsum(-1) - (tgt == STOI[EOS]).long()  # 1 strictly after the first EOS
    return (j >= K) & (after_eos == 0) & (tgt != STOI[PAD])


def template_table() -> dict:
    """Every template sequence, row = set_id · N_VARIANTS + variant: tokens (176, CTX), mask (176, CTX − 1)."""
    tok = torch.tensor([encode(sequence(y, v)) for y in SETS for v in range(N_VARIANTS)])
    return {"tokens": tok, "mask": target_mask(tok)}


def row_of(set_ids: torch.Tensor, variants: torch.Tensor) -> torch.Tensor:
    return set_ids * N_VARIANTS + variants


def clean_text(s: str, max_chars: int = MAX_CAP) -> tuple[str, bool]:
    """Lower case, without figure labels, '(Fig. 2)'-style references and panel letters, letters outside the
    vocabulary as spaces; then whole sentences while they fit in max_chars. Only when the first sentence alone is
    longer is it cut at a word boundary (returned flag True)."""
    t = s.lower()
    t = re.sub(r"^\s*(fig(ure)?\.?\s*\d+[a-z]?\s*[.:]?\s*)", "", t)
    t = re.sub(r"[(\[]\s*(see\s+)?(fig|figure|figs)\b[^)\]]*[)\]]", " ", t)
    t = re.sub(r"\(\s*[a-z]\s*\)|^\s*[a-z]\s*[:.)]\s+", " ", t)
    t = "".join(c if c in STOI and c not in (PAD, IMG, BOS, EOS) else " " for c in t)
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,])", r"\1", t)
    out = ""
    for sent in re.split(r"(?<=\.)\s+", t):
        sent = sent if sent.endswith(".") else sent + "."
        if len(out) + len(sent) + (1 if out else 0) > max_chars:
            break
        out = f"{out} {sent}" if out else sent
    if out:
        return out, False
    return t[: max_chars + 1].rsplit(" ", 1)[0].rstrip(" ,.-"), True  # one over-long first sentence


def caption_tokens(texts: list[str]) -> dict:
    """Stage-2 sequences: a blank header '------' (replaced by the soft prefix), BOS, text, EOS: (N, CTX_CAP)."""
    tok = torch.tensor([encode(("-" * K + BOS + c + EOS).ljust(CTX_CAP, PAD)) for c in texts])
    return {"tokens": tok, "mask": target_mask(tok)}


# ---------------------------------------------------------------- data
def parse_nih(s: str) -> tuple:
    """'Cardiomegaly|Effusion' → (1, 1, 0, 0, 0, 0); 'No Finding' → zeros; Mass and Nodule share one finding."""
    y = [0] * 6
    for name in s.split("|"):
        if name != "No Finding":
            y[NIH_NAMES[name]] = 1
    return tuple(y)


def _standardise(images: np.ndarray, device) -> torch.Tensor:
    """uint8 (N, H, W) → float, each image to zero mean and unit variance (the two sources have different
    exposure and contrast)."""
    x = torch.from_numpy(images).float()
    x = (x - x.mean((1, 2), keepdim=True)) / (x.std((1, 2), keepdim=True) + 1e-3)
    return x.to(device)


def load_nih(device) -> dict:
    """x (N, 64, 64) standardised, y (N, 6) multi-hot float, sets (N,) index into SETS; split 0 train / 1 test."""
    with np.load(DATA / "nih_cxr_subset.npz") as z:  # images (N,64,64) uint8, labels (N,), findings (N,6), split
        images, findings, split = z["images"], z["findings"], z["split"]
    out = {"blank": torch.zeros(IMG_SIZE, IMG_SIZE, device=device)}
    for name, s in (("train", 0), ("test", 1)):
        y = torch.from_numpy(findings[split == s].astype(np.float32))
        out[f"x_{name}"] = _standardise(images[split == s], device)
        out[f"y_{name}"] = y.to(device)
        out[f"sets_{name}"] = torch.tensor([SET_INDEX[tuple(int(v) for v in r)] for r in y.tolist()], device=device)
    return out


def load_pmc(device) -> dict:
    """PMC figures x (N, 64, 64) standardised and their stage-2 texts (caption + the article's sentences that cite
    the figure, cleaned, whole sentences up to MAX_CAP characters); also the caption alone, for comparison."""
    with np.load(DATA / "pmc_cxr_subset.npz") as z:  # images (N,64,64) uint8, labels (N,), split (N,)
        images, split = z["images"], z["split"]
    with open(DATA / "pmc_captions.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    caps = [clean_text(r["report_text"])[0] for r in rows]
    only = [clean_text(r["caption"])[0] for r in rows]
    out = {"cut": sum(clean_text(r["report_text"])[1] for r in rows)}
    for name, s in (("train", 0), ("test", 1)):
        out[f"x_{name}"] = _standardise(images[split == s], device)
        out[f"cap_{name}"] = [c for c, sp in zip(caps, split) if sp == s]
        out[f"caponly_{name}"] = [c for c, sp in zip(only, split) if sp == s]
        out[f"tok_{name}"] = {k: v.to(device) for k, v in caption_tokens(out[f"cap_{name}"]).items()}
    return out


# ---------------------------------------------------------------- stacked weights (model axis first)
def init_cnn(seeds: list[int], salt: int, device, emb: int = 64) -> dict:
    """conv 5×5 1→8, pool, conv 3×3 8→16, pool, conv 3×3 16→16, pool (64 → 6×6), 576 → emb features → 6 logits."""

    def make(g):
        def u(*shape, fan):
            return (torch.rand(*shape, generator=g) * 2 - 1) * math.sqrt(6 / fan)  # He-uniform

        return {"c1": u(8, 1, 5, 5, fan=25), "b1": torch.zeros(8), "c2": u(16, 8, 3, 3, fan=72),
                "b2": torch.zeros(16), "c3": u(16, 16, 3, 3, fan=144), "b3": torch.zeros(16),
                "fc": u(576, emb, fan=576), "bf": torch.zeros(emb), "head": u(emb, 6, fan=emb) / 2,
                "bh": torch.zeros(6)}  # fmt: skip

    return _stack(seeds, salt, make, device)


def init_lm(seeds: list[int], salt: int, device, d: int = 32, heads: int = 4, layers: int = 2) -> dict:
    """The tiny GPT of chain_of_thought / stitch_llm_cnn_dag, with this problem's vocabulary and context."""
    v, dk, s = len(VOCAB), d // heads, 1 / math.sqrt(d)

    def make(g):
        def r(*shape, scale):
            return scale * torch.randn(*shape, generator=g)

        def layer():
            return {"g1": torch.ones(d), "b1": torch.zeros(d), "Wq": r(heads, d, dk, scale=s),
                    "bq": torch.zeros(heads, dk), "Wk": r(heads, d, dk, scale=s), "bk": torch.zeros(heads, dk),
                    "Wv": r(heads, d, dk, scale=s), "bv": torch.zeros(heads, dk),
                    "Wo": r(heads, dk, d, scale=s / math.sqrt(2 * layers)), "bo": torch.zeros(d),
                    "g2": torch.ones(d), "b2": torch.zeros(d), "W1": r(d, 4 * d, scale=s), "c1": torch.zeros(4 * d),
                    "W2": r(4 * d, d, scale=0.5 * s / math.sqrt(2 * layers)), "c2": torch.zeros(d)}  # fmt: skip

        return {"tok": r(v, d, scale=0.3), "pos": r(CTX_CAP, d, scale=0.3), "layers": [layer() for _ in range(layers)],
                "gf": torch.ones(d), "bf": torch.zeros(d), "head": r(d, v, scale=1e-3),
                "bias": torch.zeros(v)}  # fmt: skip

    return _stack(seeds, salt, make, device)


def init_connector(seeds: list[int], salt: int, device, d: int = 32, emb: int = 6, hidden: int = 64) -> dict:
    """c_t(q) = GELU(q W1 + S_t) W2 + b2: the CNN's six finding probabilities q and a slot vector S_t → an LM input
    vector at header slot t."""

    def make(g):
        return {"W1": torch.randn(emb, hidden, generator=g) / math.sqrt(emb),
                "S": 0.1 * torch.randn(K, hidden, generator=g),
                "W2": torch.randn(hidden, d, generator=g) * 0.3 / math.sqrt(hidden), "b2": torch.zeros(d)}  # fmt: skip

    return _stack(seeds, salt, make, device)


def init_router(seeds: list[int], salt: int, device) -> dict:
    """Linear router over the position features (8×8 pooled image ⊕ one-hot id) → 2 expert logits."""
    f = 64 + len(VOCAB)

    def make(g):
        return {"W": 0.01 * torch.randn(f, 2, generator=g), "b": torch.zeros(2)}

    return _stack(seeds, salt, make, device)


def cat_models(ws: list):
    """Several stacks of the same architecture joined along the model axis (to run them in ONE batched call)."""
    if isinstance(ws[0], dict):
        return {k: cat_models([w[k] for w in ws]) for k in ws[0]}
    if isinstance(ws[0], list):
        return [cat_models([w[i] for w in ws]) for i in range(len(ws[0]))]
    return torch.cat([w.detach() for w in ws])
