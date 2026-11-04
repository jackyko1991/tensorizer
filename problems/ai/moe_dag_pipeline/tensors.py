"""Build the concrete tensors: the bundled MNIST subset, the three-word texts, the router of the workflow, and the
weights of M models stacked on a leading model axis (one per seed, no nn.Module).

data/mnist_train.npz    3000 training images (300 per digit), data/mnist_test.npz 1000 test images (100 per digit),
                        from MNIST: `images` (N, 28, 28) uint8, `labels` (N,) uint8, `index` (N,) in the original file
                        (LeCun, Cortes & Burges, CC BY-SA 3.0, see data/SOURCES.md)
data/Snakefile          the workflow: encode_image, encode_text → fuse_{op} (one expert per operation) → classify

Task: an image of a digit d and a text of three words in any order, one operation word, one number word k and one
filler ("now plus three", "three then minus") → y = (d op k) mod 10. Neither modality alone gives y.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data"
SNAKEFILE = DATA / "Snakefile"

OPS = ("plus", "minus", "times")
NUMBERS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
FILLERS = ("now", "then")
VOCAB = list(OPS) + list(NUMBERS) + list(FILLERS)  # token ids: operations 0–2, numbers 3–12, fillers 13–14
E, L, P = len(OPS), 3, 28 * 28  # experts (one per operation), words per text, pixels per image
OP_SYMBOL = ("+", "−", "×")


@dataclass
class Config:
    seeds: int = 2  # models on the model axis m, each with its own weights and minibatches
    split_steps: int = 1000  # stage 2: image, text and fuse experts, each on its own sub-task
    router_steps: int = 800  # stage 1b: fresh experts behind a learned softmax gate
    global_steps: int = 400  # stage 3: end-to-end fine-tuning through the whole DAG
    student_steps: int = 3000  # stage 4: distillation into one dense network
    batch: int = 128
    lr: float = 2e-3
    lr_global: float = 3e-4
    hidden_image: int = 128  # image expert 784 → 128 → 10
    d_text: int = 16  # text expert: embedding width
    hidden_fuse: int = 32  # each fuse expert 20 → 32 → 32
    student_hidden: int = 64  # student (784 + 45) → 64 → 64 → 10
    temperature: float = 4.0
    balance_weight: float = 0.01  # Switch load-balancing coefficient α
    kd_hard: float = 0.1  # weight of the true label in the student loss (1 − kd_hard on the teacher)
    test_size: int = 1000
    time_repeats: int = 5  # timing: median of this many calls
    seq_samples: int = 100  # timing of one-job-per-call: this many samples, scaled to the test size


# ---------------------------------------------------------------- data
def load_mnist() -> dict:
    """The bundled subset: x (N, 784) float32 in [0, 1], y (N,) int64, for 'train' (3000) and 'test' (1000)."""
    out = {}
    for split in ("train", "test"):
        with np.load(DATA / f"mnist_{split}.npz") as z:
            out[f"y_{split}"] = torch.from_numpy(z["labels"].astype(np.int64))
            out[f"x_{split}"] = torch.from_numpy(z["images"].reshape(len(z["images"]), P).astype(np.float32) / 255.0)
    return out


def make_texts(n: int, gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n random texts → tokens (n, 3), operation (n,), number (n,). The three words are scattered to a random
    permutation of the positions, so the text expert has to find them."""
    op = torch.randint(E, (n,), generator=gen)
    k = torch.randint(10, (n,), generator=gen)
    filler = torch.randint(len(FILLERS), (n,), generator=gen)
    perm = torch.argsort(torch.rand(n, L, generator=gen), dim=1)  # (n, 3) where each word goes
    words = torch.stack([op, len(OPS) + k, len(OPS) + 10 + filler], 1)
    return torch.zeros(n, L, dtype=torch.long).scatter(1, perm, words), op, k


def parse_op(tok: torch.Tensor) -> torch.Tensor:
    """The workflow's input function: the operation named in each text, by rule (no learning). (n, 3) → (n,)."""
    is_op = tok < E
    return (tok * is_op).sum(1)


def parse_number(tok: torch.Tensor) -> torch.Tensor:
    """The number named in each text, by rule (the ground truth of the text expert's sub-task). (n, 3) → (n,)."""
    is_num = (tok >= E) & (tok < E + 10)
    return ((tok - E) * is_num).sum(1)


def router_matrix(tok: torch.Tensor) -> torch.Tensor:
    """G (n, E): G_ne = 1 when the text of sample n names operation e, so the workflow sends it to fuse_e."""
    return torch.nn.functional.one_hot(parse_op(tok), E).float()


def answer(d: torch.Tensor, op: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """y = (d + k, d − k or d · k) mod 10 (never negative: torch's % follows the divisor's sign)."""
    return torch.stack([d + k, d - k, d * k], 1).gather(1, op[:, None])[:, 0] % 10


def words(tok_row) -> str:
    return " ".join(VOCAB[int(t)] for t in tok_row)


# ---------------------------------------------------------------- weights, stacked on the model axis m
def _randn(gen, *shape, scale=1.0):
    return torch.randn(*shape, generator=gen) * scale


def init_experts(M: int, cfg: Config, gen: torch.Generator) -> dict:
    """The fuse experts (one per operation, stacked on e), the classify head and the gate of a learned router."""
    Hf, d = cfg.hidden_fuse, cfg.d_text
    return {
        "fuse": {"W1": _randn(gen, M, E, 20, Hf, scale=(2 / 20) ** 0.5), "b1": torch.zeros(M, E, Hf),
                 "W2": _randn(gen, M, E, Hf, Hf, scale=(1 / Hf) ** 0.5), "b2": torch.zeros(M, E, Hf)},
        "cls": {"W": _randn(gen, M, Hf, 10, scale=(2 / Hf) ** 0.5), "b": torch.zeros(M, 10)},
        "gate": {"W": _randn(gen, M, d, E, scale=0.01), "b": torch.zeros(M, E)},
    }  # fmt: skip


def init_weights(M: int, cfg: Config, gen: torch.Generator) -> dict:
    """All stages of the pipeline for M models: image expert, text expert, fuse experts, classify head, gate."""
    Hi, d = cfg.hidden_image, cfg.d_text
    w = {
        "img": {"W1": _randn(gen, M, P, Hi, scale=(2 / P) ** 0.5), "b1": torch.zeros(M, Hi),
                "W2": _randn(gen, M, Hi, 10, scale=(1 / Hi) ** 0.5), "b2": torch.zeros(M, 10)},
        "txt": {"emb": _randn(gen, M, len(VOCAB), d, scale=0.5), "pos": _randn(gen, M, L, d, scale=0.1),
                **{k: _randn(gen, M, d, d, scale=d**-0.5) for k in ("Wq", "Wk", "Wv", "Wo")},
                "Wn": _randn(gen, M, d, 10, scale=d**-0.5), "bn": torch.zeros(M, 10)},
    }  # fmt: skip
    return {**w, **init_experts(M, cfg, gen)}


def init_student(M: int, cfg: Config, gen: torch.Generator) -> dict:
    """One dense network: [image pixels, one-hot words] (784 + 3·15) → H → H → 10."""
    H, n_in = cfg.student_hidden, P + L * len(VOCAB)
    return {"W1": _randn(gen, M, n_in, H, scale=(2 / n_in) ** 0.5), "b1": torch.zeros(M, H),
            "W2": _randn(gen, M, H, H, scale=(2 / H) ** 0.5), "b2": torch.zeros(M, H),
            "W3": _randn(gen, M, H, 10, scale=(1 / H) ** 0.5), "b3": torch.zeros(M, 10)}  # fmt: skip


def leaves(w: dict) -> list[torch.Tensor]:
    return [v for x in w.values() for v in (leaves(x) if isinstance(x, dict) else [x])]


def tmap(w: dict, f) -> dict:
    return {k: tmap(v, f) if isinstance(v, dict) else f(v) for k, v in w.items()}


def cast(w: dict, dtype: torch.dtype) -> dict:
    return tmap(w, lambda t: t.to(dtype))


def clone(w: dict, grad: bool = False) -> dict:
    return tmap(w, lambda t: t.detach().clone().requires_grad_(grad))


def select(w: dict, m: int) -> dict:
    """Model m alone, keeping a model axis of length 1."""
    return tmap(w, lambda t: t[m : m + 1].detach().clone())


def n_params(w: dict) -> int:
    """Weights per model (the model axis divided out)."""
    return sum(t[0].numel() for t in leaves(w))
