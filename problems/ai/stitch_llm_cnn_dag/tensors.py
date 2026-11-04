"""Build the concrete tensors: the MNIST subset, the synthetic sentences, the joint task, and stacked weights.

data/mnist_subset.npz   3000 training and 1000 test MNIST digits (300 and 100 per class), uint8, see data/SOURCES.md
data/training.smk       the training process as a Snakefile-style rule graph, run by the tensor DAG engine

Every weight tensor has the model axis m first: model m = (seed, joint-set size) row-major, so one optimiser step
trains every seed (and every size) at once, each model only touching its own slice.
"""

import math
from pathlib import Path

import numpy as np
import torch

DATA = Path(__file__).parent / "data"
TRAINING_GRAPH = DATA / "training.smk"

WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
OPS = ["plus", "minus", "times"]  # mod 10
SYM = "+-*"  # how the text writes them
K = 5  # the first operand word is right-aligned in a 5-character slot: the K soft prefix tokens replace it
CTX = 18  # "seven*seven=seven." is 5 + 13 characters
PAD, IMG = "_", "#"  # padding after the full stop; the placeholder id of an image position (never in the text)
VOCAB = PAD + IMG + " .=" + SYM + "".join(sorted({c for w in WORDS for c in w}))
STOI = {c: i for i, c in enumerate(VOCAB)}
MEAN, STD = 0.1307, 0.3081  # MNIST pixel normalisation


def result(a: torch.Tensor, op: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """r = a + b, a − b or a × b (mod 10) for op = 0, 1, 2.  Any shape (broadcast)."""
    return torch.where(op == 0, a + b, torch.where(op == 1, a - b, a * b)) % 10


def sentence(a: int, op: int, b: int) -> str:
    """'  two+three=five.' padded with '_' to CTX characters."""
    r = int(result(torch.tensor(a), torch.tensor(op), torch.tensor(b)))
    return f"{WORDS[a]:>{K}}{SYM[op]}{WORDS[b]}={WORDS[r]}.".ljust(CTX, PAD)


def encode(s: str) -> list[int]:
    return [STOI[c] for c in s]


def decode(ids) -> str:
    return "".join(VOCAB[int(i)] for i in ids)


def sentence_table() -> dict:
    """Every sentence of the language at once, indexed by (a, op, b): 10 × 3 × 10 = 300.

    tokens (10, 3, 10, CTX) ids;  text_mask (…, CTX − 1) targets up to and including the full stop;  answer_mask
    (…, CTX − 1) targets of the result word and its full stop: getting all of them right under teacher forcing is
    exactly what greedy decoding writes, so accuracy needs one forward pass.
    """
    tok = torch.zeros(10, 3, 10, CTX, dtype=torch.long)
    text = torch.zeros(10, 3, 10, CTX - 1, dtype=torch.bool)
    ans = torch.zeros(10, 3, 10, CTX - 1, dtype=torch.bool)
    for a in range(10):
        for op in range(3):
            for b in range(10):
                s = sentence(a, op, b)
                tok[a, op, b] = torch.tensor(encode(s))
                stop = s.index(".")
                start = s.index("=") + 1  # first character of the result word
                text[a, op, b, :stop] = True  # target j is character j + 1
                ans[a, op, b, start - 1 : stop] = True
    return {"tokens": tok, "text_mask": text, "answer_mask": ans}


def load_mnist(device: torch.device) -> dict:
    """The bundled subset as normalised float images (N, 28, 28) and integer labels (N,)."""
    with np.load(DATA / "mnist_subset.npz") as z:  # images (N, 28, 28) uint8, labels (N,), split 0 train / 1 test
        images, labels, split = z["images"], z["labels"], z["split"]
    arr = {"x_train": images[split == 0], "y_train": labels[split == 0], "x_test": images[split == 1],
           "y_test": labels[split == 1]}  # fmt: skip

    def img(x):
        return ((torch.from_numpy(x).float() / 255 - MEAN) / STD).to(device)

    return {"x_train": img(arr["x_train"]), "y_train": torch.from_numpy(arr["y_train"]).long().to(device),
            "x_test": img(arr["x_test"]), "y_test": torch.from_numpy(arr["y_test"]).long().to(device),
            "blank": torch.full((28, 28), -MEAN / STD, device=device)}  # fmt: skip


def joint_pool(seeds: list[int], n_images: int, device: torch.device) -> dict:
    """Joint training examples per seed: a random order of the training images, each with a random (op, b).
    A set of size n is the first n rows, so the sets are nested (100 ⊂ 300 ⊂ …).  (S, n_images) each."""
    out = {"img": [], "op": [], "b": []}
    for s in seeds:
        g = torch.Generator().manual_seed(10_000 + s)
        out["img"].append(torch.randperm(n_images, generator=g))
        out["op"].append(torch.randint(3, (n_images,), generator=g))
        out["b"].append(torch.randint(10, (n_images,), generator=g))
    return {k: torch.stack(v).to(device) for k, v in out.items()}


def joint_test(n: int, device: torch.device) -> dict:
    """The fixed joint test set: test image i with its own random (op, b), the same for every model."""
    g = torch.Generator().manual_seed(4242)
    return {"op": torch.randint(3, (n,), generator=g).to(device), "b": torch.randint(10, (n,), generator=g).to(device)}


def draws(seeds: list[int], n_per_model: list[int], steps: int, batch: int, salt: int, device) -> torch.Tensor:
    """Minibatch row indices (steps, M, B): model m draws from range(n_m) with its own generator, so a model's
    batches never depend on which other models share the stack (batched run = per-model runs)."""
    out = [torch.randint(n, (steps, batch), generator=torch.Generator().manual_seed(s * 7919 + salt))
           for s, n in zip(seeds, n_per_model)]  # fmt: skip
    return torch.stack(out, 1).to(device)


# ---------------------------------------------------------------- stacked weights (model axis first)
def _stack(seeds: list[int], salt: int, make, device) -> dict:
    """make(gen) → dict of tensors for ONE model; stacked over the seeds, each drawn from its own generator."""
    ones = [make(torch.Generator().manual_seed(s * 100 + salt)) for s in seeds]

    def stack(xs):
        if isinstance(xs[0], dict):
            return {k: stack([x[k] for x in xs]) for k in xs[0]}
        if isinstance(xs[0], list):
            return [stack([x[i] for x in xs]) for i in range(len(xs[0]))]
        return torch.stack(xs).to(device).requires_grad_(True)

    return stack(ones)


def init_cnn(seeds: list[int], salt: int, device, emb: int = 32) -> dict:
    """conv 5×5 1→6, pool, conv 5×5 6→12, pool, 192 → emb features, emb → 10 digit logits."""

    def make(g):
        def u(*shape, fan):
            return (torch.rand(*shape, generator=g) * 2 - 1) * math.sqrt(6 / fan)  # He-uniform

        return {"c1": u(6, 1, 5, 5, fan=25), "b1": torch.zeros(6), "c2": u(12, 6, 5, 5, fan=150),
                "b2": torch.zeros(12), "fc": u(192, emb, fan=192), "bf": torch.zeros(emb),
                "head": u(emb, 10, fan=emb) / 2, "bh": torch.zeros(10)}  # fmt: skip

    return _stack(seeds, salt, make, device)


def init_lm(seeds: list[int], salt: int, device, d: int = 32, heads: int = 4, layers: int = 2) -> dict:
    """The tiny GPT of chain_of_thought / tiny_transformer, one per model: tok (M, V, d), pos (M, CTX, d), blocks."""
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

        return {"tok": r(v, d, scale=0.3), "pos": r(CTX, d, scale=0.3), "layers": [layer() for _ in range(layers)],
                "gf": torch.ones(d), "bf": torch.zeros(d), "head": r(d, v, scale=1e-3),
                "bias": torch.zeros(v)}  # fmt: skip

    return _stack(seeds, salt, make, device)


def init_connector(seeds: list[int], salt: int, device, d: int = 32, emb: int = 10, hidden: int = 64) -> dict:
    """c_t(q) = W2ᵀ GELU(W1ᵀ q + S_t) + b2: the CNN output q (10 digit probabilities) and a slot vector S_t → an LM
    input vector."""

    def make(g):
        return {"W1": torch.randn(emb, hidden, generator=g) / math.sqrt(emb),
                "S": 0.1 * torch.randn(CTX - 1, hidden, generator=g),
                "W2": torch.randn(hidden, d, generator=g) * 0.3 / math.sqrt(hidden), "b2": torch.zeros(d)}  # fmt: skip

    return _stack(seeds, salt, make, device)


def init_router(seeds: list[int], salt: int, device) -> dict:
    """Linear router over the position features (7×7 pooled image ⊕ one-hot id) → 2 expert logits."""
    f = 49 + len(VOCAB)

    def make(g):
        return {"W": 0.01 * torch.randn(f, 2, generator=g), "b": torch.zeros(2)}

    return _stack(seeds, salt, make, device)


def leaves(w) -> list[torch.Tensor]:
    """Every tensor of a nested weight dict/list."""
    if isinstance(w, dict):
        return [t for v in w.values() for t in leaves(v)]
    if isinstance(w, list):
        return [t for v in w for t in leaves(v)]
    return [w]


def clone(w, grad: bool):
    """Detached copy of a nested weight dict; trainable (requires_grad) or frozen."""
    if isinstance(w, dict):
        return {k: clone(v, grad) for k, v in w.items()}
    if isinstance(w, list):
        return [clone(v, grad) for v in w]
    return w.detach().clone().requires_grad_(grad)


def expand(w, n: int):
    """Repeat each model n times along the model axis (seed → (seed, size)): the I ⊗ 𝟙 edge of the job DAG."""
    if isinstance(w, dict):
        return {k: expand(v, n) for k, v in w.items()}
    if isinstance(w, list):
        return [expand(v, n) for v in w]
    return w.detach().repeat_interleave(n, 0)


def select(w, idx: list[int]):
    """Models idx of a stack."""
    if isinstance(w, dict):
        return {k: select(v, idx) for k, v in w.items()}
    if isinstance(w, list):
        return [select(v, idx) for v in w]
    return w[idx]


def count(w) -> int:
    """Number of weights of ONE model of the stack."""
    return sum(t[0].numel() for t in leaves(w))
