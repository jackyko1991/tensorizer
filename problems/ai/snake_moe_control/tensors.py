"""Build the concrete tensors: the state of B Snake games, the weights of M controllers stacked on a leading model
axis (one per variant × seed, no nn.Module), and the run settings.

State of game b on an N × N board (rows r, columns c, row 0 at the top):
    age (B, N, N)   0 = empty, else the number of steps until that body cell is free again: the head has
                    age = length, the tail age 1 (so a body cell's age also tells its order along the snake)
    head (B, 2)     row, column of the head          dir (B,)  heading 0 up, 1 right, 2 down, 3 left
    food (B, 2)     row, column of the food          length (B,), alive (B,), hunger (B,) steps since the last food
    eaten (B,)      food eaten so far                steps (B,) moves made
Actions are relative to the heading: 0 turn left, 1 straight, 2 turn right.
"""

from dataclasses import dataclass

import torch

from problems.ai.moe_dag_pipeline.tensors import clone, leaves, n_params, tmap  # noqa: F401  (tree helpers)

DIRS = torch.tensor([[-1, 0], [0, 1], [1, 0], [0, -1]])  # (4, 2) row/column step of up, right, down, left
LEFT, STRAIGHT, RIGHT = 0, 1, 2
ACTIONS = ("left", "straight", "right")
WALL, BODY, AGE, FOOD = 0, 1, 2, 3  # channels of the view
C = 4
START_LENGTH = 3
VARIANTS = ("dense", "switch", "dense_top1", "mono")  # (a) (b) (c) (d)
LABELS = {"random": "random", "dense": "(a) dense", "switch": "(b) Switch top-1", "dense_top1": "(c) dense→top-1",
          "mono": "(d) monolithic", "teacher": "teacher"}  # fmt: skip


@dataclass
class State:
    age: torch.Tensor
    head: torch.Tensor
    dir: torch.Tensor
    food: torch.Tensor
    length: torch.Tensor
    alive: torch.Tensor
    hunger: torch.Tensor
    eaten: torch.Tensor
    steps: torch.Tensor

    def index(self, i) -> "State":
        return State(**{k: v[i] for k, v in vars(self).items()})

    def to(self, dev) -> "State":
        return State(**{k: v.to(dev) for k, v in vars(self).items()})


@dataclass
class Config:
    grid: int = 10  # board N × N
    seeds: int = 2  # models per variant on the model axis
    experts: int = 4  # K
    features: int = 64  # CNN feature width F
    hidden: int = 32  # each expert F → H → 3
    teacher_games: int = 192  # games played by the teacher for the imitation data
    teacher_steps: int = 300  # longest teacher game recorded
    steps: int = 600  # training steps (AdamW), all models at once
    batch: int = 128  # states per model per step
    lr: float = 3e-3
    balance_weight: float = 0.01  # Switch load-balancing coefficient α
    dagger: bool = True  # one DAgger round: label the states the students visit, then train on all
    dagger_games: int = 48
    dagger_steps: int = 200
    eval_games: int = 128  # the same games (same seed) for every policy
    eval_steps: int = 300
    hunger_factor: int = 1  # a game ends after hunger_factor · N² steps without food
    show_games: int = 6  # games in the GIF
    show_steps: int = 160

    def hunger_limit(self) -> int:
        return self.hunger_factor * self.grid * self.grid


# ---------------------------------------------------------------- weights, stacked on the model axis m
def _randn(gen, *shape, scale=1.0):
    return torch.randn(*shape, generator=gen) * scale


def init_cnn(M: int, F: int, grid: int, gen: torch.Generator) -> dict:
    """Perception: conv 3×3 (C → 8) → conv 3×3 stride 2 (8 → 16) → linear → F features, per model m.
    The view is (2N − 1) × (2N − 1); after the stride-2 conv it is S × S with S = N − 1."""
    S = grid - 1
    return {"c1": _randn(gen, M, 8, C, 3, 3, scale=(2 / (9 * C)) ** 0.5), "b1": torch.zeros(M, 8),
            "c2": _randn(gen, M, 16, 8, 3, 3, scale=(2 / 72) ** 0.5), "b2": torch.zeros(M, 16),
            "fc": _randn(gen, M, 16 * S * S, F, scale=(2 / (16 * S * S)) ** 0.5), "bf": torch.zeros(M, F)}  # fmt: skip


def init_moe(M: int, K: int, F: int, H: int, gen: torch.Generator) -> dict:
    """Control: a router F → K and K experts F → H → 3 (stacked on k), per model m."""
    return {
        "router": {"W": _randn(gen, M, F, K, scale=0.1), "b": torch.zeros(M, K)},
        "experts": {"W1": _randn(gen, M, K, F, H, scale=(2 / F) ** 0.5), "b1": torch.zeros(M, K, H),
                    "W2": _randn(gen, M, K, H, 3, scale=(1 / H) ** 0.5), "b2": torch.zeros(M, K, 3)},
    }  # fmt: skip


def init_mono(M: int, F: int, H: int, gen: torch.Generator) -> dict:
    """Control without routing: one MLP F → H → 3 with H chosen to match the MoE's weight count."""
    return {"W1": _randn(gen, M, F, H, scale=(2 / F) ** 0.5), "b1": torch.zeros(M, H),
            "W2": _randn(gen, M, H, 3, scale=(1 / H) ** 0.5), "b2": torch.zeros(M, 3)}  # fmt: skip


def moe_params(cfg: Config) -> int:
    K, F, H = cfg.experts, cfg.features, cfg.hidden
    return F * K + K + K * (F * H + H + H * 3 + 3)


def mono_hidden(cfg: Config) -> int:
    """H of the monolithic MLP so that F·H + H + 3·H + 3 ≈ the MoE controller's weights."""
    return round((moe_params(cfg) - 3) / (cfg.features + 4))


def stack(ws: list[dict]) -> dict:
    """Concatenate models on the model axis."""
    out = {}
    for k, v in ws[0].items():
        out[k] = stack([w[k] for w in ws]) if isinstance(v, dict) else torch.cat([w[k] for w in ws])
    return out


def to(w: dict, dev) -> dict:
    return tmap(w, lambda t: t.to(dev))
