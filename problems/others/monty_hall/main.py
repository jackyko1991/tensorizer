"""Main loop: parameters → tensors → solve → plot."""

from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors
from .equation import exact_rates


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    host, g = params["host"], params["n_games"]
    doors = tensors.door_counts(int(params["max_doors"]))  # (B,)
    res = solve.play(doors, g, host, 0, dev)  # running rates (B, G)
    stay_th, switch_th = exact_rates(doors, host)
    print(f"{host} host: {len(doors)} door counts × {g:,} games = {len(doors) * g:,} games in one (B, G) tensor")
    for b, n in enumerate(doors.tolist()):
        extra = f"   valid games {res['valid'][b]:.4f} (theory {2 / n:.4f})" if host == "ignorant" else ""
        print(
            f"N = {n:4d}: stay {res['stay'][b, -1]:.4f} (theory {stay_th[b]:.4f})   "
            f"switch {res['switch'][b, -1]:.4f} (theory {switch_th[b]:.4f}){extra}"
        )
    by_pick = solve.rate_by_first_pick(res["picks"][0], res["switch_wins"][0], res["valid_games"][0], 3)
    print(f"N = 3, switch win rate by first pick: {[round(x, 4) for x in by_pick.tolist()]} (no door is special)")
    if host == "knowing":
        print("switching loses only when the first pick was already the car: probability 1/N")
    else:
        print("an ignorant host carries no information: when the car survives, stay and switch are a coin flip")
    return plot.animate(doors, res, host, out_dir / "monty_hall.gif")


if __name__ == "__main__":
    cli(run, __file__)
