"""Main loop: enumerate machines → run them all at once → find the busy beavers → plot."""

import time
from pathlib import Path

from problems._common.runtime import cli, device

from . import plot, solve, tensors

KNOWN = {2: (6, 4), 3: (21, 6), 4: (107, 13)}  # (S(n), Σ(n)): Radó 1962, Lin & Radó 1965, Brady 1983


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    n, budget = params["states"], params["budget"]
    tic = time.perf_counter()

    # 1. Every 2-state machine, no shortcuts: (2·2·3)^4 = 20 736 transition tables in one tensor.
    two = tensors.all_machines(2, normalise=False, device=dev)  # (B, 2, 2, 3)
    res2 = solve.run(two, budget)
    halted2 = res2["halt_time"] >= 0
    s2, sigma2 = int(res2["halt_time"].max()), int(res2["ones"][halted2].max())
    print(f"2 states: all {len(two):,} machines, budget {budget} steps: {int(halted2.sum()):,} halt, "
          f"S(2) = {s2} {'✓' if s2 == 6 else '✗'}, Σ(2) = {sigma2} {'✓' if sigma2 == 4 else '✗'}")  # fmt: skip

    # 2. The chosen number of states: exhaustive (normalised) for n ≤ 3, a random sample for n = 4.
    if n == 2:
        table, res, what = two, res2, "all 2-state machines"
    else:
        if n == 3:
            table, what = tensors.all_machines(3, normalise=True, device=dev), "all 3-state machines (A0 = 1RB)"
        else:
            table = tensors.random_machines(4, params["n_sample"], seed=0, device=dev)
            what = f"{params['n_sample']:,} random 4-state machines"
        res = solve.run(table, budget)
    print(f"tape {2 * budget + 1} cells × {len(table):,} machines = {res['tape'].numel() / 1e6:.0f} M cells")
    halt_time, ones = res["halt_time"], res["ones"]
    halted = halt_time >= 0
    s_n, sigma_n = int(halt_time.max()), int(ones[halted].max())
    frac = solve.halted_fraction(halt_time, budget)  # (budget + 1,)
    exact = n <= 3
    s_known, sigma_known = KNOWN[n]
    mark = (lambda found, known: "✓" if found == known else "✗") if exact else (lambda found, known: "")  # noqa: E731
    print(f"{what}: {int(halted.sum()):,} halt within {budget} steps, {int((~halted).sum()):,} undecided "
          f"({int(res['off_tape'].sum())} ran off the tape)")  # fmt: skip
    print(f"  longest run: S = {s_n} steps {mark(s_n, s_known)}   most 1s: Σ = {sigma_n} {mark(sigma_n, sigma_known)}"
          f"   (known S({n}) = {s_known}, Σ({n}) = {sigma_known})")  # fmt: skip
    best = solve.champions(halt_time, ones)
    print(f"  step champion {tensors.to_text(table[best[s_n]])},"
          f"  ones champion {tensors.to_text(table[int((ones * halted).argmax())])}")  # fmt: skip
    for t in sorted({1, 2, 5, 10, 20, 50, 100, 200, 500, budget} & set(range(budget + 1))):
        print(f"  budget {t:4d}: halted {frac[t]:7.2%}   undecided {1 - frac[t]:7.2%}")
    last = int(halt_time.max())
    print(f"  nothing halts after step {last}: the remaining {1 - frac[-1]:.2%} look like they run forever, "
          f"but no budget can prove it in general (the halting problem)")  # fmt: skip

    # 3. The known 4-state champion, run alone.
    bb4 = tensors.parse(tensors.CHAMPIONS["BB(4)"])
    one = solve.run(bb4[None], 200)
    s4, sigma4 = int(one["halt_time"][0]), int(one["ones"][0])
    print(f"BB(4) champion {tensors.CHAMPIONS['BB(4)']}: halts after {s4} steps {'✓' if s4 == 107 else '✗'} "
          f"with {sigma4} ones {'✓' if sigma4 == 13 else '✗'}")  # fmt: skip
    print(f"simulated {len(two) + (len(table) if n > 2 else 0) + 1:,} machines in {time.perf_counter() - tic:.1f} s "
          f"on {dev}")  # fmt: skip

    records = {s: solve.space_time(table[i], s) for s, i in best.items()}  # champion halting at each step count s
    check = f"S({n})={s_n} {mark(s_n, s_known)}  Σ({n})={sigma_n} {mark(sigma_n, sigma_known)}"
    if not exact:
        check = f"best in sample: S={s_n}, Σ={sigma_n}\n(true S(4)=107, Σ(4)=13)"
    ok4 = f"S={s4} {'✓' if s4 == 107 else '✗'}  Σ={sigma4} {'✓' if sigma4 == 13 else '✗'}"
    labels = {"batch": what, "check": check, "bb4": f"BB(4): {ok4}"}
    return plot.animate(records, frac, solve.halted_fraction(res2["halt_time"], budget), solve.space_time(bb4, s4),
                        labels, out_dir / "turing_halting.gif")  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
