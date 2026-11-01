"""Main loop: parameters → tensors → encrypt → break → plot."""

import time
from pathlib import Path

import torch

from problems._common.runtime import cli, device, fast_mode

from . import plot, solve, tensors

RINGS = "AAA"  # ring settings of the secret key; the attacker is assumed to know them


def run(params: dict, out_dir: Path) -> Path:
    dev = device()
    rings, plugs = tensors.to_index(RINGS, dev), tensors.plugboard("", dev)  # empty plugboard S = I
    order = tuple(params["rotor_order"].split("-"))
    start = tensors.to_index(params["start_left"] + params["start_middle"] + params["start_right"], dev)  # (3,)
    m = tensors.machine(order, dev)

    # 1. The machine: a known test vector, then the secret message in matrix form and in index form.
    test = solve.run_machine(tensors.to_index("AAAAA", dev), tensors.machine(("I", "II", "III"), dev),
                             tensors.to_index("AAA", dev)[None], rings, plugs)  # fmt: skip
    print(f"test vector: rotors I-II-III, reflector B, rings AAA, start AAA: AAAAA → {tensors.to_text(test[0])}")
    plain = tensors.to_index(tensors.MESSAGES[params["message"]], dev)  # (T,)
    e = solve.keypress_matrices(m, start, rings, plugs, len(plain))  # (T, 26, 26), one E_t per key press
    eye = torch.eye(26, device=dev)
    print(f"secret key: rotors {params['rotor_order']}, rings {RINGS}, start {tensors.to_text(start)}, no plugs")
    print(f"{len(plain)} key presses: E_t E_t = I for all t: {torch.equal(e @ e, eye.expand_as(e))};"
          f"  letters mapped to themselves: {int(e.diagonal(dim1=-2, dim2=-1).sum())}")  # fmt: skip
    cipher_matrix = torch.einsum("tij,tj->ti", e, tensors.matrix(plain).transpose(-1, -2)).argmax(-1)  # E_t e_{x_t}
    cipher = solve.run_machine(plain, m, start[None], rings, plugs)[0]  # (T,) the same with table look-ups
    pos = solve.positions(start[None], m["notch"], len(plain))[0]  # (T, 3) window letters at each key press
    before = torch.cat([start[None], pos[:-1]])  # (T, 3) window letters just before each key press
    double_steps = int((pos[:, 0] != before[:, 0]).sum())  # each left step comes with a middle double step
    print(f"matrix form = index form: {torch.equal(cipher_matrix, cipher)};  double steps: {double_steps}")
    groups = " ".join(tensors.to_text(cipher[i : i + 5]) for i in range(0, 60, 5))
    print(f"plaintext : {tensors.to_text(plain[:60])}…\nciphertext: {groups} …")

    # 2. The attack: decrypt with all 26³ start positions for every rotor order and keep the most English-looking.
    orders = tensors.rotor_orders(params["search_rotors"])
    n = min(params["n_letters"], len(cipher), 120 if fast_mode() else len(cipher))
    tic = time.perf_counter()
    result = solve.break_enigma(cipher[:n], orders, rings, params["score"])
    seconds = time.perf_counter() - tic
    scores = result["scores"]  # (O, 26, 26, 26)
    best_per_order = scores.flatten(1).amax(1)  # (O,)
    print(f"scored {len(orders)} rotor orders × 26³ = {len(orders) * 26**3:,} keys on {n} letters "
          f"({params['score']}) in {seconds:.1f} s")  # fmt: skip
    for i in best_per_order.argsort(descending=True)[:4].tolist():
        print(f"  {'-'.join(orders[i]):>11}: best score {best_per_order[i].item():.4f}")
    best_order, best_start = orders[result["order"]], result["start"]
    found = best_order == order and torch.equal(best_start, start)
    print(f"recovered: rotors {'-'.join(best_order)}, start {tensors.to_text(best_start)}  "
          f"({'correct' if found else 'wrong: try more letters or the other score'})")  # fmt: skip

    # 3. Decrypt the whole message with the recovered key (decryption = encryption), and a preview per order.
    decrypted = solve.run_machine(cipher, tensors.machine(best_order, dev), best_start[None], rings, plugs)[0]
    print(f"decrypted : {tensors.to_text(decrypted)}")
    previews = []
    for i, o in enumerate(orders):
        b = int(scores[i].flatten().argmax())
        s_i = tensors.all_starts(dev)[b]
        y = solve.run_machine(cipher[:40], tensors.machine(o, dev), s_i[None], rings, plugs)[0]
        previews.append(f"{'-'.join(o)} at {tensors.to_text(s_i)}: {tensors.to_text(y)}…")
    return plot.animate(
        scores.amax(1), best_per_order, orders, previews, tensors.to_text(cipher), tensors.to_text(decrypted),
        result["order"], best_start, params["score"], out_dir / "enigma_solver.gif",
    )  # fmt: skip


if __name__ == "__main__":
    cli(run, __file__)
