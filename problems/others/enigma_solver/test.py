"""Physics checks, written first (TDD): what any correct Enigma code must satisfy.

Run with `pytest problems/others/enigma_solver/test.py`. Each test is a mathematical invariant of the machine, not
an implementation detail: known test vectors, every key press is a fixed-point-free involution (so decrypting the
ciphertext returns the message), the middle rotor double-steps, and the attack recovers the seeded settings.
"""

import torch

from . import equation, solve, tensors

RINGS = tensors.to_index("AAA")
NO_PLUGS = tensors.plugboard("")


def encrypt(text: str, order: tuple, start: str, rings: str = "AAA", plugs: str = "") -> str:
    out = solve.run_machine(
        tensors.to_index(text), tensors.machine(order), tensors.to_index(start)[None], tensors.to_index(rings),
        tensors.plugboard(plugs),
    )  # fmt: skip
    return tensors.to_text(out[0])


def test_known_test_vectors():
    assert encrypt("AAAAA", ("I", "II", "III"), "AAA") == "BDZGO"
    assert encrypt("AAAAA", ("I", "II", "III"), "AAA", rings="BBB") == "EWTYX"  # Wikipedia, Enigma rotor details


def test_ring_setting_shifts_the_wiring_like_the_start_position():
    # Rings and start both B: same rotor offsets as AAA/AAA and no turnover in 5 letters ⇒ the same output.
    assert encrypt("AAAAA", ("I", "II", "III"), "BBB", rings="BBB") == "BDZGO"
    assert encrypt("AAAAA", ("I", "II", "III"), "AAA", rings="BBB") != "BDZGO"


def test_rotors_and_reflector_are_permutations():
    for wiring, _ in tensors.ROTORS.values():
        assert sorted(tensors.to_index(wiring).tolist()) == list(range(26))  # bijection of the 26 letters
    u = tensors.perm(tensors.REFLECTOR_B)
    assert torch.equal(u[u], torch.arange(26)) and not (u == torch.arange(26)).any()  # reflector: U² = I, no fixed pt


def test_encryption_is_an_involution_without_fixed_points():
    m = tensors.machine(("IV", "II", "V"))
    e = solve.keypress_matrices(m, tensors.to_index("QEV"), tensors.to_index("CKM"), tensors.plugboard("AQ BW EX"), 80)
    assert torch.equal(e @ e, torch.eye(26).expand_as(e))  # E² = I: the machine is its own inverse
    assert torch.equal(e.transpose(-1, -2), e)  # E is symmetric
    assert e.diagonal(dim1=-2, dim2=-1).sum() == 0  # no letter maps to itself
    text = "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG" * 3
    cipher = encrypt(text, ("IV", "II", "V"), "QEV", "CKM", "AQ BW EX")
    assert encrypt(cipher, ("IV", "II", "V"), "QEV", "CKM", "AQ BW EX") == text  # decoded message equals input
    assert all(a != b for a, b in zip(text, cipher))


def test_matrix_form_matches_index_form():
    m = tensors.machine(("II", "V", "III"))
    start, plugs = tensors.to_index("XZK"), tensors.plugboard("PO ML")
    e = solve.keypress_matrices(m, start, RINGS, plugs, 26)  # (26, 26, 26): one matrix per key press
    x = tensors.to_index("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    via_matrix = e[torch.arange(26), :, x].argmax(-1)  # column x_t of E_t
    assert torch.equal(via_matrix, solve.run_machine(x, m, start[None], RINGS, plugs)[0])
    assert torch.equal(equation.rotor_table(m["w"][0])[0], m["w"][0])  # offset 0 is the bare wiring


def test_double_step():
    notch = tensors.machine(("I", "II", "III"))["notch"]  # III turns over at V, II at E
    seen = [tensors.to_text(p) for p in solve.positions(tensors.to_index("ADU")[None], notch, 3)[0]]
    assert seen == ["ADV", "AEW", "BFX"]  # the middle rotor steps on two key presses in a row


def test_all_positions_matches_stepping_every_start():
    notch = tensors.machine(("V", "I", "II"))["notch"]
    full = solve.positions(tensors.all_starts(), notch, 30)
    assert torch.equal(solve.all_positions(notch, 30), full)


def test_index_of_coincidence_of_english_and_uniform_text():
    english = tensors.to_index(tensors.MESSAGES["lighthouse"])[None]
    ioc = equation.index_of_coincidence(equation.letter_counts(english))
    assert 0.058 < ioc.item() < 0.075  # English letter frequencies: IoC ≈ 0.066
    uniform = torch.arange(26).repeat(10)[None]
    expected = 26 * 10 * 9 / (260 * 259)  # closed form Σ nᵢ(nᵢ − 1) / N(N − 1)
    assert abs(equation.index_of_coincidence(equation.letter_counts(uniform)).item() - expected) < 1e-6


def test_solver_recovers_seeded_settings():
    gen = torch.Generator().manual_seed(7)
    orders = tensors.rotor_orders("I-III")
    order = orders[int(torch.randint(len(orders), (1,), generator=gen))]
    start = torch.randint(26, (3,), generator=gen)
    plain = tensors.to_index(tensors.MESSAGES["permutations"])
    cipher = solve.run_machine(plain, tensors.machine(order), start[None], RINGS, NO_PLUGS)[0]
    for method in ("ioc", "unigram"):
        result = solve.break_enigma(cipher[:150], orders, RINGS, method)
        assert orders[result["order"]] == order
        assert torch.equal(result["start"], start)
