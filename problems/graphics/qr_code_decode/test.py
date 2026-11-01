"""Physics checks, written first (TDD): what any correct QR decoder code must satisfy.

Run with `pytest problems/graphics/qr_code_decode/test.py`. Each test is a mathematical invariant, not an implementation
detail: GF(256) obeys the field axioms, Reed–Solomon corrects up to T/2 byte errors, the BCH format code has distance 7,
masks are involutions, and the decoded message equals the input.
"""

import itertools

import torch

from . import equation, solve, tensors

ALL = torch.arange(256)
NONZERO = torch.arange(1, 256)
MESSAGE = b"Reed-Solomon 1960"


def test_gf256_field_axioms():
    a, b = NONZERO[:, None], NONZERO[None]
    prod = equation.gf_mul(a, b)
    assert (prod != 0).all()  # no zero divisors
    assert torch.equal(equation.gf_div(prod, b.expand_as(prod)), a.expand_as(prod))  # (a ⊗ b) ⊘ b = a
    assert torch.equal(prod, prod.T)  # commutative
    inv = equation.gf_div(torch.ones(255, dtype=torch.long), NONZERO)
    assert (equation.gf_mul(NONZERO, inv) == 1).all()  # every non-zero element has an inverse
    assert (equation.gf_mul(ALL, torch.zeros_like(ALL)) == 0).all() and torch.equal(
        equation.gf_mul(ALL, torch.ones_like(ALL)), ALL
    )
    g = torch.Generator().manual_seed(0)
    x, y, z = (torch.randint(0, 256, (2000,), generator=g) for _ in range(3))
    assert torch.equal(
        equation.gf_mul(x, equation.gf_mul(y, z)), equation.gf_mul(equation.gf_mul(x, y), z)
    )  # associative
    assert torch.equal(equation.gf_mul(x, y ^ z), equation.gf_mul(x, y) ^ equation.gf_mul(x, z))  # distributes over XOR


def test_alpha_generates_the_multiplicative_group():
    powers = equation.gf_pow_alpha(torch.arange(255))
    assert sorted(powers.tolist()) == list(range(1, 256))  # α^0 … α^254 are all non-zero bytes, once each
    assert equation.gf_pow_alpha(torch.tensor(255)).item() == 1  # order 255
    assert all(equation.gf_pow_alpha(torch.tensor(255 // p)).item() != 1 for p in (3, 5, 17))  # not a smaller order
    assert equation.gf_pow_alpha(torch.tensor(8)).item() == 0x1D  # α^8 = x^4 + x^3 + x^2 + 1 modulo 0x11D


def test_gf_sum_is_xor_reduction():
    x = torch.randint(0, 256, (7, 13), generator=torch.Generator().manual_seed(1))
    expect = torch.zeros(7, dtype=torch.long)
    for k in range(13):
        expect ^= x[:, k]
    assert torch.equal(equation.gf_sum(x, 1), expect)


def test_rs_encoder_matches_known_vectors_and_gives_zero_syndromes():
    # ISO/IEC 18004 worked example "01234567", version 1-M (16 data + 10 ECC codewords).
    iso = torch.tensor([[0x10, 0x20, 0x0C, 0x56, 0x61, 0x80] + [0xEC, 0x11] * 5])
    assert equation.rs_encode(iso, 10)[0].tolist() == [0xA5, 0x24, 0xD4, 0xC1, 0xED, 0x36, 0xC7, 0x87, 0x2C, 0x55]
    g = torch.Generator().manual_seed(2)
    for k, t in [(19, 7), (16, 10), (13, 22), (34, 18)]:
        d = torch.randint(0, 256, (20, k), generator=g)
        c = torch.cat([d, equation.rs_encode(d, t)], 1)
        assert (equation.syndromes(c, t) == 0).all()
        assert (equation.syndromes(c ^ (torch.arange(k + t) == 3).long() * 5, t) != 0).any(1).all()


def test_generator_polynomial_has_roots_alpha_0_to_T_minus_1():
    g = equation.rs_generator(10)  # descending coefficients
    for j in range(10):
        assert equation.gf_poly_eval(g.flip(0)[None], equation.gf_pow_alpha(torch.tensor([j])))[0, 0] == 0
    assert equation.gf_poly_eval(g.flip(0)[None], equation.gf_pow_alpha(torch.tensor([10])))[0, 0] != 0


def corrupt(c: torch.Tensor, n_err: int, g: torch.Generator) -> torch.Tensor:
    r = c.clone()
    for b in range(len(r)):
        pos = torch.randperm(c.shape[1], generator=g)[:n_err]
        r[b, pos] ^= torch.randint(1, 256, (n_err,), generator=g)  # non-zero error values: real byte errors
    return r


def test_rs_corrects_up_to_t_errors_and_fails_gracefully_beyond():
    g = torch.Generator().manual_seed(3)
    k, T = 22, 22  # version 2-Q block: t = 11
    d = torch.randint(0, 256, (300, k), generator=g)
    c = torch.cat([d, equation.rs_encode(d, T)], 1)
    for n_err in (1, 5, T // 2):
        fixed, count, ok = solve.rs_decode(corrupt(c, n_err, g), T)
        assert ok.all() and torch.equal(fixed, c) and (count == n_err).all()
    received = corrupt(c, T // 2 + 1, g)
    fixed, count, ok = solve.rs_decode(received, T)
    assert ok.float().mean() < 0.05  # almost always detected as uncorrectable
    assert torch.equal(fixed[~ok], received[~ok]) and (count[~ok] == 0).all()  # left untouched, nothing claimed
    assert (equation.syndromes(fixed[ok], T) == 0).all()  # a rare mis-correction still lands on a codeword


def test_format_codes_match_the_standard_and_are_7_apart():
    expect = {("L", 0): "111011111000100", ("M", 0): "101010000010010", ("Q", 0): "011010101011111",
              ("H", 0): "001011010001001"}  # fmt: skip
    for (level, mask), bits in expect.items():
        assert f"{equation.format_code(tensors.ECL_BITS[level], mask):015b}" == bits
    table = equation.format_table()
    dist = (table[:, None] != table[None]).sum(-1) + 99 * torch.eye(32, dtype=torch.long)
    assert dist.min() == 7


def test_format_decoding_fixes_up_to_three_bit_errors():
    table = equation.format_table()
    for n_err in range(4):
        flips = torch.zeros(len(list(itertools.combinations(range(15), n_err))) or 1, 15, dtype=torch.long)
        for row, pos in enumerate(itertools.combinations(range(15), n_err)):
            flips[row, list(pos)] = 1
        received = (table[:, None] ^ flips[None]).reshape(-1, 15)  # every codeword × every error pattern
        index, errors = equation.format_decode(received)
        assert torch.equal(index, torch.arange(32).repeat_interleave(len(flips)))
        assert (errors == n_err).all()


def test_masks_are_involutions_and_unmasking_recovers_the_codewords():
    for version in (1, 2, 3):
        n = tensors.size(version)
        grid = torch.randint(0, 2, (n, n), generator=torch.Generator().manual_seed(version))
        for k in range(8):
            m = equation.mask_pattern(k, n).long()
            assert torch.equal((grid ^ m) ^ m, grid)
        for level in tensors.LEVELS:
            sym = tensors.symbol(MESSAGE[: tensors.layout(version, level)["capacity"]], version, level)
            stream, _ = tensors.codewords(MESSAGE[: tensors.layout(version, level)["capacity"]], version, level)
            read = solve.read_codewords(sym["grid"][None], version, torch.tensor([sym["mask"]]))
            assert torch.equal(read[0], stream)
            blocks = solve.deinterleave(read, version, level)[0]
            assert torch.equal(blocks, sym["blocks"])


def test_symbol_layout_counts_match_the_standard():
    for version in (1, 2, 3):
        _, fun = tensors.function_patterns(version)
        raw = int((~fun).sum())
        assert raw // 8 == tensors.TOTAL_CODEWORDS[version] and raw % 8 == (0 if version == 1 else 7)
        assert len(tensors.data_positions(fun)) == raw
    assert [tensors.layout(1, e)["capacity"] for e in "LMQH"] == [17, 14, 11, 7]  # byte-mode capacities
    assert [tensors.layout(3, e)["capacity"] for e in "LMQH"] == [53, 42, 32, 24]


def test_clean_render_round_trip_for_every_level():
    g = torch.Generator().manual_seed(4)
    for level in tensors.LEVELS:
        version = tensors.smallest_version(MESSAGE, level, 2)
        sym = tensors.symbol(MESSAGE, version, level)
        img, _ = tensors.photos(sym["grid"], 2, 240, 0.0, 0.0, 0.0, g)
        results = solve.decode(img)
        for r in results:
            assert r["text"] == MESSAGE.decode()
            assert r["version"] == version and r["ecl"] == level and r["mask"] == sym["mask"]
            assert r["byte_errors"] == [0] * tensors.layout(version, level)["blocks"]


def test_perspective_warped_noisy_photos_decode():
    g = torch.Generator().manual_seed(5)
    sym = tensors.symbol(MESSAGE, 3, "Q")
    img, _ = tensors.photos(sym["grid"], 6, 240, 0.1, 0.0, 0.04, g)
    results = solve.decode(img)
    assert sum(r["text"] == MESSAGE.decode() for r in results) >= 5


def test_occlusion_is_corrected_by_reed_solomon():
    g = torch.Generator().manual_seed(6)
    sym = tensors.symbol(MESSAGE, 3, "H")
    img, _ = tensors.photos(sym["grid"], 4, 240, 0.05, 0.08, 0.03, g)
    results = solve.decode(img)
    good = [r for r in results if r["text"] == MESSAGE.decode()]
    assert len(good) >= 3
    assert all(sum(r["byte_errors"]) > 0 for r in good)  # the blob really caused byte errors that RS fixed
