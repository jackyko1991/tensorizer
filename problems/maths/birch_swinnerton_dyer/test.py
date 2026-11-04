"""Mathematical checks, written first (TDD): what any correct point count and L-function of an elliptic curve satisfy.

Run with `pytest problems/maths/birch_swinnerton_dyer/test.py`. Each test is a theorem or a published value: Hasse's
bound |a_p| ≤ 2√p, the a_p of 11a1 are the coefficients of the modular form η(q)²η(q¹¹)² (modularity), the bad
primes are those dividing the discriminant, the L-values of 11a1 and 37a1 match the LMFDB, 11a1 satisfies the
rank-0 Birch–Swinnerton-Dyer formula L(E,1)/Ω = 1/5, and ∏ N_p/p grows faster for curves of higher rank.
"""

import torch

from . import equation, solve, tensors

CPU = torch.device("cpu")


def test_point_count_matches_brute_force():
    coeffs = tensors.curve_coefficients(CPU)  # (C, 5)
    primes = tensors.primes_up_to(60, CPU)
    a_p = equation.frobenius_traces(coeffs, primes)  # (C, P)
    # slow reference: count every (x, y) in F_p × F_p that solves the Weierstrass equation
    for c, (a1, a2, a3, a4, a6) in enumerate(coeffs.tolist()):
        for i, p in enumerate(primes.tolist()):
            count = sum(
                (y * y + a1 * x * y + a3 * y - x**3 - a2 * x * x - a4 * x - a6) % p == 0
                for x in range(p)
                for y in range(p)
            )
            assert a_p[c, i].item() == p - count  # #E(F_p) = p + 1 − a_p, counting the point at infinity


def test_hasse_bound():
    coeffs = tensors.curve_coefficients(CPU)
    primes = tensors.primes_up_to(5000, CPU)
    a_p = equation.frobenius_traces(coeffs, primes).double()
    assert torch.all(a_p.abs() <= 2 * primes.double().sqrt())  # Hasse 1933: |a_p| ≤ 2√p


def test_11a1_is_modular():
    """Eichler–Shimura: the a_p of y² + y = x³ − x² − 10x − 20 are the coefficients of
    η(q)²η(q¹¹)² = q∏(1−qⁿ)²(1−q¹¹ⁿ)²."""
    n_max = 400
    series = torch.zeros(n_max + 1, dtype=torch.float64)
    series[1] = 1.0  # the leading q
    for n in range(1, n_max + 1):
        for step in (n, n, 11 * n, 11 * n):  # multiply by (1 − q^step), truncated at q^n_max
            if step <= n_max:
                series[step:] = series[step:] - series[:-step].clone()
    primes = tensors.primes_up_to(n_max, CPU)
    coeffs = tensors.curve_coefficients(CPU)[:1]  # 11a1
    a_p = equation.frobenius_traces(coeffs, primes)[0]
    good = primes != 11
    assert torch.equal(a_p[good].double(), series[primes[good]])
    assert a_p[:6].tolist() == [-2, -1, 1, -2, 1, 4]  # a_2 … a_13 as in the LMFDB


def test_37a1_traces_and_bad_primes():
    coeffs = tensors.curve_coefficients(CPU)
    primes = tensors.primes_up_to(30, CPU)
    a_p = equation.frobenius_traces(coeffs[1:2], primes)[0]
    assert a_p.tolist() == [-2, -3, -2, -1, -5, -2, 0, 0, 2, 6]  # 37a1: q − 2q² − 3q³ + … (LMFDB 37.a1)
    disc = equation.discriminant(coeffs)
    assert disc.tolist() == [-161051, 37, 389, 5077, 468892]  # −11⁵, 37, 389, 5077, 2²·117223: bad primes | conductor


def test_l_values_match_lmfdb():
    res = solve.l_function(tensors.curve_coefficients(CPU)[:2], tensors.CONDUCTORS[:2], n_max=400)
    assert res["root_number"].tolist() == [1, -1]
    assert abs(res["leading"][0].item() - 0.2538418608559107) < 1e-9  # L(11a1, 1)
    assert abs(res["leading"][1].item() - 0.3059997738340523) < 1e-9  # L′(37a1, 1)
    assert res["order"].tolist() == [0, 1]


def test_bsd_formula_for_11a1():
    """Rank 0: L(E,1)/Ω = #Sha · ∏c_p / #E(Q)_tors² = 1 · 5 / 5² = 1/5 for 11a1 (a theorem for this curve)."""
    coeffs = tensors.curve_coefficients(CPU)[:1]
    omega = equation.real_period(coeffs)[0].item()
    assert abs(omega - 1.2692093042795534) < 1e-9
    l1 = solve.l_function(coeffs, tensors.CONDUCTORS[:1], n_max=200)["leading"][0].item()
    assert abs(l1 / omega - 0.2) < 1e-9


def test_analytic_ranks_match_known_ranks():
    res = solve.l_function(tensors.curve_coefficients(CPU), tensors.CONDUCTORS, n_max=3000)
    assert res["order"].tolist() == tensors.RANKS  # analytic rank = Mordell–Weil rank (0, 1, 2, 3, 4)
    assert res["root_number"].tolist() == [1, -1, 1, -1, 1]  # ε = (−1)^rank (parity)
    # leading Taylor coefficients L^{(r)}(E,1)/r! from the LMFDB for 389a1 and 5077a1
    assert abs(res["taylor"][2].item() - 0.7593165002884) < 1e-8
    assert abs(res["taylor"][3].item() - 1.7318499001193) < 1e-8


def test_partial_products_grow_with_rank():
    coeffs = tensors.curve_coefficients(CPU)
    primes = tensors.primes_up_to(20000, CPU)
    log_prod = solve.log_partial_products(coeffs, primes)  # (C, P)
    assert torch.all(log_prod[:, -1].diff() > 0)  # ∏ N_p/p is larger for higher rank
    slope = solve.rank_estimate(log_prod, primes)
    assert torch.all(slope[:4].diff() > 0.5)  # ranks 0–3 well separated (rank 4 converges too slowly here)
    assert torch.all((slope[:4] - torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float64)).abs() < 0.6)
