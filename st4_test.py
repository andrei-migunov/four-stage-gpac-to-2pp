"""
test_make_lpp.py
Claude 5/28
================
A single unit test for St4_Fns.make_lpp on a 5-species PP-implementable
quadratic-form ODE system.


THE TEST SYSTEM
---------------
Species:  a, b, c, d, e   (lowercase, to verify make_lpp uppercases correctly).

ODEs:
    a' = -a*b + 2*c*d - a*c + 3*b*e
    b' = -a*b + 2*c*d - 3*b*e - 2*b**2
    c' =  a*b - 2*c*d - a*c + 3*b*e + 2*b**2
    d' =  a*b - 2*c*d
    e' =  2*a*c - 3*b*e

Provenance (intuition only -- the test feeds the ODEs in directly):
this is the deterministic mass-action ODE of five 2-in-2-out reactions

    a + b ---> c + d        (rate 1)
    c + d ---> a + b        (rate 2)
    a + c ---> e + e        (rate 1)
    b + e ---> a + c        (rate 3)
    b + b ---> c + c        (rate 1).

The b+b -> c+c reaction is the one that introduces a b**2 monomial,
exercising the "j = k" branch of Construction 1.

Properties verifiable from the ODEs alone:
  * homogeneous quadratic; rational coefficients;
  * conservative:  a' + b' + c' + d' + e'  identically zero;
  * every negative term in d[x_i] has x_i as a factor (CRN-implementable);
  * no positive x_i^2 term anywhere (PP-implementable);
  * q_max = 3 (largest negative coefficient magnitude; attained on
    -3*b*e in both b' and e');
  * default epsilon = 1/3.


DESIRED OUTCOME of make_lpp(system, iv, mainvar=e)
--------------------------------------------------
  (1)  Returns (lpp, lpp_iv) without raising.

  (2)  Species names in the lpp are UPPERCASE strings drawn from
       {'A', 'B', 'C', 'D', 'E'} -- make_lpp uppercases via str(sym).upper().

  (3)  All 25 ordered-pair keys (X, Y) with X, Y in {A..E} are present:
       the +2 x_X x_Y term in f_X always contributes (X, Y) -> (X, X)
       at probability >= 2*(1/3)/4 = 1/6, so no ordered pair is empty.

  (4)  For EACH ordered-pair key, the probabilities sum to exactly 1
       (sympy Rational equality -- not float-approximate).

  (5)  Construction is symmetric: lpp[(X, Y)] equals lpp[(Y, X)] as
       distributions, for every X != Y.

  (6)  Round-trip: lpp_to_odes(lpp) reproduces (1/3) * system exactly as
       quadratic POLYNOMIALS -- no sum_l x_l = 1 substitution required.
       This is the strongest possible verification of make_lpp's
       correctness on this input.

  (7)  Hand-checkable spot checks:

       lpp[('A', 'B')]  ==  { ('A', 'A') : 5/12,
                              ('B', 'B') : 5/12,
                              ('C', 'C') : 1/12,
                              ('D', 'D') : 1/12 }

         Derivation: the coefficient of A*B in f_i, summed over i, equals 4:
           f_a: eps*(-1) + 2  (from 2*a*sum, l=b)  =  -1/3 + 2 =  5/3
           f_b: eps*(-1) + 2  (from 2*b*sum, l=a)  =  -1/3 + 2 =  5/3
           f_c: eps*(+1) + 0                        =          1/3
           f_d: eps*(+1) + 0                        =          1/3
           f_e: eps*( 0) + 0                        =            0
         Probabilities = coefficient / 4.

       lpp[('B', 'B')]  ==  { ('B', 'B') : 2/3,
                              ('C', 'C') : 1/3 }

         Derivation: the coefficient of B**2 in f_i, summed over i, equals 2:
           f_a: 0
           f_b: eps*(-2) + 2  (from 2*b*sum, l=b)  =  -2/3 + 2 =  4/3
           f_c: eps*(+2) + 0                        =          2/3
           f_d: 0
           f_e: 0
         Probabilities = coefficient / 2.

  (8)  lpp_iv keys are uppercase strings {'A', ..., 'E'} with the input
       proportions preserved (and summing to 1).
"""

import sys
sys.path.insert(0, '/mnt/user-data/outputs')

import sympy as sp
from sympy import Symbol, Rational, expand
from St4_Fns import make_lpp, lpp_to_odes, compute_default_eps


def test_make_lpp_on_5_species_system():
    a, b, c, d, e = sp.symbols('a b c d e')

    system = {
        a: -a*b + 2*c*d - a*c + 3*b*e,
        b: -a*b + 2*c*d - 3*b*e - 2*b**2,
        c:  a*b - 2*c*d - a*c + 3*b*e + 2*b**2,
        d:  a*b - 2*c*d,
        e:  2*a*c - 3*b*e,
    }
    iv = {
        a: Rational(2, 5),
        b: Rational(1, 5),
        c: Rational(1, 5),
        d: Rational(1, 10),
        e: Rational(1, 10),
    }
    mainvar = e

    # ------------------------------------------------------------------
    # (1) make_lpp runs without raising
    # ------------------------------------------------------------------
    lpp, lpp_iv = make_lpp(system, iv, mainvar, checks=True, verbose=True)

    # ------------------------------------------------------------------
    # (2) Species names uppercased
    # ------------------------------------------------------------------
    found = set()
    for (x, y) in lpp:
        found.add(x); found.add(y)
    assert found == {'A', 'B', 'C', 'D', 'E'}, \
        f"Unexpected species names: {found}"

    # ------------------------------------------------------------------
    # (3) All 25 ordered-pair keys present
    # ------------------------------------------------------------------
    expected_keys = {(x, y) for x in 'ABCDE' for y in 'ABCDE'}
    missing = expected_keys - set(lpp.keys())
    extra   = set(lpp.keys()) - expected_keys
    assert not missing and not extra, \
        f"Missing ordered pairs: {missing}; extra: {extra}"

    # ------------------------------------------------------------------
    # (4) Per-ordered-pair sums == 1 exactly
    # ------------------------------------------------------------------
    for pair, dist in lpp.items():
        s = sum(dist.values())
        assert s == 1, f"Pair {pair} sums to {s}, not 1"

    # ------------------------------------------------------------------
    # (5) Symmetry of the construction
    # ------------------------------------------------------------------
    for (x, y), dist_xy in lpp.items():
        if x != y:
            dist_yx = lpp[(y, x)]
            assert dist_xy == dist_yx, \
                f"({x},{y}) != ({y},{x}): {dist_xy} vs {dist_yx}"

    # ------------------------------------------------------------------
    # (6) Round-trip: lpp_to_odes(lpp) == eps * system (uppercased)
    # ------------------------------------------------------------------
    eps = compute_default_eps(system)
    assert eps == Rational(1, 3), f"Default eps = {eps}, expected 1/3"

    odes_back = lpp_to_odes(lpp)

    A, B, C, D, E = sp.symbols('A B C D E')
    lower_to_upper = {a: A, b: B, c: C, d: D, e: E}
    for sym, expr in system.items():
        upper_sym = Symbol(str(sym).upper())
        expected  = expand(eps * expr.subs(lower_to_upper))
        recovered = odes_back[upper_sym]
        diff = expand(recovered - expected)
        assert diff == 0, (
            f"ROUND-TRIP FAILED on d[{sym}]/dt.\n"
            f"  Expected (eps * d[{sym}]):  {expected}\n"
            f"  Recovered from lpp:          {recovered}\n"
            f"  Difference:                  {diff}"
        )

    # ------------------------------------------------------------------
    # (7) Spot checks (hand-derivable)
    # ------------------------------------------------------------------
    expected_AB = {
        ('A', 'A'): Rational(5, 12),
        ('B', 'B'): Rational(5, 12),
        ('C', 'C'): Rational(1, 12),
        ('D', 'D'): Rational(1, 12),
    }
    actual_AB = dict(lpp[('A', 'B')])
    assert actual_AB == expected_AB, (
        f"(A,B) distribution mismatch.\n"
        f"  Expected: {expected_AB}\n"
        f"  Got:      {actual_AB}"
    )

    expected_BB = {
        ('B', 'B'): Rational(2, 3),
        ('C', 'C'): Rational(1, 3),
    }
    actual_BB = dict(lpp[('B', 'B')])
    assert actual_BB == expected_BB, (
        f"(B,B) distribution mismatch.\n"
        f"  Expected: {expected_BB}\n"
        f"  Got:      {actual_BB}"
    )

    # ------------------------------------------------------------------
    # (8) Initial values: renamed to uppercase, values preserved
    # ------------------------------------------------------------------
    assert set(lpp_iv) == {'A', 'B', 'C', 'D', 'E'}
    assert lpp_iv['A'] == Rational(2, 5)
    assert lpp_iv['B'] == Rational(1, 5)
    assert lpp_iv['C'] == Rational(1, 5)
    assert lpp_iv['D'] == Rational(1, 10)
    assert lpp_iv['E'] == Rational(1, 10)
    assert sum(sp.sympify(v) for v in lpp_iv.values()) == 1

    print("\nAll 8 checks passed.")
    return lpp, lpp_iv


if __name__ == '__main__':
    test_make_lpp_on_5_species_system()