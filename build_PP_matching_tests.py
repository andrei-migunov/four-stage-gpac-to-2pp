from St3_Fns import buildPP
from main import pp_reactions_to_odes
import sympy as sp

# ── Pretty-print helpers ───────────────────────────────────────────────────────

def print_system(sys, title="ODE system"):
    print(f"  {title}:")
    for var, expr in sys.items():
        print(f"    d[{var}]/dt = {expr}")

def print_reactions(rxns, title="Reactions"):
    print(f"  {title}:")
    for coeff, lhs, rhs in rxns:
        lhs_str = " + ".join(lhs)
        rhs_str = " + ".join(rhs)
        print(f"    {coeff} : {lhs_str} -> {rhs_str}")

def print_odes(odes, title="Recovered ODEs"):
    print(f"  {title}:")
    for var, expr in sorted(odes.items(), key=lambda item: str(item[0])):
        print(f"    d[{var}]/dt = {expr}")

def print_separator(label, n=60):
    print("\n" + "─" * n)
    print(f"  {label}")
    print("─" * n)

def assert_round_trip(odes, sys, label):
    """Assert that `odes` (uppercase keys) matches `sys` (lowercase keys)
    for every variable present in `sys`."""
    all_syms = set(sys.keys())
    for expr in sys.values():
        all_syms |= expr.free_symbols
    subs = {var: sp.Symbol(str(var).upper()) for var in all_syms}
    for var, expr in sys.items():
        upper_var  = subs[var]
        upper_expr = sp.expand(expr.subs(subs))
        assert upper_var in odes, \
            f"{label}: '{upper_var}' missing from recovered ODEs"
        assert sp.expand(odes[upper_var] - upper_expr) == 0, \
            f"{label}: d[{upper_var}]/dt mismatch\n" \
            f"  expected: {upper_expr}\n" \
            f"  got:      {odes[upper_var]}"

# ── Tests ──────────────────────────────────────────────────────────────────────

def test_build_PP():
    x, y, z = sp.symbols('x y z')
    X, Y, Z = sp.symbols('X Y Z')   # uppercase counterparts expected in reactions

    # ------------------------------------------------------------------
    # Test 1: single negative term
    #   d[x]/dt = -2*x*y,  d[y]/dt = +2*x*y
    #   Expected reaction: X + Y -> Y + Y  (X converts to Y, rate 2)
    #   Reaction appears twice (once per equation).
    # ------------------------------------------------------------------
    print_separator("Test 1: single negative term")
    sys1 = {x: sp.Integer(-2) * x * y, y: sp.Integer(2) * x * y}
    print_system(sys1)
    rxns1 = buildPP(sys1, x)
    print_reactions(rxns1)
    odes1 = pp_reactions_to_odes(rxns1)
    print_odes(odes1)

    assert len(rxns1) == 1,                "Test 1: The ODEs have matching terms that produce one reaction - but buildPP produced more."
    c1, lhs1, rhs1 = rxns1[0]
    assert c1 == 2,                        "Test 1: wrong rate"
    assert sorted(lhs1) == ['X', 'Y'],     "Test 1: wrong reactants"
    assert sorted(rhs1) == ['Y', 'Y'],     "Test 1: wrong products"
    assert_round_trip(odes1, sys1,         "Test 1")
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 2: single positive term (var is a factor)
    #   d[x]/dt = +3*x*y,  d[y]/dt = -3*x*y
    #   Expected reaction: X + Y -> X + X  (Y converts to X, rate 3)
    #   Reaction appears twice (once per equation).
    # ------------------------------------------------------------------
    print_separator("Test 2: single positive term")
    sys2 = {x: sp.Integer(3) * x * y, y: sp.Integer(-3) * x * y}
    print_system(sys2)
    rxns2 = buildPP(sys2, x)
    print_reactions(rxns2)
    odes2 = pp_reactions_to_odes(rxns2)
    print_odes(odes2)

    assert len(rxns2) == 1,                "Test 2: The ODEs have matching terms that produce one reaction - but buildPP produced more."
    c2, lhs2, rhs2 = rxns2[0]
    assert c2 == 3,                        "Test 2: wrong rate"
    assert sorted(lhs2) == ['X', 'Y'],     "Test 2: wrong reactants"
    assert sorted(rhs2) == ['X', 'X'],     "Test 2: wrong products"
    assert_round_trip(odes2, sys2,         "Test 2")
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 3: round-trip for a single-equation system
    #   d[x]/dt = 2*x*y - 3*x*z
    #   Reactions: X+Y->X+X (rate 2),  X+Z->Z+Z (rate 3)
    #   Only x's equation is passed, so no duplication; pp_reactions_to_odes
    #   should recover the full implied system exactly.
    # ------------------------------------------------------------------
    print_separator("Test 3: single-equation round-trip")
    sys3 = {x: 2*x*y - 3*x*z}
    print_system(sys3)
    rxns3 = buildPP(sys3, x)
    print_reactions(rxns3)
    odes3 = pp_reactions_to_odes(rxns3)
    print_odes(odes3)

    assert len(rxns3) == 2,                              "Test 3: expected 2 reactions"
    assert sp.expand(odes3[X] - ( 2*X*Y - 3*X*Z)) == 0, "Test 3: d[X]/dt mismatch"
    assert sp.expand(odes3[Y] - (-2*X*Y        )) == 0,  "Test 3: d[Y]/dt mismatch"
    assert sp.expand(odes3[Z] - ( 3*X*Z        )) == 0,  "Test 3: d[Z]/dt mismatch"
    assert_round_trip(odes3, sys3,                       "Test 3")
    print("  ✓ passed")

    # ------------------------------------------------------------------
    # Test 4: full system round-trip — rock-paper-scissors
    #   True CRN: X+Y->X+X (1),  Y+Z->Y+Y (1),  Z+X->Z+Z (1)
    #   d[x]/dt =  x*y - x*z
    #   d[y]/dt = -x*y + y*z
    #   d[z]/dt = -y*z + x*z
    #   Passing all 3 equations causes each reaction to be generated
    #   twice; pp_reactions_to_odes recovers 2× the original system.
    # ------------------------------------------------------------------
    print_separator("Test 4: full system — rock-paper-scissors")
    sys4 = {
        x:  x*y - x*z,
        y: -x*y + y*z,
        z: -y*z + x*z,
    }
    print_system(sys4)
    rxns4 = buildPP(sys4, x)
    print_reactions(rxns4)
    odes4 = pp_reactions_to_odes(rxns4)
    print_odes(odes4, title="Recovered ODEs (expect 2× original)")

    assert len(rxns4) == 3,                              "Test 4: expected 3 reactions"
    assert sp.expand(odes4[X] - (X*Y - X*Z)) == 0,      "Test 4: d[x]/dt mismatch"
    assert sp.expand(odes4[Y] - (-X*Y + Y*Z)) == 0,     "Test 4: d[y]/dt mismatch"
    assert sp.expand(odes4[Z] - (-Y*Z + X*Z)) == 0,     "Test 4: d[z]/dt mismatch"
    assert_round_trip(odes4, sys4,                       "Test 4")
    print("  ✓ passed")

    print("\n" + "═" * 60)
    print("  All tests passed!")
    print("═" * 60)

test_build_PP()