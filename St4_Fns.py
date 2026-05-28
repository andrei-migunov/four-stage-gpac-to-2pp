"""
St4_Fns.py
Claude 5/28
==========
Stage 4 of the Huang--Huls compiler.

Converts a Stage-3 PP-implementable quadratic-form ODE system into a
Probabilistic Large-Population Protocol (PLPP) via "Construction 1" from

    Xiang Huang and Rachel N. Huls,
    "Computing Real Numbers with Large-Population Protocols Having a
     Continuum of Equilibria," DNA 28 (2022), Section 3.6.

The construction relies on three properties of the Stage-3 input
(Corollary 3 in the paper):

    (i)   each x'_i = p_i - q_i * x_i with p_i, q_i in P^+
          (rational positive coefficients);
    (ii)  no positive x_i^2 term appears in x'_i;
    (iii) x' is a homogeneous quadratic form;
    (iv)  the system is conservative, i.e. sum_i x'_i = 0
          (equivalently sum_l x_l = 1 is an invariant: "the one trick").

THE MATHEMATICS (terse; see paper for full derivation)
------------------------------------------------------
For each species i we form the *production polynomial*

       f_i(x) = epsilon * x'_i  +  2 * x_i * sum_l x_l                    [Eq. (10)]

so that x'_i = f_i - 2 x_i (Observation 8) up to the epsilon-rescaling.  We
*never* substitute sum_l x_l = 1 here; f_i must remain a degree-2 polynomial
so Construction 1 can read off coefficients of x_j * x_k.

For every monomial alpha_{i,j,k} * x_j * x_k in f_i (where
alpha_{i,j,k} = C(x_j x_k, f_i) >= 0) we emit reaction rules:

    j == k :  (x_j, x_j) -> (x_i, x_i)  with probability alpha_{i,j,j}/2
    j != k :  (x_j, x_k) -> (x_i, x_i)  AND
              (x_k, x_j) -> (x_i, x_i)  each with probability alpha_{i,j,k}/4

WHY THE PROBABILITIES SUM TO 1 EXACTLY (the crucial conservation argument)
-------------------------------------------------------------------------
For an ordered pair (j, k) with j != k, the total assigned mass is

    Sum_i C(x_j x_k, f_i) / 4

By linearity:

    Sum_i C(x_j x_k, f_i)
      = epsilon * C(x_j x_k, Sum_i x'_i)                    (Part 1)
      + Sum_i C(x_j x_k, 2 x_i * Sum_l x_l)                 (Part 2)
      = epsilon * C(x_j x_k, 0)                             (conservation)
      + 2 [from i=j, l=k]  +  2 [from i=k, l=j]
      = 0 + 4 = 4

so the mass is 4/4 = 1 exactly.  Similarly, for j == k:

    Sum_i C(x_j^2, f_i)
      = epsilon * C(x_j^2, Sum_i x'_i) + Sum_i C(x_j^2, 2 x_i Sum_l x_l)
      = 0 + 2 [only i=j, l=j]
      = 2

so mass = 2/2 = 1.  Conclusion: for a valid conservative input the idle
reaction has probability exactly 0; we keep the idle-padding code only as
a defensive safeguard.

WHY epsilon IS BOUNDED
----------------------
For monomial x_i x_m of f_i the only negative contribution comes through
-epsilon * q_i * x_i (since p_i contributes positively and 2 x_i Sum_l x_l
contributes +2 to every x_i x_m).  Hence

    C(x_i x_m, f_i) = 2 + epsilon * (C(x_i x_m, p_i) - C(x_m, q_i))
                    >= 2 - epsilon * q_max

where q_max := max coefficient appearing in any q_i (equivalently, the
maximum magnitude of any negative coefficient anywhere in the system).
Any 0 < epsilon <= 2/q_max preserves f_i in P^+.  Default below is 1/q_max
which gives every coefficient a margin of at least 1.

WHY no-positive-x_i^2 MATTERS
-----------------------------
The probability of (x_i, x_i) -> (x_i, x_i) from the x_i^2 monomial of f_i is

    C(x_i^2, f_i) / 2  =  (2 + epsilon * C(x_i^2, x'_i)) / 2
                       =  1 + epsilon * C(x_i^2, x'_i) / 2

For this to be <= 1 we need C(x_i^2, x'_i) <= 0 -- i.e. PP-implementability
condition (ii).  We enforce this check up front.

OUTPUT FORMAT
-------------
    lpp = {
        (a, b): { (c, d): probability, ... },   # ordered reactant pair ->
        (b, a): { ... },                        #   distribution over product
        ...                                     #   pairs.  Probabilities are
    }                                           #   sympy Rationals summing to 1.

Species names are uppercased strings (matching buildPP).  lpp_iv uses the
same name convention.

VERIFICATION
------------
The helper `lpp_to_odes(lpp)` reconstructs the balance-equation ODE system
from the PLPP.  By the algebra above, for every species r:

    lpp_to_odes(lpp)[Symbol(name_r)]   ==   epsilon * stage3_system[symbol_r]

(equality as quadratic polynomials, with NO Sum_l x_l = 1 substitution
required, because the symbolic consumption -2 x_r * Sum_l x_l from the
balance equation cancels exactly against the +2 x_r * Sum_l x_l in f_r).

This is the round-trip check to use on large systems where reasoning by
hand is infeasible.
"""

from collections import defaultdict
import sympy as sp
from sympy import Symbol, Rational, Integer, expand


# =============================================================================
# Naming convention.  Species names are uppercased strings, matching buildPP,
# so the two PP representations (deterministic pp_reactions vs probabilistic
# lpp) refer to the same entities by the same names.
# =============================================================================

def _name(sym):
    """Canonical species name (uppercase string)."""
    return str(sym).upper()


# =============================================================================
# Parsing one degree-2 monomial out of a SymPy expression.
# =============================================================================

def _parse_monomial(mono_expr):
    """
    Convert a SymPy monomial (a key from as_coefficients_dict()) into a
    sorted 2-tuple of uppercased species names.

    Raises ValueError unless the monomial is homogeneous of degree EXACTLY 2,
    which the Stage-3 invariant guarantees.  This check is structural --
    the algorithm cannot proceed on non-degree-2 input -- so it always runs.
    """
    if mono_expr == 1:
        raise ValueError(
            "Encountered a constant (degree-0) term; Stage-3 system must be "
            "a homogeneous quadratic form."
        )
    factors = []
    for base, exp in mono_expr.as_powers_dict().items():
        if not getattr(base, "is_Symbol", False):
            raise ValueError(
                f"Unexpected non-symbol factor {base!r} in monomial {mono_expr}."
            )
        if not (exp.is_Integer and exp > 0):
            raise ValueError(
                f"Bad exponent {exp} on {base} in monomial {mono_expr}."
            )
        factors.extend([_name(base)] * int(exp))
    if len(factors) != 2:
        raise ValueError(
            f"Monomial {mono_expr} has total degree {len(factors)}; "
            f"Stage 4 requires degree exactly 2."
        )
    return tuple(sorted(factors))


# =============================================================================
# Per-equation coefficient extraction and structural validation.
# Always runs (the structural checks are cheap and embedded in the
# iteration we need anyway for coefficient extraction).
# =============================================================================

def _equation_coeffs(s_sym, s_name, rhs_expr):
    """
    Convert d[s]/dt into a {sorted-monomial-tuple: sympy Rational} dict.

    Side checks performed inline (always; cheap):
      - Each monomial is homogeneous degree 2 (via _parse_monomial).
      - Every coefficient is exactly rational (Stage 4 needs exact rationals
        to produce exact PLPP probabilities; if you have floats, scale them
        upstream).
      - Every negative-coefficient monomial has s_name as a factor.  This is
        the CRN-implementability requirement (Theorem 2 of the paper): a
        non-reactant species cannot be destroyed.
      - No positive s_name^2 term.  This is PP-implementability condition
        (ii) (Corollary 3) -- without it the self-reaction probability would
        exceed 1.

    Returns (coeffs, max_negative_magnitude).
    """
    coeffs = {}
    rhs_expanded = expand(rhs_expr)
    if rhs_expanded != 0:
        for mono, c in rhs_expanded.as_coefficients_dict().items():
            if c == 0:
                continue
            if not c.is_Rational:
                raise ValueError(
                    f"Coefficient {c} in d[{s_sym}]/dt is non-rational "
                    f"({type(c).__name__}).  Stage 4 needs exact rational "
                    f"coefficients to emit exact PLPP probabilities; rationalize "
                    f"upstream before calling make_lpp."
                )
            factors = _parse_monomial(mono)
            coeffs[factors] = coeffs.get(factors, Integer(0)) + c

    max_neg = Integer(0)
    for factors, c in coeffs.items():
        if c < 0:
            if s_name not in factors:
                raise ValueError(
                    f"d[{s_sym}]/dt has a negative term ({c}) * "
                    f"{factors[0]} * {factors[1]} that does NOT contain "
                    f"{s_name}; system is not CRN-implementable "
                    f"(Theorem 2: a non-reactant cannot be destroyed)."
                )
            if -c > max_neg:
                max_neg = -c
        elif c > 0 and factors == (s_name, s_name):
            raise ValueError(
                f"d[{s_sym}]/dt has a positive {s_name}^2 term "
                f"(coefficient {c}); system is not PP-implementable "
                f"(Corollary 3, condition ii).  Construction 1 cannot "
                f"produce valid probabilities here."
            )
    return coeffs, max_neg


# =============================================================================
# Main entry point.
# =============================================================================

def make_lpp(system, iv, mainvar, eps=None, checks=True, verbose=False):
    """
    Build a PLPP from a Stage-3 PP-implementable quadratic-form ODE system.

    Parameters
    ----------
    system  : dict { sympy.Symbol -> sympy expression }
        Stage-3 PP-implementable system.  Each value must be a homogeneous
        degree-2 polynomial in the system keys with rational coefficients.
    iv      : dict { sympy.Symbol -> rational }
        Initial proportions for the PLPP states.  Should sum to 1.
    mainvar : sympy.Symbol
        Species tracking the computed real number.  Accepted for signature
        compatibility and sanity-checked but not used in the construction.
    eps     : Optional[rational]
        The epsilon of the epsilon-trick.  Default: 1 / q_max where q_max is
        the largest magnitude of any negative coefficient in the system.
        Any 0 < eps <= 2/q_max is admissible; smaller eps means slower
        convergence of the LPP but does not change the limiting value.
    checks  : bool
        Run the (potentially expensive) global conservation check across all
        equations and the IV-sum soft check.  Default True.
    verbose : bool
        Print diagnostics (epsilon used, number of rules, etc.).

    Returns
    -------
    (lpp, lpp_iv) where
      lpp     : dict { (a_name, b_name) : { (c_name, d_name) : Rational, ... } }
                Ordered reactant pair -> distribution over product pairs.
                Probabilities sum to exactly 1 for each ordered key.
      lpp_iv  : dict { species_name (str) : initial proportion }

    Notes
    -----
    Use lpp_to_odes(lpp) to verify the result.  It should reproduce
    eps * system[r] for each species r (equality of polynomials).
    """

    # ------------------------------------------------------------------------
    # Step 0: Species set + name map.  We pin the species list to the system
    # KEYS (not free_symbols of expressions): the one-trick is only valid when
    # summing over the full state set.  If an expression mentions a symbol not
    # in keys, that's a malformed system; we raise.
    # ------------------------------------------------------------------------
    species_syms = list(system.keys())
    species_names = [_name(s) for s in species_syms]
    if len(set(species_names)) != len(species_names):
        # E.g., both 'x' and 'X' present -- uppercasing collides them.
        raise ValueError(
            "Species names collide after uppercasing.  Rename species so that "
            "str(sym).upper() is unique."
        )
    sym_to_name = dict(zip(species_syms, species_names))

    if checks:
        unknown = set()
        for expr in system.values():
            unknown |= expr.free_symbols
        unknown -= set(species_syms)
        if unknown:
            raise ValueError(
                f"Expressions reference symbols not present as system keys: "
                f"{sorted(map(str, unknown))}.  Every variable must have its own "
                f"ODE entry for the one-trick (sum_l x_l = 1) to be valid."
            )
        if mainvar is not None and mainvar not in species_syms:
            raise ValueError(
                f"mainvar = {mainvar!r} is not present in system.keys(); the "
                f"PLPP cannot track an undefined species."
            )

    # ------------------------------------------------------------------------
    # Step 1: per-equation coefficient extraction + structural validation.
    # This also tracks q_max for the default-epsilon choice.
    # ------------------------------------------------------------------------
    coeff_dicts = {}
    q_max = Integer(0)
    for s in species_syms:
        s_name = sym_to_name[s]
        cd, eqn_q_max = _equation_coeffs(s, s_name, system[s])
        coeff_dicts[s] = cd
        if eqn_q_max > q_max:
            q_max = eqn_q_max

    # ------------------------------------------------------------------------
    # Step 2: global conservation check.  Done by accumulating coefficients
    # across all equations into one big monomial->coeff dict and checking it
    # is identically zero.  This is O(total terms in system), much cheaper
    # than expand(sum(system.values())) which would symbolically reconstruct
    # the whole expression.
    #
    # If the input is not conservative, the Construction 1 probability sums
    # WILL NOT equal 1 per ordered pair (the conservation argument fails),
    # and the resulting PLPP is invalid.  Worth catching here loudly.
    # ------------------------------------------------------------------------
    if checks:
        global_sum = defaultdict(lambda: Integer(0))
        for s in species_syms:
            for factors, c in coeff_dicts[s].items():
                global_sum[factors] += c
        offenders = {m: c for m, c in global_sum.items() if c != 0}
        if offenders:
            sample = list(offenders.items())[:5]
            raise ValueError(
                f"System is not conservative: {len(offenders)} monomial(s) "
                f"have nonzero net coefficient across the system.  Sample: {sample}"
            )

    # ------------------------------------------------------------------------
    # Step 3: choose epsilon.
    # ------------------------------------------------------------------------
    if eps is None:
        # Safe default: 1/q_max.  Tightest coefficient of f_i becomes
        # 2 - q_max * (1/q_max) = 1, comfortably positive.
        eps = Rational(1) if q_max == 0 else Rational(1) / q_max
    else:
        eps = sp.sympify(eps)
        if not eps.is_Rational:
            raise ValueError(f"eps must be rational; got {eps!r}.")
    if eps <= 0:
        raise ValueError(f"eps must be positive; got {eps}.")
    if q_max > 0 and eps > Rational(2) / q_max:
        # Strict upper bound: 2/q_max gives the tightest f_i coefficient
        # equal to 0; anything larger drives it negative.
        raise ValueError(
            f"eps = {eps} exceeds the admissible maximum 2/q_max = "
            f"{Rational(2)/q_max}.  f_i would have negative coefficients."
        )

    if verbose:
        print(f"[make_lpp] species: {len(species_syms)}, "
              f"q_max = {q_max}, epsilon = {eps}")

    # ------------------------------------------------------------------------
    # Step 4: build the rule set.
    #
    # lpp[(a, b)][(c, d)] = probability that ordered pair (a, b) produces (c, d).
    #
    # We iterate species i, build f_i's coefficient dict, then emit rules
    # for each monomial of f_i.  Products are always (x_i, x_i) ("all-in
    # greedy" strategy from the paper).
    # ------------------------------------------------------------------------
    lpp = defaultdict(lambda: defaultdict(lambda: Integer(0)))

    for i_sym in species_syms:
        i_name = sym_to_name[i_sym]
        prod_pair = (i_name, i_name)

        # Build f_i as a {sorted-tuple-monomial: rational coeff} dict:
        #   Part 1: epsilon * x'_i
        #   Part 2: 2 * x_i * sum_l x_l = sum_l 2 * x_i * x_l
        fi = defaultdict(lambda: Integer(0))
        for factors, c in coeff_dicts[i_sym].items():
            fi[factors] += eps * c
        for l_name in species_names:
            key = tuple(sorted((i_name, l_name)))
            fi[key] += Integer(2)

        # Emit reactions from each monomial of f_i.
        for (a, b), alpha in fi.items():
            if alpha == 0:
                continue
            if alpha < 0:
                # By the non-negativity argument this is impossible for
                # eps <= 2/q_max on a valid PP-implementable system.  If
                # we hit it, either eps was bumped up by the caller past
                # the admissible range or the input violated assumptions.
                raise ValueError(
                    f"f_[{i_name}] has negative coefficient {alpha} on "
                    f"{a} * {b}.  Either eps = {eps} is too large or the "
                    f"input is not PP-implementable."
                )
            if a == b:
                # Monomial x_a^2 -> ordered pair (a, a), single rule, /2.
                lpp[(a, a)][prod_pair] += alpha / Integer(2)
            else:
                # Monomial x_a x_b -> rules for BOTH ordered pairs, each /4.
                p = alpha / Integer(4)
                lpp[(a, b)][prod_pair] += p
                lpp[(b, a)][prod_pair] += p

    # ------------------------------------------------------------------------
    # Step 5: per-ordered-pair normalization + defensive idle padding.
    # In exact arithmetic on a valid conservative input every total is
    # exactly 1, but we still:
    #   - drop any incidental zero entries (can arise if user-supplied
    #     eps exactly cancels a coefficient);
    #   - raise on total > 1 (would indicate broken conservation);
    #   - pad with an idle reaction (a, b) -> (a, b) if total < 1.
    # ------------------------------------------------------------------------
    lpp_out = {}
    for pair, dist in lpp.items():
        d = {p_pair: pr for p_pair, pr in dist.items() if pr != 0}
        total = sum(d.values(), Integer(0))
        if total > 1:
            raise ValueError(
                f"Ordered pair {pair} has total assigned probability {total} "
                f"> 1.  Indicates either non-conservative input or that the "
                f"per-monomial coefficient sum exceeded the expected value of "
                f"4 (or 2 for self-pairs)."
            )
        if total < 1:
            # Idle absorbs the deficit.  Should not fire for valid input.
            d[pair] = d.get(pair, Integer(0)) + (Integer(1) - total)
        lpp_out[pair] = d

    # ------------------------------------------------------------------------
    # Step 6: initial values -- rename keys to uppercased species names.
    # ------------------------------------------------------------------------
    lpp_iv = {}
    for s, v in iv.items():
        key = _name(s) if isinstance(s, sp.Basic) else str(s).upper()
        lpp_iv[key] = v

    if checks:
        # Soft check: a PLPP's state proportions should sum to 1.
        try:
            iv_sum = sum(sp.sympify(v) for v in lpp_iv.values())
            if iv_sum != 1 and verbose:
                print(f"[make_lpp] NOTE: initial proportions sum to {iv_sum}, "
                      f"not 1.  PLPP semantics expect a probability distribution.")
        except Exception:
            # Non-numeric IV values -- silently skip.
            pass

    if verbose:
        n_pairs = len(lpp_out)
        n_entries = sum(len(d) for d in lpp_out.values())
        print(f"[make_lpp] produced {n_pairs} ordered-pair rules across "
              f"{n_entries} distribution entries.")

    return lpp_out, lpp_iv


# =============================================================================
# Verification & interop helpers.
# =============================================================================

def lpp_to_odes(lpp):
    """
    Reconstruct the balance-equation ODE system from a PLPP, per Equation (5)
    of the paper.

    Returns a dict { sympy.Symbol -> sympy expression } keyed by Symbol(name)
    for each species name in the PLPP.  By construction:

        lpp_to_odes(lpp)[Symbol(name_r)]   ==   epsilon * stage3_system[symbol_r]

    as quadratic POLYNOMIALS (no sum_l x_l = 1 substitution): the symbolic
    consumption -2 x_r * sum_l x_l from the balance equation cancels exactly
    against the +2 x_r * sum_l x_l inside f_r, leaving epsilon * x'_r.

    This is the round-trip check.  For a verification on a large system:

        eps = ...                                # the eps make_lpp used
        odes_back = lpp_to_odes(lpp)
        for sym, expr in stage3_system.items():
            recovered = odes_back[Symbol(str(sym).upper())]
            assert expand(recovered - eps * expr) == 0, sym
    """
    dvars = defaultdict(lambda: Integer(0))
    for (a_name, b_name), dist in lpp.items():
        a = Symbol(a_name)
        b = Symbol(b_name)
        mono = a * b
        # Consumption: every ordered-pair interaction destroys one reactant
        # of each kind.  This contributes -e_a - e_b to the change vector
        # times x_a x_b -- the - sum_{(i,j)} x_i x_j (e_i + e_j) part of
        # Equation (5).
        dvars[a] = dvars[a] - mono
        dvars[b] = dvars[b] - mono
        # Production: each product (c, d) with probability p restores p
        # molecules of c and p molecules of d per interaction.
        for (c_name, d_name), p in dist.items():
            c = Symbol(c_name)
            d = Symbol(d_name)
            dvars[c] = dvars[c] + p * mono
            dvars[d] = dvars[d] + p * mono
    return {k: expand(v) for k, v in dvars.items()}


def lpp_to_reaction_list(lpp):
    """
    Flatten the PLPP into a list of (probability, [a, b], [c, d]) tuples,
    matching the (rate, lhs, rhs) shape used by buildPP's output.

    Useful for piping to `to_PP_reactions_str` for readable display.
    Note: this preserves the asymmetric (a, b) vs (b, a) distinction by
    emitting both as separate entries.  If you feed this to
    `to_ppsim_format`, be aware that helper sorts and SUMS by (sorted lhs,
    sorted rhs), which would double-count the symmetric pairs -- use the
    dict form directly with ppsim instead.
    """
    out = []
    for (a, b), dist in lpp.items():
        for (c, d), p in dist.items():
            out.append((p, [a, b], [c, d]))
    return out


def compute_default_eps(system):
    """
    Return the epsilon that make_lpp(system, ...) would pick by default.

    Useful for verification: callers who pass the result of make_lpp to
    lpp_to_odes can recompute the epsilon to compare against
    epsilon * stage3_system.
    """
    q_max = Integer(0)
    for s, expr in system.items():
        s_name = _name(s)
        _, eqn_q_max = _equation_coeffs(s, s_name, expr)
        if eqn_q_max > q_max:
            q_max = eqn_q_max
    return Rational(1) if q_max == 0 else Rational(1) / q_max