import sympy as sp
from collections import defaultdict
 
 
def buildPP(sys, mainvar):
    '''   Convert a degree-2 ODE system into bimolecular reactions A + B -> C + D
    by matching and transcribing monomials.
  
 
    Returns a list of (rate, lhs, rhs) tuples.  rate is a SymPy number (hopefully rates are rational), lhs and rhs are 2-element lists of uppercase species name strings.
    '''
 
    # Build a table of coeffiients
    coeffs = defaultdict(lambda: defaultdict(lambda: sp.Integer(0)))
 
    for var, expr in sys.items():
        expr = sp.expand(expr)
        if expr == 0:
            continue
        terms = list(expr.args) if isinstance(expr, sp.Add) else [expr]
 
        for term in terms:
            scalar = sp.Integer(1)
            factors = []
            args = term.args if isinstance(term, sp.Mul) else [term]
            for arg in args:
                if arg.is_Number:
                    scalar *= arg
                elif isinstance(arg, sp.Pow):
                    base, exp = arg.args
                    if not exp.is_Integer or exp < 1:
                        raise ValueError(f"Bad exponent in term {term}")
                    factors.extend([str(base).upper()] * int(exp))
                elif isinstance(arg, sp.Symbol):
                    factors.append(str(arg).upper())
                else:
                    raise ValueError(f"Unexpected factor in term {term}")
 
            if len(factors) != 2:
                raise ValueError(
                    f"Term {term} in d[{var}]/dt has {len(factors)} factors; "
                    f"expected exactly 2."
                )
 
            monomial = tuple(sorted(factors))
            coeffs[monomial][str(var).upper()] += scalar
 
    
    reactions = []
 
    for monomial, c_dict in coeffs.items():
        c = {z: v for z, v in c_dict.items() if v != 0}
        if not c:
            continue
 
        # Check mass conservation
        total = sum(c.values())
        if total != 0:
            raise ValueError(
                f"Monomial {monomial[0]}*{monomial[1]}: column sums to "
                f"{total}, not 0.  Not implementable as 2-in-2-out reactions "
                f"with reactants {{{monomial[0]}, {monomial[1]}}}.  c = {dict(c)}"
            )
 
        A, B = monomial
        same = (A == B)
 
        if same:
            T = max(-c.get(A, sp.Integer(0)) / 2, sp.Integer(0))
        else:
            T = max(-c.get(A, sp.Integer(0)),
                    -c.get(B, sp.Integer(0)),
                    sp.Integer(0))
 
        if T == 0:
            continue
 
        prods = dict(c)
        if same:
            prods[A] = prods.get(A, sp.Integer(0)) + 2 * T
        else:
            prods[A] = prods.get(A, sp.Integer(0)) + T
            prods[B] = prods.get(B, sp.Integer(0)) + T
        prods = {z: v for z, v in prods.items() if v != 0}
 
        for z, v in prods.items():
            if v < 0:
                raise ValueError(
                    f"Monomial {monomial[0]}*{monomial[1]}: non-reactant "
                    f"species {z} has negative coefficient {v}.  "
                    f"Impossible for 2-in-2-out reactions."
                )
 
        # Perform matching between ODE terms.
        lhs = list(monomial)   # [A, B] or [X, X]
        while prods:
            # Highest production first, ties broken by name 
            ordered = sorted(prods.items(), key=lambda kv: (-kv[1], kv[0]))
 
            if len(ordered) == 1:
                Z, p_Z = ordered[0]
                rate = p_Z / 2
                reactions.append((rate, lhs, [Z, Z]))
                del prods[Z]
            else:
                (Z1, p1), (Z2, p2) = ordered[0], ordered[1]
                rate = sp.Min(p1, p2) if isinstance(p1, sp.Expr) else min(p1, p2)
                reactions.append((rate, lhs, sorted([Z1, Z2])))
                prods[Z1] -= rate
                prods[Z2] -= rate
                if prods[Z1] == 0:
                    del prods[Z1]
                if Z2 in prods and prods[Z2] == 0:
                    del prods[Z2]
 
    return reactions
 




# """
# Convert a degree-2 ODE system to bimolecular reactions A + B -> C + D
# consistent with mass-action kinetics.

# sys     : dict mapping variable to ODE expression (SymPy)
# mainvar : variable we care about, though it has no role here yet
# Returns a list of (coeff, lhs, rhs) where coeff is a positive rate constant, lhs and rhs are lists of exactly two uppercase species-name strings.
# """
# def buildPP_old(sys, mainvar):
#     def extract_terms(expr):
#         expr = sp.expand(expr)
#         return list(expr.args) if isinstance(expr, sp.Add) else [expr]

#     def decompose_term(term):
#         term = sp.expand(term)
#         if term.is_Number:
#             return abs(term), []
#         coeff = 1
#         factors = []
#         args = term.args if isinstance(term, sp.Mul) else [term]
#         for arg in args:
#             if arg.is_Number:
#                 coeff *= arg
#             elif isinstance(arg, sp.Pow):
#                 base, exp = arg.args
#                 if not exp.is_Integer or exp < 1:
#                     raise ValueError(f"Non-integer or negative exponent in term: {term}")
#                 factors.extend([str(base).upper()] * int(exp))
#             elif isinstance(arg, sp.Symbol):
#                 factors.append(str(arg).upper())
#             else:
#                 raise ValueError(f"Unexpected factor in term: {arg}")
#         return abs(coeff), factors

#     reactions = []

#     for var, expr in sys.items():
#         var_str = str(var).upper()
#         for term in extract_terms(expr):
#             coeff, factors = decompose_term(term)
#             if len(factors) != 2:
#                 raise ValueError(
#                     f"Monomial '{term}' in equation for '{var}' is not degree-2 "
#                     f"(found {len(factors)} symbolic factors)."
#                 )
#             A, B = factors
#             lhs = [A, B]

#             if term.could_extract_minus_sign():
#                 if var_str not in factors:
#                     raise ValueError(
#                         f"Negative monomial '{term}' in equation for '{var}' does not "
#                         f"contain '{var_str}'."
#                     )
#                 C = A if B == var_str else B
#                 rhs = [C, C]
#             else:
#                 if var_str in factors:
#                     C = A if B == var_str else B
#                     if C == var_str:
#                         raise ValueError(
#                             f"Positive monomial '{term}' is purely '{var_str}^2', which "
#                             f"cannot be represented as a 2-in-2-out reaction for '{var_str}'."
#                         )
#                     rhs = [var_str, var_str]
#                 else:
#                     rhs = [A, var_str]

#             reactions.append((coeff, lhs, rhs))

#     # Deduplicate: A+B->C+D and B+A->D+C are the same reaction
#     seen = set()
#     unique = []
#     for coeff, lhs, rhs in reactions:
#         key = (round(float(coeff), 10), tuple(sorted(lhs)), tuple(sorted(rhs)))
#         if key not in seen:
#             seen.add(key)
#             unique.append((coeff, lhs, rhs))
#     return unique


    # groups     = defaultdict(list) 
    # canonical  = {}

    # for coeff, lhs, rhs in reactions:
    #     key = (tuple(sorted(lhs)), tuple(sorted(rhs)))
    #     groups[key].append(float(coeff))
    #     if key not in canonical:
    #         canonical[key] = (lhs, rhs)

    # unique = []
    # for key, rates in groups.items():
    #     avg_rate = sum(rates) / len(rates)
    #     lhs, rhs = canonical[key]
    #     unique.append((avg_rate, lhs, rhs))

    # return unique