from St0_Fns import *
from St1_Fns import *
from sympy import *
from sympy.utilities.lambdify import *
import sympy as sp
import numpy as np
import scipy.integrate as intg
import ast  # Used to evaluate lists in string format for variable names
import re   # Used to quickly drop all non-numeric information when evaluating variable names for indexing


# =============================================================================
# Stage 3 settings
# =============================================================================

STAGE3_CHOP_TOL = 1e-10

'''
# Creates the (x_ix_j)' combinations
# * Input:
#   sys1, sys2:
#       - Data Type: Dict of Sympy Symbol/Equations
#       - Desc: The systems that are being used in the self product. Should be in a format similar to: {x0:x0+4x1x0+x1,...}
#   var1, var2:
#       - Data Type: Sympy Symbol
#       - Desc: The variables from the system being processed at the moment
# * Output:
#   "expand(simplify(sys1[var1] * var2 + var1 * sys2[var2]))":
#       - Data Type: sympy equation
#       - Desc: The equation resulting from performing the product of the system for the current variable
'''
def get_z_derivative_full_prod(sys1, var1, sys2, var2):
    return expand(sys1[var1] * var2 + var1 * sys2[var2])


'''
# Creates the half prod of the z derivatives
# * Input:
#       - sys:
#         - Data Type: Dict of Sympy Equations/Symbols
#         - Desc: The starting ODE system we wish to take the self-product of. Should be passed in a form similiar to: {x0:x0+4x1x0+x1,...}
# * Output:
#       - z_system:
#         - Data Type: Dict of Sympy Symbols:Equations
#         - Desc: The result of taking the product on sys1, sys2. Will be in a form similar to: {z_[0,0]:x0+4x1x0+x1,...}}
#       - z_var_map:
#         - Data Type: Dict of Sympy Symbols:Equations
#         - Desc: A dictionary of valid variable substitutions that can be made on the system to later attempt to
#                 convert it into a PP-implementable quadratic form system.
'''
def half_prod(sys1):
    # Store the z variables
    z_system = {}
    # Creates the mappings for the z_vars to be utilized in substitution later
    z_var_map = {}

    total_vars = len(sys1)
    total_pairs = total_vars * (total_vars + 1) // 2
    print(f"half_prod: computing {total_pairs} upper-triangular z entries from {total_vars} variables...")

    # Loops to make all of the pairs
    m = 0
    pair_index = 0
    for var1 in sys1.keys():
        n = 0
        for var2 in sys1.keys():
            # Checks for lower half of matrix
            if m <= n:
                pair_index += 1
                if pair_index % 20 == 0 or pair_index == 1 or pair_index == total_pairs:
                    print(f"  half_prod progress: {pair_index}/{total_pairs} (var1={var1}, var2={var2})")

                i = sym_idx_parser(var1)
                j = sym_idx_parser(var2)

                # if, wlog, i is an integer and j is a list, append i to j
                if (isinstance(i, list) and isinstance(j, int)) or (isinstance(i, int) and isinstance(j, list)):

                    # Create the index for a z_[i,j,k] variable
                    if isinstance(i, list):
                        i_and_j = i + [j]
                    else:
                        i_and_j = [i] + j

                    z_system[Symbol(f"z_{i_and_j}")] = get_z_derivative_half_prod(sys1, var1, var2, i, j)

                    if i == j:
                        z_var_map[Symbol(f"z_{i_and_j}")] = var1**2
                    else:
                        z_var_map[Symbol(f"z_{i_and_j}")] = 2 * var1 * var2

                else:
                    # Adds new z variables to system
                    z_system[Symbol(f"z_[{i},{j}]")] = get_z_derivative_half_prod(sys1, var1, var2, i, j)

                    if i == j:
                        z_var_map[Symbol(f"z_[{i},{j}]")] = var1**2
                    else:
                        z_var_map[Symbol(f"z_[{i},{j}]")] = 2 * var1 * var2

            n += 1
        m += 1

    return z_system, z_var_map


'''
# Creates the (x_ix_j)' combinations efficiently
# Assumes we're only writing a z_[0,1] and never z_[1,0]
# * Input:
#   sys1:
#       - Data Type: Dict of Sympy Symbol/Equations
#       - Desc: The system that's being used in the self product. Should be in a format similar to: {x0:x0+4x1x0+x1,...}
#   var1, var2:
#       - Data Type: Sympy Symbol
#       - Desc: The variables from the system being processed at the moment
#   i,j:
#       - Data Type: Integer
#       - Desc: The indexes of the var1, var2 variables being processed at the moment.
# * Output:
#   "expand(simplify(2 * var1 * sys1[var1]))"/"expand(simplify(2 * sys1[var1] * var2 + 2 * var1 * sys1[var2]))":
#       - Data Type: sympy equation
#       - Desc: The equation resulting from performing the product of the system for the current variable
'''
def get_z_derivative_half_prod(sys1, var1, var2, i, j):
    # If variables are the same i.e X_0*X_0
    if i == j:
        return expand(2 * var1 * sys1[var1], expand_mul=True, expand_power_exp=True)

    # If variables are different i.e X_0*X_1
    else:
        return expand(2 * sys1[var1] * var2 + 2 * var1 * sys1[var2], expand_mul=True, expand_power_exp=True)


'''
# Creates the full prod of the z derivatives
# * Input:
#       - sys1, sys2:
#         - Data Type: Dict of Sympy Equations/Symbols
#         - Desc: The starting ODE systems we wish to take the product of. Should be passed in a form similiar to: {x0:x0+4x1x0+x1,...}
# * Output:
#       - z_system:
#         - Data Type: Dict of Sympy Symbols:Equations
#         - Desc: The result of taking the product on sys1, sys2. Will be in a form similar to: {z_[0,0]:x0+4x1x0+x1,...}}
#       - z_var_map:
#         - Data Type: Dict of Sympy Symbols:Equations
#         - Desc: A dictionary of valid variable substitutions that can be made on the system to later attempt to
#                 convert it into a PP-implementable quadratic form system.
'''
def full_prod(sys1, sys2):
    # Store the z variables
    z_system = {}
    # Store the mappings for converting x vars into z vars
    z_var_map = {}

    # Assume the variables in the systems all have proper indexing
    for var1 in sys1.keys():
        for var2 in sys2.keys():
            i = sym_idx_parser(var1)
            j = sym_idx_parser(var2)

            # if, wlog, i is an integer and j is a list, append i to j
            if (isinstance(i, list) and isinstance(j, int)) or (isinstance(i, int) and isinstance(j, list)):
                if isinstance(i, list):
                    i_and_j = i + [j]
                else:
                    i_and_j = [i] + j

                z_system[Symbol(f"z_{i_and_j}")] = get_z_derivative_full_prod(sys1, var1, sys2, var2)
                z_var_map[Symbol(f"z_{i_and_j}")] = simplify(var1 * var2)

            else:
                # Adds new z variables to system
                z_system[Symbol(f"z_[{i},{j}]")] = get_z_derivative_full_prod(sys1, var1, sys2, var2)
                z_var_map[Symbol(f"z_[{i},{j}]")] = simplify(var1 * var2)

    return z_system, z_var_map


'''
# Parses the i or i,j out of "x_i" or "x_i,j" variables
# * Input:
#   - raw_symbol:
#     - Data Type: Sympy symbol
#     - Desc: A sympy variable with the naming format of var_name## (ex: x0, x1) or var_name_## (ex: x_0,x_1)
# * Output:
#   - idx:
#     - Data Type: Integer
#     - Desc: The index part of the passed variable's name (ex: z8 -> idx = 8, f_77 -> idx = 77)
'''
def sym_idx_parser(raw_symbol):
    idx = 0

    # Ensure the raw_symbol is in string format
    raw_symbol = str(raw_symbol)

    # Variables are sometimes written as x0 and sometimes as x_0.
    if "_" in raw_symbol:
        numeric_part = raw_symbol.split("_")[1]
        idx = ast.literal_eval(numeric_part)

    else:
        if "[" in raw_symbol:
            numeric_part = raw_symbol.split("[")[1]
            numeric_part = "[" + numeric_part
            idx = ast.literal_eval(numeric_part)

        else:
            numeric_part = re.sub(r'[^0-9]', '', raw_symbol)
            idx = ast.literal_eval(numeric_part)

    return idx


# =============================================================================
# Stage 3 transfer-lift implementation
# =============================================================================

def _stage3_zeroish(expr, tol=STAGE3_CHOP_TOL):
    """
    Treats tiny float roundoff as zero during transfer construction.
    """
    expr = simplify(expr)

    if expr == 0:
        return True

    try:
        return abs(float(N(expr))) < tol
    except Exception:
        return False


def _stage3_coeff_sign(coeff, tol=1e-12):
    """
    Returns:
        -1 for negative
         1 for positive
         0 for zero/tiny
    """
    coeff = simplify(coeff)

    if coeff == 0:
        return 0

    try:
        val = float(N(coeff))

        if val < -tol:
            return -1
        if val > tol:
            return 1

        return 0

    except Exception:
        if coeff.is_negative:
            return -1
        if coeff.is_positive:
            return 1

        return 0


def _stage3_make_z_symbol(var1, var2, var_order=None):
    """
    Creates the same z-symbol names as half_prod.

    Uses upper-triangular order when var_order is provided.
    """
    if var_order is not None:
        order = {v: i for i, v in enumerate(var_order)}

        if order[var2] < order[var1]:
            var1, var2 = var2, var1

    i = sym_idx_parser(var1)
    j = sym_idx_parser(var2)

    if (isinstance(i, list) and isinstance(j, int)) or \
       (isinstance(i, int) and isinstance(j, list)):

        if isinstance(i, list):
            i_and_j = i + [j]
        else:
            i_and_j = [i] + j

        return Symbol(f"z_{i_and_j}")

    return Symbol(f"z_[{i},{j}]")


def _stage3_pair_scale(var1, var2):
    """
    Half-product convention:
        z_[i,i] = x_i^2       scale 1
        z_[i,j] = 2*x_i*x_j   scale 2
    """
    if var1 == var2:
        return Integer(1)

    return Integer(2)


def _stage3_old_factor_sort_key(var, var_order):
    order = {v: i for i, v in enumerate(var_order)}
    return order.get(var, 10**9)


def _stage3_term_coeff_and_old_factors(term, old_vars, var_order):
    """
    Splits a monomial term into:
        coefficient
        list of old variable factors

    Example:
        -3*x_0*x_1*x_1 -> (-3, [x_0, x_1, x_1])
    """
    coeff, factors = term.as_coeff_mul()

    old_factors = []
    leftovers = []

    for factor in factors:
        if factor in old_vars:
            old_factors.append(factor)

        elif isinstance(factor, Pow) and factor.base in old_vars and factor.exp.is_Integer:
            exp = int(factor.exp)

            if exp < 0:
                raise ValueError(f"Negative old-variable power is not supported: {factor}")

            old_factors.extend([factor.base] * exp)

        elif factor.is_number:
            coeff *= factor

        else:
            leftovers.append(factor)

    if leftovers:
        raise ValueError(
            f"Unexpected non-old-variable factor in Stage 3 input term.\n"
            f"Term: {term}\n"
            f"Leftover factors: {leftovers}"
        )

    old_factors = sorted(old_factors, key=lambda v: _stage3_old_factor_sort_key(v, var_order))

    return simplify(coeff), tuple(old_factors)


def _stage3_collect_cubic_monomial_coeffs(sys):
    """
    Groups the old cubic system by monomial.

    Returns:
        monomial_coeffs[M][x_i] = coefficient of monomial M in x_i'
    """
    var_order = list(sys.keys())
    old_vars = set(var_order)

    monomial_coeffs = {}

    for var, expr in sys.items():
        expr = expand(expr)

        for term in Add.make_args(expr):
            if term == 0:
                continue

            coeff, old_factors = _stage3_term_coeff_and_old_factors(term, old_vars, var_order)

            if _stage3_zeroish(coeff):
                continue

            if len(old_factors) != 3:
                raise ValueError(
                    f"Stage 3 transfer lift expects a homogeneous cubic system.\n"
                    f"Equation: {var}'\n"
                    f"Term: {term}\n"
                    f"Old factors: {old_factors}\n"
                    f"Degree found: {len(old_factors)}"
                )

            if old_factors not in monomial_coeffs:
                monomial_coeffs[old_factors] = {v: Integer(0) for v in var_order}

            monomial_coeffs[old_factors][var] += coeff

    # Clean tiny roundoff during coefficient collection.
    # This is used by the transfer construction itself.
    for monomial, coeffs in monomial_coeffs.items():
        for v in var_order:
            coeffs[v] = simplify(coeffs[v])

            if _stage3_zeroish(coeffs[v]):
                coeffs[v] = Integer(0)

        total = simplify(sum(coeffs.values()))

        # If total is tiny, remove the tiny symbolic/float residual exactly
        # by adjusting one nonzero coefficient.
        if _stage3_zeroish(total) and total != 0:
            for v in var_order:
                if coeffs[v] != 0:
                    coeffs[v] = simplify(coeffs[v] - total)
                    break

    return monomial_coeffs, var_order


def _stage3_remove_one_factor(factors, factor_to_remove):
    """
    Removes one copy of factor_to_remove from a tuple/list of old factors.
    """
    factors = list(factors)

    for i, f in enumerate(factors):
        if f == factor_to_remove:
            del factors[i]
            return tuple(factors)

    raise ValueError(
        f"Could not remove factor {factor_to_remove} from monomial factors {factors}."
    )


def _stage3_decompose_monomial_coeffs_into_transfers(monomial, coeffs, var_order):
    """
    For one cubic monomial M, decompose the coefficient vector into transfers.

    If:
        x_a' has -r*M
        x_b' has +r*M

    we create transfer:
        a -> b at rate r*M

    Returns list of tuples:
        (source_var, target_var, amount)
    """
    negatives = []
    positives = []

    for v in var_order:
        c = simplify(coeffs[v])
        sign = _stage3_coeff_sign(c)

        if sign < 0:
            # CRN/TPP form requires the consumed variable to occur in the monomial.
            if v not in monomial:
                raise ValueError(
                    f"Negative term does not contain consumed variable.\n"
                    f"Variable: {v}\n"
                    f"Monomial: {monomial}\n"
                    f"Coefficient: {c}"
                )

            negatives.append([v, simplify(-c)])

        elif sign > 0:
            positives.append([v, c])

    transfers = []

    i = 0
    j = 0

    while i < len(negatives) and j < len(positives):
        source, neg_amount = negatives[i]
        target, pos_amount = positives[j]

        neg_float = float(N(neg_amount))
        pos_float = float(N(pos_amount))

        if neg_float <= pos_float + 1e-12:
            amount = neg_amount
        else:
            amount = pos_amount

        if not _stage3_zeroish(amount):
            transfers.append((source, target, simplify(amount)))

        negatives[i][1] = simplify(neg_amount - amount)
        positives[j][1] = simplify(pos_amount - amount)

        if _stage3_zeroish(negatives[i][1]):
            i += 1

        if _stage3_zeroish(positives[j][1]):
            j += 1

    leftover_neg = sum(n[1] for n in negatives[i:])
    leftover_pos = sum(p[1] for p in positives[j:])

    if not _stage3_zeroish(leftover_neg) or not _stage3_zeroish(leftover_pos):
        raise ValueError(
            f"Could not fully decompose monomial into transfers.\n"
            f"Monomial: {monomial}\n"
            f"Leftover negative amount: {leftover_neg}\n"
            f"Leftover positive amount: {leftover_pos}"
        )

    return transfers


def _stage3_initialize_z_system(var_order):
    """
    Creates all half-product z variables initialized to zero derivative.
    """
    z_system = {}

    total_vars = len(var_order)
    total_pairs = total_vars * (total_vars + 1) // 2

    print(f"stage3 transfer lift: creating {total_pairs} z variables from {total_vars} old variables...")

    for m, var1 in enumerate(var_order):
        for n, var2 in enumerate(var_order):
            if m <= n:
                z = _stage3_make_z_symbol(var1, var2, var_order)
                z_system[z] = Integer(0)

    return z_system


def _stage3_add_lifted_transfer(z_system, var_order, monomial, source, target, amount):
    """
    Lift one old-variable transfer source -> target at rate amount*M
    into the half-product z-system.

    Old transfer:
        x_source' -= amount*M
        x_target' += amount*M

    For each partner j:
        z_[source,j]' -= interaction
        z_[target,j]' += interaction

    The interaction is chosen so that:
        z_[source,j] appears in the negative term,
        the same term is added to the target equation,
        conservation is preserved by construction.
    """

    remaining_after_source = _stage3_remove_one_factor(monomial, source)

    if len(remaining_after_source) != 2:
        raise ValueError(
            f"Expected two remaining factors after removing {source} from {monomial}, "
            f"got {remaining_after_source}."
        )

    r, s = remaining_after_source

    context_z = _stage3_make_z_symbol(r, s, var_order)
    context_scale = _stage3_pair_scale(r, s)

    for partner in var_order:
        source_z = _stage3_make_z_symbol(source, partner, var_order)
        target_z = _stage3_make_z_symbol(target, partner, var_order)

        source_scale = _stage3_pair_scale(source, partner)



        lifted_coeff = simplify(
            Rational(2, 1) * amount / (source_scale * context_scale)
        )

        interaction = expand(lifted_coeff * source_z * context_z)

        z_system[source_z] = expand(z_system[source_z] - interaction)
        z_system[target_z] = expand(z_system[target_z] + interaction)


def stage_three_transfer_lift(sys):
    """
    New Stage 3 implementation.

    Instead of expanding z' and then guessing substitutions, this function:
      1. Groups the old cubic system by monomial.
      2. Decomposes each monomial's coefficient vector into conservative transfers.
      3. Lifts each old-variable transfer into a z-variable transfer.

    """

    monomial_coeffs, var_order = _stage3_collect_cubic_monomial_coeffs(sys)

    z_system = _stage3_initialize_z_system(var_order)

    total_monomials = len(monomial_coeffs)

    print(f"stage3 transfer lift: processing {total_monomials} cubic monomials...")

    for idx, (monomial, coeffs) in enumerate(monomial_coeffs.items(), start=1):
        if idx == 1 or idx % 20 == 0 or idx == total_monomials:
            print(f"  stage3 transfer lift progress: monomial {idx}/{total_monomials}")

        transfers = _stage3_decompose_monomial_coeffs_into_transfers(
            monomial,
            coeffs,
            var_order  
        )

        for source, target, amount in transfers:
            _stage3_add_lifted_transfer(
                z_system,
                var_order,
                monomial,
                source,
                target,
                amount
            )

    print("[stage3] Final transfer-lift PP system")
    print(f"  equations:       {len(z_system)}")

    return z_system


'''
# Takes the product of an ODE using the self-product optimization of only needing to compute the upper triangular of possible combinations
# * Input:
#       - sys:
#         - Data Type: Dict of Sympy Equations/Symbols
#         - Desc: The starting ODE system we wish to take the self-product of. Should be passed in a form similiar to: {x0:x0+4x1x0+x1,...}
# * Output:
#       - final_sys:
#         - Data Type: Dict of Sympy Equations/Symbols
#         - Desc: The self-product of the passed systems. If sys was a TPP-implementable cubic form system, then final_sys should be a PP-implementable quadratic form system
'''
def self_product(sys):
    print("Running transfer-lift Stage 3 self product...")
    final_sys = stage_three_transfer_lift(sys)
    return final_sys


'''
# Takes the general product of an ODE
# * Input:
#       - sys1, sys2:
#         - Data Type: Dict of Sympy Equations/Symbols
#         - Desc: The starting ODE systems we wish to take the product of. Should be passed in a form similiar to: {x0:x0+4x1x0+x1,...}
# * Output:
#       - final_sys:
#         - Data Type: Dict of Sympy Equations/Symbols
#         - Desc: The product of the two passed systems.
'''
def general_product(sys1, sys2):
    # Kept for backwards compatibility.
    # The reliable Stage 3 path uses self_product/sys only.
    prod_sys, var_map = full_prod(sys1, sys2)
    final_sys = simple_sub(prod_sys, var_map)

    return final_sys


'''
# Inverts the order of the list of tuples for substititon. We know z_00, z_01, z_10, etc. will be unique, but in the general case what they map to won't be unique
# * Input:
#       - tuples:
#         - Data Type: List of tuples
#         - Desc: A list of tuples for which we want the objects within each tuple to have their order flipped in the tuple
# * Output:
#       - "[tuple[::-1] for tuple in tuples]":
#         - Data Type: List of tuples
#         - Desc: A list of tuples in which every tuple has the object in its first position swapped with the second position
'''
def reverse_tuples(tuples):
    return [tuple[::-1] for tuple in tuples]


'''
# A simple substitution function designed for converting from x0, x1, etc to z00, z01, etc.
# Kept for backwards compatibility with general_product.
'''
def simple_sub(sys, sub_map):
    sub_sys = {}
    sub_list = reverse_tuples(list(sub_map.items()))

    old_vars = set()
    for sub_key in sub_map:
        old_vars = old_vars.union(sub_map[sub_key].free_symbols)

    for eq_key in sys:
        sub_sys[eq_key] = simple_sub_loop(sys[eq_key], sub_list, old_vars)

    return sub_sys


'''
# Handles the loop so we can return early if the substitutions are complete.
# Kept for backwards compatibility with general_product.
'''
def simple_sub_loop(eq, sub_list, old_vars):
    subbed_eq = eq

    for i in range(len(sub_list)):
        subbed_eq = subbed_eq.subs(sub_list)

        if old_vars.isdisjoint(subbed_eq.free_symbols):
            return subbed_eq

    if not old_vars.isdisjoint(subbed_eq.free_symbols):
        raise ValueError(
            f"Legacy simple_sub_loop failed to close the system.\n"
            f"Remaining old variables: {old_vars.intersection(subbed_eq.free_symbols)}\n"
            f"Expression: {subbed_eq}"
        )

    return subbed_eq


'''
### A comprehensive function for running stage three start to finish ###
# * Input:
#       - sys:
#         - Data Type: Dict
#         - Desc: A TPP-implementable cubic form system. Should be passed in a format similar to: {"x_0": "x_0 * (-x_0**2 + 7*x_0*x_1 - x_1**2)", "x_1":"x_0 * (x_0**2 - 7*x_0*x_1 + x_1**2)"}
#       - sys2 (optional):
#         - Data Type: Dict
#         - Desc: A second TPP-implementable cubic form system. Should be passed in a format similar to: {"x_0": "x_0 * (-x_0**2 + 7*x_0*x_1 - x_1**2)", "x_1":"x_0 * (x_0**2 - 7*x_0*x_1 + x_1**2)"}
#       - full_prod (optional):
#         - Data Type: Binary
#         - Desc: A toggle to run the longer, slower, but non-shortcut simplified full product calculation instead of the
#                 faster half product calculation. This variable is irrelevant if you're passing two different
#                 systems and the full product will be used no matter what.
#       - standardize_main_var (optional):
#           - Data Type: String/Char
#           - Desc: If the TPP-implementable cubic form system passed as "sys" does not have
#                   variable names in the format base_var## (ex: x0, x1, x2 or a0, a1, a2), then a string can
#                   be passed here that will standardize all the variables in "sys" to have enumerated variable
#                   names with the root name as whatever was passed here
#       - standardize_main_var_2 (optional):
#           - Data Type: String/Char
#           - Desc: If the TPP-implementable cubic form system passed as "sys_2" does not have
#                   variable names in the format base_var## (ex: x0, x1, x2 or a0, a1, a2), then a string can
#                   be passed here that will standardize all the variables in "sys_2" to have enumerated variable
#                   names with the root name as whatever was passed here
#
# * Output:
#       - new_sys:
#         - Data Type: Dict
#         - Desc: A PP-implementable quadratic form sys. Will look something similar to: {"z_[0,0]":"z_[0,0]^2 + 2*z_[0,1]","z_[0,1]":"4*z_[1,1]", ...}
'''
def stage_three(sys, sys_2={}, full_prod=False, standardize_main_var="", standardize_main_var_2=""):
    clean_sys = None
    clean_sys_2 = None

    print("Beginning Stage 3...")

    # If a main variable has been passed for standardization, standardize the system's names first
    if standardize_main_var:
        clean_sys = clean_names(sys, standardize_main_var)

    # If there is a second system in need of cleaning, clean the names for that as well
    if sys_2 != {} and standardize_main_var_2:
        clean_sys_2 = clean_names(sys_2, standardize_main_var_2)

    # If the system(s) required no cleaning, simply create a copy
    if not clean_sys:
        clean_sys = sys.copy()

    if sys_2 != {} and not clean_sys_2:
        clean_sys_2 = sys_2.copy()

    print("Finished cleaning names for stage 3. Starting product calculation...")

    # If a second system was passed, do the general product
    if sys_2 != {}:
        print("Two systems passed. Running legacy general product calculation for stage 3...")
        new_sys = general_product(clean_sys, clean_sys_2)

    # If a specifier was passed to run the slower general product method for a self product, do the general method
    elif full_prod:
        print("Running legacy full product calculation for stage 3 self product...")
        new_sys = general_product(clean_sys, clean_sys)

    # If only one system was passed and no instruction was passed specifying otherwise, run the transfer-lift self_product
    else:
        print("Running transfer-lift self product calculation for stage 3...")
        new_sys = self_product(clean_sys)

    return new_sys


'''
### A version of the stage 3 runner function for running in sequence with the other stages
### when run in sequence, it can be assumed there is no system 2, and no need to run the general product
'''
def stage_three_quick(sys):
    new_sys = self_product(sys)
    return new_sys


def stage3_main(sys, standardize_main_var=""):
    # If a main variable has been passed for standardization, standardize the system's names first
    if standardize_main_var:
        clean_sys = clean_names(sys, standardize_main_var)
    else:
        clean_sys = sys.copy()

    new_sys = self_product(clean_sys)

    pp = PopulationProtocol(ode=new_sys)

    pp.protocol = pp.from_ode_system(new_sys)

    return pp


# sys = sympify({"x_0": "x_0 * (-x_0**2 +7*x_0*x_1 - x_1**2)", "x_1" : "x_0 * -1*(-x_0**2 +7*x_0*x_1 - x_1**2)"})
# stage3_main(sys, "x_0")

def _stage3_parse_idx(raw_symbol):
    """
    Local helper for creating Stage 3 initial values.
    Parses symbols like x_1, x_0, v_[0,1,0], etc.
    """
    raw_symbol = str(raw_symbol)

    if "_" in raw_symbol:
        numeric_part = raw_symbol.split("_", 1)[1]
        return ast.literal_eval(numeric_part)

    if "[" in raw_symbol:
        numeric_part = raw_symbol.split("[", 1)[1]
        numeric_part = "[" + numeric_part
        return ast.literal_eval(numeric_part)

    numeric_part = ''.join(ch for ch in raw_symbol if ch.isdigit())
    return ast.literal_eval(numeric_part)


def _stage3_make_z_symbol(var1, var2, var_order=None):
    """
    Creates the same kind of z-symbol used by the Stage 3 half product.

    Examples:
        x_1, x_2 -> z_[1,2]
        x_1, v_[0,1,0] -> z_[1,0,1,0]
    """

    if var_order is not None:
        order = {v: i for i, v in enumerate(var_order)}
        if order[var2] < order[var1]:
            var1, var2 = var2, var1

    i = _stage3_parse_idx(var1)
    j = _stage3_parse_idx(var2)

    if (isinstance(i, list) and isinstance(j, int)) or \
       (isinstance(i, int) and isinstance(j, list)):

        if isinstance(i, list):
            i_and_j = i + [j]
        else:
            i_and_j = [i] + j

        return Symbol(f"z_{i_and_j}")

    return Symbol(f"z_[{i},{j}]")


def make_stage_three_iv(old_sys, old_iv):
    """
    Builds the initial values for the Stage 3 half-product system.

    Convention:
        z_[i,i] = x_i^2
        z_[i,j] = 2*x_i*x_j for i != j
    """

    new_iv = {}
    var_order = list(old_sys.keys())

    for m, var1 in enumerate(var_order):
        for n, var2 in enumerate(var_order):
            if m <= n:
                z_var = _stage3_make_z_symbol(var1, var2, var_order)

                if var1 == var2:
                    new_iv[z_var] = old_iv[var1] ** 2
                else:
                    new_iv[z_var] = 2 * old_iv[var1] * old_iv[var2]

    return new_iv


def make_stage_three_square_mainvar(old_mainvar, old_sys):
    """
    If old tracked variable is x_1, this returns z_[1,1].
    """
    var_order = list(old_sys.keys())
    return _stage3_make_z_symbol(old_mainvar, old_mainvar, var_order)


