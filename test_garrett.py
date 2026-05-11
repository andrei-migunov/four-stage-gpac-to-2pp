from main import *


def _stage3_coeff_sign(coeff, tol=1e-10):
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


def _near_zero_polynomial(expr, state_vars, tol=1e-9, verbose=True):
    """
    Checks whether every coefficient of a polynomial is numerically close to zero.
    This is needed because floating point coefficients can leave residuals like 1e-14.
    """
    expr = expand(expr)

    if expr == 0:
        if verbose:
            print("Conservation residual max coefficient: 0")
        return True

    try:
        poly = Poly(expr, *sorted(list(state_vars), key=lambda s: str(s)))
        coeffs = poly.coeffs()

        if not coeffs:
            if verbose:
                print("Conservation residual max coefficient: 0")
            return True

        max_coeff = max(abs(float(N(c))) for c in coeffs)

        if verbose:
            print(f"Conservation residual max coefficient: {max_coeff}")

        return max_coeff < tol

    except Exception:
        try:
            val = abs(float(N(expr)))

            if verbose:
                print(f"Conservation residual numeric value: {val}")

            return val < tol
        except Exception:
            return False


def check_stage3_good_for_paper(sys, require_z_names=True, verbose=True, tol=1e-9):
    """
    Checks whether a Stage 3 output system has the expected paper-style PP form.

    Checks:
      1. Nonempty system.
      2. All state variables are z variables, if require_z_names=True.
      3. Closed system: RHS variables are all state variables.
      4. Every nonzero RHS term is quadratic in the state variables.
      5. Conservative: sum of all derivatives is zero up to tolerance.
      6. Every negative term in z_i' contains z_i.
      7. No positive own-square term z_i^2 appears in z_i'.

    Returns True if all checks pass.
    Raises ValueError with details if a check fails.
    """

    if not sys:
        raise ValueError("Stage 3 check failed: system is empty.")

    state_vars = set(sys.keys())

    # 1. Optional z-name check
    if require_z_names:
        non_z_vars = [v for v in state_vars if not str(v).startswith("z_")]

        if non_z_vars:
            raise ValueError(
                "Stage 3 check failed: not all state variables are z variables.\n"
                f"Non-z state variables: {non_z_vars[:20]}"
            )

    if verbose:
        print("Stage 3 check passed: state variables look correct.")

    # 2. Closed system check
    rhs_vars = set()

    for expr in sys.values():
        rhs_vars |= expand(expr).free_symbols

    missing_vars = rhs_vars - state_vars

    if missing_vars:
        raise ValueError(
            "Stage 3 check failed: system is not closed.\n"
            f"RHS contains variables that are not state variables:\n"
            f"{sorted(missing_vars, key=lambda s: str(s))}"
        )

    if verbose:
        print("Stage 3 check passed: system is closed.")

    # 3. Quadratic-form check
    for var, expr in sys.items():
        expr = expand(expr)

        for term in Add.make_args(expr):
            if term == 0:
                continue

            coeff = term.as_coeff_Mul()[0]

            # Ignore tiny floating-point roundoff terms.
            if _stage3_coeff_sign(coeff, tol=tol) == 0:
                continue

            powers = term.as_powers_dict()
            degree_sum = 0

            for z in state_vars:
                power = sympify(powers.get(z, 0))

                if power != 0 and power.is_integer is not True:
                    raise ValueError(
                        "Stage 3 check failed: non-integer power found.\n"
                        f"Equation: {var}'\n"
                        f"Term: {term}\n"
                        f"Variable: {z}\n"
                        f"Power: {power}"
                    )

                degree_sum += power

            if degree_sum != 2:
                raise ValueError(
                    "Stage 3 check failed: non-quadratic term found.\n"
                    f"Equation: {var}'\n"
                    f"Term: {term}\n"
                    f"Degree in state variables: {degree_sum}"
                )

    if verbose:
        print("Stage 3 check passed: every term is quadratic.")

    # 4. Conservation check with tolerance
    total_derivative = simplify(expand(sum(sys.values())))

    if not _near_zero_polynomial(total_derivative, state_vars, tol=tol, verbose=verbose):
        raise ValueError(
            "Stage 3 check failed: system is not conservative.\n"
            f"Sum of all derivatives simplifies to:\n{total_derivative}"
        )

    if verbose:
        print("Stage 3 check passed: system is conservative.")

    # 5. Negative terms must contain their own variable
    for var, expr in sys.items():
        expr = expand(expr)

        for term in Add.make_args(expr):
            if term == 0:
                continue

            coeff = term.as_coeff_Mul()[0]
            sign = _stage3_coeff_sign(coeff, tol=tol)

            # Ignore tiny floating-point roundoff terms.
            if sign == 0:
                continue

            if sign < 0:
                if not term.has(var):
                    raise ValueError(
                        "Stage 3 check failed: negative term does not contain its own variable.\n"
                        f"Equation: {var}'\n"
                        f"Bad term: {term}"
                    )

    if verbose:
        print("Stage 3 check passed: negative terms contain their own variable.")

    # 6. No positive own-square terms
    for var, expr in sys.items():
        expr = expand(expr)

        for term in Add.make_args(expr):
            if term == 0:
                continue

            coeff = term.as_coeff_Mul()[0]
            sign = _stage3_coeff_sign(coeff, tol=tol)

            # Ignore tiny floating-point roundoff terms.
            if sign == 0:
                continue

            power_of_own_var = sympify(term.as_powers_dict().get(var, 0))

            if sign > 0 and power_of_own_var >= 2:
                raise ValueError(
                    "Stage 3 check failed: positive own-square term found.\n"
                    f"Equation: {var}'\n"
                    f"Bad term: {term}"
                )

    if verbose:
        print("Stage 3 check passed: no positive own-square terms.")

    if verbose:
        print("ALL STAGE 3 PAPER-FORM CHECKS PASSED.")

    return True


sys = ""