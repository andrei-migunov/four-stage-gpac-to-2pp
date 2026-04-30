from main import *

def check_stage3_good_for_paper(sys, require_z_names=True, verbose=True):
    """
    Checks whether a Stage 3 output system has the expected paper-style PP form.

    Checks:
      1. Nonempty system.
      2. All state variables are z variables, if require_z_names=True.
      3. Closed system: RHS variables are all state variables.
      4. Every nonzero RHS term is quadratic in the state variables.
      5. Conservative: sum of all derivatives is 0.
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

            powers = term.as_powers_dict()

            degree_sum = 0

            for z in state_vars:
                power = powers.get(z, 0)

                if power != 0 and not Integer(power).is_integer:
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

    # 4. Conservation check
    total_derivative = simplify(expand(sum(sys.values())))

    if total_derivative != 0:
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

            if coeff.is_negative:
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
            power_of_own_var = term.as_powers_dict().get(var, 0)

            if coeff.is_positive and power_of_own_var >= 2:
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