import sympy as sp

def z_to_pow(vvar):
    try:
        return [int(x) for x in str(vvar)[2:].strip('][').split(',')]
    except ValueError:
        # Plain variable name (e.g. 'x', 'y', 'z'): sort alphabetically
        return [ord(c) for c in str(vvar)]
        
def compare_ode_systems(sys_in, sys_recovered, label="", max_show=50):
    """
    Compare an input ODE system (lowercase symbol keys) against a recovered
    ODE system (uppercase symbol keys, e.g. from pp_reactions_to_odes).

    For each variable present in either system the function checks:
      - whether the variable is missing from one side
      - whether the expressions differ (via expand(recovered - expected) == 0)

    Parameters
    ----------
    sys_in        : dict  {sympy.Symbol (lower) -> sympy expr}
    sys_recovered : dict  {sympy.Symbol (upper) -> sympy expr}
    label         : str   printed in the header for identification
    max_show      : int   max mismatches to print in full before truncating

    Returns
    -------
    dict with keys:
        'match'    : bool    True iff systems are identical
        'missing'  : list    vars in input but absent from recovered
        'extra'    : list    vars in recovered but absent from input
        'mismatch' : list    (var, expected_expr, got_expr, diff) tuples
    """

    # ── Build uppercase substitution from ALL symbols that appear anywhere ──
    all_syms = set(sys_in.keys())
    for expr in sys_in.values():
        all_syms |= expr.free_symbols
    to_upper = {s: sp.Symbol(str(s).upper()) for s in all_syms}

    # Pre-expand input expressions once (uppercase), keyed by uppercase symbol
    sys_upper = {
        to_upper[var]: sp.expand(expr.subs(to_upper))
        for var, expr in sys_in.items()
    }

    # ── Classify every variable seen on either side ──
    all_vars    = sorted(sys_upper.keys() | sys_recovered.keys(),
                         key=lambda v: z_to_pow(v))
    missing     = []   # in input, absent from recovered
    extra       = []   # in recovered, absent from input
    mismatches  = []   # present on both sides but different

    for var in all_vars:
        in_input     = var in sys_upper
        in_recovered = var in sys_recovered

        if in_input and not in_recovered:
            missing.append(var)
        elif in_recovered and not in_input:
            extra.append(var)
        else:
            diff = sp.expand(sys_recovered[var] - sys_upper[var])
            if diff != 0:
                mismatches.append((var, sys_upper[var], sys_recovered[var], diff))

    # ── Report ──────────────────────────────────────────────────────────────
    title = f"ODE System Comparison  {label}".strip()
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")
    print(f"  Variables in input:     {len(sys_upper)}")
    print(f"  Variables in recovered: {len(sys_recovered)}")

    all_ok = not missing and not extra and not mismatches

    if all_ok:
        print(f"  ✓  All {len(sys_upper)} equations match exactly.")
    else:
        total = len(missing) + len(extra) + len(mismatches)
        print(f"  ✗  {total} issue(s) found:\n")

        if missing:
            print(f"  Missing from recovered ({len(missing)} var(s)):")
            for v in missing:
                print(f"    d[{v}]/dt = {sys_upper[v]}")

        if extra:
            print(f"  Extra in recovered, not in input ({len(extra)} var(s)):")
            for v in extra:
                print(f"    d[{v}]/dt = {sys_recovered[v]}")

        if mismatches:
            shown = mismatches[:max_show]
            print(f"  Expression mismatches ({len(mismatches)} var(s)"
                  + (f", showing first {max_show}" if len(mismatches) > max_show else "")
                  + "):")
            for var, expected, got, diff in shown:
                print(f"\n    d[{var}]/dt")
                print(f"      expected:   {expected}")
                print(f"      got:        {got}")
                print(f"      difference: {diff}")
            if len(mismatches) > max_show:
                print(f"\n    ... and {len(mismatches) - max_show} more.")

    print(f"{'─' * 60}")

    return {
        "match":    all_ok,
        "missing":  missing,
        "extra":    extra,
        "mismatch": mismatches,
    }
