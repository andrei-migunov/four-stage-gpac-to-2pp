import sympy as sp
from sympy import expand, Symbol, Add, Mul
from collections import deque
from St1_Fns import *

import sympy as sp
from collections import deque

# Gemini, 1/7/26
import sympy as sp
from collections import deque

def carothers_observation1_ode_system_v2(odes, input_variables=None):
    # 1. Setup and Standardization
    sym_odes = {}
    for k, v in odes.items():
        sym_odes[sp.sympify(k)] = sp.sympify(v).expand()
        
    if input_variables:
        variables = list(input_variables)
    else:
        variables = sorted(list(sym_odes.keys()), key=lambda x: x.name)

    # 2. Helper Functions
    def get_powers(term):
        """Extracts power tuple from a term (ignoring coeff)."""
        term_no_coeff = term.as_coeff_Mul()[1]
        powers = []
        for var in variables:
            p = sp.degree(term_no_coeff, gen=var)
            powers.append(int(p))
        return tuple(powers)

    def get_symbol(powers):
        """Returns x_i for degree 1, v_[...] for others, 1 for degree 0."""
        total_degree = sum(powers)
        if total_degree == 1:
            return variables[powers.index(1)]
        elif total_degree == 0:
            return sp.Integer(1)
        else:
            p_str = ",".join(map(str, powers))
            return sp.Symbol(f"v_[{p_str}]")

    def monomial_to_var(term, queue, visited):
        """
        Converts a monomial term into its variable representation.
        Does NOT multiply variables together; just finds the symbol for ONE monomial.
        """
        powers = get_powers(term)
        sym = get_symbol(powers)
        
        # If it's a v-variable (sum > 1) and we haven't seen it, queue it
        if sum(powers) > 1 and powers not in visited:
            visited.add(powers)
            queue.append(powers)
        return sym

    # 3. Initialization
    final_system = {}
    visited_powers = set()
    queue = deque()

    # Mark original variables as visited
    for i in range(len(variables)):
        p = [0]*len(variables)
        p[i] = 1
        visited_powers.add(tuple(p))

    # 4. Process Original Variables
    # We factor out the variable itself from negative terms if possible.
    for var in variables:
        rhs = sym_odes[var]
        new_rhs = 0
        var_powers = get_powers(var)
        
        terms = rhs.args if rhs.is_Add else [rhs]
        for term in terms:
            coeff, var_part = term.as_coeff_Mul()
            
            if coeff < 0:
                # Try to factor out 'var' to enforce CRN safety
                # Check divisibility by checking if exponents are >= 1 at the var's index
                term_powers = get_powers(var_part)
                idx = variables.index(var)
                
                if term_powers[idx] >= 1:
                    # Divisible! Remainder = term / var
                    rem_powers = list(term_powers)
                    rem_powers[idx] -= 1
                    rem_term = sp.Mul(*[v**p for v, p in zip(variables, rem_powers)])
                    
                    # Result: -k * var * v(remainder)
                    new_rhs += coeff * var * monomial_to_var(rem_term, queue, visited_powers)
                else:
                    # Not divisible (e.g. x' = -y). Cannot enforce constraint.
                    new_rhs += coeff * monomial_to_var(var_part, queue, visited_powers)
            else:
                new_rhs += coeff * monomial_to_var(var_part, queue, visited_powers)
                
        final_system[var] = new_rhs

    # 5. Process New Variables (The fix for Blowup + CRN Safety)
    while queue:
        current_powers = queue.popleft()
        current_sym = get_symbol(current_powers)
        rhs_accum = 0
        
        # Chain Rule: d(v)/dt = sum( p_i * (v / x_i) * dx_i/dt )
        for i, p_val in enumerate(current_powers):
            if p_val > 0:
                xi = variables[i]
                
                # 1. Identify v_reduced (The partial derivative part)
                # v_reduced = v_current / x_i
                red_powers = list(current_powers)
                red_powers[i] -= 1
                red_term = sp.Mul(*[v**p for v, p in zip(variables, red_powers)])
                
                # We get the SYMBOL for v_reduced. We do NOT multiply it algebraically yet.
                v_reduced_sym = monomial_to_var(red_term, queue, visited_powers)
                
                # 2. Iterate through terms of dx_i/dt (The ODE part)
                xi_rhs = sym_odes[xi]
                xi_terms = xi_rhs.args if xi_rhs.is_Add else [xi_rhs]
                
                for term in xi_terms:
                    coeff, term_monomial = term.as_coeff_Mul()
                    
                    if coeff >= 0:
                        # --- POSITIVE TERM ---
                        # Logic: d(v) += coeff * p * v_reduced * v(term_monomial)
                        # We keep v_reduced and v(term) SEPARATE. 
                        # This ensures degree <= 2 (Quadratization).
                        v_term_sym = monomial_to_var(term_monomial, queue, visited_powers)
                        rhs_accum += coeff * p_val * v_reduced_sym * v_term_sym
                        
                    else:
                        # --- NEGATIVE TERM ---
                        # Logic: We must form -k * v_current * v_something
                        # We check if 'term_monomial' is divisible by x_i
                        
                        t_powers = get_powers(term_monomial)
                        if t_powers[i] >= 1:
                            # It is divisible! We can perform the swap.
                            # Standard: v_reduced * (x_i * alpha)
                            # Swapped:  (v_reduced * x_i) * alpha  ==  v_current * alpha
                            
                            # Calculate alpha (remainder)
                            alpha_powers = list(t_powers)
                            alpha_powers[i] -= 1
                            alpha_term = sp.Mul(*[v**p for v, p in zip(variables, alpha_powers)])
                            
                            v_alpha_sym = monomial_to_var(alpha_term, queue, visited_powers)
                            
                            # Result: -k * p * v_current * v_alpha
                            rhs_accum += coeff * p_val * current_sym * v_alpha_sym
                        else:
                            # Not divisible. We cannot factor out v_current.
                            # Fallback to standard chain rule (mathematically correct, but maybe not CRN safe)
                            v_term_sym = monomial_to_var(term_monomial, queue, visited_powers)
                            rhs_accum += coeff * p_val * v_reduced_sym * v_term_sym

        final_system[current_sym] = sp.simplify(rhs_accum)

    return final_system




"""
Convert the initial values from self.input_iv (for the original system)
to a new dictionary of initial values for the compiled system.

- Original variables retain their initial values.
- v-variables are initialized as the product of corresponding input_iv powers.

crn: the original CRN before degree squishing, must be cleaned
vsys: the output CRN after degree squishing
crn_iv: the initial values for the original CRN
returns: new_iv, the initial values for vsys
"""
def convert_to_deg2_IV(crn,vsys,crn_iv):
    new_iv = {}
    input_iv = crn_iv
    original_variables = list(crn.keys())

    for var in list(vsys.keys()):
        if var in input_iv:
            new_iv[var] = input_iv[var]

        elif is_vvar(var):
            powers = v_to_pow(var)
            val = 1
            for base, power in zip(original_variables, powers):
                val *= input_iv.get(base, 0)**power
            new_iv[var] = val                
        else:
            raise ValueError(f"Variable {var} not found in input IV and is not a v-variable.")

    return new_iv


def is_vvar(symbol):
    return symbol.is_Symbol and str(symbol).startswith("v_[")