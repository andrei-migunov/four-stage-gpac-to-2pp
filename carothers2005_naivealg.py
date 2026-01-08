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
        """Returns x_i for degree 1, v_[...] for others."""
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
        Takes a monomial (e.g. x*y) and returns its variable representation.
        If it's linear (x), returns x.
        If it's higher order (v_[...]), returns that symbol and adds to queue if new.
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

    # 4. Process Original ODEs
    # We essentially "lift" the original system first.
    # If dx/dt = x*y, we change it to dx/dt = v_[1,1]
    for var in variables:
        rhs = sym_odes[var]
        new_rhs = 0
        for term in (rhs.args if rhs.is_Add else [rhs]):
            coeff, variable_part = term.as_coeff_Mul()
            # Convert the variable part to a symbol (x or v_...)
            var_sym = monomial_to_var(variable_part, queue, visited_powers)
            new_rhs += coeff * var_sym
        final_system[var] = new_rhs

    # 5. Process the Queue (The recursive variable generation)
    while queue:
        current_powers = queue.popleft() # e.g., (1, 1, 0)
        current_sym = get_symbol(current_powers)
        
        # We build the RHS for d(v_current)/dt
        # Using the chain rule: sum( p_i * (M/x_i) * dx_i/dt )
        rhs_accum = 0
        
        for i, p_val in enumerate(current_powers):
            if p_val > 0:
                # A. Identify the "Reduced" variable (M / x_i)
                # This guarantees we step down in degree (The logic you requested)
                reduced_powers = list(current_powers)
                reduced_powers[i] -= 1
                reduced_term = sp.Mul(*[v**p for v, p in zip(variables, reduced_powers)])
                
                # Get the symbol for this reduced term (add to queue if needed)
                # This is the "alpha with one lower degree"
                v_reduced = monomial_to_var(reduced_term, queue, visited_powers)
                
                # B. Multiply by the ODE of the variable we differentiated
                # We use the ORIGINAL definition of the ODE to capture all interactions
                xi_rhs = sym_odes[variables[i]]
                
                for term in (xi_rhs.args if xi_rhs.is_Add else [xi_rhs]):
                    coeff, var_part = term.as_coeff_Mul()
                    
                    # C. Convert the term from the ODE into a variable
                    # If ODE has term 'a*b', this becomes v_[1,1]
                    # We DO NOT combine v_reduced and var_part algebraically here.
                    v_from_ode = monomial_to_var(var_part, queue, visited_powers)
                    
                    # D. Construct the final term: coeff * power * v_reduced * v_from_ode
                    # This ensures the result is strictly Quadratic (product of two vars)
                    rhs_accum += coeff * p_val * v_reduced * v_from_ode

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