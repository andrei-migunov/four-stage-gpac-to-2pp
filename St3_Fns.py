from St0_Fns import *
from St1_Fns import *
from sympy import *
from sympy import Integral#, Zero
from sympy.utilities.lambdify import *
import numpy as np
import scipy.integrate as intg
#from Old.PopulationProtocol import *
import ast # Used to evaluate lists in string format for variable names
import re # Used to quickly drop all non-numeric information when evaluating variable names for indexing

# TODO: Create a test script for running stage 3 in-sequence
# TODO: Try writing the better substitution function
# TODO: Finish the test functions 


    
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
#       - Data Type: Sympy equation
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
            #Checks for lower half of matrix
            if m <= n:
                pair_index += 1
                if pair_index % 20 == 0 or pair_index == 1 or pair_index == total_pairs:
                    print(f"  half_prod progress: {pair_index}/{total_pairs} (var1={var1}, var2={var2})")

                #Adds new z variables to system
                #z_system[symbols(f"z_[{i},{j}]")] = get_z_derivative_half_prod(sys1, var1, var2, i, j)
                #print(f"z_[{i},{j}]: ",get_z_derivative(sys1, var1, var2, i, j))
                i = sym_idx_parser(var1)
                j = sym_idx_parser(var2)

                # if, wlog, i is an integer and j is a list, append i to j (i.e. if i = 1, and j = [1,2] then create a variable with indexing as z_[1,1,2])
                if (isinstance(i, list) and isinstance(j,int)) or (isinstance(i, int) and isinstance(j,list)):
                    
                    # Create the index for a z_[i,j,k] variable
                    if isinstance(i, list): 
                        i_and_j = i + [j]
                    else:
                        i_and_j = [i] + j

                    z_system[Symbol(f"z_{i_and_j}")] = get_z_derivative_half_prod(sys1, var1, var2, i, j)
                    if i == j: 
                        z_var_map[Symbol(f"z_{i_and_j}")] = var1**2
                    else:
                        z_var_map[Symbol(f"z_{i_and_j}")] = 2* var1 * var2

                else:
                    #Adds new z variables to system
                    z_system[Symbol(f"z_[{i},{j}]")] = get_z_derivative_half_prod(sys1, var1, var2, i, j)
                    if i == j: 
                        z_var_map[Symbol(f"z_[{i},{j}]")] = var1 ** 2
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
#       - Data Type: Sympy equation
#       - Desc: The equation resulting from performing the product of the system for the current variable
'''
def get_z_derivative_half_prod(sys1, var1, var2, i, j):
    #If variables are the same i.e X_0*X_0
    if i == j:
        # simplified = simplify(2 * var1 * sys1[var1])
        # expanded = expand(simplified)
        return expand(2 * var1 * sys1[var1], expand_mul=True,expand_power_exp=True)
    #If variable are the different i.e X_0*X_1
    else:
        # simplified = simplify(2 * sys1[var1] * var2 + 2 * var1 * sys1[var2])
        # expanded = expand(simplified)
        return expand(2 * sys1[var1] * var2 + 2 * var1 * sys1[var2], expand_mul=True,expand_power_exp=True)
    

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
def full_prod(sys1,sys2):
    # Store the z variables
    z_system = {}
    # Store the mappings for convertings x vars into z vars
    z_var_map = {}

    # Assume the variables in the systems all have proper indexing
    for var1 in sys1.keys():
        for var2 in sys2.keys():
            i = sym_idx_parser(var1)
            j = sym_idx_parser(var2)

            # if, wlog, i is an integer and j is a list, append i to j (i.e. if i = 1, and j = [1,2] then create a variable with indexing as z_[1,1,2])
            if (isinstance(i, list) and isinstance(j,int)) or (isinstance(i, int) and isinstance(j,list)):
                if isinstance(i, list): 
                    i_and_j = i + [j]
                else:
                    i_and_j = [i] + j
                z_system[Symbol(f"z_{i_and_j}")] = get_z_derivative_full_prod(sys1, var1, sys2, var2)
                z_var_map[Symbol(f"z_{i_and_j}")] = simplify(var1 * var2)
            else:
                #Adds new z variables to system
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

    # Variables are sometimes written as x0 and sometimes as x_0. First we handle the case with an "_"
    if "_" in raw_symbol:
        # Get the part of the string after the "_", which should contain the indexing information
        numeric_part = raw_symbol.split("_")[1]
        # This should either evaluate to a list or an integer. I think its better to handle the distinction after than to force return just lists
        idx = ast.literal_eval(numeric_part)
    else:
        # Without the "_", we're looking for either an integer or a list. So let's check for list, and if we can't find a list we strip anything that isn't an integer and assume the remainder must be the index
        if "[" in raw_symbol:
            # Get the part of the string after the "[", which should contain the indexing information
            numeric_part = raw_symbol.split("[")[1]
            # Add back the "[" that got removed in the split
            numeric_part = "[" + numeric_part
            # This should evaluate to a list
            idx = ast.literal_eval(numeric_part)
        # Next we cover the case for variables such as x0 or z12
        else: 
            numeric_part = re.sub(r'[^0-9]', '', raw_symbol)
            # This should evaluate to an integer
            idx = ast.literal_eval(numeric_part)

    # This return may return either a list or an integer depending on what the input variable's name was like!        
    return idx

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
    # Take the half product
    print("Running half product calculation for stage 3 self product...")
    prod_sys, var_map = half_prod(sys) 
    # Convert the variables to z's
    print("Running simple_sub for stage 3 self product...")
    final_sys = simple_sub(prod_sys, var_map)
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
    # Take the full product, as required for general products on ODEs
    prod_sys, var_map = full_prod(sys1,sys2)
    # Convert the variables to z's
    final_sys = simple_sub(prod_sys, var_map)
    return final_sys

'''
# Inverts the order of the list of tuples for substititon. We know z_00, z_01, z_10, etc. will be unique, but in the general case what they map to won't be unique
# * Input:
#       - tuples:
#         - Data Type: List of tuples
#         - Desc: A list of tuples for which we want the objects within each tuple to have their order flipped in the tuple
# * Outpu:
#       - "[tuple[::-1] for tuple in tuples]":
#         - Data Type: List of tuples
#         - Desc: A list of tuples in which every tuple has the object in its first position swapped with the second position
'''
'''
# A simple substitution function designed for converting from x0, x1, etc to z00, z01, etc.
# * Input:
#       - sys: 
#         - Data Type: Dict of sympy equations
#         - Desc: A system of equations. Should be passed in a format similar to: {a:x+y,b:x^2+xy+y^2,...}
#       - sub_map: 
#         - Data Type: Dict
#         - Desc: A dictionary containing mapping information from the old variables to the new variables
# * Output:
#       - sub_sys:
#         - Data Type: Dict
#         - Desc: A system of equations with the variables defined in the sub_map removed. Should be passed in a format similar to: {"x_0": "x_0 * (-x_0**2 + 7*x_0*x_1 - x_1**2)", "x_1":"x_0 * (x_0**2 - 7*x_0*x_1 + x_1**2)"}
'''
def simple_sub(sys, sub_map):
    sub_sys = {}

    # Map old expressions to z variables for a single substitution pass.
    ordered_subs = sorted(
        [(sub_map[key], key) for key in sub_map],
        key=lambda item: item[0].count_ops(),
        reverse=True,
    )

    total_eqs = len(sys)
    total_subs = len(ordered_subs)
    print(f"simple_sub: substituting {total_eqs} equations with {total_subs} mappings...")

    for eq_index, (eq_key, expr) in enumerate(sys.items(), start=1):
        if eq_index % 20 == 0 or eq_index == 1 or eq_index == total_eqs:
            print(f"  simple_sub progress: equation {eq_index}/{total_eqs} -> {eq_key}")
        sub_sys[eq_key] = expr.subs(ordered_subs)

    return sub_sys

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
def stage_three(sys, sys_2={}, full_prod = False, standardize_main_var="", standardize_main_var_2=""):
    print("Beginning Stage 3...")
    clean_sys = None
    # If a main variable has been passed for standardization, standardize the system's names first
    if standardize_main_var:
        clean_sys = clean_names(sys,standardize_main_var)
    # If there is a second system in need of cleaning, clean the names for that as well
    if sys_2 != {} and standardize_main_var_2:
        clean_sys_2 = clean_names(sys_2,standardize_main_var_2)
    

    # If the system(s) required no cleaning, simply create a copy
    if not clean_sys:
        clean_sys = sys.copy()
    if sys_2 != {} and not clean_sys_2:
        clean_sys_2 = sys_2.copy()
    print("Finished cleaning names for stage 3. Starting product calculation...")

    # If a second system was passed, do the general product
    if sys_2 != {}:
        print("Two systems passed. Running general product calculation for stage 3...")
        new_sys = general_product(clean_sys,clean_sys_2)
    # If a specifier was passed to run the slower general product method for a self product, do the general method
    elif full_prod: 
        print("Running full product calculation for stage 3 self product...")
        new_sys = general_product(clean_sys,clean_sys)
    # If only one system was passed and no instruction was passed specifying otherwise, run the faster self_product calculation
    else:
        print("Running optimized self product calculation for stage 3...")
        new_sys = self_product(clean_sys)

    return new_sys

'''
### A version of the stage 3 runner function for running in sequence with the other stages
### (when run in sequence, it can be assumed there is no system 2, and no need to run the general product)

# * Input:
#       - sys: 
#         - Data Type: Dict
#         - Desc: A TPP-implementable cubic form system. Should be passed in a format similar to: {"x_0": "x_0 * (-x_0**2 + 7*x_0*x_1 - x_1**2)", "x_1":"x_0 * (x_0**2 - 7*x_0*x_1 + x_1**2)"}
# * Output:
#       - new_sys:
#         - Data Type: Dict
#         - Desc: A PP-implementable quadratic form sys. Will look something similar to: {"z_[0,0]":"z_[0,0]^2 + 2*z_[0,1]","z_[0,1]":"4*z_[1,1]", ...} 
'''
def stage_three_quick(sys):
    new_sys = self_product(sys)

    return new_sys

def stage3_main(sys, standardize_main_var=""):
    # If a main variable has been passed for standardization, standardize the system's names first
    if standardize_main_var:
        clean_sys = clean_names(sys,standardize_main_var)
    else:
        clean_sys = sys.copy()
    
    new_sys = self_product(clean_sys)

    pp = PopulationProtocol(ode = new_sys)

    pp.protocol = pp.from_ode_system(new_sys)
    return pp



# sys = sympify({"x_0": "x_0 * (-x_0**2 +7*x_0*x_1 - x_1**2)", "x_1" : "x_0 * -1*(-x_0**2 +7*x_0*x_1 - x_1**2)"})
# stage3_main(sys,"x_0")
