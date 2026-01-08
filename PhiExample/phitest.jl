
using Catalyst 
using Latexify
phipp = @reaction_network begin
    1/16, X + X --> X + Y
    7/8, X + X --> X + X
    1/16, X + Y --> X + Z
    7/16, X + Y --> X + X
    1/2, X + Y --> X + Y
    1/8, X + Z --> Y + Z
    7/8, X + Z -->  X + Y
    1/16, Y + Z --> Z + Z
    15/16, Y + Z --> Y + Z
end

diffsys = convert(ODESystem, phipp)
display(diffsys)

diffsys = convert(ODESystem, phipp)
for eq in equations(diffsys)
    println(eq)
end



# using DifferentialEquations
# tspan = (0.0,30.0)
# u0    = [:X => 1, :Y => 0, :Z => 0]
# op    = ODEProblem(phipp, u0, tspan)
# sol   = solve(op, Tsit5())       # use Tsit5 ODE solver



# display(op)

# using Plots
# plot(sol)

# latexify(phipp; form=:ode)


# phi = {
#     (x,x): {(x,y): 1/8, (x,x): 7/8},
#     (x,y): {(x,z):1/16, (x,x):7/16, (x,y):1/2},
#     (x,z): {(y,z):1/8, (x,y): 7/8},
#     (y,z): {(z,z): 1/16, (y,z):15/16},
# }

#Previously ... I had plugged the incorrectly derived reactions into Julia, and it gave me back a set of ODEs that didn't work.
#From there, we found the mistake. It's like using Julia to backward-compile to ODEs

# I had given it this, originally, which is incorrect and, unfortunately, in our abstract at CFW:
# phipp = @reaction_network begin
#     1/8, X + X --> X + Y
#     7/8, X + X --> X + X
#     1/16, X + Y --> X + Z
#     7/16, X + Y --> X + X
#     1/2, X + Y --> X + Y
#     1/8, X + Z --> Y + Z
#     7/8, X + Z -->  X + Y
#     1/16, Y + Z --> Z + Z
#     15/16, Y + Z --> Y + Z
# end