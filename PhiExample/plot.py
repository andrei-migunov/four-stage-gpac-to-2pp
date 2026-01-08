from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import numpy as np

# Define the differential equations
from scipy.integrate import solve_ivp
import numpy as np
import matplotlib.pyplot as plt

# Define the differential equations
# def system(t, y):
#     x, y, z = y
#     dxdt = -2*x**2 + 7*x*y -2*x*z
#     dydt = 2*x**2  -8*x*y + 16*x*z - y*z
#     dzdt = x*y + y*z -14*x*z
#     return [dxdt, dydt, dzdt]

# How Julia converted the Phi PP back into an ODE for the X X -> X Y:
# def system(t, y):
#     x, y, z = y
#     dxdt = -(1/16)*x**2 + (7/16)*x*y -(1/8)*x*z
#     dydt = (1/16)*x**2  -(1/2)*x*y + 1*x*z - (1/16)*y*z
#     dzdt = (1/16)*x*y + (1/16)*y*z -(7/8)*x*z
#     return [dxdt, dydt, dzdt]

#Julia generated ODE for the X X -> Y Y version
def system(t, y):
    x, y, z = y
    dxdt = -(1/8)*x**2 + (7/16)*x*y -(1/8)*x*z
    dydt = (1/16)*x**2  -(1/2)*x*y + 1*x*z - (1/16)*y*z
    dzdt = (1/16)*x*y + (1/16)*y*z -(7/8)*x*z
    return [dxdt, dydt, dzdt]


# Initial conditions
y0 = [1, 0, 0]

# Time points where the solution is computed
t = np.linspace(0, 50, 20000)

# Solve the system of differential equations
sol = solve_ivp(system, [0, 50], y0, t_eval=t)

# Compute z + y/2
z_plus_half_y = sol.y[2] + sol.y[1]/2

# Calculate the negative root of the golden ratio
phi = (1 - np.sqrt(5.0)) / 2
horizontal_line_value = (phi + 1) / 3
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(sol.t, z_plus_half_y, label='$z_{11} + z_{01}/2$')
plt.plot(sol.t, sol.y[0], label='$z_{00}$')
plt.plot(sol.t, sol.y[1], label='$z_{01}$')
plt.plot(sol.t, sol.y[2], label='$z_{11}$')
plt.axhline(horizontal_line_value, color='r', linestyle='--', label='$(\phi +1)/3$') #y = .127
plt.title('Evolution of $z_{11} + z_{01}/2$ over Time')
plt.xlabel('Time')
plt.ylabel('$z_{11} + z_{01}/2$')
plt.legend()
plt.grid(True)
plt.show()

