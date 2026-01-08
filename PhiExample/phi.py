
from ppsim import Simulation
import matplotlib.pyplot as plt
import numpy as np




x, y, z = 'x', 'y',"z"

##  Dave's method
#   That will give you PP without restriction.
# phi = {
#    (x,x): {(x,y): 1/8, (x,x): 7/8},
#     (x,y): {(x,z):1/16, (x,x):7/16, (x,y):1/2},
#     (x,z): {(y,z):1/8, (x,y): 7/8},
#     (y,z): {(z,z): 1/16, (y,z):15/16},
# }

#fix 4/18/24 THIS ONE WORKS
phi = {
   (x,x): {(y,y): 1/8, (x,x): 7/8},
    (x,y): {(x,z):1/16, (x,x):7/16, (x,y):1/2},
    (x,z): {(y,z):1/8, (x,y): 7/8},
    (y,z): {(z,z): 1/16, (y,z):15/16},
}

# OR THIS WORKS TOO? It does not...
# phi = {
#    (x,x): {(x,y): 1/16, (x,x): 15/16},
#     (x,y): {(x,z):1/16, (x,x):7/16, (x,y):1/2},
#     (x,z): {(y,z):1/8, (x,y): 7/8},
#     (y,z): {(z,z): 1/16, (y,z):15/16},
# }

# LPP paper  : urn model
# phi = {
#     (x,x): {(x,x): 13.0/14, (y, y): 1.0/14},
#     (x,y): {(x,x):3.0/4, (y,y):3.0/14, (z,z):1.0/28},
#     (x,z): {(x,x):3.0/7, (y,y):4.0/7},
#     (y,z): {(y,y): 13.0/28, (z,z): 15.0/28},
# }



n = 10 ** 9

init_config = {x: 1.0* n, y: 0.0 * n, z: 0.0 * n}
sim = Simulation(init_config, phi)

sim.run(100)
print(sim.history/n)
df= sim.history

# Step 2: Add the two columns together
res = (df[z] + (df[y]/2))

print(res)
# phi

# Calculate the negative root of the golden ratio
ph = (1 - np.sqrt(5.0)) / 2
horizontal_line_value = (ph + 1) / 3

print(f'The actual value of ((Phi +1) /3) is {horizontal_line_value}')
print(f'The pop (/n) of x (which is z00) by end of sim is {df[x].iloc[-1]/n}')
print(f'The pop (/n) of y (which is z01) by end of sim is {df[y].iloc[-1]/n}')
print(f'The pop (/n) of z (which is z11) by end of sim is {df[z].iloc[-1]/n}')
print(f'The result (/n) we compute in the paper as z11 + .5*z01 by end of sim is {res.iloc[-1]/n}')


# Step 3: Plot the sum
ax = df.plot(y = [x,y,z])
res.plot(label="res") 

#I am dumb - the line below needed to be multiplied by the enormous population n in order to not be on x axis
ax.axhline(y = horizontal_line_value*n, color='r', linestyle='-', label=f'Horizontal Line at {horizontal_line_value:.4f}')
plt.title('Large-population Protocol that computes (phi +1)/3')
plt.xlabel('Time')  # Adjust as per your DataFrame index
plt.ylabel('Ratio')
plt.legend()
plt.show()


plt.show()


print("done")