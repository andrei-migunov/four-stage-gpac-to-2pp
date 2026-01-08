import numpy as np
import plotly.graph_objects as go

# Set total bound and generate all valid simplex points
max_sum = 7
k = 3
points = []
for a in range(max_sum + 1):
    for b in range(max_sum + 1 - a):
        for c in range(max_sum + 1 - a - b):
            points.append((a, b, c))
points = np.array(points)

# Specify any number of lattice peaks here
peaks = [(3, 1, 3), (1,5,1)]  # Add more peaks as needed

# Check if a point is in the lattice defined by a peak
def in_lattice(p, peak):
    return all(p[i] <= peak[i] for i in range(3))

def child_of_peak(p,peak):
    diff = [p[i] - peak[i] for i in range(k)]
    return diff.count(0) == k-1 and diff.count(-1) == 1

# Assign colors
colors = []
for p in points:
    if any(np.array_equal(p, peak) for peak in peaks):
        colors.append('red')  # the peak itself
    elif any (child_of_peak(p, peak) for peak in peaks):
        colors.append('purple')
    elif any(in_lattice(p, peak) for peak in peaks):
        colors.append('orange')  # inside the lattice
    else:
        colors.append('blue')  # not in any lattice

# Plot using Plotly
scatter = go.Scatter3d(
    x=points[:, 0],
    y=points[:, 1],
    z=points[:, 2],
    mode='markers',
    marker=dict(size=8, color=colors, opacity=0.8)
)

layout = go.Layout(
    scene=dict(
        xaxis=dict(title='a', range=[0, 7]),
        yaxis=dict(title='b', range=[0, 7]),
        zaxis=dict(title='c', range=[0, 7]),
        aspectmode='cube'
    ),
    title='3-Simplex: Lattice Points Below Peak(s)',
    margin=dict(l=0, r=0, b=0, t=30)
)

fig = go.Figure(data=[scatter], layout=layout)
fig.show()
