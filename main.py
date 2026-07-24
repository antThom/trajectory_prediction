import numpy as np
import matplotlib.pyplot as plt
from reachable_set.zonotope import Zonotope, reachable_sets
dt = 0.1

A = np.array([
    [1, dt],
    [0, 1]
])

B = np.array([
    [0],
    [dt]
])

X0 = Zonotope(
    center=np.array([0,0]),
    generators=np.zeros((2,0))
)

U = Zonotope(
    center=np.array([0]),
    generators=np.array([[1]])
)

reach = reachable_sets(A,B,X0,U,40)

plt.figure(figsize=(8,8))

for Z in reach:

    pts = sample(Z,3000)

    hull = ConvexHull(pts)

    for simplex in hull.simplices:
        plt.plot(pts[simplex,0],
                 pts[simplex,1],
                 'b',
                 linewidth=0.6)

plt.xlabel("Position")
plt.ylabel("Velocity")
plt.axis("equal")
plt.grid(True)
plt.show()