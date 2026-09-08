import numpy as np
import matplotlib.pyplot as plt
from continuoussets import Zonotope

# 1. System Dimension & Scaling Definitions
N  = 2 # Number of states
M  = 1 # Number of inputs
P  = N//2 # Number of outpus
I = np.eye(N)
dt = 0.1

A_base = np.array([[0,1],[0,0]])
A = np.kron(A_base, I)*dt + np.eye(N*2)

B_base = np.array([[0],[1]])
B = np.kron(B_base, I)

M = B.shape[1]
numStates = A.shape[1]

C_base = np.array([1,0])
C = np.kron(C_base, I)

# 3. Define the Point Initial State Set (X0)
# Center at origin, empty generator matrix (0 generators)
X0 = Zonotope(c=np.zeros(numStates), G=np.zeros((numStates, 0)))

# 4. IMPLEMENT THE COMBINED ENERGY CONSTRAINT (U)
# To map the energy limits (u_1^2 + u_2^2 <= R^2), we distribute generators 
# radially to form a tight, multi-faceted regular polygon over-approximation.
R = 1.0            # Target Combined Energy Radius
num_facets = 10    # Increase for a smoother circle approximation
angles = np.linspace(0, np.pi, num_facets, endpoint=False)

# Stack radial coordinates and scale to match the uniform norm bounds
circle_generators = np.vstack([np.cos(angles), np.sin(angles)]) * (R * np.sqrt(2 / num_facets))

# Construct the combined input space zonotope using mandatory c and G parameters
U = Zonotope(c=np.zeros(M), G=circle_generators)

# 5. Reachability Parametrization & Iterative Propagation
reach_steps = 3
reach_sets = [X0]

# Linearly propagate the exact structural geometry through the matrix steps:
# X_next = A * X_current + B * U
for k in range(reach_steps):
    X_current = reach_sets[-1]
    
    # continuoussets natively overrides the @ operator for linear maps 
    # and the + operator for Minkowski sum calculations
    AX = X_current.matmul(A)
    BU = U.matmul(B)
    X_next = AX.minkowski_sum(BU)
    reach_sets.append(X_next)

# 6. Plotting the Exact Projected Subspace
fig, ax = plt.subplots(figsize=(8, 8))

for Z in reach_sets:
    # Analytically project the high-dimensional states down to the output workspace subspace (C)
    # The framework warps the generator bounds directly, without any numerical point sampling
    Z_projected = Z.matmul(C)
    
    # Extract the boundary vertices of the projected zonotope to plot cleanly
    verts = Z_projected.vertices()

    # Close the loop on the polyhedral hull shape for rendering
    if verts.shape[1] > 0:
        # Re-append the first vertex to the end to close the polygon stroke path
        verts_closed = np.hstack([verts, verts[:, [0]]])
        ax.plot(verts_closed[0, :], verts_closed[1, :], 'b-', linewidth=1.5)
        ax.fill(verts_closed[0, :], verts_closed[1, :], 'b', alpha=0.15)

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.axis("equal")
ax.grid(True)
plt.show()