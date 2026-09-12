import numpy as np
import matplotlib.pyplot as plt
from continuoussets import Zonotope

# ============================================================
# 1. System definition
# ============================================================

DOF = 2
N = 2 * DOF       # states: [x, y, vx, vy]
dt = 0.1

I_dof = np.eye(DOF)

# Continuous-time double-integrator structure
A_base = np.array([
    [0.0, 1.0],
    [0.0, 0.0]
])

B_base = np.array([
    [0.0],
    [1.0]
])

C_base = np.array([[1.0, 0.0]])

# Discrete-time approximation:
# x_{k+1} = A x_k + B u_k
A = np.eye(N) + dt * np.kron(A_base, I_dof)

# Input matrix
B = dt * np.kron(B_base, I_dof)

# Output matrix: extract x and y position
C = np.kron(C_base, I_dof)

num_states = A.shape[0]
num_inputs = B.shape[1]

print("A shape:", A.shape)
print("B shape:", B.shape)
print("C shape:", C.shape)

# Expected:
# A: (4, 4)
# B: (4, 2)
# C: (2, 4)


# ============================================================
# 2. Initial state set
# ============================================================

X0 = Zonotope(
    c=np.zeros(num_states),
    G=np.zeros((num_states, 0))
)


# ============================================================
# 3. Input set
#
# Approximate
#
#       u_x^2 + u_y^2 <= R^2
#
# using a 2-D zonotope.
# ============================================================

R = 1.0

# Number of generator directions.
# More generators -> smoother approximation.
num_generators = 20

angles = np.linspace(
    0.0,
    np.pi,
    num_generators,
    endpoint=False
)

directions = np.vstack([
    np.cos(angles),
    np.sin(angles)
])

# A zonotope is:
#
# U = { G xi : ||xi||_inf <= 1 }
#
# Scale generators so the resulting zonotope is approximately
# unit-radius rather than growing with the number of generators.
#
# For evenly distributed directions, sum |cos(theta_i)|
# gives the support-function scaling.
test_angles = np.linspace(0, 2 * np.pi, 2000)

support = np.array([
    np.sum(np.abs(np.cos(theta - angles)))
    for theta in test_angles
])

min_support = np.min(support)

generator_scale = R / min_support

G_u = generator_scale * directions

U = Zonotope(
    c=np.zeros(num_inputs),
    G=G_u
)

print("Input center shape:", U.c.shape)
print("Input generator shape:", U.G.shape)


# ============================================================
# 4. Reachability calculation
#
# X_{k+1} = A X_k (+) B U
# ============================================================

reach_steps = 4

reach_sets = [X0]

# Compute B*U once since U does not change
BU = U.matmul(B)

for k in range(reach_steps):

    X_current = reach_sets[-1]

    AX = X_current.matmul(A)

    X_next = AX.minkowski_sum(BU)

    reach_sets.append(X_next)


# ============================================================
# 5. Plot projected reachable sets
# ============================================================

fig, ax = plt.subplots(figsize=(8, 8))

for k, Z in enumerate(reach_sets):

    # Project [x, y, vx, vy] -> [x, y]
    Z_projected = Z.matmul(C)

    # Initial point
    if (
        Z_projected.G is None
        or Z_projected.G.shape[1] == 0
    ):
        ax.plot(
            Z_projected.c[0],
            Z_projected.c[1],
            "ko",
            markersize=4
        )
        continue

    try:

        verts = Z_projected.vertices()

        if verts is None:
            continue

        # continuoussets typically represents vertices as
        # dimensions x number_of_vertices
        if verts.shape[0] == 2:

            verts_closed = np.hstack([
                verts,
                verts[:, [0]]
            ])

            ax.plot(
                verts_closed[0, :],
                verts_closed[1, :],
                linewidth=1.2
            )

            ax.fill(
                verts_closed[0, :],
                verts_closed[1, :],
                alpha=0.08
            )

        # Handle alternative vertex orientation just in case
        elif verts.shape[1] == 2:

            verts_closed = np.vstack([
                verts,
                verts[[0], :]
            ])

            ax.plot(
                verts_closed[:, 0],
                verts_closed[:, 1],
                linewidth=1.2
            )

            ax.fill(
                verts_closed[:, 0],
                verts_closed[:, 1],
                alpha=0.08
            )

    except Exception as e:

        print(f"Could not plot reachable set {k}: {e}")

        ax.plot(
            Z_projected.c[0],
            Z_projected.c[1],
            "."
        )


# ============================================================
# 6. Plot formatting
# ============================================================

ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_title("Position Reachable Sets")

ax.axis("equal")
ax.grid(True)

plt.show()