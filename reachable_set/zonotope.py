import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull


class Zonotope:

    def __init__(self, center, generators):
        self.c = np.asarray(center).reshape(-1)
        self.G = np.asarray(generators)

        if self.G.ndim == 1:
            self.G = self.G.reshape(len(center),1)

    @property
    def dim(self):
        return len(self.c)

    def linear_map(self, A):
        return Zonotope(
            A @ self.c,
            A @ self.G
        )

    def minkowski_sum(self, other):

        return Zonotope(
            self.c + other.c,
            np.hstack((self.G, other.G))
        )


def reachable_sets(A,B,X0,U,N):

    reachable = [X0]

    X = X0

    for k in range(N):

        AX = X.linear_map(A)
        BU = U.linear_map(B)

        X = AX.minkowski_sum(BU)

        reachable.append(X)

    return reachable

def sample(zono, N=5000):

    p = zono.G.shape[1]

    alpha = np.random.uniform(-1,1,(p,N))

    return (zono.c[:,None] + zono.G @ alpha).T