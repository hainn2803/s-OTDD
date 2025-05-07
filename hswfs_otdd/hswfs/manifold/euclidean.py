import torch

import torch.nn.functional as F

from geoopt import linalg

from .manifold import BaseManifold


class Euclidean(BaseManifold):
    def __init__(self, d, device="cpu"):
        self.d = d
        self.device = device

    def exp(self, x, v):
        return x + v

    def proj(self, x, v):
        return (x@v.T).T

    def grad_proj(self, x, v):
        return v[:, None, :]

    def sample_geodesics(self, n_projs):
        theta = torch.randn(n_projs, self.d)
        theta = F.normalize(theta, p=2, dim=1)
        return theta.to(self.device)


class Mahalanobis(BaseManifold):
    def __init__(self, d, A):
        self.d = d
        self.A = A
        self.L = torch.linalg.cholesky(self.A)

    def exp(self, x, v):
        return x + v

    def proj(self, x, v):
        return torch.einsum("ni,ij,mj -> mn", x, self.A, v)  # (x.T@A@v).T

    def grad_proj(self, x, v):
        return v[:, None, :]

    def sample_geodesics(self, n_projs):
        """
            https://math.stackexchange.com/questions/973101/how-to-generate-points-uniformly-distributed-on-the-surface-of-an-ellipsoid
        """
        device = self.A.device

        # theta = torch.randn(n_projs, self.d).type(self.A.dtype).to(device)
        # denom = torch.einsum("ni,ij,nj -> n", theta, self.A, theta)
        # return theta / torch.sqrt(denom[:, None])

        X = torch.randn(n_projs, self.d).type(self.A.dtype).to(device)
        X = F.normalize(X, p=2, dim=1)

        Y = X @ torch.linalg.inv(self.L)
        return Y
