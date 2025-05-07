import torch

import torch.nn.functional as F
import torch.distributions as D
import numpy as np

from .manifold import BaseManifold


class Lorentz(BaseManifold):
    def __init__(self, d, K=-1., projection="geodesic", device="cpu"):
        """
            projection: "geodesic" or "horospheric"
            K: negative float
        """
        self.d = d
        self.projection = projection
        self.device = device
        self.K = K

        self.x0 = torch.zeros((1, self.d), device=device, dtype=torch.float64)
        self.x0[0, 0] = 1 / np.sqrt(-self.K)

    def exp(self, x, v):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)

        norm_v = self.minkowski_ip(v, v)**(1/2)
        sqrt_K = np.sqrt(- self.K)

        y = torch.cosh(sqrt_K * norm_v) * x + \
            torch.sinh(sqrt_K * norm_v) * v / (norm_v * sqrt_K)

        return y

    def proj(self, x, v):
        sqrt_K = np.sqrt(-self.K)

        if self.projection == "geodesic":
            ip_x0_X = self.minkowski_ip(self.x0, x)
            ip_v_X = self.minkowski_ip2(v, x)
            return torch.arctanh(-ip_v_X / (ip_x0_X * sqrt_K)).T / sqrt_K

        elif self.projection == "horospheric":
            return self.busemann(v, x).T

    def grad_proj(self, x, v):
        if self.projection == "geodesic":
            ip_x0_X = self.minkowski_ip(self.x0, x)
            ip_v_X = self.minkowski_ip2(v, x)

            num = ip_x0_X[None] * v[:, None] - \
                ip_v_X.T[:, :, None] * self.x0[None]
            denom = self.K * ip_x0_X[None]**2 + ip_v_X.T[:, :, None]**2

            nabla = self.K**2 * num / denom
            return nabla

        elif self.projection == "horospheric":
            sqrt_K = np.sqrt(-self.K)

            ip = self.minkowski_ip2(x, sqrt_K * self.x0 + v)
            cpt = (sqrt_K * self.x0 + v)[:, None] / ip[:, :, None]
            nabla = self.K * x[None] - cpt
            return self.K * sqrt_K * nabla

    def sample_geodesics(self, n_projs):
        vs = np.random.normal(size=(n_projs, self.d-1))
        vs = F.normalize(torch.from_numpy(vs), p=2, dim=1)
        vs = F.pad(vs, (1, 0))
        return vs.to(self.device)

    def minkowski_ip(self, x, y):
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        if len(y.shape) == 1:
            y = y.reshape(1, -1)

        if x.shape[0] != y.shape[0]:
            return -x[..., 0][None]*y[..., 0][:, None] + \
                torch.sum(x[..., 1:][None]*y[..., 1:][:, None], axis=-1)
        else:
            return (-x[..., 0]*y[..., 0])[:, None] + \
                torch.sum(x[..., 1:]*y[..., 1:], axis=-1, keepdim=True)

    def minkowski_ip2(self, x, y):
        return -x[:, 0][None]*y[:, 0][:, None] + \
            torch.sum(x[:, 1:][None]*y[:, 1:][:, None], axis=-1)

    def busemann(self, v, z):
        sqrt_K = np.sqrt(-self.K)
        ip = self.minkowski_ip2(v+sqrt_K*self.x0, z)
        return torch.log(-sqrt_K * ip) / sqrt_K

    def to_poincare(self, x):
        return x[..., 1:]/(1 + np.sqrt(-self.K) * x[..., 0][:, None])

    def parallel_transport(self, v, x, y):
        n, d = v.shape
        if len(x.shape) == 1:
            x = x.reshape(-1, d)
        if len(y.shape) == 1:
            y = y.reshape(-1, d)

        num = self.minkowski_ip(y, v) * (x + y)
        denom = 1 + self.K * self.minkowski_ip(x, y)

        u = v - self.K * num / denom
        return u

    def sample_wrapped_normal(self, n, mu, sigma):
        assert np.abs(self.minkowski_ip(mu, mu).item() - 1/self.K) < 1e-3, \
            "mu not on the manifold"

        normal = D.MultivariateNormal(torch.zeros((self.d-1,),
                                                  device=mu.device), sigma)

        # Sample in T_x0 L
        v_ = normal.sample((n,))
        v = F.pad(v_, (1, 0))

        # Transport to T_mu L
        u = self.parallel_transport(v, self.x0, mu)

        # Project to L
        y = self.exp(mu, u)

        return y
