import torch

import torch.nn.functional as F
import numpy as np

from .manifold import BaseManifold
from .lorentz import Lorentz


class Poincare(BaseManifold):
    def __init__(self, d, K=-1., projection="horospheric", device="cpu"):
        """
            projection: "geodesic" or "horospheric"
            K: negative float
        """
        self.d = d
        self.K = K
        self.projection = projection
        self.device = device

    def exp(self, x, v):
        sqrt_K = np.sqrt(-self.K)

        lx = self.lambd(x)
        norm_v = torch.norm(v, dim=-1, keepdim=True)
        normalized_v = v / torch.clamp(norm_v, min=1e-6)
        th = torch.tanh(sqrt_K * lx * norm_v / 2)
        y = th * normalized_v / sqrt_K

        ip_xy = torch.sum(x * y, dim=-1, keepdim=True)
        norm_y = torch.norm(y, dim=-1, keepdim=True)
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        num = (1-2*self.K*ip_xy-self.K*norm_y**2)*x + (1+self.K*norm_x**2)*y
        denom = 1 - 2*self.K*ip_xy + self.K**2 * norm_x**2 * norm_y**2

        return num / denom

    def proj(self, x, v):
        if self.projection == "geodesic":
            norm_x = torch.norm(x, dim=-1, keepdim=True)
            ip_xv = torch.sum(x[None] * v[:, None], dim=-1, keepdim=True)
            sqrt_K = np.sqrt(-self.K)

            sqrt = torch.sqrt((1-self.K*norm_x**2)**2 + 4*self.K*ip_xv**2)
            num = 1 - self.K * norm_x**2 - sqrt
            denom = -2*self.K*ip_xv

            s_x = (num / denom)[:, :, 0]

            return 2 * torch.arctanh(sqrt_K * s_x) / sqrt_K

        elif self.projection == "horospheric":
            return self.busemann(v, x).T

    def grad_proj(self, x, v):
        if self.projection == "geodesic":
            return  # TODO

        elif self.projection == "horospheric":
            norm_x2 = torch.sum(x**2, dim=-1)

            diff = v[:, None] - x[None]
            norm_diff2 = torch.sum(diff**2, dim=-1)

            cpt1 = x / (1-norm_x2[:, None])
            cpt2 = diff / norm_diff2[:, :, None]

            nabla_B = 2 * (cpt1[None] - cpt2)
            nabla = ((1 + self.K * norm_x2[None, :, None]) / 2)**2 * nabla_B

            return nabla

    def sample_geodesics(self, n_projs):
        ps = np.random.normal(size=(n_projs, self.d))
        ps = F.normalize(torch.from_numpy(ps), p=2, dim=-1)
        return ps.to(self.device)

    def busemann(self, v, z):
        sqrt_K = np.sqrt(-self.K)
        norm_z = torch.norm(z, dim=-1, keepdim=True)**2
        num = torch.norm(v[None] - sqrt_K * z[:, None], dim=-1)**2
        log_term = torch.clamp(num/(1 + self.K * norm_z), min=1e-10)
        return torch.log(log_term) / sqrt_K

    def lambd(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        return 2/(1+self.K * norm_x**2)

    def to_lorentz(self, x):
        norm_x = torch.norm(x, dim=-1, keepdim=True)
        x0 = (1-self.K*norm_x**2) / np.sqrt(-self.K)
        return torch.cat([x0, 2*x], dim=-1) / (1+self.K*norm_x**2)

    def sample_wrapped_normal(self, n, mu, sigma):
        lorentz = Lorentz(self.d+1, self.K)

        mu_lorentz = self.to_lorentz(mu)
        x_lorentz = lorentz.sample_wrapped_normal(n, mu_lorentz, sigma)
        x_poincare = lorentz.to_poincare(x_lorentz)

        return x_poincare
