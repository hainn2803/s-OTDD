import torch

import torch.nn.functional as F
import numpy as np

from geoopt import linalg

from .manifold import BaseManifold


class LogEuclidean(BaseManifold):
    def __init__(self, d, device="cpu"):
        """
        """
        self.d = d
        self.device = device

    def exp(self, x, v):
        if len(v.shape) == 2:
            v = v.reshape(v.shape[0], self.d, self.d)

        log_x = linalg.sym_logm(x)
        diff_x = self.diff_x2(x, v)
        return linalg.sym_expm(log_x + diff_x)

    def proj(self, x, v):
        n, _, _ = x.shape
        n_proj, _, _ = v.shape

        log_x = linalg.sym_logm(x)
        prod_x = (v[None] * log_x[:, None]).reshape(n, n_proj, -1)
        return prod_x.sum(-1).T

    def grad_proj(self, x, v):
        grad_euc = self.diff_x(x, v)
        out = self.inv_diff_x(x, self.inv_diff_x(x, grad_euc))
        return out.reshape(v.shape[0], x.shape[0], self.d*self.d)

    def sample_geodesics(self, n_projs):
        theta = np.random.normal(size=(n_projs, self.d))
        theta = F.normalize(
            torch.from_numpy(theta), p=2, dim=-1
        ).to(self.device)

        D = theta[:, None] * torch.eye(
            theta.shape[-1],
            device=self.device
        )

        # Random orthogonal matrices
        Z = np.random.normal(size=(n_projs, self.d, self.d))
        Z = torch.tensor(
            Z,
            device=self.device
        )
        Q, R = torch.linalg.qr(Z)
        lambd = torch.diagonal(R, dim1=-2, dim2=-1)
        lambd = lambd / torch.abs(lambd)
        P = lambd[:, None] * Q

        A = torch.matmul(
            P,
            torch.matmul(D, torch.transpose(P, -2, -1))
        )

        return A

    def diff_x(self, x, v):
        D, U = torch.linalg.eigh(x, "U")
        S = self.sigma(D, U, v)
        return torch.einsum("nij, nmjk, nlk -> nmil", U, S, U)  # U @ S @ U.T

    def diff_x2(self, x, v):
        D, U = torch.linalg.eigh(x, "U")
        M = torch.einsum("nij, nik, nkl -> njl", U, v, U)  # U.T @ V @ U
        G = self.get_gamma(D)
        S = M * G
        return torch.einsum("nij, njk, nlk -> nil", U, S, U)  # U @ S @ U.T

    def sigma(self, D, U, V):
        M = torch.einsum("nij, mik, nkl -> nmjl", U, V, U)  # U.T@V@U
        G = self.get_gamma(D)
        S = M * G[:, None]
        return S

    def inv_diff_x(self, x, v):
        D, U = torch.linalg.eigh(x, "U")
        G = self.get_gamma(D)

        # U.T @ V @ U
        M = (torch.einsum("nji, nmjk, nlk -> nmil", U, v, U)) / G[:, None]

        return torch.einsum("nij, nmjk, nlk -> nmil", U, M, U)  # U @ V @ U.T

    def get_gamma(self, D):
        log_D = torch.log(D)[:, None]-torch.log(D)[:, :, None]
        log_D /= (D[:, None] - D[:, :, None])

        i = torch.arange(self.d)
        log_D[:, i, i] = 0

        D_repeat = 1 / D[:, :, None].expand(-1, -1, D.shape[1])
        log_D = torch.where(torch.isnan(log_D), D_repeat, log_D)
        # log_D = torch.nan_to_num(log_D, 0)
        G = log_D + torch.diag_embed(1/D)

        return G


class SPD_Euclidean(BaseManifold):
    def __init__(self, d, device="cpu"):
        """
        """
        self.d = d
        self.device = device

    def exp(self, x, v):
        if len(v.shape) == 2:
            v = v.reshape(v.shape[0], self.d, self.d)
        return x + v

    def proj(self, x, v):
        n, _, _ = x.shape
        n_proj, _, _ = v.shape

        prod_x = (v[None] * x[:, None]).reshape(n, n_proj, -1)
        return prod_x.sum(-1).T

    def grad_proj(self, x, v):
        return v.reshape(v.shape[0], 1, self.d * self.d)

    def sample_geodesics(self, n_projs):
        theta = torch.randn(size=(n_projs, self.d, self.d))
        A = theta / torch.norm(theta, dim=(1, 2), keepdim=True)

        return A
