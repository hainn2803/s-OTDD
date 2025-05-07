import torch

from .manifold import BaseManifold


class ProductManifold(BaseManifold):
    def __init__(self, manifolds, weights, projection="horospheric"):
        """
            manifolds: List of manifolds
            weights
        """

        self.manifolds = manifolds
        self.projection = projection
        self.weights = weights

    def exp(self, x, v):
        return [manifold.exp(x[i], v[i]) for i, manifold
                in enumerate(self.manifolds)]

    def proj(self, x, v):
        if self.projection == "horospheric":
            busemann = [manifold.proj(x[i], v[i])[None] * self.weights[i]
                        for i, manifold in enumerate(self.manifolds)]
            busemann_cat = torch.cat(busemann, dim=0)

            return torch.sum(busemann_cat, dim=0)

        elif self.projection == "geodesic":
            return  # TODO

    def grad_proj(self, x, v):
        if self.projection == "horospheric":
            grad_list = [manifold.grad_proj(x[i], v[i])[None]*self.weights[i]
                         for i, manifold in enumerate(self.manifolds)]
            grads = torch.cat(grad_list, dim=0)
            return torch.sum(grads, dim=0)

        elif self.projection == "geodesic":
            return  # TODO

    def sample_geodesics(self, n_projs):
        return [manifold.sample_geodesics(n_projs)
                for manifold in self.manifolds]
