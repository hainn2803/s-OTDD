from abc import ABC, abstractmethod


class BaseManifold(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def exp(self, x, v):
        """
            Input:
            - x: samples on the manifold, of shape (n_samples, d)
            - v: directions of the geodesics in the tangent space at the origin
            of shape (n_projs, d)

            Return shape (,) TODO
        """
        pass

    @abstractmethod
    def proj(self, x, v):
        """
            Input:
            - x: samples on the manifold, of shape (n_samples, d)
            - v: directions of the geodesics in the tangent space at the origin
            of shape (n_projs, d)

            Return shape (n_projs, n_samples)
        """
        pass

    @abstractmethod
    def grad_proj(self, x, v):
        """
            Input:
            - x: samples on the manifold, of shape (n_samples, d)
            - v: directions of the geodesics in the tangent space at the origin
            of shape (n_projs, d)

            Return shape (n_projs, n_samples, d)
        """
        pass

    @abstractmethod
    def sample_geodesics(self, n_projs):
        """
            Input:
            - n_projs: Number of geodesics to sample

            Return shape (n_projs, d)
        """
        pass
