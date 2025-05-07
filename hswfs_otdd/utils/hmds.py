import torch

from tqdm import trange


class HyperMDS():
    def __init__(self, dim, manifold, optimizer, loss="ads", scaling=1, use_pbar=False):
        """
            dim: dimension of the embedding
            manifold: manifold from geoopt
            optimizer: Euclidean optimizer
            loss: From "ads" or "contrastive"
            
            If loss=="ads", use the Absolute different squared loss from [1] to perform Hyperbolic MDS
            
            [1] Cvetkovski, Andrej, and Mark Crovella. "Multidimensional scaling in the Poincaré disk." arXiv preprint arXiv:1105.5332 (2011).
        """
        self.dim = dim
        self.manifold = manifold
        self.type_loss = loss
        self.scaling = scaling
        self.optimizer = optimizer
        self.use_pbar = use_pbar
        
        
    def fit_transform(self, X, n_epochs=100, lr=1):
        self.dissimilarity_matrix = X
        z, L = self.transform(n_epochs, lr)
        return z, L
    
    def transform(self, n_epochs=100, lr=1):
        """
            The gradient descent is performed in the tangent space following the method of 
            [1] Mishne, G., Wan, Z., Wang, Y., & Yang, S. (2023, July). The numerical stability of hyperbolic representation learning. In International Conference on Machine Learning (pp. 24925-24949). PMLR.
        """
        n = self.dissimilarity_matrix.shape[0]
        d = self.dim
        
        if self.manifold.__class__.__name__ == "Lorentz":
            x = torch.randn((n, d-1)).type(torch.float64)
        else:
            x = torch.randn((n, d)).type(torch.float64)
        x.requires_grad_(True)  

        optimizer = self.optimizer([x], lr=lr)
        
        if self.use_pbar:
            pbar = trange(n_epochs)
        else:
            pbar = range(n_epochs)
                
        L = []
        
        for e in pbar:
            optimizer.zero_grad()
            
#             norm_x = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))
            
            
            if self.manifold.__class__.__name__ == "Lorentz":
                z = self.manifold.expmap0(torch.cat([torch.zeros((x.shape[0], 1), dtype=torch.float64), x], axis=-1))
#                 norm_x = torch.sqrt(torch.sum(x**2, dim=-1, keepdim=True))
#                 z = torch.cat([torch.cosh(norm_x), torch.sinh(norm_x) * x / torch.clamp(norm_x, min=1e-6)], axis=-1)
            elif self.manifold.__class__.__name__ == "PoincareBall":
                z = self.manifold.expmap0(x)
            
            l = self.loss(z)            
            l.backward()
            optimizer.step()
            
            L.append(l.item())
                        
            if torch.isnan(l):
                return z, L

        if self.manifold.__class__.__name__ == "Lorentz":
            z = self.manifold.expmap0(torch.cat([torch.zeros((x.shape[0], 1), dtype=torch.float64), x], axis=-1))
        elif self.manifold.__class__.__name__ == "PoincareBall":
            z = self.manifold.expmap0(x)
            
        return z, L
        
    def loss(self, z):
        n = self.dissimilarity_matrix.shape[0]
        
        dist = self.manifold.dist(z[None], z[:,None])
        
        diff_dist = dist - self.scaling * self.dissimilarity_matrix
                
        if self.type_loss == "ads":
            loss = torch.sum(torch.square(torch.triu(diff_dist, diagonal=1))) * 2 / (n * (n-1))
#         elif self.type_loss == "rds":
#             loss = torch.sum(torch.square(torch.triu(diff_dist / (self.scaling * self.dissimilarity_matrix), 
#                                                      diagonal=1)))
#         elif self.type_loss == "sam":
#             num = torch.sum(torch.triu(torch.square(diff_dist) / (self.scaling * self.dissimilarity_matrix), 
#                                        diagonal=1))
#             loss = num / (self.scaling * torch.sum(torch.triu(self.dissimilarity_matrix)))

        return loss
