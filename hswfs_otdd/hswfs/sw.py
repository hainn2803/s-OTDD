import torch


def emd1D(u_values, v_values, u_weights=None, v_weights=None,
          p=2, require_sort=True):
    n = u_values.shape[-1]
    m = v_values.shape[-1]

    device = u_values.device
    dtype = u_values.dtype

    if u_weights is None:
        u_weights = torch.full((n,), 1/n, dtype=dtype, device=device)

    if v_weights is None:
        v_weights = torch.full((m,), 1/m, dtype=dtype, device=device)

    if require_sort:
        u_values, u_sorter = torch.sort(u_values, -1)
        v_values, v_sorter = torch.sort(v_values, -1)

        u_weights = u_weights[..., u_sorter]
        v_weights = v_weights[..., v_sorter]

    u_cdf = torch.cumsum(u_weights, -1)
    v_cdf = torch.cumsum(v_weights, -1)

    cdf_axis, _ = torch.sort(torch.cat((u_cdf, v_cdf), -1), -1)

    u_index = torch.searchsorted(u_cdf, cdf_axis)
    v_index = torch.searchsorted(v_cdf, cdf_axis)

    u_icdf = torch.gather(u_values, -1, u_index.clip(0, n-1))
    v_icdf = torch.gather(v_values, -1, v_index.clip(0, m-1))

    cdf_axis = torch.nn.functional.pad(cdf_axis, (1, 0))
    delta = cdf_axis[..., 1:] - cdf_axis[..., :-1]

    if p == 1:
        return torch.sum(delta * torch.abs(u_icdf - v_icdf), axis=-1)
    if p == 2:
        return torch.sum(delta * torch.square(u_icdf - v_icdf), axis=-1)  
    return torch.sum(delta * torch.pow(torch.abs(u_icdf - v_icdf), p), axis=-1)


def sliced_wasserstein(Xs, Xt, num_projections, manifold,
                       u_weights=None, v_weights=None, p=2):

    if torch.is_tensor(Xs):
        device = Xs.device
        dtype = Xs.dtype

        # Random projection directions, shape (num_features, num_projections)
        v = manifold.sample_geodesics(num_projections).type(dtype).to(device)
    else:
        v = manifold.sample_geodesics(num_projections)

    Xps = manifold.proj(Xs, v)
    Xpt = manifold.proj(Xt, v)

    w = emd1D(Xps, Xpt, u_weights=u_weights, v_weights=v_weights, p=p)

    return w


def get_quantiles(x, ts, weights=None):
    """
        Inputs:
        - x: 1D values, size: n_projs * n_batch
        - ts: points at which to evaluate the quantile
    """
    n_projs, n_batch = x.shape

    if weights is None:
        X_weights = torch.full(
            (n_batch,), 1/n_batch, dtype=x.dtype, device=x.device
        )
        X_values, X_sorter = torch.sort(x, -1)
        X_weights = X_weights[..., X_sorter]

    X_cdf = torch.cumsum(X_weights, -1)

    X_index = torch.searchsorted(X_cdf, ts.repeat(n_projs, 1))
    X_icdf = torch.gather(X_values, -1, X_index.clip(0, n_batch-1))

    return X_icdf


def get_features(x, num_projs, manifold, weights=None, ts=None, p=2):
    """
        Inputs:
        - x: ndarray, shape (n_batch, d)
        - num_projs
        - manifold
        - weights: weight of each sample, if None, uniform weights
        - ts: points at which to evaluate the quantile function
        - p: order of sw, default = 2
    """
    device = x.device
    dtype = x.dtype

    if ts is None:
        ts = torch.linspace(0, 1, 100, dtype=dtype, device=device)

    num_unifs = len(ts)

    v = manifold.sample_geodesics(num_projs).type(dtype).to(device)
    Xp = manifold.proj(x, v)

    q_Xp = get_quantiles(Xp, ts, weights)

    return q_Xp / ((num_projs * num_unifs) ** (1 / p))
