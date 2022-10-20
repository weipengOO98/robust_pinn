import numpy as np
from scipy import io


def get_truth():
    data = io.loadmat('cylinder_nektar_wake.mat')

    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data 
    XX = np.tile(X_star[:, 0:1], (1, T))  # N x T
    YY = np.tile(X_star[:, 1:2], (1, T))  # N x T
    TT = np.tile(t_star, (1, N)).T  # N x T

    UU = U_star[:, 0, :]  # N x T
    VV = U_star[:, 1, :]  # N x T
    PP = P_star  # N x T

    x = XX.flatten()[:, None]  # NT x 1
    y = YY.flatten()[:, None]  # NT x 1
    t = TT.flatten()[:, None]  # NT x 1

    u = UU.flatten()[:, None]  # NT x 1
    v = VV.flatten()[:, None]  # NT x 1
    p = PP.flatten()[:, None]  # NT x 1
    return x, y, t, u, v, p


def get_noise_data(N, noise_type="none", sigma=0.5, size=-1, seed=0):
    np.random.seed(0)
    x, y, t, u, v, p = get_truth()

    idx = np.random.choice(x.shape[0], N, replace=False)
    x = x[idx, :]
    y = y[idx, :]
    t = t[idx, :]
    u = u[idx, :]
    v = v[idx, :]
    p = p[idx, :]
    u = u.ravel()
    v = v.ravel()
    sigma_u = sigma * np.std(u)
    sigma_v = sigma * np.std(v)
    if noise_type == "none":
        return x, y, t, u[:, None], v[:, None], p
    elif noise_type == "normal":
        eps_u = np.random.randn(N) * sigma_u
        eps_v = np.random.randn(N) * sigma_v
        return x, y, t, (u + eps_u)[:, None], (v + eps_v)[:, None], p
    elif noise_type == "contamined":
        eps_u = np.random.randn(N) * sigma_u
        eps_v = np.random.randn(N) * sigma_v
        idx = np.random.choice(np.arange(0, N), size=N // 5, replace=False)
        eps_u[idx] = np.random.randn(len(eps_u[idx])) * sigma_u * 10
        eps_v[idx] = np.random.randn(len(eps_v[idx])) * sigma_v * 10
        return x, y, t, (u + eps_u)[:, None], (v + eps_v)[:, None], p
    elif noise_type == "t1":
        eps_u = np.random.standard_cauchy(N) * sigma_u
        eps_v = np.random.standard_cauchy(N) * sigma_v
        return x, y, t, (u + eps_u)[:, None], (v + eps_v)[:, None], p
    elif noise_type == 'outlinear':
        eps_u = np.random.randn(N) * sigma_u
        eps_v = np.random.randn(N) * sigma_v
        if size < 0:
            idx = np.random.choice(np.arange(0, N), size=N // 10, replace=False)
        else:
            idx = np.random.choice(np.arange(0, N), size=size, replace=False)
        u += eps_u
        v += eps_v
        u[idx] = 30
        v[idx] = 30
        return x, y, t, u[:, None], v[:, None], p
    else:
        raise NotImplementedError(f'noise type {noise_type} not implemented.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    for ti in [20, 25, 30, 35, 40]:
        x, y, t, u, v, p = get_truth()
        print(f"x_min: {x.min()}, y_min: {y.min()}, t_min: {t.min()}")
        print(f"x_max: {x.max()}, y_max: {y.max()}, t_max: {t.max()}")
        x = x[ti::200, 0].reshape(50, 100)
        y = y[ti::200, 0].reshape(50, 100)
        u = u[ti::200, 0].reshape(50, 100)
        v = v[ti::200, 0].reshape(50, 100)
        p = p[ti::200, 0].reshape(50, 100)
        fig, ax = plt.subplots(3, 1, figsize=(10, 10), dpi=200)
        fig.set_tight_layout(True)
        print(u.min(), u.max())
        ax[0].pcolormesh(x, y, u, vmin=-0.3, vmax=1.4)
        print(v.min(), v.max())
        ax[1].pcolormesh(x, y, v, vmin=-0.7, vmax=0.7)
        ax[2].pcolormesh(x, y, p, vmin=-0.6, vmax=0.1)
        print(p.min(), p.max())
        plt.savefig(f'noise_data_{ti}.png')
        plt.show()
