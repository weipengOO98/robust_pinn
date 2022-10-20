from pyDOE import lhs
from scipy.io import loadmat
import numpy as np

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    '''
    delete points within cylinder
    '''
    dst = np.array([((xy[0] - xc) ** 2 + (xy[1] - yc) ** 2) ** 0.5 for xy in XY_c])
    return XY_c[dst > r, :]

_sol = []


def get_truth(pressure=False):
    if len(_sol) == 0:
        sol = loadmat("FluentSol.mat")
        print(sol.keys())
        _sol.append(sol)
    sol = _sol[0]

    x = sol['x'].ravel()[:, None]
    y = sol['y'].ravel()[:, None]
    u = sol['vx'].ravel()[:, None]
    v = sol['vy'].ravel()[:, None]
    p = sol['p'].ravel()[:, None]
    if not pressure:
        return x, y, u, v
    else:
        return x, y, u, v, p


def get_noise_data(N, noise_type="none", sigma=0.5, size=-1, seed=0, return_index=False):
    np.random.seed(seed)
    x, y, u, v = get_truth()

    idx = np.random.choice(x.shape[0], N, replace=False)
    x = x[idx, :]
    y = y[idx, :]
    u = u[idx, :]
    v = v[idx, :]
    u = u.ravel()
    v = v.ravel()
    sigma_u = sigma * np.std(u)
    sigma_v = sigma * np.std(v)

    if noise_type == "none":
        return x, y, u[:, None], v[:, None]
    elif noise_type == "normal":
        eps_u = np.random.randn(N) * sigma_u
        eps_v = np.random.randn(N) * sigma_v
        # print("noise_level: ", eps_u, eps_v)
        return x, y, (u + eps_u)[:, None], (v + eps_v)[:, None]
    elif noise_type == "contamined":
        eps_u = np.random.randn(N) * sigma_u
        eps_v = np.random.randn(N) * sigma_v
        idx = np.random.choice(np.arange(0, N), size=N // 5, replace=False)
        eps_u[idx] = np.random.randn(len(eps_u[idx])) * sigma_u * 10
        eps_v[idx] = np.random.randn(len(eps_v[idx])) * sigma_v * 10
        return x, y, (u + eps_u)[:, None], (v + eps_v)[:, None]
    elif noise_type == "t1":
        eps_u = np.random.standard_cauchy(N) * sigma_u
        eps_v = np.random.standard_cauchy(N) * sigma_v
        return x, y, (u + eps_u)[:, None], (v + eps_v)[:, None]
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
        if return_index:
            return x, y, u[:, None], v[:, None], idx
        else:
            return x, y, u[:, None], v[:, None]
    else:
        raise NotImplementedError(f'noise type {noise_type} not implemented.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(0)
    x, y, u, v = get_truth()
    plt.plot(x, v, 'r.')
    print(len(x))
    x, y, u, v = get_noise_data(N=19340, noise_type="normal", sigma=0.1)
    plt.plot(x, v, 'b.', markersize=1)
    plt.savefig('noise_0.1.png')
