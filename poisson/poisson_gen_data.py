import numpy as np


def get_truth(N, full=False):
    if not full:
        xx = np.linspace(- np.pi, 0, N)
    else:
        xx = np.linspace(- np.pi, np.pi, N)
    yy = np.sin(4 * xx[:, None]) + 1.
    return xx[:, None], yy


def get_noise_data(N, noise_type="none", sigma=0.5, size=-1, seed=0):
    np.random.seed(seed)
    xx, yy = get_truth(N)
    yy = yy.ravel()
    sigma = sigma / np.sqrt(2)  # std of arcsin distribution
    if noise_type == "none":
        return xx, yy[:, None]
    elif noise_type == "normal":
        eps = np.random.randn(N) * sigma
        return xx, (yy + eps)[:, None]
    elif noise_type == "contamined":
        eps = np.random.randn(N) * sigma
        idx = np.random.choice(np.arange(0, N), size=N // 5, replace=False)
        eps[idx] = np.random.randn(len(eps[idx])) * sigma * 10
        return xx, (yy + eps)[:, None]
    elif noise_type == "t1":
        eps = np.random.standard_cauchy(size=N) * sigma
        return xx, (yy + eps)[:, None]
    elif noise_type == 'outlinear':
        eps = np.random.randn(N) * sigma
        if size < 0:
            idx = np.random.choice(np.arange(0, N), size=N // 10, replace=False)
        else:
            idx = np.random.choice(np.arange(0, N), size=size, replace=False)
        yy += eps
        yy[idx] = 10
        return xx, yy[:, None]
    else:
        raise NotImplementedError(f'noise type {noise_type} not implemented.')


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xx_n, yy_n = get_noise_data(100, noise_type="normal", sigma=0.5)
    xx_c, yy_c = get_truth(1000, full=True)

    fig, ax = plt.subplots(2, 1, figsize=(10, 10), dpi=200)
    fig.set_tight_layout(True)
    ax[0].plot(xx_c.ravel(), yy_c.ravel())
    ax[0].scatter(xx_n, yy_n, s=4)

    ax[1].plot(xx_c.ravel(), yy_c.ravel())
    ax[1].scatter(xx_n, yy_n, s=4)
    ax[1].set_ylim([-1.5, 1.5])

    plt.savefig('noise_data.png')
    plt.show()
