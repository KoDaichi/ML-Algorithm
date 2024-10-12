# https://qiita.com/phyblas/items/38bcff139e67a41d9e16

import numpy as np
import matplotlib.pyplot as plt
import emcee
# from mpl_toolkits.mplot3d import Axes3D


# 分布を描く関数
def bunpuplot(x, y):
    plt.figure(figsize=[8, 8])
    plt.subplot(221, aspect=1)
    plt.scatter(x, y, alpha=0.002, marker='.')
    plt.subplot(222)
    plt.hist(y, bins=100, orientation='horizontal')
    plt.subplot(223)
    plt.hist(x, bins=100)
    plt.subplot(224, aspect=1)
    plt.hist2d(x, y, bins=50, cmap='rainbow')
    plt.colorbar()
    plt.tight_layout()
    plt.show()

# +-------------+-------------+
# | Subplot 1   | Subplot 2   |
# | (221)       | (222)       |
# +-------------+-------------+
# | Subplot 3   | Subplot 4   |
# | (223)       | (224)       |
# +-------------+-------------+


def fn(xy):
    x, y = xy
    xy2 = x**2 + y**2

    # # オーバーフローとゼロ除算を防ぐためのチェック
    # if np.any(xy2 == 0):
    #     return -np.inf  # log-probabilityが無限小の場合
    # if np.any(xy2 > 1e10):  # 例として、非常に大きな値を制限
    #     return -np.inf
    #
    # result = np.sin(xy2)**2 / np.sqrt(xy2)
    # if not np.isfinite(result):  # 無効な値が発生した場合のチェック
    #     return -np.inf
    #
    # return result
    return np.sin(xy2)**2 / xy2**0.5


def lnfn(xy, b):
    if np.any(xy < b[0]) | np.any(xy > b[1]):
        return -np.inf
    x, y = xy
    xy2 = x**2+y**2
    return np.log(np.sin(xy2/2)**2/xy2**0.5)


mx, my = np.meshgrid(np.linspace(-10, 10, 41), np.linspace(-10, 10, 41))
mz = fn([mx, my])
ax = plt.figure(figsize=[8, 8]).add_axes([0, 0, 1, 1], projection='3d')
ax.plot_surface(mx, my, mz, rstride=1, cstride=1, alpha=0.2, edgecolor='k', cmap='rainbow')
plt.show()


b = np.array([[-4, -2], [3, 5]])
ndim = 2
nwalker = 20
nstep = 10000
xy0 = np.random.uniform(-1, 1, [nwalker, ndim])
sampler = emcee.EnsembleSampler(nwalker, ndim, lnfn, args=[b])
sampler.run_mcmc(xy0, nstep)
xy = sampler.flatchain
x, y = xy[:, 0], xy[:, 1]
bunpuplot(x, y)


