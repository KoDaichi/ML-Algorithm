# https://qiita.com/phyblas/items/38bcff139e67a41d9e16

import numpy as np
import matplotlib.pyplot as plt
import emcee
from matplotlib import rcParams
# from mpl_toolkits.mplot3d import Axes3D

# 日本語フォントを設定
rcParams['font.family'] = 'IPAGothic'  # IPAフォントがインストールされている場合

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
    return np.exp(-(5**2-(x**2+y**2))**2/200 + xy[1]/20) * (6./5+np.sin(6*np.arctan2(x,y)))


def lnfn(xy):
    return np.log(max(1e-10, fn(xy)))


ndim = 2
nwalker = 6
xy0 = np.random.uniform(-5, 5, [nwalker, ndim])
sampler = emcee.EnsembleSampler(nwalker, ndim, lnfn)
sampler.run_mcmc(xy0, 4000)

xy = sampler.flatchain
x, y = xy[:, 0], xy[:, 1]
x_max, y_max = sampler.flatchain[sampler.flatlnprobability.argmax()]
plt.figure(figsize=[7, 6])
plt.subplot()
plt.scatter(x, y, alpha=0.1, c=sampler.flatlnprobability, marker='.', cmap='rainbow')
plt.colorbar()
plt.scatter(x_max, y_max, c='k')  # 最大値の位置を描く
plt.show()


saidaichi = np.empty(4000)
lnprobmax = -np.inf
for i, a in enumerate(sampler.lnprobability.max(0)):
    if a > lnprobmax:
        lnprobmax = a
    saidaichi[i] = lnprobmax

plt.ylabel(u'最大値', size=16)
plt.xlabel(u'鎖の長さ', size=16)
plt.plot(np.exp(saidaichi))
plt.loglog()
plt.show()


def lnfn(x):
    if np.any(x**2 > 1):
        return -np.inf
    p = (x[0]*np.sin(x[0]*3))**2*(1-x[1]**2)*(1-np.abs(x[2])-np.abs(x[3])/2-x[3]**2/2)
    return np.log(max(1e-10, p))


ndim = 4
nwalker = 20
nstep = 6000
xy0 = np.random.uniform(-0.5, 0.5, [nwalker, ndim])
sampler = emcee.EnsembleSampler(nwalker, ndim, lnfn)
sampler.run_mcmc(xy0, nstep)
xy = sampler.flatchain
x, y = xy[:, 0], xy[:, 1]

plt.figure(figsize=[15, 15])
for i in range(ndim):
    for j in range(i+1):
        plt.subplot(ndim, ndim, 1+i*ndim+j)
        if i == j:
            plt.hist(sampler.flatchain[:, i], 50, color='#BB3300')
        else:
            plt.hist2d(sampler.flatchain[:, j], sampler.flatchain[:, i], bins=50, cmap='coolwarm')
plt.show()

