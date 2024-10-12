# https://qiita.com/phyblas/items/38bcff139e67a41d9e16

import numpy as np
import matplotlib.pyplot as plt
import emcee
# from mpl_toolkits.mplot3d import Axes3D


def fn(xy):
    x, y = xy
    return np.maximum(0, (10-np.abs(2+x)-np.abs(y-1)))**2


def lnfn(xy):
    x, y = xy
    prop = np.maximum(1e-10, (10-np.abs(2+x)-np.abs(y-1)))**2
    return np.log(prop)


mx, my = np.meshgrid(np.linspace(-10, 10, 41), np.linspace(-10, 10, 41))
mz = fn([mx, my])
ax = plt.figure(figsize=[8, 8]).add_axes([0, 0, 1, 1], projection='3d')
ax.plot_surface(mx, my, mz, rstride=1, cstride=1, alpha=0.2, edgecolor='k', cmap='rainbow')
plt.show()

ndim = 2  # 鎖の数
nwalker = 100  # 次元の数
nstep = 4000  # 鎖の長さ
xy0 = np.random.uniform(-4, 4, [nwalker, ndim])  # 初期のxとy
sampler = emcee.EnsembleSampler(nwalker, ndim, lnfn)  # サンプラーを作る
sampler.run_mcmc(xy0, nstep)  # サンプリング開始

xy = sampler.flatchain  # サンプリングでできた結果を取得
x, y = xy[:, 0], xy[:, 1]

plt.figure()
plt.plot(sampler.chain[:, :, 0].mean(0).T)
plt.show()

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

bunpuplot(x,y)

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
plt.gca(aspect=1)
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

plt.ylabel(u'最大値', fontname='AppleGothic', size=16)
plt.xlabel(u'鎖の長さ', fontname='AppleGothic', size=16)
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

