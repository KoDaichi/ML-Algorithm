import numpy as np
import matplotlib.pyplot as plt
import emcee
# https://qiita.com/phyblas/items/38bcff139e67a41d9e16

from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.gaussian_process import GaussianProcessRegressor as GPR


def y(x):
    return np.cos(np.pi/10*x) + x**2/400  # 実際の関数

n = 40
x0 = np.random.uniform(0, 50, n)
y0 = y(x0) + np.random.normal(0, 0.25, n)
x1 = np.linspace(0, 50, 201)

kernel = 1*RBF()+WhiteKernel()  # sklearnのカーネル
gp = GPR(kernel, alpha=0, optimizer=None)  # MCMCで最適化するので、ここではoptimizer=None
gp.fit(x0[:, None], y0)

bound = np.array([[0.001, 1000], [0.001, 1000], [0.001, 1000]])  # 下限上限
logbound = np.log(bound)

def lllh(theta):  # emceeに使う分布関数
    if np.any(theta < logbound[:, 0]) | np.any(theta > logbound[:, 1]):
        return -np.inf
    return gp.log_marginal_likelihood(theta)

nwalker = 20
ndim = len(gp.kernel.theta)
nstep = 50
theta0 = np.random.uniform(-2, 2, [nwalker, ndim])  # パラメータの初期値
sampler = emcee.EnsembleSampler(nwalker, ndim, lllh)
sampler.run_mcmc(theta0, nstep)
theta = sampler.flatchain[sampler.flatlnprobability.argmax()]  # 尤度を一番高くするパラメータ
gp.kernel.theta = theta  # 新しく得られたパラメータを設定する
gp.fit(x0[:, None], y0)  # 新しいパラメータでもう一度学習させる

# サンプリングされたパラメータの値の分布
fig = plt.figure(figsize=[6, 4.5])
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
sc = ax.scatter(*sampler.flatchain.T, alpha=0.2, s=100, c=sampler.flatlnprobability, marker='.', edgecolor='k')
plt.colorbar(sc, pad=0)
ax.scatter(*theta, s=600, c='r', edgecolor='k', marker='*')  # 最大値の位置を描く

# 近似の結果
plt.figure(figsize=[5, 4])
plt.plot(x0, y0, '. ', label='Known data')
mu, std = gp.predict(x1[:, None], return_std=True)
plt.plot(x1, y(x1), '--r', label='True function')
plt.plot(x1, mu, 'g', label='Predict function')
plt.fill_between(x1, mu-std, mu+std, alpha=0.2, color='g')
plt.tight_layout()
plt.show()

