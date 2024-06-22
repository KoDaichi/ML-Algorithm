import numpy as np
import matplotlib.pyplot as plt


# === マルコフ連鎖モンテカルロ法（MCMC）の実装 ===
def fn(xy):
    x, y = xy
    return np.exp(-(5**2-(x**2+y**2))**2/250 + xy[1]/10) * (7./4-np.sin(7*np.arctan2(x, y)))

plt.figure(figsize=[6, 6])
mx, my = np.meshgrid(np.linspace(-10, 10, 101), np.linspace(-10, 10, 101))
mz = fn([mx, my])
ax = plt.axes([0, 0, 1, 1], projection='3d')
ax.plot_surface(mx, my, mz, rstride=2, cstride=2, alpha=0.2, edgecolor='k', cmap='rainbow')
plt.show()

xy0 = np.array([3, -3])  # 開始の位置
bound = np.array([[-6, 6], [-6, 6]])  # 下限上限
s = (bound[:, 1]-bound[:, 0])/10.  # 毎回どれくらい遠く移動するか
n = 16000  # 何度繰り返すか
xy = []  # 毎回の位置を格納するリスト
p = []  # 毎回の確率を格納するリスト
p0 = fn(xy0)  # 開始の位置の確率
for i in range(n):
    idou = np.random.normal(0, s, 2)  # 移動する距離
    hazure = (xy0 + idou < bound[:, 0]) | (xy0 + idou > bound[:, 1])  # 下限上限から外れたか
    while np.any(hazure):
        idou[hazure] = np.random.normal(0, s, 2)[hazure]  # 外れたものだけもう一度ランダムする
        hazure = (xy0 + idou < bound[:, 0]) | (xy0 + idou > bound[:, 1])
    xy1 = xy0 + idou  # 新しい位置の候補
    p1 = fn(xy1)  # 新しい位置の確率
    r = p1 / p0  # 新しい位置と現在の位置の確率の比率
    # 比率は1より高い場合は常に移動するが、低い場合は確率で移動する (メトロポリス, SAでも使われている)
    if (r > 1 or r > np.random.random()):
        xy0 = xy1  # 現在の位置を新しい位置に移動する
        p0 = p1
        xy.append(xy0)  # 新しい位置を格納
        p.append(p0)  # 新しい確率を格納

xy = np.stack(xy)
x, y = xy[:, 0], xy[:, 1]
plt.figure(figsize=[7, 6])
plt.axes(aspect=1)
plt.scatter(x, y, c=p, alpha=0.1, edgecolor='k')
plt.colorbar(pad=0.01)
plt.scatter(*xy[np.argmax(p)], s=150, c='r', marker='*', edgecolor='k')  # 最大値を星で示す
plt.tight_layout()
plt.show()


# === ガウス過程のパラメータの調整 ===
class Kernel:
    def __init__(self, param, bound=None):
        self.param = np.array(param)
        if (bound==None):
            bound = np.zeros([len(param), 2])
            bound[:, 1] = np.inf
        self.bound = np.array(bound)

    def __call__(self, x1, x2):
        a, s, w = self.param
        return a**2*np.exp(-0.5*((x1-x2)/s)**2) + w**2*(x1==x2)

class Gausskatei:
    def __init__(self, kernel):
        self.kernel = kernel

    def gakushuu(self, x0, y0): # パラメータを調整せず学習
        self.x0 = x0
        self.y0 = y0
        self.k00 = self.kernel(*np.meshgrid(x0, x0))
        self.k00_1 = np.linalg.inv(self.k00)

    def yosoku(self, x): # xからyを予測
        k00_1 = self.k00_1
        k01 = self.kernel(*np.meshgrid(self.x0, x, indexing='ij'))
        k10 = k01.T
        k11 = self.kernel(*np.meshgrid(x, x))

        mu = k10.dot(k00_1.dot(self.y0))
        sigma = k11 - k10.dot(k00_1.dot(k01))
        std = np.sqrt(sigma.diagonal())
        return mu, std

    def logyuudo(self, param=None): # 対数尤度
        if (param is None):
            k00 = self.k00
            k00_1 = self.k00_1
        else:
            self.kernel.param = param
            k00 = self.kernel(*np.meshgrid(self.x0, self.x0))
            k00_1 = np.linalg.inv(k00)
        return -(np.linalg.slogdet(k00)[1]+self.y0.dot(k00_1.dot(self.y0)))

    def saitekika(self, x0, y0, kurikaeshi=1000): # パラメータを調整して学習
        self.x0 = x0
        self.y0 = y0
        param = self.kernel.param
        logbound = np.log(self.kernel.bound)
        s = (logbound[:, 1]-logbound[:, 0])/10.
        n_param = len(param)
        theta0 = np.log(param)
        p0 = self.logyuudo(param)
        lis_theta = []
        lis_p = []
        for i in range(kurikaeshi):
            idou = np.random.normal(0, s, n_param)
            hazure = (theta0 + idou < logbound[:, 0]) | (theta0 + idou > logbound[:, 1])
            while np.any(hazure):
                idou[hazure] = np.random.normal(0, s, n_param)[hazure]
                hazure = (theta0 + idou < logbound[:, 0]) | (theta0 + idou > logbound[:, 1])
            theta1 = theta0 + idou
            param = np.exp(theta1)
            p1 = self.logyuudo(param)
            r = np.exp(p1-p0)
            if (r >= 1 or r > np.random.random()):
                theta0 = theta1
                p0 = p1
                lis_theta.append(theta0)
                lis_p.append(p0)
        self.ar_theta = np.array(lis_theta)
        self.ar_p = np.array(lis_p)
        self.kernel.param = np.exp(lis_theta[np.argmax(lis_p)])
        self.k00 = self.kernel(*np.meshgrid(x0, x0))
        self.k00_1 = np.linalg.inv(self.k00)


def y(x): # 実際の関数
    return 5*np.sin(np.pi/15*x)*np.exp(-x/50)

n = 100  # 既知の点の数
x0 = np.random.uniform(0, 100, n)  # 既知の点
y0 = y(x0) + np.random.normal(0, 1, n)
param0 = [3, 0.6, 0.5]  # パラメータの初期値
bound = [[1e-2, 1e2], [1e-2, 1e2], [1e-2, 1e2]]  # 下限上限
kernel = Kernel(param0, bound)
x1 = np.linspace(0, 100, 200)
gp = Gausskatei(kernel)
gp.gakushuu(x0, y0)  # パラメータを調整せずに学習
plt.figure(figsize=[5, 8])
for i in [0, 1]:
    if i:
        gp.saitekika(x0, y0, 10000)  # パラメータを調整する
    plt.subplot(211+i)
    plt.plot(x0, y0, '. ', label='Known data')
    mu, std = gp.yosoku(x1)
    plt.plot(x1, y(x1), '--r', label='True function')
    plt.plot(x1, mu, 'g', label='Predict mean')
    plt.legend()
    plt.fill_between(x1, mu-std, mu+std, alpha=0.2, color='g')
    plt.title('a=%.3f,  s=%.3f,  w=%.3f' % tuple(gp.kernel.param))
plt.tight_layout()
plt.show()


fig = plt.figure(figsize=[6, 5])
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
sz = np.ones(len(gp.ar_p))*50
sz[gp.ar_p.argmax()] = 300  # 尤度の一番高いパラメータを大きく表示する
sc = ax.scatter(*gp.ar_theta.T, c=gp.ar_p, s=sz, alpha=0.7, edgecolor='k')
fig.colorbar(sc, pad=0)
plt.show()


# === カーネルから生成されたノイズのパラメータの再現 ===
n = 100
x0 = np.random.uniform(0, 100, n)
param = [20, 5, 3]  # 実際のパラメータ
sigma = Kernel(param)(*np.meshgrid(x0, x0))
param0 = [4, 0.4, 10]  # パラメータの初期値
bound = [[1e-2, 1e2], [1e-2, 1e2], [1e-2, 1e2]]
y0 = np.random.multivariate_normal(np.zeros(n), sigma)
kernel = Kernel(param0, bound)
x1 = np.linspace(0, 100, 200)
gp = Gausskatei(kernel)
plt.figure(figsize=[5, 4])
gp.saitekika(x0, y0, 10000)
plt.plot(x0, y0, '. ')
mu, std = gp.yosoku(x1)
plt.plot(x1, mu, 'g')
plt.fill_between(x1, mu-std, mu+std, alpha=0.2, color='g')
plt.tight_layout()

fig = plt.figure(figsize=[6, 5])
ax = fig.add_axes([0, 0, 1, 1], projection='3d')
sc = ax.scatter(*gp.ar_theta.T, c=gp.ar_p, s=30, alpha=0.3, edgecolor='k')
fig.colorbar(sc, pad=0)
ax.scatter(*np.log(param), c='r', edgecolor='k', marker='v', s=300)  # 実際のパラメータを三角で示す
ax.scatter(*np.log(gp.kernel.param), c='b', edgecolor='k', marker='*', s=300)  # 尤度の一番高いパラメータを星で示す

plt.show()
print('最適化の後のパラメータ: %s\n実際のパラメータ: %s'%(gp.kernel.param, param))

