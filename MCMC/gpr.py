# https://qiita.com/phyblas/items/d756803ec932ab621c56

import numpy as np
import matplotlib.pyplot as plt
import imageio


# === ガウス過程の実装 ===
class RBFkernel:
    def __init__(self, *param):
        self.param = list(param)

    def __call__(self, x1, x2):
        a, s, w = self.param
        return a**2*np.exp(-((x1-x2) / s)**2) + w*(x1 == x2)


def y(x):  # 知りたい関数の正体
    return 0.1*x**3-x**2+2*x+5


x0 = np.random.uniform(0, 10, 30)  # 既知の点
y0 = y(x0) + np.random.normal(0, 2, 30)  # 関数にノイズを加える
x1 = np.linspace(-1, 11, 101)  # 探す点

kernel = RBFkernel(8, 0.5, 3.5)  # 適当なパラメータを使うカーネル関数

k00 = kernel(*np.meshgrid(x0, x0))
k00_1 = np.linalg.inv(k00)  # 逆行列
k01 = kernel(*np.meshgrid(x0, x1, indexing='ij'))
k10 = k01.T
k11 = kernel(*np.meshgrid(x1, x1))

# ここでは上述の方程式の通りのμとΣ
mu = k10.dot(k00_1.dot(y0))
sigma = k11 - k10.dot(k00_1.dot(k01))

plt.scatter(x0, y0, c='#ff77aa', label='Known data')
plt.plot(x1, y(x1), '--r', label='True function')  # 本物の関数
plt.plot(x1, mu, 'g', label='Predict mean')  # 推測された平均
std = np.sqrt(sigma.diagonal())  # 各点の標準偏差は共分散行列の対角成分
plt.fill_between(x1, mu-std, mu+std, alpha=0.2, color='g')  # 推測された標準偏差の中の領域
plt.legend()
plt.show()


# === ガウス過程のアニメーション化 ===
def y(x):
    return 10*np.sin(np.pi*x/2)

n = 30
x0 = np.random.permutation(np.linspace(0.1, 9.9, n))
y0 = y(x0) + np.random.normal(0, 0.1, n)
gif = []
for i in range(n):
    x1 = np.linspace(0, 10, 1000)
    kernel = RBFkernel(8, 0.5, 0.1)

    k00 = kernel(*np.meshgrid(x0[:i], x0[:i]))
    k00_1 = np.linalg.inv(k00)
    k01 = kernel(*np.meshgrid(x0[:i], x1, indexing='ij'))
    k10 = k01.T
    k11 = kernel(*np.meshgrid(x1, x1))

    mu = k10.dot(k00_1.dot(y0[:i]))
    sigma = k11 - k10.dot(k00_1.dot(k01))
    std = np.sqrt(sigma.diagonal())

    fig = plt.figure()
    plt.scatter(x0[:i], y0[:i], color='w', edgecolor='b', label='Known data')
    plt.plot(x1, y(x1), 'r--', label='True function')
    plt.plot(x1, mu, 'b', label='Predict mean')
    plt.legend()
    plt.ylim(-15, 15)
    plt.fill_between(x1, mu-std, mu+std, alpha=0.1, color='b')
    plt.tight_layout()
    fig.canvas.draw()
    gif.append(np.array(fig.canvas.renderer._renderer))
    plt.close()
imageio.mimsave('gp.gif', gif, fps=2.5)

