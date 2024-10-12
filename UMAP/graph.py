# https://qiita.com/odanny/items/7c550010f5915ae4acdc

import networkx as nx
import matplotlib.pyplot as plt

N = 500  # 頂点数
K = 5
P = 0.05

G = nx.watts_strogatz_graph(N,K,P)  # 適当なグラフ
# pos = nx.random_layout(G, seed=0)
pos = nx.spring_layout(G)  # Fruchterman-Reingold algorithm

# グラフを描画
plt.figure(figsize=(15, 15))
nx.draw_networkx(G, pos, with_labels=False, node_shape='.')
plt.axis("off")
plt.show()
