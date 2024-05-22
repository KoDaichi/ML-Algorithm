# https://qiita.com/tk-tatsuro/items/1228c90e8803db378f31

# データ取得①（ワインデータ）
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import linkage, dendrogram
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

wine = datasets.load_wine()
wine_df = pd.DataFrame(wine.data, columns=wine.feature_names) # データフレームに変形
print(wine_df)
print(wine_df.sort_values(by='alcohol'))

# リネーム
name_list = ['アルコール',
             'リンゴ酸',
             '灰',
             'アルカリ性の灰',
             'マグネシウム',
             'トータルフェノール',
             'フラボノイド',
             '非フラボノイドフェノール',
             'プロアントシアニン',
             '色の強度',
             '色相',
             '希釈値',
             'プロリン']
wine_df.columns = name_list
# print(wine_df.shape)
print(wine_df.head())


def plot_dendrograme(df):
    li = linkage(df.corr())
    r = dendrogram(li, labels=df.columns)
    plt.figure(figsize=[20, 5])
    plt.show()

# ワインデータ
wine_data = wine_df.copy()
dframe = pd.DataFrame()
dframe['項目名'] = wine_df.columns
# dframe

number_list = list(range(0, 13))
wine_data.columns = number_list
plot_dendrograme(wine_data)


# クラスタリング（k-means）
def clustering(df, num):
    sc = StandardScaler()
    sc.fit_transform(df)
    data_norm = sc.transform(df)

    cls = KMeans(n_clusters=num)
    result = cls.fit(data_norm)
    pred = cls.fit_predict(data_norm)

    plt.figure(figsize=[10, 5])
    sns.set(rc={'axes.facecolor':'cornflowerblue', 'figure.facecolor':'cornflowerblue'})
    sns.scatterplot(x=data_norm[:, 0], y=data_norm[:, 1], c=result.labels_)
    plt.scatter(result.cluster_centers_[:, 0], result.cluster_centers_[:, 1], s=250, marker='*', c='blue')
    plt.grid('darkgray')
    plt.show()

# ワインデータ
clustering(wine_data, 3)

# クラスター数の探索
def elbow(df):
    wcss = []

    for i in range(1, 10):
        kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 30, random_state = 0)
        kmeans.fit(df.iloc[:, :])
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=[10,4])
    plt.plot(range(1, 10), wcss)
    # plt.title('エルボー法')
    # plt.xlabel('クラスター数')
    # plt.ylabel('クラスター内平方和（WCSS）')
    plt.title('Elbor metod')
    plt.xlabel('Cluster num')
    plt.ylabel('WCSS')
    plt.grid()
    plt.show()

# ワインデータ
elbow(wine_df)

## クラスタリング（k-means）＋主成分分析
def cross(df, num):
    df_cls = df.copy()
    sc = StandardScaler()
    clustering_sc = sc.fit_transform(df_cls)
 
    kmeans = KMeans(n_clusters=num, random_state=42) # n_clusters：クラスター数
    clusters = kmeans.fit(clustering_sc)
    df_cls['cluster'] = clusters.labels_
 
    x = clustering_sc
    pca = PCA(n_components=num) # n_components：削減結果の次元数
    pca.fit(x)
    x_pca = pca.transform(x)
    pca_df = pd.DataFrame(x_pca)
    pca_df['cluster'] = df_cls['cluster']
 
    for i in df_cls['cluster'].unique():
        tmp = pca_df.loc[pca_df['cluster'] == i]
        plt.scatter(tmp[0], tmp[1])
    plt.grid()
    plt.show()

# ワインデータ
cross(wine_df, 3)
