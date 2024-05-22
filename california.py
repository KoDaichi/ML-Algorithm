# https://qiita.com/nuco_fn/items/75272b5f4a3c27da132a

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

sns.set_style('whitegrid')

# カリフォルニア住宅価格のデータセット
housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['Price'] = housing.target
print(df.head())

# 各データの最大・最小など
print(df.describe())

df.hist(bins=50, figsize=(15, 13))
plt.show()

print(df.corr())

# データの前処理(HouseAgeとPriceの不自然なデータを除く)
df = df[df['HouseAge'] < df['HouseAge'].max()]
df = df[df['Price'] < df['Price'].max()]

df.hist(bins=50, figsize=(15, 13))
plt.show()

# 説明変数
X = df.drop(['Price'], axis=1)
# 目的変数
y = df['Price']

scaler = StandardScaler()
# 標準化(平均を0, 分散を1にしてデータの散らばり具合を縮小 --> 範囲の異なる変数を同様に扱える)
X = scaler.fit_transform(X)

# データセットの分割(訓練データとテストデータ 8:2)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
print(X_train.shape, X_test.shape)

# 学習(線型回帰)
model = LinearRegression()
model.fit(X_train, y_train)
print(model.intercept_)
print(model.coef_)

X_some = X_train[:4]
y_some = y_train[:4]
print(f'予測結果{np.round(model.predict(X_some), 3)}')
print(f'正解ラベル{list(y_some)}\n')

# 訓練データのRMSE(Root Mean Squared Eroor)計算
model_pred = model.predict(X_train)
err_sum = ((y_train - model_pred) ** 2).sum()
mse = err_sum / len(y_train)
rmse = np.sqrt(mse)

print(f'誤差の合計:{err_sum}')
print(f'誤差の平均値(MSE):{mse}')
print(f'RMSE:{rmse}\n')

# テストデータで予測
model_test_pred = model.predict(X_test)
model_test_mse = mean_squared_error(y_test, model_test_pred)
model_test_rmse = np.sqrt(model_test_mse)

print(f'誤差の平均値(MSE):{model_test_mse}')
print(f'RMSE:{model_test_rmse}')

# 結果の可視化
plt.figure(figsize=(5,5))
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.plot(model.predict(X_test), y_test, 'o')
plt.title('RMSE: {:.3f}'.format(model_test_rmse))
plt.xlabel('Predict')
plt.ylabel('Actual')
plt.show()

