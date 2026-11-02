import pandas as pd
import numpy as np
import xgboost as xgb
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(r'..\driver_stability\other\OCO2_global_final.csv')
df = df[(df['R4'] > 0) & (df['R2'] > 0) & (df['R3'] > 0) & (df['R1'] > 0)]
df['R1'] = df['R1'] / 10000
df['R2'] = df['R2'] / 10000
df['R3'] = df['R3'] / 10000
df['R4'] = df['R4'] / 10000

df = df[(df['PAR'] > 0)& (df['FPAR'] > 0)]
df['PAR'] = df['PAR'] / 24 / 3600
df['SIF'] = df['SIF'] / df['PAR'] / (df['FPAR'] / 100) * 100

np.random.seed(42)
df['date'] = df[['year', 'month', 'day']].apply(lambda x: f"{x['year']}-{x['month']}-{x['day']}", axis=1)
unique_dates = df['date'].unique()
shuffled_dates = np.random.permutation(unique_dates)
n_dates = len(unique_dates)
train_end = int(n_dates * 0.7)
date_to_split = {}
date_to_split.update({date: 'train' for date in shuffled_dates[:train_end]})
date_to_split.update({date: 'test' for date in shuffled_dates[train_end:]})
df['split'] = df['date'].map(date_to_split)
df_train = df[df['split'] == 'train'].drop(columns=['split', 'date'])
df_test = df[df['split'] == 'test'].drop(columns=['split', 'date'])

model = xgb.XGBRegressor(random_state=42)
model.fit(df_train[['R1', 'R2', 'R3', 'R4', 'COSSZA']], df_train['SIF'])
preds = model.predict(df_test[['R1', 'R2', 'R3', 'R4', 'COSSZA']])

from sklearn.metrics import r2_score; r2 = r2_score(df_test['SIF'], preds)
rmse = np.sqrt(np.mean((preds - df_test['SIF']) ** 2))
rrmse = rmse / np.mean(df_test['SIF']) * 100
print(f"R² = {r2:.3f}, RMSE = {rmse:.3f}")

y_test = df_test['SIF'].values.flatten()
y_predicted = preds

np.random.seed(43)
sample_size = 20000
indices = np.random.choice(len(y_test), size=sample_size, replace=False)
sampled_y_test = y_test[indices]
sampled_y_predicted = y_predicted[indices]

y_test = sampled_y_test
y_predicted = sampled_y_predicted

lr = LinearRegression()
lr.fit(y_test.reshape(-1, 1), y_predicted.reshape(-1, 1))
slope = lr.coef_[0][0]
intercept = lr.intercept_[0]

plt.figure(figsize=(5, 5))

plt.scatter(y_test, y_predicted, s=0.5, color='gray')
sns.kdeplot(x=y_test, y=y_predicted, cmap="Purples", fill=True, levels=5, alpha=0.85)
plt.plot([0, 2], [0, 2], 'k--', linewidth=1.5, label='1:1 line')

x_line = np.array([0, 2])
y_line = slope * x_line + intercept
plt.plot(x_line, y_line, color='lightgray', linewidth=2, label='Regression line')

plt.annotate(f'y = {slope:.2f}x + {intercept:.2f}', xy=(0.05, 0.92), xycoords='axes fraction',
             fontsize=14, fontweight='normal')
plt.annotate(f'R² = {r2:.3f}', xy=(0.05, 0.84), xycoords='axes fraction', fontsize=14)
plt.annotate(f'RMSE = {rmse:.3f}', xy=(0.05, 0.76), xycoords='axes fraction', fontsize=14)

plt.xlabel(r'Observed SIF/PAR/FPAR', fontsize=16)
plt.ylabel(r'Predicted SIF/PAR/FPAR', fontsize=16)

plt.title('CSIF Normalized by APAR', fontsize=16)

plt.xticks(np.arange(0, 2.01, 0.4), fontsize=14)
plt.yticks(np.arange(0, 2.01, 0.4), fontsize=14)
plt.xlim(-0.4, 1.6)
plt.ylim(-0.4, 1.6)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.gca().set_aspect('equal')
plt.legend(loc='lower right', frameon=False, fontsize=14)
plt.tight_layout()
plt.savefig(r'other/csifapar.png', dpi=600)
plt.show()
