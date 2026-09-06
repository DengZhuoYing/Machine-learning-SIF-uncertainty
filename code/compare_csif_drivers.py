import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from matplotlib import patches
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
import scipy

df = pd.read_csv(r'other\OCO2_global_final.csv')
df = df[(df['R4'] > 0) & (df['R2'] > 0) & (df['R3'] > 0) & (df['R1'] > 0)]
df['R1'] = df['R1'] / 10000
df['R2'] = df['R2'] / 10000
df['R3'] = df['R3'] / 10000
df['R4'] = df['R4'] / 10000

df20142017 = df[(df['year'] >= 2014) & (df['year'] <= 2017)]
df20222025 = df[(df['year'] >= 2022) & (df['year'] <= 2025)]

model = xgb.XGBRegressor(random_state=42)
model.fit(df20142017[['R1', 'R2', 'R3', 'R4', 'COSSZA']], df20142017['SIF'])
joblib.dump(model, 'other/OCO_20142017_CSIF.pkl')

model = xgb.XGBRegressor(random_state=42)
model.fit(df20222025[['R1', 'R2', 'R3', 'R4', 'COSSZA']], df20222025['SIF'])
joblib.dump(model, 'other/OCO_20222025_CSIF.pkl')

df = pd.read_csv(r'other\OCO2_global_final.csv')
df = df[(df['R4'] > 0) & (df['R2'] > 0) & (df['R3'] > 0) & (df['R1'] > 0)]
df['R1'] = df['R1'] / 10000
df['R2'] = df['R2'] / 10000
df['R3'] = df['R3'] / 10000
df['R4'] = df['R4'] / 10000

models = {
    '2014-2017': joblib.load('other/OCO_20142017_CSIF.pkl'),
    '2022-2025': joblib.load('other/OCO_20222025_CSIF.pkl'),
}

shap_values_dict = {}
explainers = {}

feature_names = ['R1', 'R2', 'R3', 'R4', 'COSSZA']
X = df[feature_names]
X_sample = X.sample(n=10000, random_state=42)

for period, model in models.items():
    explainer = shap.TreeExplainer(model)
    explainers[period] = explainer
    shap_values = explainer.shap_values(X_sample)
    shap_values_dict[period] = {
        'shap_values': shap_values,
        'X_sample': X_sample
    }

R1_shap_comparison = []
for period in models.keys():
    shap_values = shap_values_dict[period]['shap_values']
    X_sample = shap_values_dict[period]['X_sample']
    R1_idx = feature_names.index('R1')
    R1_shap = shap_values[:, R1_idx]
    R1_values = X_sample['R1'].values
    period_df = pd.DataFrame({
        'Period': period,
        'R1_value': R1_values,
        'R1_SHAP': R1_shap
    })
    R1_shap_comparison.append(period_df)
R1_shap_df = pd.concat(R1_shap_comparison, ignore_index=True)

R2_shap_comparison = []
for period in models.keys():
    shap_values = shap_values_dict[period]['shap_values']
    X_sample = shap_values_dict[period]['X_sample']
    R2_idx = feature_names.index('R2')
    R2_shap = shap_values[:, R2_idx]
    R2_values = X_sample['R2'].values
    period_df = pd.DataFrame({
        'Period': period,
        'R2_value': R2_values,
        'R2_SHAP': R2_shap
    })
    R2_shap_comparison.append(period_df)
R2_shap_df = pd.concat(R2_shap_comparison, ignore_index=True)

R3_shap_comparison = []
for period in models.keys():
    shap_values = shap_values_dict[period]['shap_values']
    X_sample = shap_values_dict[period]['X_sample']
    R3_idx = feature_names.index('R3')
    R3_shap = shap_values[:, R3_idx]
    R3_values = X_sample['R3'].values
    period_df = pd.DataFrame({
        'Period': period,
        'R3_value': R3_values,
        'R3_SHAP': R3_shap
    })
    R3_shap_comparison.append(period_df)
R3_shap_df = pd.concat(R3_shap_comparison, ignore_index=True)

R4_shap_comparison = []
for period in models.keys():
    shap_values = shap_values_dict[period]['shap_values']
    X_sample = shap_values_dict[period]['X_sample']
    R4_idx = feature_names.index('R4')
    R4_shap = shap_values[:, R4_idx]
    R4_values = X_sample['R4'].values
    period_df = pd.DataFrame({
        'Period': period,
        'R4_value': R4_values,
        'R4_SHAP': R4_shap
    })
    R4_shap_comparison.append(period_df)
R4_shap_df = pd.concat(R4_shap_comparison, ignore_index=True)

colorlist = ['black', 'gray']
colorlistscatter = ['#3953A4', '#ED2123']

plt.figure(figsize=(14, 3))
plt.subplot(1, 4, 1)
for i, period in enumerate(models.keys()):
    period_data = R1_shap_df[R1_shap_df['Period'] == period]
    plt.scatter(period_data['R1_value'], period_data['R1_SHAP'], alpha=0.1, s=10, label=period,
                color=colorlistscatter[i])
    lowessline = sm.nonparametric.lowess(period_data['R1_SHAP'], period_data['R1_value'], frac=0.1)
    plt.plot(lowessline[:, 0], lowessline[:, 1], linewidth=2, color=colorlist[i])

plt.tick_params(labelsize=12)
plt.xlim(0, 0.55)
plt.xlabel('Red Refl', fontsize=12)
plt.ylabel('SHAP values', fontsize=12, labelpad=-0.5)

plt.subplot(1, 4, 2)
for i, period in enumerate(models.keys()):
    period_data = R2_shap_df[R2_shap_df['Period'] == period]
    plt.scatter(period_data['R2_value'], period_data['R2_SHAP'], alpha=0.1, s=10, label=period,
                color=colorlistscatter[i])
    lowessline = sm.nonparametric.lowess(period_data['R2_SHAP'], period_data['R2_value'], frac=0.1)
    plt.plot(lowessline[:, 0], lowessline[:, 1], linewidth=2, color=colorlist[i])
plt.tick_params(labelsize=12)
plt.xlim(0, 0.75)
plt.ylim(-0.15, 0.35)
plt.xlabel('NIR Refl', fontsize=12)

plt.subplot(1, 4, 3)
for i, period in enumerate(models.keys()):
    period_data = R3_shap_df[R3_shap_df['Period'] == period]
    plt.scatter(period_data['R3_value'], period_data['R3_SHAP'], alpha=0.1, s=10, label=period,
                color=colorlistscatter[i])
    lowessline = sm.nonparametric.lowess(period_data['R3_SHAP'], period_data['R3_value'], frac=0.1)
    plt.plot(lowessline[:, 0], lowessline[:, 1], linewidth=2, color=colorlist[i])
plt.tick_params(labelsize=12)
plt.xlim(0, 0.25)
plt.ylim(-0.04, 0.04)
plt.xlabel('Blue Refl', fontsize=12)

plt.subplot(1, 4, 4)
for i, period in enumerate(models.keys()):
    period_data = R4_shap_df[R4_shap_df['Period'] == period]
    plt.scatter(period_data['R4_value'], period_data['R4_SHAP'], alpha=0.1, s=10, label=period,
                color=colorlistscatter[i])
    lowessline = sm.nonparametric.lowess(period_data['R4_SHAP'], period_data['R4_value'], frac=0.1)
    plt.plot(lowessline[:, 0], lowessline[:, 1], linewidth=2, color=colorlist[i])
plt.xlim(0, 0.35)
plt.ylim(-0.06, 0.05)
plt.tick_params(labelsize=12)
plt.xlabel('Green Refl', fontsize=12)

plt.subplots_adjust(wspace=0.3)

import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.legend_handler import HandlerTuple

legend_elements = [
    (mpatches.Patch(facecolor=colorlistscatter[0], linewidth=2),
     mlines.Line2D([0, 3], [0, 0], color='black', lw=3)),

    (mpatches.Patch(facecolor=colorlistscatter[1], linewidth=2),
     mlines.Line2D([0, 3], [0, 0], color='gray', lw=3))
]

plt.legend(
    handles=[tuple(elements) for elements in legend_elements],
    labels=['2014‒2017', '2022‒2025'],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc='upper center',
    fontsize=14,
    frameon=False,
    bbox_to_anchor=(-1.45, 1.22),
    handletextpad=1.0,
    handlelength=6.0,
    handleheight=1,
    borderpad=0.5,
    ncol=2
)

plt.tight_layout()
plt.savefig('other/shap_comparison_csif.png', dpi=600, bbox_inches='tight')
plt.show()
