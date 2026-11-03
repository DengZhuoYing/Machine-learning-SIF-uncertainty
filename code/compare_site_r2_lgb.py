import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import lightgbm as lgb


def get_sif_versions(df):
    sif_raw = df['SIF_raw'].values
    sif_div_par = df['SIF_raw'].values / df['PAR'].values
    sif_div_par_fpar = df['SIF_raw'].values / df['PAR'].values / df['FPAR'].values
    return {'SIF': sif_raw, 'SIF/PAR': sif_div_par, 'SIF/PAR/FPAR': sif_div_par_fpar}


def get_date_blocks(df):
    """Group all samples from each date into one sampling unit."""
    sample_dates = df['timestamp_start'].dt.normalize()
    unique_dates = pd.DatetimeIndex(sample_dates.unique()).sort_values()

    if len(unique_dates) < 2:
        raise ValueError('Fewer than two valid dates; cannot create training and test sets.')

    date_blocks = [
        df.index[sample_dates == date].to_numpy()
        for date in unique_dates
    ]
    return date_blocks


def split_date_blocks(date_blocks, rng, train_ratio=0.7):
    """Randomly select 70% of dates for training and use the remaining 30% for testing."""
    n_dates = len(date_blocks)
    n_train_dates = int(n_dates * train_ratio)
    n_train_dates = min(max(n_train_dates, 1), n_dates - 1)

    train_positions = rng.choice(n_dates, size=n_train_dates, replace=False)
    is_train = np.zeros(n_dates, dtype=bool)
    is_train[train_positions] = True
    test_positions = np.flatnonzero(~is_train)

    train_idx = np.concatenate([date_blocks[position] for position in train_positions])
    test_idx = np.concatenate([date_blocks[position] for position in test_positions])
    return train_idx, test_idx


sitenamelist = ['All sites', 'Reifsteck', 'Reinhart', 'Rund', 'US-Ne2', 'US-Ne3', 'US-UiB', 'US-UiC']

# sitenamelist = ['All sites', 'Reifsteck', 'Reinhart', 'Rund']

for site in sitenamelist:

    df = pd.read_csv(fr'other/{site}.csv')
    df = df[(df['SIF_iFLD_raw'] > -9999) & (df['NDVI'] > -9999) & (df['EVI'] > -9999) & (df['NIRv'] > -9999) & (
            df['CI_red_edge'] > -9999) & (df['PRI'] > -9999) & (df['FPAR_measured'] > -9999) & (df['PAR'] > -9999)]

    df['timestamp_start'] = pd.to_datetime(df['timestamp_start'], errors='coerce')
    df = df.dropna(subset=['timestamp_start'])
    df.sort_values('timestamp_start', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(df.shape)

    df['SIF_raw'] = df['SIF_iFLD_raw'] * df["f_cal_corr_QEPRO"]
    df['CI'] = df['CI_red_edge']
    df['FPAR'] = df['FPAR_measured']

    driver_combinations = {
        'NDVI+EVI+NIRv': ['NDVI', 'EVI', 'NIRv'],
        '+CI': ['NDVI', 'EVI', 'NIRv', 'CI'],
        '+CI+PRI': ['NDVI', 'EVI', 'NIRv', 'CI', 'PRI']
    }

    n_repeats = 1000
    results = []
    sif_versions = get_sif_versions(df)

    date_blocks = get_date_blocks(df)
    n_train_dates = int(len(date_blocks) * 0.7)
    n_test_dates = len(date_blocks) - n_train_dates
    print(
        f'{site}: {len(date_blocks)} valid dates; each run randomly selects '
        f'{n_train_dates} dates for training, {n_test_dates} dates for testing'
    )

    for sif_name, sif_values in sif_versions.items():
        for driver_name, driver_labels in driver_combinations.items():
            r2_list = []

            for i in tqdm(range(n_repeats), desc="Model repetitions"):
                # Sample 70% of complete dates without replacement and use the rest for testing.
                # The same i gives each model combination the same date split for comparison.
                rng = np.random.default_rng(42 + i)
                train_idx, test_idx = split_date_blocks(
                    date_blocks,
                    rng,
                    train_ratio=0.7,
                )

                X_train = df.loc[train_idx, driver_labels].values
                y_train = sif_values[train_idx]
                X_test = df.loc[test_idx, driver_labels].values
                y_test = sif_values[test_idx]

                model = lgb.LGBMRegressor(random_state=i, verbose=-1)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)

                r2 = r2_score(y_test, preds)
                r2_list.append(r2)

            for r2 in r2_list:
                results.append({
                    'SIF_type': sif_name,
                    'Driver_type': driver_name,
                    'R2': r2,
                })

            print(
                f"SIF: {sif_name}, Drivers: {driver_name} -> R2: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(f'lgb/{site} results.csv', index=False)

    df_results = pd.read_csv(f'lgb/{site} results.csv')
    df_results['R2'] = np.where(df_results['R2'] < 0, 0, df_results['R2'])
    plt.figure(figsize=(6, 5))
    sns.boxplot(data=df_results, x='SIF_type', y='R2', hue='Driver_type', order=['SIF', 'SIF/PAR', 'SIF/PAR/FPAR'],
                palette='Set1', legend=False)
    plt.title(f'{site}', fontsize=16)
    plt.xlabel('', fontsize=16)
    plt.ylabel('R²', fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend(loc='lower left', fontsize=16, frameon=False)
    plt.tight_layout()
    plt.savefig(f'lgb/{site}.png', dpi=600)
    # plt.show()

site = 'All sites'
df_results = pd.read_csv(f'lgb/{site} results.csv')
df_results['R2'] = np.where(df_results['R2'] < 0, 0, df_results['R2'])
plt.figure(figsize=(6, 5))
sns.boxplot(data=df_results, x='SIF_type', y='R2', hue='Driver_type', order=['SIF', 'SIF/PAR', 'SIF/PAR/FPAR'],
            palette='Set1')
plt.title(f'{site}', fontsize=16)
plt.xlabel('', fontsize=16)
plt.ylabel('R²', fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(loc='lower left', fontsize=16, frameon=False)
plt.tight_layout()
plt.savefig(f'lgb/{site}.png', dpi=600)
plt.show()
