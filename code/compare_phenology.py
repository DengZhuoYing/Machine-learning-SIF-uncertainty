import rasterio
from rasterio import open as rasterio_open
from rasterio.transform import from_origin
import numpy as np
import os

for year in range(2014, 2026):
    SOSyear1 = rasterio_open(fr'D:\SIFcompare\GOSIF_phe\band1\SOS_{year}.tif').read(1)
    SOSyear2 = rasterio_open(fr'D:\SIFcompare\GOSIF_phe\band2\SOS_{year}.tif').read(1)
    SOSyear1[SOSyear1 == -9999] = np.nan
    SOSyear2[SOSyear2 == -9999] = np.nan
    SOSdelta = SOSyear2 - SOSyear1

    POSyear1 = rasterio_open(fr'D:\SIFcompare\GOSIF_phe\band1\POS_{year}.tif').read(1)
    POSyear2 = rasterio_open(fr'D:\SIFcompare\GOSIF_phe\band2\POS_{year}.tif').read(1)
    POSyear1[POSyear1 == -9999] = np.nan
    POSyear2[POSyear2 == -9999] = np.nan
    POSdelta = POSyear2 - POSyear1

    EOSyear1 = rasterio_open(fr'D:\SIFcompare\GOSIF_phe\band1\EOS_{year}.tif').read(1)
    EOSyear2 = rasterio_open(fr'D:\SIFcompare\GOSIF_phe\band2\EOS_{year}.tif').read(1)
    EOSyear1[EOSyear1 == -9999] = np.nan
    EOSyear2[EOSyear2 == -9999] = np.nan
    EOSdelta = EOSyear2 - EOSyear1

    if year > 2024:
        landcover = rasterio_open(fr'D:\SIFcompare\lc05\lc_2024.tif').read(1)
    else:
        landcover = rasterio_open(fr'D:\SIFcompare\lc05\lc_{year}.tif').read(1)

    landcover = landcover[:120, :]

    # Keep only the first 120 rows for each dataset
    SOSdelta = SOSdelta[:120, :]
    POSdelta = POSdelta[:120, :]
    EOSdelta = EOSdelta[:120, :]

    SOSdelta[
        (landcover == 0) | (landcover == 15) | (landcover == 16) | (landcover == 17) | (landcover == 255)] = np.nan
    POSdelta[
        (landcover == 0) | (landcover == 15) | (landcover == 16) | (landcover == 17) | (landcover == 255)] = np.nan
    EOSdelta[
        (landcover == 0) | (landcover == 15) | (landcover == 16) | (landcover == 17) | (landcover == 255)] = np.nan

    os.makedirs(fr'D:\SIFcompare\GOSIF_phe\delta', exist_ok=True)
    raster = fr'D:\SIFcompare\GOSIF_phe\delta\delta_{year}.tif'

    with rasterio.open(raster, 'w', driver='GTiff', height=120, width=720, count=3, dtype=np.float32,
                       crs='+proj=longlat +datum=WGS84 +no_defs',
                       transform=from_origin(-180, 90, 0.5, 0.5)) as dst:
        dst.write(SOSdelta, 1)
        dst.write(POSdelta, 2)
        dst.write(EOSdelta, 3)

SOSdelta = np.zeros((120, 720))
POSdelta = np.zeros((120, 720))
EOSdelta = np.zeros((120, 720))
for year in range(2014, 2026):
    SOSdelta += rasterio_open(fr'D:\SIFcompare\GOSIF_phe\delta\delta_{year}.tif').read(1)
    POSdelta += rasterio_open(fr'D:\SIFcompare\GOSIF_phe\delta\delta_{year}.tif').read(2)
    EOSdelta += rasterio_open(fr'D:\SIFcompare\GOSIF_phe\delta\delta_{year}.tif').read(3)

SOSdelta /= 12
POSdelta /= 12
EOSdelta /= 12

raster = fr'other\delta_mean_GOSIF.tif'

with rasterio.open(raster, 'w', driver='GTiff', height=120, width=720, count=3, dtype=np.float32,
                   crs='+proj=longlat +datum=WGS84 +no_defs',
                   transform=from_origin(-180, 90, 0.5, 0.5)) as dst:
    dst.write(SOSdelta, 1)
    dst.write(POSdelta, 2)
    dst.write(EOSdelta, 3)

SOSdelta = SOSdelta[~np.isnan(SOSdelta)]
POSdelta = POSdelta[~np.isnan(POSdelta)]
EOSdelta = EOSdelta[~np.isnan(EOSdelta)]
SOSmean = np.mean(SOSdelta)
POSmean = np.mean(POSdelta)
EOSmean = np.mean(EOSdelta)
print(SOSmean, POSmean, EOSmean)
print(len(SOSdelta), len(POSdelta), len(EOSdelta))

import rasterio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

periods = [(2014, 2015), (2016, 2017), (2018, 2019), (2020, 2021), (2022, 2023), (2024, 2025)]
period_labels = [f"{p[0]}–{p[1]}" for p in periods]

data_SOS = {label: {'year1': [], 'year2': []} for label in period_labels}
data_POS = {label: {'year1': [], 'year2': []} for label in period_labels}
data_EOS = {label: {'year1': [], 'year2': []} for label in period_labels}

for year in range(2014, 2026):

    with rasterio.open(fr'D:\SIFcompare\GOSIF_phe\band1\SOS_{year}.tif') as src:
        SOSyear1 = src.read(1)
    with rasterio.open(fr'D:\SIFcompare\GOSIF_phe\band2\SOS_{year}.tif') as src:
        SOSyear2 = src.read(1)
    SOSyear1[SOSyear1 == -9999] = np.nan
    SOSyear2[SOSyear2 == -9999] = np.nan

    with rasterio.open(fr'D:\SIFcompare\GOSIF_phe\band1\POS_{year}.tif') as src:
        POSyear1 = src.read(1)
    with rasterio.open(fr'D:\SIFcompare\GOSIF_phe\band2\POS_{year}.tif') as src:
        POSyear2 = src.read(1)
    POSyear1[POSyear1 == -9999] = np.nan
    POSyear2[POSyear2 == -9999] = np.nan

    with rasterio.open(fr'D:\SIFcompare\GOSIF_phe\band1\EOS_{year}.tif') as src:
        EOSyear1 = src.read(1)
    with rasterio.open(fr'D:\SIFcompare\GOSIF_phe\band2\EOS_{year}.tif') as src:
        EOSyear2 = src.read(1)
    EOSyear1[EOSyear1 == -9999] = np.nan
    EOSyear2[EOSyear2 == -9999] = np.nan

    if year > 2024:
        lc_file = r'D:\SIFcompare\lc05\lc_2024.tif'
    else:
        lc_file = fr'D:\SIFcompare\lc05\lc_{year}.tif'
    with rasterio.open(lc_file) as src:
        landcover = src.read(1)

    landcover = landcover[:120, :]

    SOSyear1 = SOSyear1[:120, :]
    POSyear1 = POSyear1[:120, :]
    EOSyear1 = EOSyear1[:120, :]
    SOSyear2 = SOSyear2[:120, :]
    POSyear2 = POSyear2[:120, :]
    EOSyear2 = EOSyear2[:120, :]

    mask_land = (landcover == 0) | (landcover == 15) | (landcover == 16) | (landcover == 17) | (landcover == 255)
    SOSyear1[mask_land] = np.nan
    POSyear1[mask_land] = np.nan
    EOSyear1[mask_land] = np.nan
    SOSyear2[mask_land] = np.nan
    POSyear2[mask_land] = np.nan
    EOSyear2[mask_land] = np.nan

    idx = (year - 2014) // 2
    period_label = period_labels[idx]

    valid_sos = ~(np.isnan(SOSyear1) | np.isnan(SOSyear2))
    data_SOS[period_label]['year1'].extend(SOSyear1[valid_sos])
    data_SOS[period_label]['year2'].extend(SOSyear2[valid_sos])

    valid_pos = ~(np.isnan(POSyear1) | np.isnan(POSyear2))
    data_POS[period_label]['year1'].extend(POSyear1[valid_pos])
    data_POS[period_label]['year2'].extend(POSyear2[valid_pos])

    valid_eos = ~(np.isnan(EOSyear1) | np.isnan(EOSyear2))
    data_EOS[period_label]['year1'].extend(EOSyear1[valid_eos])
    data_EOS[period_label]['year2'].extend(EOSyear2[valid_eos])


def plot_split_violin(data_dict, var_name, period_labels, save_path=None):
    rows = []

    for period in period_labels:
        for src in ['year1', 'year2']:
            values = data_dict[period][src]
            for val in values:
                rows.append((period, src, val))
    df = pd.DataFrame(rows, columns=['Period', 'Source', 'Value'])

    q_low = df['Value'].quantile(0.01)
    q_high = df['Value'].quantile(0.99)
    df = df[(df['Value'] >= q_low) & (df['Value'] <= q_high)]

    plt.figure(figsize=(8, 4))

    plt.style.use('default')
    sns.set(style="ticks", font_scale=1.2, rc={
        'axes.edgecolor': 'black',
        'axes.linewidth': 1.5,
        'xtick.major.size': 4,
        'xtick.major.width': 1.2,
        'ytick.major.size': 4,
        'ytick.major.width': 1.2,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'font.sans-serif': 'Arial',
        'axes.grid': False,
        'grid.color': '#E0E0E0',
        'grid.linestyle': '--',
        'grid.linewidth': 0.6,
        'legend.frameon': False
    })

    palette = {'year1': (40 / 255, 114 / 255, 113 / 255), 'year2': (120 / 255, 0 / 255, 1 / 255)}
    ax = sns.violinplot(data=df, x='Period', y='Value', hue='Source',
                        split=True, palette=palette, inner='quart', linewidth=1.8, cut=0, legend=False)

    plt.xticks(fontsize=16, rotation=30)
    plt.yticks(fontsize=16)
    plt.xlabel('')
    plt.ylabel(f'{var_name} (days)', fontsize=18)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


plot_split_violin(data_SOS, 'SOS', period_labels, save_path=r'other\SOS_GOSIF.png')
plot_split_violin(data_POS, 'POS', period_labels, save_path=r'other\POS_GOSIF.png')
plot_split_violin(data_EOS, 'EOS', period_labels, save_path=r'other\EOS_GOSIF.png')
