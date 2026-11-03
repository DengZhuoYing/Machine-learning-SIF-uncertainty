import xarray as xr
from osgeo import gdal, osr
import numpy as np
import rasterio.features
import geopandas as gpd
import warnings
from rasterio.mask import mask
import rasterio.features
import joblib
from numpy import zeros, mean, array, datetime64, pi, degrees, arcsin, cos, radians, sin, nan, nanmean
import pandas as pd
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from numba import jit
from numpy import nan
from scipy import stats as st
from calendar import monthrange
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
import scipy.stats as stats
from rasterio.enums import Resampling

warnings.filterwarnings("ignore")


def load_raster_to_memory(raster_path):
    with open(raster_path, 'rb') as f:
        memfile = MemoryFile(f.read())
    return memfile


def callc(lc, SIF):
    result = np.zeros_like(SIF)
    non_zero_mask = (SIF != 0)
    lc_non_zero = lc[non_zero_mask]
    SIF_non_zero = SIF[non_zero_mask]
    result_non_zero = np.zeros_like(SIF_non_zero)

    mask = (lc_non_zero == 1)
    result_non_zero[mask] = SIF_non_zero[mask] * 27.35 + 0.92

    mask = (lc_non_zero == 2)
    result_non_zero[mask] = SIF_non_zero[mask] * 17.87 + 3.70

    mask = (lc_non_zero == 4)
    result_non_zero[mask] = SIF_non_zero[mask] * 22.10 - 0.03

    mask = (lc_non_zero == 5)
    result_non_zero[mask] = SIF_non_zero[mask] * 21.80 - 0.03

    mask = (lc_non_zero == 7)
    result_non_zero[mask] = SIF_non_zero[mask] * 32.23 + 0.13

    mask = (lc_non_zero == 9)
    result_non_zero[mask] = SIF_non_zero[mask] * 24.85 + 0.09

    mask = (lc_non_zero == 8)
    result_non_zero[mask] = SIF_non_zero[mask] * 24.41 + 0.28

    mask = (lc_non_zero == 10)
    result_non_zero[mask] = SIF_non_zero[mask] * 24.65 - 0.06

    mask = (lc_non_zero == 11)
    result_non_zero[mask] = SIF_non_zero[mask] * 24.38 + 0.17

    mask = (lc_non_zero == 12)
    result_non_zero[mask] = SIF_non_zero[mask] * 26.16 - 0.59

    mask = ~(
            (lc_non_zero == 1) | (lc_non_zero == 2) | (lc_non_zero == 4) | (lc_non_zero == 5) |
            (lc_non_zero == 7) | (lc_non_zero == 8) | (lc_non_zero == 9) | (lc_non_zero == 10) |
            (lc_non_zero == 11) | (lc_non_zero == 12)
    )
    result_non_zero[mask] = SIF_non_zero[mask] * 24.08 + 0.53
    result[non_zero_mask] = result_non_zero

    return result


df = pd.DataFrame()
for year in range(2001, 2021):

    rasterlc = load_raster_to_memory(fr'D:\SIFcompare\lc05\lc_{year}.tif')
    with rasterlc.open() as lc:
        lc_data = lc.read(1)

    SIFnc = xr.open_dataset(fr'F:\CSIF\OCO2.SIF.clear.daily.{year}.nc')
    SIF12 = SIFnc['clear_daily_sif'][:, :, :]
    GPP_year = np.zeros(360 * 720)

    for month in range(1, 13):
        SIF_data = SIF12[month - 1, :, :]
        SIF_data = np.flipud(SIF_data)
        SIF_data = np.nan_to_num(SIF_data, nan=0.0)
        GPPtmp = callc(lc_data, SIF_data)
        GPP_year += GPPtmp.flatten()

        print(year, month)

    df[f'GPP_{year}'] = GPP_year / 12

gpp_2001 = df['GPP_2001'].values
epsilon = 1e-10
gpp_2001_safe = np.where(gpp_2001 > epsilon, gpp_2001, np.nan)
df_change = pd.DataFrame(index=df.index)

for year in range(2001, 2021):
    if year == 2001:
        df_change[f'Change_{year}'] = 0.0
    else:
        gpp_current = df[f'GPP_{year}'].values
        relative_change = ((gpp_current - gpp_2001_safe) / gpp_2001_safe) * 100
        df_change[f'Change_{year}'] = relative_change

years = np.arange(2001, 2021)  # Year sequence
n_pixels = df.shape[0]  # Number of pixels

slopes = np.zeros(n_pixels)  # Slope (trend)
intercepts = np.zeros(n_pixels)  # Intercept
r_values = np.zeros(n_pixels)  # Correlation coefficient
p_values = np.zeros(n_pixels)  # p-value
std_errs = np.zeros(n_pixels)  # Standard error

for i in range(n_pixels):
    y_vals = df_change.iloc[i].values.astype(float)

    valid_mask = ~np.isnan(y_vals)
    x_valid = years[valid_mask]
    y_valid = y_vals[valid_mask]

    if len(x_valid) >= 3:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_valid, y_valid)
        slopes[i] = slope  # Slope represents the annual percentage change
        intercepts[i] = intercept
        r_values[i] = r_value
        p_values[i] = p_value
        std_errs[i] = std_err
    else:
        slopes[i] = np.nan
        intercepts[i] = np.nan
        r_values[i] = np.nan
        p_values[i] = np.nan
        std_errs[i] = np.nan

    if (i + 1) % 10000 == 0:
        print(f"Processed {i + 1}/{n_pixels} pixels")

df_trend = pd.DataFrame({
    'slope': slopes,  # Trend slope (% per year)
    'intercept': intercepts,  # Regression intercept
    'r_value': r_values,  # Correlation coefficient
    'p_value': p_values,  # p-value (significance)
    'std_err': std_errs  # Standard error
})
print(df_trend)
df_trend.to_csv('other\CSIF-linear.csv', index=False)

df_trend = pd.read_csv('other\CSIF-linear.csv')
slope = df_trend['slope'].values.flatten()
p_value = df_trend['p_value'].values.flatten()
# slopes = np.where(p_value >= 0.05, 0.0, slope)
slope_grid = slope.reshape(360, 720)

file = fr'other\CSIF-linear.tif'
with rasterio.open(file, 'w', driver='GTiff', height=360, width=720, count=1, dtype=slope_grid.dtype,
                   crs='+proj=longlat +datum=WGS84 +no_defs',
                   transform=from_origin(-180, 90, 0.5, 0.5)) as dst:
    dst.write(slope_grid, 1)


def resample_tif(input_path, output_path, target_resolution=0.05):
    with rasterio.open(input_path) as src:
        new_width = int(src.width * (src.res[0] / target_resolution))
        new_height = int(src.height * (src.res[1] / target_resolution))

        new_transform = src.transform * src.transform.scale(
            (src.width / new_width),
            (src.height / new_height)
        )

        data = src.read()

        resampled_data = np.empty((src.count, new_height, new_width), dtype=src.dtypes[0])

        for i in range(src.count):
            rasterio.warp.reproject(
                source=data[i],
                destination=resampled_data[i],
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=new_transform,
                dst_crs=src.crs,
                resampling=Resampling.bilinear  # Bilinear interpolation
            )

        profile = src.profile.copy()
        profile.update({
            'height': new_height,
            'width': new_width,
            'transform': new_transform,
            'dtype': src.dtypes[0]
        })

        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(resampled_data)


resample_tif(fr'other\CSIF-linear.tif', fr'other\CSIF-linear-005.tif', target_resolution=0.05)