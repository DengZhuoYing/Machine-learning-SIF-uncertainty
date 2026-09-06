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
import sys
import xarray as xr

sys.path.append(r'D:\Global_SIF_Simulate\code\utils')

from util import area_of_polygon

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


resolution = 0.5
latitudes = np.linspace(90 - resolution / 2, -90 + resolution / 2, int(180 / resolution))
longitudes = np.linspace(-180 + resolution / 2, 180 - resolution / 2, int(360 / resolution))
area = np.zeros((int(180 / resolution), int(360 / resolution)))
for i in range(int(180 / resolution)):
    for j in range(int(360 / resolution)):
        area[i, j] = area_of_polygon(latitudes[i], resolution)

df = pd.DataFrame()
for year in range(2001, 2021):

    if year <= 2019:
        rasterC4 = load_raster_to_memory(fr'other\C405\C4_{year}.tif')
    else:
        rasterC4 = load_raster_to_memory(fr'other\C405\C4_2019.tif')
    with rasterC4.open() as C4:
        C4_data = C4.read(1)
    C4_data = np.nan_to_num(C4_data, nan=0.0) / 100
    C3_data = 1 - C4_data

    rasterlc = load_raster_to_memory(fr'D:\SIFcompare\lc05\lc_{year}.tif')
    with rasterlc.open() as lc:
        lc_data = lc.read(1)

    SIFnc = xr.open_dataset(fr'F:\CSIF\OCO2.SIF.clear.daily.{year}.nc')
    SIF12 = SIFnc['clear_daily_sif'][:, :, :]
    GPP_year = np.zeros(360 * 720)

    rasterfai = load_raster_to_memory(fr'other\fai05\fai_2001.tif')
    with rasterfai.open() as fai:
        fai_data = fai.read(1)
    rasterfai = load_raster_to_memory(fr'other\fai05\fai_{year}.tif')
    with rasterfai.open() as fai:
        fai_data2 = fai.read(1)
    fai_data = fai_data2 / fai_data

    for month in range(1, 13):
        SIF_data = SIF12[month - 1, :, :]
        SIF_data = np.flipud(SIF_data)
        SIF_data = np.nan_to_num(SIF_data, nan=0.0)
        areatmp = area.copy()
        areatmp[SIF_data == 0] = 0

        GPPtmp = callc(lc_data, SIF_data)
        GPPtmpco2 = GPPtmp * fai_data * C3_data + GPPtmp * C4_data
        GPP_year += (GPPtmpco2 * areatmp).flatten() / np.sum(areatmp)
        print(year, month)

    df[f'GPP_{year}'] = GPP_year / 12

yearsGPP = df.columns[:].astype(str).str[-4:].astype(int).tolist()
sumGPP = df.sum().values
sumGPP = (sumGPP - sumGPP[0]) / sumGPP[0] * 100
dfGPPsum = pd.DataFrame({'Year': yearsGPP, 'Sum': sumGPP})
dfGPPsum.to_csv(r'..\compare_long_term_trend\other\CSIF-CO2-linear.csv', index=False)
