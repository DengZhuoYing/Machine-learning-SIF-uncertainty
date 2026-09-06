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
from rasterio.enums import Resampling


def load_raster_to_memory(raster_path):
    with open(raster_path, 'rb') as f:
        memfile = MemoryFile(f.read())
    return memfile


# Calculate atmospheric pressure in Pa
def calc_patm(elv, patm0=101325):
    kTo = 298.15
    kL = 0.0065
    kG = 9.80665
    kR = 8.3145
    kMa = 0.028963

    patm = patm0 * (1 - kL * elv / kTo) ** (kG * kMa / (kR * kL))

    return patm


# Calculate the temperature correction factor with the Arrhenius equation
def ftemp_arrh(tk, dha, tkref=298.15):
    kR = 8.3145
    ftemp = np.exp(dha * (tk - tkref) / (tkref * kR * tk))
    return ftemp


# Calculate the CO2 compensation point in ppm
def calc_gammastar(tk, patm):
    dha = 37830
    gs25_0 = 4.332  # Pa
    patm0 = calc_patm(0)
    ftemp = ftemp_arrh(tk, dha)
    gammastar = gs25_0 * (patm / patm0) * ftemp
    gammastar = gammastar / patm * 1e6

    return gammastar


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


DEM_path = load_raster_to_memory(r'other\DEM.tif')
with DEM_path.open() as DEM:
    DEM_data = DEM.read(1)

Tnc = xr.open_dataset(r'other\monthT.nc')
T = Tnc['t2m'][:, 1:, :]

# CO2nc1 = xr.open_dataset(r'other\CO2_1deg_month_1850-2013.nc')
# CO2_1 = CO2nc1['value'][:, :, :].values
#
# CO2nc2 = xr.open_dataset(r'other\CO2_SSP119_2015_2150.nc')
# CO2_2 = CO2nc2['CO2'][:, :, :].values

dfCO2 = pd.read_csv(r'other\co2_mm_gl.csv')

for year in range(2001, 2021):
    for month in range(1, 13):
        T_month = T[(year - 2001) * 12 + month - 1, :, :]

        T_month = np.concatenate((T_month[:, 1800:], T_month[:, :1800]), axis=1)
        T_month = np.repeat(np.repeat(T_month, 2, axis=0), 2, axis=1)

        patm = calc_patm(DEM_data)
        gammastar = calc_gammastar(T_month, patm)

        # if year <= 2013:
        #     CO2_month = CO2_1[(year - 2001) * 12 + month - 1, :, :]
        # elif year >= 2015:
        #     CO2_month = CO2_2[(year - 2015) * 12 + month - 1, :, :]
        #     CO2_month = np.flipud(CO2_month)
        # else:
        #     CO2_month_2013 = CO2_1[(2013 - 2001) * 12 + month - 1, :, :]
        #     CO2_month_2015 = CO2_2[(2015 - 2015) * 12 + month - 1, :, :]
        #     CO2_month_2015 = np.flipud(CO2_month_2015)
        #     CO2_month = (CO2_month_2013 + CO2_month_2015) / 2

        # CO2_month = np.repeat(np.repeat(CO2_month, 20, axis=0), 20, axis=1)

        CO2_month = dfCO2[(dfCO2['year'] == year) & (dfCO2['month'] == month)]['average'].values

        fai = (0.7 * CO2_month - gammastar) / (0.7 * CO2_month + 2 * gammastar)
        file = fr'other\fai005\fai_{year}{month:02d}.tif'
        with rasterio.open(file, 'w', driver='GTiff', height=3600, width=7200, count=1, dtype=fai.dtype,
                           crs='+proj=longlat +datum=WGS84 +no_defs',
                           transform=from_origin(-180, 90, 0.05, 0.05)) as dst:
            dst.write(fai, 1)

        print(year, month)

for year in range(2001, 2021):
    for month in range(1, 13):
        resample_tif(fr'other\fai005\fai_{year}{month:02d}.tif', fr'other\fai05\fai_{year}{month:02d}.tif',
                     target_resolution=0.5)

        print(year, month)
