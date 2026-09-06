from os.path import exists
from numpy import zeros, nanmean, full, linspace
from pandas import DataFrame, read_csv, date_range
from numba import jit
import numba
import time
from xarray import DataArray, Dataset
from rasterio import open as rasterio_open
import warnings
import joblib
from multiprocessing import Pool
from calendar import monthrange
import rasterio.features
import numpy as np
from rasterio.transform import from_origin
import pandas as pd
import netCDF4 as nc

warnings.filterwarnings("ignore")


def generate(year):
    lgbm_1 = joblib.load(fr'D:\SIFcompare\code\driver_stability\other\OCO_20142017_GOSIF.pkl')
    lgbm_2 = joblib.load(fr'D:\SIFcompare\code\driver_stability\other\OCO_20222025_GOSIF.pkl')

    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] + 1):

            raster = fr'D:\SIFcompare\GOSIF\{year}{month:02d}{day:02d}.tif'
            if exists(raster):
                continue

            start = time.time()

            raster_MCD43C4 = fr'D:\SIFcompare\MCD43C4_05\{year}{month:02d}{day:02d}.tif'
            R1 = rasterio_open(raster_MCD43C4).read(1) / 10000
            R2 = rasterio_open(raster_MCD43C4).read(2) / 10000
            R3 = rasterio_open(raster_MCD43C4).read(3) / 10000

            nc1 = fr'H:\ENVI05\ENVI_{year}{month:02d}{day:02d}.nc4'
            dataset1 = nc.Dataset(nc1)
            par24 = dataset1.variables['par'][:]
            t2m24 = dataset1.variables['t2m'][:]
            vpd24 = dataset1.variables['vpd'][:]
            par = np.sum(par24, axis=0) / 24 / 3600
            t2m = np.sum(t2m24, axis=0) / 24 / 10
            vpd = np.sum(vpd24, axis=0) / 24 / 100

            x = DataFrame({'R1': R1.flatten(), 'R2': R2.flatten(), 'R3': R3.flatten(), 'PAR': par.flatten(),
                           'T2M': t2m.flatten(), 'VPD': vpd.flatten()})
            x['EVI'] = 2.5 * (x['R2'] - x['R1']) / (x['R2'] + 6 * x['R1'] - 7.5 * x['R3'] + 1)

            xinput = x[['EVI', 'PAR', 'T2M', 'VPD']].copy()
            df = DataFrame()
            df['GOSIF_1'] = lgbm_1.predict(xinput)
            df['GOSIF_2'] = lgbm_2.predict(xinput)

            df['GOSIF_1'] = np.where((x['EVI'] >= 1) | (x['EVI'] <= 0), np.nan, df['GOSIF_1'])
            df['GOSIF_1'] = np.where(x['VPD'] <= 0, np.nan, df['GOSIF_1'])
            df['GOSIF_1'] = np.where(x['PAR'] <= 0, np.nan, df['GOSIF_1'])
            df['GOSIF_1'] = np.where(x['T2M'] <= -30, np.nan, df['GOSIF_1'])
            df['GOSIF_2'] = np.where((x['EVI'] > 1) | (x['EVI'] < 0), np.nan, df['GOSIF_2'])
            df['GOSIF_2'] = np.where(x['VPD'] <= 0, np.nan, df['GOSIF_2'])
            df['GOSIF_2'] = np.where(x['PAR'] <= 0, np.nan, df['GOSIF_2'])
            df['GOSIF_2'] = np.where(x['T2M'] <= -30, np.nan, df['GOSIF_2'])

            GOSIF_1 = df['GOSIF_1'].values.reshape(360, 720)
            GOSIF_2 = df['GOSIF_2'].values.reshape(360, 720)

            with rasterio.open(raster, 'w', driver='GTiff', height=360, width=720, count=2, dtype=np.float32,
                               crs='+proj=longlat +datum=WGS84 +no_defs',
                               transform=from_origin(-180, 90, 0.5, 0.5)) as dst:
                dst.write(GOSIF_1, 1)
                dst.write(GOSIF_2, 2)

            print(f'{year}{month:02d}{day:02d}', time.time() - start)


if __name__ == '__main__':
    years = [year for year in range(2014, 2026)]
    pool = Pool(processes=12)
    pool.map(generate, years)
    pool.close()
    pool.join()
