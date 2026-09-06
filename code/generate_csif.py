from os import makedirs
from os.path import exists
from calendar import monthrange
from multiprocessing import Pool, freeze_support
import time

import joblib
import numpy as np
import pandas as pd
import rasterio
from numpy import datetime64, pi, cos, sin, degrees, arcsin, radians


# OCO-2 crosses at approximately 13:35 local solar time.
# If training used a different solar time, set this to the same value used for training.
OCO_LOCAL_SOLAR_TIME = 13.0 + 35.0 / 60.0

MODEL_1 = r'D:\SIFcompare\code\driver_stability\other\OCO_20142017_CSIF.pkl'
MODEL_2 = r'D:\SIFcompare\code\driver_stability\other\OCO_20222025_CSIF.pkl'
MCD43C4_DIR = r'D:\SIFcompare\MCD43C4_05'
OUTPUT_DIR = r'D:\SIFcompare\CSIF'

FEATURE_COLUMNS = ['R1', 'R2', 'R3', 'R4', 'COSSZA']


def calculate_cos_sza(lon, lat, yearf, monthf, dayf,
                      hourf, minutef, secondf):
    """
    Calculate the cosine of the solar zenith angle with the supplied method.

    lon and lat may be scalars or NumPy arrays with matching shapes.
    hourf is the UTC hour; local solar time is calculated from longitude.
    """
    yearf = int(yearf)
    monthf = int(monthf)
    dayf = int(dayf)

    d = (
        datetime64(f'{yearf}-{monthf:02d}-{dayf:02d}')
        - datetime64(f'{yearf}-01-01')
    ).astype('timedelta64[D]').astype(int) + 1
    T = 2 * pi * (d - 1) / 365

    SD = (
        0.006918
        - 0.399912 * cos(T)
        + 0.070257 * sin(T)
        - 0.006758 * cos(2 * T)
        + 0.000907 * sin(2 * T)
        - 0.002697 * cos(3 * T)
        + 0.00148 * sin(3 * T)
    )
    ST = (
        hourf
        + (lon - 0) / 15
        + (minutef / 60)
        + (secondf / 3600)
    )
    w = 15 * (ST - 12)

    sun_altitude = degrees(
        arcsin(
            sin(radians(lat)) * sin(SD)
            + cos(radians(lat)) * cos(SD) * cos(radians(w))
        )
    )
    SZA = 90 - sun_altitude

    cos_sza = cos(radians(SZA))

    return cos_sza


def generate_cossza_raster(year, month, day, height, width, transform,
                            local_solar_time=OCO_LOCAL_SOLAR_TIME):
    """Generate COSSZA values corresponding to every raster row and column."""
    row_indices = np.arange(height, dtype=np.float64)
    column_indices = np.arange(width, dtype=np.float64)

    latitudes = transform.f + (row_indices + 0.5) * transform.e
    longitudes = transform.c + (column_indices + 0.5) * transform.a
    longitude_grid, latitude_grid = np.meshgrid(longitudes, latitudes)

    # hourf in calculate_cos_sza is the UTC hour.
    # OCO_LOCAL_SOLAR_TIME is local solar time, so convert it to UTC using longitude.
    # Values may fall outside 0 to 24 because the function uses continuous solar time,
    # while the date remains the local date for the current pixel.
    utc_hour_grid = local_solar_time - longitude_grid / 15.0

    cossza = calculate_cos_sza(
        lon=longitude_grid,
        lat=latitude_grid,
        yearf=year,
        monthf=month,
        dayf=day,
        hourf=utc_hour_grid,
        minutef=0,
        secondf=0,
    )

    # Set the nighttime cosine of the solar zenith angle to zero to avoid negative irradiance inputs.
    return np.clip(cossza, 0.0, 1.0).astype(np.float32)


def get_model_feature_count(model):
    """Read the input feature count from a scikit-learn, XGBoost, or LightGBM model."""
    feature_count = getattr(model, 'n_features_in_', None)
    if feature_count is not None:
        return int(feature_count)

    if hasattr(model, 'get_booster'):
        return int(model.get_booster().num_features())

    if hasattr(model, 'booster_'):
        return int(model.booster_.num_feature())

    return None


def check_model_features(model, model_path):
    """Prevent five-feature data from being passed to the original four-feature model."""
    feature_count = get_model_feature_count(model)
    if feature_count is not None and feature_count != len(FEATURE_COLUMNS):
        raise ValueError(
            f'Model {model_path} requires {feature_count} input features,'
            f'The prediction code provides {len(FEATURE_COLUMNS)} features:{FEATURE_COLUMNS}.'
            'Retrain the model with R1, R2, R3, R4, and COSSZA before prediction.'
        )


def generate(year):
    model_1 = joblib.load(MODEL_1)
    model_2 = joblib.load(MODEL_2)
    check_model_features(model_1, MODEL_1)
    check_model_features(model_2, MODEL_2)

    makedirs(OUTPUT_DIR, exist_ok=True)

    for month in range(1, 13):
        for day in range(1, monthrange(year, month)[1] + 1):
            raster = fr'{OUTPUT_DIR}\{year}{month:02d}{day:02d}.tif'
            if exists(raster):
                continue

            start = time.time()

            raster_mcd43c4 = fr'{MCD43C4_DIR}\{year}{month:02d}{day:02d}.tif'
            if not exists(raster_mcd43c4):
                print(f'Input file not found; skipping: {raster_mcd43c4}')
                continue

            with rasterio.open(raster_mcd43c4) as src:
                if src.count < 4:
                    raise ValueError(f'{raster_mcd43c4} has fewer than four bands.')

                r1 = src.read(1).astype(np.float32) / 10000.0
                r2 = src.read(2).astype(np.float32) / 10000.0
                r3 = src.read(3).astype(np.float32) / 10000.0
                r4 = src.read(4).astype(np.float32) / 10000.0
                height = src.height
                width = src.width
                transform = src.transform
                crs = src.crs

            if r1.shape != (height, width):
                raise ValueError(f'{raster_mcd43c4} has an invalid raster shape.')

            cossza = generate_cossza_raster(
                year=year,
                month=month,
                day=day,
                height=height,
                width=width,
                transform=transform,
            )

            x = pd.DataFrame({
                'R1': r1.flatten(),
                'R2': r2.flatten(),
                'R3': r3.flatten(),
                'R4': r4.flatten(),
                'COSSZA': cossza.flatten(),
            })

            df = pd.DataFrame()
            df['CSIF_1'] = model_1.predict(x[FEATURE_COLUMNS])
            df['CSIF_2'] = model_2.predict(x[FEATURE_COLUMNS])

            invalid = (
                ~np.isfinite(x['R1'])
                | ~np.isfinite(x['R2'])
                | ~np.isfinite(x['R3'])
                | ~np.isfinite(x['R4'])
                | (x['R1'] <= 0)
                | (x['R1'] >= 1)
                | (x['R2'] <= 0)
                | (x['R2'] >= 1)
                | (x['R3'] <= 0)
                | (x['R3'] >= 1)
                | (x['R4'] <= 0)
                | (x['R4'] >= 1)
                | (x['COSSZA'] <= 0)
            )

            df.loc[invalid, 'CSIF_1'] = np.nan
            df.loc[invalid, 'CSIF_2'] = np.nan

            csif_1 = df['CSIF_1'].to_numpy(dtype=np.float32).reshape(height, width)
            csif_2 = df['CSIF_2'].to_numpy(dtype=np.float32).reshape(height, width)

            with rasterio.open(
                raster,
                'w',
                driver='GTiff',
                height=height,
                width=width,
                count=2,
                dtype=np.float32,
                crs=crs,
                transform=transform,
                nodata=np.nan,
                compress='deflate',
            ) as dst:
                dst.write(csif_1, 1)
                dst.write(csif_2, 2)
                dst.set_band_description(1, 'CSIF_1')
                dst.set_band_description(2, 'CSIF_2')

            print(f'{year}{month:02d}{day:02d}', time.time() - start)


if __name__ == '__main__':
    freeze_support()
    years = [year for year in range(2014, 2026)]

    with Pool(processes=12) as pool:
        pool.map(generate, years)
