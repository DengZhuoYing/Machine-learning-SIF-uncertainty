import numpy as np
import warnings
import pandas as pd
from rasterio.io import MemoryFile
import xarray as xr

warnings.filterwarnings("ignore")


def load_raster_to_memory(raster_path):
    with open(raster_path, 'rb') as f:
        memfile = MemoryFile(f.read())
    return memfile


def callc_with_mask(lc, SIF, lc_mask_values, c4_mask=None):
    result = np.zeros_like(SIF)
    non_zero_mask = (SIF != 0)

    if not non_zero_mask.any():
        return result

    lc_non_zero = lc[non_zero_mask]
    SIF_non_zero = SIF[non_zero_mask]
    result_non_zero = np.zeros_like(SIF_non_zero)

    lc_mask = np.isin(lc_non_zero, lc_mask_values)

    if c4_mask is not None:
        c4_non_zero = c4_mask[non_zero_mask]
        lc_mask = lc_mask & (c4_non_zero <= 0.5)

    lc_masked = lc_non_zero[lc_mask]
    SIF_masked = SIF_non_zero[lc_mask]

    for lc_val in np.unique(lc_masked):
        if lc_val == 1:
            coeff = 27.35
            intercept = 0.92
        elif lc_val == 2:
            coeff = 17.87
            intercept = 3.70
        elif lc_val == 4:
            coeff = 22.10
            intercept = -0.03
        elif lc_val == 5:
            coeff = 21.80
            intercept = 0.70
        elif lc_val == 7:
            coeff = 32.23
            intercept = 0.13
        elif lc_val == 8:
            coeff = 24.41
            intercept = 0.28
        elif lc_val == 9:
            coeff = 24.85
            intercept = 0.09
        elif lc_val == 10:
            coeff = 24.65
            intercept = -0.06
        elif lc_val == 11:
            coeff = 24.38
            intercept = 0.17
        elif lc_val == 12:
            coeff = 26.16
            intercept = -0.59
        else:
            coeff = 24.08
            intercept = 0.53

        mask_val = (lc_masked == lc_val)
        result_masked_val = result_non_zero[lc_mask]
        result_masked_val[mask_val] = SIF_masked[mask_val] * coeff + intercept
        result_non_zero[lc_mask] = result_masked_val

    result[non_zero_mask] = result_non_zero
    return result


def calculate_c4_only(GPPtmp, C4_data, SIF_data, fai_data):
    result = np.zeros_like(GPPtmp)
    c4_mask = C4_data > 0.5
    sif_mask = SIF_data != 0
    mask = c4_mask & sif_mask
    C3_data = 1 - C4_data
    result[mask] = GPPtmp[mask] * fai_data[mask] * C3_data[mask] + GPPtmp[mask] * C4_data[mask]

    return result


def process_landcover_type(lc_values, suffix, years_range, output_filename):
    area = np.load(r'..\generate_long_term_trend\other\area05.npy')
    df = pd.DataFrame()

    for year in range(years_range[0], years_range[1]):
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

            GPPtmp = callc_with_mask(lc_data, SIF_data, lc_values, C4_data)
            GPPtmpco2 = GPPtmp * fai_data * C3_data + GPPtmp * C4_data
            GPP_year += (GPPtmpco2 * areatmp).flatten() / np.sum(areatmp)

            print(year, month)

        df[f'GPP_{year}'] = GPP_year / 12

    yearsGPP = df.columns[:].astype(str).str[-4:].astype(int).tolist()
    sumGPP = df.sum().values
    sumGPP = (sumGPP - sumGPP[0]) / sumGPP[0] * 100
    dfGPPsum = pd.DataFrame({'Year': yearsGPP, 'Sum': sumGPP})
    dfGPPsum.to_csv(output_filename + suffix + '.csv', index=False)


def process_c4_only(years_range, output_filename):
    area = np.load(r'..\generate_long_term_trend\other\area05.npy')
    df = pd.DataFrame()

    for year in range(years_range[0], years_range[1]):
        if year <= 2019:
            rasterC4 = load_raster_to_memory(fr'other\C405\C4_{year}.tif')
        else:
            rasterC4 = load_raster_to_memory(fr'other\C405\C4_2019.tif')

        with rasterC4.open() as C4:
            C4_data = C4.read(1)
        C4_data = np.nan_to_num(C4_data, nan=0.0) / 100

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

            GPPtmp = callc_with_mask(lc_data, SIF_data, list(range(1, 13)) + [14])
            GPPtmpco2 = calculate_c4_only(GPPtmp, C4_data, SIF_data, fai_data)
            GPP_year += (GPPtmpco2 * areatmp).flatten() / np.sum(areatmp)

            print(year, month)

        df[f'GPP_{year}'] = GPP_year / 12

    yearsGPP = df.columns[:].astype(str).str[-4:].astype(int).tolist()
    sumGPP = df.sum().values
    sumGPP = (sumGPP - sumGPP[0]) / sumGPP[0] * 100
    dfGPPsum = pd.DataFrame({'Year': yearsGPP, 'Sum': sumGPP})
    dfGPPsum.to_csv(output_filename + '-C4.csv', index=False)


if __name__ == "__main__":
    years_range = (2001, 2021)
    output_filename = r'..\compare_long_term_trend\other\CSIF-CO2-linear'
    process_landcover_type([2], '-EBF', years_range, output_filename)
    process_landcover_type([1, 3, 4, 5, 8], '-OF', years_range, output_filename)
    process_landcover_type([6, 7, 9, 11], '-SW', years_range, output_filename)
    process_landcover_type([10], '-GRA', years_range, output_filename)
    process_landcover_type([12, 14], '-CRO', years_range, output_filename)
    process_c4_only(years_range, output_filename)
