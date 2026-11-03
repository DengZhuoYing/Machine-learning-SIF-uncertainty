import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline


def smooth_hants(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta, fill_val):
    '''
    Harmonic Analysis of Time Series (HANTS) implementation
    Parameters:
        ni : Time-series length
        nb : Base period length (for example, 365 days)
        nf : Number of harmonics to retain
        y  : Input time series (one-dimensional array)
        ts : Time position for each observation (for example, a day index)
        HiLo : 'Hi' Reject points above the fitted values,'Lo' Reject points below the fitted values
        low, high : Valid data range used for initial outlier rejection
        fet : Fit error tolerance; stop when the maximum error is below this value
        dod : Extra degrees of freedom used to calculate the maximum rejection count
        delta : Regularization parameter that prevents a singular matrix
        fill_val : Fill value used when there are too few valid points
    Returns:
        yr : Fitted time series


    '''

    yr = np.zeros((ni, 1))

    sHiLo = -1 if HiLo == 'Hi' else 1 if HiLo == 'Lo' else 0
    nr = min(2 * nf + 1, ni)
    noutmax = ni - nr - dod
    ang = 2 * math.pi * np.arange(nb) / nb
    cs, sn = np.cos(ang), np.sin(ang)

    i_vals = np.arange(1, nf + 1)
    ts_col = ts.reshape(-1, 1)
    indices = ((i_vals * ts_col) % nb).astype(int)

    mat = np.zeros((2 * nf + 1, ni))
    mat[0, :] = 1.0
    mat[1:2 * nf + 1:2, :] = cs[indices.T]
    mat[2:2 * nf + 1:2, :] = sn[indices.T]
    mat_T = mat.T

    p = np.ones_like(y)
    outliers = (y < low) | (y > high)
    p[outliers] = 0
    nout = np.sum(outliers)

    if nout > noutmax:
        return np.full_like(y, fill_val)

    nloop, nloopmax = 0, ni
    while nloop < nloopmax:
        nloop += 1

        weighted_mat = mat * p
        A = weighted_mat @ mat_T
        A[np.diag_indices_from(A)] += delta
        A[0, 0] -= delta
        za = weighted_mat @ y
        zr = np.linalg.solve(A, za)
        yr = mat_T @ zr

        diffVec = sHiLo * (yr - y)
        err = p * diffVec

        rank_idx = np.argsort(err)
        maxerr = diffVec[rank_idx[-1]]
        ready = (maxerr <= fet) or (nout >= noutmax)

        if ready:
            break

        desc_idx = rank_idx[::-1]
        condition = (p[desc_idx] * diffVec[desc_idx] > 0.5 * maxerr) & (p[desc_idx] != 0)
        n_remove = min(np.sum(condition), noutmax - nout)
        if n_remove > 0:
            remove_idx = desc_idx[:n_remove][condition[:n_remove]]
            p[remove_idx] = 0
            outliers[remove_idx] = 1
            nout += n_remove

    return yr


def smooth_spline(y, s=None):
    x = np.arange(len(y))
    if s is None:
        s = len(y) * np.var(y) * 0.2
    spl = UnivariateSpline(x, y, s=s, k=2)
    return spl(x)


def smooth_sg(y, window=30, order=3):
    return savgol_filter(y, window, order, mode='interp')


def linear_interpolate(arr):
    x = np.arange(len(arr))
    mask = ~np.isnan(arr)
    if mask.sum() < 2:
        return arr
    return np.interp(x, x[mask], arr[mask])


def extract_phenology(ts, thresholds):
    n_years = len(ts) // days_per_year
    sos_list, pos_list, eos_list = [], [], []
    for i in range(n_years):
        y = ts[i * days_per_year: (i + 1) * days_per_year]
        pos = np.nanargmax(y) + 1
        pos_list.append(pos)
        sos_vals, eos_vals = [], []
        for th in thresholds:
            idx = np.where(y >= th)[0]
            if len(idx) > 0:
                sos_vals.append(idx[0] + 1)
                eos_vals.append(idx[-1] + 1)
        sos_list.append(np.nanmean(sos_vals) if sos_vals else np.nan)
        eos_list.append(np.nanmean(eos_vals) if eos_vals else np.nan)
    return sos_list, pos_list, eos_list


if __name__ == '__main__':

    np.random.seed(42)
    years = np.arange(2014, 2026)
    n_years = len(years)
    days_per_year = 365  # Simplified handling for non-leap years
    total_days = 12 * 365

    t = np.arange(total_days)
    base = 1 + 0.8 * np.sin(2 * np.pi * t / 365 - np.pi / 2)
    base = base * 2  # Increase the amplitude
    noise = np.random.normal(0, 0.15, size=total_days)
    signal = base + noise

    missing_mask = np.random.random(total_days) < 0.2
    signal_missing = signal.copy()
    signal_missing[missing_mask] = np.nan

    # aa = 0
    # signal_missing[:90] = aa
    # signal_missing[365 * 1:90 + 365 * 1] = aa
    # signal_missing[365 * 2:90 + 365 * 2] = aa
    # signal_missing[365 * 3:90 + 365 * 3] = aa
    # signal_missing[365 * 4:90 + 365 * 4] = aa
    # signal_missing[365 * 5:90 + 365 * 5] = aa
    # signal_missing[365 * 6:90 + 365 * 6] = aa
    # signal_missing[365 * 7:90 + 365 * 7] = aa
    # signal_missing[365 * 8:90 + 365 * 8] = aa
    # signal_missing[365 * 9:90 + 365 * 9] = aa
    # signal_missing[365 * 10:90 + 365 * 10] = aa
    # signal_missing[365 * 11:90 + 365 * 11] = aa
    # signal_missing[-90:] = aa
    # signal_missing[-90 - 365 * 1:-365 * 1] = aa
    # signal_missing[-90 - 365 * 2:-365 * 2] = aa
    # signal_missing[-90 - 365 * 3:-365 * 3] = aa
    # signal_missing[-90 - 365 * 4:-365 * 4] = aa
    # signal_missing[-90 - 365 * 5:-365 * 5] = aa
    # signal_missing[-90 - 365 * 6:-365 * 6] = aa
    # signal_missing[-90 - 365 * 7:-365 * 7] = aa
    # signal_missing[-90 - 365 * 8:-365 * 8] = aa
    # signal_missing[-90 - 365 * 9:-365 * 9] = aa
    # signal_missing[-90 - 365 * 10:-365 * 10] = aa
    # signal_missing[-90 - 365 * 11:-365 * 11] = aa

    year_min = np.nanmin(signal_missing.reshape(n_years, days_per_year), axis=1)
    year_max = np.nanmax(signal_missing.reshape(n_years, days_per_year), axis=1)

    min_avg = np.nanmean(year_min)
    amp_avg = np.nanmean(year_max - year_min)
    thresh10 = min_avg + 0.1 * amp_avg
    thresh20 = min_avg + 0.2 * amp_avg
    thresh30 = min_avg + 0.3 * amp_avg
    thresholds = [thresh10, thresh20, thresh30]

    print(thresholds)

    true_SOS, true_POS, true_EOS = [], [], []
    for i in range(n_years):
        y = signal_missing[i * days_per_year: (i + 1) * days_per_year]
        pos = np.nanargmax(y) + 1
        true_POS.append(pos)
        sos_vals, eos_vals = [], []
        for th in thresholds:
            idx = np.where(y >= th)[0]
            if len(idx) > 0:
                sos_vals.append(idx[0] + 1)
                eos_vals.append(idx[-1] + 1)
        true_SOS.append(np.nanmean(sos_vals) if sos_vals else np.nan)
        true_EOS.append(np.nanmean(eos_vals) if eos_vals else np.nan)

    data_filled = linear_interpolate(signal_missing)

    sg = smooth_sg(data_filled)
    spline = smooth_spline(data_filled)
    hants = smooth_hants(len(data_filled), 365, 3, data_filled, np.arange(len(data_filled)), 'Hi', 0,
                         np.max(data_filled) * 1.2, 0.1, 1, 1e-3, np.nan)

    sos_sg, pos_sg, eos_sg = extract_phenology(sg, thresholds)
    sos_spline, pos_spline, eos_spline = extract_phenology(spline, thresholds)
    sos_hants, pos_hants, eos_hants = extract_phenology(hants, thresholds)

    plt.figure(figsize=(14, 6))
    doy = np.arange(1, days_per_year + 1)
    plt.plot(doy, signal_missing[:days_per_year], 'k.', markersize=2, label='Original')
    plt.plot(doy, data_filled[:days_per_year], 'r.', markersize=2, label='Nan-Filled', alpha=0.5)
    plt.plot(doy, sg[:days_per_year], 'b-', label='Savitzky-Golay')
    plt.plot(doy, spline[:days_per_year], 'g-', label='Spline')
    plt.plot(doy, hants[:days_per_year], 'm-', label='HANTS')

    plt.axhline(thresh10, color='gray', linestyle='--', alpha=0.5, label='10%')
    plt.axhline(thresh20, color='gray', linestyle='-', alpha=0.5, label='20%')
    plt.axhline(thresh30, color='gray', linestyle='-.', alpha=0.5, label='30%')
    plt.xlabel('Day of Year', fontsize=14)
    plt.ylabel('GPP', fontsize=14)
    plt.tick_params(labelsize=14)
    plt.legend(fontsize=14)
    plt.tight_layout()
    plt.savefig(r'other\2.png', dpi=300)
    plt.show()
