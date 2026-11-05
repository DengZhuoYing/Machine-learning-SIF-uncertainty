from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from scipy.interpolate import UnivariateSpline


HANTS_START_YEARS = {365: 2014, 92: 2001, 46: 2001}
HANTS_STEP_DAYS = {92: 4, 46: 8}
POS_SUPPORT_DAYS = 15


def _make_hants_dates(ts, nb, start_year):
    if start_year is None:
        start_year = HANTS_START_YEARS[nb]

    sample_indices = np.asarray(ts, dtype=np.int64)
    if nb == 365:
        first_day = np.datetime64(f'{start_year}-01-01', 'D')
        return first_day + sample_indices.astype('timedelta64[D]')

    years = start_year + sample_indices // nb
    year_starts = (years - 1970).astype('datetime64[Y]').astype('datetime64[D]')
    day_offsets = (sample_indices % nb) * HANTS_STEP_DAYS[nb]
    return year_starts + day_offsets.astype('timedelta64[D]')


def _annual_coordinates(dates):
    dates = np.asarray(dates, dtype='datetime64[D]')
    calendar_years = dates.astype('datetime64[Y]')
    for year in np.unique(calendar_years):
        indices = np.flatnonzero(calendar_years == year)
        first_day = year.astype('datetime64[D]')
        next_year = (year + np.timedelta64(1, 'Y')).astype('datetime64[D]')
        period = float((next_year - first_day) / np.timedelta64(1, 'D'))
        day_offsets = (dates[indices] - first_day).astype('timedelta64[D]').astype(float)
        yield indices, day_offsets, period


def _fit_hants_year(y, ts, period, nf, HiLo, low, high, fet, dod, delta, fill_val):
    """Fit an independent set of harmonic coefficients for one calendar year."""
    valid = np.isfinite(y) & (y >= low) & (y <= high)
    coefficient_count = 2 * nf + 1
    min_points = coefficient_count + dod
    if np.count_nonzero(valid) < min_points:
        return np.full(len(y), fill_val, dtype=float)

    angles = 2 * np.pi * np.arange(1, nf + 1)[:, None] * ts[None, :] / period
    mat = np.empty((coefficient_count, len(y)), dtype=float)
    mat[0, :] = 1.0
    mat[1::2, :] = np.cos(angles)
    mat[2::2, :] = np.sin(angles)

    weights = valid.astype(float)
    observed = np.where(valid, y, 0.0)
    direction = -1 if HiLo == 'Hi' else 1 if HiLo == 'Lo' else 0

    while True:
        weighted_mat = mat * weights
        normal_matrix = weighted_mat @ mat.T
        normal_matrix[np.diag_indices_from(normal_matrix)] += delta
        normal_matrix[0, 0] -= delta
        coefficients = np.linalg.solve(normal_matrix, weighted_mat @ observed)
        fitted = mat.T @ coefficients

        errors = direction * (fitted - observed)
        errors[weights == 0] = -np.inf
        max_error = np.max(errors)
        remaining = int(np.sum(weights)) - min_points
        if max_error <= fet or remaining == 0:
            return fitted

        candidates = np.flatnonzero(errors > 0.5 * max_error)
        candidates = candidates[np.argsort(errors[candidates])[::-1]]
        remove_count = min(len(candidates), remaining)
        weights[candidates[:remove_count]] = 0.0


def smooth_hants(ni, nb, nf, y, ts, HiLo, low, high, fet, dod, delta, fill_val,
                 *, dates=None, start_year=None):
    """
    Fit HANTS independently for each calendar year.

    ni: Number of observations.
    nb: Annual sampling scheme: 365 for daily observations, 92 for four-day
        observations, or 46 for eight-day observations. Calendar dates determine
        the actual annual period of 365 or 366 days.
    nf: Number of harmonics fitted separately for each year.
    y: One-dimensional observations. Non-finite and out-of-range values have
        zero fitting weight.
    ts: Zero-based sample indices used when dates are omitted.
    HiLo: Hi rejects points above the fitted curve; Lo rejects points below it.
    low, high: Inclusive observation limits.
    fet: Fit error tolerance.
    dod: Valid observations retained in addition to the fitted coefficients.
    delta: Regularization applied to harmonic coefficients, excluding the mean.
    fill_val: Output for years with insufficient valid observations.
    dates: Actual observation dates, including gaps and partial years.
    start_year: First year used to construct dates from the sampling scheme.
        Defaults are defined by HANTS_START_YEARS.

    Return a one-dimensional fitted series in the original observation order.
    """
    y = np.asarray(y, dtype=float)
    if dates is None:
        dates = _make_hants_dates(ts, nb, start_year)

    result = np.full(ni, fill_val, dtype=float)
    for indices, year_time, period in _annual_coordinates(dates):
        result[indices] = _fit_hants_year(
            y[indices], year_time, period, nf, HiLo,
            low, high, fet, dod, delta, fill_val
        )
    return result


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


def extract_supported_pos(year_fit, year_doy, year_observed,
                          max_gap_days=POS_SUPPORT_DAYS):
    """Select the highest peak supported by the original observations.

    A candidate must lie within max_gap_days of a finite observation, measured
    in calendar days. Annual endpoint maxima are candidates too. Return NaN
    for a flat curve or when no candidate has observation support.
    """
    year_fit = np.asarray(year_fit, dtype=float)
    year_doy = np.asarray(year_doy, dtype=float)
    observed_days = year_doy[np.isfinite(year_observed)]
    finite_fit = year_fit[np.isfinite(year_fit)]
    if observed_days.size == 0 or finite_fit.size == 0:
        return np.nan
    if np.ptp(finite_fit) == 0:
        return np.nan

    peaks, _ = find_peaks(year_fit)
    peaks = np.unique(np.append(peaks, np.nanargmax(year_fit)))
    gaps = np.min(np.abs(year_doy[peaks, None] - observed_days[None, :]), axis=1)
    peaks = peaks[gaps <= max_gap_days]
    if peaks.size == 0:
        return np.nan

    peak_index = peaks[np.argmax(year_fit[peaks])]
    return float(year_doy[peak_index])


def extract_phenology(ts, thresholds, days_per_year=365, dates=None,
                      *, observed=None, max_gap_days=POS_SUPPORT_DAYS):
    """Extract annual SOS, POS and EOS using actual day-of-year coordinates.

    observed: Original observations before interpolation, zero filling or
        smoothing. Preserve NaN for missing and invalid observations. The
        array has the same dates and ordering as ts. When omitted, ts supplies
        the observation support for direct extraction from observed data.
    max_gap_days: Maximum calendar-day distance from a POS candidate to a
        finite original observation. Defaults to POS_SUPPORT_DAYS.

    SOS and EOS use the existing threshold crossings. POS uses the highest
    supported local peak instead of a maximum extrapolated across a long gap.
    """
    ts = np.asarray(ts, dtype=float)
    observed = ts if observed is None else np.asarray(observed, dtype=float)
    if dates is None:
        groups = (
            (np.arange(start, start + days_per_year),
             np.arange(days_per_year, dtype=float), float(days_per_year))
            for start in range(0, len(ts) - days_per_year + 1, days_per_year)
        )
    else:
        groups = _annual_coordinates(dates)

    sos_list, pos_list, eos_list = [], [], []
    for indices, day_offsets, _ in groups:
        y = ts[indices]
        doy = day_offsets + 1
        if not np.any(np.isfinite(y)):
            sos_list.append(np.nan)
            pos_list.append(np.nan)
            eos_list.append(np.nan)
            continue

        pos_list.append(extract_supported_pos(
            y, doy, observed[indices], max_gap_days=max_gap_days
        ))
        sos_vals, eos_vals = [], []
        for threshold in thresholds:
            above = np.flatnonzero(y >= threshold)
            if len(above) > 0:
                sos_vals.append(doy[above[0]])
                eos_vals.append(doy[above[-1]])
        sos_list.append(np.mean(sos_vals) if sos_vals else np.nan)
        eos_list.append(np.mean(eos_vals) if eos_vals else np.nan)
    return sos_list, pos_list, eos_list


if __name__ == '__main__':
    rng = np.random.default_rng(42)
    years = np.arange(2014, 2026)
    dates = np.arange(np.datetime64('2014-01-01'), np.datetime64('2026-01-01'))
    annual_groups = list(_annual_coordinates(dates))

    # Give each year a different peak date.
    base = np.empty(len(dates), dtype=float)
    for year_index, (indices, day_offsets, period) in enumerate(annual_groups):
        shift = 15 * np.sin(2 * np.pi * year_index / len(years))
        base[indices] = 2 + 1.6 * np.sin(
            2 * np.pi * (day_offsets - shift) / period - np.pi / 2
        )

    signal = base + rng.normal(0, 0.15, size=len(dates))
    signal_missing = signal.copy()
    signal_missing[rng.random(len(dates)) < 0.2] = np.nan

    year_min = np.array([np.nanmin(signal_missing[idx]) for idx, _, _ in annual_groups])
    year_max = np.array([np.nanmax(signal_missing[idx]) for idx, _, _ in annual_groups])
    min_avg = np.mean(year_min)
    amp_avg = np.mean(year_max - year_min)
    thresholds = min_avg + np.array([0.1, 0.2, 0.3]) * amp_avg

    data_filled = linear_interpolate(signal_missing)
    sg = smooth_sg(data_filled)
    spline = smooth_spline(data_filled)
    hants = smooth_hants(
        len(signal_missing), 365, 3, signal_missing,
        np.arange(len(signal_missing)), 'Hi', 0,
        np.nanmax(signal_missing) * 1.2, 0.1, 1, 1e-3, np.nan,
        dates=dates
    )

    true_sos, true_pos, true_eos = extract_phenology(base, thresholds, dates=dates)
    sos_sg, pos_sg, eos_sg = extract_phenology(
        sg, thresholds, dates=dates, observed=signal_missing
    )
    sos_spline, pos_spline, eos_spline = extract_phenology(
        spline, thresholds, dates=dates, observed=signal_missing
    )
    sos_hants, pos_hants, eos_hants = extract_phenology(
        hants, thresholds, dates=dates, observed=signal_missing
    )

    first_year = annual_groups[0][0]
    doy = annual_groups[0][1] + 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(doy, signal_missing[first_year], 'k.', markersize=2, label='Original')
    axes[0].plot(doy, sg[first_year], 'b-', label='Savitzky-Golay')
    axes[0].plot(doy, spline[first_year], 'g-', label='Spline')
    axes[0].plot(doy, hants[first_year], 'm-', label='HANTS')
    for threshold in thresholds:
        axes[0].axhline(threshold, color='gray', linestyle='--', alpha=0.5)
    axes[0].set(xlabel='Day of year', ylabel='GPP', title=str(years[0]))
    axes[0].legend()

    axes[1].plot(years, true_pos, 'k--', label='True POS')
    axes[1].plot(years, pos_sg, 'bo-', label='Savitzky-Golay')
    axes[1].plot(years, pos_spline, 'go-', label='Spline')
    axes[1].plot(years, pos_hants, 'mo-', label='HANTS by year')
    axes[1].set(xlabel='Year', ylabel='POS (day of year)')
    axes[1].legend()
    fig.tight_layout()
    output_dir = Path(__file__).resolve().parent / 'other'
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / '2.png', dpi=300)
    plt.show()

