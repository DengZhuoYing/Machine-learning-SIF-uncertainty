"""
Compare observed SIF under the CSIF and GOSIF predictor configurations.

Open and run this file in PyCharm. Dependencies: numpy, pandas, scipy, matplotlib.
Match observations by predictors and spatiotemporal groups, without using SIF or fitting models.
"""

from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


# ====================== User configuration ======================
DATA_FILE = Path(__file__).resolve().parents[1] / "data" / "OCO2_global_final.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "matched_sif_results"

PERIOD_EARLY = (2014, 2017)
PERIOD_LATE = (2022, 2025)
REFLECTANCE_SCALE = 10000.0  # Use 1.0 if the input reflectances have already been scaled.

# Every predictor in a pair must satisfy its absolute-difference tolerance.
# For example, R1 = 0.02 permits a scaled reflectance difference of at most 0.02.
DRIVER_TOLERANCE = {
    "R1": 0.02,
    "R2": 0.02,
    "R3": 0.02,
    "R4": 0.02,
    "COSSZA": 0.02,
}

# GOSIF tolerances apply after the unit conversions used in the reference script.
# These absolute tolerances are approximately 5% to 13% of the predictor IQRs in this dataset.
GOSIF_DRIVER_TOLERANCE = {
    "EVI": 0.02,
    "PAR": 5.0,   # W/m^2; divide raw PAR by 24 * 3600 first.
    "T2M": 1.0,   # Degrees C; divide raw T2M by 24 * 10 first.
    "VPD": 0.10,  # kPa; divide raw VPD by 24 * 100 first.
}

MATCH_SAME_MONTH = True
SPATIAL_GRID_DEG = 2.0  # Match within the same latitude-longitude cell; None disables spatial grouping.
MAX_SAMPLES_PER_PERIOD = None  # None uses all samples; set to 200000 to subsample each period.
RANDOM_SEED = 42

N_CANDIDATE_NEIGHBORS = 16  # Maximum candidate neighbors retained for each observation in group A.
QUERY_BATCH_SIZE = 50000
WORKERS = -1

SIF_DIFF_THRESHOLD = 0.10  # Threshold for a large observed difference, in the input SIF units.
EXAMPLE_MAX_DISTANCE = 0.50  # Examples use closer pairs, with every predictor difference within half its tolerance.
N_EXAMPLES = 12
N_RESPONSE_BINS = 12
N_DISTANCE_BINS = 10
MIN_PAIRS_PER_DISTANCE_BIN = 30
BASE_FONT_SIZE = 18
AXIS_LABEL_FONT_SIZE = 20
TICK_FONT_SIZE = 18
LEGEND_FONT_SIZE = 17
MODE_LEGEND_FONT_SIZE = 20
GOSIF_LEGEND_ANCHOR = (0.0, 1.0)  # Anchor the legend in panel f at the upper left.
PANEL_LABEL_FONT_SIZE = 26
PANEL_LABEL_X = -0.14  # Position relative to the left axis boundary; negative values shift labels left.
PANEL_LABEL_Y = 1.03
CASE_RIGHT_PANEL_LABEL_X_SHIFT = 0.10  # Additional rightward offset for labels b and d in figure 03.
ANNOTATION_FONT_SIZE = 18
FIGURE_DPI = 300
SHOW_FIGURES = True
# =================================================================

REFLECTANCE_COLUMNS = ["R1", "R2", "R3", "R4"]
METADATA_COLUMNS = ["source_row", "year", "month", "day", "lon", "lat"]
FEATURE_LABELS = {
    "R1": "Red refl",
    "R2": "NIR refl",
    "R3": "Blue refl",
    "R4": "Green refl",
    "COSSZA": "Cosine of solar zenith angle",
}
MODE_CONFIGS = {
    "CSIF": {
        "tolerances": DRIVER_TOLERANCE,
        "labels": FEATURE_LABELS,
        "units": {"R1": "reflectance", "R2": "reflectance", "R3": "reflectance",
                  "R4": "reflectance", "COSSZA": "dimensionless"},
    },
    "GOSIF": {
        "tolerances": GOSIF_DRIVER_TOLERANCE,
        "labels": {"EVI": "EVI", "PAR": r"PAR (W m$^{-2}$)",
                   "T2M": "Temperature (°C)", "VPD": "VPD (kPa)"},
        "units": {"EVI": "dimensionless", "PAR": "W/m²", "T2M": "℃", "VPD": "kPa"},
    },
}
EPS = 1e-10


def year_label(period):
    return f"{period[0]}-{period[1]}"


def comparison_definitions():
    early_mid = sum(PERIOD_EARLY) // 2
    late_mid = sum(PERIOD_LATE) // 2
    return {
        "cross_period": (PERIOD_EARLY, PERIOD_LATE),
        "within_early": ((PERIOD_EARLY[0], early_mid), (early_mid + 1, PERIOD_EARLY[1])),
        "within_late": ((PERIOD_LATE[0], late_mid), (late_mid + 1, PERIOD_LATE[1])),
    }


def load_data(raw, mode):
    features = list(MODE_CONFIGS[mode]["tolerances"])
    columns = ["SIF", "year", "month", "day", "lon", "lat"] + features
    n_input = len(raw)
    in_period = raw["year"].between(*PERIOD_EARLY) | raw["year"].between(*PERIOD_LATE)
    finite = np.isfinite(raw[columns].to_numpy(dtype=float)).all(axis=1)
    if mode == "CSIF":
        valid_drivers = (raw[REFLECTANCE_COLUMNS] > 0).all(axis=1)
    else:
        # Follow the reference GOSIF script: apply these filters before unit conversion.
        valid_drivers = (raw["VPD"] > 0) & (raw["PAR"] > 0) & (raw["T2M"] > 0) & (raw["EVI"] > 0)
    data = raw.loc[in_period & finite & valid_drivers, columns + ["source_row"]].copy()
    if mode == "CSIF":
        data[REFLECTANCE_COLUMNS] = data[REFLECTANCE_COLUMNS] / REFLECTANCE_SCALE
    else:
        data["PAR"] = data["PAR"] / 24 / 3600
        data["T2M"] = data["T2M"] / 24 / 10
        data["VPD"] = data["VPD"] / 24 / 100

    quartiles = data[features].quantile([0.25, 0.75])
    scales = pd.DataFrame({
        "mode": mode, "feature": features,
        "unit": [MODE_CONFIGS[mode]["units"][f] for f in features],
        "q25": quartiles.loc[0.25].to_numpy(),
        "q75": quartiles.loc[0.75].to_numpy(),
        "tolerance": [MODE_CONFIGS[mode]["tolerances"][f] for f in features],
    })
    scales["iqr"] = scales["q75"] - scales["q25"]
    scales["tolerance_over_iqr"] = scales["tolerance"] / scales["iqr"]

    group_columns = []
    if MATCH_SAME_MONTH:
        group_columns.append("month")
    if SPATIAL_GRID_DEG is not None:
        data["grid_lon"] = np.floor((data["lon"] + 180) / SPATIAL_GRID_DEG).astype(int)
        data["grid_lat"] = np.floor((data["lat"] + 90) / SPATIAL_GRID_DEG).astype(int)
        group_columns.extend(["grid_lon", "grid_lat"])
    data["match_group"] = data.groupby(group_columns, sort=True).ngroup() if group_columns else 0

    n_valid = len(data)
    if MAX_SAMPLES_PER_PERIOD is not None:
        selected = []
        for i, period in enumerate([PERIOD_EARLY, PERIOD_LATE]):
            part = data.loc[data["year"].between(*period)]
            if len(part) > MAX_SAMPLES_PER_PERIOD:
                part = part.sample(MAX_SAMPLES_PER_PERIOD, random_state=RANDOM_SEED + i)
            selected.append(part)
        data = pd.concat(selected).sort_values("source_row")
    data = data.reset_index(drop=True)
    counts = {"input_rows": n_input, "valid_rows_in_periods": n_valid, "selected_rows": len(data)}
    print(f"{mode}: {n_valid:,} valid rows in the target periods; {len(data):,} rows selected.", flush=True)
    return data, counts, scales


def find_pairs(a, b, mode="CSIF"):
    """Greedily match pairs without replacement, ordered by the maximum normalized predictor difference."""
    tolerances = MODE_CONFIGS[mode]["tolerances"]
    features = list(tolerances)
    tolerance = np.array([tolerances[f] for f in features])
    za = a[features].to_numpy() / tolerance
    zb = b[features].to_numpy() / tolerance
    # Separate group codes by 4 so candidates within distance 1 share a spatiotemporal group.
    za = np.column_stack([za, 4.0 * a["match_group"].to_numpy()])
    zb = np.column_stack([zb, 4.0 * b["match_group"].to_numpy()])
    tree = cKDTree(zb)
    ranks = np.arange(1, min(N_CANDIDATE_NEIGHBORS, len(b)) + 1)

    a_edges, b_edges, distances = [], [], []
    for start in range(0, len(a), QUERY_BATCH_SIZE):
        stop = min(start + QUERY_BATCH_SIZE, len(a))
        distance, index = tree.query(
            za[start:stop], k=ranks, p=np.inf,
            distance_upper_bound=1.0 + EPS, workers=WORKERS,
        )
        row, neighbor = np.nonzero(np.isfinite(distance))
        a_edges.append(row + start)
        b_edges.append(index[row, neighbor])
        distances.append(distance[row, neighbor])

    a_edges = np.concatenate(a_edges)
    b_edges = np.concatenate(b_edges)
    distances = np.concatenate(distances)
    # Break distance ties by sample index for reproducible matching.
    order = np.lexsort((b_edges, a_edges, distances))
    used_a = np.zeros(len(a), dtype=bool)
    used_b = np.zeros(len(b), dtype=bool)
    picked_a, picked_b = [], []
    for edge in order:
        ia, ib = a_edges[edge], b_edges[edge]
        if not used_a[ia] and not used_b[ib]:
            used_a[ia] = True
            used_b[ib] = True
            picked_a.append(ia)
            picked_b.append(ib)
    if not picked_a:
        raise ValueError("No pairs satisfy the configured tolerances and spatiotemporal groups. Adjust the matching settings.")

    aa = a.iloc[picked_a].reset_index(drop=True)
    bb = b.iloc[picked_b].reset_index(drop=True)
    pairs = pd.DataFrame({"pair_id": np.arange(1, len(aa) + 1)})
    for name in METADATA_COLUMNS + ["SIF"]:
        pairs[f"{name}_a"] = aa[name]
        pairs[f"{name}_b"] = bb[name]
    for feature in features:
        pairs[f"{feature}_a"] = aa[feature]
        pairs[f"{feature}_b"] = bb[feature]
        pairs[f"{feature}_mean"] = (aa[feature] + bb[feature]) / 2
        pairs[f"{feature}_scaled_delta"] = (bb[feature] - aa[feature]) / tolerances[feature]
    pairs["driver_distance"] = pairs[[f"{f}_scaled_delta" for f in features]].abs().max(axis=1)
    pairs["sif_diff"] = pairs["SIF_b"] - pairs["SIF_a"]
    pairs["abs_sif_diff"] = pairs["sif_diff"].abs()
    pairs["large_gap"] = pairs["abs_sif_diff"] >= SIF_DIFF_THRESHOLD
    pairs["distance_bin"] = np.minimum(
        (pairs["driver_distance"] * N_DISTANCE_BINS).astype(int), N_DISTANCE_BINS - 1
    )
    return pairs


def standardize_driver_distance(pair_sets):
    """Weight within-period controls to match the cross-period driver-distance distribution."""
    counts = {
        key: np.bincount(p["distance_bin"], minlength=N_DISTANCE_BINS)
        for key, p in pair_sets.items()
    }
    common = np.all(np.stack(list(counts.values())) >= MIN_PAIRS_PER_DISTANCE_BIN, axis=0)
    if not common.any():
        raise ValueError("Too few pairs in the shared distance strata. Reduce the number of distance bins or adjust tolerances.")
    for key, pairs in pair_sets.items():
        bins = pairs["distance_bin"].to_numpy()
        pairs["common_distance_support"] = common[bins]
        pairs["comparison_weight"] = np.where(
            common[bins], counts["cross_period"][bins] / counts[key][bins], 0.0
        )
    return common


def weighted_quantile(values, quantile, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    keep = weights > 0
    values, weights = values[keep], weights[keep]
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    positions = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(quantile, positions, values)


def make_statistics(pair_sets):
    control_values, control_weights = [], []
    for key in ["within_early", "within_late"]:
        pairs = pair_sets[key]
        control_values.append(pairs["abs_sif_diff"].to_numpy())
        weights = pairs["comparison_weight"].to_numpy()
        control_weights.append(weights / weights.sum() / 2)
    reference_q95 = weighted_quantile(np.concatenate(control_values), 0.95, np.concatenate(control_weights))

    rows, strata = [], []
    for key, pairs in pair_sets.items():
        w = pairs["comparison_weight"].to_numpy()
        abs_diff = pairs["abs_sif_diff"].to_numpy()
        rows.append({
            "comparison": key,
            "n_pairs_all": len(pairs),
            "n_pairs_common_distance_support": int((w > 0).sum()),
            "mean_sif_diff": np.average(pairs["sif_diff"], weights=w),
            "median_sif_diff": weighted_quantile(pairs["sif_diff"], 0.5, w),
            "median_abs_sif_diff": weighted_quantile(abs_diff, 0.5, w),
            "q90_abs_sif_diff": weighted_quantile(abs_diff, 0.9, w),
            "q95_abs_sif_diff": weighted_quantile(abs_diff, 0.95, w),
            "large_gap_fraction": np.average(abs_diff >= SIF_DIFF_THRESHOLD, weights=w),
            "reference_q95": reference_q95,
            "fraction_above_reference_q95": np.average(abs_diff > reference_q95, weights=w),
            "raw_median_abs_sif_diff": np.median(abs_diff),
            "raw_large_gap_fraction": np.mean(abs_diff >= SIF_DIFF_THRESHOLD),
        })
        for bin_id, part in pairs.groupby("distance_bin", sort=True):
            strata.append({
                "comparison": key, "distance_bin": bin_id,
                "distance_center": (bin_id + 0.5) / N_DISTANCE_BINS,
                "n_pairs": len(part),
                "in_common_distance_support": bool(part["common_distance_support"].iloc[0]),
                "median_abs_sif_diff": part["abs_sif_diff"].median(),
                "q25_abs_sif_diff": part["abs_sif_diff"].quantile(0.25),
                "q75_abs_sif_diff": part["abs_sif_diff"].quantile(0.75),
                "large_gap_fraction": part["large_gap"].mean(),
            })
    return pd.DataFrame(rows), pd.DataFrame(strata)


def configure_plotting():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": BASE_FONT_SIZE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelsize": AXIS_LABEL_FONT_SIZE,
        "axes.titlesize": PANEL_LABEL_FONT_SIZE,
        "xtick.labelsize": TICK_FONT_SIZE,
        "ytick.labelsize": TICK_FONT_SIZE,
        "legend.fontsize": LEGEND_FONT_SIZE,
        "savefig.facecolor": "white", "figure.facecolor": "white",
        "pdf.fonttype": 42, "svg.fonttype": "none",
    })


def set_panel_label(ax, letter, x_offset=0.0):
    ax.text(PANEL_LABEL_X + x_offset, PANEL_LABEL_Y, letter, transform=ax.transAxes,
            fontsize=PANEL_LABEL_FONT_SIZE, fontweight="bold",
            ha="left", va="bottom", clip_on=False)


def save_figure(fig, filename):
    fig.savefig(OUTPUT_DIR / filename, dpi=FIGURE_DPI, bbox_inches="tight")
    if not SHOW_FIGURES:
        plt.close(fig)


def draw_response_panel(ax, cross, mode, feature, letter):
    bins = pd.qcut(cross[f"{feature}_mean"], N_RESPONSE_BINS, duplicates="drop")
    rows = []
    for bin_id, (_, part) in enumerate(cross.groupby(bins, observed=True)):
        rows.append({
            "mode": mode, "feature": feature, "panel": letter,
            "bin_id": bin_id, "n_pairs": len(part),
            "driver_median": part[f"{feature}_mean"].median(),
            "SIF_a_median": part["SIF_a"].median(), "SIF_b_median": part["SIF_b"].median(),
            "SIF_a_q25": part["SIF_a"].quantile(0.25), "SIF_a_q75": part["SIF_a"].quantile(0.75),
            "SIF_b_q25": part["SIF_b"].quantile(0.25), "SIF_b_q75": part["SIF_b"].quantile(0.75),
            "median_sif_diff": part["sif_diff"].median(),
            "median_abs_sif_diff": part["abs_sif_diff"].median(),
            "large_gap_fraction": part["large_gap"].mean(),
        })
    values = pd.DataFrame(rows)
    x = values["driver_median"]
    ax.fill_between(x, values["SIF_a_q25"], values["SIF_a_q75"], color="#246B99", alpha=0.14)
    ax.fill_between(x, values["SIF_b_q25"], values["SIF_b_q75"], color="#BA3B46", alpha=0.14)
    ax.plot(x, values["SIF_a_median"], "o-", color="#246B99", label=year_label(PERIOD_EARLY), markersize=3)
    ax.plot(x, values["SIF_b_median"], "o-", color="#BA3B46", label=year_label(PERIOD_LATE), markersize=3)
    ax.set(xlabel=MODE_CONFIGS[mode]["labels"][feature], ylabel="Observed SIF")
    set_panel_label(ax, letter)
    ax.grid(axis="y", alpha=0.18)
    if letter in ["a", "f"]:
        if letter == "f":
            legend = ax.legend(title='GOSIF mode', title_fontsize=MODE_LEGEND_FONT_SIZE,
                               frameon=False, fontsize=LEGEND_FONT_SIZE,
                               loc="upper left", bbox_to_anchor=GOSIF_LEGEND_ANCHOR,
                               borderaxespad=0.0)
        else:
            legend = ax.legend(title='CSIF mode', title_fontsize=MODE_LEGEND_FONT_SIZE,
                               frameon=False, fontsize=LEGEND_FONT_SIZE, loc="upper right")
        # legend.get_title().set_fontweight("bold")
    return rows


def plot_response_curves(results):
    fig = plt.figure(figsize=(19, 22), layout="constrained")
    mode_blocks = fig.add_gridspec(2, 1, hspace=0.12)
    response_rows = []
    for block, (mode, letter_offset) in enumerate([("CSIF", 0), ("GOSIF", 5)]):
        if mode == "CSIF":
            grid = mode_blocks[block].subgridspec(2, 6)
            positions = [(0, 0, 2), (0, 2, 4), (0, 4, 6), (1, 1, 3), (1, 3, 5)]
        else:
            grid = mode_blocks[block].subgridspec(2, 2)
            positions = [(0, 0, 1), (0, 1, 2), (1, 0, 1), (1, 1, 2)]
        features = list(MODE_CONFIGS[mode]["tolerances"])
        for panel, feature in enumerate(features):
            row, start, stop = positions[panel]
            ax = fig.add_subplot(grid[row, start:stop])
            response_rows.extend(draw_response_panel(
                ax, results[mode]["cross"], mode, feature, chr(97 + letter_offset + panel)
            ))
    save_figure(fig, "02_matched_sif_response_curves.png")
    return pd.DataFrame(response_rows)


def draw_examples_row(fig, grid, top, mode, row, first_letter):
    features = list(MODE_CONFIGS[mode]["tolerances"])
    n = len(top)
    ax1 = fig.add_subplot(grid[row, 0])
    ax2 = fig.add_subplot(grid[row, 1], sharey=ax1)
    values = top[[f"{f}_scaled_delta" for f in features]].to_numpy()
    image = ax1.imshow(values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for index in range(n):
        for col in range(len(features)):
            ax1.text(col, index, f"{values[index, col]:+.2f}", ha="center", va="center", fontsize=ANNOTATION_FONT_SIZE)
    ax1.set_xticks(np.arange(len(features)), features)
    ax1.set_yticks(np.arange(n), [f"#{i + 1}  D={d:.2f}" for i, d in enumerate(top["driver_distance"])])
    ax1.set(ylabel=f"{mode}\nExample rank and driver distance")
    set_panel_label(ax1, first_letter)
    cax = fig.add_subplot(grid[row + 1, 0])
    fig.colorbar(image, cax=cax, orientation="horizontal", label="(Later value - earlier value) / tolerance")

    y = np.arange(n)
    ax2.hlines(y, top[["SIF_a", "SIF_b"]].min(axis=1), top[["SIF_a", "SIF_b"]].max(axis=1), color="#B7C2CA", linewidth=2)
    ax2.scatter(top["SIF_a"], y, color="#246B99", label=year_label(PERIOD_EARLY), s=45, zorder=3)
    ax2.scatter(top["SIF_b"], y, color="#BA3B46", label=year_label(PERIOD_LATE), s=45, zorder=3)
    left = top[["SIF_a", "SIF_b"]].min().min()
    right = top[["SIF_a", "SIF_b"]].max().max()
    span = right - left
    for index, (_, pair) in enumerate(top.iterrows()):
        ax2.text(max(pair["SIF_a"], pair["SIF_b"]) + span * 0.035, index, f"delta = {pair['sif_diff']:+.3f}", va="center", fontsize=ANNOTATION_FONT_SIZE)
    ax2.set(xlabel="Observed SIF", xlim=(left - 0.05 * span, right + 0.35 * span))
    set_panel_label(ax2, chr(ord(first_letter) + 1), x_offset=CASE_RIGHT_PANEL_LABEL_X_SHIFT)
    ax2.tick_params(axis="y", labelleft=False)
    ax1.set_ylim(n - 0.5, -0.5)
    legend_ax = fig.add_subplot(grid[row + 1, 1])
    legend_ax.axis("off")
    legend_ax.legend(*ax2.get_legend_handles_labels(), loc="center", ncol=2, frameon=False)


def plot_examples(results):
    height = max(7.5, N_EXAMPLES * 0.65 + 2.0) * 2
    fig = plt.figure(figsize=(20, height), layout="constrained")
    grid = fig.add_gridspec(4, 2, height_ratios=[1, 0.045, 1, 0.045], width_ratios=[1, 1.5])
    draw_examples_row(fig, grid, results["CSIF"]["top"], "CSIF", 0, "a")
    draw_examples_row(fig, grid, results["GOSIF"]["top"], "GOSIF", 2, "c")
    save_figure(fig, "03_large_sif_differences_under_matched_drivers.png")


def write_run_notes(summary, counts, top, common, scales, mode):
    table = summary.set_index("comparison")
    cross = table.loc["cross_period"]
    control_median = float(np.mean(table.loc[["within_early", "within_late"], "median_abs_sif_diff"]))
    controls = comparison_definitions()
    notes = [
        f"SIF difference analysis under the {mode} predictor configuration",
        "",
        f"Data file: {DATA_FILE}",
        f"Input rows: {counts['input_rows']:,}",
        f"Valid rows in the target periods: {counts['valid_rows_in_periods']:,}",
        f"Selected rows: {counts['selected_rows']:,}",
        f"Comparison periods: {year_label(PERIOD_EARLY)} and {year_label(PERIOD_LATE)}",
        f"Predictors: {', '.join(MODE_CONFIGS[mode]['tolerances'])}",
        f"Predictor tolerances: {MODE_CONFIGS[mode]['tolerances']}",
        f"Match calendar month: {MATCH_SAME_MONTH}; spatial cell size: {SPATIAL_GRID_DEG} degrees",
        "CSIF: divide R1, R2, R3, and R4 by 10000.",
        "GOSIF: retain EVI; divide PAR by 24 and 3600, T2M by 24 and 10, and VPD by 24 and 100.",
        "GOSIF filters follow the reference script: require VPD > 0, PAR > 0, T2M > 0, and EVI > 0 before unit conversion.",
        "Retain original SIF values, including negative values. Match and select examples separately for CSIF and GOSIF.",
        "",
        "Matching: find neighbors using predictors, then accept one-to-one pairs in ascending distance order.",
        f"Retain up to {N_CANDIDATE_NEIGHBORS} candidate neighbors per group A observation. Do not reuse observations within a comparison.",
        "Driver distance D = max_j(abs(difference in predictor j) / tolerance for predictor j).",
        "All accepted pairs satisfy D <= 1. SIF is excluded from matching.",
        "Within-period controls use observations from non-overlapping groups of years.",
        "Match each comparison independently. An observation may occur once in both a cross-period and a within-period comparison.",
        "",
        "Statistics: divide D from 0 to 1 into equal-width bins.",
        f"Shared support includes bins with at least {MIN_PAIRS_PER_DISTANCE_BIN} pairs in every comparison.",
        f"Shared-support bin indices: {np.flatnonzero(common).tolist()}; zero-based indices, {N_DISTANCE_BINS} bins in total.",
        "Weight within-period controls by the cross-period distance-bin proportions. Main metrics in summary.csv use these weights.",
        "Metrics prefixed with raw_ use all pairs. Each within-period control receives half the weight in the pooled reference distribution.",
        "",
        f"Cross-period pairs: {int(cross['n_pairs_all']):,}",
        f"Cross-period pairs in shared distance support: {int(cross['n_pairs_common_distance_support']):,}",
        f"Median signed cross-period SIF difference: {cross['median_sif_diff']:.6f}",
        f"Median absolute cross-period SIF difference: {cross['median_abs_sif_diff']:.6f}",
        f"Mean of the two within-period median absolute differences: {control_median:.6f}",
        f"Cross-period / within-period median ratio: {cross['median_abs_sif_diff'] / control_median:.4f}",
        f"Large-difference threshold: {SIF_DIFF_THRESHOLD:.6f}, in the original SIF units",
        f"Fraction of large cross-period differences: {cross['large_gap_fraction']:.3%}",
        f"95th percentile of the pooled within-period reference: {cross['reference_q95']:.6f}",
        f"Cross-period fraction above the reference 95th percentile: {cross['fraction_above_reference_q95']:.3%}",
        f"Examples show {len(top)} pairs satisfying D <= {EXAMPLE_MAX_DISTANCE} and absolute SIF difference >= {SIF_DIFF_THRESHOLD}.",
        "",
        "Differences are defined as group B minus group A. In cross-period comparisons, A is earlier and B is later.",
        "Figure outputs are numbered 02 and 03.",
        "CSIF and GOSIF occupy separate row blocks in each figure.",
        "In figure 02, panels a to e show the five CSIF predictors and panels f to i show the four GOSIF predictors.",
        "The two period curves in figure 02 use the same matched pairs in each bin; shading shows the observed interquartile ranges.",
        "Figure 02 shows CSIF above GOSIF, with mode names in the legend titles of panels a and f.",
        "Anchor the legend of figure 02 panel f at the upper left; shift labels b and d in figure 03 to the right.",
        "Figure 03 shows CSIF in panels a and b above GOSIF in panels c and d. Rank each mode by absolute SIF difference.",
        "Use bold panel letters at the upper left of both figures, with a leftward base offset.",
        "Rank examples to display individual observations. Overall statistics use the complete matching results.",
        "Within-period distributions provide an empirical reference for observed differences. Thresholds and reference quantiles describe their magnitude.",
        "source_row_a and source_row_b are zero-based data-row indices in the original CSV.",
        "",
        "Comparisons:",
    ]
    notes.extend(["", "Predictor scales and tolerances:", scales.to_string(index=False)])
    for key, (pa, pb) in controls.items():
        notes.append(f"{key}: {year_label(pa)} versus {year_label(pb)}")
    mode_dir = OUTPUT_DIR / mode
    (mode_dir / "run_notes.txt").write_text("\n".join(notes), encoding="utf-8-sig")
    settings = {
        "data_file": str(DATA_FILE), "period_early": PERIOD_EARLY, "period_late": PERIOD_LATE,
        "mode": mode, "reflectance_scale": REFLECTANCE_SCALE,
        "driver_tolerance": MODE_CONFIGS[mode]["tolerances"],
        "match_same_month": MATCH_SAME_MONTH, "spatial_grid_deg": SPATIAL_GRID_DEG,
        "max_samples_per_period": MAX_SAMPLES_PER_PERIOD, "random_seed": RANDOM_SEED,
        "n_candidate_neighbors": N_CANDIDATE_NEIGHBORS, "sif_diff_threshold": SIF_DIFF_THRESHOLD,
        "example_max_distance": EXAMPLE_MAX_DISTANCE, "n_distance_bins": N_DISTANCE_BINS,
        "min_pairs_per_distance_bin": MIN_PAIRS_PER_DISTANCE_BIN,
        "panel_label_x": PANEL_LABEL_X, "panel_label_y": PANEL_LABEL_Y,
        "mode_legend_font_size": MODE_LEGEND_FONT_SIZE,
        "gosif_legend_anchor": GOSIF_LEGEND_ANCHOR,
        "case_right_panel_label_x_shift": CASE_RIGHT_PANEL_LABEL_X_SHIFT,
    }
    (mode_dir / "run_settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")


def analyze_mode(raw, mode):
    mode_dir = OUTPUT_DIR / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    data, counts, scales = load_data(raw, mode)
    pair_sets = {}
    for key, (period_a, period_b) in comparison_definitions().items():
        a = data.loc[data["year"].between(*period_a)].reset_index(drop=True)
        b = data.loc[data["year"].between(*period_b)].reset_index(drop=True)
        print(f"Starting {mode} / {key}: {len(a):,} versus {len(b):,} observations.", flush=True)
        pair_sets[key] = find_pairs(a, b, mode)
        print(f"Matched {len(pair_sets[key]):,} pairs.", flush=True)
    common = standardize_driver_distance(pair_sets)
    summary, strata = make_statistics(pair_sets)
    summary.insert(0, "mode", mode)
    strata.insert(0, "mode", mode)
    cross = pair_sets["cross_period"]
    large = cross.loc[cross["large_gap"]].sort_values("abs_sif_diff", ascending=False)
    top = large.loc[large["driver_distance"] <= EXAMPLE_MAX_DISTANCE + EPS].head(N_EXAMPLES).copy()

    for key, pairs in pair_sets.items():
        pairs.to_csv(mode_dir / f"pairs_{key}.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    for filename, frame in [
        ("summary.csv", summary), ("distance_strata_summary.csv", strata),
        ("large_gap_cases.csv", large), ("top_examples.csv", top),
        ("driver_scale_and_tolerances.csv", scales),
    ]:
        frame.to_csv(mode_dir / filename, index=False, encoding="utf-8-sig", float_format="%.10g")
    write_run_notes(summary, counts, top, common, scales, mode)
    print(summary[["mode", "comparison", "n_pairs_all", "median_abs_sif_diff", "large_gap_fraction"]].to_string(index=False), flush=True)
    return {"cross": cross, "top": top, "summary": summary, "scales": scales, "counts": counts}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_plotting()
    columns = ["SIF", "year", "month", "day", "lon", "lat"]
    for config in MODE_CONFIGS.values():
        columns.extend(config["tolerances"])
    raw = pd.read_csv(DATA_FILE, usecols=list(dict.fromkeys(columns)))
    raw["source_row"] = np.arange(len(raw))
    print(f"Loaded {len(raw):,} input rows.", flush=True)
    results = {mode: analyze_mode(raw, mode) for mode in MODE_CONFIGS}
    response = plot_response_curves(results)
    response.to_csv(OUTPUT_DIR / "response_bins.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    for mode, part in response.groupby("mode", sort=False):
        part.to_csv(OUTPUT_DIR / mode / "response_bins.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    plot_examples(results)
    combined_summary = pd.concat([result["summary"] for result in results.values()], ignore_index=True)
    combined_scales = pd.concat([result["scales"] for result in results.values()], ignore_index=True)
    combined_summary.to_csv(OUTPUT_DIR / "summary.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    combined_scales.to_csv(OUTPUT_DIR / "driver_scale_and_tolerances.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
    print("\nResults for both predictor configurations:", flush=True)
    print(combined_summary[["mode", "comparison", "n_pairs_all", "median_abs_sif_diff", "large_gap_fraction"]].to_string(index=False), flush=True)
    print(f"\nOutput directory: {OUTPUT_DIR}", flush=True)
    if SHOW_FIGURES:
        plt.show()


if __name__ == "__main__":
    main()
