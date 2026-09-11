from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb


DATA_FILE = Path(r"other\OCO2_global_final.csv")
OUTPUT_FIGURE_NAME = "shap_interaction_pairwise_model_differences_CSIF.png"

FEATURE_NAMES = ["R1", "R2", "R3", "R4", "COSSZA"]
FEATURE_LABELS = {
    "R1": "Red Refl",
    "R2": "NIR Refl",
    "R3": "Blue Refl",
    "R4": "Green Refl",
    "COSSZA": "cos(SZA)",
}
REFLECTANCE_FEATURES = ["R1", "R2", "R3", "R4"]
TARGET_NAME = "SIF"
YEAR_NAME = "year"

PERIODS = {
    "2014-2017": (2014, 2017),
    "2022-2025": (2022, 2025),
}

# =========================
# Interaction pair selection
# =========================
# Specify six distinct variable pairs.
# In each tuple, the first variable defines the x-axis and the second controls color.
# Available variables: R1, R2, R3, R4, and COSSZA.

SELECTED_INTERACTION_PAIRS = [
    ("R1", "R2"),
    ("R2", "R3"),
    ("R1", "R3"),
    ("R1", "R4"),
    ("R2", "R4"),
    ("R3", "R4"),
]


RANDOM_STATE = 42
INTERACTION_SAMPLE_SIZE = 10000

# Use the same samples to calculate interaction SHAP values for both models.
# When True, retain common feature support before sampling each period equally.
USE_COMMON_SUPPORT = True
COMMON_SUPPORT_QUANTILES = (0.01, 0.99)
MIN_REFERENCE_ROWS = 500

N_CURVE_BINS = 30
FIGURE_DPI = 600
SHOW_FIGURE = True

# Cover the middle 96% of the combined interaction values on the y-axis.
# Allow rounded ticks to clip at most 5% of this range to avoid doubled intervals.
Y_AXIS_QUANTILES = (0.02, 0.98)
Y_TICK_MAX_CLIP_FRACTION = 0.05

EARLY_LINE_COLOR = "black"
LATE_LINE_COLOR = "gray"

MODEL_PARAMS = {
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}


def resolve_data_file(data_file: Path) -> Path:
    """Find the data file in the working directory or near this script."""
    script_dir = Path(__file__).resolve().parent

    if data_file.is_absolute():
        candidates = [data_file]
    else:
        candidates = [
            Path.cwd() / data_file,
            script_dir / data_file,
            script_dir.parent / data_file,
        ]

    checked_paths = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate not in checked_paths:
            checked_paths.append(candidate)
        if candidate.is_file():
            return candidate

    checked_text = "\n".join(f"  {path}" for path in checked_paths)
    raise FileNotFoundError(
        "OCO2_global_final.csv was not found. Update DATA_FILE.\n"
        f"Checked paths:\n{checked_text}"
    )


def prepare_data(data_path: Path) -> pd.DataFrame:
    """Read data, retain positive reflectances, and scale reflectance values."""
    df = pd.read_csv(data_path)
    required_columns = [YEAR_NAME, TARGET_NAME, *FEATURE_NAMES]
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise KeyError(f"Missing data columns: {missing_columns}")

    df = df.copy()
    for column in required_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    original_count = len(df)
    df.dropna(subset=required_columns, inplace=True)

    positive_mask = (df[REFLECTANCE_FEATURES] > 0).all(axis=1)
    df = df.loc[positive_mask].copy()
    df.loc[:, REFLECTANCE_FEATURES] = df[REFLECTANCE_FEATURES] / 10000.0

    print(f"Loaded rows: {original_count:,}")
    print(f"Rows after cleaning: {len(df):,}")
    return df


def split_period_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Extract training data for both periods."""
    period_data = {}

    for period, (start_year, end_year) in PERIODS.items():
        subset = df.loc[df[YEAR_NAME].between(start_year, end_year)].copy()
        if subset.empty:
            raise ValueError(f"No usable data for period {period}")

        period_data[period] = subset
        print(f"{period} training rows: {len(subset):,}")

    if len(period_data) != 2:
        raise ValueError("PERIODS must define exactly two periods")

    return period_data


def train_models(
    period_data: dict[str, pd.DataFrame],
) -> dict[str, xgb.XGBRegressor]:
    """Train one XGBoost model for each period without saving model files."""
    models = {}

    for period, subset in period_data.items():
        print(f"Training the {period} model")
        model = xgb.XGBRegressor(**MODEL_PARAMS)
        model.fit(subset[FEATURE_NAMES], subset[TARGET_NAME])
        models[period] = model

    return models


def calculate_common_support(
    period_data: dict[str, pd.DataFrame],
) -> dict[str, tuple[float, float]]:
    """Calculate the shared feature support across both periods."""
    lower_quantile, upper_quantile = COMMON_SUPPORT_QUANTILES
    common_support = {}

    for feature in FEATURE_NAMES:
        period_bounds = []
        for subset in period_data.values():
            period_bounds.append(
                (
                    float(subset[feature].quantile(lower_quantile)),
                    float(subset[feature].quantile(upper_quantile)),
                )
            )

        common_lower = max(bounds[0] for bounds in period_bounds)
        common_upper = min(bounds[1] for bounds in period_bounds)
        common_support[feature] = (common_lower, common_upper)

    return common_support


def filter_by_common_support(
    subset: pd.DataFrame,
    common_support: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    mask = pd.Series(True, index=subset.index)

    for feature, (lower, upper) in common_support.items():
        if lower >= upper:
            return subset.iloc[0:0].copy()
        mask &= subset[feature].between(lower, upper)

    return subset.loc[mask].copy()


def build_reference_sample(
    period_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Sample both periods equally to create a shared explanation dataset."""
    common_support = calculate_common_support(period_data)

    if USE_COMMON_SUPPORT:
        candidate_data = {
            period: filter_by_common_support(subset, common_support)
            for period, subset in period_data.items()
        }

        if min(len(subset) for subset in candidate_data.values()) < MIN_REFERENCE_ROWS:
            print("Too few samples in common support; using all cleaned period data")
            candidate_data = {
                period: subset.copy()
                for period, subset in period_data.items()
            }
    else:
        candidate_data = {
            period: subset.copy()
            for period, subset in period_data.items()
        }

    requested_per_period = max(
        1,
        INTERACTION_SAMPLE_SIZE // len(candidate_data),
    )
    available_per_period = min(len(subset) for subset in candidate_data.values())
    sample_per_period = min(requested_per_period, available_per_period)

    if sample_per_period < 2:
        raise ValueError("Too few reference samples for interaction SHAP")

    sampled_parts = []
    for period_index, (period, subset) in enumerate(candidate_data.items()):
        sampled = subset.sample(
            n=sample_per_period,
            replace=False,
            random_state=RANDOM_STATE + period_index,
        ).copy()
        sampled["reference_source_period"] = period
        sampled_parts.append(sampled)

    reference = pd.concat(sampled_parts, ignore_index=True)
    reference = reference.sample(
        frac=1.0,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print(f"Shared explanation rows: {len(reference):,}")
    for period, count in reference["reference_source_period"].value_counts().sort_index().items():
        print(f"  From {period}: {count:,} rows")

    return reference[FEATURE_NAMES].copy()


def normalize_interaction_array(
    values: object,
    sample_count: int,
) -> np.ndarray:
    """Normalize single-output return formats from different SHAP versions."""
    if isinstance(values, list):
        if len(values) != 1:
            raise ValueError("Only single-output regression models are supported")
        values = values[0]

    array = np.asarray(values)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]

    feature_count = len(FEATURE_NAMES)
    expected_shape = (sample_count, feature_count, feature_count)
    if array.shape != expected_shape:
        raise ValueError(
            f"Interaction SHAP array shape is {array.shape}; expected {expected_shape}"
        )

    return array.astype(np.float64, copy=False)


def compute_interaction_values(
    models: dict[str, xgb.XGBRegressor],
    X_reference: pd.DataFrame,
) -> dict[str, np.ndarray]:
    """Calculate interaction SHAP values for both models and check each matrix."""
    interaction_values = {}

    for period, model in models.items():
        print(f"Calculating SHAP interaction values for {period}")
        explainer = shap.TreeExplainer(
            model,
            feature_perturbation="tree_path_dependent",
        )

        raw_values = explainer.shap_interaction_values(X_reference)
        values = normalize_interaction_array(raw_values, len(X_reference))

        predictions = np.asarray(model.predict(X_reference), dtype=np.float64)
        expected_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        symmetry_error = float(
            np.max(np.abs(values - np.swapaxes(values, 1, 2)))
        )
        additivity_error = float(
            np.max(
                np.abs(
                    values.sum(axis=(1, 2))
                    + expected_value
                    - predictions
                )
            )
        )

        print(f"  Maximum symmetry error: {symmetry_error:.3e}")
        print(f"  Maximum additivity error: {additivity_error:.3e}")
        interaction_values[period] = values

    return interaction_values


def validate_selected_pairs(
    selected_pairs: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Validate variable names, pair count, and duplicates."""
    if len(selected_pairs) != 6:
        raise ValueError(
            "SELECTED_INTERACTION_PAIRS must contain six variable pairs"
        )

    normalized_pairs = []
    unordered_pairs = set()

    for pair in selected_pairs:
        if len(pair) != 2:
            raise ValueError(f"Invalid variable-pair format: {pair}")

        feature_x, feature_color = pair
        if feature_x not in FEATURE_NAMES or feature_color not in FEATURE_NAMES:
            raise ValueError(
                f"Pair {pair} contains an invalid variable. Choose from {FEATURE_NAMES}"
            )
        if feature_x == feature_color:
            raise ValueError(f"A variable cannot interact with itself: {pair}")

        unordered_pair = frozenset((feature_x, feature_color))
        if unordered_pair in unordered_pairs:
            raise ValueError(f"Duplicate interaction pair: {pair}")

        unordered_pairs.add(unordered_pair)
        normalized_pairs.append((feature_x, feature_color))

    return normalized_pairs


def binned_mean_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    n_bins: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Represent the response trend with means from quantile bins."""
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]

    if len(x_values) < 2:
        return x_values, y_values

    edges = np.unique(
        np.quantile(
            x_values,
            np.linspace(0.0, 1.0, n_bins + 1),
        )
    )

    if len(edges) < 3:
        order = np.argsort(x_values)
        return x_values[order], y_values[order]

    bin_index = np.digitize(x_values, edges[1:-1], right=True)
    curve_x = []
    curve_y = []

    for current_bin in range(len(edges) - 1):
        selected = bin_index == current_bin
        if np.any(selected):
            curve_x.append(float(np.mean(x_values[selected])))
            curve_y.append(float(np.mean(y_values[selected])))

    return np.asarray(curve_x), np.asarray(curve_y)


def robust_limits(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[float, float]:
    """Set axis and color-scale limits with robust quantiles."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return -1.0, 1.0

    lower, upper = np.quantile(
        finite_values,
        [lower_quantile, upper_quantile],
    )

    if lower == upper:
        padding = max(abs(float(lower)) * 0.05, 1e-9)
    else:
        padding = 0.06 * float(upper - lower)

    return float(lower - padding), float(upper + padding)


def nice_five_ticks(
    values: np.ndarray,
    lower_quantile: float,
    upper_quantile: float,
    include_zero: bool,
    max_clip_fraction: float = 0.0,
) -> np.ndarray:
    """Generate five rounded tick values without scientific notation."""
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.linspace(-1.0, 1.0, 5)

    lower, upper = np.quantile(
        finite_values,
        [lower_quantile, upper_quantile],
    )
    lower = float(lower)
    upper = float(upper)

    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)

    if lower == upper:
        scale = max(abs(lower), 1.0)
        lower -= scale * 0.05
        upper += scale * 0.05

    raw_step = (upper - lower) / 4.0
    exponent = int(np.floor(np.log10(raw_step)))
    multipliers = [1.0, 1.25, 2.0, 2.5, 5.0, 10.0]
    minimum_step = raw_step * (1.0 - max_clip_fraction)
    candidates = sorted(
        {
            multiplier * 10.0 ** current_exponent
            for current_exponent in range(exponent - 1, exponent + 3)
            for multiplier in multipliers
            if multiplier * 10.0 ** current_exponent >= minimum_step * (1.0 - 1e-12)
        }
    )

    allowed_clip = (upper - lower) * max_clip_fraction
    for step in candidates:
        first_index = int(np.floor(lower / step)) - 4
        last_index = int(np.ceil(upper / step))
        possible_ticks = []

        for start_index in range(first_index, last_index + 1):
            ticks = (start_index + np.arange(5)) * step

            if lower >= 0 and ticks[0] < 0:
                continue
            if include_zero and not (ticks[0] <= 0 <= ticks[-1]):
                continue

            clipped_range = (
                max(0.0, ticks[0] - lower)
                + max(0.0, upper - ticks[-1])
            )
            if clipped_range > allowed_clip + step * 1e-10:
                continue

            unused_range = (
                max(0.0, lower - ticks[0])
                + max(0.0, ticks[-1] - upper)
            )
            center_difference = abs(
                (ticks[0] + ticks[-1]) / 2.0
                - (lower + upper) / 2.0
            )
            possible_ticks.append(
                (
                    clipped_range,
                    unused_range,
                    center_difference,
                    ticks,
                )
            )

        if possible_ticks:
            possible_ticks.sort(key=lambda item: item[:3])
            ticks = possible_ticks[0][3]
            ticks[np.isclose(ticks, 0.0, atol=step * 1e-10)] = 0.0
            return ticks

    return np.linspace(lower, upper, 5)


def plain_tick_formatter(ticks: np.ndarray) -> FuncFormatter:
    """Format ticks as decimals based on step size and remove trailing zeros."""
    step = abs(float(ticks[1] - ticks[0]))
    decimal_places = 8

    for digits in range(9):
        scaled_step = step * 10.0 ** digits
        if abs(scaled_step - round(scaled_step)) < 1e-8:
            decimal_places = digits
            break

    def format_value(value: float, position: int) -> str:
        del position
        text = f"{value:.{decimal_places}f}"
        if "." in text:
            text = text.rstrip("0").rstrip(".")
        return "0" if text == "-0" else text

    return FuncFormatter(format_value)


def plot_selected_interactions(
    interaction_values: dict[str, np.ndarray],
    X_reference: pd.DataFrame,
    selected_pairs: list[tuple[str, str]],
    output_path: Path,
) -> None:
    """Plot selected interaction responses in a two-row, six-column layout."""
    period_names = list(PERIODS)
    column_count = len(selected_pairs)

    fig = plt.figure(figsize=(3.1 * column_count, 6.4))
    grid = fig.add_gridspec(
        3,
        column_count,
        height_ratios=[1.0, 1.0, 0.05],
        wspace=0.32,
        hspace=0.38,
    )

    axes = np.empty((2, column_count), dtype=object)
    for period_index in range(2):
        for column_index in range(column_count):
            axes[period_index, column_index] = fig.add_subplot(
                grid[period_index, column_index]
            )

    for column_index, (feature_x, feature_color) in enumerate(selected_pairs):
        x_index = FEATURE_NAMES.index(feature_x)
        color_index = FEATURE_NAMES.index(feature_color)

        x_values = X_reference[feature_x].to_numpy(dtype=np.float64)
        color_values = X_reference[feature_color].to_numpy(dtype=np.float64)

        x_ticks = nice_five_ticks(
            x_values,
            0.0,
            1.0,
            include_zero=False,
        )

        color_min, color_max = robust_limits(
            color_values,
            0.01,
            0.99,
        )
        color_normalization = Normalize(
            vmin=color_min,
            vmax=color_max,
            clip=True,
        )

        combined_interactions = np.concatenate(
            [
                interaction_values[period][:, x_index, color_index]
                for period in period_names
            ]
        )
        y_ticks = nice_five_ticks(
            combined_interactions,
            Y_AXIS_QUANTILES[0],
            Y_AXIS_QUANTILES[1],
            include_zero=True,
            max_clip_fraction=Y_TICK_MAX_CLIP_FRACTION,
        )

        for period_index, period in enumerate(period_names):
            ax = axes[period_index, column_index]
            y_values = interaction_values[period][:, x_index, color_index]

            ax.scatter(
                x_values,
                y_values,
                c=color_values,
                cmap="coolwarm",
                norm=color_normalization,
                alpha=0.16,
                s=7,
                linewidths=0,
                rasterized=True,
            )

            curve_x, curve_y = binned_mean_curve(
                x_values,
                y_values,
                N_CURVE_BINS,
            )
            line_color = (
                EARLY_LINE_COLOR
                if period_index == 0
                else LATE_LINE_COLOR
            )
            ax.plot(
                curve_x,
                curve_y,
                color=line_color,
                linewidth=1.8,
            )
            ax.axhline(
                0,
                color="black",
                linewidth=0.6,
                alpha=0.45,
            )
            ax.set_xlim(x_ticks[0], x_ticks[-1])
            ax.set_ylim(y_ticks[0], y_ticks[-1])
            ax.set_xticks(x_ticks)
            ax.set_yticks(y_ticks)
            ax.xaxis.set_major_formatter(plain_tick_formatter(x_ticks))
            ax.yaxis.set_major_formatter(plain_tick_formatter(y_ticks))
            ax.grid(alpha=0.13)
            ax.tick_params(labelsize=10)

            if period_index == 0:
                ax.set_title(
                    f"{FEATURE_LABELS[feature_x]} × {FEATURE_LABELS[feature_color]}",
                    fontsize=11.5,
                    pad=6,
                )

            if period_index == 1:
                ax.set_xlabel(
                    FEATURE_LABELS[feature_x],
                    fontsize=11,
                    labelpad=3,
                )

            if column_index == 0:
                ax.set_ylabel(
                    f"{period}\nSHAP interaction value (CSIF)",
                    fontsize=11,
                )

        colorbar_axis = fig.add_subplot(grid[2, column_index])
        colorbar = fig.colorbar(
            ScalarMappable(
                norm=color_normalization,
                cmap="coolwarm",
            ),
            cax=colorbar_axis,
            orientation="horizontal",
        )
        colorbar.set_ticks(
            [
                color_min,
                (color_min + color_max) / 2.0,
                color_max,
            ]
        )
        colorbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        colorbar.set_label(
            FEATURE_LABELS[feature_color],
            fontsize=10,
            labelpad=2,
        )
        colorbar.ax.tick_params(labelsize=9, pad=2)

    fig.subplots_adjust(
        left=0.065,
        right=0.992,
        bottom=0.09,
        top=0.96,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_path,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    print(f"Figure saved to: {output_path}")

    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.linewidth": 0.8,
            "savefig.facecolor": "white",
        }
    )

    selected_pairs = validate_selected_pairs(SELECTED_INTERACTION_PAIRS)
    data_path = resolve_data_file(DATA_FILE)
    output_path = data_path.parent / OUTPUT_FIGURE_NAME

    df = prepare_data(data_path)
    period_data = split_period_data(df)
    models = train_models(period_data)
    X_reference = build_reference_sample(period_data)
    interaction_values = compute_interaction_values(models, X_reference)

    plot_selected_interactions(
        interaction_values,
        X_reference,
        selected_pairs,
        output_path,
    )


if __name__ == "__main__":
    main()
