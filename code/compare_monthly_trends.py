import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

SatelliteGPP = ['MCI_GPP', 'ST_CFE-ML_NT', 'ST_CFE-Hybrid_NT']
SatelliteGPP2 = ['FluxSat', 'FluxSat-V22', 'NIRv-GPP']
MLSIF_linear = ['CSIF-linear', 'GOSIF-linear', 'RTSIF-linear']
MLSIF_CO2_linear = ['CSIF-CO2-linear-month', 'GOSIF-CO2-linear-month', 'RTSIF-CO2-linear-month']

ensembles = {
    'Satellite GPP (CFE)': SatelliteGPP,
    'EEO-inferred GPP': [],
    'Satellite GPP without CFE': SatelliteGPP2,
    'GPP cons. by SIF': MLSIF_linear,
    'GPP cons. by SIF ($\mathrm{CO}_{\mathrm{2}}$ fertilized)': MLSIF_CO2_linear,
}

custom_labels = {
    'MCI_GPP': 'MCI GPP',
    'ST_CFE-ML_NT': 'ML-NT GPP',
    'ST_CFE-Hybrid_NT': 'Hybrid-NT GPP',

    'FluxSat': 'FluxSat GPP',
    'FluxSat-V22': 'FluxSat GPP V2.2',
    'Sunshade': 'Revised TL-LUE GPP',
    'Fluxcom-X': 'Fluxcom-X GPP',

    'CSIF-linear': 'CSIF-cons.',
    'GOSIF-linear': 'GOSIF-cons.',
    'RTSIF-linear': 'RTSIF-cons.',
    'LHSIF-linear': 'LHSIF-cons.',

    'CSIF-CO2-linear-month': 'CSIF-cons. (CFE)',
    'GOSIF-CO2-linear-month': 'GOSIF-cons. (CFE)',
    'RTSIF-CO2-linear-month': 'RTSIF-cons. (CFE)',
    'LHSIF-CO2-linear-month': 'LHSIF-cons. (CFE)',

    # Add the EEO-inferred GPP label
    'EEO-inferred_GPP_A': 'EEO-inferred GPP (direct)',
    'EEO-inferred_GPP_B': 'EEO-inferred GPP (diag.)',
}

folder = r'other'

slopelist = []
component_info = []

markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>', 'd']
component_colors = plt.cm.tab10(np.linspace(0, 1, 9))

for ensemble_idx, (ensemble_name, model_list) in enumerate(ensembles.items()):
    ensemble_component_info = []

    if ensemble_name == 'EEO-inferred GPP':
        # EEO-inferred GPP has two values
        eeo_slopes = [0.45]
        eeo_labels = ['EEO-inferred GPP']

        # Use the mean as the bar height
        avg_slope = np.mean(eeo_slopes)
        slopelist.append(avg_slope)

        # Add information for both components
        for label, slope in zip(eeo_labels, eeo_slopes):
            custom_label = custom_labels.get(label, label)
            ensemble_component_info.append((custom_label, slope))

        component_info.append(ensemble_component_info)
        continue

    component_slopes = []

    for csv_file in os.listdir(folder):
        component_name = os.path.splitext(csv_file)[0]

        if component_name in model_list:
            print(csv_file)
            file_path = os.path.join(folder, csv_file)
            df = pd.read_csv(file_path)
            df_filtered = df[df['Year'] <= 2020]
            slope, _, _, _, _ = stats.linregress(df_filtered['Year'], df_filtered['Sum'])
            component_slopes.append(slope)

            custom_label = custom_labels.get(component_name, component_name)
            ensemble_component_info.append((custom_label, slope))

    if component_slopes:  # Process only when data is available
        # Use the mean slope as the bar height
        avg_slope = np.mean(component_slopes)
        slopelist.append(avg_slope)
        component_info.append(ensemble_component_info)
    else:
        # Add an empty list when data is unavailable
        slopelist.append(0)
        component_info.append([])

fig, ax = plt.subplots(figsize=(10, 4.5))
bar_colors = ['black', 'blue', 'red', 'orange', 'green']
bar_positions = [0, 1, 2, 3, 4]

bar_labels = ['L1: Satellite-derived GPP (CFE)',
              'L2: EEO-inferred GPP',
              'L3: Satellite Optical Observation-Driven GPP',
              'L4: SIF-cons. GPP',
              'L5: SIF-cons. GPP (CFE)']

bars = ax.bar(bar_positions, slopelist, color=bar_colors, alpha=0.7, capsize=10,
              error_kw={'elinewidth': 2, 'capthick': 2})

for i, (pos, comp_info, bar_color) in enumerate(zip(bar_positions, component_info, bar_colors)):
    num_components = len(comp_info)

    if num_components == 0:
        continue

    x_offset = 0.25 if num_components > 1 else 0
    x_positions = np.linspace(pos - x_offset, pos + x_offset, num_components) if num_components > 1 else [pos]

    for j, (x_pos, (custom_label, slope_val)) in enumerate(zip(x_positions, comp_info)):
        marker_idx = (i * 5 + j) % len(markers)
        color_idx = (i * 3 + j) % len(component_colors)

        ax.scatter(x_pos, slope_val, s=200,
                   marker=markers[marker_idx],
                   color=component_colors[color_idx],
                   linewidth=1.5, zorder=5)

ax.set_xticks(bar_positions)
ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'L5'], fontsize=16)
ax.set_ylabel('Trend (% / year)', fontsize=16)
ax.tick_params(axis='y', labelsize=16)

bar_legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=color, alpha=0.7, label=label)
                       for color, label in zip(bar_colors, bar_labels)]

scatter_legend_elements = []
for i, comp_info in enumerate(component_info):
    for j, (custom_label, slope_val) in enumerate(comp_info):
        marker_idx = (i * 5 + j) % len(markers)
        color_idx = (i * 3 + j) % len(component_colors)

        print(i, j, marker_idx, color_idx)

        scatter_legend_elements.append(plt.Line2D([0], [0], marker=markers[marker_idx],
                                                  color=component_colors[color_idx],
                                                  markerfacecolor=component_colors[color_idx],
                                                  markersize=12, label=custom_label,
                                                  linestyle='None'))
    if i == 2:
        scatter_legend_elements.append(plt.Line2D([0], [0], markersize=0, linestyle='None'))

ax.legend(handles=bar_legend_elements,
          bbox_to_anchor=(1, 1.05), fontsize=16, frameon=False, ncol=1)

ax2 = fig.add_axes([0.53, 0.1, 0.2, 0.6])
ax2.axis('off')
ax2.legend(handles=scatter_legend_elements, loc='center left', fontsize=16, frameon=False, ncol=2, columnspacing=0.2)
plt.tight_layout(rect=[0, 0, 1.14, 1.15])

plt.savefig(r'other\1.svg', bbox_inches='tight')
plt.show()
