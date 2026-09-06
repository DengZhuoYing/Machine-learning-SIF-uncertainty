import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

lclist = ['EBF', 'OF', 'SW', 'GRA', 'CRO', 'C4']

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.3, right=0.75)  # Reserve space on the right for the legend

axes_flat = axes.flatten()

bar_colors = ['black', 'orange', 'green']
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', 'X', 'P', '<', '>', 'd']
component_colors = plt.cm.tab10(np.linspace(0, 1, 9))

for idx, (lc, ax) in enumerate(zip(lclist, axes_flat)):

    SatelliteGPP = [f'MCI_GPP-{lc}', f'ST_CFE-ML_NT-{lc}', f'ST_CFE-Hybrid_NT-{lc}']
    MLSIF_linear = [f'CSIF-linear-{lc}', f'GOSIF-linear-{lc}', f'RTSIF-linear-{lc}']
    MLSIF_CO2_linear = [f'CSIF-CO2-linear-{lc}', f'GOSIF-CO2-linear-{lc}', f'RTSIF-CO2-linear-{lc}']

    ensembles = {
        'Satellite GPP (CFE)': SatelliteGPP,
        'GPP cons. by SIF': MLSIF_linear,
        'GPP cons. by SIF ($\mathrm{CO}_{\mathrm{2}}$ fertilized)': MLSIF_CO2_linear,
    }

    folder = r'other'

    slopelist = []
    component_info = []

    for ensemble_idx, (ensemble_name, model_list) in enumerate(ensembles.items()):
        component_slopes = []
        ensemble_component_info = []

        for csv_file in os.listdir(folder):
            component_name = os.path.splitext(csv_file)[0]

            if component_name in model_list:
                print(csv_file)
                file_path = os.path.join(folder, csv_file)
                df = pd.read_csv(file_path)
                df_filtered = df[df['Year'] <= 2020]
                slope, _, _, _, _ = stats.linregress(df_filtered['Year'], df_filtered['Sum'])

                component_slopes.append(slope)
                ensemble_component_info.append(('', slope))

        if component_slopes:
            avg_slope = np.mean(component_slopes)
            slopelist.append(avg_slope)
            component_info.append(ensemble_component_info)
        else:
            slopelist.append(0)
            component_info.append([])

    bar_positions = [0, 1, 2]
    bars = ax.bar(bar_positions, slopelist, color=bar_colors, alpha=0.7, capsize=10,
                  error_kw={'elinewidth': 2, 'capthick': 2})

    for i, (pos, comp_info, bar_color) in enumerate(zip(bar_positions, component_info, bar_colors)):
        num_components = len(comp_info)

        x_offset = 0.25 if num_components > 1 else 0
        x_positions = np.linspace(pos - x_offset, pos + x_offset, num_components) if num_components > 1 else [pos]

        for j, (x_pos, (custom_label, slope_val)) in enumerate(zip(x_positions, comp_info)):
            if i == 0:
                marker_idx = (i * 5 + j) % len(markers)
                color_idx = (i * 3 + j) % len(component_colors)
            elif i >= 1:
                marker_idx = ((i + 2) * 5 + j) % len(markers)
                color_idx = ((i + 2) * 3 + j) % len(component_colors)

            ax.scatter(x_pos, slope_val, s=200, marker=markers[marker_idx],
                       color=component_colors[color_idx], linewidth=1.5, zorder=5)

    ax.set_xticks(bar_positions)
    ax.set_xticklabels(['L1', 'L4', 'L5'], fontsize=14)

    if idx % 3 == 0:
        ax.set_ylabel('Trend (% / year)', fontsize=14)

    ax.tick_params(axis='y', labelsize=14)

    if lc != 'C4':
        ax.text(0.5, 1, f'{lc}', transform=ax.transAxes, fontsize=16, ha='center', va='bottom')
    else:
        ax.text(0.5, 1, f'C$_4$', transform=ax.transAxes, fontsize=16, ha='center', va='bottom')


bar_labels = ['L1: Satellite-derived GPP (CFE)',
              'L4: SIF-cons. GPP',
              'L5: SIF-cons. GPP (CFE)']
bar_legend_elements = [plt.Rectangle((0, 0), 1, 1, facecolor=bar_colors[i],
                                     alpha=0.7, label=label) for i, label in enumerate(bar_labels)]

# Scatter plot legend for each component
# Define base names and display labels for each group
group1 = [('MCI_GPP', 'MCI GPP'),
          ('ST_CFE-ML_NT', 'ML-NT GPP'),
          ('ST_CFE-Hybrid_NT', 'Hybrid-NT GPP')]
group2 = [('CSIF-linear', 'CSIF-cons.'),
          ('GOSIF-linear', 'GOSIF-cons.'),
          ('RTSIF-linear', 'RTSIF-cons.')]
group3 = [('CSIF-CO2-linear', 'CSIF-cons. (CFE)'),
          ('GOSIF-CO2-linear', 'GOSIF-cons. (CFE)'),
          ('RTSIF-CO2-linear', 'RTSIF-cons. (CFE)')]

scatter_legend_elements = []

# L1 components (i=0)
for j, (_, label) in enumerate(group1):
    marker_idx = (0 * 5 + j) % len(markers)
    color_idx = (0 * 3 + j) % len(component_colors)
    scatter_legend_elements.append(plt.Line2D([0], [0], marker=markers[marker_idx],
                                              color=component_colors[color_idx],
                                              markerfacecolor=component_colors[color_idx],
                                              markersize=12, label=label, linestyle='None'))
# L4 components (use i+2=3 when i=1 in the subplot)
for j, (_, label) in enumerate(group2):
    marker_idx = ((1 + 2) * 5 + j) % len(markers)
    color_idx = ((1 + 2) * 3 + j) % len(component_colors)
    scatter_legend_elements.append(plt.Line2D([0], [0], marker=markers[marker_idx],
                                              color=component_colors[color_idx],
                                              markerfacecolor=component_colors[color_idx],
                                              markersize=12, label=label, linestyle='None'))
# L5 components (use i+2=4 when i=2 in the subplot)
for j, (_, label) in enumerate(group3):
    marker_idx = ((2 + 2) * 5 + j) % len(markers)
    color_idx = ((2 + 2) * 3 + j) % len(component_colors)
    scatter_legend_elements.append(plt.Line2D([0], [0], marker=markers[marker_idx],
                                              color=component_colors[color_idx],
                                              markerfacecolor=component_colors[color_idx],
                                              markersize=12, label=label, linestyle='None'))

# Place the legend outside on the right
fig.legend(handles=bar_legend_elements, bbox_to_anchor=(0.4, 0.08), fontsize=14, frameon=False)
fig.legend(handles=scatter_legend_elements, bbox_to_anchor=(0.75, 0.08), fontsize=14,
           frameon=False, ncol=2, columnspacing=0.5)

plt.savefig(r'other\lc.svg', bbox_inches='tight', dpi=300)
plt.show()