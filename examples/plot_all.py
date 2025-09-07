#!/usr/bin/env python3
"""
Plot all spectral parameters from the h5 dataset.

This script reads spectral parameters from an h5 file and creates comprehensive visualizations:

BOX PLOTS (Distribution Analysis):
- One box plot per spectral parameter showing the distribution across all segments and subjects
- Includes mean, standard deviation, and sample count statistics
- Shows outliers, quartiles, and overall spread of each parameter

1D SCATTER PLOTS (Subject-wise Analysis):
- One scatter plot per spectral parameter with points color-coded by subject category
- Each point represents one segment from one subject
- Y-axis shows different subjects (jittered for visibility)
- Color coding helps identify patterns within/between subject categories

2D SCATTER PLOTS (Relationship Analysis):
- Band power relationships: Delta vs Alpha, Theta vs Alpha, Beta vs Alpha
- Frequency domain relationships: Median frequency vs Spectral centroid, etc.
- Entropy relationships: Shannon vs Renyi, Shannon vs Tsallis
- Alpha-related relationships: IAF vs Alpha power, IAF vs Median frequency
- Spectral shape relationships: Crest factor vs Centroid, etc.
- Each plot includes correlation coefficient and color-coded subject categories

CORRELATION MATRIX:
- Heatmap showing correlations between all spectral parameters
- Helps identify strongly related parameters and potential redundancies

RELATIVE POWERS (Special Handling):
- Box plot showing all 6 frequency bands side-by-side
- Individual scatter plots for each frequency band (Delta, Theta, Alpha, Beta1, Beta2, Gamma)

USAGE:
    python plot_all.py --h5_file ../h5test.h5 --output_dir ../plots

The script automatically:
- Extracts subject categories from filenames (e.g., 'ADMIL_092.mat' -> 'ADMIL')
- Handles missing data and NaN values gracefully
- Uses proper frequency band names from the h5 file attributes
- Saves all plots as high-resolution PNG files
- Creates ~40+ different visualizations for comprehensive analysis
"""

import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
from pathlib import Path
import argparse

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def extract_subject_category(subject_name: str) -> str:
    """Extract category from subject filename (e.g., 'ADMIL_092.mat' -> 'ADMIL')"""
    return subject_name.split('_')[0]

def load_spectral_data(h5_path: str) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load spectral parameters from h5 file.

    Returns:
        Dict[subject_name, Dict[param_name, array]]
    """
    spectral_data = {}

    with h5py.File(h5_path, 'r') as f:
        if "subjects" not in f:
            raise ValueError("No 'subjects' group found in h5 file")

        subjects_group = f["subjects"]

        for subject_key in subjects_group.keys():
            subject = subjects_group[subject_key]

            if "spectral" in subject:
                spectral_group = subject["spectral"]
                if "spectral_parameters" in spectral_group:
                    sp_group = spectral_group["spectral_parameters"]

                    # Extract all spectral parameters
                    subject_params = {}
                    for param_key in sp_group.keys():
                        data = sp_group[param_key][()]
                        if isinstance(data, np.ndarray) and data.size > 0:
                            subject_params[param_key] = data

                    if subject_params:  # Only add if we have parameters
                        spectral_data[subject_key] = subject_params

    return spectral_data

def create_box_plots(data: Dict[str, Dict[str, np.ndarray]], output_dir: str = "plots"):
    """
    Create box plots for each spectral parameter.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Get all parameter names (use first subject as reference)
    if not data:
        print("No data to plot")
        return

    first_subject = next(iter(data.values()))
    param_names = list(first_subject.keys())

    # Remove relative_powers from individual plots (will handle separately)
    if "relative_powers" in param_names:
        param_names.remove("relative_powers")

    # Create box plots for each parameter
    for param_name in param_names:
        plt.figure(figsize=(12, 8))

        # Collect data for all subjects
        all_values = []
        subject_labels = []

        for subject_name, params in data.items():
            if param_name in params:
                values = params[param_name]
                if values.size > 0 and np.issubdtype(values.dtype, np.number):
                    # Remove NaN values for cleaner plots
                    clean_values = values[~np.isnan(values)]
                    if len(clean_values) > 0:
                        all_values.extend(clean_values)
                        subject_labels.extend([subject_name] * len(clean_values))

        if all_values:
            # Create box plot
            plt.boxplot([all_values], tick_labels=[param_name], patch_artist=True,
                       boxprops=dict(facecolor='lightblue', color='blue'),
                       medianprops=dict(color='red'),
                       whiskerprops=dict(color='blue'),
                       capprops=dict(color='blue'))

            plt.title(f'Distribution of {param_name.replace("_", " ").title()}')
            plt.ylabel(param_name.replace("_", " ").title())
            plt.grid(True, alpha=0.3)

            # Add statistics as text
            if all_values:
                mean_val = np.mean(all_values)
                std_val = np.std(all_values)
                plt.figtext(0.02, 0.02, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}\nN: {len(all_values)}',
                           fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

            plt.tight_layout()
            plt.savefig(f"{output_dir}/{param_name}_boxplot.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved box plot for {param_name}")
        else:
            print(f"No valid data for {param_name}")
            plt.close()

    # Handle relative_powers separately (6 bands)
    if any("relative_powers" in params for params in data.values()):
        plt.figure(figsize=(15, 10))

        # Get classical bands from file attributes (if available)
        band_names = [f"Band {i+1}" for i in range(6)]  # Default names

        try:
            with h5py.File("../h5test.h5", 'r') as f:
                if "classical_bands" in f.attrs:
                    import json
                    bands_dict = json.loads(f.attrs["classical_bands"])
                    band_names = list(bands_dict.keys())
        except:
            pass  # Use default names

        # Plot each band as a separate box
        band_data = [[] for _ in range(6)]

        for subject_name, params in data.items():
            if "relative_powers" in params:
                rp_data = params["relative_powers"]  # Shape: (n_segments, 6)
                if rp_data.shape[1] >= 6:
                    for band_idx in range(6):
                        clean_values = rp_data[:, band_idx][~np.isnan(rp_data[:, band_idx])]
                        band_data[band_idx].extend(clean_values)

        # Create box plot for all bands
        plt.boxplot(band_data, tick_labels=band_names, patch_artist=True)

        plt.title('Distribution of Relative Powers by Frequency Band')
        plt.ylabel('Relative Power')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{output_dir}/relative_powers_boxplot.png", dpi=300, bbox_inches='tight')
        plt.close()

        print("Saved box plot for relative powers")

def create_scatter_plots(data: Dict[str, Dict[str, np.ndarray]], output_dir: str = "plots"):
    """
    Create scatter plots for each spectral parameter, color-coded by subject category.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Get unique categories and assign colors
    categories = list(set(extract_subject_category(subj) for subj in data.keys()))
    color_map = plt.colormaps['tab10'](np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, color_map))

    # Get all parameter names
    if not data:
        print("No data to plot")
        return

    first_subject = next(iter(data.values()))
    param_names = list(first_subject.keys())

    # Remove relative_powers from individual plots
    if "relative_powers" in param_names:
        param_names.remove("relative_powers")

    # Create scatter plots for each parameter
    for param_name in param_names:
        plt.figure(figsize=(14, 10))

        # Track data for legend
        legend_elements = []

        # Plot data for each subject
        y_positions = []  # For vertical spacing of subjects
        subject_names = []

        for i, (subject_name, params) in enumerate(data.items()):
            if param_name in params:
                values = params[param_name]
                if values.size > 0 and np.issubdtype(values.dtype, np.number):
                    # Remove NaN values
                    clean_values = values[~np.isnan(values)]
                    if len(clean_values) > 0:
                        category = extract_subject_category(subject_name)
                        color = category_colors[category]

                        # Create y positions for this subject (jittered)
                        y_pos = np.random.normal(i, 0.1, len(clean_values))

                        plt.scatter(clean_values, y_pos, alpha=0.6, color=color,
                                  label=category if category not in [e.get_label() for e in legend_elements] else "",
                                  s=30, edgecolors='black', linewidth=0.5)

                        y_positions.append(i)
                        subject_names.append(subject_name)

        if y_positions:
            plt.title(f'{param_name.replace("_", " ").title()} by Subject Category')
            plt.xlabel(param_name.replace("_", " ").title())
            plt.ylabel('Subject')

            # Set y-axis ticks and labels
            plt.yticks(range(len(subject_names)), subject_names, fontsize=8)
            plt.ylim(-0.5, len(subject_names) - 0.5)

            # Add legend
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{param_name}_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved scatter plot for {param_name}")
        else:
            print(f"No valid data for {param_name}")
            plt.close()

    # Handle relative_powers scatter plots (one plot per band)
    if any("relative_powers" in params for params in data.values()):
        # Get band names
        band_names = [f"Band {i+1}" for i in range(6)]
        try:
            with h5py.File("../h5test.h5", 'r') as f:
                if "classical_bands" in f.attrs:
                    import json
                    bands_dict = json.loads(f.attrs["classical_bands"])
                    band_names = list(bands_dict.keys())
        except:
            pass

        for band_idx in range(6):
            plt.figure(figsize=(14, 10))

            for i, (subject_name, params) in enumerate(data.items()):
                if "relative_powers" in params:
                    rp_data = params["relative_powers"]
                    if rp_data.shape[1] > band_idx:
                        values = rp_data[:, band_idx]
                        clean_values = values[~np.isnan(values)]
                        if len(clean_values) > 0:
                            category = extract_subject_category(subject_name)
                            color = category_colors[category]

                            y_pos = np.random.normal(i, 0.1, len(clean_values))

                            plt.scatter(clean_values, y_pos, alpha=0.6, color=color,
                                      label=category if category not in plt.gca().get_legend_handles_labels()[1] else "",
                                      s=30, edgecolors='black', linewidth=0.5)

            plt.title(f'Relative Power - {band_names[band_idx]} by Subject Category')
            plt.xlabel('Relative Power')
            plt.ylabel('Subject')

            # Set y-axis ticks and labels
            subject_names = list(data.keys())
            plt.yticks(range(len(subject_names)), subject_names, fontsize=8)
            plt.ylim(-0.5, len(subject_names) - 0.5)

            # Add legend
            handles, labels = plt.gca().get_legend_handles_labels()
            if handles:
                plt.legend(title='Category', bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/relative_power_{band_names[band_idx].replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_').lower()}_scatter.png",
                       dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved scatter plot for relative power {band_names[band_idx]}")

def create_2d_scatter_plots(data: Dict[str, Dict[str, np.ndarray]], output_dir: str = "plots"):
    """
    Create 2D scatter plots to reveal relationships between spectral parameters.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Get unique categories and assign colors
    categories = list(set(extract_subject_category(subj) for subj in data.keys()))
    color_map = plt.colormaps['tab10'](np.linspace(0, 1, len(categories)))
    category_colors = dict(zip(categories, color_map))

    # Get band names for relative powers
    band_names = [f"Band {i+1}" for i in range(6)]
    band_indices = {}
    try:
        with h5py.File("../h5test.h5", 'r') as f:
            if "classical_bands" in f.attrs:
                import json
                bands_dict = json.loads(f.attrs["classical_bands"])
                band_names = list(bands_dict.keys())
                band_indices = {name: idx for idx, name in enumerate(band_names)}
    except:
        # Create default band indices
        band_indices = {f"Band {i+1}": i for i in range(6)}

    # Define interesting 2D relationships to plot
    plot_pairs = [
        # Band power relationships (most common in neuroscience)
        ("delta_vs_alpha", "Delta relative power", "Alpha relative power", "relative_powers", "relative_powers", band_indices.get("Delta (0.5-4 Hz)", 0), band_indices.get("Alpha (8-13 Hz)", 2)),
        ("theta_vs_alpha", "Theta relative power", "Alpha relative power", "relative_powers", "relative_powers", band_indices.get("Theta (4-8 Hz)", 1), band_indices.get("Alpha (8-13 Hz)", 2)),
        ("beta1_vs_alpha", "Beta1 relative power", "Alpha relative power", "relative_powers", "relative_powers", band_indices.get("Beta1 (13-19 Hz)", 3), band_indices.get("Alpha (8-13 Hz)", 2)),

        # Frequency domain relationships
        ("median_vs_centroid", "Median frequency", "Spectral centroid", "median_frequency", "spectral_centroid", None, None),
        ("median_vs_edge", "Median frequency", "Spectral edge 95%", "median_frequency", "spectral_edge_frequency_95", None, None),
        ("centroid_vs_bandwidth", "Spectral centroid", "Spectral bandwidth", "spectral_centroid", "spectral_bandwidth", None, None),

        # Entropy relationships
        ("shannon_vs_renyi", "Shannon entropy", "Renyi entropy", "shannon_entropy", "renyi_entropy", None, None),
        ("shannon_vs_tsallis", "Shannon entropy", "Tsallis entropy", "shannon_entropy", "tsallis_entropy", None, None),
        ("renyi_vs_tsallis", "Renyi entropy", "Tsallis entropy", "renyi_entropy", "tsallis_entropy", None, None),

        # Alpha-related relationships
        ("iaf_vs_alpha_power", "Individual alpha frequency", "Alpha relative power", "individual_alpha_frequency", "relative_powers", None, band_indices.get("Alpha (8-13 Hz)", 2)),
        ("iaf_vs_median", "Individual alpha frequency", "Median frequency", "individual_alpha_frequency", "median_frequency", None, None),

        # Spectral shape relationships
        ("crest_vs_centroid", "Spectral crest factor", "Spectral centroid", "spectral_crest_factor", "spectral_centroid", None, None),
        ("transition_vs_iaf", "Transition frequency", "Individual alpha frequency", "transition_frequency", "individual_alpha_frequency", None, None),
    ]

    for plot_name, x_label, y_label, x_param, y_param, x_band_idx, y_band_idx in plot_pairs:
        plt.figure(figsize=(12, 10))

        # Collect data for this plot
        x_data = []
        y_data = []
        colors = []
        subject_labels = []

        for subject_name, params in data.items():
            if x_param in params and y_param in params:
                # Handle relative powers specially
                if x_param == "relative_powers":
                    if x_band_idx is not None and params[x_param].shape[1] > x_band_idx:
                        x_values = params[x_param][:, x_band_idx]
                    else:
                        continue
                else:
                    x_values = params[x_param]

                if y_param == "relative_powers":
                    if y_band_idx is not None and params[y_param].shape[1] > y_band_idx:
                        y_values = params[y_param][:, y_band_idx]
                    else:
                        continue
                else:
                    y_values = params[y_param]

                # Ensure same length
                min_len = min(len(x_values), len(y_values))
                x_values = x_values[:min_len]
                y_values = y_values[:min_len]

                # Remove NaN values
                valid_mask = ~(np.isnan(x_values) | np.isnan(y_values))
                x_values = x_values[valid_mask]
                y_values = y_values[valid_mask]

                if len(x_values) > 0:
                    x_data.extend(x_values)
                    y_data.extend(y_values)

                    category = extract_subject_category(subject_name)
                    colors.extend([category_colors[category]] * len(x_values))
                    subject_labels.extend([subject_name] * len(x_values))

        if x_data and y_data:
            # Convert to numpy arrays for easier plotting
            x_data = np.array(x_data)
            y_data = np.array(y_data)
            colors = np.array(colors)

            # Create scatter plot
            scatter = plt.scatter(x_data, y_data, c=colors, alpha=0.7, s=50,
                                edgecolors='black', linewidth=0.5)

            plt.xlabel(x_label)
            plt.ylabel(y_label)
            plt.title(f'{x_label} vs {y_label}')

            # Add colorbar legend
            legend_elements = []
            for category in categories:
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                markerfacecolor=category_colors[category],
                                                markersize=10, label=category))
            plt.legend(handles=legend_elements, title='Subject Category', bbox_to_anchor=(1.05, 1), loc='upper left')

            # Add correlation coefficient
            if len(x_data) > 1:
                corr = np.corrcoef(x_data, y_data)[0, 1]
                plt.figtext(0.02, 0.02, f'Correlation: {corr:.3f}', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{plot_name}_2d_scatter.png", dpi=300, bbox_inches='tight')
            plt.close()

            print(f"Saved 2D scatter plot for {plot_name}")
        else:
            print(f"No valid data for {plot_name}")
            plt.close()

def create_correlation_matrix(data: Dict[str, Dict[str, np.ndarray]], output_dir: str = "plots"):
    """
    Create a correlation matrix heatmap for all spectral parameters.
    """
    Path(output_dir).mkdir(exist_ok=True)

    # Get all parameter names (excluding relative_powers for simplicity)
    if not data:
        return

    first_subject = next(iter(data.values()))
    param_names = [name for name in first_subject.keys() if name != "relative_powers"]

    # Collect all data across subjects
    all_data = {}
    for param in param_names:
        all_values = []
        for subject_name, params in data.items():
            if param in params:
                values = params[param]
                # Remove NaN values
                clean_values = values[~np.isnan(values)]
                all_values.extend(clean_values)
        if all_values:
            all_data[param] = all_values

    if len(all_data) < 2:
        print("Not enough parameters for correlation matrix")
        return

    # Create correlation matrix
    param_list = list(all_data.keys())
    n_params = len(param_list)
    corr_matrix = np.zeros((n_params, n_params))

    for i, param1 in enumerate(param_list):
        for j, param2 in enumerate(param_list):
            if i <= j:  # Only compute upper triangle (symmetric matrix)
                values1 = all_data[param1]
                values2 = all_data[param2]
                # Ensure same length
                min_len = min(len(values1), len(values2))
                values1 = values1[:min_len]
                values2 = values2[:min_len]

                if len(values1) > 1:
                    corr = np.corrcoef(values1, values2)[0, 1]
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr

    # Create heatmap
    plt.figure(figsize=(14, 12))

    # Create mask for upper triangle
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

    # Create heatmap
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                xticklabels=[p.replace('_', ' ').title() for p in param_list],
                yticklabels=[p.replace('_', ' ').title() for p in param_list],
                square=True, cbar_kws={'shrink': 0.8})

    plt.title('Spectral Parameters Correlation Matrix')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()

    print("Saved correlation matrix heatmap")

def main():
    parser = argparse.ArgumentParser(description='Plot spectral parameters from h5 dataset')
    parser.add_argument('--h5_file', default='../h5test.h5', help='Path to h5 file (default: ../h5test.h5)')
    parser.add_argument('--output_dir', default='../plots', help='Output directory for plots (default: ../plots)')

    args = parser.parse_args()

    print(f"Loading data from {args.h5_file}...")

    try:
        # Load spectral data
        spectral_data = load_spectral_data(args.h5_file)

        if not spectral_data:
            print("No spectral data found in the h5 file")
            return

        print(f"Found {len(spectral_data)} subjects with spectral parameters")

        # Create plots
        print("\nCreating box plots...")
        create_box_plots(spectral_data, args.output_dir)

        print("\nCreating 1D scatter plots...")
        create_scatter_plots(spectral_data, args.output_dir)

        print("\nCreating 2D scatter plots...")
        create_2d_scatter_plots(spectral_data, args.output_dir)

        print("\nCreating correlation matrix...")
        create_correlation_matrix(spectral_data, args.output_dir)

        print(f"\nAll plots saved to '{args.output_dir}' directory")

    except Exception as e:
        print(f"Error: {e}")
        return

if __name__ == "__main__":
    main()
