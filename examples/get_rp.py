import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import List, Dict

import core.eeg_utils as eeg
from spectral.relative_powers import calcular_rp # Import the Relative Power calculation function

# --- Configuration ---
DATA_FOLDER_PATH = '/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH'
CLASSICAL_BANDS: Dict[str, List[float]] = {
    "Delta (0.5-4 Hz)": [0.5, 4.0],
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta1 (13-19 Hz)": [13.0, 19.0],
    "Beta2 (19-30 Hz)": [19.0, 30.0],
    "Gamma (30-70 Hz)": [30.0, 70.0]
}
# Convert dict to list of lists for calcular_rp and keep names for plotting
SUB_BANDAS_LIST: List[List[float]] = list(CLASSICAL_BANDS.values())
SUB_BANDAS_NAMES: List[str] = list(CLASSICAL_BANDS.keys())

# --- Load Data ---
all_files, names = eeg.load_files_from_folder(DATA_FOLDER_PATH)

if not all_files:
    logger.error(f"No files found in {DATA_FOLDER_PATH}. Exiting.")
    exit()

# Process the first file as an example
# You can loop through all_files and names to process all files
one_file = all_files[0]
one_name = names[0]

logger.info(f"Processing file: {one_name}")

# --- Preprocessing and PSD Calculation ---
signal, cfg, target = eeg.get_nice_data(raw_data=one_file, name=one_name, comes_from_bbdds=True)
if target is None:
    logger.error(f"File {one_name} has no target. Skipping.")
    exit()
    
n_segments, n_samples, n_channels = signal.shape

logger.info(f"Signal shape: {signal.shape} (segments, samples, channels)")

f, Pxx = eeg.get_spectral_density(signal, cfg)
logger.success(f"Shape of frequency vector f: {f.shape}")
logger.success(f"Shape of PSD matrix Pxx: {Pxx.shape} (segments, frequencies)")

# --- Determine Total Power Band for Relative Power Calculation ---
banda_total_for_rp = None
if 'filtering' in cfg and isinstance(cfg['filtering'], list):
    for filt_item in cfg['filtering']:
        if isinstance(filt_item, dict) and filt_item.get('type') == 'BandPass' and 'band' in filt_item:
            banda_total_for_rp = filt_item['band']
            logger.info(f"Using bandpass filter range from cfg for Pxx total power: {banda_total_for_rp} Hz")
            break

if banda_total_for_rp is None:
    # Default to a band covering all defined sub-bands if not found in cfg
    min_freq_all_sub_bands = min(band[0] for band in SUB_BANDAS_LIST)
    max_freq_all_sub_bands = max(band[1] for band in SUB_BANDAS_LIST)
    banda_total_for_rp = [min_freq_all_sub_bands, max_freq_all_sub_bands]
    logger.warning(f"No BandPass filter range found in cfg. Defaulting Pxx total power band to: {banda_total_for_rp} Hz")

# --- Relative Power Calculation --- 
all_relative_powers = [] # List to store RP arrays for each segment

if Pxx.ndim == 1: # Handle case where Pxx might be 1D (single segment)
    Pxx_proc = Pxx.reshape(1, -1) # Treat as a single segment
    n_plot_segments = 1
else:
    Pxx_proc = Pxx
    n_plot_segments = Pxx_proc.shape[0]

for seg_idx in range(n_plot_segments):
    current_psd_segment = Pxx_proc[seg_idx, :]
    relative_powers_segment = calcular_rp(current_psd_segment, f, banda_total_for_rp, SUB_BANDAS_LIST)
    all_relative_powers.append(relative_powers_segment)

# --- Logging Results ---
logger.info("\n--- Relative Power (RP) per Segment ---")
for seg_idx, rp_values in enumerate(all_relative_powers):
    logger.info(f"Segment {seg_idx + 1}:")
    for band_name, rp_val in zip(SUB_BANDAS_NAMES, rp_values):
        logger.info(f"  {band_name}: {rp_val:.4f}")

# --- Visualization ---
try:
    # Plot 1: PSDs per segment with band indications (optional, can be very verbose)
    # For simplicity, we'll focus on the summary bar chart as it directly shows relative powers.
    # If PSD plots are desired, they can be adapted from get_iaftf.py

    # Plot 2: Bar chart of Relative Powers
    if n_plot_segments > 0 and all_relative_powers:
        fig_bar, ax_bar = plt.subplots(figsize=(max(10, n_plot_segments * 1.5), 6))
        
        n_bands = len(SUB_BANDAS_NAMES)
        segment_indices = np.arange(n_plot_segments)
        bar_width = 0.8 / n_bands # Adjust bar width based on number of bands
        
        colors = plt.cm.get_cmap('viridis', n_bands) # or any other colormap

        for band_idx, band_name in enumerate(SUB_BANDAS_NAMES):
            # Extract RP for this band across all segments
            rp_for_current_band = [rp_seg[band_idx] if rp_seg is not None and len(rp_seg) > band_idx else np.nan 
                                   for rp_seg in all_relative_powers]
            
            offsets = bar_width * band_idx - (bar_width * (n_bands -1) / 2) # Center group of bars
            ax_bar.bar(segment_indices + offsets, rp_for_current_band, bar_width, 
                       label=band_name, color=colors(band_idx))

        ax_bar.set_xlabel('Segment Number', fontsize=10)
        ax_bar.set_ylabel('Relative Power', fontsize=10)
        ax_bar.set_title(f'Relative Power in Classical Bands for {one_name}', fontsize=12)
        ax_bar.set_xticks(segment_indices)
        ax_bar.set_xticklabels([f'{i+1}' for i in segment_indices], fontsize=8)
        ax_bar.legend(title='Frequency Bands', fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=(0, 0, 0.88, 1)) # Adjust rect to make space for legend outside
        plt.show()
    else:
        logger.info("No data to plot for relative powers.")

except ImportError:
    logger.warning("Matplotlib is not installed or other import error. Plotting will be skipped.")
except Exception as e:
    logger.error(f"Error during visualization: {e}", exc_info=True)

logger.info("Relative Power (RP) processing finished.") 