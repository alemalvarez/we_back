import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import List, Dict, Any, Optional

import core.eeg_utils as eeg
from spectral.spectral_crest_factor import calcular_scf # Import the SCF calculation function

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
# Keep names for plotting
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

logger.info(f"Processing file: {one_name} for Spectral Crest Factor (SCF)")

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

# --- Determine Overall Band for SCF Calculation ---
overall_band_for_scf: Optional[List[float]] = None
if 'filtering' in cfg and isinstance(cfg['filtering'], list):
    for filt_item in cfg['filtering']:
        if isinstance(filt_item, dict) and filt_item.get('type') == 'BandPass' and 'band' in filt_item:
            overall_band_for_scf = filt_item['band']
            logger.info(f"Using bandpass filter range from cfg for Overall SCF: {overall_band_for_scf} Hz")
            break

if overall_band_for_scf is None:
    if f.size > 0:
        overall_band_for_scf = [f.min(), f.max()]
        logger.warning(
            f"No BandPass filter range found in cfg. Defaulting Overall SCF band to: {overall_band_for_scf} Hz"
        )
    else:
        logger.error("Frequency vector 'f' is empty. Cannot determine overall band for SCF.")
        # Handle error appropriately, perhaps by exiting or skipping overall SCF
        # For now, we'll let it be None and it will be skipped in calculation if it remains None.

# --- SCF Calculation --- 
all_scf_values_per_segment: List[List[Optional[float]]] = [] # List to store SCF arrays for each segment

if Pxx.ndim == 1: # Handle case where Pxx might be 1D (single segment)
    Pxx_proc = Pxx.reshape(1, -1) # Treat as a single segment
    n_plot_segments = 1
else:
    Pxx_proc = Pxx
    n_plot_segments = Pxx_proc.shape[0]

for seg_idx in range(n_plot_segments):
    current_psd_segment = Pxx_proc[seg_idx, :]
    scf_values_for_current_segment: List[Optional[float]] = []
    # Calculate SCF for classical bands
    for band_name, band_limits in CLASSICAL_BANDS.items():
        scf_val = calcular_scf(current_psd_segment, f, band_limits)
        scf_values_for_current_segment.append(scf_val)
        if scf_val is not None:
            logger.trace(f"Segment {seg_idx + 1}, Band '{band_name}': SCF = {scf_val:.4f}")
        else:
            logger.trace(f"Segment {seg_idx + 1}, Band '{band_name}': SCF = None")
    
    # Calculate SCF for the overall band
    if overall_band_for_scf is not None and f.size > 0:
        overall_scf_val = calcular_scf(current_psd_segment, f, overall_band_for_scf)
        scf_values_for_current_segment.append(overall_scf_val)
        if overall_scf_val is not None:
            logger.trace(f"Segment {seg_idx + 1}, Band 'Overall': SCF = {overall_scf_val:.4f}")
        else:
            logger.trace(f"Segment {seg_idx + 1}, Band 'Overall': SCF = None")
    else:
        scf_values_for_current_segment.append(None) # Append None if overall band is not available
        logger.trace(f"Segment {seg_idx + 1}, Band 'Overall': SCF = None (overall band not defined or f empty)")

    all_scf_values_per_segment.append(scf_values_for_current_segment)

# --- Logging Results ---
logger.info("\n--- Spectral Crest Factor (SCF) per Segment ---")
# Define band names for logging and plotting, including "Overall"
LOG_PLOT_BAND_NAMES = SUB_BANDAS_NAMES + ["Overall SCF"]

for seg_idx, scf_values_segment in enumerate(all_scf_values_per_segment):
    logger.info(f"Segment {seg_idx + 1}:")
    # Ensure scf_values_segment has the same length as LOG_PLOT_BAND_NAMES
    # This should be true if overall_scf_val was appended (even as None)
    for band_name, scf_val in zip(LOG_PLOT_BAND_NAMES, scf_values_segment):
        if scf_val is not None:
            logger.info(f"  {band_name}: {scf_val:.4f}")
        else:
            logger.info(f"  {band_name}: None")

# --- Visualization ---
try:
    if n_plot_segments > 0 and any(any(val is not None for val in seg) for seg in all_scf_values_per_segment):
        fig_bar, ax_bar = plt.subplots(figsize=(max(10, n_plot_segments * 2), 7)) # Adjusted figsize slightly
        
        n_bands = len(LOG_PLOT_BAND_NAMES) # Use the updated list of band names
        segment_indices = np.arange(n_plot_segments)
        bar_width = 0.8 / n_bands 
        
        if n_bands <= 10:
            colors = plt.cm.get_cmap('tab10', n_bands)
        elif n_bands <=20:
            colors = plt.cm.get_cmap('tab20', n_bands)
        else: 
            colors = plt.cm.get_cmap('viridis', n_bands)

        for band_idx, band_name in enumerate(LOG_PLOT_BAND_NAMES): # Use updated list
            scf_for_current_band_list = [
                scf_seg[band_idx] if scf_seg and len(scf_seg) > band_idx and scf_seg[band_idx] is not None else np.nan 
                for scf_seg in all_scf_values_per_segment
            ]
            scf_for_current_band_np = np.array(scf_for_current_band_list, dtype=float) 
            
            offsets = bar_width * band_idx - (bar_width * (n_bands -1) / 2) 
            ax_bar.bar(segment_indices + offsets, scf_for_current_band_np, bar_width, 
                       label=band_name, color=colors(band_idx / (n_bands -1 if n_bands > 1 else 1)))

        ax_bar.set_xlabel('Segment Number', fontsize=10)
        ax_bar.set_ylabel('Spectral Crest Factor (SCF)', fontsize=10)
        ax_bar.set_title(f'Spectral Crest Factor (SCF) in Classical Bands for {one_name}', fontsize=12)
        ax_bar.set_xticks(segment_indices)
        ax_bar.set_xticklabels([f'{i+1}' for i in segment_indices], fontsize=8)
        ax_bar.legend(title='Frequency Bands', fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout(rect=(0, 0, 0.88, 1)) # Adjust rect to make space for legend outside
        plt.show()
    else:
        logger.info("No valid SCF data to plot.")

except ImportError:
    logger.warning("Matplotlib is not installed or other import error. Plotting will be skipped.")
except Exception as e:
    logger.error(f"Error during visualization: {e}", exc_info=True)

logger.info("Spectral Crest Factor (SCF) processing finished.") 