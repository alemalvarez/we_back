import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from typing import List, Dict, Optional

import core.eeg_utils as eeg  # Assuming eeg_utils.py is in the Python path
from spectral.spectral_bandwidth import calcular_sb

# Attempt to import calcular_sc, which is a prerequisite
try:
    from spectral.spectral_centroid import calcular_sc
except ImportError:
    logger.error("Failed to import 'calcular_sc' from 'CalculoSC'. "
                 "This function is required to calculate Spectral Centroid (SC) "
                 "before calculating Spectral Bandwidth (SB). "
                 "Please ensure 'CalculoSC.py' exists and 'calcular_sc' is defined correctly.")
    # Define a placeholder if not found, so the script can be parsed, but calculations will fail or be incorrect.
    def calcular_sc(psd: np.ndarray, f: np.ndarray, banda: List[float]) -> Optional[float]:
        logger.warning("Using placeholder for 'calcular_sc'. Results will be incorrect.")
        if not isinstance(psd, np.ndarray) or not isinstance(f, np.ndarray) or not isinstance(banda, list):
            return None # Basic type check for placeholder
        if len(banda) != 2 or not all(isinstance(x, (int,float)) for x in banda):
             return None # Basic structure check
        if f.size == 0: # Guard against empty frequency array
            return None
        valid_f_indices = np.where((f >= banda[0]) & (f <= banda[1]))[0]
        if valid_f_indices.size == 0:
            return None
        return float((f[valid_f_indices[0]] + f[valid_f_indices[-1]]) / 2.0)

# --- Configuration ---
DATA_FOLDER_PATH = '/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH' # Update if necessary
CLASSICAL_BANDS: Dict[str, List[float]] = {
    "Delta (0.5-4 Hz)": [0.5, 4.0],
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta1 (13-19 Hz)": [13.0, 19.0],
    "Beta2 (19-30 Hz)": [19.0, 30.0],
    "Gamma (30-70 Hz)": [30.0, 70.0]
}
SUB_BANDAS_LIST: List[List[float]] = list(CLASSICAL_BANDS.values())
SUB_BANDAS_NAMES: List[str] = list(CLASSICAL_BANDS.keys())

# --- Load Data ---
all_files, names = eeg.load_files_from_folder(DATA_FOLDER_PATH)

if not all_files:
    logger.error(f"No files found in {DATA_FOLDER_PATH}. Exiting.")
    exit()

# Process the first file as an example
one_file = all_files[0]
one_name = names[0]
logger.info(f"Processing file: {one_name}")

# --- Preprocessing and PSD Calculation ---
signal, cfg, target = eeg.get_nice_data(raw_data=one_file, name=one_name, comes_from_bbdds=True)
if signal is None or target is None: 
    logger.error(f"File {one_name} yielded no valid signal or target. Skipping.")
    exit()
    
n_segments, n_samples, n_channels = signal.shape
logger.info(f"Signal shape: {signal.shape} (segments, samples, channels)")

f, Pxx = eeg.get_spectral_density(signal, cfg) # Assuming Pxx is (segments, frequencies)
if f is None or Pxx is None:
    logger.error(f"Could not compute PSD for {one_name}. Exiting.")
    exit()
logger.success(f"Shape of frequency vector f: {f.shape}")
logger.success(f"Shape of PSD matrix Pxx: {Pxx.shape}")

# --- Determine Overall Operational Band for SC and SB ---
banda_total_operativa = None
if isinstance(cfg, dict) and 'filtering' in cfg and isinstance(cfg['filtering'], list):
    for filt_item in cfg['filtering']:
        if isinstance(filt_item, dict) and filt_item.get('type') == 'BandPass' and 'band' in filt_item:
            banda_total_operativa = filt_item['band']
            logger.info(f"Using bandpass filter range from cfg for overall SC/SB: {banda_total_operativa} Hz")
            break

if banda_total_operativa is None:
    min_f_val = np.min(f) if f.size > 0 else 0.0
    max_f_val = np.max(f) if f.size > 0 else 1.0 
    min_freq_all_sub_bands = min((b[0] for b in SUB_BANDAS_LIST), default=min_f_val) if SUB_BANDAS_LIST else min_f_val
    max_freq_all_sub_bands = max((b[1] for b in SUB_BANDAS_LIST), default=max_f_val) if SUB_BANDAS_LIST else max_f_val
    banda_total_operativa = [min_freq_all_sub_bands, max_freq_all_sub_bands]
    if not (f.size > 0) and not SUB_BANDAS_LIST:
        logger.warning("Frequency vector 'f' is empty and no sub-bands defined. Defaulting overall SC/SB band to [0.0, 1.0] Hz")
    elif not SUB_BANDAS_LIST:
        logger.warning(f"No sub-bands defined. Defaulting overall SC/SB band to full frequency range: {banda_total_operativa} Hz")
    else:
        logger.warning(f"No BandPass filter range found in cfg. Defaulting overall SC/SB band based on sub-bands or full range: {banda_total_operativa} Hz")

# --- SC and SB Calculation ---
all_total_sc: List[Optional[float]] = []
all_total_sb: List[Optional[float]] = []
all_sub_band_sc: List[List[Optional[float]]] = [] 
all_sub_band_sb: List[List[Optional[float]]] = []

if Pxx.ndim == 1: 
    Pxx_proc = Pxx.reshape(1, -1)
else:
    Pxx_proc = Pxx

num_proc_segments = Pxx_proc.shape[0]

for seg_idx in range(num_proc_segments):
    current_psd_segment = Pxx_proc[seg_idx, :]
    logger.debug(f"Processing segment {seg_idx + 1}/{num_proc_segments}")

    sc_total_seg = calcular_sc(current_psd_segment, f, banda_total_operativa)
    sb_total_seg = None
    if sc_total_seg is not None:
        sb_total_seg = calcular_sb(current_psd_segment, f, banda_total_operativa, sc_total_seg)
    all_total_sc.append(sc_total_seg)
    all_total_sb.append(sb_total_seg)

    sc_values_current_segment: List[Optional[float]] = []
    sb_values_current_segment: List[Optional[float]] = []
    for sub_band in SUB_BANDAS_LIST:
        sc_sub = calcular_sc(current_psd_segment, f, sub_band)
        sb_sub = None
        if sc_sub is not None:
            sb_sub = calcular_sb(current_psd_segment, f, sub_band, sc_sub)
        sc_values_current_segment.append(sc_sub)
        sb_values_current_segment.append(sb_sub)
    all_sub_band_sc.append(sc_values_current_segment)
    all_sub_band_sb.append(sb_values_current_segment)

# --- Logging Results ---
logger.info("\n--- Spectral Centroid (SC) and Spectral Bandwidth (SB) per Segment ---")
for seg_idx in range(num_proc_segments):
    logger.info(f"Segment {seg_idx + 1}:")
    sc_total_val = all_total_sc[seg_idx]
    sb_total_val = all_total_sb[seg_idx]
    band_label = f"{banda_total_operativa[0]:.1f}-{banda_total_operativa[1]:.1f} Hz"
    if sc_total_val is not None and sb_total_val is not None:
        logger.info(f"  Overall Band ({band_label}): "
                    f"SC = {sc_total_val:.2f} Hz, SB = {sb_total_val:.2f} Hz^2")
    else:
        logger.info(f"  Overall Band ({band_label}): SC/SB calculation failed or no power.")

    for band_idx, band_name in enumerate(SUB_BANDAS_NAMES):
        sc_val = all_sub_band_sc[seg_idx][band_idx]
        sb_val = all_sub_band_sb[seg_idx][band_idx]
        if sc_val is not None and sb_val is not None:
            logger.info(f"  {band_name}: SC = {sc_val:.2f} Hz, SB = {sb_val:.2f} Hz^2")
        else:
            logger.info(f"  {band_name}: SC/SB calculation failed or no power.")


# --- Visualization ---
# PlotDataType = Union[float, np.dtype[np.float64]] # np.nan is a float, so this simplifies to float
PlotDataType = float # np.nan is of type float

def plot_sb_results(
    sb_values_list: List[List[Optional[float]]], 
    band_names: List[str],
    segment_indices: np.ndarray,
    title_prefix: str,
    file_name_for_title: str,
    num_total_segments: int 
):
    has_data_to_plot = False
    # Prepare data for plotting: List of Lists, where inner list is for bands, outer for segments
    # Values are float or np.nan
    processed_sb_values_for_plot: List[List[PlotDataType]] = [] 
    for seg_data_list in sb_values_list:
        segment_plot_data: List[PlotDataType] = []
        for val_optional in seg_data_list:
            if val_optional is not None:
                # Ensure it's a Python float or np.nan, not np.float64 object for type consistency if linter is strict
                val_float = float(val_optional) 
                segment_plot_data.append(val_float)
                if not np.isnan(val_float): # Check after conversion
                    has_data_to_plot = True
            else:
                segment_plot_data.append(np.nan)
        processed_sb_values_for_plot.append(segment_plot_data)
            
    if not has_data_to_plot:
        logger.info(f"No valid (non-NaN) SB data to plot for {title_prefix}.")
        return

    fig_bar, ax_bar = plt.subplots(figsize=(max(12, num_total_segments * 1.5), 7))
    n_bands_to_plot = len(band_names)
    bar_width = 0.8 / n_bands_to_plot if n_bands_to_plot > 0 else 0.8
    
    cmap = plt.cm.get_cmap('viridis') 
    colors_list = [cmap(i/n_bands_to_plot) for i in range(n_bands_to_plot)] if n_bands_to_plot > 0 else [cmap(0.5)]

    for band_idx, band_name_plot in enumerate(band_names):
        bar_heights_for_band: List[PlotDataType] = []
        # Ensure band_idx is valid for all segments in processed_sb_values_for_plot
        if processed_sb_values_for_plot and all(len(seg_plot_data) > band_idx for seg_plot_data in processed_sb_values_for_plot):
             bar_heights_for_band = [processed_sb_values_for_plot[seg_idx][band_idx] for seg_idx in range(num_total_segments)]
        else: 
            # This case implies inconsistent number of bands per segment in input, fill with NaN
            bar_heights_for_band = [np.nan] * num_total_segments
        
        offsets = bar_width * band_idx - (bar_width * (n_bands_to_plot -1) / 2) if n_bands_to_plot > 1 else 0
        ax_bar.bar(segment_indices + offsets, np.array(bar_heights_for_band, dtype=float), bar_width,
                   label=band_name_plot, color=colors_list[band_idx])

    ax_bar.set_xlabel('Segment Number', fontsize=10)
    ax_bar.set_ylabel('Spectral Bandwidth (SB) (Hz^2)', fontsize=10)
    ax_bar.set_title(f'{title_prefix} SB for {file_name_for_title}', fontsize=12)
    ax_bar.set_xticks(segment_indices)
    ax_bar.set_xticklabels([f'{i+1}' for i in segment_indices], fontsize=8)
    legend_needed = n_bands_to_plot > 1 or \
                    (n_bands_to_plot == 1 and not band_names[0].startswith("Overall"))
    if legend_needed: 
        ax_bar.legend(title='Frequency Bands', fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    ax_bar.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout(rect=(0, 0, 0.88 if legend_needed else 0.98, 1))
    plt.show()

try:
    if num_proc_segments > 0 :
        current_segment_indices = np.arange(num_proc_segments)
        all_total_sb_for_plot: List[List[Optional[float]]] = [[val] for val in all_total_sb]
        plot_sb_results(
            sb_values_list=all_total_sb_for_plot,
            band_names=[f"Overall ({banda_total_operativa[0]:.1f}-{banda_total_operativa[1]:.1f} Hz)"],
            segment_indices=current_segment_indices,
            title_prefix='Overall Band',
            file_name_for_title=one_name,
            num_total_segments=num_proc_segments
        )

        if SUB_BANDAS_NAMES:
            plot_sb_results(
                sb_values_list=all_sub_band_sb,
                band_names=SUB_BANDAS_NAMES,
                segment_indices=current_segment_indices,
                title_prefix='Sub-Band',
                file_name_for_title=one_name,
                num_total_segments=num_proc_segments
            )
        elif not SUB_BANDAS_NAMES:
            logger.info("No sub-bands defined, skipping sub-band SB plot.")
            
    else:
        logger.info("No segments processed, skipping plotting.")

except ImportError: 
    logger.warning("Matplotlib is not installed or other import error. Plotting will be skipped.")
except Exception as e:
    logger.error(f"Error during visualization: {e}", exc_info=True)

logger.info("Spectral Bandwidth (SB) processing finished.")
