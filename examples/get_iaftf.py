import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import core.eeg_utils as eeg
from spectral.individual_alpha_frequency_transition_frequency import calcular_iaftf # Import the IAF/TF calculation function

# Load data
# Replace with the actual path to your data folder
all_files, names = eeg.load_files_from_folder('/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH') 

# Process the first file as an example
one_file = all_files[0]
one_name = names[0]

logger.info(f"Processing file: {one_name}")

signal, cfg, target = eeg.get_nice_data(raw_data=one_file, name=one_name, comes_from_bbdds=True)

n_segments, n_samples, n_channels = signal.shape

logger.info(f"Signal shape: {signal.shape}")
logger.info(f"Number of segments: {n_segments}")
logger.info(f"Samples per segment: {n_samples}")
logger.info(f"Number of channels: {n_channels}")

# Get spectral density
f, Pxx = eeg.get_spectral_density(signal, cfg, nperseg=256)

logger.success(f"Shape of frequency vector f: {f.shape}")
logger.success(f"Shape of PSD matrix Pxx: {Pxx.shape}")

# --- IAF/TF Calculation and Plotting Logic ---

# Determine PSD to use (first segment if Pxx is 2D for initial test)
if Pxx.ndim == 2:
    # Assuming Pxx is (segments, frequencies)
    psd_to_use_for_single_calc = Pxx[0, :] 
    logger.info("Pxx is 2D, using data from the first segment (index 0) for initial IAF/TF calculation.")
elif Pxx.ndim == 1:
    psd_to_use_for_single_calc = Pxx
    logger.info("Pxx is 1D, using it directly for initial IAF/TF calculation.")
else:
    logger.error(f"Unexpected shape for Pxx: {Pxx.shape}")
    raise ValueError(f"Unexpected shape for Pxx: {Pxx.shape}")

# Get the overall band of interest from the cfg dictionary (for the 'banda' parameter of calcular_iaftf)
banda_param_for_iaftf = None
if 'filtering' in cfg and isinstance(cfg['filtering'], list):
    for filt in cfg['filtering']:
        if isinstance(filt, dict) and filt.get('type') == 'BandPass' and 'band' in filt:
            banda_param_for_iaftf = filt['band']
            break

if banda_param_for_iaftf is None:
    logger.warning("No BandPass filter with 'band' key found in cfg. Defaulting 'banda' to a wide range.")
    if f.size > 0:
        banda_param_for_iaftf = [f[0], f[-1]]
    else:
        logger.error("Frequency vector 'f' is empty. Cannot set default 'banda'.")
        raise ValueError("Frequency vector 'f' is empty and BandPass info not found in cfg.")

# Define the 'q' band for IAF calculation (typical alpha range)
q_band_for_iaf = [7.0, 13.0] 
logger.info(f"Using 'q' band for IAF calculation: {q_band_for_iaf} Hz")
logger.success(f"Overall 'banda' parameter for calcular_iaftf (from filter or default): {banda_param_for_iaftf} Hz")


# Calculate IAF and TF for the selected PSD (first segment)
iaf_initial, tf_initial = calcular_iaftf(psd_to_use_for_single_calc, f, banda_param_for_iaftf, q_band_for_iaf)

if iaf_initial is not None:
    logger.success(f"Initial IAF for the first segment (q_band {q_band_for_iaf} Hz): {iaf_initial:.2f} Hz")
else:
    logger.warning(f"Could not calculate initial IAF for the first segment (q_band {q_band_for_iaf} Hz).")
if tf_initial is not None:
    logger.success(f"Initial TF for the first segment: {tf_initial:.2f} Hz")
else:
    logger.warning("Could not calculate initial TF for the first segment.")
    

# Optional: Visualización
try:
    n_plot_segments = Pxx.shape[0] if Pxx.ndim == 2 else 1
    n_cols = 4 # Adjust as needed
    n_rows = (n_plot_segments + n_cols - 1) // n_cols

    plt.figure(figsize=(15, 3 * n_rows if n_rows > 0 else 3)) # Adjusted figure size
    
    all_iaf_values = []
    all_tf_values = []

    for seg_idx in range(n_plot_segments):
        current_psd_segment = Pxx[seg_idx, :] if Pxx.ndim == 2 else Pxx
        
        iaf_segment, tf_segment = calcular_iaftf(current_psd_segment, f, banda_param_for_iaftf, q_band_for_iaf)
        all_iaf_values.append(iaf_segment)
        all_tf_values.append(tf_segment)
        
        # Plotting band, can be banda_param_for_iaftf or a wider fixed range like [0, 30] or [0, 40]
        # Using banda_param_for_iaftf for consistency with SEF95 plotting, but ensure q_band is visible.
        plot_band_display = banda_param_for_iaftf 
        
        # For plotting, we show the PSD within the plot_band_display
        ind_plot_band = np.where((f >= plot_band_display[0]) & (f <= plot_band_display[1]))[0]
        
        if ind_plot_band.size > 0:
            psd_in_plot_band = current_psd_segment[ind_plot_band]
            f_in_plot_band = f[ind_plot_band]
            
            if not (psd_in_plot_band.size == 0 or np.all(psd_in_plot_band <= 0)):
                plt.subplot(n_rows if n_rows > 0 else 1, n_cols if n_rows > 0 else 1, seg_idx + 1)
                plt.plot(f_in_plot_band, psd_in_plot_band, 'b-', linewidth=0.8, label='PSD')
                
                # Plot q_band region for IAF
                plt.axvspan(q_band_for_iaf[0], q_band_for_iaf[1], color='whitesmoke', alpha=0.9, zorder=-1, label=f'IAF q-band ({q_band_for_iaf[0]}-{q_band_for_iaf[1]} Hz)')


                if iaf_segment is not None:
                    plt.axvline(iaf_segment, color='g', linestyle='--', linewidth=1.0, label=f'IAF: {iaf_segment:.1f} Hz')
                if tf_segment is not None:
                    plt.axvline(tf_segment, color='r', linestyle=':', linewidth=1.0, label=f'TF: {tf_segment:.1f} Hz')
                
                plt.title(f'Segment {seg_idx+1}', fontsize=9)
                plt.grid(True, alpha=0.3)
                
                if seg_idx % n_cols == 0 or (n_rows == 1 and seg_idx == 0) :
                    plt.ylabel('PSD', fontsize=8)
                if seg_idx >= (n_plot_segments - n_cols) or n_rows == 1 : # Ensure xlabel on last row
                    plt.xlabel('Freq (Hz)', fontsize=8)
                
                plt.xlim(plot_band_display)
                plt.yticks([])
                plt.xticks(fontsize=7)
                plt.legend(fontsize=7, loc='upper right')
        else:
            logger.warning(f"Segment {seg_idx+1}: No data in the specified plot display band {plot_band_display} Hz.")

    plt.tight_layout(rect=(0, 0, 1, 0.95)) 
    plt.suptitle(f"PSD, IAF, and TF for {one_name} (Overall band: {banda_param_for_iaftf[0]:.1f}-{banda_param_for_iaftf[1]:.1f} Hz; IAF q-band: {q_band_for_iaf[0]}-{q_band_for_iaf[1]} Hz)", fontsize=12)
    plt.show()
    
    logger.info("Summary of IAF and TF values:")
    for seg_idx in range(n_plot_segments):
        logger.info(f"Segment {seg_idx+1}: IAF = {all_iaf_values[seg_idx]:.2f} Hz, TF = {all_tf_values[seg_idx]:.2f} Hz" 
                    if all_iaf_values[seg_idx] is not None and all_tf_values[seg_idx] is not None 
                    else f"Segment {seg_idx+1}: IAF = {'N/A' if all_iaf_values[seg_idx] is None else f'{all_iaf_values[seg_idx]:.2f} Hz'}, TF = {'N/A' if all_tf_values[seg_idx] is None else f'{all_tf_values[seg_idx]:.2f} Hz'}")

except ImportError:
    logger.warning("Matplotlib is not installed or other import error. Plotting will be skipped.")
except Exception as e:
    logger.error(f"Error during visualization or processing: {e}", exc_info=True)


logger.info("IAF/TF processing finished.") 