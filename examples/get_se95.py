import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import core.eeg_utils as eeg
from spectral_95_limit_frequency import calcular_sef95 # Import the SEF95 calculation function

# Load data
all_files, names = eeg.load_files_from_folder('/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH')

# Process the first file
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
f, Pxx = eeg.get_spectral_density(signal, cfg)

logger.success(f"Shape of frequency vector f: {f.shape}")
logger.success(f"Shape of PSD matrix Pxx: {Pxx.shape}")

# --- SEF95 Calculation and Plotting Logic ---

# Determine PSD to use (first segment if Pxx is 2D)
if Pxx.ndim == 2:
    # Assuming Pxx is (segments, frequencies)
    # If you want to process a specific channel, you might need to adjust Pxx selection
    # For now, assuming Pxx from get_spectral_density is averaged over channels or for a primary channel
    psd_to_use_for_single_calc = Pxx[0, :] 
    logger.info("Pxx is 2D, using data from the first segment (index 0) for initial SEF95 calculation.")
elif Pxx.ndim == 1:
    psd_to_use_for_single_calc = Pxx
    logger.info("Pxx is 1D, using it directly for initial SEF95 calculation.")
else:
    logger.error(f"Unexpected shape for Pxx: {Pxx.shape}")
    raise ValueError(f"Unexpected shape for Pxx: {Pxx.shape}")

# Get the band of interest from the cfg dictionary
banda_interes = None
if 'filtering' in cfg and isinstance(cfg['filtering'], list):
    for filt in cfg['filtering']:
        if isinstance(filt, dict) and filt.get('type') == 'BandPass' and 'band' in filt:
            banda_interes = filt['band']
            break

if banda_interes is None:
    logger.error("No BandPass filter with 'band' key found in cfg dictionary or cfg['filtering'] is not a list of dicts.")
    # Defaulting to a wide band if not found, or you can raise an error
    # For example, if f is available: banda_interes = [f[0], f[-1]]
    # This part might need adjustment based on how critical the band is.
    # For now, let's raise an error if no band is found as SEF95 is band-dependent.
    raise ValueError("BandPass filter information not found or improperly formatted in cfg.")


logger.success(f"Band of interest from bandpass filter: {banda_interes} Hz")

# Calculate the SEF95 for the selected PSD
sef95 = calcular_sef95(psd_to_use_for_single_calc, f, banda_interes)

if sef95 is not None:
    logger.success(f"The Spectral Edge Frequency 95% (SEF95) in the band {banda_interes} Hz for the first segment is: {sef95:.2f} Hz")
    
    # Optional: Visualización para verificar, similar to MF plotting
    try:
        n_plot_segments = Pxx.shape[0] if Pxx.ndim == 2 else 1
        n_cols = 5 # Adjust as needed
        n_rows = (n_plot_segments + n_cols - 1) // n_cols

        plt.figure(figsize=(15, 2.5 * n_rows if n_rows > 0 else 2.5))
        
        for seg_idx in range(n_plot_segments):
            if Pxx.ndim == 2:
                psd_segment = Pxx[seg_idx, :]
            else: # Pxx is 1D, only one segment to plot
                psd_segment = Pxx
            
            sef95_segment = calcular_sef95(psd_segment, f, banda_interes)
            
            indbanda = np.where((f >= banda_interes[0]) & (f <= banda_interes[1]))[0]
            
            if indbanda.size > 0:
                psd_banda = psd_segment[indbanda]
                f_banda = f[indbanda]
                
                if not (psd_banda.size == 0 or np.all(psd_banda <= 0)):
                    plt.subplot(n_rows if n_rows > 0 else 1, n_cols if n_rows > 0 else 1, seg_idx + 1)
                    plt.plot(f_banda, psd_banda, 'b-', linewidth=0.8)
                    if sef95_segment is not None:
                        plt.axvline(sef95_segment, color='g', linestyle='--', linewidth=0.8, label=f'SEF95: {sef95_segment:.1f} Hz')
                    
                    plt.title(f'Segment {seg_idx+1}', fontsize=9)
                    plt.grid(True, alpha=0.3)
                    
                    if seg_idx % n_cols == 0 or n_rows == 1 and seg_idx == 0 :
                        plt.ylabel('PSD', fontsize=8)
                    if seg_idx >= (n_rows - 1) * n_cols or n_rows == 1:
                        plt.xlabel('Freq (Hz)', fontsize=8)
                    
                    plt.xlim(banda_interes)
                    plt.yticks([])
                    plt.xticks(fontsize=7)
                    if sef95_segment is not None:
                        plt.legend(fontsize=7)
            else:
                logger.warning(f"Segment {seg_idx+1}: No data in the specified band {banda_interes} Hz.")

        plt.tight_layout(rect=(0, 0, 1, 0.96)) # Corrected list to tuple for rect
        plt.suptitle(f"PSD and SEF95 for {one_name} (Band: {banda_interes[0]}-{banda_interes[1]} Hz)", fontsize=12)
        plt.show()
        
        logger.info("\nSummary of SEF95 values:")
        for seg_idx in range(n_plot_segments):
            if Pxx.ndim == 2:
                psd_segment = Pxx[seg_idx, :]
            else:
                psd_segment = Pxx
            sef95_segment = calcular_sef95(psd_segment, f, banda_interes)
            if sef95_segment is not None:
                logger.info(f"Segment {seg_idx+1}: {sef95_segment:.2f} Hz")
            else:
                logger.warning(f"Segment {seg_idx+1}: SEF95 could not be calculated.")

    except ImportError:
        logger.warning("Matplotlib is not installed. Plotting will be skipped.")
    except Exception as e:
        logger.error(f"Error during visualization: {e}", exc_info=True)

else:
    logger.warning(f"Could not calculate SEF95 for the first segment in the band {banda_interes} Hz.")

logger.info("SEF95 processing finished.")
