import numpy as np
import matplotlib.pyplot as plt
from loguru import logger

import core.eeg_utils as eeg
from spectral.median_frequency import calcular_mf

all_files, names = eeg.load_files_from_folder('/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH')

one_file = all_files[0]
one_name = names[0]

signal, cfg, target = eeg.get_nice_data(raw_data=one_file, name=one_name)

n_segments, n_samples, n_channels = signal.shape

logger.info(f"n_samples: {n_samples}")
logger.info(f"n_channels: {n_channels}")
logger.info(f"n_segments: {n_segments}")

f, Pxx = eeg.get_spectral_density(signal, cfg)

logger.success(f"Shape of f: {f.shape}")
logger.success(f"Shape of Pxx: {Pxx.shape}")

# --- Added MF calculation and plotting logic ---

# Assuming Pxx might be 2D (segments, frequencies), use the first segment
if Pxx.ndim == 2:
    psd_to_use = Pxx[0, :] 
    logger.info("Pxx is 2D, using data from the first segment (index 0).")
elif Pxx.ndim == 1:
    psd_to_use = Pxx
    logger.info("Pxx is 1D, using it directly.")
else:
    raise ValueError(f"Unexpected shape for Pxx: {Pxx.shape}")

# Get the band of interest from the cfg dictionary
banda_interes = None
for filt in cfg['filtering']:
    if filt['type'] == 'BandPass':
        banda_interes = filt['band']
        break

if banda_interes is None:
    raise ValueError("No BandPass filter found in cfg dictionary")

logger.success(f"Band of interest from bandpass filter: {banda_interes}")

# Calculate the median frequency
mf = calcular_mf(psd_to_use, f, banda_interes)

if mf is not None:
    logger.success(f"La Frecuencia Mediana (MF) en la banda {banda_interes} Hz es: {mf:.2f} Hz")
    # Optional: Visualización para verificar
    try:
        # Calculate number of rows and columns for the grid
        n_segments = Pxx.shape[0]
        n_cols = 10  # Increased number of columns
        n_rows = (n_segments + n_cols - 1) // n_cols  # Ceiling division
        
        # Create a figure for all segments with smaller size
        plt.figure(figsize=(15, 2.5 * n_rows))
        
        # Process each segment
        for seg_idx in range(n_segments):
            psd_segment = Pxx[seg_idx, :]
            
            # Calculate MF for this segment
            mf_segment = calcular_mf(psd_segment, f, banda_interes)
            
            # Get indices for the band of interest
            indbanda = np.where((f >= banda_interes[0]) & (f <= banda_interes[1]))[0]
            
            if indbanda.size > 0:
                psd_banda = psd_segment[indbanda]
                f_banda = f[indbanda]
                
                if not (psd_banda.size == 0 or np.all(psd_banda <= 0)):
                    # Calculate cumulative power
                    vector_suma = np.cumsum(psd_banda)
                    potencia_total = np.sum(psd_banda)
                    
                    # Create subplot for this segment
                    plt.subplot(n_rows, n_cols, seg_idx + 1)
                    
                    # Plot PSD
                    plt.plot(f_banda, psd_banda, 'b-', linewidth=0.5)  # Reduced line width
                    if mf_segment is not None:
                        plt.axvline(mf_segment, color='r', linestyle='--', linewidth=0.5)  # Reduced line width
                    
                    # Add segment number and MF value with smaller font
                    plt.title(f'Seg {seg_idx+1}\nMF: {mf_segment:.1f} Hz' if mf_segment is not None else f'Seg {seg_idx+1}', 
                             fontsize=8)  # Reduced font size
                    plt.grid(True, alpha=0.2)  # Reduced grid opacity
                    
                    # Only show x and y labels for the first plot in each row
                    if seg_idx % n_cols == 0:
                        plt.ylabel('PSD', fontsize=8)  # Reduced font size
                    if seg_idx >= (n_rows - 1) * n_cols:
                        plt.xlabel('Freq (Hz)', fontsize=8)  # Reduced font size
                    
                    # Set x-axis limits to the band of interest
                    plt.xlim(banda_interes)
                    
                    # Remove y-axis ticks to save space
                    plt.yticks([])
                    
                    # Reduce tick label size
                    plt.xticks(fontsize=6)  # Reduced tick label size
        
        plt.tight_layout()
        plt.show()
        
        # Print summary of MF values
        logger.info("\nSummary of Median Frequencies:")
        for seg_idx in range(n_segments):
            psd_segment = Pxx[seg_idx, :]
            mf_segment = calcular_mf(psd_segment, f, banda_interes)
            if mf_segment is not None:
                logger.info(f"Segment {seg_idx+1}: {mf_segment:.2f} Hz")

    except ImportError:
        logger.warning("Matplotlib no está instalado. No se generará la gráfica.")
    except Exception as e:
        logger.error(f"Error durante la visualización: {e}")

else:
    logger.warning(f"No se pudo calcular la Frecuencia Mediana en la banda {banda_interes} Hz.")


