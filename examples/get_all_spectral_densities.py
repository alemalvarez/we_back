import core.eeg_utils as eeg
import numpy as np
import matplotlib.pyplot as plt
from loguru import logger
from spectral.median_frequency import calcular_mf
from typing import List, Tuple, Optional

# Load all files
all_files, names = eeg.load_files_from_folder('/Users/alemalvarez/code-workspace/TFG/DATA')

# Lists to store results
mean_frequencies: List[float] = []
targets: List[int] = []
subject_names: List[str] = []

# Process each file
for file, name in zip(all_files, names):
    # Get signal and configuration
    signal, cfg, target = eeg.get_nice_data(raw_data=file, name=name)
    
    # Get spectral density
    f, Pxx = eeg.get_spectral_density(signal, cfg)
    
    # Get band of interest from cfg
    banda_interes = None
    for filt in cfg['filtering']:
        if filt['type'] == 'BandPass':
            banda_interes = filt['band']
            break
    
    if banda_interes is None:
        logger.warning(f"No BandPass filter found in cfg for {name}, skipping...")
        continue
    
    # Calculate MF for each segment
    segment_mfs: List[float] = []
    for seg_idx in range(Pxx.shape[0]):
        psd_segment = Pxx[seg_idx, :]
        mf_segment = calcular_mf(psd_segment, f, banda_interes)
        if mf_segment is not None:
            segment_mfs.append(mf_segment)
    
    if segment_mfs:
        # Calculate mean frequency across segments
        mean_freq = float(np.mean(segment_mfs))
        mean_frequencies.append(mean_freq)
        targets.append(target)
        subject_names.append(name)
        logger.info(f"Subject {name}: Mean Frequency = {mean_freq:.2f} Hz")
    else:
        logger.warning(f"Could not calculate MF for any segment in {name}, skipping...")

# Create the plot
plt.figure(figsize=(10, 6))

# Convert lists to numpy arrays for easier indexing
targets_array = np.array(targets, dtype=int)
mean_frequencies_array = np.array(mean_frequencies, dtype=float)

# Plot positive and negative cases with different colors
positive_mask = targets_array == 1
negative_mask = ~positive_mask

# Get indices for plotting
positive_indices = np.arange(np.sum(positive_mask))
negative_indices = np.arange(np.sum(negative_mask))

plt.scatter(positive_indices, 
           mean_frequencies_array[positive_mask], 
           color='red', label='Positive', alpha=0.6)
plt.scatter(negative_indices, 
           mean_frequencies_array[negative_mask], 
           color='blue', label='Negative', alpha=0.6)

# Add mean lines for each group
if np.any(positive_mask):
    mean_positive = float(np.mean(mean_frequencies_array[positive_mask]))
    plt.axhline(y=mean_positive, 
                color='red', linestyle='--', alpha=0.3,
                label=f'Mean Positive: {mean_positive:.2f} Hz')
if np.any(negative_mask):
    mean_negative = float(np.mean(mean_frequencies_array[negative_mask]))
    plt.axhline(y=mean_negative, 
                color='blue', linestyle='--', alpha=0.3,
                label=f'Mean Negative: {mean_negative:.2f} Hz')

plt.title('Mean Frequency by Subject')
plt.xlabel('Subject Index')
plt.ylabel('Mean Frequency (Hz)')
plt.legend()
plt.grid(True, alpha=0.3)

# Add some statistics to the plot
stats_text = f"Total Subjects: {len(mean_frequencies)}\n"
stats_text += f"Positive Cases: {np.sum(positive_mask)}\n"
stats_text += f"Negative Cases: {np.sum(negative_mask)}\n"
stats_text += f"Overall Mean: {float(np.mean(mean_frequencies_array)):.2f} Hz\n"
stats_text += f"Overall Std: {float(np.std(mean_frequencies_array)):.2f} Hz"

plt.text(0.02, 0.98, stats_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.show()

# Print detailed statistics
logger.info("\nDetailed Statistics:")
logger.info(f"Total Subjects: {len(mean_frequencies)}")
logger.info(f"Positive Cases: {np.sum(positive_mask)}")
logger.info(f"Negative Cases: {np.sum(negative_mask)}")
logger.info(f"Overall Mean: {float(np.mean(mean_frequencies_array)):.2f} Hz")
logger.info(f"Overall Std: {float(np.std(mean_frequencies_array)):.2f} Hz")

if np.any(positive_mask) and np.any(negative_mask):
    from scipy import stats  # type: ignore
    t_stat, p_value = stats.ttest_ind(mean_frequencies_array[positive_mask], 
                                     mean_frequencies_array[negative_mask])
    logger.info(f"\nT-test Results:")
    logger.info(f"t-statistic: {t_stat:.4f}")
    logger.info(f"p-value: {p_value:.4f}")



