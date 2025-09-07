import core.eeg_utils as eeg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D # For custom legend
from loguru import logger
from typing import List
from sklearn.cluster import KMeans # type: ignore
from sklearn import metrics # type: ignore

from spectral.median_frequency import calcular_mf
from spectral.spectral_95_limit_frequency import calcular_sef95

# Load all files
logger.info("Loading EEG files...")
all_files, names = eeg.load_files_from_folder('/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH')
logger.success(f"Loaded {len(all_files)} files.")

# Lists to store results
mean_mfs: List[float] = []
mean_sef95s: List[float] = []
targets: List[int] = []
subject_names: List[str] = []

# Process each file
for file_idx, (file, name) in enumerate(zip(all_files, names)):
    logger.info(f"Processing file {file_idx + 1}/{len(all_files)}: {name}")
    try:
        # Get signal and configuration
        signal, cfg, target = eeg.get_nice_data(raw_data=file, name=name, positive_classes_binary=['AD'])
        if target is None:
            logger.error(f"File {name} has no target. Skipping.")
            continue
        # Get spectral density
        f, Pxx = eeg.get_spectral_density(signal, cfg)
        
        if Pxx.ndim == 1: # If Pxx is 1D, make it 2D with one segment
            Pxx = Pxx[np.newaxis, :]
            logger.debug(f"Pxx for {name} was 1D, reshaped to {Pxx.shape}")
        elif Pxx.ndim != 2 or Pxx.shape[0] == 0:
            logger.warning(f"Pxx for {name} has unexpected shape {Pxx.shape} or is empty, skipping.")
            continue
            
        # Get band of interest from cfg
        banda_interes = None
        if 'filtering' in cfg and isinstance(cfg['filtering'], list):
            for filt in cfg['filtering']:
                if isinstance(filt, dict) and filt.get('type') == 'BandPass' and 'band' in filt:
                    banda_interes = filt['band']
                    break
        
        if banda_interes is None:
            logger.warning(f"No BandPass filter found in cfg for {name}, skipping...")
            continue
        logger.debug(f"Band of interest for {name}: {banda_interes}")

        # Calculate MF and SEF95 for each segment
        segment_mfs: List[float] = []
        segment_sef95s: List[float] = []
        
        for seg_idx in range(Pxx.shape[0]):
            psd_segment = Pxx[seg_idx, :]
            
            mf_segment = calcular_mf(psd_segment, f, banda_interes)
            sef95_segment = calcular_sef95(psd_segment, f, banda_interes)
            
            if mf_segment is not None:
                segment_mfs.append(mf_segment)
            if sef95_segment is not None:
                segment_sef95s.append(sef95_segment)
        
        # Ensure we have valid data for both MF and SEF95 to form pairs
        if len(segment_mfs) > 0 and len(segment_sef95s) > 0:
            # If differing numbers of valid segments, take the minimum length or handle appropriately
            # For simplicity, let's assume we want to average if at least one of each is found.
            # A more robust approach might pair them if calculated per segment, 
            # but here we average them separately then pair the averages.
            current_mean_mf = float(np.mean(segment_mfs))
            current_mean_sef95 = float(np.mean(segment_sef95s))
            
            mean_mfs.append(current_mean_mf)
            mean_sef95s.append(current_mean_sef95)
            targets.append(target) # Keep original target
            subject_names.append(name)
            logger.success(f"Subject {name}: Mean MF = {current_mean_mf:.2f} Hz, Mean SEF95 = {current_mean_sef95:.2f} Hz, Target = {target}")
        else:
            logger.warning(f"Could not calculate MF and/or SEF95 for enough segments in {name} to form a pair, skipping...")

    except Exception as e:
        logger.error(f"Error processing file {name}: {e}", exc_info=True)
        continue

# Convert lists to numpy arrays for easier indexing
mean_mfs_array = np.array(mean_mfs, dtype=float)
mean_sef95s_array = np.array(mean_sef95s, dtype=float)
targets_array = np.array(targets, dtype=int) # Original targets

if len(mean_mfs_array) < 2 or len(mean_mfs_array) != len(mean_sef95s_array) or len(mean_mfs_array) != len(targets_array):
    logger.error("Not enough data, mismatched data lengths, or missing targets. Exiting.")
    exit()

# Prepare data for clustering
X_for_clustering = np.column_stack((mean_mfs_array, mean_sef95s_array))

if X_for_clustering.shape[0] < 2: # Need at least 2 samples for k=2
    logger.error("Not enough samples for clustering. Exiting.")
    exit()

# Apply KMeans clustering
num_clusters = 2 # Starting with 2 clusters
logger.info(f"Applying KMeans clustering with k={num_clusters}...")
kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
kmeans.fit(X_for_clustering)
cluster_labels = kmeans.labels_
centroids = kmeans.cluster_centers_

logger.success(f"Clustering complete. Found {num_clusters} clusters.")

# --- Clustering Evaluation Metrics ---
ari_score = metrics.adjusted_rand_score(targets_array, cluster_labels)
nmi_score = metrics.normalized_mutual_info_score(targets_array, cluster_labels)
homogeneity, completeness, v_measure = metrics.homogeneity_completeness_v_measure(targets_array, cluster_labels)
contingency_mat = metrics.cluster.contingency_matrix(targets_array, cluster_labels)

logger.info("--- Clustering Evaluation Metrics ---")
logger.info(f"Adjusted Rand Index (ARI): {ari_score:.4f}")
logger.info(f"Normalized Mutual Information (NMI): {nmi_score:.4f}")
logger.info(f"Homogeneity: {homogeneity:.4f}")
logger.info(f"Completeness: {completeness:.4f}")
logger.info(f"V-measure: {v_measure:.4f}")
logger.info(f"Contingency Matrix (True Labels vs Cluster Labels):\n{contingency_mat}")

# --- Plotting with Cluster Color and True Target Marker ---
logger.info("Generating 2D scatter plot (color by cluster, marker by true target)...")
plt.figure(figsize=(14, 10))

cluster_colors = ['purple', 'orange', 'green', 'cyan', 'magenta'] 
# Ensure enough colors if num_clusters is high
target_markers = ['o', 's', '^', 'X', 'P'] 
# Ensure enough markers for unique target values. Assume 0 and 1 for now.
unique_targets = np.unique(targets_array)

for i in range(num_clusters):
    for target_val_idx, target_val in enumerate(unique_targets):
        mask = (cluster_labels == i) & (targets_array == target_val)
        if np.any(mask):
            plt.scatter(mean_mfs_array[mask], 
                        mean_sef95s_array[mask], 
                        color=cluster_colors[i % len(cluster_colors)], 
                        marker=target_markers[target_val_idx % len(target_markers)], 
                        alpha=0.7, s=50, # s is marker size
                        label=f'Cluster {i}, Target {target_val}' if i == 0 and target_val_idx == 0 else "_nolegend_") # Only label once per combo for cleaner legend

# Plot centroids
for i in range(num_clusters):
    plt.scatter(centroids[i, 0], centroids[i, 1], 
                color=cluster_colors[i % len(cluster_colors)], marker='X', s=250, 
                edgecolors='black', linewidth=1.5, label=f'Cluster {i} Centroid')

plt.title(f'Mean MF vs. SEF95 (k={num_clusters})\nColor: Cluster | Marker: True Target', fontsize=16)
plt.xlabel('Mean Median Frequency (MF) (Hz)', fontsize=12)
plt.ylabel('Mean Spectral Edge Frequency 95% (SEF95) (Hz)', fontsize=12)

# Custom legend
legend_elements = []
for i in range(num_clusters):
    legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                  label=f'Cluster {i}', 
                                  markerfacecolor=cluster_colors[i % len(cluster_colors)], markersize=10))
for target_val_idx, target_val in enumerate(unique_targets):
    legend_elements.append(Line2D([0], [0], marker=target_markers[target_val_idx % len(target_markers)], 
                                  color='w', label=f'True Target {target_val}', 
                                  markerfacecolor='gray', markersize=10))
legend_elements.append(Line2D([0], [0], marker='X', color='w', label='Cluster Centroid',
                           markerfacecolor='black', markersize=10, markeredgecolor='black'))

plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

plt.grid(True, alpha=0.4)

# Add statistics to the plot (inside the plot area)
stats_text = f"Total Subjects: {len(mean_mfs_array)}\n"
stats_text += f"ARI: {ari_score:.3f} | NMI: {nmi_score:.3f}\n"
stats_text += f"Homogeneity: {homogeneity:.3f}\nCompleteness: {completeness:.3f}\nV-measure: {v_measure:.3f}\n\n"
stats_text += "Cluster Sizes:\n"
for i in range(num_clusters):
    stats_text += f"  Cluster {i}: {np.sum(cluster_labels == i)}\n"
stats_text = stats_text.strip()

plt.text(0.02, 0.02, stats_text,
         transform=plt.gca().transAxes,
         fontsize=9,
         verticalalignment='bottom',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8))

plt.tight_layout(rect=(0, 0, 0.85, 1)) # Adjust rect to make space for legend outside
logger.info("Displaying plot...")
plt.show()

logger.success("Script finished.") 