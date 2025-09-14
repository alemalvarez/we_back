import numpy as np
from loguru import logger
from typing import List, Dict, Optional

import core.eeg_utils as eeg
from spectral.relative_powers import calcular_rp
from spectral.individual_alpha_frequency_transition_frequency import calcular_iaftf
from spectral.median_frequency import calcular_mf
from spectral.spectral_95_limit_frequency import calcular_sef95

# --- Configuration ---
# IMPORTANT: Update this path to your actual data folder
DATA_FOLDER_PATH = '/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/HURH'

# For Relative Power
CLASSICAL_BANDS: Dict[str, List[float]] = {
    "Delta (0.5-4 Hz)": [0.5, 4.0],
    "Theta (4-8 Hz)": [4.0, 8.0],
    "Alpha (8-13 Hz)": [8.0, 13.0],
    "Beta1 (13-19 Hz)": [13.0, 19.0],
    "Beta2 (19-30 Hz)": [19.0, 30.0],
    "Gamma (30-70 Hz)": [30.0, 70.0]
}
SUB_BANDAS_LIST_RP: List[List[float]] = list(CLASSICAL_BANDS.values())
SUB_BANDAS_NAMES_RP: List[str] = list(CLASSICAL_BANDS.keys())

# For IAF/TF
Q_IAFTF: List[float] = [4.0, 15.0]  # Typical range [q_min, q_max] for IAF calculation
BANDA_IAFTF: List[float] = [0.5, 70.0]  # Broad band for context for IAF/TF function

# For MF and SEF95 - default band, can be overridden by config's BandPass filter
DEFAULT_WIDE_BAND_FOR_PARAMS: List[float] = [0.5, 70.0]


def main() -> None:
    """
    Main function to load EEG files, calculate average spectral parameters per file, 
    and print summaries.
    """
    logger.info(f"Starting average spectral parameter calculation for files in: {DATA_FOLDER_PATH}")
    
    try:
        all_files, names = eeg.load_files_from_folder(DATA_FOLDER_PATH)
    except Exception as e:
        logger.error(f"Failed to load files from {DATA_FOLDER_PATH}: {e}")
        return

    if not all_files:
        logger.error(f"No .mat files found in {DATA_FOLDER_PATH}. Exiting.")
        return

    for file_idx, (mat_file_content, file_name) in enumerate(zip(all_files, names)):
        logger.info(f"\\n--- Processing file {file_idx + 1}/{len(all_files)}: {file_name} ---")

        try:
            signal_data, cfg, _ = eeg.get_nice_data(
                raw_data=mat_file_content, 
                name=file_name, 
                comes_from_bbdds=True
            )
            
            if signal_data.ndim == 2:
                logger.warning(f"Signal data for {file_name} is 2D, reshaping to (1, samples, channels).")
                signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])
            
            if signal_data.size == 0:
                logger.warning(f"Signal data for {file_name} is empty. Skipping.")
                continue

            logger.info(f"Signal segments: {signal_data.shape[0]}, Samples/segment: {signal_data.shape[1]}, Channels: {signal_data.shape[2]}")
            fs = cfg.get('fs')
            if fs is None:
                logger.error(f"Sampling frequency 'fs' not found in config for {file_name}. Skipping.")
                continue
            logger.info(f"Sampling frequency (fs): {fs} Hz")

            f_freqs, pxx_segments = eeg.get_spectral_density(signal_data, cfg)

            if f_freqs.size == 0 or pxx_segments.size == 0:
                logger.warning(f"PSD calculation resulted in empty arrays for {file_name}. Skipping.")
                continue
            
            logger.success(f"PSD calculated. Frequencies shape: {f_freqs.shape}, Pxx segments shape: {pxx_segments.shape}")

            param_band: List[float] = DEFAULT_WIDE_BAND_FOR_PARAMS
            band_source_info: str = "default wide band"
            if 'filtering' in cfg and isinstance(cfg['filtering'], list):
                for filt_item in cfg['filtering']:
                    if isinstance(filt_item, dict) and \
                       filt_item.get('type') == 'BandPass' and \
                       'band' in filt_item and \
                       isinstance(filt_item['band'], list) and \
                       len(filt_item['band']) == 2:
                        param_band = filt_item['band']
                        band_source_info = f"BandPass filter from cfg: {param_band} Hz"
                        break
            if param_band == DEFAULT_WIDE_BAND_FOR_PARAMS and SUB_BANDAS_LIST_RP:
                min_freq_all_sub_bands = min(band[0] for band in SUB_BANDAS_LIST_RP)
                max_freq_all_sub_bands = max(band[1] for band in SUB_BANDAS_LIST_RP)
                param_band = [min_freq_all_sub_bands, max_freq_all_sub_bands]
                band_source_info = f"derived from RP sub-bands extent: {param_band} Hz"
            logger.info(f"Using reference band for RP, MF, SEF95: {param_band} Hz (source: {band_source_info})")

            # Store parameters for all segments in this file
            all_segment_rps: List[np.ndarray] = []
            all_segment_iafs: List[Optional[float]] = []
            all_segment_tfs: List[Optional[float]] = []
            all_segment_mfs: List[Optional[float]] = []
            all_segment_sef95s: List[Optional[float]] = []

            for seg_idx in range(pxx_segments.shape[0]):
                current_psd_segment: np.ndarray = pxx_segments[seg_idx, :]
                
                try:
                    rp_values = calcular_rp(current_psd_segment, f_freqs, param_band, SUB_BANDAS_LIST_RP)
                    all_segment_rps.append(rp_values)
                except Exception as e_rp:
                    logger.error(f"Error calculating RP for file {file_name}, segment {seg_idx + 1}: {e_rp}. Appending NaNs.")
                    all_segment_rps.append(np.full(len(SUB_BANDAS_LIST_RP) if SUB_BANDAS_LIST_RP else 0, np.nan))


                try:
                    iaf, tf = calcular_iaftf(current_psd_segment, f_freqs, BANDA_IAFTF, Q_IAFTF)
                    all_segment_iafs.append(iaf)
                    all_segment_tfs.append(tf)
                except Exception as e_iaftf:
                    logger.error(f"Error calculating IAF/TF for file {file_name}, segment {seg_idx + 1}: {e_iaftf}")
                    all_segment_iafs.append(None)
                    all_segment_tfs.append(None)

                try:
                    mf = calcular_mf(current_psd_segment, f_freqs, param_band)
                    all_segment_mfs.append(mf)
                except Exception as e_mf:
                    logger.error(f"Error calculating MF for file {file_name}, segment {seg_idx + 1}: {e_mf}")
                    all_segment_mfs.append(None)

                try:
                    sef95 = calcular_sef95(current_psd_segment, f_freqs, param_band)
                    all_segment_sef95s.append(sef95)
                except Exception as e_sef95:
                    logger.error(f"Error calculating SEF95 for file {file_name}, segment {seg_idx + 1}: {e_sef95}")
                    all_segment_sef95s.append(None)

            # Calculate and log average parameters for the file
            logger.info(f"\\n  Average Spectral Parameters for file: {file_name}")

            # Average RP
            if all_segment_rps and SUB_BANDAS_NAMES_RP:
                # Convert list of arrays to a 2D NumPy array for nanmean
                # Ensure all arrays in all_segment_rps have the same length (as SUB_BANDAS_NAMES_RP)
                # This should be guaranteed by calcular_rp or the error handling above
                try:
                    np_all_segment_rps = np.array(all_segment_rps)
                    if np_all_segment_rps.ndim == 2 and np_all_segment_rps.shape[1] == len(SUB_BANDAS_NAMES_RP):
                         with np.errstate(all='ignore'): # Suppress warnings for all-NaN slices
                            avg_rp_values = np.nanmean(np_all_segment_rps, axis=0)
                         logger.info("    Average Relative Powers (RP):")
                         for band_name, avg_rp_val in zip(SUB_BANDAS_NAMES_RP, avg_rp_values):
                             logger.info(f"      {band_name}: {avg_rp_val:.4f}" if not np.isnan(avg_rp_val) else f"      {band_name}: N/A")
                    else:
                        logger.warning("    Could not compute average RP: Mismatch in RP array dimensions or empty sub-band names.")
                except Exception as e_avg_rp:
                    logger.error(f"    Error averaging RP values: {e_avg_rp}")
            elif not SUB_BANDAS_NAMES_RP:
                 logger.info("    Average Relative Powers (RP): No sub-bands defined.")
            else: # all_segment_rps is empty
                logger.info("    Average Relative Powers (RP): No RP values to average.")

            # Average IAF, TF, MF, SEF95 (filtering out None before nanmean)
            with np.errstate(all='ignore'): # Suppress warnings for all-NaN slices
                avg_iaf = np.nanmean([val for val in all_segment_iafs if val is not None])
                avg_tf = np.nanmean([val for val in all_segment_tfs if val is not None])
                avg_mf = np.nanmean([val for val in all_segment_mfs if val is not None])
                avg_sef95 = np.nanmean([val for val in all_segment_sef95s if val is not None])

            logger.info("    Average IAF & TF:")
            logger.info(f"      Individual Alpha Frequency (IAF): {avg_iaf:.2f} Hz" if not np.isnan(avg_iaf) else "      Individual Alpha Frequency (IAF): N/A")
            logger.info(f"      Transition Frequency (TF): {avg_tf:.2f} Hz" if not np.isnan(avg_tf) else "      Transition Frequency (TF): N/A")
            
            logger.info("    Average Median Frequency (MF):")
            logger.info(f"      MF: {avg_mf:.2f} Hz (within band {param_band})" if not np.isnan(avg_mf) else f"      MF: N/A (within band {param_band})")

            logger.info("    Average Spectral Edge Frequency 95% (SEF95):")
            logger.info(f"      SEF95: {avg_sef95:.2f} Hz (within band {param_band})" if not np.isnan(avg_sef95) else f"      SEF95: N/A (within band {param_band})")

        except Exception as e_file:
            logger.error(f"Could not process file {file_name} due to an error: {e_file}", exc_info=True)

    logger.info("\\n--- All files processed. ---")

if __name__ == "__main__":
    main()
