import numpy as np
from loguru import logger
from typing import List, Dict, Any, Tuple, Optional

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
    Main function to load EEG files, calculate spectral parameters, and print summaries.
    """
    logger.info(f"Starting spectral parameter calculation for files in: {DATA_FOLDER_PATH}")
    
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
            # Assuming 'comes_from_bbdds=True' based on typical usage. Adjust if needed.
            signal_data, cfg, _ = eeg.get_nice_data(
                raw_data=mat_file_content, 
                name=file_name, 
                comes_from_bbdds=True 
            )
            
            if signal_data.ndim == 2:  # Reshape if (samples, channels)
                logger.warning(f"Signal data for {file_name} is 2D, reshaping to (1, samples, channels).")
                signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])
            
            if signal_data.size == 0:
                logger.warning(f"Signal data for {file_name} is empty after loading. Skipping.")
                continue

            n_segments = signal_data.shape[0]
            logger.info(f"Signal segments: {n_segments}, Samples/segment: {signal_data.shape[1]}, Channels: {signal_data.shape[2]}")
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

            # Determine the reference band for RP, MF, SEF95 calculations
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
                # Fallback: if no BandPass in cfg, use extent of RP sub-bands if defined
                min_freq_all_sub_bands = min(band[0] for band in SUB_BANDAS_LIST_RP)
                max_freq_all_sub_bands = max(band[1] for band in SUB_BANDAS_LIST_RP)
                param_band = [min_freq_all_sub_bands, max_freq_all_sub_bands]
                band_source_info = f"derived from RP sub-bands extent: {param_band} Hz"
            
            logger.info(f"Using reference band for RP, MF, SEF95: {param_band} Hz (source: {band_source_info})")

            for seg_idx in range(pxx_segments.shape[0]):
                current_psd_segment: np.ndarray = pxx_segments[seg_idx, :]
                logger.info(f"\\n  Segment {seg_idx + 1}/{pxx_segments.shape[0]}:")

                # 1. Relative Power (RP)
                try:
                    rp_values: np.ndarray = calcular_rp(current_psd_segment, f_freqs, param_band, SUB_BANDAS_LIST_RP)
                    logger.info("    Relative Powers (RP):")
                    if rp_values.size == len(SUB_BANDAS_NAMES_RP):
                        for band_name, rp_val in zip(SUB_BANDAS_NAMES_RP, rp_values):
                            logger.info(f"      {band_name}: {rp_val:.4f}")
                    elif rp_values.size == 0 and not SUB_BANDAS_LIST_RP:
                         logger.info("      No sub-bands defined for RP calculation.")
                    else:
                        logger.warning(f"      RP calculation output size ({rp_values.size}) "
                                       f"mismatches number of sub-band names ({len(SUB_BANDAS_NAMES_RP)}).")
                        logger.debug(f"      RP values: {rp_values}")
                except Exception as e_rp:
                    logger.error(f"      Error calculating RP for segment {seg_idx + 1}: {e_rp}")

                # 2. IAF and TF
                try:
                    iaf, tf = calcular_iaftf(current_psd_segment, f_freqs, BANDA_IAFTF, Q_IAFTF)
                    logger.info("    IAF & TF:")
                    logger.info(f"      Individual Alpha Frequency (IAF): {iaf if iaf is not None else 'N/A'} Hz (q_band: {Q_IAFTF} Hz)")
                    logger.info(f"      Transition Frequency (TF): {tf if tf is not None else 'N/A'} Hz")
                except Exception as e_iaftf:
                    logger.error(f"      Error calculating IAF/TF for segment {seg_idx + 1}: {e_iaftf}")

                # 3. Median Frequency (MF)
                try:
                    mf: Optional[float] = calcular_mf(current_psd_segment, f_freqs, param_band)
                    logger.info("    Median Frequency (MF):")
                    logger.info(f"      MF: {mf if mf is not None else 'N/A'} Hz (within band {param_band})")
                except Exception as e_mf:
                    logger.error(f"      Error calculating MF for segment {seg_idx + 1}: {e_mf}")

                # 4. Spectral Edge Frequency 95% (SEF95)
                try:
                    sef95: Optional[float] = calcular_sef95(current_psd_segment, f_freqs, param_band)
                    logger.info("    Spectral Edge Frequency 95% (SEF95):")
                    logger.info(f"      SEF95: {sef95 if sef95 is not None else 'N/A'} Hz (within band {param_band})")
                except Exception as e_sef95:
                    logger.error(f"      Error calculating SEF95 for segment {seg_idx + 1}: {e_sef95}")
            
        except Exception as e_file:
            logger.error(f"Could not process file {file_name} due to an error: {e_file}", exc_info=True)

    logger.info("\\n--- All files processed. ---")

if __name__ == "__main__":
    main() 