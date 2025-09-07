import os
import numpy as np
import pandas as pd # type: ignore
from loguru import logger
from typing import List, Dict, Optional, Any
from collections import defaultdict

from sklearn.ensemble import RandomForestClassifier # type: ignore
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # type: ignore
from sklearn.preprocessing import StandardScaler # type: ignore
from sklearn.impute import SimpleImputer # type: ignore

import core.eeg_utils as eeg
from spectral.relative_powers import calcular_rp
from spectral.individual_alpha_frequency_transition_frequency import calcular_iaftf
from spectral.median_frequency import calcular_mf
from spectral.spectral_95_limit_frequency import calcular_sef95

# --- Configuration ---
DATA_FOLDER_PATH = '/Users/alemalvarez/code-workspace/TFG/DATA/BBDDs/POCTEP'

# Configuration for binary classification
# Patterns (substrings) to identify classes in filenames.
# Order of checking: ignore -> positive -> negative. 
# If a file doesn't match any, its binary_target will be None.
POSITIVE_CLASSES_BINARY_CONFIG: List[str] = ['ADMIL', 'ADMOD', 'ADSEV'] # e.g. AD, MCI are positive
NEGATIVE_CLASSES_BINARY_CONFIG: List[str] = ['HC']              # e.g. HC is negative
IGNORE_CLASSES_BINARY_CONFIG: List[str] = ['MCI']            # e.g. ['OTHER'] to ignore files with OTHER in name for binary task

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
FEATURE_NAMES_RP: List[str] = [f"RP_{name.split(' ')[0]}" for name in SUB_BANDAS_NAMES_RP]

Q_IAFTF: List[float] = [4, 15]
BANDA_IAFTF: List[float] = [0.5, 70.0]
DEFAULT_WIDE_BAND_FOR_PARAMS: List[float] = [0.5, 70.0]

FEATURE_NAMES_OTHER: List[str] = ["IAF", "TF", "MF", "SEF95"]
ALL_FEATURE_NAMES: List[str] = FEATURE_NAMES_RP + FEATURE_NAMES_OTHER

MULTI_CLASS_CATEGORIES_MAP: Dict[str, str] = {
    "ADMIL": "ADMIL",
    "ADMOD": "ADMOD",
    "HC": "HC",
    "MCI": "MCI",
    "ADSEV": "ADSEV"
}

def get_multiclass_label(file_name: str) -> Optional[str]:
    for prefix, label in MULTI_CLASS_CATEGORIES_MAP.items():
        if file_name.upper().startswith(prefix.upper()): # Ensure case-insensitive prefix matching
            return label
    return None

def extract_features_for_segment(
    psd_segment: np.ndarray, 
    f_freqs: np.ndarray, 
    param_band: List[float]
) -> Dict[str, Optional[float]]:
    features = {}
    try:
        rp_values = calcular_rp(psd_segment, f_freqs, param_band, SUB_BANDAS_LIST_RP)
        if rp_values.size == len(FEATURE_NAMES_RP):
            for name, val in zip(FEATURE_NAMES_RP, rp_values):
                features[name] = val if not np.isnan(val) else None
        else: raise ValueError("RP values size mismatch")
    except Exception:
        for name in FEATURE_NAMES_RP: features[name] = None
    try:
        iaf, tf = calcular_iaftf(psd_segment, f_freqs, BANDA_IAFTF, Q_IAFTF)
        features["IAF"] = iaf; features["TF"] = tf
    except Exception:
        features["IAF"] = None; features["TF"] = None
    try:
        features["MF"] = calcular_mf(psd_segment, f_freqs, param_band)
    except Exception:
        features["MF"] = None
    try:
        features["SEF95"] = calcular_sef95(psd_segment, f_freqs, param_band)
    except Exception:
        features["SEF95"] = None
    return features

def load_and_extract_features_from_files(
    file_list: List[str],
    mat_files_map: Dict[str, Any] 
) -> List[Dict[str, Any]]:
    segment_data_list = []
    for file_name in file_list:
        mat_content = mat_files_map.get(file_name)
        if mat_content is None:
            logger.warning(f"Content for {file_name} not found in map. Skipping.")
            continue
        try:
            signal_data, cfg, binary_target_value = eeg.get_nice_data(
                raw_data=mat_content, name=file_name, 
                positive_classes_binary=POSITIVE_CLASSES_BINARY_CONFIG,
                negative_classes_binary=NEGATIVE_CLASSES_BINARY_CONFIG,
                ignore_classes_binary=IGNORE_CLASSES_BINARY_CONFIG,
                comes_from_bbdds=True
            )
            if binary_target_value is None: # File was globally ignored by get_nice_data
                logger.info(f"File {file_name} skipped by get_nice_data (globally ignored).")
                continue
            
            multiclass_label_file = get_multiclass_label(file_name)

            if multiclass_label_file is None: # If still no multiclass label, skip (should be caught by categorization earlier too)
                logger.warning(f"File {file_name} has no multi-class label after get_nice_data. Skipping feature extraction for it.")
                continue

            if signal_data.ndim == 2: signal_data = signal_data.reshape(1, *signal_data.shape)
            if signal_data.size == 0: 
                logger.warning(f"Signal data for {file_name} is empty. Skipping.")
                continue

            f_freqs, pxx_segments = eeg.get_spectral_density(signal_data, cfg)
            if f_freqs.size == 0 or pxx_segments.size == 0: 
                logger.warning(f"PSD for {file_name} is empty. Skipping.")
                continue

            param_band = DEFAULT_WIDE_BAND_FOR_PARAMS
            if 'filtering' in cfg and isinstance(cfg['filtering'], list):
                 for filt_item in cfg['filtering']:
                    if isinstance(filt_item, dict) and filt_item.get('type') == 'BandPass' and \
                       'band' in filt_item and isinstance(filt_item['band'], list) and len(filt_item['band']) == 2:
                        param_band = filt_item['band']; break
            
            for seg_idx in range(pxx_segments.shape[0]):
                segment_features = extract_features_for_segment(pxx_segments[seg_idx, :], f_freqs, param_band)
                segment_data = {
                    'file_name': file_name, 
                    'segment_idx': seg_idx,
                    'binary_target': binary_target_value, # Can be 0, 1, or None
                    'multiclass_target': multiclass_label_file
                }
                segment_data.update(segment_features)
                segment_data_list.append(segment_data)
        except Exception as e:
            logger.error(f"Error processing file {file_name} for feature extraction: {e}", exc_info=False)
    return segment_data_list

def main() -> None:
    logger.info(f"Starting stratified segment-level modeling from: {DATA_FOLDER_PATH}")

    try:
        all_mat_contents, all_file_names = eeg.load_files_from_folder(DATA_FOLDER_PATH)
        mat_files_map = {name: content for name, content in zip(all_file_names, all_mat_contents)}
    except Exception as e:
        logger.error(f"Failed to load files: {e}"); return
    if not all_file_names:
        logger.error("No files found. Exiting."); return

    files_by_category: Dict[str, List[str]] = defaultdict(list)
    for file_name in all_file_names:
        label = get_multiclass_label(file_name)
        if label: files_by_category[label].append(file_name)
        else: logger.warning(f"Could not determine category for file: {file_name}, it will be excluded.")

    train_files: List[str] = []
    test_files: List[str] = []
    logger.info("\n--- File Stratification for Train/Test Split ---")
    for category, cat_files in files_by_category.items():
        cat_files.sort()
        if len(cat_files) < 2:
            logger.warning(f"Category '{category}' has {len(cat_files)} file(s). Needs at least 2 for 1-train/1-test split. Skipping this category for stratified split.")
            continue
        train_files.append(cat_files[0])
        test_files.append(cat_files[1])
        logger.info(f"Category '{category}': Assigned '{cat_files[0]}' to TRAIN, '{cat_files[1]}' to TEST.")
        if len(cat_files) > 2:
            logger.warning(f"Category '{category}' has {len(cat_files)} files. Files {cat_files[2:]} are currently not used in this 1-train/1-test stratified split.")
    
    train_files = [f for f in train_files if f in mat_files_map] # Ensure files exist
    test_files = [f for f in test_files if f in mat_files_map]

    if not train_files or not test_files:
        logger.error("Not enough files for a stratified train/test split after validation. Exiting.")
        return

    logger.info(f"Final training files ({len(train_files)}): {train_files}")
    logger.info(f"Final testing files ({len(test_files)}): {test_files}")

    train_segment_data_all = load_and_extract_features_from_files(train_files, mat_files_map)
    test_segment_data_all = load_and_extract_features_from_files(test_files, mat_files_map)

    if not train_segment_data_all or not test_segment_data_all:
        logger.error("No segments extracted for training or testing after feature extraction. Exiting."); return

    train_df_all = pd.DataFrame(train_segment_data_all)
    test_df_all = pd.DataFrame(test_segment_data_all)

    logger.info(f"Total training segments loaded: {len(train_df_all)}")
    logger.info(f"Total testing segments loaded: {len(test_df_all)}")

    # --- Binary Classification ---
    logger.info("\n--- Stratified Binary Classification (Positive/Negative) ---")
    train_df_bin = train_df_all[train_df_all['binary_target'].notna()].copy()
    test_df_bin = test_df_all[test_df_all['binary_target'].notna()].copy()

    if train_df_bin.empty or test_df_bin.empty:
        logger.warning("Binary classification skipped: No data after filtering for non-None binary targets.")
    elif train_df_bin['binary_target'].nunique() < 2 or test_df_bin['binary_target'].nunique() < 2:
        logger.warning("Binary classification skipped: At least one set (train/test) has only one class for binary target.")
        logger.info(f"Train binary unique values: {train_df_bin['binary_target'].unique()}")
        logger.info(f"Test binary unique values: {test_df_bin['binary_target'].unique()}")
    else:
        X_train_bin_raw = train_df_bin[ALL_FEATURE_NAMES]
        y_train_binary = train_df_bin['binary_target'].astype(int)
        X_test_bin_raw = test_df_bin[ALL_FEATURE_NAMES]
        y_test_binary = test_df_bin['binary_target'].astype(int)

        imputer_b = SimpleImputer(strategy='mean')
        X_train_bin_imputed = imputer_b.fit_transform(X_train_bin_raw)
        X_test_bin_imputed = imputer_b.transform(X_test_bin_raw)

        scaler_b = StandardScaler()
        X_train_bin_scaled = scaler_b.fit_transform(X_train_bin_imputed)
        X_test_bin_scaled = scaler_b.transform(X_test_bin_imputed)

        X_train_b_df = pd.DataFrame(X_train_bin_scaled, columns=ALL_FEATURE_NAMES)
        X_test_b_df = pd.DataFrame(X_test_bin_scaled, columns=ALL_FEATURE_NAMES) # type: ignore

        logger.info(f"Train binary segments: {len(X_train_b_df)}, Test binary segments: {len(X_test_b_df)}")
        logger.info(f"Train binary class distribution:\n{y_train_binary.value_counts(normalize=True)}")
        logger.info(f"Test binary class distribution:\n{y_test_binary.value_counts(normalize=True)}")
        
        model_b = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model_b.fit(X_train_b_df, y_train_binary)
        y_pred_b = model_b.predict(X_test_b_df)
        logger.info(f"Accuracy: {accuracy_score(y_test_binary, y_pred_b):.4f}")
        logger.info("Classification Report:\n" + str(classification_report(y_test_binary, y_pred_b, zero_division=0)))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test_binary, y_pred_b)))
        try:
            importances_b = pd.Series(model_b.feature_importances_, index=ALL_FEATURE_NAMES).sort_values(ascending=False)
            logger.info("Feature Importances (Binary):\n" + str(importances_b))
        except Exception as fe_b_ex: logger.warning(f"Could not get binary feature importances: {fe_b_ex}")

    # --- Multi-class Classification ---
    logger.info("\n--- Stratified Multi-class Classification ---")
    # Using all loaded segments for multiclass as multiclass_target should always be present if file processed
    X_train_multi_raw = train_df_all[ALL_FEATURE_NAMES]
    y_train_multiclass = train_df_all['multiclass_target']
    X_test_multi_raw = test_df_all[ALL_FEATURE_NAMES]
    y_test_multiclass = test_df_all['multiclass_target']

    if y_train_multiclass.nunique() < 2 or y_test_multiclass.nunique() < 2:
         logger.warning("Multi-class classification skipped: At least one set (train/test) has less than two unique multi-class labels.")
    elif X_train_multi_raw.empty or X_test_multi_raw.empty:
        logger.warning("Multi-class classification skipped: No data available for training or testing.")
    else:
        imputer_m = SimpleImputer(strategy='mean')
        X_train_multi_imputed = imputer_m.fit_transform(X_train_multi_raw)
        X_test_multi_imputed = imputer_m.transform(X_test_multi_raw)

        scaler_m = StandardScaler()
        X_train_multi_scaled = scaler_m.fit_transform(X_train_multi_imputed)
        X_test_multi_scaled = scaler_m.transform(X_test_multi_imputed)
        
        X_train_m_df = pd.DataFrame(X_train_multi_scaled, columns=ALL_FEATURE_NAMES)
        X_test_m_df = pd.DataFrame(X_test_multi_scaled, columns=ALL_FEATURE_NAMES) # type: ignore

        logger.info(f"Train multi-class segments: {len(X_train_m_df)}, Test multi-class segments: {len(X_test_m_df)}")
        logger.info(f"Train multi-class distribution:\n{y_train_multiclass.value_counts(normalize=True)}")
        logger.info(f"Test multi-class distribution:\n{y_test_multiclass.value_counts(normalize=True)}")
        model_m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model_m.fit(X_train_m_df, y_train_multiclass)
        y_pred_m = model_m.predict(X_test_m_df)
        
        all_possible_labels = sorted(list(set(y_train_multiclass.unique()) | set(y_test_multiclass.unique())))
        logger.info(f"Accuracy: {accuracy_score(y_test_multiclass, y_pred_m):.4f}")
        logger.info("Classification Report (Multi-class):\n" + str(classification_report(y_test_multiclass, y_pred_m, labels=all_possible_labels, zero_division=0)))
        logger.info("Confusion Matrix (Multi-class):\n" + str(confusion_matrix(y_test_multiclass, y_pred_m, labels=all_possible_labels)))
        try:
            importances_m = pd.Series(model_m.feature_importances_, index=ALL_FEATURE_NAMES).sort_values(ascending=False)
            logger.info("Feature Importances (Multi-class):\n" + str(importances_m))
        except Exception as fe_m_ex: logger.warning(f"Could not get multi-class feature importances: {fe_m_ex}")

    logger.info("\n--- Stratified segment-level modeling finished. ---")

if __name__ == "__main__":
    main()
