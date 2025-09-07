import os
import numpy as np
import pandas as pd # type: ignore
from loguru import logger
from typing import List, Dict, Tuple, Optional, Any

from sklearn.model_selection import train_test_split # type: ignore
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
# Positive classes for binary classification as defined in eeg_utils or specific to this model
POSITIVE_CLASSES_BINARY = ['AD', 'MCI'] # ADMIL, ADMOD, MCI will be positive

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

Q_IAFTF: List[float] = [4.0, 15.0]
BANDA_IAFTF: List[float] = [0.5, 70.0]
DEFAULT_WIDE_BAND_FOR_PARAMS: List[float] = [0.5, 70.0]

FEATURE_NAMES_OTHER: List[str] = ["IAF", "TF", "MF", "SEF95"]
ALL_FEATURE_NAMES: List[str] = FEATURE_NAMES_RP + FEATURE_NAMES_OTHER

# Multi-class categories and their mapping from filename prefixes
MULTI_CLASS_CATEGORIES: Dict[str, str] = {
    "ADMIL": "ADMIL",
    "ADMOD": "ADMOD",
    "HC": "HC",
    "MCI": "MCI",
    "ADSEV": "ADSEV"
}

def get_multiclass_label(file_name: str) -> Optional[str]:
    """Determines the multi-class label from the file name."""
    for prefix, label in MULTI_CLASS_CATEGORIES.items():
        if file_name.upper().startswith(prefix):
            return label
    return None

def extract_features_for_segment(
    psd_segment: np.ndarray, 
    f_freqs: np.ndarray, 
    param_band: List[float]
) -> Dict[str, Optional[float]]:
    """Extracts all spectral features for a single PSD segment."""
    features = {}

    # Relative Power
    try:
        rp_values = calcular_rp(psd_segment, f_freqs, param_band, SUB_BANDAS_LIST_RP)
        if rp_values.size == len(FEATURE_NAMES_RP):
            for name, val in zip(FEATURE_NAMES_RP, rp_values):
                features[name] = val if not np.isnan(val) else None
        else:
            for name in FEATURE_NAMES_RP: features[name] = None # Fill with None if error
    except Exception:
        for name in FEATURE_NAMES_RP: features[name] = None

    # IAF & TF
    try:
        iaf, tf = calcular_iaftf(psd_segment, f_freqs, BANDA_IAFTF, Q_IAFTF)
        features["IAF"] = iaf
        features["TF"] = tf
    except Exception:
        features["IAF"] = None
        features["TF"] = None

    # MF
    try:
        mf = calcular_mf(psd_segment, f_freqs, param_band)
        features["MF"] = mf
    except Exception:
        features["MF"] = None

    # SEF95
    try:
        sef95 = calcular_sef95(psd_segment, f_freqs, param_band)
        features["SEF95"] = sef95
    except Exception:
        features["SEF95"] = None
    
    return features

def main() -> None:
    logger.info(f"Starting segment-level modeling from: {DATA_FOLDER_PATH}")

    all_segment_features_list: List[Dict[str, Any]] = []

    try:
        mat_files_contents, file_names = eeg.load_files_from_folder(DATA_FOLDER_PATH)
    except Exception as e:
        logger.error(f"Failed to load files: {e}")
        return

    if not mat_files_contents:
        logger.error("No files found. Exiting.")
        return

    for mat_content, file_name in zip(mat_files_contents, file_names):
        logger.info(f"Processing file: {file_name}")
        try:
            signal_data, cfg, is_positive_file = eeg.get_nice_data(
                raw_data=mat_content, 
                name=file_name, 
                positive_classes_binary=POSITIVE_CLASSES_BINARY, # Ensure this matches desired binary definition
                comes_from_bbdds=True
            )
            if is_positive_file is None:
                logger.error(f"File {file_name} has no target. Skipping.")
                continue
            multiclass_label_file = get_multiclass_label(file_name)

            if signal_data.ndim == 2:
                signal_data = signal_data.reshape(1, signal_data.shape[0], signal_data.shape[1])
            if signal_data.size == 0 or multiclass_label_file is None:
                logger.warning(f"Skipping {file_name} due to empty signal or no multi-class label.")
                continue

            f_freqs, pxx_segments = eeg.get_spectral_density(signal_data, cfg)
            if f_freqs.size == 0 or pxx_segments.size == 0:
                logger.warning(f"Skipping {file_name} due to empty PSD.")
                continue

            param_band = DEFAULT_WIDE_BAND_FOR_PARAMS # Determine band as in other scripts
            if 'filtering' in cfg and isinstance(cfg['filtering'], list):
                for filt_item in cfg['filtering']:
                    if isinstance(filt_item, dict) and filt_item.get('type') == 'BandPass' and \
                       'band' in filt_item and isinstance(filt_item['band'], list) and len(filt_item['band']) == 2:
                        param_band = filt_item['band']
                        break
            
            for seg_idx in range(pxx_segments.shape[0]):
                current_psd_segment = pxx_segments[seg_idx, :]
                segment_spectral_features = extract_features_for_segment(current_psd_segment, f_freqs, param_band)
                
                segment_data = {
                    'file_name': file_name,
                    'segment_idx': seg_idx,
                    'binary_target': int(is_positive_file), # Convert boolean to int (0 or 1)
                    'multiclass_target': multiclass_label_file
                }
                segment_data.update(segment_spectral_features)
                all_segment_features_list.append(segment_data)

        except Exception as e_file:
            logger.error(f"Error processing file {file_name}: {e_file}", exc_info=False) # exc_info=False for less verbose logs in loop
    
    if not all_segment_features_list:
        logger.error("No segments extracted for modeling. Exiting.")
        return

    dataset_df = pd.DataFrame(all_segment_features_list)
    logger.info(f"Total segments processed: {len(dataset_df)}")
    logger.info(f"Dataset columns: {dataset_df.columns.tolist()}")

    # Prepare data for modeling
    X = dataset_df[ALL_FEATURE_NAMES].copy()
    
    # Impute missing values (e.g., from failed feature calculations)
    imputer = SimpleImputer(strategy='mean') # or 'median'
    X_imputed = imputer.fit_transform(X)
    X_imputed_df = pd.DataFrame(X_imputed, columns=ALL_FEATURE_NAMES)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed_df)
    X_scaled_df = pd.DataFrame(X_scaled, columns=ALL_FEATURE_NAMES)

    # --- Binary Classification (Positive/Negative) ---
    logger.info("\n--- Binary Classification (Positive/Negative) ---")
    y_binary = dataset_df['binary_target']
    if y_binary.nunique() < 2:
        logger.warning("Binary classification skipped: Only one class present in binary_target.")
    else:
        logger.info(f"Binary class distribution:\n{y_binary.value_counts(normalize=True)}")
        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(X_scaled_df, y_binary, test_size=0.3, random_state=42, stratify=y_binary)
        
        model_b = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        model_b.fit(X_train_b, y_train_b)
        y_pred_b = model_b.predict(X_test_b)
        
        logger.info(f"Accuracy: {accuracy_score(y_test_b, y_pred_b):.4f}")
        logger.info("Classification Report:\n" + str(classification_report(y_test_b, y_pred_b)))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test_b, y_pred_b)))
        try:
            importances_b = pd.Series(model_b.feature_importances_, index=ALL_FEATURE_NAMES).sort_values(ascending=False)
            logger.info("Feature Importances (Binary):\n" + str(importances_b))
        except Exception as fe_b_ex:
             logger.warning(f"Could not retrieve feature importances for binary model: {fe_b_ex}")

    # --- Multi-class Classification (ADMIL, ADMOD, HC, MCI) ---
    logger.info("\n--- Multi-class Classification (ADMIL, ADMOD, HC, MCI) ---")
    y_multiclass = dataset_df['multiclass_target']
    # Convert string labels to numerical for scikit-learn if they aren't already
    # For RandomForest, it can often handle string labels directly if consistent,
    # but explicit encoding is safer.
    # We'll rely on scikit-learn's internal handling for now for simplicity.
    # If issues arise, LabelEncoder would be used here.
    
    if y_multiclass.nunique() < 2:
        logger.warning("Multi-class classification skipped: Only one class present in multiclass_target.")
    else:
        logger.info(f"Multi-class distribution:\n{y_multiclass.value_counts(normalize=True)}")
        X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(X_scaled_df, y_multiclass, test_size=0.3, random_state=42, stratify=y_multiclass)

        model_m = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced') # class_weight for imbalanced data
        model_m.fit(X_train_m, y_train_m)
        y_pred_m = model_m.predict(X_test_m)
        
        logger.info(f"Accuracy: {accuracy_score(y_test_m, y_pred_m):.4f}")
        logger.info("Classification Report:\n" + str(classification_report(y_test_m, y_pred_m, zero_division=0)))
        logger.info("Confusion Matrix:\n" + str(confusion_matrix(y_test_m, y_pred_m, labels=model_m.classes_)))
        try:
            importances_m = pd.Series(model_m.feature_importances_, index=ALL_FEATURE_NAMES).sort_values(ascending=False)
            logger.info("Feature Importances (Multi-class):\n" + str(importances_m))
        except Exception as fe_m_ex:
             logger.warning(f"Could not retrieve feature importances for multi-class model: {fe_m_ex}")

    logger.info("\n--- Segment-level modeling finished. ---")

if __name__ == "__main__":
    # Configure Loguru for cleaner output if not done globally
    # logger.remove() 
    # logger.add(lambda msg: print(msg, end=''), 
    #            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>", 
    #            level="INFO")
    main()
