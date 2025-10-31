#!/usr/bin/env python3
"""
Harmonize H5 file using control-based normalization per database.
Applies the same harmonization logic as shown in norm2.ipynb.
"""

import h5py      # type: ignore
import pandas as pd      # type: ignore
import numpy as np
import re
import argparse

def load_h5_to_dataframe(h5_path: str) -> pd.DataFrame:
    """Load H5 file to DataFrame, same as in modellling.ipynb"""
    spectral_features = [
        "individual_alpha_frequency",
        "median_frequency",
        "relative_powers",
        "renyi_entropy",
        "shannon_entropy",
        "spectral_bandwidth",
        "spectral_centroid",
        "spectral_crest_factor",
        "spectral_edge_frequency_95",
        "transition_frequency",
        "tsallis_entropy",
    ]

    relative_power_band_names = [
        "Delta (0.5-4 Hz)",
        "Theta (4-8 Hz)",
        "Alpha (8-13 Hz)",
        "Beta1 (13-19 Hz)",
        "Beta2 (19-30 Hz)",
        "Gamma (30-70 Hz)"
    ]

    all_subjects_data = []

    with h5py.File(h5_path, "r") as f:
        subjects_group = f["subjects"]
        for subj_key in subjects_group.keys():
            subj_group = subjects_group[subj_key]
            sp = subj_group["spectral"]["spectral_parameters"]

            data_dict = {}
            n_segments: int = 0
            for feat in spectral_features:
                data = sp[feat][()]
                if feat == "relative_powers" and data.ndim == 2:
                    if n_segments == 0:
                        n_segments = data.shape[0]
                    for j, band_name in enumerate(relative_power_band_names):
                        col_name = f"relative_powers_{band_name}"
                        data_dict[col_name] = data[:, j]
                elif data.ndim == 1:
                    if n_segments == 0:
                        n_segments = data.shape[0]
                    data_dict[feat] = data
                elif data.ndim == 2:
                    if n_segments == 0:
                        n_segments = data.shape[0]
                    for j in range(data.shape[1]):
                        col_name = f"{feat}_{j}"
                        data_dict[col_name] = data[:, j]

            # Get category from H5 attributes (more reliable)
            category = subj_group.attrs.get('category', '')
            
            # Get origin from folder_id attribute if available
            folder_id = subj_group.attrs.get('folder_id', '')
            is_eeg = subj_group.attrs.get('is_eeg', True)
            
            # Extract ID from subject key for backwards compatibility
            match = re.match(r"([^_]+)_([^\.]+)\.mat", subj_key)
            subj_id = match.group(2) if match else ""

            subj_df = pd.DataFrame(data_dict)
            subj_df.insert(0, "Segment", range(n_segments))
            subj_df.insert(0, "Subject", subj_key)
            subj_df.insert(1, "category", category)
            subj_df.insert(2, "ID", subj_id)
            subj_df.insert(3, "is_eeg", is_eeg)

            # Use folder_id as origin (e.g., "POCTEP", "HURH", "DK")
            # This properly distinguishes MEG databases from EEG databases
            origin_val = folder_id if folder_id else ("POCTEP" if subj_id.endswith(("ES", "PT")) else "HURH")
            subj_df["origin"] = origin_val

            all_subjects_data.append(subj_df)

    POSITIVE_CATEGORIES = ['ADMIL', 'ADMOD']
    NEGATIVE_CATEGORIES = ['HC']

    df = pd.concat(all_subjects_data, ignore_index=True)
    df['category_binary'] = df['category'].apply(
        lambda c: 'POSITIVE' if c in POSITIVE_CATEGORIES else ('NEGATIVE' if c in NEGATIVE_CATEGORIES else 'OTHER')
    )

    return df

def apply_harmonization(df: pd.DataFrame) -> pd.DataFrame:
    """Apply harmonization exactly as in norm2.ipynb"""
    # Create a copy to avoid changing original dataframe
    df_harmonized = df.copy()

    # Feature columns (exclude label columns)
    feature_cols = [col for col in df_harmonized.columns
                    if col not in ["category", "category_binary", "ID", "origin", "Subject", "Segment", "is_eeg"]]

    # First, harmonize all MEG subjects together (is_eeg=False)
    mask_meg = ~df_harmonized["is_eeg"]
    if mask_meg.any():
        mask_controls_meg = mask_meg & (df_harmonized["category_binary"] == "NEGATIVE")
        controls_meg = df_harmonized.loc[mask_controls_meg, feature_cols]

        if not controls_meg.empty:
            mean_ctrl = controls_meg.mean()
            std_ctrl = controls_meg.std()
            
            # Handle features with zero standard deviation (constant values)
            # Replace std=0 with 1 to avoid division by zero (these features won't be normalized)
            std_ctrl = std_ctrl.replace(0, 1)

            normalized_vals = (
                df_harmonized.loc[mask_meg, feature_cols].astype(float) - mean_ctrl
            ) / std_ctrl

            df_harmonized.loc[mask_meg, feature_cols] = normalized_vals
        else:
            print("WARNING: No control subjects found for MEG data. Skipping harmonization for MEG.")

    # Then, harmonize each EEG database separately
    mask_eeg = df_harmonized["is_eeg"]
    for db in df_harmonized.loc[mask_eeg, "origin"].unique():
        mask_db = (df_harmonized["origin"] == db) & mask_eeg
        mask_controls = mask_db & (df_harmonized["category_binary"] == "NEGATIVE")
        controls = df_harmonized.loc[mask_controls, feature_cols]

        if controls.empty:
            print(f"WARNING: No control subjects found for origin '{db}'. Skipping harmonization for this database.")
            continue

        mean_ctrl = controls.mean()
        std_ctrl = controls.std()
        
        # Handle features with zero standard deviation (constant values)
        # Replace std=0 with 1 to avoid division by zero (these features won't be normalized)
        std_ctrl = std_ctrl.replace(0, 1)

        normalized_vals = (
            df_harmonized.loc[mask_db, feature_cols].astype(float) - mean_ctrl
        ) / std_ctrl

        df_harmonized.loc[mask_db, feature_cols] = normalized_vals

    return df_harmonized

def dataframe_to_h5(df: pd.DataFrame, output_path: str, original_h5_path: str):
    """Convert harmonized DataFrame back to H5 format identical to original"""

    # Group features back by subject
    subject_groups = df.groupby("Subject")

    with h5py.File(original_h5_path, "r") as orig_f, h5py.File(output_path, "w") as out_f:
        # Copy root-level attributes
        for attr_name, attr_value in orig_f.attrs.items():
            out_f.attrs[attr_name] = attr_value
        
        # Copy the root structure
        subjects_group = out_f.create_group("subjects")

        for subj_key in orig_f["subjects"].keys():
            orig_subj = orig_f["subjects"][subj_key]
            out_subj = subjects_group.create_group(subj_key)

            # Copy attributes
            for attr_name, attr_value in orig_subj.attrs.items():
                out_subj.attrs[attr_name] = attr_value

            # Copy raw_segments if it exists
            if "raw_segments" in orig_subj:
                out_subj.copy(orig_subj["raw_segments"], "raw_segments")

            # Copy spectral group structure
            spectral_group = out_subj.create_group("spectral")
            spectral_params_group = spectral_group.create_group("spectral_parameters")

            # Get harmonized data for this subject
            subj_data = subject_groups.get_group(subj_key) if subj_key in subject_groups.groups else None
            if subj_data is None:
                continue

            # Copy spectral parameters with harmonized data
            sp_orig = orig_subj["spectral"]["spectral_parameters"]

            # Handle each feature
            spectral_features = [
                "individual_alpha_frequency", "median_frequency", "renyi_entropy",
                "shannon_entropy", "spectral_bandwidth", "spectral_centroid",
                "spectral_crest_factor", "spectral_edge_frequency_95",
                "transition_frequency", "tsallis_entropy"
            ]

            for feat in spectral_features:
                if feat in sp_orig:
                    if feat in subj_data.columns:
                        # Use harmonized data
                        data = subj_data[feat].values.astype(np.float32)
                        spectral_params_group.create_dataset(feat, data=data)
                    else:
                        # Copy original if not in harmonized data
                        spectral_params_group.copy(sp_orig[feat], spectral_params_group)

            # Handle relative_powers specially (6 bands)
            if "relative_powers" in sp_orig:
                relative_power_cols = [
                    "relative_powers_Delta (0.5-4 Hz)",
                    "relative_powers_Theta (4-8 Hz)",
                    "relative_powers_Alpha (8-13 Hz)",
                    "relative_powers_Beta1 (13-19 Hz)",
                    "relative_powers_Beta2 (19-30 Hz)",
                    "relative_powers_Gamma (30-70 Hz)"
                ]

                if all(col in subj_data.columns for col in relative_power_cols):
                    # Use harmonized data
                    rp_data = np.column_stack([subj_data[col].values for col in relative_power_cols]).astype(np.float32)
                    spectral_params_group.create_dataset("relative_powers", data=rp_data)
                else:
                    # Copy original
                    spectral_params_group.copy(sp_orig["relative_powers"], spectral_params_group)

            # Copy any other datasets/groups we might have missed
            for key in sp_orig.keys():
                if key not in spectral_params_group:
                    spectral_params_group.copy(sp_orig[key], spectral_params_group)

def main():
    parser = argparse.ArgumentParser(description="Harmonize H5 file using control-based normalization")
    parser.add_argument("input_h5", help="Input H5 file path")
    parser.add_argument("output_h5", help="Output H5 file path")
    args = parser.parse_args()

    print(f"Loading {args.input_h5}...")
    df = load_h5_to_dataframe(args.input_h5)
    print(f"Loaded DataFrame with shape: {df.shape}")

    print("Applying harmonization...")
    df_harmonized = apply_harmonization(df)
    print("Harmonization complete")

    print(f"Writing harmonized data to {args.output_h5}...")
    dataframe_to_h5(df_harmonized, args.output_h5, args.input_h5)
    print("Done!")

if __name__ == "__main__":
    main()
