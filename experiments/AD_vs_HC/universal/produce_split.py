#!/usr/bin/env python3
"""
Script to create stratified test vs cross-validation splits for universal experiments.
Filters subjects by folder_id and category, then creates stratified splits.
Appends results to a JSON file for easy dataset switching.
"""
# pyright: reportMissingTypeStubs=false

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import h5py  # type: ignore[import]
import numpy as np
from dotenv import load_dotenv
from sklearn.model_selection import train_test_split  # type: ignore[import]

load_dotenv()

# ============================================================================
# CONFIGURATION - Edit these for each dataset
# ============================================================================

H5_FILE_PATH = os.getenv("H5_FILE_PATH", "artifacts/combined_DK_features_only:v0/combined_DK_features_only.h5")
DATASET_NAME = "eeg"  # Unique name for this dataset configuration
FOLDER_IDS = ["HURH", "POCTEP"]  # List of folder_ids to include
CATEGORIES = ["HC", "ADMIL", "ADMOD"]  # List of categories to include
TEST_RATIO = 0.15  # Proportion for test set (rest goes to CV)
RANDOM_STATE = int(os.getenv("RANDOM_SEED", "42"))
OUTPUT_JSON = "universal_splits.json"  # Path relative to this script's directory

# ============================================================================


def load_and_filter_subjects(
    h5_path: str,
    folder_ids: List[str],
    categories: List[str],
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Load subject IDs filtered by folder_id and category from the H5 dataset.

    Args:
        h5_path: Path to the H5 file
        folder_ids: List of folder_ids to include
        categories: List of categories to include

    Returns:
        Tuple of (subjects_by_category, category_counts)
    """
    subjects_by_category: Dict[str, List[str]] = defaultdict(list)
    category_counts: Dict[str, int] = defaultdict(int)

    folder_id_set = set(folder_ids)
    category_set = set(categories)

    with h5py.File(h5_path, "r") as f:
        if "subjects" not in f:
            raise ValueError("No 'subjects' group found in H5 file")

        subjects_group = f["subjects"]

        for subject_id in subjects_group.keys():
            subject_group = subjects_group[subject_id]

            # Check if subject has required attributes
            if "category" not in subject_group.attrs or "folder_id" not in subject_group.attrs:
                continue

            category = subject_group.attrs["category"]
            folder_id = subject_group.attrs["folder_id"]

            # Filter by folder_id and category
            if folder_id in folder_id_set and category in category_set:
                subjects_by_category[category].append(subject_id)
                category_counts[category] += 1

    return dict(subjects_by_category), dict(category_counts)


def create_stratified_test_cv_split(
    subjects_by_category: Dict[str, List[str]],
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Create stratified split for test vs cross-validation sets.

    Args:
        subjects_by_category: Dictionary mapping category to list of subject IDs
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility

    Returns:
        Tuple of (cv_subjects, test_subjects)
    """
    if not (0.0 < test_ratio < 1.0):
        raise ValueError("Test ratio must be between 0 and 1")

    # Flatten subjects and corresponding labels
    subjects: List[str] = []
    labels: List[str] = []
    for category, ids in subjects_by_category.items():
        subjects.extend(ids)
        labels.extend([category] * len(ids))

    # Ensure reproducibility
    np.random.seed(random_state)

    # Split: cv (train+val) vs test
    cv_subjects, test_subjects, cv_labels, test_labels = train_test_split(
        subjects,
        labels,
        test_size=test_ratio,
        stratify=labels,
        random_state=random_state,
        shuffle=True,
    )

    return list(cv_subjects), list(test_subjects)


def print_category_distribution(
    subjects: List[str],
    subjects_by_category: Dict[str, List[str]],
    split_name: str,
) -> None:
    """Print category distribution for a split."""
    # Build a mapping from subject to category
    subject_to_category = {}
    for category, subs in subjects_by_category.items():
        for s in subs:
            subject_to_category[s] = category

    # Count categories in the split
    categories = [subject_to_category[s] for s in subjects if s in subject_to_category]
    counter = Counter(categories)
    total = len(subjects)
    print(f"\n{split_name} category distribution:")
    for category in sorted(counter.keys()):
        count = counter[category]
        percentage = (count / total) * 100 if total > 0 else 0.0
        print(f"  {category}: {count} ({percentage:.1f}%)")


def load_existing_splits(json_path: Path) -> Dict:
    """Load existing splits from JSON file, or return empty dict if not exists."""
    if json_path.exists():
        with open(json_path, "r") as f:
            return json.load(f)
    return {}


def save_splits_to_json(
    dataset_name: str,
    cv_subjects: List[str],
    test_subjects: List[str],
    json_path: Path,
) -> None:
    """Save splits to JSON file, appending to existing data."""
    # Load existing data
    data = load_existing_splits(json_path)

    # Add or update this dataset
    data[dataset_name] = {
        "test_subjects": sorted(test_subjects),
        "cv_subjects": sorted(cv_subjects),
    }

    # Save back to file
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Saved splits for '{dataset_name}' to {json_path}")


def main():
    """Main function to create the dataset splits."""

    # Get H5 file path
    h5_path = H5_FILE_PATH
    if not h5_path:
        raise ValueError("H5_FILE_PATH environment variable not set")

    # Output JSON path (relative to this script)
    output_path = Path(__file__).parent / OUTPUT_JSON

    print("=" * 70)
    print(f"Universal Split Generator: {DATASET_NAME}")
    print("=" * 70)
    print("\nConfiguration:")
    print(f"  Dataset name: {DATASET_NAME}")
    print(f"  Folder IDs: {FOLDER_IDS}")
    print(f"  Categories: {CATEGORIES}")
    print(f"  Test ratio: {TEST_RATIO}")
    print(f"  Random state: {RANDOM_STATE}")
    print(f"  Output: {output_path}")

    print("\nLoading and filtering subjects from H5 file...")
    subjects_by_category, category_counts = load_and_filter_subjects(
        h5_path, FOLDER_IDS, CATEGORIES
    )

    if not subjects_by_category:
        print("\n⚠ No subjects found matching the filter criteria!")
        sys.exit(1)

    print("\n=== Filtered Dataset Summary ===")
    total_subjects = sum(category_counts.values())
    print(f"Total subjects: {total_subjects}")

    print("\nSubjects per category:")
    for category in sorted(category_counts.keys()):
        count = category_counts[category]
        percentage = (count / total_subjects) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

    print("\n=== Creating Stratified Split ===")
    cv_subjects, test_subjects = create_stratified_test_cv_split(
        subjects_by_category,
        test_ratio=TEST_RATIO,
        random_state=RANDOM_STATE,
    )

    # Print split summary
    print(f"\nCross-validation set: {len(cv_subjects)} subjects ({(1-TEST_RATIO)*100:.0f}%)")
    print(f"Test set: {len(test_subjects)} subjects ({TEST_RATIO*100:.0f}%)")

    # Verify no overlap
    cv_set = set(cv_subjects)
    test_set = set(test_subjects)

    assert len(cv_set & test_set) == 0, "CV and test sets overlap!"
    print("\n✓ No overlap between splits")

    # Print category distributions
    print_category_distribution(cv_subjects, subjects_by_category, "Cross-validation")
    print_category_distribution(test_subjects, subjects_by_category, "Test")

    # Save to JSON
    print("\n=== Saving to JSON ===")
    save_splits_to_json(DATASET_NAME, cv_subjects, test_subjects, output_path)

    print("\n✓ All done!")


if __name__ == "__main__":
    main()

