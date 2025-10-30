#!/usr/bin/env python3
"""
Script to create stratified train/validation/test splits for the combined dataset.
Specifically for ADMOD vs HC binary classification.
Creates subject-level splits to avoid data leakage between train/val/test sets.
"""
# pyright: reportMissingTypeStubs=false

import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from collections import Counter

import h5py  # type: ignore[import]

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def load_dataset_info(h5_path: str) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Load subject IDs and their categories from the H5 dataset.
    Only includes ADSEV (positives) and HC (negatives) for binary classification.

    Args:
        h5_path: Path to the H5 file

    Returns:
        Tuple of (subjects_by_category, category_counts)
    """
    subjects_by_category: Dict[str, List[str]] = defaultdict(list)
    category_counts: Dict[str, int] = defaultdict(int)

    # Only include ADMOD and HC categories
    target_categories = {'ADMOD', 'HC'}

    with h5py.File(h5_path, 'r') as f:
        if 'subjects' not in f:
            raise ValueError("No 'subjects' group found in H5 file")

        subjects_group = f['subjects']

        for subject_id in subjects_group.keys():
            subject_group = subjects_group[subject_id]
            if 'category' in subject_group.attrs:
                category = subject_group.attrs['category']
                # Only include ADMOD and HC subjects
                if category in target_categories:
                    subjects_by_category[category].append(subject_id)
                    category_counts[category] += 1

    return dict(subjects_by_category), dict(category_counts)

def create_custom_split(
    subjects_by_category: Dict[str, List[str]],
    random_state: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """
    Create custom split where:
    - Validation set: All subjects ending with '_PT.mat' or '_ES.mat'
    - Training set: All subjects that don't end with '_PT.mat' or '_ES.mat'
    - Test set: Empty (not used in this split strategy)

    Args:
        subjects_by_category: Dictionary mapping category to list of subject IDs
        random_state: Random seed for reproducibility (not used but kept for compatibility)

    Returns:
        Tuple of (train_subjects, val_subjects, test_subjects)
    """
    # Flatten subjects
    all_subjects: List[str] = []
    for category, ids in subjects_by_category.items():
        all_subjects.extend(ids)

    # Split based on suffix
    val_subjects: List[str] = []
    train_subjects: List[str] = []
    
    for subject in all_subjects:
        if subject.endswith('_PT.mat') or subject.endswith('_ES.mat'):
            train_subjects.append(subject)
        else:
            val_subjects.append(subject)
            
    
    # No test set in this configuration
    test_subjects: List[str] = []

    return train_subjects, val_subjects, test_subjects

def save_split_to_file(subject_ids: List[str], filename: str, output_dir: str = ".") -> None:
    """
    Save list of subject IDs to a text file.

    Args:
        subject_ids: List of subject IDs
        filename: Output filename
        output_dir: Output directory
    """
    output_path = Path(output_dir) / filename
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        for subject_id in sorted(subject_ids):  # Sort for consistent output
            f.write(f"{subject_id}\n")

    print(f"Saved {len(subject_ids)} subjects to {output_path}")

def main():
    """Main function to create the dataset splits."""

    # Configuration
    h5_path = "artifacts/combined_features_df2.h5:v0/combined_features_df2.h5"
    # Save splits to a sibling folder named "splits" relative to this script
    output_dir = str((Path(__file__).parent / "splits").resolve())
    random_state = 42

    print("Loading dataset information...")
    subjects_by_category, category_counts = load_dataset_info(h5_path)

    print("\n=== Dataset Summary (ADMOD vs HC) ===")
    total_subjects = sum(category_counts.values())
    print(f"Total subjects: {total_subjects}")

    print("\nSubjects per category:")
    for category in sorted(category_counts.keys()):
        count = category_counts[category]
        percentage = (count / total_subjects) * 100
        print(f"  {category}: {count} ({percentage:.1f}%)")

    print("=== Creating Custom Split ===")
    print("Validation set: All subjects ending with '_PT.mat' or '_ES.mat'")
    print("Training set: All other subjects")
    
    # Create custom split
    train_subjects, val_subjects, test_subjects = create_custom_split(
        subjects_by_category,
        random_state=random_state
    )

    # Print split summary
    print(f"\nTrain set: {len(train_subjects)} subjects")
    print(f"Validation set: {len(val_subjects)} subjects")
    print(f"Test set: {len(test_subjects)} subjects (empty in this configuration)")

    # Verify no overlap
    train_set = set(train_subjects)
    val_set = set(val_subjects)

    assert len(train_set & val_set) == 0, "Train and validation sets overlap!"

    print("\n✓ No overlap between splits")

    # Verify split criteria
    pt_es_count = sum(1 for s in val_subjects if s.endswith('_PT.mat') or s.endswith('_ES.mat'))
    non_pt_es_count = sum(1 for s in train_subjects if not (s.endswith('_PT.mat') or s.endswith('_ES.mat')))
    
    print(f"\n✓ Validation set contains {pt_es_count}/{len(val_subjects)} subjects ending with '_PT.mat' or '_ES.mat'")
    print(f"✓ Training set contains {non_pt_es_count}/{len(train_subjects)} subjects not ending with '_PT.mat' or '_ES.mat'")

    # Save splits to files
    print("=== Saving Split Files ===")
    save_split_to_file(train_subjects, "training_subjects.txt", output_dir)
    save_split_to_file(val_subjects, "validation_subjects.txt", output_dir)
    
    # Only save test file if it has subjects
    if test_subjects:
        save_split_to_file(test_subjects, "test_subjects.txt", output_dir)

    print("✓ All splits saved successfully!")



    def print_category_distribution(subjects: List[str], subjects_by_category: Dict[str, List[str]], split_name: str) -> None:
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

    print_category_distribution(train_subjects, subjects_by_category, "Train")
    print_category_distribution(val_subjects, subjects_by_category, "Validation")
    if test_subjects:
        print_category_distribution(test_subjects, subjects_by_category, "Test")


if __name__ == "__main__":
    main()
