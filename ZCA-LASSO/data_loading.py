"""
data_loading - TE Dataset Loader (Handling Test Set Fault Injection Delay)

Author: Worromot
Date: 2026/5/7
"""

import os
import glob
import pandas as pd
import numpy as np

DATA_DIR = r"C:\Users\TianmingXie\Desktop\paper3\tennessee-eastman"
EXCLUDE_FAULTS = [3, 9, 15, 21] #


def load_te_raw_dat_files(data_dir):
    """
    Read the original dXX.dat (train set) and dXX_te.dat (test set) files
    d00 represents normal (label 0)
    d01-d21 represents faults 1 to 21 (labels 1-21)
    *Special case handling: the first 160 samples of the test set (d01_te~d21_te) are normal data, the labels need to be reset to 0*
    """

    x_train_list, y_train_list = [], []
    x_test_list, y_test_list = [], []

    file_pattern = os.path.join(data_dir, "*.dat")
    files = glob.glob(file_pattern)

    if not files:
        print(f"No .dat files found in {data_dir}, please check the path!")
        return None, None, None, None

    for file_path in files:
        file_name = os.path.basename(file_path)

        # 1. Parse file name
        name_without_ext = file_name.split('.')[0]
        is_test = "_te" in name_without_ext

        try:
            if is_test:
                label_str = name_without_ext.split('_')[0][1:]
            else:
                label_str = name_without_ext[1:]
            label = int(label_str)
        except ValueError:
            print(f"Skipping file with unparsable label: {file_name}")
            continue

        # ================== Core modification 1: Skip undiagnosable faults ==================
        if label in EXCLUDE_FAULTS:
            print(f"⏩ [Filter] Excluding undiagnosable fault file: {file_name} (original label: {label})")
            continue
        # ====================================================================

        # 2. Read data matrix
        data_matrix = pd.read_csv(file_path, header=None, sep='\s+').values

        if data_matrix.shape[0] == 52:
            data_matrix = data_matrix.T

        n_samples = data_matrix.shape[0]

        # ================== Core modification ==================
        # Initialize labels for all samples in this file
        current_y = np.full(n_samples, label)

        # Handle the "fault injection delay" mechanism of the TE test set
        if is_test and label != 0:
            # Only the first 160 points of the fault test set are normal data
            # Ensure the number of samples is greater than 160 before slicing to prevent out-of-bounds
            safe_idx = min(160, n_samples)
            current_y[:safe_idx] = 0
        # ===============================================

        # 3. Distribute to train set or test set
        if is_test:
            x_test_list.append(data_matrix)
            y_test_list.append(current_y)
        else:
            x_train_list.append(data_matrix)
            y_train_list.append(current_y)

        print(f"Loaded {'[Test set]' if is_test else '[Train set]'} {file_name}, Main label: {label}, Number of samples: {n_samples}")

    # 4. Concatenate
    x_train = np.vstack(x_train_list) if x_train_list else np.array([])
    y_train = np.concatenate(y_train_list) if y_train_list else np.array([])

    x_test = np.vstack(x_test_list) if x_test_list else np.array([])
    y_test = np.concatenate(y_test_list) if y_test_list else np.array([])

    print("-" * 40)
    print(f"✅ Dataset construction completed (filtered faults {EXCLUDE_FAULTS})")
    print(f"✅ Dataset construction completed (processed normal label switching for the first 160 samples of the test set)!")
    print(f"Train set -> x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"Test set -> x_test:  {x_test.shape}, y_test:  {y_test.shape}")
    x_train = x_train.astype(float)
    x_test = x_test.astype(float)
    return x_train, y_train, x_test, y_test


if __name__ == "__main__":
    x_train, y_train, x_test, y_test = load_te_raw_dat_files(DATA_DIR)