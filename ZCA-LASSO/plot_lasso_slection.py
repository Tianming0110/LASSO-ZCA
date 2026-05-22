import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ================= 1. configuration =================
DATA_DIR = r'.\ZCA-LASSO\results'
NUM_RUNS = 20
NUM_FEATURES = 52

# colors definition
COLOR_LASSO = '#22467B'  #
COLOR_ZCA_LASSO = '#C29B40'

# Initialize frequency statistical array
counts_lasso = np.zeros(NUM_FEATURES)
counts_zca_lasso = np.zeros(NUM_FEATURES)

# Pre-generate 52 standard feature names of TEP data
feature_names = [f"XMEAS({i + 1})" for i in range(41)] + [f"XMV({i + 1})" for i in range(11)]

# ================= 2. Data Reading and Statistics =================
print("Reading CSV data and counting feature selection frequency...")

for run in range(1, NUM_RUNS + 1):
    file_lasso = os.path.join(DATA_DIR, f"LASSO_features_ZCAFalse_Run{run}.csv")
    file_zca_lasso = os.path.join(DATA_DIR, f"LASSO_features_ZCATrue_Run{run}.csv")

    # Counting frequency using only LASSO
    if os.path.exists(file_lasso):
        # Read with pd.read_csv
        df_lasso = pd.read_csv(file_lasso)
        counts_lasso[df_lasso['Feature_Index'].values] += 1
    else:
        print(f"Warning: Can not find file {file_lasso}")

    # Counting frequency using ZCA-LASSO
    if os.path.exists(file_zca_lasso):
        # Read with pd.read_csv
        df_zca_lasso = pd.read_csv(file_zca_lasso)
        counts_zca_lasso[df_zca_lasso['Feature_Index'].values] += 1
    else:
        print(f"Warning: Can not find file {file_zca_lasso}")

print("Data statistics completed, ready to plot!")

# ================= 3. Draw two separate charts respectively =================
x_pos = np.arange(NUM_FEATURES)

# ---------------- Fig 1：Only use LASSO ----------------
plt.figure(figsize=(16, 5), dpi=500)
plt.bar(x_pos, counts_lasso, color=COLOR_LASSO, edgecolor=COLOR_LASSO, linewidth=0.8, alpha=1)

# plt.title('Feature Selection Stability over 20 Runs: LASSO Only', fontsize=16, fontweight='bold', pad=10)
plt.ylabel('Selection Frequency', fontsize=14, fontweight='bold')
plt.xlabel('TE Process Variables (Sensors)', fontsize=14, fontweight='bold')

plt.ylim(0, 22)
plt.yticks([0, 5, 10, 15, 20])
plt.xticks(x_pos, feature_names, rotation=90, fontsize=10)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.gca().set_axisbelow(True)
# plt.axhline(y=20, color='red', linestyle='-', linewidth=1.5, alpha=0.5)

output_lasso = 'Feature_Selection_Stability_LASSO_Only.png'
plt.savefig(output_lasso, dpi=500, bbox_inches='tight')
print(f"✅ 图 1 已保存: {output_lasso}")
plt.close()

# ---------------- Fig 2：use ZCA-LASSO ----------------
plt.figure(figsize=(16, 5), dpi=500)
plt.bar(x_pos, counts_zca_lasso, color=COLOR_ZCA_LASSO, edgecolor=COLOR_ZCA_LASSO, linewidth=0.8, alpha=1)

# plt.title('Feature Selection Stability over 20 Runs: Proposed ZCA-LASSO', fontsize=16, fontweight='bold', pad=10)
plt.ylabel('Selection Frequency', fontsize=14, fontweight='bold')
plt.xlabel('TE Process Variables (Sensors)', fontsize=14, fontweight='bold')

plt.ylim(0, 22)
plt.yticks([0, 5, 10, 15, 20])
plt.xticks(x_pos, feature_names, rotation=90, fontsize=10)

# plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.gca().set_axisbelow(True)
# plt.axhline(y=20, color='red', linestyle='-', linewidth=1.5, alpha=0.5)

# 保存图 2
output_zca_lasso = 'Feature_Selection_Stability_ZCA_LASSO.png'
plt.savefig(output_zca_lasso, dpi=500, bbox_inches='tight')
print(f"✅ 图 2 saved: {output_zca_lasso}")
plt.close()