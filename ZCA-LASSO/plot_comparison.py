import numpy as np
import matplotlib.pyplot as plt

# ================= 1. Data Preparation =================
# Model names
models = ['SVM', 'MLP', 'CNN', 'LSTM', 'Transformer']

# Mean and standard deviation of Baseline (Raw Data)
baseline_means = [61.74, 77.14, 65.73, 72.08, 79.80]
baseline_stds = [0.00, 0.52, 1.62, 1.05, 0.54]

# Mean and standard deviation after introducing ZCA-LASSO
zca_lasso_means = [71.66, 84.09, 77.26, 80.75, 85.88]
zca_lasso_stds = [0.00, 0.84, 0.21, 0.95, 0.45]

# ================= 2. Image Style Settings =================
# Set the width and X-axis position of the bars
x = np.arange(len(models))
width = 0.35

# Create canvas
fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

# ================= 3. Draw Bar Chart with Error Bars =================
# Draw baseline bars (blue series, capsize controls the width of the horizontal lines at the ends of the error bars)
rects1 = ax.bar(x - width/2, baseline_means, width, yerr=baseline_stds,
                label='Baseline (Raw Data)', capsize=6, color='#22467B',
                edgecolor='#22467B', linewidth=1, alpha=1, error_kw={'elinewidth':1.5})

# Draw ZCA-LASSO bars (orange series)
rects2 = ax.bar(x + width/2, zca_lasso_means, width, yerr=zca_lasso_stds,
                label='Proposed ZCA-LASSO', capsize=6, color='#C29B40',
                edgecolor='#C29B40', linewidth=1, alpha=1, error_kw={'elinewidth':1.5})

# ================= 4. Add Labels and Decorations =================
ax.set_ylabel('Mean Accuracy (%)', fontsize=16, fontweight='bold')
# ax.set_title('Overall Fault Diagnosis Performance Comparison', fontsize=16, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=12, fontweight='bold')
ax.set_ylim(50, 95)  # Set the Y-axis range to leave space for text

# Add grid lines to make the chart look more academic
# ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.set_axisbelow(True) # Place grid lines behind the bars

# Set legend
ax.legend(fontsize=16, loc='upper left', frameon=True, edgecolor='black')

# ================= 5. Automatically Add Accuracy Values Above Bars =================
def autolabel(rects):
    """Add numerical labels rounded to two decimal places above each bar"""
    for rect in rects:
        height = rect.get_height()
        if height == 65.73:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 13),  # Offset vertically upward by 6 pixels to avoid error bars
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        elif height == 72.08 or height == 80.75:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 8),  # Offset vertically upward by 6 pixels to avoid error bars
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')
        else:
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 6),  # Offset vertically upward by 6 pixels to avoid error bars
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

# ================= 6. Save and Display =================
plt.tight_layout()
plt.savefig('Overall_Performance_Comparison.png', dpi=300, bbox_inches='tight')
print("✅ Chart successfully generated and saved as Overall_Performance_Comparison.png")
# plt.show()
# a = 1