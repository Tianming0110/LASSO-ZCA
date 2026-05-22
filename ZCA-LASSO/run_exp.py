"""
run_experiments.py - Batch automated experiment and result statistical analysis script

Author: Worromot
Date: 2026/5/8
"""

import pandas as pd
import numpy as np
import time

# Import Pipeline function from the newly refactored train.py
from train import run_pipeline

# ================= Experimental Environment Configuration =================
# You can modify this as needed, for example, if you only want to test MLP and CNN, delete the others
MODELS = ["SVM"]    #SVM "MLP", "CNN", "LSTM", "Transformer"

# Four combinations: Baseline (no preprocessing), ZCA only, LASSO only, ZCA+LASSO
ZCA_OPTIONS = [False, True]
LASSO_OPTIONS = [False, True]

NUM_RUNS = 20  # Repeat 20 times for each combination
BASE_SEED = 42  # Base seed, each run will accumulate on this basis
EPOCHS = 100  # DL training epochs
LASSO_C = 0.01  # LASSO penalty coefficient


# ================================================

def main():
    print(f"🚀 Starting batch automated experiments...")
    print(f"Configuration: Models={MODELS}, Number of runs={NUM_RUNS}")

    # Used to record the detailed results of each run
    all_results = []
    # Used to record the final statistical indicators table (mean and variance)
    summary_stats = []

    total_combinations = len(MODELS) * len(ZCA_OPTIONS) * len(LASSO_OPTIONS)
    current_combo = 0

    start_total_time = time.time()

    # 1. Iterate through all models
    for model in MODELS:
        # 2. Iterate through ZCA on and off
        for use_zca in ZCA_OPTIONS:
            # 3. Iterate through LASSO on and off
            for use_lasso in LASSO_OPTIONS:
                current_combo += 1
                combo_name = f"{model} | ZCA:{use_zca} | LASSO:{use_lasso}"
                print(f"\n=======================================================")
                print(f"🔥 Testing combination [{current_combo}/{total_combinations}]: {combo_name}")
                print(f"=======================================================")

                acc_list = []

                # 4. Repeat the experiment 20 times
                for run_id in range(1, NUM_RUNS + 1):
                    current_seed = BASE_SEED + run_id
                    print(f"  --> Run {run_id}/{NUM_RUNS} (Random seed: {current_seed}) starting...")

                    # try:
                    # Call the encapsulated core pipeline
                    acc = run_pipeline(
                        model_type=model,
                        use_zca=use_zca,
                        use_lasso=use_lasso,
                        lasso_c=LASSO_C,
                        batch_size=256,
                        epochs=EPOCHS,
                        run_id=run_id,
                        seed=current_seed
                    )

                    acc_list.append(acc)
                    print(f"      Run {run_id} finished, Test set accuracy: {acc:.2f}%")

                    # Save the record for each run
                    all_results.append({
                        "Model": model,
                        "ZCA": use_zca,
                        "LASSO": use_lasso,
                        "Run_ID": run_id,
                        "Seed": current_seed,
                        "Accuracy": acc
                    })

                    # except Exception as e:
                    #     print(f"      ❌ Error occurred in Run {run_id}: {e}")

                # 5. Calculate the mean and standard deviation of this combination over 20 experiments
                if len(acc_list) > 0:
                    mean_acc = np.mean(acc_list)
                    std_acc = np.std(acc_list)
                    print(f"\n✅ Combination {combo_name} testing completed!")
                    print(f"   Final score: {mean_acc:.2f}% ± {std_acc:.2f}%")

                    summary_stats.append({
                        "Model": model,
                        "ZCA": use_zca,
                        "LASSO": use_lasso,
                        "Mean_Accuracy": round(mean_acc, 2),
                        "Std_Accuracy": round(std_acc, 2)
                    })

    # ================= End of experiment, save all data =================
    print("\n🎉 All experiments finished running! Exporting results...")

    # Save detailed run records
    df_all = pd.DataFrame(all_results)
    df_all.to_csv("results/Experiment_All_Runs_Detail.csv", index=False)

    # Save what your paper needs most: the summary comparison table
    df_summary = pd.DataFrame(summary_stats)
    df_summary.to_csv("results/Experiment_Summary_Table.csv", index=False)

    end_total_time = time.time()
    hours, rem = divmod(end_total_time - start_total_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"💾 Data saved to the results folder. Total time elapsed: {int(hours)} hours {int(minutes)} minutes")
    print(df_summary)


if __name__ == "__main__":
    main()