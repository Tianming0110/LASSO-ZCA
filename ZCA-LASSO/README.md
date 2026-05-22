
# Fault Diagnosis for the Tennessee Eastman Process (TEP) using ZCA-LASSO and Deep Learning

This project provides an end-to-end automated pipeline for fault diagnosis in the Tennessee Eastman Process (TEP). It incorporates a novel preprocessing framework combining **ZCA Whitening** and **LASSO Feature Selection** to enhance the stability and accuracy of various machine learning and deep learning models, including SVM, MLP, 1D-CNN, LSTM, and Transformers.

## ⚠️ Important: Dataset Download & Preparation

Before running any scripts, you must download the TEP dataset and configure the data directory:

1. Download the raw `.dat` files from the following GitHub repository:
   👉 [https://github.com/camaramm/tennessee-eastman-profBraatz](https://github.com/camaramm/tennessee-eastman-profBraatz)
2. Extract the dataset files into a directory on your local machine.
3. Open `data_loading.py`, locate the `DATA_DIR` variable, and change it to your actual local dataset path:
   ```python
   DATA_DIR = r"C:\Your\Local\Path\To\tennessee-eastman"

```

## 🛠️ Requirements & Dependencies

Make sure you have Python installed along with the following packages:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn torch

```

*(Note: If you intend to use GPU acceleration, please ensure that your PyTorch installation is compatible with your local CUDA version).*

## 📂 File Structure

* **`run_exp.py`** (Main File): The primary entry point. It automates batch experiments across different models and configurations (Baseline, ZCA-only, LASSO-only, and ZCA+LASSO) for a specified number of runs to gather reliable statistics (mean and standard deviation).
* **`train.py`**: Contains the core training pipeline logic, including data transformation (ZCA whitening matrix computation, L1-regularized LASSO feature selection), the PyTorch training loop, evaluation metrics, and confusion matrix generation.
* **`Model.py`**: Defines the deep learning architectures built in PyTorch (`TE_MLP`, `TE_1DCNN`, `TE_LSTM`, and `TE_Transformer`).
* **`data_loading.py`**: Handles loading the raw TEP data matrices, automatically adjusting labels for the "fault injection delay" mechanism in the test set (first 160 samples reset to 0), and filtering out undiagnosable faults (Faults 3, 9, 15, 21).
* **Visualization Scripts**:
* `plot_lasso_slection.py`: Computes and plots the feature selection frequency/stability across multiple runs.
* `plot_comparison.py`: Generates the overall diagnostic performance bar charts with error bars comparing the Baseline vs. Proposed ZCA-LASSO.



## 🚀 How to Run

### Step 1: Run the Core Experiments

After setting up your local dataset directory, launch the automated grid-search experiments by running:

```bash
python run_exp.py

```

*By default, this will run batch experiments for the specified models and configurations. All logs, trained model weights, configuration statistics, and classification summaries will be automatically exported to a newly generated `results/` folder.*

### Step 2: Generate Visualizations

Once the experimental data is collected in the `results/` directory, you can run the visualization scripts to reproduce figures for papers or reports:

1. **Feature Selection Stability Analysis**:
```bash
python plot_lasso_slection.py

```


2. **Performance Comparison Charts**:
```bash
python plot_comparison.py

```


3. **Data Space Transformation Demo**:
```bash
python plot_architecture.py

```



## 📊 Expected Outputs & Results

The code automatically tracks results and saves them in the `results/` directory:

* **`Experiment_Summary_Table.csv`**: A structured table containing the mean accuracy and standard deviation for each model and preprocessing combination (ideal for direct inclusion in academic tables).
* **`Experiment_All_Runs_Detail.csv`**: Granular tracking records for every individual seed and iteration.
* **`LASSO_features_*.csv`**: Log files capturing exactly which TEP process sensors (XMEAS and XMV variables) were retained during each automated run.
* **`.png` Plots**: Automatically generated confusion matrices and summary comparison charts.

```

```