"""
train.py - Pipeline for ZCA-LASSO Preprocessing & Deep Learning Training

Author: Worromot
Date: 2026/5/8
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

# Import previously written modules
from data_loading import load_te_raw_dat_files, DATA_DIR
from Model import TE_MLP, TE_1DCNN, TE_LSTM, TE_Transformer

# ================= Create result saving directory =================
RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)
# ====================================================

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def apply_zca_whitening(x_train, x_test):
    scaler = StandardScaler()
    x_train_s = scaler.fit_transform(x_train)
    x_test_s = scaler.transform(x_test)

    N = x_train_s.shape[0]
    cov_matrix = np.dot(x_train_s.T, x_train_s) / (N - 1)
    U, S, V = np.linalg.svd(cov_matrix)
    epsilon = 1e-5
    W_zca = np.dot(U, np.dot(np.diag(1.0 / np.sqrt(S + epsilon)), U.T))

    x_train_zca = np.dot(x_train_s, W_zca)
    x_test_zca = np.dot(x_test_s, W_zca)
    return x_train_zca, x_test_zca


def apply_lasso_selection(x_train, y_train, x_test, penalty_C=0.01, use_zca=False, run_id=1, seed=42):
    l1_model = LogisticRegression(penalty='l1', solver='saga', C=penalty_C, max_iter=2000, n_jobs=-1, random_state=seed)
    selector = SelectFromModel(l1_model)
    selector.fit(x_train, y_train)

    x_train_selected = selector.transform(x_train)
    x_test_selected = selector.transform(x_test)
    selected_indices = selector.get_support(indices=True)

    # ================= New: Automatically save feature indices locally (with Run_ID tag) =================
    te_var_names = [f"XMEAS({i + 1})" for i in range(41)] + [f"XMV({i + 1})" for i in range(11)]
    selected_names = [te_var_names[i] for i in selected_indices]

    df_features = pd.DataFrame({
        'Feature_Index': selected_indices,
        'Variable_Name': selected_names
    })
    csv_filename = os.path.join(RESULT_DIR, f"LASSO_features_ZCA{use_zca}_Run{run_id}.csv")
    df_features.to_csv(csv_filename, index=False)
    # ==============================================================================

    return x_train_selected, x_test_selected, len(selected_indices)


# ================= 2. Training and Evaluation Module =================

def train_and_evaluate(model, train_loader, test_loader, epochs=100, lr=0.001, device='cpu', use_zca=False, use_lasso=False, run_id=1):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_test_acc = 0.0
    model_name = model.__class__.__name__
    save_path = os.path.join(RESULT_DIR, f"best_{model_name}_ZCA{use_zca}_LASSO{use_lasso}_Run{run_id}.pth")

    for epoch in range(epochs):
        model.train()
        train_loss, train_correct, total_train = 0.0, 0, 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        model.eval()
        test_loss, test_correct, total_test = 0.0, 0, 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                test_loss += criterion(outputs, batch_y).item() * batch_x.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_test += batch_y.size(0)
                test_correct += (predicted == batch_y).sum().item()
        test_acc = 100 * test_correct / total_test

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(model.state_dict(), save_path)

    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.numpy())

    generate_evaluation_reports(all_targets, all_preds, model_name, best_test_acc, use_zca, use_lasso, run_id)
    return best_test_acc


def generate_evaluation_reports(all_targets, all_preds, model_name, acc, use_zca, use_lasso, run_id):
    num_classes = len(np.unique(all_targets))
    cm = confusion_matrix(all_targets, all_preds)

    plt.figure(figsize=(16, 12))

    # Strictly keep your original plotting settings, removed xlabel, ylabel and title
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=range(num_classes), yticklabels=range(num_classes),
                cbar=False, square=True, annot_kws={'size': 18})

    fig_name = os.path.join(RESULT_DIR, f'CM_{model_name}_ZCA{use_zca}_LASSO{use_lasso}_Run{run_id}.png')
    plt.savefig(fig_name, dpi=150, bbox_inches='tight')
    plt.close()


def train_and_evaluate_svm(x_train, y_train, x_test, y_test, use_zca=False, use_lasso=False, run_id=1):
    svm_model = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
    svm_model.fit(x_train, y_train)
    all_preds = svm_model.predict(x_test)
    all_targets = y_test

    test_acc = 100 * np.sum(all_preds == all_targets) / len(all_targets)
    generate_evaluation_reports(all_targets, all_preds, "SVM", test_acc, use_zca, use_lasso, run_id)
    return test_acc


# ================= 3. Encapsulate into an externally callable Pipeline function =================
def run_pipeline(model_type="MLP", use_zca=False, use_lasso=False, lasso_c=0.01, batch_size=256, epochs=100, run_id=1, seed=42):
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x_train, y_train, x_test, y_test = load_te_raw_dat_files(DATA_DIR)

    unique_labels = sorted(np.unique(y_train))
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
    y_train_mapped = np.copy(y_train)
    y_test_mapped = np.copy(y_test)
    for old_label, new_label in label_map.items():
        y_train_mapped[y_train == old_label] = new_label
        y_test_mapped[y_test == old_label] = new_label
    y_train, y_test = y_train_mapped, y_test_mapped

    x_train, x_test = x_train.astype(np.float32), x_test.astype(np.float32)
    y_train, y_test = y_train.astype(np.int64), y_test.astype(np.int64)

    current_input_dim = x_train.shape[1]
    num_actual_classes = len(np.unique(y_train))

    if use_zca:
        x_train, x_test = apply_zca_whitening(x_train, x_test)
    else:
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x_train)
        x_test = scaler.transform(x_test)

    if use_lasso:
        x_train, x_test, current_input_dim = apply_lasso_selection(x_train, y_train, x_test, penalty_C=lasso_c, use_zca=use_zca, run_id=run_id, seed=seed)

    if model_type == "SVM":
        final_acc = train_and_evaluate_svm(x_train, y_train, x_test, y_test, use_zca, use_lasso, run_id)
    else:
        train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train))
        test_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test))

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if model_type == "MLP":
            model = TE_MLP(input_dim=current_input_dim, num_classes=num_actual_classes)
        elif model_type == "CNN":
            model = TE_1DCNN(input_dim=current_input_dim, num_classes=num_actual_classes)
        elif model_type == "LSTM":
            model = TE_LSTM(input_dim=current_input_dim, num_classes=num_actual_classes)
        elif model_type == "Transformer":
            model = TE_Transformer(input_dim=current_input_dim, num_classes=num_actual_classes)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        final_acc = train_and_evaluate(model, train_loader, test_loader, epochs=epochs, lr=0.001, device=device, use_zca=use_zca, use_lasso=use_lasso, run_id=run_id)

    return final_acc

if __name__ == "__main__":
    acc = run_pipeline("MLP", use_zca=True, use_lasso=True, run_id=999, seed=42)
    print(f"Test Run Accuracy: {acc}%")