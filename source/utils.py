import numpy as np
import matplotlib.pyplot as plt
from config import FINAL_ORDER, COLOR_MAP, CHEMICAL_ACCURACY_CM

def transform_r_to_x(R: np.ndarray, a: float = 1.0) -> np.ndarray:
    return np.exp(-R / a)

def transform_x_to_r(X: np.ndarray, a: float = 1.0) -> np.ndarray:
    return -a * np.log(np.maximum(X, 1e-9))

def interpolation_split(X, y, R, train_frac=0.5, seed=42):
    N = len(X)
    rng = np.random.RandomState(seed)
    indices = np.arange(N)
    rng.shuffle(indices)
    n_train = int(N * train_frac)
    train_idx, test_idx = np.sort(indices[:n_train]), np.sort(indices[n_train:])
    return X[train_idx], y[train_idx], R[train_idx], X[test_idx], y[test_idx], R[test_idx]

def extrapolation_split(X, y, R, center_frac=0.5):
    N = len(X)
    n_train = int(N * center_frac)
    start = (N - n_train) // 2
    train_idx = np.arange(start, start + n_train)
    test_idx = np.setdiff1d(np.arange(N), train_idx)
    return X[train_idx], y[train_idx], R[train_idx], X[test_idx], y[test_idx], R[test_idx]

def plot_aggregated_results(summary, R_train, y_train, R_test, y_test, R_full, scenario, molecule):
    LABEL_SIZE = 18
    TICK_SIZE = 17
    LEGEND_SIZE = 16

    plt.figure(figsize=(12, 7))

    plt.scatter(R_train, y_train, s=90, color='blue', edgecolor='white', linewidth=0.8, label='Train', zorder=5)
    plt.scatter(R_test, y_test, s=120, marker='x', color='red', linewidth=2, label='Test', zorder=6)

    for name in FINAL_ORDER:
        if name not in summary:
            continue

        res = summary[name]
        y_mu = res['y_mu_mean']
        y_std = res['y_mu_std']

        plt.plot(R_full, y_mu, label=name, color=COLOR_MAP[name], linewidth=3 if 'VQE' in name else 2)
        plt.fill_between(R_full.ravel(), y_mu - y_std, y_mu + y_std, color=COLOR_MAP[name], alpha=0.10)

    x_label = "R (Å)" if molecule.lower() == 'lih' else "R (Å) [Be-H Distance]"
    plt.xlabel(x_label, fontsize=22)
    plt.ylabel("Energy (cm⁻¹)", fontsize=22)
    
    if molecule.lower() == 'beh2':
        plt.title(f"PES Prediction: {scenario} ({molecule.upper()})", fontsize=20)

    plt.xticks(fontsize=TICK_SIZE)
    plt.yticks(fontsize=TICK_SIZE)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"results/{molecule.lower()}_pes_{scenario.lower()}.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))

    methods = [k for k in FINAL_ORDER if k in summary]
    means =[summary[k]['rmse_mean'] for k in methods]
    stds = [summary[k]['rmse_std'] for k in methods]

    plt.bar(methods, means, yerr=stds, capsize=5, color=[COLOR_MAP[k] for k in methods], alpha=0.8)
    plt.axhline(CHEMICAL_ACCURACY_CM, color='red', linestyle='--', label='Chemical Accuracy')
    plt.ylabel("RMSE (cm⁻¹)", fontsize=LABEL_SIZE)
    plt.xticks(fontsize=15, rotation=20)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=LEGEND_SIZE)
    plt.tight_layout()
    plt.savefig(f"results/{molecule.lower()}_bar_{scenario.lower()}.png", dpi=300)
    plt.close()
