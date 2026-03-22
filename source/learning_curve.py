import os
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning

from config import FINAL_ORDER, COLOR_MAP, CHEMICAL_ACCURACY_CM, CHEM_ACC_COLOR, LEARNING_CURVE_SEEDS, TRAIN_SIZES
from dataset import MoleculeActiveSpaceCache, MoleculeDataset
from models import (
    VQEProxy, VQEParamRegressor, DNNVQEKernelGPR, 
    ClassicalRBFKernel, QuantumKernelFixedAnsatz, QuantumKernelAnsatz,
    GPZeroNoise, GPRWithNoise, RandomSearch, CompositionSearch
)

warnings.filterwarnings("ignore", category=ConvergenceWarning)

def run_learning_curve(molecule, X, y, R, seeds, train_sizes):
    print("\n" + "="*50)
    print(f" SCENARIO: Learning Curve Analysis ({molecule.upper()}) (Averaged over {len(seeds)} seeds)")
    print("="*50)

    max_train = max(train_sizes)
    test_size = max(30, len(X) - max_train)
    if len(X) < max_train + 1:
        raise ValueError(f"Insufficient data points ({len(X)}) for max training size ({max_train}).")

    history = {name: {size:[] for size in train_sizes} for name in FINAL_ORDER}

    vqe_proxy = VQEProxy(molecule)
    rbf = ClassicalRBFKernel()
    kf = QuantumKernelFixedAnsatz()

    for seed_idx, seed in enumerate(seeds):
        print(f"\n>>> Running Seed {seed_idx + 1}/{len(seeds)} (Random Seed: {seed}) <<<")

        np.random.seed(seed)
        rng = np.random.RandomState(seed)

        indices = np.arange(len(X))
        rng.shuffle(indices)

        test_idx = indices[-test_size:]
        train_pool_idx = indices[:-test_size]
        X_test, y_test, R_test = X[test_idx], y[test_idx], R[test_idx]

        for size in train_sizes:
            print(f"\n--- Seed {seed} | Training Size: {size} ---")
            curr_train_idx = train_pool_idx[:size]
            X_train, y_train, R_train = X[curr_train_idx], y[curr_train_idx], R[curr_train_idx]

            bo_rbf = RandomSearch([(0.1, 4.0)])
            xb_rbf, _ = bo_rbf.maximize(lambda v: GPZeroNoise(rbf).fit(X_train, y_train, {"ell": v[0]}))
            if xb_rbf is None:
                xb_rbf = [1.0]
            gp_rbf = GPZeroNoise(rbf)
            gp_rbf.fit(X_train, y_train, {"ell": xb_rbf[0]})
            rmse_rbf = gp_rbf.rmse(y_test, gp_rbf.predict(X_test, {"ell": xb_rbf[0]}))
            history['RBF'][size].append(rmse_rbf)

            bo_qf = RandomSearch([(0.1, 5.0), (0.1, 6.0)])
            xb_qf, _ = bo_qf.maximize(lambda v: GPZeroNoise(kf).fit(X_train, y_train, {"theta_ry": v[0], "theta_rzz": v[1]}))
            if xb_qf is None:
                xb_qf = [1.0, 1.0]
            gp_qf = GPZeroNoise(kf)
            th_qf = {"theta_ry": xb_qf[0], "theta_rzz": xb_qf[1]}
            gp_qf.fit(X_train, y_train, th_qf)
            rmse_qf = gp_qf.rmse(y_test, gp_qf.predict(X_test, th_qf))
            history['Q Fixed'][size].append(rmse_qf)

            comp_search = CompositionSearch(2)
            s_vq, th_vq = comp_search.search(X_train, y_train,[(0.1, 5.0), (0.1, 5.0)])
            gp_qv = GPZeroNoise(QuantumKernelAnsatz(s_vq))
            gp_qv.fit(X_train, y_train, th_vq)
            rmse_qv = gp_qv.rmse(y_test, gp_qv.predict(X_test, th_vq))
            history['Q Variable'][size].append(rmse_qv)

            gp_nngp = GPRWithNoise(rbf)
            gp_nngp.fit(X_train, y_train, {"ell": xb_rbf[0]})
            rmse_nngp = gp_nngp.rmse(y_test, gp_nngp.predict(X_test, {"ell": xb_rbf[0]}))
            history['NNGP Proxy'][size].append(rmse_nngp)

            dnn = VQEParamRegressor(vqe_proxy, seed=seed)
            dnn.fit(X_train, R_train)
            vqe_gp = DNNVQEKernelGPR(vqe_proxy, dnn)
            vqe_gp.fit(X_train, y_train)
            vqe_rmse = vqe_gp.rmse(y_test, vqe_gp.predict(X_test))
            history['VQE Ansatz Kernel'][size].append(vqe_rmse)

            print(f"  -> RMSE | RBF: {rmse_rbf:.2f} | Q Fixed: {rmse_qf:.2f} | Q Variable: {rmse_qv:.2f} | NNGP: {rmse_nngp:.2f} | VQE: {vqe_rmse:.2f}")

    mean_history = {name:[] for name in FINAL_ORDER}
    std_history = {name:[] for name in FINAL_ORDER}
    csv_data = {}

    for name in FINAL_ORDER:
        means = [np.mean(history[name][size]) for size in train_sizes]
        stds = [np.std(history[name][size]) for size in train_sizes]

        mean_history[name] = means
        std_history[name] = stds

        csv_data[f"{name}_mean"] = means
        csv_data[f"{name}_std"] = stds

    df_lc = pd.DataFrame(csv_data, index=train_sizes)
    df_lc.index.name = 'Train_Size'
    csv_filepath = f"results/{molecule.lower()}_learning_curve_averaged.csv"
    df_lc.to_csv(csv_filepath)
    print(f"\n[Saved] {csv_filepath}")

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['mathtext.fontset'] = 'stix'

    markers = {'RBF': 'o', 'Q Fixed': 's', 'Q Variable': '^', 'NNGP Proxy': 'D', 'VQE Ansatz Kernel': 'p'}
    linestyles = {'RBF': '--', 'Q Fixed': '-.', 'Q Variable': ':', 'NNGP Proxy': '--', 'VQE Ansatz Kernel': '-'}

    def plot_learning_curve(x_scale_type):
        fig, ax = plt.subplots(figsize=(8, 5))

        for name in FINAL_ORDER:
            means = np.array(mean_history[name])
            stds = np.array(std_history[name])

            lower_bound = np.maximum(means - stds, 1e-1)
            upper_bound = means + stds

            ax.plot(train_sizes, means,
                    marker=markers.get(name, 'o'),
                    markersize=8,
                    linewidth=2.5 if 'VQE' in name else 1.5,
                    color=COLOR_MAP[name],
                    linestyle=linestyles.get(name, '-'),
                    label=name,
                    alpha=0.9)

            ax.fill_between(train_sizes, lower_bound, upper_bound, color=COLOR_MAP[name], alpha=0.10)

        ax.axhline(CHEMICAL_ACCURACY_CM, color=CHEM_ACC_COLOR, linestyle='--', linewidth=2.0, label='Chemical Accuracy')

        if x_scale_type == 'log':
            ax.set_xlabel("Number of Training Samples $N$ (log scale)", fontsize=14, fontweight='bold')
            ax.set_xscale('log')
        else:
            ax.set_xlabel("Number of Training Samples $N$ (linear scale)", fontsize=14, fontweight='bold')
            ax.set_xscale('linear')

        ax.set_xticks(train_sizes)
        ax.set_xticklabels([str(s) for s in train_sizes])
        ax.tick_params(axis='x', rotation=45)

        ax.set_ylabel("RMSE (cm$^{-1}$)", fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.set_ylim(10**2, 10**5)

        ax.tick_params(axis='both', which='major', labelsize=12, width=1.5, length=6)
        ax.tick_params(axis='both', which='minor', width=1.0, length=4)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        for spine in ['left', 'bottom']:
            ax.spines[spine].set_linewidth(1.5)

        ax.grid(True, which='major', linestyle='-', alpha=0.3)
        ax.grid(True, which='minor', linestyle=':', alpha=0.2)
        ax.legend(fontsize=10, loc='upper right', framealpha=1.0, edgecolor='black')

        plt.tight_layout()
        filepath = f"results/{molecule.lower()}_learning_curve_averaged_{x_scale_type}.png"
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[Saved] {filepath}")

    plot_learning_curve('log')
    plot_learning_curve('linear')

def main():
    parser = argparse.ArgumentParser(description="Run Learning Curve Analysis")
    parser.add_argument('--molecule', type=str, default='lih', choices=['lih', 'beh2'], help="Molecule to simulate")
    args = parser.parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')

    print(f"Starting Learning Curve Analysis for {args.molecule.upper()}...")

    cache = MoleculeActiveSpaceCache(molecule_type=args.molecule, num=5100)
    dataset = MoleculeDataset(cache)
    X, y, R, E0 = dataset.generate()

    run_learning_curve(args.molecule, X, y, R, LEARNING_CURVE_SEEDS, TRAIN_SIZES)

if __name__ == "__main__":
    main()
