import os
import argparse
import numpy as np
import pandas as pd
from config import FINAL_ORDER, SEEDS_DICT
from utils import interpolation_split, extrapolation_split, transform_r_to_x, plot_aggregated_results
from dataset import MoleculeActiveSpaceCache, MoleculeDataset
from models import (
    VQEProxy, VQEParamRegressor, DNNVQEKernelGPR, 
    ClassicalRBFKernel, QuantumKernelFixedAnsatz, QuantumKernelAnsatz,
    GPZeroNoise, GPRWithNoise, RandomSearch, CompositionSearch
)

def run_scenario(molecule, scenario, X, y, R, cache, E0, ha_to_cm):
    print(f"=============================================")
    print(f" SCENARIO: {scenario} ({molecule.upper()})")
    print(f"=============================================")

    R_full = np.linspace(R.min(), R.max(), 100).reshape(-1, 1)
    X_full = transform_r_to_x(R_full)

    all_results = {m: {'rmse': [], 'y_mu':[]} for m in FINAL_ORDER}
    first_data = None

    seed_key = f"{molecule.lower()}_{scenario.lower()}"
    seeds = SEEDS_DICT.get(seed_key, [42])

    for seed in seeds:
        np.random.seed(seed)
        
        if scenario.lower() == "interpolation":
            Xt, yt, Rt, Xte, yte, Rte = interpolation_split(X, y, R, seed=seed)
        else:
            Xt, yt, Rt, Xte, yte, Rte = extrapolation_split(X, y, R)

        if first_data is None:
            first_data = (Rt, yt, Rte, yte)

        vqe_proxy = VQEProxy(molecule)

        rbf = ClassicalRBFKernel()
        bo = RandomSearch([(0.1, 4.0)])
        xb_rbf, _ = bo.maximize(lambda v: GPZeroNoise(rbf).fit(Xt, yt, {"ell": v[0]}))
        if xb_rbf is None:
            xb_rbf =[1.0]
        
        gp_rbf = GPZeroNoise(rbf)
        gp_rbf.fit(Xt, yt, {"ell": xb_rbf[0]})
        pred_rbf = gp_rbf.predict(Xte, {"ell": xb_rbf[0]})
        all_results['RBF']['rmse'].append(gp_rbf.rmse(yte, pred_rbf))
        all_results['RBF']['y_mu'].append(gp_rbf.predict(X_full, {"ell": xb_rbf[0]}))

        kf = QuantumKernelFixedAnsatz()
        bo_qf = RandomSearch([(0.1, 5.0), (0.1, 6.0)])
        xb_qf, _ = bo_qf.maximize(lambda v: GPZeroNoise(kf).fit(Xt, yt, {"theta_ry": v[0], "theta_rzz": v[1]}))
        if xb_qf is None:
            xb_qf =[1.0, 1.0]
            
        gp_qf = GPZeroNoise(kf)
        th_qf = {"theta_ry": xb_qf[0], "theta_rzz": xb_qf[1]}
        gp_qf.fit(Xt, yt, th_qf)
        pred_qf = gp_qf.predict(Xte, th_qf)
        all_results['Q Fixed']['rmse'].append(gp_qf.rmse(yte, pred_qf))
        all_results['Q Fixed']['y_mu'].append(gp_qf.predict(X_full, th_qf))

        comp_search = CompositionSearch(2)
        s_vq, th_vq = comp_search.search(Xt, yt,[(0.1, 5.0), (0.1, 5.0)])
        gp_vq = GPZeroNoise(QuantumKernelAnsatz(s_vq))
        gp_vq.fit(Xt, yt, th_vq)
        pred_vq = gp_vq.predict(Xte, th_vq)
        all_results['Q Variable']['rmse'].append(gp_vq.rmse(yte, pred_vq))
        all_results['Q Variable']['y_mu'].append(gp_vq.predict(X_full, th_vq))

        gp_nngp = GPRWithNoise(rbf)
        gp_nngp.fit(Xt, yt, {"ell": xb_rbf[0]})
        pred_nngp = gp_nngp.predict(Xte, {"ell": xb_rbf[0]})
        all_results['NNGP Proxy']['rmse'].append(gp_nngp.rmse(yte, pred_nngp))
        all_results['NNGP Proxy']['y_mu'].append(gp_nngp.predict(X_full, {"ell": xb_rbf[0]}))

        dnn = VQEParamRegressor(vqe_proxy, seed=seed)
        dnn.fit(Xt, Rt)
        vqe_gp = DNNVQEKernelGPR(vqe_proxy, dnn)
        vqe_gp.fit(Xt, yt)
        pred_vqe = vqe_gp.predict(Xte)
        rmse_vqe = vqe_gp.rmse(yte, pred_vqe)
        all_results['VQE Ansatz Kernel']['rmse'].append(rmse_vqe)
        all_results['VQE Ansatz Kernel']['y_mu'].append(vqe_gp.predict(X_full))

    summary = {}
    csv_rows =[]

    for method in FINAL_ORDER:
        rmses = np.array(all_results[method]['rmse'])
        y_mus = np.array(all_results[method]['y_mu'])

        rmse_mean = np.mean(rmses)
        rmse_std = np.std(rmses)
        y_mu_mean = np.mean(y_mus, axis=0)
        y_mu_std = np.std(y_mus, axis=0)

        summary[method] = {
            'rmse_mean': rmse_mean,
            'rmse_std': rmse_std,
            'y_mu_mean': y_mu_mean,
            'y_mu_std': y_mu_std
        }

        csv_rows.append({
            'Scenario': scenario,
            'Method': method,
            'RMSE_Mean': rmse_mean,
            'RMSE_Std': rmse_std
        })

    df_metrics = pd.DataFrame(csv_rows)
    metrics_path = f"results/{molecule.lower()}_metrics_{scenario.lower()}.csv"
    df_metrics.to_csv(metrics_path, index=False)

    y_pred_df = pd.DataFrame({'R': R_full.ravel()})
    for method in FINAL_ORDER:
        y_pred_df[f"{method}_mean"] = summary[method]['y_mu_mean']
        y_pred_df[f"{method}_std"] = summary[method]['y_mu_std']
    pred_path = f"results/{molecule.lower()}_predictions_{scenario.lower()}.csv"
    y_pred_df.to_csv(pred_path, index=False)

    plot_aggregated_results(summary, first_data[0], first_data[1], first_data[2], first_data[3], R_full, scenario, molecule)

def main():
    parser = argparse.ArgumentParser(description="Run PES Simulation")
    parser.add_argument('--molecule', type=str, required=True, choices=['lih', 'beh2'], help="Molecule to simulate")
    parser.add_argument('--scenario', type=str, required=True, choices=['interpolation', 'extrapolation'], help="Data split scenario")
    args = parser.parse_args()

    if not os.path.exists('results'):
        os.makedirs('results')

    cache = MoleculeActiveSpaceCache(molecule_type=args.molecule, num=30)
    dataset = MoleculeDataset(cache)
    X, y, R, E0 = dataset.generate()
    ha_to_cm = dataset.ha_to_cm

    run_scenario(args.molecule, args.scenario, X, y, R, cache, E0, ha_to_cm)

if __name__ == "__main__":
    main()
