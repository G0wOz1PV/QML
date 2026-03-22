import os
import datetime
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize
from scipy.linalg import cho_solve
from sklearn.neural_network import MLPRegressor

from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.primitives import BackendEstimatorV2, StatevectorEstimator

from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD

from config import BASIS, CHEMICAL_ACCURACY_CM
from utils import transform_x_to_r, interpolation_split
from dataset import MoleculeActiveSpaceCache, MoleculeDataset

NUM_TRIALS = 10
TIMESTAMP = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = f"results_noise_analysis_{TIMESTAMP}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_csv(df, filename):
    df.to_csv(os.path.join(OUTPUT_DIR, filename), index=False)

def get_noise_model(noise_level: float):
    if noise_level <= 0.0:
        return None
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level / 10.0, 1),['u1', 'u2', 'u3', 'rx', 'ry', 'rz', 'h', 'x', 'y', 'z'])
    noise_model.add_all_qubit_quantum_error(depolarizing_error(noise_level, 2),['cx', 'cz'])
    return noise_model

class NoisyVQEProxy:
    def __init__(self, noise_level=0.0, basis=BASIS):
        self.basis = basis
        self.noise_level = noise_level
        self.mapper = JordanWignerMapper()
        self._ansatz_cache = {}

        if self.noise_level > 0.0:
            nm = get_noise_model(self.noise_level)
            self.backend = AerSimulator(noise_model=nm)
            self.estimator = BackendEstimatorV2(backend=self.backend)
            self.estimator.options.default_shots = 2048
            self.pm = generate_preset_pass_manager(target=self.backend.target, optimization_level=1)
        else:
            self.estimator = StatevectorEstimator()
            self.pm = None

    def get_ansatz_and_ops(self, r: float):
        r_key = round(r, 4)
        if r_key in self._ansatz_cache:
            return self._ansatz_cache[r_key]
        
        driver = PySCFDriver(atom=f"Li 0 0 0; H 0 0 {r}", basis=self.basis)
        problem = driver.run()
        transformer = ActiveSpaceTransformer(num_electrons=2, num_spatial_orbitals=2)
        problem = transformer.transform(problem)
        H_op = self.mapper.map(problem.hamiltonian.second_q_op())
        init_state = HartreeFock(problem.num_spatial_orbitals, problem.num_particles, self.mapper)
        ansatz = UCCSD(problem.num_spatial_orbitals, problem.num_particles, self.mapper, initial_state=init_state)
        self._ansatz_cache[r_key] = (ansatz, H_op)
        return ansatz, H_op

    def run_vqe_optimization(self, r: float, initial_point=None):
        ansatz, H_op = self.get_ansatz_and_ops(r)
        
        if self.noise_level > 0.0:
            isa_ansatz = self.pm.run(ansatz)
            isa_H = H_op.apply_layout(isa_ansatz.layout)
            def cost(params):
                return float(self.estimator.run([(isa_ansatz, isa_H, params)]).result()[0].data.evs)
            method, max_iter = 'COBYLA', 200
        else:
            def cost(params):
                return float(self.estimator.run([(ansatz, H_op, params)]).result()[0].data.evs)
            method, max_iter = 'L-BFGS-B', 500

        if initial_point is None:
            initial_point = np.zeros(ansatz.num_parameters)
            
        return minimize(cost, initial_point, method=method, options={'maxiter': max_iter}).x

def generate_vqe_training_data(vqe_proxy, R_train):
    sorted_indices = np.argsort(R_train.ravel())
    R_sorted = R_train.ravel()[sorted_indices]

    theta_list =[]
    last_theta = None
    desc_str = f"Generating VQE thetas (Noise {vqe_proxy.noise_level*100}%)"
    
    for r in tqdm(R_sorted, desc=desc_str, leave=False):
        try:
            theta_opt = vqe_proxy.run_vqe_optimization(r, initial_point=last_theta)
            theta_list.append(theta_opt)
            last_theta = theta_opt
        except Exception:
            ansatz, _ = vqe_proxy.get_ansatz_and_ops(r)
            theta_list.append(last_theta if last_theta is not None else np.zeros(ansatz.num_parameters))

    inv_indices = np.argsort(sorted_indices)
    return np.array(theta_list)[inv_indices]

class NoisyVQEParamRegressor:
    def __init__(self, hidden_layer_sizes=(30, 30), seed=42):
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes, activation='tanh', solver='adam',
            max_iter=3000, random_state=seed, learning_rate_init=0.001
        )

    def fit(self, X_train, thetas_train):
        self.mlp.fit(X_train, thetas_train)

    def predict_theta(self, X):
        return self.mlp.predict(X)

class NoisyDNNVQEKernelGPR:
    def __init__(self, vqe_proxy, dnn_regressor, noise_level=0.0):
        self.vqe = vqe_proxy
        self.dnn = dnn_regressor
        self.noise_level = noise_level
        self.a, self.gamma, self.jitter = 1.0, 1.0, 1e-7

        if self.noise_level > 0.0:
            nm = get_noise_model(self.noise_level)
            self.backend = AerSimulator(noise_model=nm)
            self.estimator = BackendEstimatorV2(backend=self.backend)
            self.estimator.options.default_shots = 2048
            self.pm = generate_preset_pass_manager(target=self.backend.target, optimization_level=1)
        else:
            self.estimator = StatevectorEstimator()
            self.pm = None

    def _get_zero_projector(self, num_qubits):
        pauli_strs =["".join(p) for p in itertools.product(["I", "Z"], repeat=num_qubits)]
        coeffs = [1.0 / (2**num_qubits)] * (2**num_qubits)
        return SparsePauliOp(pauli_strs, coeffs)

    def _calc_fidelity_matrix(self, R_X, thetas_X, R_Z, thetas_Z):
        N_x, N_z = len(R_X), len(R_Z)
        circuits =[]
        for i in range(N_x):
            ansatz_x, _ = self.vqe.get_ansatz_and_ops(R_X[i])
            qc_x = ansatz_x.assign_parameters(thetas_X[i])
            m_qubits = qc_x.num_qubits
            for j in range(N_z):
                ansatz_z, _ = self.vqe.get_ansatz_and_ops(R_Z[j])
                qc_z = ansatz_z.assign_parameters(thetas_Z[j])
                qc = QuantumCircuit(m_qubits)
                qc.compose(qc_x, inplace=True)
                qc.compose(qc_z.inverse(), inplace=True)
                circuits.append(qc)

        proj = self._get_zero_projector(m_qubits)
        if self.noise_level > 0.0:
            isa_circuits = self.pm.run(circuits)
            pubs = [(qc, proj.apply_layout(qc.layout)) for qc in isa_circuits]
        else:
            pubs = [(qc, proj) for qc in circuits]

        evs =[float(res.data.evs) for res in self.estimator.run(pubs).result()]
        return np.array(evs).reshape((N_x, N_z))

    def fit(self, X_train, y_train):
        self.train_thetas = self.dnn.predict_theta(X_train)
        self.train_R = transform_x_to_r(X_train, self.a).ravel()

        F = self._calc_fidelity_matrix(self.train_R, self.train_thetas, self.train_R, self.train_thetas)
        F = (F + F.T) / 2.0
        K = np.exp(-(1.0 - np.maximum(F, 0)) / self.gamma)

        current_jitter = self.jitter
        while current_jitter < 10.0:
            try:
                K_fit = K.copy()
                K_fit[np.diag_indices_from(K_fit)] += current_jitter
                self.L = np.linalg.cholesky(K_fit)
                self.alpha = cho_solve((self.L, True), y_train)
                break
            except np.linalg.LinAlgError:
                current_jitter *= 10.0
        return self

    def predict(self, X_star):
        thetas_star = self.dnn.predict_theta(X_star)
        R_star = transform_x_to_r(X_star, self.a).ravel()
        F_star = self._calc_fidelity_matrix(R_star, thetas_star, self.train_R, self.train_thetas)
        K_star = np.exp(-(1.0 - np.maximum(F_star, 0)) / self.gamma)
        return (K_star @ self.alpha).ravel()

def main_noise_analysis():
    print(f"Starting Noise Impact Analysis ({NUM_TRIALS} Trials)...")

    cache = MoleculeActiveSpaceCache(molecule_type='lih', num=12)
    dataset = MoleculeDataset(cache)
    X, y, R, _ = dataset.generate()
    
    Xt, yt, Rt, Xte, yte, _ = interpolation_split(X, y, R, train_frac=0.6, seed=42)

    noise_levels =[0.0, 0.0001, 0.001, 0.01, 0.1]
    results_all = {
        'VQE Only Noisy': {p:[] for p in noise_levels},
        'Fidelity Only Noisy': {p:[] for p in noise_levels},
        'Both Noisy': {p: [] for p in noise_levels}
    }

    print("\n>>> [Phase 1] Preparing VQE Training Data (Optimizing thetas once per noise level)...")
    vqe_proxies = {}
    theta_train_dict = {}

    for p in noise_levels:
        vqe_proxy = NoisyVQEProxy(noise_level=p)
        vqe_proxies[p] = vqe_proxy
        theta_train_dict[p] = generate_vqe_training_data(vqe_proxy, Rt)

    print(f"\n>>> [Phase 2] Starting {NUM_TRIALS} Trials for DNN Prediction & GPR Evaluation...")

    for trial in range(NUM_TRIALS):
        seed = 42 + trial
        print(f"\n--- TRIAL {trial + 1} / {NUM_TRIALS} ---")

        dnn_ideal = NoisyVQEParamRegressor(seed=seed)
        dnn_ideal.fit(Xt, theta_train_dict[0.0])
        gpr_ideal = NoisyDNNVQEKernelGPR(vqe_proxies[0.0], dnn_ideal, noise_level=0.0)
        gpr_ideal.fit(Xt, yt)
        rmse_ideal = np.sqrt(np.mean((yte.ravel() - gpr_ideal.predict(Xte).ravel())**2))

        for p in noise_levels:
            if p == 0.0:
                results_all['VQE Only Noisy'][p].append(rmse_ideal)
                results_all['Fidelity Only Noisy'][p].append(rmse_ideal)
                results_all['Both Noisy'][p].append(rmse_ideal)
                continue

            dnn_noisy = NoisyVQEParamRegressor(seed=seed)
            dnn_noisy.fit(Xt, theta_train_dict[p])

            gpr_A = NoisyDNNVQEKernelGPR(vqe_proxies[p], dnn_noisy, noise_level=0.0)
            gpr_A.fit(Xt, yt)
            results_all['VQE Only Noisy'][p].append(np.sqrt(np.mean((yte.ravel() - gpr_A.predict(Xte).ravel())**2)))

            gpr_B = NoisyDNNVQEKernelGPR(vqe_proxies[0.0], dnn_ideal, noise_level=p)
            gpr_B.fit(Xt, yt)
            results_all['Fidelity Only Noisy'][p].append(np.sqrt(np.mean((yte.ravel() - gpr_B.predict(Xte).ravel())**2)))

            gpr_C = NoisyDNNVQEKernelGPR(vqe_proxies[p], dnn_noisy, noise_level=p)
            gpr_C.fit(Xt, yt)
            results_all['Both Noisy'][p].append(np.sqrt(np.mean((yte.ravel() - gpr_C.predict(Xte).ravel())**2)))

    rows =[]
    for scenario, data_dict in results_all.items():
        for p in noise_levels:
            for trial_id, val in enumerate(data_dict[p]):
                rows.append({
                    "scenario": scenario,
                    "noise_level": p,
                    "trial": trial_id,
                    "rmse_cm": val
                })

    df_raw = pd.DataFrame(rows)
    save_csv(df_raw, "rmse_raw_all_trials.csv")

    summary_rows =[]
    for scenario, data_dict in results_all.items():
        for p in noise_levels:
            vals = np.array(data_dict[p])
            summary_rows.append({
                "scenario": scenario,
                "noise_level": p,
                "mean_rmse": np.mean(vals),
                "std_rmse": np.std(vals)
            })

    df_summary = pd.DataFrame(summary_rows)
    save_csv(df_summary, "rmse_summary.csv")

    noise_labels =["0", "0.01", "0.1", "1", "10"]
    x_vals =[1e-5, 1e-4, 1e-3, 1e-2, 1e-1]

    means_A = [np.mean(results_all['VQE Only Noisy'][p]) for p in noise_levels]
    stds_A = [np.std(results_all['VQE Only Noisy'][p]) for p in noise_levels]
    means_B = [np.mean(results_all['Fidelity Only Noisy'][p]) for p in noise_levels]
    stds_B =[np.std(results_all['Fidelity Only Noisy'][p]) for p in noise_levels]
    means_C = [np.mean(results_all['Both Noisy'][p]) for p in noise_levels]
    stds_C = [np.std(results_all['Both Noisy'][p]) for p in noise_levels]

    plt.figure(figsize=(10, 6))

    plt.errorbar(x_vals, means_A, yerr=stds_A, marker='o', label='VQE Noisy, Fidelity Ideal',
                 linewidth=2.5, color='#1f77b4', markersize=8, capsize=6, capthick=2, elinewidth=2)
    plt.errorbar(x_vals, means_B, yerr=stds_B, marker='s', label='VQE Ideal, Fidelity Noisy',
                 linewidth=2.5, color='#ff7f0e', markersize=8, capsize=6, capthick=2, elinewidth=2)
    plt.errorbar(x_vals, means_C, yerr=stds_C, marker='^', label='Both Noisy',
                 linewidth=2.5, color='#d62728', markersize=8, capsize=6, capthick=2, elinewidth=2)

    plt.axhline(CHEMICAL_ACCURACY_CM, color='#B10303', linestyle='--', label='Chemical Accuracy', alpha=0.8)

    plt.xscale('log')
    plt.xticks(x_vals, noise_labels)
    plt.xlabel("Depolarizing Noise Level (%) [Log Scale]", fontsize=12)
    plt.ylabel("RMSE (cm⁻¹)[Mean ± Std]", fontsize=12)
    plt.title(f"Noise Impact on PES Prediction ({NUM_TRIALS} Trials)", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, which="both", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "noise_impact_variance.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    main_noise_analysis()
