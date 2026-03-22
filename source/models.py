import numpy as np
from typing import List, Callable
from dataclasses import dataclass
from scipy.optimize import minimize
from scipy.linalg import cho_solve
from sklearn.neural_network import MLPRegressor
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD
from config import BASIS, VQE_TOL, VQE_MAXITER
from utils import transform_x_to_r

class VQEProxy:
    def __init__(self, molecule_type, basis=BASIS):
        self.molecule_type = molecule_type.lower()
        self.basis = basis
        self.mapper = JordanWignerMapper()
        self._ansatz_cache = {}

    def _get_atom_string(self, r):
        if self.molecule_type == 'lih':
            return f"Li 0 0 0; H 0 0 {r}"
        elif self.molecule_type == 'beh2':
            return f"H 0 0 {-r}; Be 0 0 0; H 0 0 {r}"
        else:
            raise ValueError("Unknown molecule type")

    def get_ansatz_and_ops(self, r: float):
        r_key = round(r, 4)
        if r_key in self._ansatz_cache:
            return self._ansatz_cache[r_key]

        atom_str = self._get_atom_string(r)
        if self.molecule_type == 'beh2':
            driver = PySCFDriver(atom=atom_str, basis=self.basis, charge=0, spin=0)
        else:
            driver = PySCFDriver(atom=atom_str, basis=self.basis)
            
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

        def cost(params):
            if len(params) != ansatz.num_parameters:
                return 0.0
            qc = ansatz.assign_parameters(params)
            return Statevector(qc).expectation_value(H_op).real

        if initial_point is None:
            initial_point = np.zeros(ansatz.num_parameters)

        res = minimize(cost, initial_point, method='L-BFGS-B', tol=VQE_TOL, options={'maxiter': VQE_MAXITER})
        return res.x

    def get_state_vector(self, r: float, theta: np.ndarray):
        ansatz, _ = self.get_ansatz_and_ops(r)
        if len(theta) != ansatz.num_parameters:
            t = np.zeros(ansatz.num_parameters)
            L = min(len(theta), len(t))
            t[:L] = theta[:L]
            theta = t
        qc = ansatz.assign_parameters(theta)
        return Statevector(qc).data


class VQEParamRegressor:
    def __init__(self, vqe_proxy, hidden_layer_sizes=(60, 60, 60), seed=42):
        self.vqe = vqe_proxy
        self.mlp = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='tanh', solver='adam', max_iter=3000, random_state=seed, learning_rate_init=0.001
        )
        self.is_fitted = False

    def fit(self, X_train, R_train):
        sorted_indices = np.argsort(R_train.ravel())
        R_sorted = R_train.ravel()[sorted_indices]
        X_sorted = X_train[sorted_indices]

        theta_list =[]
        last_theta = None

        for i, r in enumerate(R_sorted):
            try:
                theta_opt = self.vqe.run_vqe_optimization(r, initial_point=last_theta)
                if np.all(theta_opt == 0) or np.any(np.isnan(theta_opt)):
                    if last_theta is not None:
                        theta_opt = last_theta
                    else:
                        continue
                theta_list.append(theta_opt)
                last_theta = theta_opt
            except Exception:
                if last_theta is not None:
                    theta_list.append(last_theta)
                continue

        if len(theta_list) == 0:
            raise ValueError("VQE Failed.")
        theta_train = np.array(theta_list)
        self.mlp.fit(X_sorted, theta_train)
        self.is_fitted = True

    def predict_theta(self, X):
        if not self.is_fitted:
            raise RuntimeError("DNN not trained")
        return self.mlp.predict(X)


class DNNVQEKernelGPR:
    def __init__(self, vqe_proxy, dnn_regressor, a_transform=1.0, jitter=1e-6):
        self.vqe = vqe_proxy
        self.dnn = dnn_regressor
        self.a = a_transform
        self.jitter = jitter
        self.gamma = 1.0
        self.train_states = None
        self.alpha = None
        self.L = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        y_train = y_train.reshape(-1, 1)
        thetas = self.dnn.predict_theta(X_train)
        R = transform_x_to_r(X_train, self.a).ravel()

        self.train_states =[]
        for r, th in zip(R, thetas):
            psi = self.vqe.get_state_vector(r, th)
            self.train_states.append(psi)
        self.train_states = np.array(self.train_states)

        F = np.abs(self.train_states @ self.train_states.conj().T)**2

        best_rmse = float('inf')
        best_gamma = 1.0
        best_model_params = None
        candidate_gammas =[0.1, 0.25, 0.5, 1.0, 2.0, 5.0]

        for g in candidate_gammas:
            K = np.exp(-(1.0 - F) / g)
            K[np.diag_indices_from(K)] += self.jitter
            try:
                L = np.linalg.cholesky(K)
                alpha = cho_solve((L, True), y_train)
                y_pred_train = (K @ alpha).ravel()
                rmse_train = np.sqrt(np.mean((y_train.ravel() - y_pred_train)**2))
                if rmse_train < best_rmse:
                    best_rmse = rmse_train
                    best_gamma = g
                    best_model_params = (L, alpha)
            except:
                continue

        self.gamma = best_gamma
        if best_model_params:
            self.L, self.alpha = best_model_params
        return self

    def predict(self, X_star, return_std=False):
        thetas = self.dnn.predict_theta(X_star)
        R = transform_x_to_r(X_star, self.a).ravel()

        test_states =[]
        for r, th in zip(R, thetas):
            psi = self.vqe.get_state_vector(r, th)
            test_states.append(psi)
        test_states = np.array(test_states)

        F_star = np.abs(test_states @ self.train_states.conj().T)**2
        K_star = np.exp(-(1.0 - F_star) / self.gamma)

        y_pred = (K_star @ self.alpha).ravel()

        if return_std:
            v = cho_solve((self.L, True), K_star.T)
            var = 1.0 - np.sum(K_star.T * v, axis=0)
            return y_pred, np.sqrt(np.maximum(var, 0))
        return y_pred

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))


class GPZeroNoise:
    def __init__(self, kernel_fn: Callable, jitter: float = 1e-6):
        self.kernel_fn = kernel_fn
        self.jitter = jitter

    def fit(self, X, y, theta=None):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).reshape(-1, 1)
        if theta is None:
            theta = {}
        K = self.kernel_fn(self.X_train, self.X_train, theta)
        K[np.diag_indices_from(K)] += self.jitter
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = cho_solve((self.L, True), self.y_train)
        except:
            return -np.inf
        return 0

    def predict(self, X_star, theta=None, return_std=False):
        if theta is None:
            theta = {}
        K_star = self.kernel_fn(self.X_train, X_star, theta)
        y_pred = K_star.T @ self.alpha
        if not return_std:
            return y_pred.ravel()
        v = cho_solve((self.L, True), K_star)
        K_star_star = self.kernel_fn(X_star, X_star, theta)
        var = np.diag(K_star_star) - np.sum(v**2, axis=0)
        return y_pred.ravel(), np.sqrt(np.maximum(var, 0))

    def rmse(self, y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))


class GPRWithNoise(GPZeroNoise):
    def __init__(self, kernel_fn, noise_level=1e-5):
        super().__init__(kernel_fn)
        self.noise_level = noise_level

    def fit(self, X, y, theta=None):
        self.X_train = np.asarray(X)
        self.y_train = np.asarray(y).reshape(-1, 1)
        if theta is None:
            theta = {}
        K = self.kernel_fn(self.X_train, self.X_train, theta)
        K[np.diag_indices_from(K)] += (self.noise_level + self.jitter)
        try:
            self.L = np.linalg.cholesky(K)
            self.alpha = cho_solve((self.L, True), self.y_train)
        except:
            return -np.inf
        return 0


class ClassicalRBFKernel:
    def __call__(self, X, Z, theta):
        ell = theta.get("ell", 1.0)
        X2 = X**2
        Z2 = Z**2
        d = np.sum(X2, 1)[:, None] + np.sum(Z2, 1) - 2 * X @ Z.T
        return np.exp(-0.5 * d / (ell**2))


class QuantumKernelFixedAnsatz:
    def __init__(self, m=2):
        self.m = m

    def _state(self, x, theta):
        qc = QuantumCircuit(self.m)
        qc.h(range(self.m))
        qc.rzz(theta.get("theta_rzz", 1.0), 0, 1)
        qc.ry(x[0] / theta.get("theta_ry", 1.0), 0)
        return Statevector(qc).data

    def __call__(self, X, Z, theta):
        sx =[self._state(x, theta) for x in X]
        sz = [self._state(z, theta) for z in Z]
        return np.abs(np.array(sx) @ np.array(sz).conj().T)**2


@dataclass
class CircuitStructure:
    m: int
    layers: List[bool]


class QuantumKernelAnsatz:
    def __init__(self, struct):
        self.struct = struct
        self.m = struct.m

    def _state(self, x, theta):
        qc = QuantumCircuit(self.m)
        qc.h(range(self.m))
        for flag in self.struct.layers:
            if flag:
                qc.rzz(np.exp(-x[0]/theta.get("Theta_rzz", 1.0)), 0, 1)
        qc.ry(x[0] / theta.get("theta_ry", 1.0), 0)
        return Statevector(qc).data

    def __call__(self, X, Z, theta):
        sx = [self._state(x, theta) for x in X]
        sz =[self._state(z, theta) for z in Z]
        return np.abs(np.array(sx) @ np.array(sz).conj().T)**2


class RandomSearch:
    def __init__(self, bounds, n_iter=15):
        self.bounds = np.array(bounds)
        self.n_iter = n_iter

    def maximize(self, func):
        bx, by = None, -np.inf
        for _ in range(self.n_iter):
            c = np.random.uniform(self.bounds[:,0], self.bounds[:,1])
            try:
                v = func(c)
                if v > by:
                    by, bx = v, c
            except:
                continue
        return bx, by


class CompositionSearch:
    def __init__(self, m):
        self.m = m

    def search(self, X, y, b):
        s = CircuitStructure(self.m, [True, True])
        def obj(v):
            gp = GPZeroNoise(QuantumKernelAnsatz(s))
            return gp.fit(X, y, {"theta_ry": v[0], "Theta_rzz": v[1]})
        opt = RandomSearch(b, n_iter=10)
        xb, _ = opt.maximize(obj)
        if xb is None:
            xb = [1.0, 1.0]
        return s, {"theta_ry": xb[0], "Theta_rzz": xb[1]}
