import numpy as np
import pandas as pd
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.transformers import ActiveSpaceTransformer
from config import BASIS
from utils import transform_r_to_x

class MoleculeActiveSpaceCache:
    def __init__(self, molecule_type, r_min=0.5, r_max=3.5, num=30, basis=BASIS):
        self.molecule_type = molecule_type.lower()
        self.r_grid = np.linspace(r_min, r_max, num).astype(float)
        self.basis = basis
        self.mapper = JordanWignerMapper()
        self.n_electrons = 2
        self.n_spatial_orbitals = 2
        self.data_points =[]
        self._build_all()

    def _get_atom_string(self, r):
        if self.molecule_type == 'lih':
            return f"Li 0 0 0; H 0 0 {r}"
        elif self.molecule_type == 'beh2':
            return f"H 0 0 {-r}; Be 0 0 0; H 0 0 {r}"
        else:
            raise ValueError("Unknown molecule type")

    def _build_all(self):
        for r in self.r_grid:
            try:
                atom_str = self._get_atom_string(r)
                if self.molecule_type == 'beh2':
                    driver = PySCFDriver(atom=atom_str, basis=self.basis, charge=0, spin=0)
                else:
                    driver = PySCFDriver(atom=atom_str, basis=self.basis)
                
                problem = driver.run()
                transformer = ActiveSpaceTransformer(
                    num_electrons=self.n_electrons,
                    num_spatial_orbitals=self.n_spatial_orbitals
                )
                problem = transformer.transform(problem)
                
                H2 = problem.hamiltonian.second_q_op()
                Hq = self.mapper.map(H2)
                H_mat = Hq.to_matrix()
                w, _ = np.linalg.eigh(H_mat)

                self.data_points.append({
                    "R": r,
                    "E_exact": w[0]
                })
            except Exception:
                pass

    def get_dataset(self):
        df = pd.DataFrame(self.data_points)
        if df.empty:
            raise ValueError("Dataset Empty.")
        R = df["R"].values
        E = df["E_exact"].values
        E0 = E.min()
        y_cm = (E - E0) * 219474.63
        X = transform_r_to_x(R)
        return X.reshape(-1, 1), y_cm.astype(float), R.reshape(-1, 1), E0


class MoleculeDataset:
    def __init__(self, cache, a_transform=1.0, ha_to_cm=219474.63):
        self.cache = cache
        self.a = a_transform
        self.ha_to_cm = ha_to_cm

    def generate(self):
        return self.cache.get_dataset()
