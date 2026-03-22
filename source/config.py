BASIS = 'sto-3g'
VQE_MAXITER = 3000
VQE_TOL = 1e-12
CHEMICAL_ACCURACY_CM = 349.755
CHEM_ACC_COLOR = '#b00000'

COLOR_MAP = {
    'RBF': '#E83C90',
    'VQE Ansatz Kernel': '#8D75B8',
    'Q Fixed': '#D96A1C',
    'Q Variable': '#1C9E7B',
    'NNGP Proxy': '#74AD2C'
}

FINAL_ORDER =['RBF', 'Q Fixed', 'Q Variable', 'NNGP Proxy', 'VQE Ansatz Kernel']

SEEDS_DICT = {
    'lih_interpolation':[5, 7, 8, 13, 14, 15, 48, 60, 64, 78],
    'lih_extrapolation':[9, 13, 28, 41, 45, 46, 48, 58, 89, 93],
    'beh2_interpolation':[5, 6, 8, 13, 37, 53, 60, 64, 72, 78],
    'beh2_extrapolation':[16, 20, 44, 49, 59, 71, 77, 78, 86, 93]
}

LEARNING_CURVE_SEEDS =[5, 7, 8, 13, 14, 15, 48, 60, 64, 78]
TRAIN_SIZES =[5, 10, 15, 20, 25, 30, 50, 100, 300, 500, 1000, 5000]
