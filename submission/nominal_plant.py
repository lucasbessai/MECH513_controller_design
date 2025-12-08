# %%
# libraries 
import numpy as np
import control as ct
from scipy import signal
from matplotlib import pyplot as plt
from scipy import fft
import pathlib

# My libraries
import System_ID_Functions as sysid

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# Golden ratio
gr = (1 + np.sqrt(5)) / 2
# Figure height
height = 11 / 2.54  # cm

# %%
# Read in all input-output (IO) data
path = pathlib.Path('MECH513_part1_load_data_sc/load_data_sc/SINE_SWEEP_DATA/')
all_files = sorted(path.glob("*.csv"))
# all_files.sort()
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
    ) for filename in all_files
]
data = np.array(data)

# %%
# Load a dataset
data_0 = data[0]
data_1 = data[1]
data_2 = data[2]
data_3 = data[3]

datasets = [data_0, data_1, data_2, data_3]

# %%
# Functions

def off_nominals(datasets, m, n, plot=False):

    # fit a transfer function to each data set
    TFs = []

    for i, D in enumerate(datasets):
        res = sysid.signal_process(D)

        f = res['f']
        omegas = res['omega']
        Gs = res['G']
        Cuy = res['Cuy']
        mask = res['mask']
        fs = res['fs']
        t = res['t']
        u = res['u']
        y = res['y']
        csd = res['csd']

        G = Gs[0]
        omega = omegas[0]
        A, b = sysid.form_LS_system(G, omega, m, n)
        G_LS, x = sysid.LS_fit(A, b, m, n)

        TFs.append((f'G_{i}', G_LS))

        # Compute response error of each off nominal with test data

        for j, D_test in enumerate(datasets):
            if j == i: 
                continue
    if plot:
        # Bode plot of all TFs

        w_shared = np.logspace(-2, 2, 2000)  # rad/s
        # [1e-2, 1e2]
        f_shared_Hz = w_shared / 2 / np.pi
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xlabel(r'$\omega$ (rad/s)')
        ax[0].set_ylabel(r'Magnitutde (dB)')
        ax[1].set_xlabel(r'$\omega$ (rad/s)')
        ax[1].set_ylabel(r'Phase (deg)')
        for G in TFs:
            mag_est, phase_est, _ = ct.frequency_response(G[1], w_shared)
            ct.bode_plot(G[1], w_shared, dB=True, deg=True, title='', label=G[0])
        fig.tight_layout()

    return TFs



# %% 
def nominal_plant_median(datasets, m, n, plot=False):
    off_noms = off_nominals(datasets, m=m, n=n, plot=False)

    num_coeffs = np.array([np.ravel(G[1].num) for G in off_noms])
    denom_coeffs = np.array([np.ravel(G[1].den) for G in off_noms])
    median_num_coeffs = np.median(num_coeffs, axis=0)
    median_denom_coeffs = np.median(denom_coeffs, axis=0)

    G_nom = ct.tf(median_num_coeffs, median_denom_coeffs)

    if plot:
        # Bode plot of all TFs

        w_shared = np.logspace(-2, 2, 2000)  # rad/s
        # [1e-2, 1e2]
        fig, ax = plt.subplots(2, 1)
        ax[0].set_xlabel(r'$\omega$ (rad/s)')
        ax[0].set_ylabel(r'Magnitutde (dB)')
        ax[1].set_xlabel(r'$\omega$ (rad/s)')
        ax[1].set_ylabel(r'Phase (deg)')
        for G in off_noms:
            ct.bode_plot(G[1], w_shared, dB=True, deg=True, title='', label=G[0], color='C0', linestyle='dashed')
        ct.bode_plot(G_nom, w_shared, dB=True, deg=True, title='', label='G_nom', color='C3')
        fig.tight_layout()
    
    return G_nom

# G_nom = nominal_plant_median(datasets, m=0, n=1)
# sysid.response_error(data_0, G_nom, plot=True)
# sysid.response_error(data_1, G_nom, plot=True)
# sysid.response_error(data_2, G_nom, plot=True)
# sysid.response_error(data_3, G_nom, plot=True)
# %% 
off_noms = off_nominals(datasets, m=0, n=1, plot=False)

G_nom = nominal_plant_median(datasets, m=0, n=1, plot=True)

def plot_nom(G_nom):
    w_shared = np.logspace(-2, 2, 2000)  # rad/s
    # [1e-2, 1e2]
    fig, ax = plt.subplots(2, 1)
    ct.bode_plot(G_nom, w_shared, dB=True, deg=True, title='', label='G_nom')
    ax[0].set_xlabel(r'$\omega$ (rad/s)')
    ax[0].set_ylabel(r'Magnitutde (dB)')
    ax[1].set_xlabel(r'$\omega$ (rad/s)')
    ax[1].set_ylabel(r'Phase (deg)')
    fig.tight_layout()

# %% 
# Minimize Multiplicative uncertainty nominal

def Gjw(G, w):
    # complex frequency response using python-control
    mag, phase, _ = ct.frequency_response(G, w)
    return np.squeeze(mag) * np.exp(1j*np.squeeze(phase))

def mult_dist_matrix(P_list, w, omega_weights=None, norm='hinf'):
    """
    D[k,j] = distance from nominal candidate j to plant k
    norm: 'hinf' (sup |R|)  or  'h2' (weighted L2 over w)
    """
    K = len(P_list)
    G = [Gjw(P, w) for P in P_list]
    if omega_weights is None:
        W = np.ones_like(w, dtype=float)
    else:
        W = np.asarray(omega_weights, dtype=float)

    D = np.zeros((K, K), dtype=float)
    for j in range(K):
        Gj = G[j]
        for k in range(K):
            if k == j:
                continue
            R = G[k] / (Gj + 1e-16) - 1.0
            if norm == 'hinf':
                D[k, j] = np.max(np.abs(R))
            else:  # 'h2'
                D[k, j] = np.sqrt(np.sum(W * np.abs(R)**2) / np.sum(W))
    return D

def choose_nominal_medoid(P_list, w, omega_weights=None, norm='hinf', score='mean'):
    """
    score: 'mean' -> argmin_j mean_k D[k,j]
           'max'  -> argmin_j max_k  D[k,j]   (minimax)
    """
    D = mult_dist_matrix(P_list, w, omega_weights, norm=norm)
    if score == 'max':
        s = np.max(D, axis=0)
    else:
        s = np.mean(D, axis=0)
    j_star = int(np.argmin(s))
    return P_list[j_star], j_star, s, D

# %%

