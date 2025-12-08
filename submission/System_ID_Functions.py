# %%
# libraries 
import numpy as np
import control as ct
from scipy import signal
from matplotlib import pyplot as plt
from scipy import fft
import pathlib

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

# %%
# Frequency Domain Transfer Function and Error
def signal_process(
        data,
        coh_thresh=0.8,
        min_puu_frac=1e-3,
        fmax_frac=0.9
    ):
    """
    Returns a dict with {omega, G, Cuy, mask, fs, t, u, y}.
    Mask keeps points: Cuy>=coh_thresh, Puu above noise floor, and f<=fmax_frac*fN.
    """
    t = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]

    T = t[1] - t[0]
    fs = 1.0 / T
    fN = fs / 2

    f, Puy = signal.csd(u, y, fs=fs, window='hann')
    _, Puu = signal.csd(u, u, fs=fs, window='hann')
    _, Pyy = signal.csd(y, y, fs =fs, window='hann')

    G = Puy / Puu
    Cuy = (np.abs(Puy)**2) / (Puu * Pyy + 1e-16)

    # Build mask
    mask = (
        (Cuy >= coh_thresh) &
        (Puu >= min_puu_frac * np.median(Puu)) &
        (f <= fmax_frac * fN)
    )

    return dict(
        f = (f[mask], f),
        omega = (2*np.pi*f[mask], 2*np.pi*f),
        G = (G[mask], G),
        Cuy = (Cuy[mask], Cuy),
        mask = mask,
        fs = fs,
        t = t,
        u = u,
        y = y,
        csd = (Puy, Puu, Pyy)
    )
  

def form_LS_system(G_csd, omega, m, n, weights=None):
    # Fit a tranfer function to discrete transfer function
    # ---------------------------------------------------- 
    # Degree of numerator, degree of denominator.

    # Number of numerator and denominator parameters.
    N_num = m + 1
    N_den = n
    Nw = omega.shape[0]

    #Weight Based on coherence
    if weights is None:
        W = np.ones(Nw)
    else:
        W = weights
    # Initialized A and b matrices that will be complex.
    b = -G_csd * (1j * omega)**n * W
    A = np.zeros((Nw, N_den + N_num), dtype=complex) 

    # Fill A matrix row by row 
    for i in range(N_num):
        A[:, i] = -(1j * omega)**(m - i) * W

    for j in range(N_den):
        A[:, j+N_num] = (1j * omega)**(n-j-1) * G_csd * W

    # Split A and b into real and complex parts. Stack real part on top of
    # complex part to create large A matrix.
    A1 = np.vstack([np.real(A), np.imag(A)])
    b1 = np.hstack([np.real(b), np.imag(b)])
    return A1, b1

def LS_fit(A1, b1, m, n, prin=True):

    # Number of numerator and denominator parameters.
    N_num = m + 1
    N_den = n

    x_lsq = np.linalg.lstsq(A1, b1, rcond=None)
    print(x_lsq)

    x = x_lsq[0]
    num_coeffs = x[:N_num]                 # [bm, bm-1, ..., b0] for s^m ... s^0
    den_coeffs = np.r_[1.0, x[N_num:]]     # [an-1, an-1, ..., a0]

    # Make TF (descending powers)
    G_LS = ct.tf(num_coeffs, den_coeffs)
    if prin:
        print("num =", num_coeffs)
        print("den =", den_coeffs)
        print("Pd_ID =", G_LS)

    return G_LS, x

def fit_error(A, b, x, m, n, prin=False):
    N = A.shape[0]
    e = b - A @ x
    e2 = np.dot(e,e) #sum of squared residuals

    # Compute the uncertainty and relative uncertainty. 
    cov_x = np.linalg.inv(A.T @ A) / (N - (n+m+1)) * e2
    sigma = np.sqrt(np.diagonal(cov_x))
    rel_unc = sigma / np.abs(x) * 100 

    # Compute the MSE, MSO, NMSE.
    MSE = e2 / N
    MSO = np.dot(b, b) / N
    NMSE = MSE / MSO

    if prin:
        print('The standard deviation is', sigma)
        print('The relative uncertainty is', rel_unc, '%\n')
        print('The MSE is', MSE)
        print('The MSO is', MSO)
        print('The NMSE is', NMSE, '\n')

    return dict(
        sigma=sigma,
        rel_unc=rel_unc,
        MSE=MSE,
        MSO=MSO,
        NMSE=NMSE
    )

def compare(x_train, data_test, m, n):
    res_test = signal_process(data_test)
    omega_test = res_test['omega'][0]
    G_test = res_test['G'][0]
    A_test, b_test = form_LS_system(G_test, omega_test, m, n)

    fit_error(A_test, b_test, x_train, m, n)

def response_error(data, G, prin=False, plot=False):

    t = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]
    N = t.size

    t_ID, y_ID = ct.forced_response(G, t, u)
    
    e = y_ID - y

    e_var = np.var(e, ddof=0)
    y_var = np.var(y, ddof=0)

    nmse_t = e_var / y_var
    VAF_test = (1 - e_var/y_var) * 100

    if prin:
        print('Time-domain NMSE:', nmse_t)
        print('The %VAF is', VAF_test)

    # Compute and plot errors
    e_abs = np.abs(e)
    y_max = np.max(np.abs(y))
    e_rel = e_abs / y_max

    if plot:
        # Plot test data
        fig, ax = plt.subplots(2, 1)   
        ax[0].set_ylabel(r'$u(t)$ (V)')
        ax[1].set_ylabel(r'$y(t)$ (LPM)')
        # Plot data
        ax[0].plot(t, u, '--', label='input', color='C0')
        ax[1].plot(t, y, label='output', color='C1')
        ax[1].plot(t_ID, y_ID, '-.', label="IDed output", color='C2')
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
            a.legend(loc='upper right')
        fig.tight_layout()

        # Plot error
        fig, ax = plt.subplots(2, 1)
        # Format axes
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
        ax[0].set_ylabel(r'$e_{abs}(t)$ (LPM)')
        ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
        # Plot data
        ax[0].plot(t, e)
        ax[1].plot(t, e_rel)
        # for a in np.ravel(ax):
        #     a.legend(loc='lower right')
        fig.tight_layout()
    
    return float(nmse_t), float(VAF_test)

def response_error_std(data, G, plot=False):
    t = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]
    N = t.size

    # standardize data
    u_mean = np.mean(u)
    u_std = np.std(u)
    y_mean =np.mean(y)
    y_std = np.std(y)

    us = (u - u_mean)/u_std
    ys = (y - y_mean)/y_std
    # SCALE THE MODEL to standardized units
    Gs = (u_std / (y_std + 1e-16)) * G
    t_ID, y_ID = ct.forced_response(Gs, t, us)
    
    e = y_ID - ys
    e_var = np.var(e, ddof=0)
    y_var = np.var(ys, ddof=0)

    VAF_test = (1 - e_var/y_var) * 100
    print('The %VAF is', VAF_test)

    # Compute and plot errors
    e_abs = np.abs(e)
    y_max = np.max(np.abs(ys))
    e_rel = e_abs / y_max

    if plot:
        # Plot test data
        fig, ax = plt.subplots(2, 1)   
        ax[0].set_ylabel(r'$u(t)$ (V)')
        ax[1].set_ylabel(r'$y(t)$ (LPM)')
        # Plot data
        ax[0].plot(t, us, '--', label='input', color='C0')
        ax[1].plot(t, ys, label='output', color='C1')
        ax[1].plot(t_ID, y_ID, '-.', label="IDed output", color='C2')
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
            a.legend(loc='upper right')
        fig.tight_layout()

        # Plot error
        fig, ax = plt.subplots(2, 1)
        # Format axes
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
        ax[0].set_ylabel(r'$e_{abs}(t)$ (LPM)')
        ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
        # Plot data
        ax[0].plot(t, e)
        ax[1].plot(t, e_rel)
        # for a in np.ravel(ax):
        #     a.legend(loc='lower right')
        fig.tight_layout()
    
    return float(VAF_test)

