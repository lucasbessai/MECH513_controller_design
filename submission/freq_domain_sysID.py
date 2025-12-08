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

#%%

def quantify_error(datasets, m, n):

    def VAF(data, G):

        t = data[:, 0]
        u = data[:, 1]
        y = data[:, 2]
        N = t.size

        t_ID, y_ID = ct.forced_response(G, t, u)
        
        e = y_ID - y

        e_var = np.var(e, ddof=0)
        y_var = np.var(y, ddof=0)

        nmse_t = e_var / y_var
        VAF_test = (1 - nmse_t) * 100

        print('Time-domain NMSE:', nmse_t)
        print('The %VAF is', VAF_test)
        
    for D, i in zip(datasets, [0,1,2,3]):
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

        print(f'train: {i} test {i}, results: ')
        VAF(D, G_LS)
        
        for j, D_test in enumerate(datasets):
            if j == i: 
                continue
            print(f'train: {i}, test{j} results: ')
            VAF(D_test, G_LS)

# quantify_error(datasets, m=0, n=1)
# A, b = sysid.form_LS_system(G, omega, m, n)
# G_LS, x = sysid.LS_fit(A, b, m, n)
# sysid.fit_error(A, b, x, m, n)
# sysid.compare(x, data_1, m, n)
# sysid.compare(x, data_2, m, n)
# sysid.compare(x, data_3, m, n)

# # %%
# sysid.response_error(data_0, G_LS, plot=False)
# sysid.response_error_std(data_0, G_LS, plot=False)

# # %%
# Puy, Puu, Pyy = csd[0], csd[1], csd[2]
 
# #Plotting Units
# G_mag = np.sqrt((np.real(G))**2 + (np.imag(G))**2)  # absolute
# G_phase = np.arctan2(np.imag(G), np.real(G))  # rad

# # Plotting units
# G_csd_mag_dB = 20 * np.log10(G_mag)
# G_csd_phase_deg = G_phase * 360 / 2 / np.pi


# %%
# Quantify error 


# Quantify (m,n) quality: train-on-i, test-on-others, freq + time metrics

def quantify_error_table(datasets, m, n,
                         coh_thresh=0.8, min_puu_frac=1e-3, fmax_frac=0.9,
                         use_weights=False, include_train_row=True,
                         standardize_time=False):
    """
    Uses only your sysid functions:
      - signal_process (masked band, coherence)
      - form_LS_system (with optional sqrt(coherence) weights)
      - LS_fit
      - fit_error  (freq-domain MSE/MSO/NMSE + sigma/rel_unc)
      - response_error (time-domain NMSE,VAF)  [standardize via response_error_std if requested]
    Prints one row per (train,test) combo and returns a list of dicts.
    """
    packs = [sysid.signal_process(D, coh_thresh=coh_thresh,
                                  min_puu_frac=min_puu_frac, fmax_frac=fmax_frac)
             for D in datasets]

    rows = []
    line = "-" * 120
    print(line)
    print(f"{'Tr':>2} {'Te':>2} | "
          f"{'m':>2} {'n':>2} | "
          f"{'F_MSE':>10} {'F_NMSE':>8} {'σ_max':>8} {'rel%_max':>9} | "
          f"{'NMSE_t':>8} {'VAF%':>8}")
    print(line)

    N = len(datasets)
    for i, D_train in enumerate(datasets):
        # --- build TRAIN spectral data on its masked band ---
        pack_i = packs[i]
        G_i   = pack_i['G'][0]
        w_i   = pack_i['omega'][0]
        W_i   = np.sqrt(pack_i['Cuy'][0]) if use_weights else None

        A1_i, b1_i = sysid.form_LS_system(G_i, w_i, m, n, weights=W_i)
        G_fit, x_i = sysid.LS_fit(A1_i, b1_i, m, n, prin=False)

        # freq metrics on TRAIN (no print)
        fe_i = sysid.fit_error(A1_i, b1_i, x_i, m, n, prin=False)

        # time metrics on TRAIN
        if standardize_time:
            # reuse your std function for VAF; compute NMSE_var from residuals/var(ys)
            VAF_train = sysid.response_error_std(D_train, G_fit, plot=False)
            nmse_t_train = 1.0 - VAF_train/100.0
        else:
            nmse_t_train, VAF_train = sysid.response_error(D_train, G_fit, prin=False, plot=False)

        if include_train_row:
            row = {
                'train': i, 'test': i, 'm': m, 'n': n,
                'F_MSE': fe_i['MSE'], 'F_NMSE': fe_i['NMSE'],
                'SIGMA_MAX': float(np.max(fe_i['sigma'])),
                'RELUNC_MAX': float(np.max(fe_i['rel_unc'])),
                'NMSE_t': float(nmse_t_train),
                'VAF': float(VAF_train),
            }
            rows.append(row)
            print(f"{i:>2} {i:>2} | {m:>2} {n:>2} | "
                  f"{row['F_MSE']:10.3f} {row['F_NMSE']:8.3f} {row['SIGMA_MAX']:8.3f} {row['RELUNC_MAX']:9.2f} | "
                  f"{row['NMSE_t']:8.3f} {row['VAF']:8.2f}")

        # --- TEST on other sets using same G_fit/x_i ---
        for j, D_test in enumerate(datasets):
            if j == i: 
                continue
            pack_j = packs[j]
            G_j  = pack_j['G'][0]
            w_j  = pack_j['omega'][0]
            W_j  = np.sqrt(pack_j['Cuy'][0]) if use_weights else None

            # freq residuals on TEST band with TRAIN parameters
            A1_j, b1_j = sysid.form_LS_system(G_j, w_j, m, n, weights=W_j)
            fe_j = sysid.fit_error(A1_j, b1_j, x_i, m, n, prin=False)

            # time metrics on TEST
            if standardize_time:
                VAF_test = sysid.response_error_std(D_test, G_fit, plot=False)
                nmse_t_test = 1.0 - VAF_test/100.0
            else:
                nmse_t_test, VAF_test = sysid.response_error(D_test, G_fit, prin=False, plot=False)

            row = {
                'train': i, 'test': j, 'm': m, 'n': n,
                'F_MSE': fe_j['MSE'], 'F_NMSE': fe_j['NMSE'],
                'SIGMA_MAX': float(np.max(fe_j['sigma'])),
                'RELUNC_MAX': float(np.max(fe_j['rel_unc'])),
                'NMSE_t': float(nmse_t_test),
                'VAF': float(VAF_test),
            }
            rows.append(row)
            print(f"{i:>2} {j:>2} | {m:>2} {n:>2} | "
                  f"{row['F_MSE']:10.3f} {row['F_NMSE']:8.3f} {row['SIGMA_MAX']:8.3f} {row['RELUNC_MAX']:9.2f} | "
                  f"{row['NMSE_t']:8.3f} {row['VAF']:8.2f}")

    print(line)
    print(f"(m,n)=({m},{n}), weights={'√coh' if use_weights else 'none'}, "
          f"mask: Cuy≥{coh_thresh}, Puu≥{min_puu_frac}·median, f≤{fmax_frac}·fN")
    return rows

# %% 
# Print Quantify Error Table
# _ = quantify_error_table(datasets, m=1, n=3)
# %%
