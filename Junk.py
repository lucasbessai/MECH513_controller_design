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

res = sysid.signal_process(data_0)

omega = res['omega']
G = res['G']
Cuy = res['Cuy']
mask = res['mask']
fs = res['fs']
t = res['t']
u = res['u']
y = res['y']
csd = res['csd']

G_csd = G[0]
Puy, Puu, Pyy = csd[0], csd[1], csd[2]


# %%

def evaluate_mn_table(
        datasets,              # list like [data_0, data_1, data_2, data_3]
        m=0, n=1,
        coh_thresh=0.8, min_puu_frac=1e-3, fmax_frac=0.9,
        # LS options (kept minimal)
        use_weights=True,
        include_train_row=True  # print the train-on/train-set row too
    ):
    """
    For each dataset as TRAIN, fit (m,n), then TEST on the other 3.
    Prints a table with time-domain (NMSE,VAF) and frequency-domain metrics.
    """
    # Precompute freq-domain data for all sets
    packs = []
    for D in datasets:
        sp = signal_process(
            D, 
            coh_thresh=coh_thresh, min_puu_frac=min_puu_frac, fmax_frac=fmax_frac
        )
        packs.append(sp)

    # Header
    line = "-" * 108
    print(line)
    print(f"{'Train':>5} {'Test':>5}  {'NMSE_t':>8} {'VAF%':>8}   {'NMSE_f':>8}  {'RMSE|.|dB':>10} {'RMSE∠ deg':>10}  {'MAE|.|dB':>9} {'MAX|.|dB':>9}")
    print(line)

    # Loop over train sets
    N = len(datasets)
    for i in range(N):
        # --- Fit on TRAIN=i ---
        Gi   = packs[i]['G']
        wi   = packs[i]['omega']
        Cuyi = packs[i]['Cuy']
        W    = np.sqrt(Cuyi) if use_weights else None

        A1, b1, scales = form_LS_system(Gi, wi, m, n, weights=W)
        G_fit, _ = LS_fit(A1, b1, m, n, scales=scales)

        # Optionally include the train row itself first
        if include_train_row:
            nmse_t, vaf_t = _time_nmse_vaf(datasets[i], G_fit)
            fmet = _freq_errors(G_fit, wi, Gi)
            print(f"{i:5d} {i:5d}  {nmse_t:8.3f} {vaf_t:8.2f}   {fmet['NMSE_freq']:8.3f}  {fmet['RMSE_mag_dB']:10.2f} {fmet['RMSE_phase_deg']:10.2f}  {fmet['MAE_mag_dB']:9.2f} {fmet['MAX_mag_dB']:9.2f}")

        # --- Test on other sets ---
        for j in range(N):
            if j == i:
                continue
            Gj  = packs[j]['G']
            wj  = packs[j]['omega']

            # time-domain metrics on raw time series
            nmse_t, vaf_t = _time_nmse_vaf(datasets[j], G_fit)

            # frequency-domain metrics on that test's masked band
            fmet = _freq_errors(G_fit, wj, Gj)

            print(f"{i:5d} {j:5d}  {nmse_t:8.3f} {vaf_t:8.2f}   {fmet['NMSE_freq']:8.3f}  {fmet['RMSE_mag_dB']:10.2f} {fmet['RMSE_phase_deg']:10.2f}  {fmet['MAE_mag_dB']:9.2f} {fmet['MAX_mag_dB']:9.2f}")

    print(line)
    print(f"(m, n) = ({m}, {n}), weights={'coh' if use_weights else 'none'}")

    # Suppose you already loaded:
# data_0, data_1, data_2, data_3 = ...
datasets = [data_0, data_1, data_2, data_3]

# First-order model:
evaluate_mn_table(
    datasets,
    m=0, n=1,
    coh_thresh=0.8, fmax_frac=0.9,
    use_weights=False,
    include_train_row=True
)

# %%
