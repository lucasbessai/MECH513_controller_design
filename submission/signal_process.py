# %%
# Libraries
import numpy as np
# import control
# from scipy import signal
from matplotlib import pyplot as plt
import pathlib
from scipy import signal, fft
from scipy.interpolate import interp1d

# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')
# path = pathlib.Path('figs')
# path.mkdir(exist_ok=True)

# Conversion
rps2Hz = lambda w: w / 2 / np.pi
Hz2rps = lambda w: w * 2 * np.pi

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25

# %%
# Read in all input-output (IO) data FORCED
path = pathlib.Path('SINE_SWEEP_DATA/')
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
data_forced = np.array(data)

# %%
# Read in all input-output (IO) data NOISE
path = pathlib.Path('DATA_noise/')
all_files = sorted(path.glob("*.csv"))
# all_files.sort()
data = [
    np.loadtxt(
        filename,
        dtype=float,
        delimiter=',',
        skiprows=1,
        usecols=(0, 1, 2),
        # max_rows=1100,
    ) for filename in all_files
]
data_noise = np.array(data)

# %% 
# PSD 

N_data = data_forced.shape[0]

fig_n_PSD, ax_n_PSD = plt.subplots()
fig_coh, ax_coh = plt.subplots()

COH_chirp = []
Pyy_noise = []
f_f_list = []
f_n_list = []

# First pass: compute PSDs and collect frequency arrays
for i in range(N_data):
    data_read_f = data_forced[i, :, :]

    t = data_read_f[:, 0]
    u = data_read_f[:, 1]
    y = data_read_f[:, 2]

    T = t[1] - t[0]
    fs = 1.0 / T
    fN = fs / 2

    f_f, COH_f = signal.coherence(u, y, fs =fs, window='hann')
    f_f_list.append(f_f)
    COH_chirp.append(COH_f)

    data_read_n = data_noise[i, :, :]

    t = data_read_n[:, 0]
    u = data_read_n[:, 1]
    y = data_read_n[:, 2]

    y = y[t>4]

    T = t[1] - t[0]
    fs = 1.0 / T
    fN = fs / 2

    
    f_n, Pyy_n = signal.welch(y, fs =fs, window='hann')
    f_n_list.append(f_n)
    Pyy_noise.append(Pyy_n)

    ax_coh.plot(f_f, COH_f, color='C0', alpha=0.3)
    ax_n_PSD.semilogy(f_n, Pyy_n, color='C1', alpha=0.3)

# plt.show()

Pyy_noise_avg = np.mean(Pyy_noise, axis=0)
COH_chirp_avg = np.mean(COH_chirp, axis=0)
COH_threshold = 0.8

# Plot PSD of signal+noise and noise (on common frequency grid)
# fig, ax = plt.subplots()
fig_n_PSD.set_size_inches(height * gr, height, forward=True)
ax_n_PSD.semilogy(f_n, Pyy_noise_avg, color='C1', label='Noise (avg)')
ax_n_PSD.vlines(0.015, 0, 1, transform=ax_n_PSD.get_xaxis_transform(), color='r', linestyle='--', label=r'$f_r$')
ax_n_PSD.set_xlabel('Frequency [Hz]')
ax_n_PSD.set_ylabel(r'PSD [$y^2/Hz$]')
# ax.set_title('PSD of LPM and noise')
ax_n_PSD.legend()

fig_coh.set_size_inches(height * gr, height, forward=True)
ax_coh.plot(f_f, COH_chirp_avg, color='C0', label=r'$C_{uy}$ (avg)')
ax_coh.vlines(0.015, 0, 1, transform=ax_coh.get_xaxis_transform(), color='r', linestyle='--', label=r'$f_r$')
ax_coh.hlines(COH_threshold, f_f[0], f_f[-1], color='b', linestyle='--', label=r'$C_{uy}(f) = 0.8$')
ax_coh.set_xlabel('Frequency [Hz]')
ax_coh.set_ylabel('Coherence (unitless)')
ax_coh.legend()


# %%
