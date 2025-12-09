"""Step responses, to be used to understand noise. 

J R Forbes, 2025/09/10
"""

# %%
# Libraries
import numpy as np
# import control
# from scipy import signal
from matplotlib import pyplot as plt
# from scipy import fft
# from scipy import integrate
import pathlib


# %%
# Plotting parameters
# plt.rc('text', usetex=True)
# plt.rc('font', family='serif', size=14)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %%
# Common parameters

# Conversion
rps2Hz = lambda w: w / 2 / np.pi
Hz2rps = lambda w: w * 2 * np.pi

# Golden ratio
gr = (1 + np.sqrt(5)) / 2

# Figure height
height = 4.25


# %%
# Read in all input-output (IO) data
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
data = np.array(data)

# %%
# 

N_data = data.shape[0]
max_input_output_std = np.zeros((N_data, 7))

for i in range(N_data): # N_data
    # Data
    data_read = data[i, :, :]

    t_full = data_read[:, 0]
    target_time = 0  # s
    t_start_index = np.argmin(np.abs(t_full - target_time))

    # Extract time
    t = data_read[t_start_index:-1, 0]
    # N = t.size
    T = t[1] - t[0]

    # Extract input and output
    u_raw = data_read[t_start_index:-1, 1]  # V, volts
    y_raw = data_read[t_start_index:-1, 2]  # LMP, force

    # Plotting
    # Plot raw data time domain
    fig, ax = plt.subplots(2, 1)
    fig.set_size_inches(height * gr, height, forward=True)
    ax[0].plot(t, u_raw)
    ax[1].plot(t, y_raw)
    ax[0].set_xlabel(r'$t$ (s)')
    ax[1].set_xlabel(r'$t$ (s)')
    ax[0].set_ylabel(r'$\tilde{u}(t)$ (V)')
    ax[1].set_ylabel(r'$\tilde{y}(t)$ (LPM)')
    fig.tight_layout()
    plt.show()
    # fig.savefig('x.pdf')


# %%
