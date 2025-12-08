# %%
# libraries 
import numpy as np
import control
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

t = data_0[:, 0]
N = t.size
T = t[1] - t[0]

# Extract input and output
u = data_0[:, 1]  # V, volts
y = data_0[:, 2]  # LPM, liters per minute

# Hanning Window
h = signal.get_window(window='hann', Nx=N, fftbins=False)


# Plot signals multiplied by sindow
fig, ax = plt.subplots(2, 1)
ax[0].set_ylabel(r'$u(t)$ (V)')
ax[1].set_ylabel(r'$y(t)$ (LPM)')
# Plot data
ax[0].plot(t, u * h, label='input', color='C0')
ax[1].plot(t, y * h, label='output', color='C1')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')
    a.legend(loc='upper right')
fig.tight_layout()
