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
import uncertainty_characterization
import nominal_plant
import freq_domain_sysID
import System_ID_power
import System_ID_Functions_table

# %% 
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
# Driver

freq_domain_sysID.quantify_error(datasets, m=0, n=1)

off_noms = nominal_plant.off_nominals(datasets, m=0, n=1, plot=False)
G_nom = nominal_plant.nominal_plant_median(datasets, m=0, n=1, plot=True)

sysid.response_error(data_0, G_nom, plot=True)
sysid.response_error(data_1, G_nom, plot=True)
sysid.response_error(data_2, G_nom, plot=True)
sysid.response_error(data_3, G_nom, plot=True)
# %%
