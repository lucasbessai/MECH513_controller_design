# %%
# Main file to generate nominal plant and uncertainty bounds
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
import unc_bound

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
# Driver - Generate nominal plant and uncertainty bounds

# System identification parameters
m, n = 0, 1

# Quantify error for different datasets
print("=== Quantifying error ===")
freq_domain_sysID.quantify_error(datasets, m=m, n=n)

# Generate off-nominal plants (first-order)
print("\n=== Generating off-nominal plants (first-order) ===")
off_noms = nominal_plant.off_nominals(datasets, m=m, n=n, plot=False)

# Add a second-order off-nominal plant
print("\n=== Adding second-order off-nominal plant ===")
# Fit a second-order plant from the first dataset (you can change this to any dataset)
#G_2nd_order = nominal_plant.fit_plant_from_data(data_0, m=0, n=3, label='G_2nd_order')
#print(f"Second-order plant: {G_2nd_order[1]}")
G_3rd_order_nominal = nominal_plant.fit_plant_from_data(data_0, m=0, n=3, label='G_3rd_order_nominal')

# Generate nominal plant (median of first-order off-nominals)
# Include the second-order plant in the plot
print("\n=== Generating nominal plant ===")
G_nom = nominal_plant.nominal_plant_median(datasets, m=m, n=n, plot=True, additional_plants=[G_3rd_order_nominal])

# Compute response errors for each dataset
print("\n=== Computing response errors ===")
sysid.response_error(data_0, G_nom, plot=True)
sysid.response_error(data_1, G_nom, plot=True)
sysid.response_error(data_2, G_nom, plot=True)
sysid.response_error(data_3, G_nom, plot=True)

# Generate uncertainty bounds
print("\n=== Generating uncertainty bounds ===")
G_off_nom = [G[1] for G in off_noms]
# Add the second-order plant to the off-nominal list for uncertainty bounds
G_off_nom.append(G_3rd_order_nominal[1])
print(f"Total off-nominal plants for uncertainty bounds: {len(G_off_nom)} (including second-order)")

uncertainty_results = uncertainty_characterization.compute_uncertainty_bounds(
    G_nom, G_off_nom, w_shared=None, deg=3
)

# Print summary
print("\n=== Summary ===")
print(f"Nominal plant: {G_nom}")
print(f"Number of off-nominal plants: {len(G_off_nom)}")
print(f"Uncertainty bound types computed: {list(uncertainty_results.keys())}")

# Print W2 transfer functions
print("\n=== W2 Uncertainty Bound Transfer Functions ===")
for bound_type, results in uncertainty_results.items():
    print(f"\n{bound_type.replace('_', ' ').title()}:")
    print(f"  W2 = {results['W2']}")
    print(f"  Slack: {results['slack']:.4f} dB")
    print(f"  Ratio: {results['ratio']:.4f}")

# Display all plots
plt.show()
# %%
