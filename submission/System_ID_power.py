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

# Extract time
t = data_0[:, 0]
N = t.size
T = t[1] - t[0]
fs = 1/T
fN = fs/2

# Extract input and output
u = data_0[:, 1]  # V, volts
y = data_0[:, 2]  # LPM, liters per minute

# %%
# Signal Power
# ------------------------------------------------------------------

# # Compute Power Spectral Density using periodogram
# u_psd, psd = signal.periodogram(u, fs=fs, window='hann')
# signal.get_window(window='hann', Nx=N, fftbins=False)

# # Plot PSD
# fig, ax = plt.subplots()
# ax.semilogy(u_psd, psd)
# ax.set_xlabel(r'$\omega$ (Hz)')
# ax.set_ylabel(r'PSD')

# Compute PSD of uu and uy
f, Puy = signal.csd(u, y, fs=fs, window='hann')
_, Puu = signal.csd(u, u, fs=fs, window='hann')
_, Pyy = signal.csd(y, y, fs =fs, window='hann')

omega_org = f * 2 * np.pi   # the frequencies in rad/s

Cuy = np.abs(Puy)**2 / Puu / Pyy #Coherence 
# Cuy=1 indicates linear relation between u and y
# Cuy<1 indicates noise or non-linearities (quantifies how non-linear)
# Cuy=0 indicates no relation between u and y

# %%
# Compute TF using PSD
G_csd = Puy/Puu # array of complex numbers

mask = (Cuy >= 0.8) & (Puu >= 1e-3*np.median(Puu)) & (f <= 0.9*fN)

# Filter out points with excessive noise (non-linearity)
G_csd = G_csd[mask]
fcsd = f[mask]
omega = omega_org[mask]

#Plotting Units
G_csd_mag = np.sqrt((np.real(G_csd))**2 + (np.imag(G_csd))**2)  # absolute
G_csd_phase = np.arctan2(np.imag(G_csd), np.real(G_csd))  # rad

# Plotting units
G_csd_mag_dB = 20 * np.log10(G_csd_mag)
G_csd_phase_deg = G_csd_phase * 360 / 2 / np.pi

# %%
# Plot CSD Bode plot

# Plot input and output CSD
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.semilogx(f, 20 * np.log10(Puu), '.', color='C0', label=r'$|Puu(j\omega)|$')
ax.semilogx(f, 20 * np.log10(Puy), '.', color='C1', label=r'$|Puy(j\omega)|$')
ax.semilogx(f, 20 * np.log10(Pyy), '.', color='C3', label=r'$|Pyy(j\omega)|$')
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower left')
fig.tight_layout()
plt.show()

# Plot input and output CSD filtering out noisy points
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.semilogx(fcsd, 20 * np.log10(Puu[mask]), '.', color='C0', label=r'$|Puu(j\omega)|$')
ax.semilogx(fcsd, 20 * np.log10(Puy[mask]), '.', color='C1', label=r'$|Puy(j\omega)|$')
ax.semilogx(fcsd, 20 * np.log10(Pyy[mask]), '.', color='C3', label=r'$|Pyy(j\omega)|$')
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower left')
fig.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
# Magnitude plot
axes[0].semilogx(fcsd, G_csd_mag_dB, '.', color='C3', label='System ID')
axes[0].set_yticks(np.arange(-80, 20, 20))
axes[0].set_xlabel(r'$\omega$ (Hz)')
axes[0].set_ylabel(r'Magnitude (dB)')
axes[0].legend(loc='best')
# Phase plot
axes[1].semilogx(fcsd, G_csd_phase_deg, '.', color='C3', label='System ID')
axes[1].set_yticks(np.arange(-90, 210, 30))
axes[1].set_xlabel(r'$\omega$ (Hz)')
axes[1].set_ylabel(r'Phase (deg)')
# fig.savefig(f'figs/system_ID_freq_resp.pdf'


# %%
# Fit a tranfer function to discrete transfer function
# ----------------------------------------------------

# Degree of numerator, degree of denominator.
m, n = 0, 1

# Number of numerator and denominator parameters.
N_num = m + 1
N_den = n
Nw = omega.shape[0]

#Weight Based on coherence
W = np.sqrt(Cuy[mask])
# Initialized A and b matrices that will be complex.
b = -G_csd * (1j * omega)**n 
A = np.zeros((Nw, N_den + N_num), dtype=complex) 

# Fill A matrix row by row 
for i in range(N_num):
    A[:, i] = -(1j * omega)**(m - i) 

for j in range(N_den):
    A[:, j+N_num] = (1j * omega)**(n-j-1) * G_csd 

# Split A and b into real and complex parts. Stack real part on top of
# complex part to create large A matrix.
A1 = np.vstack([np.real(A), np.imag(A)])
b1 = np.hstack([np.real(b), np.imag(b)])

x_lsq = np.linalg.lstsq(A1, b1, rcond=None)
print(x_lsq)

x = x_lsq[0]
num_coeffs = x[:N_num]                 # [bm, bm-1, ..., b0] for s^m ... s^0
den_coeffs = np.r_[1.0, x[N_num:]]     # [an-1, an-1, ..., a0]

# Make TF (descending powers)
G_LS = ct.tf(num_coeffs, den_coeffs)
print("num =", num_coeffs)
print("den =", den_coeffs)
print("Pd_ID =", G_LS)

# Bode plot of IDed tf
w_shared = np.logspace(-2, 2, 2000)  # rad/s
# [1e-2, 1e2]
f_shared_Hz = w_shared / 2 / np.pi

mag_est, phase_est, _ = ct.frequency_response(G_LS, w_shared)

# Convert to dB and deg.
mag_est_dB = 20 * np.log10(mag_est)
phase_est_deg = phase_est / np.pi * 180

# Plot the Bode plot of the IDed system
# Plot CSD Bode plot
fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
# Magnitude plot
axes[0].semilogx(fcsd, G_csd_mag_dB, '.', color='C3', label='System ID')
axes[0].semilogx(f_shared_Hz, mag_est_dB, '-', color="C2", label='LS fit')
axes[0].set_yticks(np.arange(-80, 20, 20))
axes[0].set_xlabel(r'$\omega$ (Hz)')
axes[0].set_ylabel(r'Magnitude (dB)')
axes[0].legend(loc='best')
# Phase plot
axes[1].semilogx(fcsd, G_csd_phase_deg, '.', color='C3')
axes[1].semilogx(f_shared_Hz, phase_est_deg, '-', color="C2")
axes[1].set_yticks(np.arange(-90, 210, 30))
axes[1].set_xlabel(r'$\omega$ (Hz)')
axes[1].set_ylabel(r'Phase (deg)')

# %%
# Bode with control library
fig, ax = plt.subplots(2, 1)
ct.bode_plot(G_LS, w_shared, dB=True, deg=True, title='')
ax[0].semilogx(omega, G_csd_mag_dB, '.', color='C3', label='System ID')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitutde (dB)')
ax[1].semilogx(omega, G_csd_phase_deg, '.', color='C3', label='System ID')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
fig.tight_layout()

