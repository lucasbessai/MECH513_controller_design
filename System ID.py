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
# FFT of the input and output

# Single-sided FFT of input
u_fft = fft.rfft(u, n=N) / N  # same units as u
u_mag = np.abs(u_fft)  # compute the magnitude of each u_fft
u_mag[1:] = 2 * u_mag[1:]  # multiply all mag's by 2, but the zero frequency
u_phase = np.angle(u_fft, deg=False)  # compute the angle

f = fft.rfftfreq(N, d=T)  # the frequencies in Hz
N_f_max = np.searchsorted(f, np.max(f))  # find index of max frequency
omega = f * 2 * np.pi   # the frequencies in rad/s
w_shared = omega[:N_f_max]

# Recompute the FFT with the correct scaling 
# Scaled vesion of the array out of fft.rfft considers factor of 2 from the left side of the transform
# Will cancel in the division to make G(s)
u_FFT = np.zeros(N_f_max, dtype=complex)
for i in range(N_f_max):
    u_FFT[i] = u_mag[i] * np.cos(u_phase[i]) + 1j * u_mag[i] * np.sin(u_phase[i])

# You must now compute y_FFT
y_fft = fft.rfft(y, n=N) / N
y_mag = np.abs(y_fft)
y_mag[1:] = 2 * y_mag[1:]
y_phase = np.angle(y_fft, deg=False)

# Rescale to conserve energy
y_FFT = np.zeros(N_f_max, dtype=complex)
for i in range(N_f_max):
    y_FFT[i] = y_mag[i] * np.cos(y_phase[i]) + 1j * y_mag[i] * np.sin(y_phase[i])

# %%
# Signal Power
# # Compute Power Spectral Density using periodogram
# u_psd, psd = signal.periodogram(u, fs=fs, window='hann')
# signal.get_window(window='hann', Nx=N, fftbins=False)

# # Plot PSD
# fig, ax = plt.subplots()
# ax.semilogy(u_psd, psd)
# ax.set_xlabel(r'$\omega$ (Hz)')
# ax.set_ylabel(r'PSD')

# Compute PSD of uu and uy
fcsd, Puy = signal.csd(u, y, fs=fs, window='hann')
fcsd, Puu = signal.csd(u, u, fs=fs, window='hann')
fcsd, Pyy = signal.csd(y, y, fs =fs, window='hann')

Cuy = np.abs(Puy)**2 / Puu / Pyy #Coherence 
# Cuy=1 indicates linear relation between u and y
# Cuy=0 indicates no relation between u and y


# %% 
# Compute TF using dft
G_div = y_fft/u_fft
G_mag = np.sqrt((np.real(G_div))**2 + (np.imag(G_div))**2)  # absolute
G_phase = np.arctan2(np.imag(G_div), np.real(G_div))  # rad

# Plotting units
G_mag_dB = 20 * np.log10(G_mag)
G_phase_deg = G_phase * 360 / 2 / np.pi

# %%
# Compute TF using PSD
G_csd = Puy/Puu # array of complex numbers

mask = (Cuy >= 0.8) & (Puu >= 1e-3*np.median(Puu)) & (fcsd <= 0.9*fN)

G_csd = G_csd[mask]
fcsd = 2*np.pi*fcsd[mask]

#Plotting Units
G_csd_mag = np.sqrt((np.real(G_csd))**2 + (np.imag(G_csd))**2)  # absolute
G_csd_phase = np.arctan2(np.imag(G_csd), np.real(G_csd))  # rad

# Plotting units
G_csd_mag_dB = 20 * np.log10(G_csd_mag)
G_csd_phase_deg = G_csd_phase * 360 / 2 / np.pi

# %%
# Plots

# Plot FFT of input
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.semilogx(f[:N_f_max], 20 * np.log10(u_mag[:N_f_max]), '.', color='C0', label=r'$|u(j\omega)|$')
ax.semilogx(f[:N_f_max], 20 * np.log10(y_mag[:N_f_max]), '.', color='C1', label=r'$|y(j\omega)|$')
ax.set_xlabel(r'$\omega$ (Hz)')
ax.set_ylabel(r'Magnitude (dB)')
ax.legend(loc='lower left')
fig.tight_layout()
plt.show()
# fig.savefig(f'figs/IO_freq_resp.pdf')

# Plot Bode plot
fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
# Magnitude plot
axes[0].semilogx(f[:N_f_max], G_mag_dB[:N_f_max], '.', color='C3', label='System ID')
axes[0].set_yticks(np.arange(-80, 20, 20))
axes[0].set_xlabel(r'$\omega$ (Hz)')
axes[0].set_ylabel(r'Magnitude (dB)')
axes[0].legend(loc='best')
# Phase plot
# axes[1].semilogx(f_shared_Hz, phase_G_deg, color='C2', label='True')
axes[1].semilogx(f[:N_f_max], G_phase_deg[:N_f_max], '.', color='C3', label='System ID')
axes[1].set_yticks(np.arange(-90, 210, 30))
axes[1].set_xlabel(r'$\omega$ (Hz)')
axes[1].set_ylabel(r'Phase (deg)')
# fig.savefig(f'figs/system_ID_freq_resp.pdf'

# %%
# Plot CSD Bode plot
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
# Degree of numerator, degree of denominator.
m, n = 0, 1

# Number of numerator and denominator parameters.
N_num = m + 1
N_den = n
len = omega.shape[0]
# Initialized A and b matrices that will be complex.
b = -G_div * (1j * omega)**n
A = np.zeros((len, N_den + N_num)) 

# Fill A matrix row by row 
for i in range(N_num):
    A[:, i] = -(1j * omega)**(N_num - (i+1)) 

for j in range(N_den):
    A[:, j+N_num] = (1j * omega)**(n-j-1) * G_div

# Split A and b into real and complex parts. Stack real part on top of
# complex part to create large A matrix.
A1 = np.vstack([np.real(A), np.imag(A)])
b1 = np.hstack([np.real(b), np.imag(b)])

x_lsq = np.linalg.lstsq(A1, b1, rcond=None)
print(x_lsq)

x = x_lsq[0]
num_coeffs = x[:N_num][::-1]                 # [b0, b1, ..., bm] for s^m ... s^0
den_coeffs = np.r_[1.0, x[N_num:][::-1]]     # [1, a, ..., dn]

# Make TF (descending powers)
G_LS = ct.tf(num_coeffs, den_coeffs)
print("num =", num_coeffs)
print("den =", den_coeffs)
print("Pd_ID =", G_LS)

# %%
# Bode with control library
w_shared = np.logspace(-2, 2, 2000)  # rad/s
# [1e-2, 1e2]
f_shared_Hz = w_shared / 2 / np.pi

fig, ax = plt.subplots(2, 1)
ct.bode_plot(G_LS, w_shared, dB=True, deg=True, title='')
ax[0].semilogx(omega[:N_f_max], G_mag_dB[:N_f_max], '.', color='C3', label='System ID')
ax[0].set_xlabel(r'$\omega$ (rad/s)')
ax[0].set_ylabel(r'Magnitutde (dB)')
ax[1].semilogx(omega[:N_f_max], G_phase_deg[:N_f_max], '.', color='C3', label='System ID')
ax[1].set_xlabel(r'$\omega$ (rad/s)')
ax[1].set_ylabel(r'Phase (deg)')
fig.tight_layout()

# %%
# Bode plot of IDed tf
mag_est, phase_est, _ = ct.frequency_response(G_LS, w_shared)

# Convert to dB and deg.
mag_est_dB = 20 * np.log10(mag_est)
phase_est_deg = phase_est / np.pi * 180

# Plot the Bode plot of the IDed system
# Plot CSD Bode plot
fig, axes = plt.subplots(2, 1, figsize=(height * gr, height))
# Magnitude plot
axes[0].semilogx(f[:N_f_max], G_mag_dB[:N_f_max], '.', color='C3', label='System ID')
axes[0].semilogx(f_shared_Hz, mag_est_dB, '-', color="C2", label='LS fit')
axes[0].set_yticks(np.arange(-80, 20, 20))
axes[0].set_xlabel(r'$\omega$ (Hz)')
axes[0].set_ylabel(r'Magnitude (dB)')
axes[0].legend(loc='best')
# Phase plot
axes[1].semilogx(f[:N_f_max], G_phase_deg[:N_f_max], '.', color='C3')
axes[1].semilogx(f_shared_Hz, phase_est_deg, '-', color="C2")
axes[1].set_yticks(np.arange(-90, 210, 30))
axes[1].set_xlabel(r'$\omega$ (Hz)')
axes[1].set_ylabel(r'Phase (deg)')

# %% 
# 