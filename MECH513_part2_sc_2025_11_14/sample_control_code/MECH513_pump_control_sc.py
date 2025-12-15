"""Pump control sample code, MECH 513.

James Forbes
2025/11/06
"""
# %%
# Packages
import numpy as np
from matplotlib import pyplot as plt
from scipy import integrate
# from scipy import signal
from scipy.stats import norm
import control
import pathlib
import dkpy

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

# Laplace variable
s = control.tf('s')

# Frequencies for Bode plot, in rad/s
w_shared_low, w_shared_high, N_w = np.log10(Hz2rps(10**(-2))), np.log10(Hz2rps(10**(2))), 500
w_shared = np.logspace(w_shared_low, w_shared_high, N_w)

# %%
# Extract uncertainty weight and nominal model.

# Uncertainty weight
# Dummy uncertainty weight! You must change!
# This weight is from a multiplicative uncertainty model. 
#3.219e4
W2 = control.TransferFunction([1.062, 60.41, 134.6, 19.3], [1, 119.2, 443.6, 64.12],
                              inputs=["u"],
                              outputs=["y_Delta"],
                              name="W2")
print("W_2(s) = ", W2)

# Nominal model, which is NOT normalized, and has units of LPM / V.
# Note, this is just something made up, and does NOT imply the pump is first order!
# Dummy plant! You must change!
# The ``tilde" means ``with units". This sample code has not done any normalization. 
m, n = 0, 1
P_tilde = control.TransferFunction([22.67], [1, 10.46])

DC_gain = P_tilde.dcgain()
max_V = 5
slope = 2.437 #From step response data
max_LPM = max_V * slope  # Dummy value. You must change. 

# Normalize. 
#normalization constants:
SD_noise = 0.107 #LPM

r_nor = max_LPM #LPM
u_nor_r = max_V #V
e_nor_r = 0.05 * max_LPM #LPM

n_nor = SD_noise #LPM
u_nor_n = 0.02*max_V #V 
e_nor_n = n_nor #LPM

d_nor = 0.256 # V
u_nor_d = d_nor
e_nor_d = 0.01 * max_LPM #LPM

#Get this from fft of reference
w_r_h_Hz = 0.015  # Hz
w_n_Hz = 3 # Hz

P0 = P_tilde * u_nor_r / e_nor_r

P_nom = control.TransferFunction(np.array(P0.num).ravel(), np.array(P0.den).ravel(),
                              inputs=["u_total"],
                              outputs=["y0"], 
                              name="P_nom")

control.bode([P_nom, W2], w_shared, Hz=True)


# %%
# Other weights

w_r_h = Hz2rps(w_r_h_Hz)

#Reference Scaling
# R = control.TransferFunction([1], [r_nor],
#                               inputs=["r_tilde"],
#                               outputs=["r_scaled"],
#                               name="R")

# N = control.TransferFunction([1], [SD_noise],
#                               inputs=["n_tilde"],
#                               outputs=["n_scaled"],
#                               name="N")


Wr_tf = (1 / (s / w_r_h + 1))
Wr = control.tf(np.array(Wr_tf.num).ravel(), np.array(Wr_tf.den).ravel(),
                inputs=["r"], outputs=["r_f"], name="Wr")


w_n_l = Hz2rps(w_n_Hz)
Wn_tf = (0.2 / (s / (w_n_l * 100)  + 1))
Wn_tf = control.tf([1], [1])
Wn = control.tf(np.array(Wn_tf.num).ravel(), np.array(Wn_tf.den).ravel(),
                inputs=["n"], outputs=["n_f"], name="Wn")


R = control.tf([r_nor/e_nor_r], [1], inputs=["r_f"], outputs=["r_scaled"], name="R")
N = control.tf([n_nor/e_nor_r], [1], inputs=["n_f"], outputs=["n_scaled"], name="N")


sum_ideal_error = control.summing_junction(
    inputs=["r_scaled", "-y0"],
    outputs=["e_ideal"],
    name="sum_ideal_error"
)

k = 2
epsilon = 10**(-30 / 20)
Me = 10**(5 / 20)
w_e = Hz2rps(w_r_h_Hz + 0.1)
We_tf = ((s / Me**(1 / k) + w_e) / (s + w_e * (epsilon)**(1 / k)))**k

# w_e = Hz2rps(w_r_h_Hz + 0.2)
We_tf = 1 / (s / (w_e / 1) + 1)
We = control.TransferFunction(np.array(We_tf.num).ravel(), np.array(We_tf.den).ravel(),
                              inputs=["e_ideal"],
                              outputs=["z[1]"],
                              name="We")

sum_noise = control.summing_junction(
    inputs = ["n_scaled","y0"],
    outputs = ["yn"],
    name = "sum_noise"
)

sum_error = control.summing_junction(
    inputs = ["r_scaled","-yn"],
    outputs = ["e"],
    name = "sum_error"
)

sum_u = control.summing_junction(
    inputs = ["u","u_Delta"],
    outputs = ["u_total"],
    name = "sum_u"
)

w_u_l = w_e
Wu_tf = (1 - 1 / (s / w_u_l + 1))**2

wbc = w_e
Mu = 10**(10/20)

# Wu_tf = ((s + wbc / Mu**(1 / k)) / (s * (epsilon)**(1 / k) + wbc))**k

Wu = control.TransferFunction(np.array(Wu_tf.num).ravel(), np.array(Wu_tf.den).ravel(),
                              inputs=["u"],
                              outputs=["z[0]"],
                              name="Wu")



# Reference weight
# Dummy value. You must change. 

# Wr_tf = (1 / (s / w_r_h + 1))
# Wr = control.TransferFunction(np.array(Wr_tf.num).ravel(), np.array(Wr_tf.den).ravel(),
#                               inputs=["r"],
#                               outputs=["r_filtered"],
#                               name="Wr")
# # control.bode(Wr)

# # Noise weight
# # Dummy value. You must change. 
# w_n_l_Hz = w_r_h_Hz * 50
# w_n_l = Hz2rps(w_n_l_Hz)
# Wn_tf = control.TransferFunction([0.2], [1])
# Wn = control.TransferFunction(np.array(Wn_tf.num).ravel(), np.array(Wn_tf.den).ravel(),
#                               inputs=["n"],
#                               outputs=["n_filtered"],
#                               name="Wn")
# control.bode(Wn)

# Control weight
# Dummy value. You must change. 

# control.bode(Wu)

# Error weight

# control.bode(Wu)

control.bode([We, Wu, Wr, Wn], w_shared, Hz=True)

"""
# Interconnect

# The interconnection is dictated by the generalized plant structure. 

"""

# IMPORTANT: Order inputs/outputs for proper M-Delta structure:
#   - Uncertainty channels FIRST
#   - Then performance/disturbance channels
#   - Controller output (u) LAST in inputs
#   - Controller input (e) LAST in outputs
#
# Correct ordering:
#   Inputs:  [u_Delta, r_tilde, n_tilde, u]  → uncertainty first, then disturbances, then control
#   Outputs: [y_Delta, z[1], z[0], e]        → uncertainty first, then performance, then controller
P = control.interconnect(
    syslist=[R, N, Wr, Wn, sum_ideal_error, We, Wu, sum_noise, sum_error, sum_u, W2, P_nom],
    inplist=['u_Delta', 'r', 'n', 'u'],
    outlist=['y_Delta', 'z[1]', 'z[0]', 'e'],
    inputs=['u_Delta', 'r', 'n', 'u'],
    outputs=['y_Delta', 'z[1]', 'z[0]', 'e']
)
# print("Generalized plant P:")
# print(P)
P.dt = 0  # Continuous-time system
n_y = 1  # Controller input dimension (e)
n_u = 1  # Controller output dimension (u)

# K_hinf_syn = control.hinfsyn(P, n_y, n_u)

#print("K_hinf_syn = ", K_hinf_syn)



# %%
# DK-iteration using dkpy

# DK-iteration setup
print("Running DK-iteration...")
import logging
logging.basicConfig(level=logging.INFO)
dk_iter = dkpy.DkIterAutoOrder(
    controller_synthesis=dkpy.HinfSynLmi(
        lmi_strictness=1e-7,
        solver_params=dict(solver="MOSEK", eps=1e-8, verbose=False),
    ),
    structured_singular_value=dkpy.SsvLmiBisection(
        bisection_atol=1e-5,
        bisection_rtol=1e-5,
        max_iterations=1000,
        lmi_strictness=1e-7,
        solver_params=dict(solver="MOSEK", eps=1e-9, verbose=False),
        n_jobs=1,
    ),
    d_scale_fit=dkpy.DScaleFitSlicot(),
    max_mu=1,
    max_mu_fit_error=1e-2,
    max_iterations=5,
    max_fit_order=2,
)

# Uncertainty structure: 1×1 (uncertainty) + 2×2 (performance) = 3×3 Delta
uncertainty_structure = [
    dkpy.ComplexFullBlock(1, 1),
    dkpy.ComplexFullBlock(2, 2),
]
block_structure = np.array([[1, 1], [2, 2]])
omega = np.logspace(-3, 3, 61)
n_y, n_u = 1, 1

# Synthesize controller
K, N, mu, iter_results, info = dk_iter.synthesize(P, n_y, n_u, omega, block_structure)

print(f"DK-iteration complete: μ = {mu:.4f}, iterations = {len(iter_results)}, controller states = {K.A.shape[0]}")

print("K = ", K)

print(f"mu={mu}")


#USE HINFSYN FIRST TO GET A CONTROLLER

# Extract controller from hinfsyn output
# hinfsyn returns (K, N, gamma, info) where:
#   K: controller (StateSpace)
#   N: closed-loop system (StateSpace)
#   gamma: H-infinity norm (float)
#   info: additional info
# K_ss = K_hinf_syn[0]  # Controller as StateSpace
# gamma = K_hinf_syn[2]  # H-infinity norm
# print(f"Controller H-infinity norm (gamma) = {gamma}")
# print("Controller (StateSpace):\n", K_ss, "\n")

# Convert controller to transfer function form for simulation
C = control.ss2tf(K)
print("Controller (TransferFunction):\n", C, "\n")

# %% 
# Bode Plot of important closed loop transfer functions

# Closed-loop T(s) and S(s) transfer functions.
S = control.feedback(1, P_nom * K, -1)
T = control.feedback(P_nom * K, 1, -1)

fig, ax = plt.subplots()
control.bode([1 / (1 + C * P_nom), 1/We], w_shared, Hz=True)

fig, ax = plt.subplots()
control.bode([S, T], w_shared, Hz=True)

fig, ax = plt.subplots()
control.bode([C*S, 1/Wu], w_shared, Hz=True)

# %%
# Reference

# Get the directory where this script is located
script_dir = pathlib.Path(__file__).parent
csv_path = script_dir / "RL_temp_motor_mod.csv"

data = np.loadtxt(
    csv_path,
    dtype=float,
    delimiter=',',
    skiprows=1,
    usecols=(0, 1),
    # max_rows=1100,
)

# Extract time and temperature data
N_temp_data = data.shape[0]
dt = 0.02

# Must specify time, because the data has some numerical rounding issues. 
t_raw = np.linspace(0, dt * N_temp_data, N_temp_data)

# All the temperatures
temp_raw_raw = data[:, 1]

# Extract a subset of time 
t_start = 900  # s
t_end = 1900  # s
# t_end = 1200  # s
t_start_index = np.where(np.abs(t_raw - t_start) <= 0.02)[0][-1]
t_end_index = np.where(np.abs(t_raw - t_end) <= 0.02)[0][-1]

# Extract time over the desired interval
t = t_raw[t_start_index:t_end_index]

# Extract temperature data over the desired interval. 
temp_raw = temp_raw_raw[t_start_index:t_end_index]

# Plotting
# Plot raw data time domain
fig, ax = plt.subplots()
fig.set_size_inches(height * gr, height, forward=True)
ax.plot(t, temp_raw)
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'Temperature (°C)')
fig.tight_layout()
# fig.savefig('x.pdf')

temp_raw_max = np.max(np.abs(temp_raw))
temp_raw_min = np.min(np.abs(temp_raw))
print("The max (in absolute value) output is", temp_raw_max, '(°C)')
print("The min (in absolute value) output is", temp_raw_min, '(°C)')

# Round the max up to 90 C
temp_raw_max = 90  # C

# Extract temperature data 
temp_raw = temp_raw_raw[t_start_index:t_end_index]

# Normalize and center temperature data
temp_norm = (temp_raw - temp_raw_min) / (temp_raw_max - temp_raw_min)

# Define Reference in terms of flow rate
# Convert temp into a reference LPM
r_raw_tilde = temp_norm * max_LPM

# Now filter using w_r_h
a = w_r_h
#a = 0.5
_, r_tilde = control.forced_response(1 / (1 / a * s + 1), t, r_raw_tilde, 0)

r_raw = r_raw_tilde  # r_raw would be unfiltered, without units, if you did some normalization
r = r_tilde  # r would be filtered, without units, if you did some normalization

fig, ax = plt.subplots()
ax.set_xlabel(r'$t$ (s)')
ax.set_ylabel(r'$r(t)$ (LPM)')
# Plot data
ax.plot(t, r, '--', label='$r(t)$', color='C3')
ax.legend(loc='upper right')
fig.tight_layout()
plt.show()
# fig.savefig('x.pdf')

# Noise
np.random.seed(123321)
noise_raw = np.random.normal(0, 1, t.shape[0])
sigma_n = SD_noise  # LPM, dummy value. You must change. 
noise =  sigma_n * noise_raw * 1  # Change the 1 to a zero to ``turn off" noise in order to debug. 


# %%
# Set up simulation.
"""
You should not need to change this part of the code. Just run the 
simulation once you've found the reference, noise, controller, 
bounds max_V and max_LPM, etc. 
"""

# Note, the simulation does take awhile to run. 
# When debugging, set t_end = 1200, to shorten the simulation. 

u_range = np.array([0, max_V])
z_range = np.array([0, max_LPM])

def simulate(P, C, t, r_raw, r, noise, u_range, z_range):
    """Nonlinear simulation.
    
    The plant is not linear, it's affine. This is why this 
    type of simulation is needed. 
    """

    # Time needs to be redefined as a new variable for solve_ivp.
    time = t

    # Plant state-space form.
    P_ss = control.tf2ss(np.array(P.num).ravel(), np.array(P.den).ravel())
    n_x_P = np.shape(P_ss.A)[0]
    
    # Control state-space form.
    C_ss = control.tf2ss(np.array(C.num).ravel(), np.array(C.den).ravel())
    n_x_C = np.shape(C_ss.A)[0]

    # ICs for plant and control. 
    x_P_IC = np.zeros((n_x_P, 1))
    x_C_IC = np.zeros((n_x_C, 1))
    
    # Set up closed-loop ICs.
    x_cl_IC = np.block([[x_P_IC], [x_C_IC]]).ravel()

    # Define closed-loop system. This will be passed to solve_ivp.
    def closed_loop(t, x):
        """Closed-loop system"""

        # Reference at current time.
        r_now = np.interp(t, time, r).reshape((1, 1))

        # Noise at current time.
        n_now = np.interp(t, time, noise).reshape((1, 1))

        # Split state.
        x_P = (x[:n_x_P]).reshape((-1, 1))
        x_C = (x[n_x_P:]).reshape((-1, 1))

        # Interpolation of u_bar and y_bar.
        # Note, r_raw is used, because we are ``linearizing" about
        # the current reference point.
        z_bar = np.interp(t, time, r_raw).reshape((1, 1))
        u_bar = np.interp(z_bar, z_range, u_range).reshape((1, 1))
        
        # Plant output, with noise.
        delta_z = P_ss.C @ x_P
        y = z_bar + delta_z + n_now
        
        # Compute error.
        error = r_now - y
        
        # Compute control signal. 
        delta_u = C_ss.C @ x_C + C_ss.D @ error
        u = u_bar + delta_u

        # Advance system state.
        dot_x_sys = P_ss.A @ x_P + P_ss.B @ u - P_ss.B @ u_bar
        
        # Advance controller state.
        dot_x_ctrl = C_ss.A @ x_C + C_ss.B @ error

        # Concatenate state derivatives.
        x_dot = np.block([[dot_x_sys], [dot_x_ctrl]]).ravel()

        return x_dot


    # Find time-domain response by integrating the ODE
    sol = integrate.solve_ivp(
        closed_loop,
        (t_start, t_end),
        x_cl_IC,
        t_eval=t,
        rtol=1e-8,
        atol=1e-6,
        method='RK45',
    )
#   rtol=1e-6,
#         atol=1e-8,
#         method='LSODA',
    # Extract states.
    sol_x = sol.y
    x_P = sol_x[:n_x_P, :]
    x_C = sol_x[n_x_P:, :]

    # Compute plant output, control signal, and ideal error.
    y = np.zeros(t.shape[0],)
    u = np.zeros(t.shape[0],)
    e = np.zeros(t.shape[0],)

    for i in range(time.size):

        # Reference at current time.
        r_now = np.interp(t[i], time, r).reshape((1, 1))
        
        # Noise at current time.
        n_now = np.interp(t[i], time, noise).reshape((1, 1))

        # Interpolation of u_bar and y_bar
        # Note, r_raw is used, because we are ``linearizing" about the current reference point.
        z_bar = np.interp(t[i], time, r_raw).reshape((1, 1))
        u_bar = np.interp(z_bar, z_range, u_range).reshape((1, 1))

        # Plant output, with noise.
        delta_z = P_ss.C @ x_P[:, [i]]
        y[i] = (z_bar + delta_z + n_now).ravel()[0]

        # Compute error.
        error = r_now - (z_bar + delta_z + n_now)
        e[i] = error.ravel()[0]

        # Compute control.
        delta_u = C_ss.C @ x_C[:, [i]] + C_ss.D @ error
        u[i] = (u_bar + delta_u).ravel()[0]

    return y, u, e


# Run simulation
# C must be in transfer function form. 
y, u, e = simulate(P_nom, C, t, r_raw, r, noise, u_range, z_range)


# %%
# Plots

y_tilde = y
u_tilde = u
e_tilde = e

# Max acceptable error and control values. 
e_nor_ref = e_nor_r 
u_nor_ref = 5  # V


# Plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
ax[0].set_ylabel(r'$\tilde{y}(t)$ (LPM)')
ax[1].set_ylabel(r'$\tilde{u}(t)$ (V)')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')

ax[0].plot(t, y_tilde, '-', label=r'$\tilde{y}(t)$', color='C0')
ax[0].plot(t, r_tilde, '--', label=r'$\tilde{r}(t)$', color='C3')
ax[1].plot(t, u_tilde, '-', label=r'$\tilde{u}(t)$', color='C1')
ax[1].plot(t, u_nor_ref * np.ones(t.shape[0],), '--', label=r'$u_{nor, r}$', color='C6')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
fig.tight_layout()
# fig.savefig('y_u_time_dom_response_tilde.pdf')

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$\tilde{y}(t)$ (LPM)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, y_tilde, '-', label=r'$\tilde{y}(t)$', color='C0')
ax.plot(t, r_tilde, '--', label=r'$\tilde{r}(t)$', color='C3')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('y_time_dom_response_tilde.pdf')

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$\tilde{e}(t)$ (LPM)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, e_tilde, '-', label=r'$\tilde{e}(t)$', color='C0')
ax.plot(t, e_nor_ref * np.ones(t.shape[0],), '--', label=r'$e_{nor, r}$', color='C6')
ax.plot(t, -e_nor_ref * np.ones(t.shape[0],), '--', color='C6')
ax.legend(loc='upper right')
fig.tight_layout()
# fig.savefig('e_time_dom_response_tilde.pdf')

# Plot
fig, ax = plt.subplots(2, 1, figsize=(height * gr, height))
ax[0].set_ylabel(r'$\tilde{e}(t)$ (LPM)')
ax[1].set_ylabel(r'$\tilde{u}(t)$ (V)')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (s)')

ax[0].plot(t, e_tilde, '-', label=r'$\tilde{e}(t)$', color='C0')
ax[0].plot(t, e_nor_ref * np.ones(t.shape[0],), '--', label=r'$e_{nor, r}$', color='C6')
ax[0].plot(t, -e_nor_ref * np.ones(t.shape[0],), '--', color='C6')
ax[1].plot(t, u_tilde, '-', label=r'$\tilde{u}(t)$', color='C1')
ax[1].plot(t, u_nor_ref * np.ones(t.shape[0],), '--', label=r'$u_{nor, r}$', color='C6')
ax[0].legend(loc='lower right')
ax[1].legend(loc='lower right')
fig.tight_layout()
# fig.savefig('u_e_time_dom_response_tilde.pdf')


# %%
# Find the mean and standard deviation of the error
mu, sigma = norm.fit(e_tilde)
print('The mean and standard deviation of the error is', mu, 'and', sigma, '\n\n')

fig, ax = plt.subplots()
count, bins, patches = plt.hist(e_tilde, np.int32(np.floor(e_tilde.size / 100)), density=True)
ax.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(bins - mu) ** 2 / (2 * sigma**2)), linewidth=2, color='r')
ax.set_xlabel(r'$\tilde{e}$ (LPM)')
ax.set_ylabel('Normalized Count')
fig.tight_layout()
# fig.savefig('e_tilde_hist.pdf')


# %%
# Energy estimate

power = np.zeros(t.size,)
for i in range(t.size):
    power[i] = u_tilde[i]**2

# Plot
fig, ax = plt.subplots(figsize=(height * gr, height))
ax.set_ylabel(r'$P(t)$ (V$^2$)')
ax.set_xlabel(r'$t$ (s)')
ax.plot(t, power, '-', label=r'$P(t)$', color='C0')
ax.legend(loc='best')
fig.tight_layout()
# fig.savefig('power_fb_vs_time.pdf')

# Integrate using Simpson's rule to get total ``energy" in units of V^2 s
energy = integrate.simpson(power, x=t)
print("The total energy consumed when using feedback control is", energy, '(V^2 s).')

# Compute total energy when pump is at max voltage all the time, or ``all out". 
energy_ao = integrate.simpson(max_V**2 * np.ones(t.size,), x=t)
print("The total energy consumed when using the max voltage all the time is", energy_ao, '(V^2 s).')
print("The percent energy saved relative to the max voltage all the time is", (energy_ao - energy) / energy_ao * 100, '(%).')


# %%
# Some stats

# print("The mean error is", np.mean(e_tilde), '(LPM). \n')
print("The mean error relative to the max flow rate is", np.mean(e_tilde) / max_LPM * 100, '(%). \n')

# print("The error standard deviation is", np.std(e_tilde), '(LPM). \n')
print("The error standard deviation relative to the max flow rate is", np.std(e_tilde) / max_LPM * 100, '(%). \n')

# print("The mean of the control signal is", np.mean(u_tilde), '(V). \n')
print("The mean of the control signal relative to the max control signal is", np.mean(u_tilde) / max_V * 100, '(%). \n')

# print("The control signal standard deviation is", np.std(u_tilde), '(V). \n')
print("The control signal standard deviation relative to the max control signal is", np.std(u_tilde) / max_V * 100, '(%). \n')

# print("The max error is", np.max(e_tilde), '(LPM). \n')
print("The max error relative to the max flow rate is", np.max(e_tilde) / max_LPM * 100, '(%). \n')

# print("The max control effort is", np.max(u_tilde), '(V). \n')
print("The mean control signal relative to the max control signal is", np.max(u_tilde) / max_V * 100, '(%). \n')



# %%
# Plot
plt.show()

# %%
