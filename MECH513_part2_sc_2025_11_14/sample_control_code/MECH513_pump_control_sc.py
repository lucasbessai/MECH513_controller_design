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
W2 = control.TransferFunction([2.196e4, 3.471e4, 1.607e4, 2269], [1, 5.577e4, 3.763e4, 3407],
                              inputs=["u"],
                              outputs=["y_Delta"],
                              name="W2")
print("W_2(s) = ", W2)

# Nominal model, which is NOT normalized, and has units of LPM / V.
# Note, this is just something made up, and does NOT imply the pump is first order!
# Dummy plant! You must change!
# The ``tilde" means ``with units". This sample code has not done any normalization. 
m, n = 0, 1
P_tilde = control.TransferFunction([30], [1, 10])

DC_gain = P_tilde.dcgain()
max_V = 5
max_LPM = 15  # Dummy value. You must change. 

# Normalize. 
# Dummy value. You must change. 
e_nor_ref = 0.7  # LPM, largest allowable error associated with the largest allowable reference
u_nor_ref = max_V # V, largest allowable change in the control input due to the largest allowable reference

P0 = control.TransferFunction(np.array(P_tilde.num).ravel(), np.array(P_tilde.den).ravel(),
                              inputs=["u0"],
                              outputs=["y0"], 
                              name="P0")

control.bode([P0, W2], w_shared, Hz=True)


# %%
# Other weights

# Reference weight
# Dummy value. You must change. 
w_r_h_Hz = 1
w_r_h = Hz2rps(w_r_h_Hz)
Wr_tf = (1 / (s / w_r_h + 1))
Wr = control.TransferFunction(np.array(Wr_tf.num).ravel(), np.array(Wr_tf.den).ravel(),
                              inputs=["r"],
                              outputs=["r_filtered"],
                              name="Wr")
# control.bode(Wr)

# Noise weight
# Dummy value. You must change. 
w_n_l_Hz = w_r_h_Hz * 50
w_n_l = Hz2rps(w_n_l_Hz)
Wn_tf = control.TransferFunction([0.2], [1])
Wn = control.TransferFunction(np.array(Wn_tf.num).ravel(), np.array(Wn_tf.den).ravel(),
                              inputs=["n"],
                              outputs=["n_filtered"],
                              name="Wn")
# control.bode(Wn)

# Control weight
# Dummy value. You must change. 
w_u_l_Hz = w_r_h_Hz * 5
Wu_tf = (1 - 1 / (s / Hz2rps(w_u_l_Hz) + 1))**2
Wu = control.TransferFunction(np.array(Wu_tf.num).ravel(), np.array(Wu_tf.den).ravel(),
                              inputs=["u"],
                              outputs=["z[0]"],
                              name="Wu")
# control.bode(Wu)

# Error weight
# Dummy value. You must change. 
We_tf = 1 / (s / (w_r_h / 1) + 1)
We = control.TransferFunction(np.array(We_tf.num).ravel(), np.array(We_tf.den).ravel(),
                              inputs=["e_ideal"],
                              outputs=["z[1]"],
                              name="We")
# control.bode(Wu)

control.bode([We, Wr, Wn, Wu], w_shared, Hz=True)

"""
# Interconnect

# The interconnection is dictated by the generalized plant structure. 

"""

# %%
# Use dkpy

# Dummy controller, you must change!
w_c = 1
L_des = w_c / s

K = L_des / P0
print("C = ", K, "\n")

# The rest of the code calls the controller C
# C must be in transfer function form. 
C = K


# %%
# Reference

data = np.loadtxt(
    "RL_temp_motor_mod.csv",
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

# Convert temp into a reference LPM
# You must change this!
r_raw_tilde = (temp_raw - temp_raw_min) * 0.3
# Now filter using w_r_h
a = w_r_h
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
sigma_n = 0.25  # LPM, dummy value. You must change. 
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
y, u, e = simulate(P0, C, t, r_raw, r, noise, u_range, z_range)


# %%
# Plots

y_tilde = y
u_tilde = u
e_tilde = e

# Max acceptable error and control values. 
e_nor_ref = 0.56  # Dummy variable, you change
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
