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

# %%
# Frequency Domain Transfer Function
def signal_process(
        data,
        coh_thresh=0.8,
        min_puu_frac=1e-3,
        fmax_frac=0.9
    ):
    """
    Returns a dict with {omega, G, Cuy, mask, fs, t, u, y}.
    Mask keeps points: Cuy>=coh_thresh, Puu above noise floor, and f<=fmax_frac*fN.
    """
    t = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]

    T = t[1] - t[0]
    fs = 1.0 / T
    fN = fs / 2

    f, Puy = signal.csd(u, y, fs=fs, window='hann')
    _, Puu = signal.csd(u, u, fs=fs, window='hann')
    _, Pyy = signal.csd(y, y, fs =fs, window='hann')

    G = Puy / Puu
    Cuy = (np.abs(Puy)**2) / (Puu * Pyy + 1e-16)

    # Build mask
    mask = (
        (Cuy >= coh_thresh) &
        (Puu >= min_puu_frac * np.median(Puu)) &
        (f <= fmax_frac * fN)
    )

    return dict(
        f = (f[mask], f),
        omega = (2*np.pi*f[mask], 2*np.pi*f),
        G = (G[mask], G),
        Cuy = (Cuy[mask], Cuy),
        mask = mask,
        fs = fs,
        t = t,
        u = u,
        y = y,
        csd = (Puy, Puu, Pyy)
    )
  

def form_LS_system(G_csd, omega, m, n, weights=None):
    # Fit a tranfer function to discrete transfer function
    # ---------------------------------------------------- 
    # Degree of numerator, degree of denominator.

    # Number of numerator and denominator parameters.
    N_num = m + 1
    N_den = n
    Nw = omega.shape[0]

    #Weight Based on coherence
    if weights is None:
        W = np.ones(Nw)
    else:
        W = weights
    # Initialized A and b matrices that will be complex.
    b = -G_csd * (1j * omega)**n * W
    A = np.zeros((Nw, N_den + N_num), dtype=complex) 

    # Fill A matrix row by row 
    for i in range(N_num):
        A[:, i] = -(1j * omega)**(m - i) * W

    for j in range(N_den):
        A[:, j+N_num] = (1j * omega)**(n-j-1) * G_csd * W

    # Split A and b into real and complex parts. Stack real part on top of
    # complex part to create large A matrix.
    A1 = np.vstack([np.real(A), np.imag(A)])
    b1 = np.hstack([np.real(b), np.imag(b)])
    return A1, b1

def LS_fit(A1, b1, m, n, prin=True):

    # Number of numerator and denominator parameters.
    N_num = m + 1
    N_den = n

    x_lsq = np.linalg.lstsq(A1, b1, rcond=None)
    print(x_lsq)

    x = x_lsq[0]
    num_coeffs = x[:N_num]                 # [bm, bm-1, ..., b0] for s^m ... s^0
    den_coeffs = np.r_[1.0, x[N_num:]]     # [an-1, an-1, ..., a0]

    # Make TF (descending powers)
    G_LS = ct.tf(num_coeffs, den_coeffs)
    if prin:
        print("num =", num_coeffs)
        print("den =", den_coeffs)
        print("Pd_ID =", G_LS)

    return G_LS, x

def fit_error(A, b, x, m, n, prin=False):
    N = A.shape[0]
    e = b - A @ x
    e2 = np.dot(e,e) #sum of squared residuals

    # Compute the uncertainty and relative uncertainty. 
    cov_x = np.linalg.inv(A.T @ A) / (N - (n+m+1)) * e2
    sigma = np.sqrt(np.diagonal(cov_x))
    rel_unc = sigma / np.abs(x) * 100 

    # Compute the MSE, MSO, NMSE.
    MSE = e2 / N
    MSO = np.dot(b, b) / N
    NMSE = MSE / MSO

    if prin:
        print('The standard deviation is', sigma)
        print('The relative uncertainty is', rel_unc, '%\n')
        print('The MSE is', MSE)
        print('The MSO is', MSO)
        print('The NMSE is', NMSE, '\n')

    return dict(
        sigma=sigma,
        rel_unc=rel_unc,
        MSE=MSE,
        MSO=MSO,
        NMSE=NMSE
    )

def compare(x_train, data_test, m, n):
    res_test = signal_process(data_test)
    omega_test = res_test['omega'][0]
    G_test = res_test['G'][0]
    A_test, b_test = form_LS_system(G_test, omega_test, m, n)

    fit_error(A_test, b_test, x_train, m, n)

def response_error(data, G, prin=False, plot=False):

    t = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]
    N = t.size

    t_ID, y_ID = ct.forced_response(G, t, u)
    
    e = y_ID - y

    e_var = np.var(e, ddof=0)
    y_var = np.var(y, ddof=0)

    nmse_t = e_var / y_var
    VAF_test = (1 - e_var/y_var) * 100

    if prin:
        print('Time-domain NMSE:', nmse_t)
        print('The %VAF is', VAF_test)

    # Compute and plot errors
    e_abs = np.abs(e)
    y_max = np.max(np.abs(y))
    e_rel = e_abs / y_max

    if plot:
        # Plot test data
        fig, ax = plt.subplots(2, 1)   
        ax[0].set_ylabel(r'$u(t)$ (V)')
        ax[1].set_ylabel(r'$y(t)$ (LPM)')
        # Plot data
        ax[0].plot(t, u, '--', label='input', color='C0')
        ax[1].plot(t, y, label='output', color='C1')
        ax[1].plot(t_ID, y_ID, '-.', label="IDed output", color='C2')
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
            a.legend(loc='upper right')
        fig.tight_layout()

        # Plot error
        fig, ax = plt.subplots(2, 1)
        # Format axes
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
        ax[0].set_ylabel(r'$e_{abs}(t)$ (LPM)')
        ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
        # Plot data
        ax[0].plot(t, e)
        ax[1].plot(t, e_rel)
        # for a in np.ravel(ax):
        #     a.legend(loc='lower right')
        fig.tight_layout()
    
    return float(nmse_t), float(VAF_test)

def response_error_std(data, G, plot=False):
    t = data[:, 0]
    u = data[:, 1]
    y = data[:, 2]
    N = t.size

    # standardize data
    u_mean = np.mean(u)
    u_std = np.std(u)
    y_mean =np.mean(y)
    y_std = np.std(y)

    us = (u - u_mean)/u_std
    ys = (y - y_mean)/y_std
    # SCALE THE MODEL to standardized units
    Gs = (u_std / (y_std + 1e-16)) * G
    t_ID, y_ID = ct.forced_response(Gs, t, us)
    
    e = y_ID - ys
    e_var = np.var(e, ddof=0)
    y_var = np.var(ys, ddof=0)

    VAF_test = (1 - e_var/y_var) * 100
    print('The %VAF is', VAF_test)

    # Compute and plot errors
    e_abs = np.abs(e)
    y_max = np.max(np.abs(ys))
    e_rel = e_abs / y_max

    if plot:
        # Plot test data
        fig, ax = plt.subplots(2, 1)   
        ax[0].set_ylabel(r'$u(t)$ (V)')
        ax[1].set_ylabel(r'$y(t)$ (LPM)')
        # Plot data
        ax[0].plot(t, us, '--', label='input', color='C0')
        ax[1].plot(t, ys, label='output', color='C1')
        ax[1].plot(t_ID, y_ID, '-.', label="IDed output", color='C2')
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
            a.legend(loc='upper right')
        fig.tight_layout()

        # Plot error
        fig, ax = plt.subplots(2, 1)
        # Format axes
        for a in np.ravel(ax):
            a.set_xlabel(r'$t$ (s)')
        ax[0].set_ylabel(r'$e_{abs}(t)$ (LPM)')
        ax[1].set_ylabel(r'$e_{rel}(t) \times 100\%$ (unitless)')
        # Plot data
        ax[0].plot(t, e)
        ax[1].plot(t, e_rel)
        # for a in np.ravel(ax):
        #     a.legend(loc='lower right')
        fig.tight_layout()
    
    return float(VAF_test)


# %%

def _freq_errors(G_model, omega, G_meas):
    """
    Compute freq-domain errors between measured G(jw) and model G(jw).
    Returns a dict with NMSE_freq, RMSE/MAE in |.|_dB and phase (deg), max |.|_dB.
    """
    # Evaluate model at jw using python-control
    mag_hat, ph_hat, _ = ct.frequency_response(G_model, omega)
    mag_hat = np.squeeze(mag_hat)
    ph_hat  = np.squeeze(ph_hat)

    mag_mea = np.abs(G_meas)
    ph_mea  = np.unwrap(np.angle(G_meas))

    # Complex NMSE
    Ghat_c  = mag_hat * np.exp(1j * ph_hat)
    e       = G_meas - Ghat_c
    nmse_f  = np.sum(np.abs(e)**2) / (np.sum(np.abs(G_meas)**2) + 1e-16)

    # Magnitude/phase errors
    mag_err_db = 20*np.log10(np.maximum(mag_hat, 1e-16)) - 20*np.log10(np.maximum(mag_mea, 1e-16))
    ph_err_deg = (np.unwrap(ph_hat) - ph_mea) * 180/np.pi

    rmse_mag_db = float(np.sqrt(np.mean(mag_err_db**2)))
    mae_mag_db  = float(np.mean(np.abs(mag_err_db)))
    max_mag_db  = float(np.max(np.abs(mag_err_db)))
    rmse_ph_deg = float(np.sqrt(np.mean(ph_err_deg**2)))
    mae_ph_deg  = float(np.mean(np.abs(ph_err_deg)))

    return dict(
        NMSE_freq=float(np.real_if_close(nmse_f)),
        RMSE_mag_dB=rmse_mag_db,
        MAE_mag_dB=mae_mag_db,
        MAX_mag_dB=max_mag_db,
        RMSE_phase_deg=rmse_ph_deg,
        MAE_phase_deg=mae_ph_deg
    )

# %% -------------------------------------------------------------------------
# Extra helper: time-domain metrics (matches your definitions; optional standardization)
# ----------------------------------------------------------------------------
def _time_error_metrics(data, G, standardize=True):
    t = data[:, 0]; u = data[:, 1]; y = data[:, 2]
    T = t[1] - t[0]
    if standardize:
        u_mean = np.mean(u); y_mean = np.mean(y)
        u_std  = np.std(u);  y_std  = np.std(y)
        us = (u - u_mean) / (u_std + 1e-16)
        ys = (y - y_mean) / (y_std + 1e-16)
        Gs = (u_std / (y_std + 1e-16)) * G   # your scaling in response_error_std
        _, yhat = ct.forced_response(Gs, t, us)
        e  = yhat - ys
        y_for_var = ys
    else:
        _, yhat = ct.forced_response(G, t, u)
        e  = y - yhat
        y_for_var = y

    mse   = float(np.mean(e**2))
    std_e = float(np.std(e, ddof=1)) if e.size > 1 else np.nan
    var_y = float(np.var(y_for_var, ddof=0))
    var_e = float(np.var(e,       ddof=0))
    nmse  = float(var_e / (var_y + 1e-16))
    vaf   = float((1.0 - var_e / (var_y + 1e-16)) * 100.0)
    cov_y_yhat = np.cov(np.vstack([y_for_var, yhat]))[0, 1] if yhat.size > 1 else np.nan
    return dict(NMSE_t=nmse, VAF=vaf, RMSE=mse**0.5, std_resid=std_e, var_y=var_y, var_resid=var_e, cov_y_yhat=cov_y_yhat)


# %% -------------------------------------------------------------------------
# Table printer: train on each dataset, test on the other three (fixed m,n)
# ----------------------------------------------------------------------------
def evaluate_tf_table(
        datasets,              # list like [data_0, data_1, data_2, data_3]
        m=0, n=1,
        # spectral estimation / masking to pass through to your signal_process:
        coh_thresh=0.8, min_puu_frac=1e-3, fmax_frac=0.9,
        # LS weighting: use √coherence on masked band if True
        use_weights=False,
        # print train-on-train row too?
        include_train_row=True,
        # standardize time-domain metrics (your preferred VAF definition)?
        standardize_time=False
    ):
    """
    For each dataset k used as TRAIN:
      • build freq-domain G(jw) from TRAIN (your signal_process)
      • fit LS TF (m,n) using your form_LS_system + LS_fit (+ coherence weights if enabled)
      • freq-domain errors on TRAIN and on each TEST (LS residual MSE/MSO/NMSE + sigma/rel%)
      • time-domain NMSE/VAF/RMSE/std(resid)/cov(y,ŷ) on TRAIN and on each TEST
    Prints a compact table. Returns list of row dicts for further use.
    """
    # Precompute spectral packs (using your exact signal_process outputs)
    packs = [signal_process(d, coh_thresh=coh_thresh,
                            min_puu_frac=min_puu_frac, fmax_frac=fmax_frac)
             for d in datasets]

    rows = []
    line = "-" * 140
    print(line)
    print(f"{'Tr':>2} {'Te':>2} | "
          f"{'F_MSE':>9} {'F_MSO':>9} {'F_NMSE':>8} {'σ_max':>8} {'rel%_max':>9} | "
          f"{'NMSE_t':>8} {'VAF%':>8} {'RMSE':>8} {'std_e':>8} {'cov(y,ŷ)':>10}")
    print(line)

    N = len(datasets)
    for i in range(N):
        # --- Fit on TRAIN i using your pipeline ---
        G_i_masked = packs[i]['G'][0]
        w_i        = packs[i]['omega'][0]
        Cuy_i      = packs[i]['Cuy'][0]
        W_i        = np.sqrt(Cuy_i) if use_weights else None

        A1_i, b1_i = form_LS_system(G_i_masked, w_i, m, n, weights=W_i)
        G_fit, x_i = LS_fit(A1_i, b1_i, m, n, prin=False)

        # Freq-domain LS residual metrics on TRAIN (using your fit_error)
        fe_train = fit_transfer_metrics = fit_error(A1_i, b1_i, x_i, m, n, prin=False)

        # Time-domain metrics on TRAIN
        tm_train = _time_error_metrics(datasets[i], G_fit, standardize=standardize_time)

        if include_train_row:
            row = dict(
                train=i, test=i,
                F_MSE=fit_transfer_metrics['MSE'],
                F_MSO=fit_transfer_metrics['MSO'],
                F_NMSE=fit_transfer_metrics['NMSE'],
                SIGMA_MAX=float(np.max(fit_transfer_metrics['sigma'])),
                RELUNC_MAX=float(np.max(fit_transfer_metrics['rel_unc'])),
                NMSE_t=tm_train['NMSE_t'], VAF=tm_train['VAF'],
                RMSE=tm_train['RMSE'], std_resid=tm_train['std_resid'],
                cov_y_yhat=tm_train['cov_y_yhat']
            )
            rows.append(row)
            print(f"{i:>2} {i:>2} | "
                  f"{row['F_MSE']:9.3f} {row['F_MSO']:9.3f} {row['F_NMSE']:8.3f} {row['SIGMA_MAX']:8.3f} {row['RELUNC_MAX']:9.2f} | "
                  f"{row['NMSE_t']:8.3f} {row['VAF']:8.2f} {row['RMSE']:8.3f} {row['std_resid']:8.3f} {row['cov_y_yhat']:10.3f}")

        # --- Test on each other set j ---
        for j in range(N):
            if j == i:
                continue
            G_j_masked = packs[j]['G'][0]
            w_j        = packs[j]['omega'][0]
            Cuy_j      = packs[j]['Cuy'][0]
            W_j        = np.sqrt(Cuy_j) if use_weights else None

            # freq-domain residuals of TRAIN θ against TEST A,b
            A1_j, b1_j = form_LS_system(G_j_masked, w_j, m, n, weights=W_j)
            fe_test = fit_error(A1_j, b1_j, x_i, m, n, prin=False)

            # time-domain metrics on TEST j
            tm_test = _time_error_metrics(datasets[j], G_fit, standardize=standardize_time)

            row = dict(
                train=i, test=j,
                F_MSE=fe_test['MSE'],
                F_MSO=fe_test['MSO'],
                F_NMSE=fe_test['NMSE'],
                SIGMA_MAX=float(np.max(fe_test['sigma'])),
                RELUNC_MAX=float(np.max(fe_test['rel_unc'])),
                NMSE_t=tm_test['NMSE_t'], VAF=tm_test['VAF'],
                RMSE=tm_test['RMSE'], std_resid=tm_test['std_resid'],
                cov_y_yhat=tm_test['cov_y_yhat']
            )
            rows.append(row)
            print(f"{i:>2} {j:>2} | "
                  f"{row['F_MSE']:9.3f} {row['F_MSO']:9.3f} {row['F_NMSE']:8.3f} {row['SIGMA_MAX']:8.3f} {row['RELUNC_MAX']:9.2f} | "
                  f"{row['NMSE_t']:8.3f} {row['VAF']:8.2f} {row['RMSE']:8.3f} {row['std_resid']:8.3f} {row['cov_y_yhat']:10.3f}")

    print(line)
    print(f"(m, n) = ({m}, {n}), weights={'√coh' if use_weights else 'none'} | "
          f"coh≥{coh_thresh}, Puu≥{min_puu_frac}·median(Puu), f≤{fmax_frac}·fN")
    return rows

datasets = [data_0, data_1, data_2, data_3]

# First-order model, train on each, test on the other three:
_ = evaluate_tf_table(
    datasets,
    m=0, n=1,
    coh_thresh=0.8, min_puu_frac=1e-3, fmax_frac=0.9,
    use_weights=True,
    include_train_row=True,
    standardize_time=True  # for sensible VAF across amplitudes
)

# %%
