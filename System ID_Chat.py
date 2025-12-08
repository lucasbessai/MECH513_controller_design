# %%
# Libraries
import numpy as np
from numpy.linalg import lstsq, solve
from scipy import signal

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def _estimate_Gjw_from_csd(u, y, fs, nperseg=1024, noverlap=None, window='hann'):
    f, Puy = signal.csd(y, u, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    _, Puu = signal.csd(u, u, fs=fs, nperseg=nperseg, noverlap=noverlap, window=window)
    G = Puy / Puu
    w = 2*np.pi*f
    return w, G

def _build_ls_system(w, G, m, n, weights=None):
    """Build stacked Re/Im LS system A θ = b for
       (s^n + a1 s^{n-1}+...+ a_n) G = b0 s^m + ... + b_m
       unknown θ = [a1..a_n, b0..b_m]^T with real coefficients.
    """
    s = 1j*w
    K = len(w)
    # columns for [a1..a_n]
    A_den = np.column_stack([ (s**(n-i))*G for i in range(1, n+1) ])  # G*s^{n-1} ... G
    # columns for [b0..b_m] with negative sign moved to LHS
    A_num = -np.column_stack([ s**(m-j) for j in range(0, m+1) ])     # -s^m ... -1
    A_cplx = np.hstack([A_den, A_num])
    b_cplx = -(s**n)*G

    # Real+Imag stacking (optionally weighted)
    if weights is None:
        W = np.ones(K)
    else:
        W = np.asarray(weights).reshape(-1)
        if W.size != K:
            raise ValueError("weights must match len(w)")

    Ar = (A_cplx.real * W[:,None])
    Ai = (A_cplx.imag * W[:,None])
    br = (b_cplx.real * W)
    bi = (b_cplx.imag * W)
    A = np.vstack([Ar, Ai])
    b = np.hstack([br, bi])
    return A, b

def _ridge_solve(A, b, ridge):
    if ridge <= 0:
        theta, *_ = lstsq(A, b, rcond=None)
        cov = None
    else:
        # (A^T A + λI) θ = A^T b
        ATA = A.T @ A
        ATb = A.T @ b
        npar = ATA.shape[0]
        theta = solve(ATA + ridge*np.eye(npar), ATb)
        # crude covariance estimate using σ^2 from residuals
        r = b - A @ theta
        dof = max(1, A.shape[0] - npar)
        sigma2 = (r @ r) / dof
        cov = sigma2 * np.linalg.inv(ATA + ridge*np.eye(npar))
    return theta, cov

def _mirror_RHP_poles_to_LHP(den):
    roots = np.roots(den)
    roots_stable = []
    for p in roots:
        if p.real > 0:
            roots_stable.append(-p.real + 1j*p.imag)  # mirror across jω-axis
        else:
            roots_stable.append(p)
    den_stable = np.real_if_close(np.poly(roots_stable))
    return den_stable

def _discretize(numc, denc, fs):
    dt = 1.0/fs
    bz, az, _ = signal.cont2discrete((numc, denc), dt, method='zoh')[:3]
    return bz.squeeze(), az.squeeze()

def _simulate_discrete(bz, az, u):
    return signal.lfilter(bz, az, u)

def _metrics_time(y, yhat):
    y = np.asarray(y).reshape(-1)
    yhat = np.asarray(yhat).reshape(-1)
    r = y - yhat
    sse = float(np.sum(r**2))
    mse = sse / y.size
    rmse = np.sqrt(mse)
    var_y = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    var_r = float(np.var(r, ddof=1)) if y.size > 1 else 0.0
    ss_tot = float(np.sum((y - np.mean(y))**2))
    nmse = sse/ss_tot if ss_tot > 0 else np.nan               # in [0, ∞)
    r2 = 1 - nmse if np.isfinite(nmse) else np.nan
    vaf = 100 * (1 - var_r/var_y) if var_y > 0 else np.nan
    std_r = float(np.std(r, ddof=1)) if y.size > 1 else 0.0
    cov = np.cov(np.vstack([y, yhat])) if y.size > 1 else np.array([[np.nan, np.nan],[np.nan,np.nan]])
    return dict(SSE=sse, MSE=mse, RMSE=rmse, NMSE=nmse, R2=r2, VAF=vaf, std_resid=std_r, cov_y_yhat=cov)

def _freq_metrics(G, Ghat):
    e = G - Ghat
    nmse = np.sum(np.abs(e)**2) / np.sum(np.abs(G)**2)
    return dict(NMSE_freq=float(np.real_if_close(nmse)))

def _aic_bic(sse, n, k):
    # n: number of time samples, k: number of parameters
    if n <= k+1 or sse <= 0:
        return dict(AIC=np.nan, BIC=np.nan, AICc=np.nan)
    sigma2 = sse/n
    AIC = n*np.log(sigma2) + 2*k
    BIC = n*np.log(sigma2) + k*np.log(n)
    AICc = AIC + (2*k*(k+1))/(n-k-1)
    return dict(AIC=float(AIC), BIC=float(BIC), AICc=float(AICc))

# -----------------------------------------------------------------------------
# Main API
# -----------------------------------------------------------------------------

def fit_transfer_function_LS(
    m, n,
    # Training data (choose ONE of the two blocks)
    u=None, y=None, fs=None,                         # time-domain
    w=None, Gjw=None,                                # freq-domain
    # Frequency-estimation options (when using time-domain):
    nperseg=1024, noverlap=None, window='hann',
    # LS options
    weights=None, ridge=0.0, enforce_stability=True,
    # Validation on time-domain sets:
    tests=None,  # list of dicts: [{'u':u2, 'y':y2, 'fs':fs2}, ...]
    # Output control
    return_discrete=True
):
    """
    Fit a real-coefficient CT transfer function:
        G(s) = (b0 s^m + ... + b_m) / (s^n + a1 s^{n-1} + ... + a_n)

    Returns a dict with:
      - 'num','den' (continuous)
      - 'numz','denz','fs' (discrete, if return_discrete)
      - 'theta','cov','stderr'
      - 'freq' : {'w','G','Ghat','metrics'}
      - 'train','tests' : metrics dict(s)
      - 'info' : bookkeeping (m,n,ridge,enforce_stability)
      - 'criteria' : AIC/BIC/AICc computed from training residuals (time-domain)
    """
    # --- Build frequency data for training ---
    if w is None or Gjw is None:
        if (u is None) or (y is None) or (fs is None):
            raise ValueError("Provide either (u,y,fs) OR (w,Gjw) for training.")
        w, Gjw = _estimate_Gjw_from_csd(u, y, fs, nperseg=nperseg, noverlap=noverlap, window=window)

    # --- LS fit on stacked Re/Im ---
    A, b = _build_ls_system(w, Gjw, m, n, weights=weights)
    theta, cov = _ridge_solve(A, b, ridge=ridge)

    a = np.array(theta[:n])
    bcoef = np.array(theta[n:])
    den = np.hstack([1.0, a])
    num = bcoef.copy()

    if enforce_stability:
        den = _mirror_RHP_poles_to_LHP(den)

    # --- Fitted frequency response on training grid ---
    s = 1j*w
    num_poly = sum(num[j]*s**(m-j) for j in range(m+1))
    den_poly = sum(den[i]*s**(n-i) for i in range(n+1))
    Ghat = num_poly / den_poly

    # --- Coefficient standard errors (if cov available) ---
    if cov is not None:
        stderr = np.sqrt(np.maximum(np.diag(cov), 0.0))
    else:
        # estimate via normal equations if possible
        try:
            ATA = A.T @ A
            r = b - A @ theta
            dof = max(1, A.shape[0] - A.shape[1])
            sigma2 = (r @ r) / dof
            cov_ = sigma2 * np.linalg.inv(ATA)
            cov = cov_
            stderr = np.sqrt(np.maximum(np.diag(cov_), 0.0))
        except Exception:
            stderr, cov = None, None

    # --- Discretize for time-domain validation ---
    numz = denz = None
    if return_discrete and (fs is not None):
        numz, denz = _discretize(num, den, fs)

    # --- Time-domain metrics on training (if provided) ---
    train_metrics = {}
    criteria = {}
    if (u is not None) and (y is not None) and (fs is not None) and (numz is not None):
        yhat = _simulate_discrete(numz, denz, u)
        train_metrics = _metrics_time(y, yhat)
        criteria = _aic_bic(train_metrics['SSE'], n=len(y), k=(m+1+n))
    # Frequency metrics always
    freq_metrics = _freq_metrics(Gjw, Ghat)

    # --- Validate on tests (time-domain) ---
    test_metrics = []
    if tests:
        for T in tests:
            ut, yt, fst = T['u'], T['y'], T['fs']
            bz, az = numz, denz
            if fst != fs:
                # re-discretize if sample rate differs
                bz, az = _discretize(num, den, fst)
            yhat_t = _simulate_discrete(bz, az, ut)
            test_metrics.append(_metrics_time(yt, yhat_t))

    return dict(
        num=num, den=den, numz=numz, denz=denz, fs=fs,
        theta=theta, cov=cov, stderr=stderr,
        freq=dict(w=w, G=Gjw, Ghat=Ghat, metrics=freq_metrics),
        train=train_metrics, tests=test_metrics,
        info=dict(m=m, n=n, ridge=ridge, enforce_stability=enforce_stability),
        criteria=criteria
    )

# -----------------------------------------------------------------------------
# Optional convenience: sweep model orders and compare metrics
# -----------------------------------------------------------------------------

def sweep_orders(m_list, n_list, train, tests=None, **kwargs):
    """
    Quickly compare NMSE/VAF/AICc across (m,n).
    train = {'u':u,'y':y,'fs':fs} or {'w':w,'Gjw':G}
    """
    results = []
    for m in m_list:
        for n in n_list:
            R = fit_transfer_function_LS(
                m, n,
                u=train.get('u'), y=train.get('y'), fs=train.get('fs'),
                w=train.get('w'), Gjw=train.get('Gjw'),
                tests=tests, **kwargs
            )
            row = dict(m=m, n=n)
            row.update(dict(
                NMSE_train=R['train'].get('NMSE', np.nan),
                VAF_train=R['train'].get('VAF', np.nan),
                AICc=R['criteria'].get('AICc', np.nan),
                NMSE_freq=R['freq']['metrics']['NMSE_freq']
            ))
            # If tests exist, average their NMSE/VAF
            if R['tests']:
                nmse_t = np.mean([t['NMSE'] for t in R['tests']])
                vaf_t  = np.mean([t['VAF']  for t in R['tests']])
                row.update(dict(NMSE_test=nmse_t, VAF_test=vaf_t))
            results.append(row)
    return results

R = fit_transfer_function_LS(
    m=1, n=3,
    u=u_train, y=y_train, fs=fs,
    tests=[{'u':u_val1,'y':y_val1,'fs':fs},
           {'u':u_val2,'y':y_val2,'fs':fs}],
    nperseg=2048, ridge=1e-6, enforce_stability=True
)

print("Continuous TF:")
print("num:", R['num'])
print("den:", R['den'])
print("Train metrics:", R['train'])
print("Freq NMSE:", R['freq']['metrics'])
print("Test metrics:", R['tests'])
print("AICc:", R['criteria']['AICc'])

R = fit_transfer_function_LS(
    m=1, n=3,
    w=w, Gjw=G_meas,
    ridge=0.0, weights=None
)

table = sweep_orders(
    m_list=[0,1,2], n_list=[1,2,3,4],
    train={'u':u_train,'y':y_train,'fs':fs},
    tests=[{'u':u_val,'y':y_val,'fs':fs}],
    ridge=1e-6, enforce_stability=True
)
# Inspect `table`: pick the lowest order with good test VAF / NMSE and reasonable AICc.

