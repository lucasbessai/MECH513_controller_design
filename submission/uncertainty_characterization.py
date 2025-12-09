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
import nominal_plant
import unc_bound

# Golden ratio (used for figure sizing)
gr = (1 + np.sqrt(5)) / 2

# %%
# Helper: fit a W2 to bound the worst residual for a chosen structure
def fit_and_plot(R_list, w, title, nW2, color_W='k'):
    # Worst residual envelope
    mag_max_dB, mag_max_abs = unc_bound.residual_max_mag(R_list, w)
    # Fit W2(s) (stable, min-phase) with chosen degree
    W2 = unc_bound.upperbound(omega=w, upper_bound=mag_max_abs, degree=nW2)
    # Plot residuals vs bound
    fig, ax = plt.subplots(figsize=(6.5, 6.5/gr))
    for R in R_list:
        ct.bode_plot(R, w, dB=True, deg=True, title='', plot_phase=False, linestyle='dashed', color='C0', label='')
    ax.semilogx(w, mag_max_dB, label=r'upper bound: $maxR|(j\omega)|$', color='C1')
    ct.bode_plot(W2, w, dB=True, deg=True, title='', plot_phase=False, label=r'fit bound: $W_2(j\omega)$', color='C3')
    ax.set_xlabel(r'$\omega$ (rad/s)')
    ax.set_ylabel(r'Residual (dB)')
    ax.set_title(f'{title} (order={nW2})')
    ax.legend()
    
    fig.tight_layout()
    # Tightness numbers
    magW, _, _ = ct.frequency_response(W2, w)
    slack_dB = float(np.max(20*np.log10(np.squeeze(magW)) - mag_max_dB))   # >0
    ratio_inf = float(np.max(np.squeeze(magW) / (mag_max_abs + 1e-16)))    # >=1
    return W2, slack_dB, ratio_inf


def compute_uncertainty_bounds(G_nom, G_off_nom, w_shared=None, deg=1):
    """Compute uncertainty bounds for different residual types.
    
    Parameters
    ----------
    G_nom : control.TransferFunction
        Nominal transfer function
    G_off_nom : List[control.TransferFunction]
        List of off-nominal transfer functions
    w_shared : np.ndarray, optional
        Frequency array for evaluation. Defaults to logspace(-2, 3, 600)
    deg : int, optional
        Degree of W2 fit. Defaults to 1
    
    Returns
    -------
    dict
        Dictionary containing all residual types and their bounds
    """
    if w_shared is None:
        w_shared = np.logspace(-2, 3, 600)
    
    # i) Additive:   R_k = P_k - P0
    R_add = [G - G_nom for G in G_off_nom]
    W2_add, slack_add, ratio_add = fit_and_plot(R_add, w_shared, 'Additive residuals: $|P_k - P_0|$', nW2=deg, color_W='k')

    # ii) Multiplicative:   R_k = P_k/P0 - 1
    R_mult = unc_bound.residuals(G_nom, G_off_nom)
    W2_mult, slack_mult, ratio_mult = fit_and_plot(R_mult, w_shared, 'Multiplicative residuals: $|P_k/P_0 - 1|$', nW2=deg, color_W='k')

    # iii) Inverse additive:   R_k = (P_k - P0)/(P_k P_0)
    R_inv_add = [(P - G_nom)/(P*G_nom) for P in G_off_nom]
    W2_inv_add, slack_inv_add, ratio_inv_add = fit_and_plot(R_inv_add, w_shared, 'Inverse-additive residuals: $|(P_k-P_0)/(P_k P_0)|$', nW2=deg, color_W='k')

    # iv) Inverse multiplicative:   R_k = (P_k - P0)/P_k
    R_inv_mult = [(P - G_nom)/P for P in G_off_nom]
    W2_inv_mult, slack_inv_mult, ratio_inv_mult = fit_and_plot(R_inv_mult, w_shared, 'Inverse-multiplicative residuals: $|(P_k-P_0)/P_k|$', nW2=deg, color_W='k')

    # Quick console report
    print("\n=== Residual bound tightness (smaller is better) ===")
    print(f"Additive              : slack = {slack_add:5.2f} dB   |  sup |W2|/max|Rk| = {ratio_add:5.2f}")
    print(f"Multiplicative        : slack = {slack_mult:5.2f} dB   |  sup |W2|/max|Rk| = {ratio_mult:5.2f}")
    print(f"Inverse additive      : slack = {slack_inv_add:5.2f} dB |  sup |W2|/max|Rk| = {ratio_inv_add:5.2f}")
    print(f"Inverse multiplicative: slack = {slack_inv_mult:5.2f} dB |  sup |W2|/max|Rk| = {ratio_inv_mult:5.2f}")

    return {
        'additive': {'R': R_add, 'W2': W2_add, 'slack': slack_add, 'ratio': ratio_add},
        'multiplicative': {'R': R_mult, 'W2': W2_mult, 'slack': slack_mult, 'ratio': ratio_mult},
        'inverse_additive': {'R': R_inv_add, 'W2': W2_inv_add, 'slack': slack_inv_add, 'ratio': ratio_inv_add},
        'inverse_multiplicative': {'R': R_inv_mult, 'W2': W2_inv_mult, 'slack': slack_inv_mult, 'ratio': ratio_inv_mult}
    }


if __name__ == "__main__":
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

    # Read in all input-output (IO) data
    path = pathlib.Path('MECH513_part1_load_data_sc/load_data_sc/SINE_SWEEP_DATA/')
    all_files = sorted(path.glob("*.csv"))
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

    # Load a dataset
    data_0 = data[0]
    data_1 = data[1]
    data_2 = data[2]
    data_3 = data[3]

    datasets = [data_0, data_1, data_2, data_3]

    # get nominal and off-nominals
    m, n = 0, 1

    G_off_nom_lab = nominal_plant.off_nominals(datasets, m=m, n=n, plot=False)
    G_off_nom = [G[1] for G in G_off_nom_lab]
    G_nom = nominal_plant.nominal_plant_median(datasets, m=m, n=n, plot=False)

    N = len(G_off_nom)
    labels = [f'G{i}' for i in range(N)]

    # Compute uncertainty bounds
    results = compute_uncertainty_bounds(G_nom, G_off_nom, deg=1)
    
    print("\nTransfer functions:")
    print(results['additive']['W2'])
    print(results['multiplicative']['W2'])
    print(results['inverse_additive']['W2'])
    print(results['inverse_multiplicative']['W2'])
