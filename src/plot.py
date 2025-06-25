import numpy as np
import matplotlib.pyplot as plt
import os

import scienceplots
plt.style.use(['science', 'no-latex', 'ieee'])
plt.rc('axes', labelsize=14) 
plt.rcParams['axes.titlesize'] = 20
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=10)  

def plot_gp_with_data(gp, E_plot, E_train, sigma_mu_train, delta_sigma_train):
    """
    Plot GP predictions with 95% confidence interval on a fine grid,
    and overlay the training data with error bars.

    Parameters
    ----------
    gp : GaussianProcessRegressor
        A fitted GP model (returned by train_gp).
    E_train : array-like, shape (N,)
        Testing point energies.
    sigma_mu_train : array-like, shape (N,)
        Mean cross-section values at E_train.
    delta_sigma_train : array-like, shape (N,)
        Variance of cross-section at E_train (heteroscedastic noise).
    num_points : int, optional (default=300)
        Number of points to use for the fine prediction grid.
    """

    # 2) Predict with the GP: get mean and standard deviation
    mu_pred, std_pred = gp.predict(E_plot.reshape(-1, 1), return_std=True)
    
    # 3) Plot the GP mean curve
    plt.plot(E_plot, mu_pred, 'b-', lw=2, label='GP mean')
    
    # 4) Shade the 95% confidence interval: mu ± 1.96 * std
    upper = mu_pred + 1.96 * std_pred
    lower = mu_pred - 1.96 * std_pred
    plt.fill_between(E_plot, lower, upper, color='blue', alpha=0.3, label='95% CI')
    
    # 5) Overlay the training data points with error bars
    #    Use sqrt(delta_sigma_train) as the error in y
    y_err = np.sqrt(delta_sigma_train)
    plt.errorbar(
        E_train, sigma_mu_train, 
        yerr=y_err, fmt='ro', markersize=5, 
        capsize=3, label='Training data'
    )
    
    # 6) Labels and legend
    plt.xlabel("E (MeV)")
    plt.ylabel(r"$\sigma(E)$")
    plt.title("GP fit with 95% confidence band + training data")
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()

def plot_D_f(E_train, d2_train, E_pred, D_f_sampled_preds, title="Barrier Distribution", 
             title_fontsize=12, axes_fontsize=10, legend_fontsize=12,xlims=None, ylims=[-1000,1500], save_dir = None, save_extra=None, legend=False):
    plt.figure(figsize=(5,3))
    for p in range(D_f_sampled_preds.shape[0]):
        plt.plot(E_pred, D_f_sampled_preds[p], lw=1, alpha=0.1, color='b')
    mean_deriv = D_f_sampled_preds.mean(axis=0)
    plt.plot(E_pred, mean_deriv, lw=1, alpha=0.9, color='b', label=r"Sampled $D_f(E)$")
    plt.scatter(E_train, d2_train, color='r', s=5, label=r"Training $D_f(E)$")
    plt.title(title, fontsize=title_fontsize)
    plt.xlabel(r'$E_{\mathrm{cm}}$ (MeV)', fontsize=axes_fontsize)
    plt.ylabel(r'$D_f(E)$ (mb/MeV)', fontsize=axes_fontsize)
    plt.ylim(ylims)
    if xlims:
        plt.xlim(xlims)
    else:
        plt.xlim(E_pred[0], E_pred[-1])
    if legend:
        plt.legend(fontsize=legend_fontsize)
    #plt.tight_layout()

    if save_dir: 
        if save_extra:
            out_png = os.path.join(save_dir, f"plot_{save_extra}.png")
        else: 
            out_png = os.path.join(save_dir, f"plot_{save_extra}.png")
        plt.savefig(out_png) 

# Example usage:
# gp_model = train_gp(E_train, sigma_mu_train, delta_sigma_train)
# plot_gp_with_data(gp_model, E_train, sigma_mu_train, delta_sigma_train)
