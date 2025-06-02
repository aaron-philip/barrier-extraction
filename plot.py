import numpy as np
import matplotlib.pyplot as plt

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

# Example usage:
# gp_model = train_gp(E_train, sigma_mu_train, delta_sigma_train)
# plot_gp_with_data(gp_model, E_train, sigma_mu_train, delta_sigma_train)
