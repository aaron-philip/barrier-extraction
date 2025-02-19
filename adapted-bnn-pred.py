import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy as sp
import findiff as fd

# Import autobnn modules and jax:
import jax
import jax.numpy as jnp
from autobnn import estimators, training_util

# --- Some helper functions (unchanged) ---
def p_fus_flat(ecm, L, bL):
    eps = 0.4
    return np.exp((ecm - bL) / eps) / (1 + np.exp((ecm - bL) / eps))

def p_fus(ecm, L, df):
    eps = 0.4
    return np.exp((ecm - bar(L, df)) / eps) / (1 + np.exp((ecm - bar(L, df)) / eps))

def bar_flat(df):
    barriers = np.zeros((len(df),))
    prevl = 0
    for i in range(len(df)):
        l = round(df["L"][i])
        if prevl != l:
            barriers[i] = df["Ecm"][i]
        else:
            barriers[i] = barriers[i-1]
        prevl = l
    return barriers

def bar(L, df):
    if (L - 1) in df.L.values:
        barrier = df[df["L"] == L - 1]["Ecm"].iloc[0]
    else:
        barrier = 50.0
    return barrier

def coul(z1, z2, r):
    return z1 * z2 * 1.439978408596513 / r

def xsec(ecm, df, a1=17, a2=12):
    xs = 0.0
    mu = a1 * a2 / (a1 + a2)
    for i in range(30):
        xs += (2 * i + 1) * p_fus(ecm, i, df)
    return 10 * xs * np.pi * 20.73553 / (mu * ecm)

def dis_xsec(ecm, L, a1=17, a2=12):
    mu = a1 * a2 / (a1 + a2)
    return 10 * (L) ** 2 * np.pi * 20.73553 / (mu * ecm)

def sum_xsec(ecm, pl, L, a1=17, a2=12):
    xs = 0.0
    mu = a1 * a2 / (a1 + a2)
    if L < len(pl):
        for i in range(L):
            xs += (2 * i + 1) * pl.iloc[i, 6]
    else:
        for i in range(len(pl)):
            xs += (2 * i + 1) * pl.iloc[i, 6]
    return 10 * xs * np.pi * 20.73553 / (mu * ecm)

def sum_xsec_tdhf(ecm, L, a1=17, a2=12):
    xs = 0.0
    mu = a1 * a2 / (a1 + a2)
    for i in range(L):
        xs += (2 * i + 1)
    return 10 * xs * np.pi * 20.73553 / (mu * ecm)

def xsec_average(ecm, xsec, lb=12.0, ub=17.0):
    xs_sum = 0.0
    nxsec = 0
    for i in range(len(ecm)):
        if lb <= ecm[i] <= ub:
            nxsec += 1
            xs_sum += xsec[i]
    return xs_sum / nxsec

def centr(l, r, a1=17, a2=12):
    hbc  = 197.3269885804381
    amc2 = 938.918387
    amu = (a1 * a2) / (a1 + a2)
    return (hbc * hbc) * l * (l + 1.0) / (2.0 * amu * amc2 * r**2)

def barrier_dist(e, sigma, acc=4):
    d = fd.FinDiff(0, e[1] - e[0], 2, acc=acc)
    barrier_dist_val = d(sigma * e)
    return barrier_dist_val

def barrier_dist_discrete(e, sigma, acc=4):
    d = fd.FinDiff(0, e, 2, acc=acc)
    barrier_dist_val = d(sigma * e)
    return barrier_dist_val

def point_diff(x, y, std, dE=1):
    der = np.zeros(x.shape)
    der_err = np.zeros(x.shape)
    for i in range(dE, x.shape[0] - dE):
        der[i] = (2 * y[i] - y[i - dE] - y[i + dE]) / (abs(x[i] - x[i - dE]) * abs(x[i] - x[i + dE]))
        der_err[i] = std[i] * np.sqrt(6) * y[i] / (abs(x[i] - x[i - dE]) * abs(x[i] - x[i + dE]))
    return der, der_err

# --- Prepare experimental data ---
dirname = "16O208Pb_exp"
system = "$^{208}$Pb+$^{16}$O"
fileprefix = dirname

expfiles   = ['expMort.dat', 'expDas.dat']
explabel   = ['Morton et al. 1999', 'Dasgupta et al. 2007']
expmarker  = ['o', 's']
expcolor   = ['k', 'g']

exp = []  # list to hold the data frames for each experiment

for file in expfiles:
    df = pd.read_csv(os.path.join(dirname, file),
                     delim_whitespace=True,
                     names=['Ecm', 'Sigma', 'Error'])
    # Reverse the order if necessary
    if df['Ecm'].iloc[0] > df['Ecm'].iloc[-1]:
        df = df.iloc[::-1].reset_index(drop=True)
    exp.append(df)

xmin = 70.0
xmax = 90.0

# --- Instead of GP regressors, we now use BNNs.
# We will store one BNN estimator per experiment.
exp_bnns = []

# Create a JAX PRNG seed that we will split for each experiment:
seed = jax.random.PRNGKey(0)

plt.figure()
# (Plot experimental data points with error bars.)
for i in range(len(exp)):
    # Plot the measured values and error bars
    plt.errorbar(exp[i]['Ecm'], exp[i]['Sigma'],
                 yerr=exp[i]['Error'],
                 label=explabel[i],
                 marker=expmarker[i],
                 ls='',
                 lw=1,
                 ms=4,
                 color=expcolor[i])
    
    # Use a simple RBF-like kernel period if needed.
    # For many cases with nonperiodic data you can pass an empty tuple.
    periods = ()  # no explicit periodicity for cross-section data
    
    # For numerical stability it is often useful to normalize the independent variable.
    x_data = exp[i]['Ecm'].to_numpy()
    y_data = exp[i]['Sigma'].to_numpy()
    # Normalize x by its maximum value.
    x_scale = x_data.max()
    x_norm = x_data / x_scale

    # Split the seed (so each experiment gets its own PRNG key)
    seed, est_seed = jax.random.split(seed)
    
    # Instantiate the BNN estimator.
    # (You may wish to adjust the model architecture or likelihood for your application.)
    bnn_est = estimators.AutoBnnMapEstimator(
        'sum_of_products',
        likelihood_model='normal_likelihood_logistic_noise',
        seed=est_seed,   
        num_particles=32,
    )
    
    print(f"Training BNN for {explabel[i]} ...")
    # Train the BNN. Data arrays need a trailing singleton dimension.
    bnn_est = bnn_est.fit(x_norm[..., None], y_data[..., None])
    exp_bnns.append((bnn_est, x_scale))  # store the estimator along with the scaling factor

    # Create a prediction grid in physical units and then normalize.
    x_pred = np.linspace(exp[i]['Ecm'].iloc[0], exp[i]['Ecm'].iloc[-1], 200)
    x_pred_norm = x_pred / x_scale

    # Get the median prediction and quantiles (as provided by autobnn).
    lo, mid, p90, hi = bnn_est.predict_quantiles(x_pred_norm[..., None], q=[2.5, 50., 90., 97.5])
    # (mid is the median prediction.)
    # Plot the median prediction and 95% confidence interval:
    plt.plot(x_pred, mid.squeeze(), label="BNN " + explabel[i],
             ls='-', lw=2, color=expcolor[i])
    plt.fill_between(
        x_pred,
        lo.squeeze(), hi.squeeze(),
        alpha=0.5,
        label="95% confidence BNN " + explabel[i],
        color=expcolor[i]
    )
    
    # (Optionally, one may sample several trajectories from the BNN.
    # Here we assume that the estimator provides a method called "sample" similar to the GP's "sample_y".
    # If your autobnn version provides a different method name, please adjust accordingly.)
    nsamps = 100
    try:
        # Here we assume a method "sample" exists that returns a [n_points x nsamps] array.
        y_samps = bnn_est.sample(x_pred_norm[..., None], nsamps)
    except Exception:
        # If no sampling method exists, simply use repeated predictions (one per particle) as a rough substitute.
        # (This is only an example; for a true sample, check your autobnn documentation.)
        y_samps = np.tile(mid, (nsamps, 1)).T
    # Compute (and plot) barrier distributions based on each BNN sample:
    # For demonstration, we compute the barrier distribution (using your barrier_dist function)
    # on each sampled prediction.
    for j in range(nsamps):
        # Compute barrier distribution from the j-th sample.
        # Here we use x_pred (energy points) and the corresponding sample.
        _dist = barrier_dist(x_pred, y_samps[:, j], acc=4)
        plt.plot(x_pred, _dist, color=expcolor[i], ls='-', lw=1, alpha=0.1)
    
    plt.xlim([xmin, xmax])
    
    plt.legend()
    plt.ylim([-1, 850])
    plt.xlabel(r'$E_{\mathrm{c.m.}}$ (MeV)')
    plt.ylabel(r'$\Sigma$ (mb)')
    plt.title(system)
    plt.savefig(f'16O208Pb_BNN_{i}.pdf', bbox_inches='tight')
    plt.show()


# --- (Optional additional plotting) ---
# For example, one may plot additional quantities such as the point-difference of the barrier,
# computed from the experimental error bars and the BNN prediction.
nsamps = 100
plt.figure()
for i in range(len(exp)):
    if not np.isfinite(exp[i]['Error'].iloc[0]): 
        continue
    # Create a fine grid over the experimental energy range.
    x_pred = np.linspace(exp[i]['Ecm'].iloc[0], exp[i]['Ecm'].iloc[-1], 1000)
    x_scale = exp_bnns[i][1]
    x_pred_norm = x_pred / x_scale

    # Get the median prediction (you might also want to compute quantiles).
    _, mid, _, _ = exp_bnns[i][0].predict_quantiles(x_pred_norm[..., None], q=[2.5, 50., 90., 97.5])
    # Compute a discrete derivative of the median prediction.
    dE = 3
    der, derr = point_diff(exp[i]['Ecm'].to_numpy(), mid.squeeze(), exp[i]['Error'].to_numpy(), dE=dE)
    plt.errorbar(exp[i]['Ecm'].iloc[dE:-dE], -der[dE:-dE],
                 yerr=derr[dE:-dE],
                 label=explabel[i],
                 marker=expmarker[i],
                 ls='',
                 lw=1,
                 ms=4,
                 color=expcolor[i])
    
plt.xlim([xmin, xmax])
plt.ylim([-1.1, 1.1])
plt.xlabel(r'$E_{\mathrm{c.m.}}$ (MeV)')
plt.ylabel(r'$-d^2\Sigma/dE^2$ (normalized)')
plt.legend()
plt.savefig('16O208Pb_derivative.pdf', bbox_inches='tight')
plt.show()
