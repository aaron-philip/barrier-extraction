import numpy as np
import findiff as fd
import jax.numpy as jnp
import pandas as pd
import os
import sys
import logging
import h5py

import numpy as np
from scipy.interpolate import interp1d


def p_fus_flat(ecm,L,bL):
    eps=0.4
    return np.exp((ecm-bL)/eps)/(1+np.exp((ecm-bL)/eps))

def p_fus(ecm,L,df):
    eps=0.4
    return np.exp((ecm-bar(L,df))/eps)/(1+np.exp((ecm-bar(L,df))/eps))

def bar_flat(df):
    barriers = np.zeros((len(df),))
    prevl=0
    for i in range(len(df)):
        l=round(df["L"][i])
        if(prevl!=l):
            barriers[i] = df["Ecm"][i]
        else:
            barriers[i] = barriers[i-1]

        prevl = l
    return barriers

def bar(L,df):
    if(L-1 in df.L.values):
        barrier = df[df["L"] == L-1]["Ecm"].iloc[0]
    else:
        barrier=50.0
        # if(L-1 in df.L.values):
        #     barrier = df[df["L"] == L-1]["Ecm"].iloc[0]
        # else:
        #     if(L-2 in df.L.values):
        #         barrier = df[df["L"] == L-2]["Ecm"].iloc[0]
        #     else:
        #         barrier=1000.0

    return barrier
        
def coul(z1,z2,r):
    return z1*z2*1.439978408596513/r

def xsec(ecm,df,a1=17,a2=12):
    xsec=0.0
    mu=a1*a2/(a1+a2)
    for i in range(30):
        xsec += (2*i+1)*p_fus(ecm,i,df)

    return 10*xsec*np.pi*20.73553/(mu*ecm)
def dis_xsec(ecm,L,a1=17,a2=12):
    mu=a1*a2/(a1+a2)
    return 10*(L)**2*np.pi*20.73553/(mu*ecm)



def sum_xsec(ecm,pl,L,a1=17,a2=12):
    xsec=0.0
    mu=a1*a2/(a1+a2)
    if(L<len(pl)):
        for i in range(L):
            xsec += (2*i+1)*pl.iloc[i,6]
    else:
        for i in range(len(pl)):
            xsec += (2*i+1)*pl.iloc[i,6]
            
    return 10*xsec*np.pi*20.73553/(mu*ecm)
def sum_xsec_tdhf(ecm,L,a1=17,a2=12):
    xsec=0.0
    mu=a1*a2/(a1+a2)
    # print(L)
    for i in range(L):
        # print(i)
        xsec += (2*i+1)
    return 10*xsec*np.pi*20.73553/(mu*ecm)

def xsec_average(ecm,xsec,lb=12.0,ub=17.0):
    xsecsum = 0.0
    nxsec = 0
    for i in range(len(ecm)):
        if(ecm[i]>=lb and ecm[i]<=ub):
            nxsec += 1
            xsecsum += xsec[i]

    return xsecsum/nxsec

def centr(l,r,a1=17,a2=12):
    hbc  = 197.3269885804381
    amc2 = 938.918387
    amu = (a1*a2)/(a1+a2)

    return (hbc*hbc)*l*(l+1.0)/(2.0*amu*amc2*r**2)

def barrier_dist(e,sigma,acc=4):
    d = fd.FinDiff(0, e[1]-e[0], 2, acc=acc)
    barrier_dist = d(sigma*e)
    return (barrier_dist)

def barrier_dist_discrete(e,sigma, acc=4):
    d = fd.FinDiff(0, e, 2, acc=acc)
    barrier_dist = d(sigma*e)
    return (barrier_dist)

def point_diff(x,y,std,dE=1):
    der = np.zeros(x.shape)
    der_err = np.zeros(x.shape)
    for i in range(dE,x.shape[0]-dE):
        der[i] = (2*y[i] - y[i-dE] - y[i+dE]) / (abs(x[i] - x[i-dE]) * abs(x[i]-x[i+dE]))
        der_err[i] = std[i] * np.sqrt(6) * y[i] / (abs(x[i] - x[i-dE]) * abs(x[i]-x[i+dE]))
        # print(i,std[i],y[i],der_err[i])
    
    return der, der_err



def summarize(err):
    if err.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, median=np.nan, MAE=np.nan, RMSE=np.nan,
                    min=np.nan, max=np.nan)
    return dict(
        n      = int(err.size),
        mean   = float(np.mean(err)),
        std    = float(np.std(err, ddof=1)) if err.size > 1 else 0.0,
        median = float(np.median(err)),
        MAE    = float(np.mean(np.abs(err))),
        RMSE   = float(np.sqrt(np.mean(err**2))),
        min    = float(np.min(err)),
        max    = float(np.max(err)),
    )

def load_packed_hdf5(path):
    """
    Lazy-load packed Wong dataset stored in HDF5 format.
    Returns:
      - N: number of datasets
      - get_i(i): function returning (E_i, Sigma_i, Sigma_err_i, params_i)
    """
    f = h5py.File(path, "r")

    E_cat = f["E_cat"]
    Sigma_cat = f["Sigma_cat"]
    Sigma_err_cat = f["Sigma_err_cat"]
    idx_ptr = f["idx_ptr"][:]
    params = f["params"][:]

    N = len(idx_ptr) - 1

    def get_i(i):
        i0, i1 = idx_ptr[i], idx_ptr[i + 1]
        E_i = E_cat[i0:i1]
        S_i = Sigma_cat[i0:i1]
        Se_i = Sigma_err_cat[i0:i1]
        p_i = params[i]
        order = np.argsort(E_i)
        return E_i[order], S_i[order], Se_i[order], p_i

    return {"N": N, "get_i": get_i, "file": f}



def compute_all_metrics(E, y_true, y_mean, y_samples, levels=(0.5, 0.68, 0.9, 0.95),
                        x_min=None, x_max=None):
    """
    Compute all cross-section and barrier-distribution metrics.

    Parameters
    ----------
    E : array (n_E,)
        Energy grid.
    y_true : array (n_E,)
        True sigma(E) values.
    y_mean : array (n_E,)
        Mean predicted sigma(E).
    y_samples : array (n_samples, n_E)
        Sampled sigma(E) predictions from posterior or MC draws.
    levels : sequence of floats
        Confidence levels for empirical coverage.
    x_min, x_max : floats, optional
        Range restriction for peak metrics.

    Returns
    -------
    dict
        Dictionary with sigma-space and barrier-space metrics plus empirical coverages.
    """
    # --------- pointwise errors in σ-space ---------
    err = y_mean - y_true
    L1 = float(np.mean(np.abs(err)))
    L2 = float(np.sqrt(np.mean(err**2)))

    # optional domain restriction
    mask = np.ones_like(E, dtype=bool)
    if x_min is not None:
        mask &= (E >= x_min)
    if x_max is not None:
        mask &= (E <= x_max)
    E_mask = E[mask]
    if E_mask.size < 2:
        return {"sigma_L1": L1, "sigma_L2": L2}

    order = np.argsort(E_mask)
    E_mask = E_mask[order]
    y_true_m = y_true[mask][order]
    y_mean_m = y_mean[mask][order]
    y_samp_m = y_samples[:, mask][:, order]

    def get_peak_metrics(E_arr, Y_arr):
        f = interp1d(E_arr, Y_arr, kind='cubic', bounds_error=False, fill_value="extrapolate")
        Xq = np.linspace(E_arr[0], E_arr[-1], max(200, 10*len(E_arr)))
        Yq = f(Xq)
        i_pk = np.argmax(Yq)
        pk_loc, pk_ht = Xq[i_pk], Yq[i_pk]
        half = 0.5 * pk_ht
        z = Yq - half
        cross = np.where(np.signbit(z[:-1]) != np.signbit(z[1:]))[0]
        if len(cross) >= 2:
            x0 = Xq[cross[0]]
            x1 = Xq[cross[-1]+1]
            fwhm = x1 - x0
        else:
            fwhm = np.nan
        return pk_loc, pk_ht, fwhm

    # --- σ-space peak metrics ---
    t_loc, t_ht, t_fwhm = get_peak_metrics(E_mask, y_true_m)
    p_loc, p_ht, p_fwhm = get_peak_metrics(E_mask, y_mean_m)

    sigma_metrics = dict(
        sigma_L1=L1,
        sigma_L2=L2,
        sigma_peak_loc_resid=p_loc - t_loc,
        sigma_peak_height_resid=p_ht - t_ht,
        sigma_fwhm_resid=(p_fwhm - t_fwhm)
    )

    # --------- Empirical coverage in σ-space ---------
    coverages = {}
    for alpha in levels:
        lower = np.percentile(y_samp_m, (1 - alpha) / 2 * 100, axis=0)
        upper = np.percentile(y_samp_m, (1 + alpha) / 2 * 100, axis=0)
        inside = (y_true_m >= lower) & (y_true_m <= upper)
        coverages[f"sigma_emp_cov_{int(alpha*100)}"] = np.mean(inside)

    # --------- Barrier distributions ---------
    D_true = barrier_dist(E_mask, y_true_m)
    D_mean = barrier_dist(E_mask, y_mean_m)
    D_samps = np.array([barrier_dist(E_mask, ys) for ys in y_samp_m])

    D_err = D_mean - D_true
    DL1 = float(np.mean(np.abs(D_err)))
    DL2 = float(np.sqrt(np.mean(D_err**2)))

    t_loc, t_ht, t_fwhm = get_peak_metrics(E_mask, D_true)
    p_loc, p_ht, p_fwhm = get_peak_metrics(E_mask, D_mean)

    barrier_metrics = dict(
        barrier_L1=DL1,
        barrier_L2=DL2,
        barrier_peak_loc_resid=p_loc - t_loc,
        barrier_peak_height_resid=p_ht - t_ht,
        barrier_fwhm_resid=(p_fwhm - t_fwhm)
    )

    # --- Empirical coverage in barrier space ---
    for alpha in levels:
        lower = np.percentile(D_samps, (1 - alpha) / 2 * 100, axis=0)
        upper = np.percentile(D_samps, (1 + alpha) / 2 * 100, axis=0)
        inside = (D_true >= lower) & (D_true <= upper)
        coverages[f"barrier_emp_cov_{int(alpha*100)}"] = np.mean(inside)

    out = {}
    out.update(sigma_metrics)
    out.update(barrier_metrics)
    out.update(coverages)
    return out

def setup_logger(
    name=None,
    dataset_path=None,
    log_dir="/mnt/home/philipaa/scratch/barrier-paper-logs",
    tag=None,
    level=logging.INFO,
):
    """
    General-purpose logger setup for SLURM or local runs.

    Parameters
    ----------
    name : str, optional
        A base name for the logger (e.g., 'train_bnn', 'train_gp').
        If None, derived from dataset basename.
    dataset_path : str, optional
        Path to dataset file (used to generate unique log filename).
    log_dir : str, default
        Directory where log files are stored.
    tag : str, optional
        Optional short string to prefix the log file (e.g. 'bnn', 'gp', 'wong').
    level : logging level, default=logging.INFO
        Minimum log level.

    Returns
    -------
    logger : logging.Logger
        Configured logger that writes to both file and stdout.
    """

    os.makedirs(log_dir, exist_ok=True)

    # === Determine unique identifiers ===
    base = "process"
    if dataset_path:
        base = os.path.basename(dataset_path).replace(".npz", "")
    if name:
        base = f"{name}_{base}"

    task_id = os.environ.get("SLURM_ARRAY_TASK_ID", os.getpid())
    prefix = f"{tag}_" if tag else ""

    logfile = os.path.join(log_dir, f"{prefix}{base}_task{task_id}.log")

    # === Configure logger ===
    logger = logging.getLogger(f"{prefix}{base}_{task_id}")
    logger.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # === File + stream handlers ===
    fh = logging.FileHandler(logfile)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)

    # Prevent multiple handlers on re-import
    if not logger.handlers:
        logger.addHandler(fh)
        logger.addHandler(sh)

    logger.propagate = False
    logger.info(f"Logger initialized -> {logfile}")
    return logger


def _interp_crossing(x0, y0, x1, y1, y_target):
    """Linear interpolation for y(x)=y_target between two points (x0,y0),(x1,y1)."""
    if y1 == y0:
        return 0.5*(x0+x1)
    t = (y_target - y0) / (y1 - y0)
    return x0 + t*(x1 - x0)

def peak_metrics_anchored(E, Y, E0_true, window=6.0, smooth=None):
    """
    Find peak location/height/FWHM near E0_true, restricted to a window.

    Parameters
    ----------
    E, Y : (n,) arrays
    E0_true : float
    window : float
        Search in [E0_true-window, E0_true+window]
    smooth : int or None
        If int >= 3, apply simple moving-average smoothing of that width
        (helps when D(E) from samples is noisy).

    Returns
    -------
    (pk_loc, pk_ht, fwhm)
    """
    E = np.asarray(E, float)
    Y = np.asarray(Y, float)
    order = np.argsort(E)
    E, Y = E[order], Y[order]

    m = (E >= E0_true - window) & (E <= E0_true + window)
    if m.sum() < 5:
        return np.nan, np.nan, np.nan

    Ew = E[m]
    Yw = Y[m].copy()

    # Optional smoothing (moving average)
    if smooth is not None and int(smooth) >= 3 and int(smooth) < Yw.size:
        k = int(smooth)
        kern = np.ones(k)/k
        Yw = np.convolve(Yw, kern, mode="same")

    i_pk = int(np.argmax(Yw))
    pk_loc = float(Ew[i_pk])
    pk_ht  = float(Yw[i_pk])

    if not np.isfinite(pk_ht) or pk_ht <= 0:
        return pk_loc, pk_ht, np.nan

    half = 0.5 * pk_ht

    # Find left crossing: scan from peak down to left
    left = np.nan
    for i in range(i_pk, 0, -1):
        if (Yw[i] >= half) and (Yw[i-1] < half):
            left = _interp_crossing(Ew[i-1], Yw[i-1], Ew[i], Yw[i], half)
            break

    # Find right crossing: scan from peak down to right
    right = np.nan
    for i in range(i_pk, Yw.size - 1):
        if (Yw[i] >= half) and (Yw[i+1] < half):
            right = _interp_crossing(Ew[i], Yw[i], Ew[i+1], Yw[i+1], half)
            break

    if np.isfinite(left) and np.isfinite(right) and right > left:
        fwhm = float(right - left)
    else:
        fwhm = np.nan

    return pk_loc, pk_ht, fwhm


import numpy as np

def empirical_coverage(y_true, y_samples, level=0.68):
    lo = np.percentile(y_samples, (1-level)/2*100, axis=0)
    hi = np.percentile(y_samples, (1+level)/2*100, axis=0)
    return float(np.mean((y_true >= lo) & (y_true <= hi)))

def mae_rmse(y_true, y_pred):
    err = (y_pred - y_true)
    return float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err**2)))

def compute_metrics_for_fit(
    x_pred, y_mean, y_samples,
    E_data,
    E0_true, R0_true, hw_true,
    sigma_true_fn, barrier_from_sigma_samples_fn,
    pad=0.0,
    levels=(0, 0.25, 0.5, 0.68, 0.9, 0.95, 1.0),
    peak_window=6.0,
    barrier_smooth=5,
):
    """
    Computes sigma-space + barrier-space MAE/RMSE, empirical coverage, and anchored peak metrics.
    Region is [min(E_data)-pad, max(E_data)+pad] evaluated on x_pred.
    """
    xmin = float(np.min(E_data) - pad)
    xmax = float(np.max(E_data) + pad)
    m = (x_pred >= xmin) & (x_pred <= xmax)
    if m.sum() < 10:
        return {}

    x = x_pred[m]
    ym = y_mean[m]
    ys = y_samples[:, m]

    # true sigma on x_pred region
    ytrue = sigma_true_fn(x, E0=E0_true, R0=R0_true, hbarw=hw_true)

    sigma_mae, sigma_rmse = mae_rmse(ytrue, ym)
    out = {
        "x_min": xmin,
        "x_max": xmax,
        "sigma_MAE": sigma_mae,
        "sigma_RMSE": sigma_rmse,
    }

    for a in levels:
        out[f"sigma_emp_cov_{int(a*100)}"] = empirical_coverage(ytrue, ys, level=a)

    # Barrier distributions from sigma samples on the same x grid
    D_samps = barrier_from_sigma_samples_fn(x, ys)                  # (nsamp, nx)
    D_mean  = barrier_from_sigma_samples_fn(x, ym[None, :])[0]      # (nx,)
    D_true  = barrier_from_sigma_samples_fn(x, ytrue[None, :])[0]   # (nx,)

    b_mae, b_rmse = mae_rmse(D_true, D_mean)
    out.update({
        "barrier_MAE": b_mae,
        "barrier_RMSE": b_rmse,
    })

    for a in levels:
        out[f"barrier_emp_cov_{int(a*100)}"] = empirical_coverage(D_true, D_samps, level=a)

    # Anchored peak metrics (in barrier space; that’s what you care about)
    t_loc, t_ht, t_fwhm = peak_metrics_anchored(x, D_true, E0_true, window=peak_window, smooth=barrier_smooth)
    p_loc, p_ht, p_fwhm = peak_metrics_anchored(x, D_mean, E0_true, window=peak_window, smooth=barrier_smooth)

    out.update({
        "barrier_peak_loc_true": t_loc,
        "barrier_peak_ht_true": t_ht,
        "barrier_fwhm_true": t_fwhm,
        "barrier_peak_loc_pred": p_loc,
        "barrier_peak_ht_pred": p_ht,
        "barrier_fwhm_pred": p_fwhm,
        "barrier_peak_loc_resid": (p_loc - t_loc) if np.isfinite(p_loc) and np.isfinite(t_loc) else np.nan,
        "barrier_peak_ht_resid":  (p_ht - t_ht)  if np.isfinite(p_ht)  and np.isfinite(t_ht)  else np.nan,
        "barrier_fwhm_resid":     (p_fwhm - t_fwhm) if np.isfinite(p_fwhm) and np.isfinite(t_fwhm) else np.nan,
    })

    return out
