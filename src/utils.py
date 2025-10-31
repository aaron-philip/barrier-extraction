import numpy as np
import findiff as fd
import jax.numpy as jnp
import pandas as pd
import os
import sys
import logging


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


def compute_peak_metrics(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 65.0,
    x_max: float = 80.0,
    kind: str = "cubic",           # 'linear', 'quadratic', or 'cubic'
    upsample_factor: float = 10.0  # how dense to resample
):
    """
    Returns (peak_loc, peak_height, fwhm) by:
      1) restricting to [x_min, x_max],
      2) interpolating with scipy.interpolate.interp1d,
      3) evaluating on a dense grid,
      4) computing peak and FWHM on the interpolated curve.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    # restrict to subrange
    mask = np.ones_like(x, dtype=bool)
    if x_min is not None:
        mask &= (x >= x_min)
    if x_max is not None:
        mask &= (x <= x_max)
    x_sub, y_sub = x[mask], y[mask]

    if x_sub.size < 2:
        return np.nan, np.nan, np.nan

    # sort by x
    order = np.argsort(x_sub)
    x_sub, y_sub = x_sub[order], y_sub[order]

    # interpolation function
    f = interp1d(x_sub, y_sub, kind=kind, bounds_error=False, fill_value="extrapolate")

    # dense grid for evaluation
    xmin, xmax = float(x_sub[0]), float(x_sub[-1])
    n_query = max(int(np.ceil(upsample_factor * len(x_sub))), 200)
    Xq = np.linspace(xmin, xmax, n_query)
    Yq = f(Xq)

    # peak
    i = int(np.argmax(Yq))
    peak_loc = float(Xq[i])
    peak_height = float(Yq[i])

    # FWHM
    half = 0.5 * peak_height
    z = Yq - half
    sign = np.signbit(z)
    cross_idxs = np.where(sign[:-1] != sign[1:])[0]

    crossings = []
    for k in cross_idxs:
        x0, x1 = Xq[k], Xq[k+1]
        z0, z1 = z[k], z[k+1]
        if z1 != z0:
            x_cross = x0 - z0 * (x1 - x0) / (z1 - z0)
            crossings.append(x_cross)

    if len(crossings) >= 2:
        fwhm = float(np.max(crossings) - np.min(crossings))
    else:
        fwhm = np.nan

    return peak_loc, peak_height, fwhm

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
