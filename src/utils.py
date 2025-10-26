import numpy as np
import findiff as fd
import jax.numpy as jnp
import pandas as pd
import os


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


def compute_all_metrics(true_curve, pred_curve, E, x_min=None, x_max=None):
    """
    Compute error metrics comparing a predicted barrier distribution to the true barrier distribution.
    - L1: mean absolute error over the entire function.
    - L2: root mean squared error over the entire function.
    - peak_loc_resid: difference in barrier peak location (predicted - true).
    - peak_height_resid: difference in barrier peak height (predicted - true).
    - fwhm_resid: difference in barrier distribution FWHM (predicted - true).
    The comparison can be restricted to [x_min, x_max] (energy range) for peak calculations if specified.
    Returns a dictionary of metrics.
    """
    true_curve = np.asarray(true_curve)
    pred_curve = np.asarray(pred_curve)
    err = pred_curve - true_curve
    # L1 and L2 errors
    L1 = float(np.mean(np.abs(err)))
    L2 = float(np.sqrt(np.mean(err**2)))
    # Restrict to specified range for peak metrics (defaults to full range if not given)
    mask = np.ones_like(E, dtype=bool)
    if x_min is not None:
        mask &= (E >= x_min)
    if x_max is not None:
        mask &= (E <= x_max)
    E_masked = E[mask]
    if E_masked.size < 2:
        # Not enough points in range to compute peak metrics
        return {
            "L1": L1,
            "L2": L2,
            "peak_loc_resid": np.nan,
            "peak_height_resid": np.nan,
            "fwhm_resid": np.nan
        }
    # Sort by energy (just to be safe in case E is not sorted)
    order = np.argsort(E_masked)
    E_masked = E_masked[order]
    true_masked = true_curve[mask][order]
    pred_masked = pred_curve[mask][order]
    # Interpolate each curve on a dense grid for peak finding
    def get_peak_metrics(E_arr, Y_arr):
        f = interp1d(E_arr, Y_arr, kind='cubic', bounds_error=False, fill_value="extrapolate")
        Nq = max(int(np.ceil(10 * len(E_arr))), 200)  # upsample factor 10 or at least 200 points
        Xq = np.linspace(E_arr[0], E_arr[-1], Nq)
        Yq = f(Xq)
        # Peak location and height
        i_peak = int(np.argmax(Yq))
        peak_loc = float(Xq[i_peak])
        peak_height = float(Yq[i_peak])
        # Full Width at Half Maximum (FWHM)
        half = 0.5 * peak_height
        z = Yq - half
        sign = np.signbit(z)
        cross_idxs = np.where(sign[:-1] != sign[1:])[0]
        crossings = []
        for k in cross_idxs:
            # Linear interpolation to find crossing point where Y = half
            x0, x1 = Xq[k], Xq[k+1]
            z0, z1 = z[k], z[k+1]
            if z1 != z0:
                x_cross = x0 - z0 * (x1 - x0) / (z1 - z0)
                crossings.append(x_cross)
        if len(crossings) >= 2:
            fwhm_val = float(np.max(crossings) - np.min(crossings))
        else:
            fwhm_val = np.nan
        return peak_loc, peak_height, fwhm_val
    true_peak_loc, true_peak_height, true_fwhm = get_peak_metrics(E_masked, true_masked)
    pred_peak_loc, pred_peak_height, pred_fwhm = get_peak_metrics(E_masked, pred_masked)
    # Compute residuals (predicted minus true)
    peak_loc_resid = float(pred_peak_loc - true_peak_loc)
    peak_height_resid = float(pred_peak_height - true_peak_height)
    fwhm_resid = float(pred_fwhm - true_fwhm) if (not np.isnan(pred_fwhm) and not np.isnan(true_fwhm)) else np.nan
    return {
        "L1": L1,
        "L2": L2,
        "peak_loc_resid": peak_loc_resid,
        "peak_height_resid": peak_height_resid,
        "fwhm_resid": fwhm_resid
    }





"""
def compute_peak_metrics(
    x: np.ndarray,
    y: np.ndarray,
    x_min: float = 65.0,
    x_max: float = 80.0
):
    """
"""   Given arrays x (shape [n,]) and y (shape [n,]), find within [x_min, x_max]:
      - peak_loc: x at max(y)
      - peak_height: max(y)
      - fwhm: full width at half maximum
    Returns (peak_loc, peak_height, fwhm)."""
"""
    x = np.asarray(x)
    y = np.asarray(y)
    # restrict to subrange
    if x_min is not None or x_max is not None:
        mask = np.ones_like(x, dtype=bool)
        if x_min is not None:
            mask &= (x >= x_min)
        if x_max is not None:
            mask &= (x <= x_max)
        x_sub = x[mask]
        y_sub = y[mask]
    else:
        x_sub, y_sub = x, y

    if x_sub.size == 0:
        return np.nan, np.nan, np.nan

    # find peak within subrange
    idx = np.argmax(y_sub)
    peak_loc    = x_sub[idx]
    peak_height = y_sub[idx]
    half        = peak_height / 2.0

    # FWHM on subrange
    inds = np.where(y_sub >= half)[0]
    if inds.size >= 2:
        fwhm = x_sub[inds[-1]] - x_sub[inds[0]]
    else:
        fwhm = np.nan

    return peak_loc, peak_height, fwhm


"""