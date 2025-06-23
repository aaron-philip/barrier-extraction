import numpy as np
import findiff as fd
import jax.numpy as jnp
import pandas as pd
import os

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
    x_max: float = 80.0
):
    """
    Given arrays x (shape [n,]) and y (shape [n,]), find within [x_min, x_max]:
      - peak_loc: x at max(y)
      - peak_height: max(y)
      - fwhm: full width at half maximum
    Returns (peak_loc, peak_height, fwhm).
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


