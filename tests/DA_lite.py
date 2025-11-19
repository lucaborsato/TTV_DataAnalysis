#!/usr/bin/env python
# coding: utf-8

# # Data Analysis

# #### Imports

# In[1]:


import numpy as np
import os
import sys
import time as timer

# In[2]:


os.environ["OMP_NUM_THREADS"] = "1"


# In[3]:


from pytrades import pytrades
from pytrades import constants as cst
from pytrades import ancillary as anc
from pytrades import plot_oc as poc


import ttvfast

from scipy.stats import norm, halfnorm, uniform


# import matplotlib as mpl
# mpl.use("Agg") # use non-interactive backend for matplotlib

# import matplotlib.pyplot as plt
# from matplotlib.gridspec import GridSpec
# anc.set_rcParams()
# plt.rcParams["figure.dpi"] = 600
# plt.rcParams["savefig.dpi"] = 600
# for key, value in plt.rcParams.items():
#     if "figsize" in key:
#         print(key, value)


# ## TOI-216 system  
# Simulate a 2-planet system based on the parameters published in [McKee et al., 2023](https://ui.adsabs.harvard.edu/abs/2023AJ....165..236M/abstract)  
# Planetary parameters at reference time $\mathrm{BJD_{TDB} - 2457000 = 1325.31$ days for TOI-216 b and TOI-216 c.  
# For demonstration purpose, we will use zero as reference time.  
# Let's define masses in solar masses (`mass`), 
# radius in solar radii (`radius`), 
# period in days (`period`), 
# arguments of periastron (`argp`), 
# mean anomalies (`meana`), 
# mean longitudes (`meanl`),
# inclinations (`inc`), 
# longitudes of ascending nodes (`longn`)
# in degrees.

# In[12]:





# ## TTVFast
# Tring the `TTVFast` code by [Deck et al., 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...787..132D/abstract)  
# available at [github@TTVFast](https://github.com/kdeck/TTVFast) and wrapped with [github@ttvfast-python](https://github.com/simonrw/ttvfast-python).  

# In[24]:

def get_transits_ttvfast(mass_in, period_in, ecc_in, argp_in, meana_in, inc_in, longn_in, t_epoch_in, t_int_in):

    planet_b = ttvfast.models.Planet(
        mass = mass_in[1],# mass: Mplanet in units of M_sun
        period = period_in[1],# period: Period in days
        eccentricity = ecc_in[1],# eccentricity: E between 0 and 1
        inclination = inc_in[1],# inclination: I in units of degrees
        longnode = longn_in[1],# longnode: Longnode in units of degrees
        argument = argp_in[1],# argument: Argument in units of degrees
        mean_anomaly = meana_in[1],# mean_anomaly: mean anomaly in units of degrees
    )

    planet_c = ttvfast.models.Planet(
        mass = mass_in[2],# mass: Mplanet in units of M_sun
        period = period_in[2],# period: Period in days
        eccentricity = ecc_in[2],# eccentricity: E between 0 and 1
        inclination = inc_in[2],# inclination: I in units of degrees
        longnode = longn_in[2],# longnode: Longnode in units of degrees
        argument = argp_in[2],# argument: Argument in units of degrees
        mean_anomaly = meana_in[2],# mean_anomaly: mean anomaly in units of degrees
    )

    planets = [planet_b, planet_c]
    gravity = cst.Giau
    stellar_mass = mass_in[0]
    dt = np.min(period_in[1:]) / 10.0

    results = ttvfast.ttvfast(
        planets, 
        stellar_mass, 
        t_epoch_in, 
        dt, 
        t_int_in, 
        rv_times=None, 
        input_flag=1 # 0 = Jacobi 1 = astrocentric elements 2 = astrocentric cartesian
    )

    ttvfast_index, ttvfast_epoch, ttvfast_transits, _, _ = results["positions"]
    SEL_OK =  np.array(ttvfast_transits) > -2.0
    ttvfast_index = np.array(ttvfast_index)[SEL_OK]
    ttvfast_epoch = np.array(ttvfast_epoch)[SEL_OK]
    ttvfast_transits = np.array(ttvfast_transits)[SEL_OK]

    return ttvfast_index, ttvfast_epoch, ttvfast_transits


def compute_epoch(Tref, Pref, transit_times):

    dt = transit_times - Tref
    epoch = np.rint(dt / Pref)

    return epoch


# Based on Numerical Recipes

# case without errors on transit times
def linear_fit_no_errors(x, y):

    nx = len(x)
    S = nx
    Sx = np.sum(x)
    Sy = np.sum(y)

    t = x - (Sx / S)
    m = np.dot(t, y)
    St2 = np.dot(t, t)
    m = m / St2
    q = (Sy - (m * Sx)) / S
    err_q = np.sqrt((1.0 + ((Sx*Sx)/(S*St2)))/S)
    err_m = np.sqrt(1.0/St2)

    t = y - (q + m * x)
    chi2 = np.dot(t,t)
    dof = nx-2
    sigdat = np.sqrt(chi2/dof)
    err_q = err_q*sigdat
    err_m = err_m*sigdat

    
    return m, err_m, q, err_q, chi2

# case with errors
def linear_fit_with_errors(x, y, ey):

    nx = len(x)
    w = 1.0 / (ey*ey)
    S = np.sum(w)
    Sx = np.dot(w, x)
    Sy = np.dot(w, y)

    t = (x - (Sx/S))/ey
    m = np.dot(t/ey, y)
    St2 = np.dot(t, t)
    m = m / St2
    q = (Sy - (Sx*m))/S
    err_q = np.sqrt((1.0 + ((Sx*Sx)/(S*St2)))/S)
    err_m = np.sqrt(1.0/St2)

    return m, err_m, q, err_q

# automatic selection of case with or without errors
def linear_fit(x, y, ey=None):

    if(ey is None):
        m, err_m, q, err_q, chi2 = linear_fit_no_errors(x, y)
        # res = y - (q + m * x)
    else:
        m, err_m, q, err_q = linear_fit_with_errors(x, y, ey)
        res = (y - (q + m * x)) / ey
        chi2 = np.dot(res, res)

    return (q, err_q), (m, err_m), chi2

# automatic linear ephemeris from input or fitting
def linear_ephemeris(T0s, eT0s=None, Tref_in = None, Pref_in = None, fit=False):

    nx = len(T0s)

    # let's define a first guess of the reference time
    Tref, Pref = [0.0, 0.0], [0.0, 0.0]
    if Tref_in is None:
        Tref[0] = T0s[nx//2]
    elif isinstance(Tref_in, (tuple, list, np.ndarray)):
        Tref = Tref_in
    else:
        Tref[0] = Tref_in

    # let's define a first guess of the reference period
    if Pref_in is None:
        Pref[0] = np.median(np.diff(T0s))
    elif isinstance(Pref_in, (tuple, list, np.ndarray)):
        Pref = Pref_in
    else:
        Pref[0] = Pref_in

    # let's define the initial epochs
    epoch = compute_epoch(Tref[0], Pref[0], T0s)
    
    if fit:
        # fit the transit times to obtain the linear ephemeris
        Tref, Pref, chi2 = linear_fit(epoch, T0s, ey=eT0s)
        # recompute the epoch (or transit numbers)
        epoch = compute_epoch(Tref[0], Pref[0], T0s)

    # compute the predicted transit times
    Tlin = Tref[0] + Pref[0] * epoch
    # compute the O-C in days
    oc = T0s - Tlin
    if not fit:
        chi2 = np.dot(oc,oc)
    
    return Tref, Pref, chi2, epoch, Tlin, oc


def run_and_get_transits_trades(
    t_e,
    t_s,
    t_i,
    mass_in, 
    radius_in,
    period_in,
    ecc_in,
    argp_in,
    meana_in,
    inc_in,
    longn_in,
    planet_names,
):

    n_body = len(mass_in)
    ttt, ooo, stable = pytrades.kelements_to_orbits_full(
        t_e,
        t_s,
        t_i,
        mass_in,
        radius_in,
        period_in,
        ecc_in,
        argp_in,
        meana_in,
        inc_in,
        longn_in,
        specific_times=None,
        step_size=None,
        n_steps_smaller_orbits=10.0,
        )
    sort_ttt = np.argsort(ttt)
    ttt = ttt[sort_ttt]
    ooo = ooo[sort_ttt, :]
    
    tra_body = 1 # Set the transiting body to 1 (all planets)
    n_transits = [len(ttt)-1]*(n_body-1) # prepare a list of the number of transits for each planet
    n_all_transits = np.sum(n_transits) # total number of transits for all the planets
    # Compute the transits, durations, lambda_rm, Kepler elements, and body flags
    tratime, durmin, l_rm, kelem, bd_flag = pytrades.orbits_to_transits(
        n_all_transits, ttt, mass_in, radius_in, ooo, tra_body
    )
    
    out_transits = {}
    for i_pl, pl_letter in enumerate(planet_names):
        pl_number = i_pl + 2
        sel_pl = bd_flag == pl_number
        n_transits = np.sum(sel_pl)
        print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_transits, t_i))
        out_transits[pl_letter] = {
            "n_transits":  n_transits,
            "transit_times": tratime[sel_pl],
            "transit_durations": durmin[sel_pl],
            "lambda_rm": l_rm[sel_pl],
            "kep_elem": kelem[sel_pl],
        }

    return ttt, ooo, stable, out_transits




def select_transit_times(
    pl_letter, pl_idx, tra_index, tra_times, err_pool, t1=0.0, t2=365.25,
    n_tra_syn=20, err_scale=5, seed=42
):
    """
    Select a subset of transit times from a larger pool and add uncertainty.

    Parameters
    ----------
    pl_letter : str
        Planet letter (e.g. "b", "c")
    pl_idx : int
        Planet index (e.g. 0, 1)
    tra_index : array
        Index of each transit (e.g. 0, 1, 2, etc.)
    tra_times : array
        Transit times (BJD)
    err_pool : array
        Pool of uncertainties to draw from
    t1 : float, optional
        Lower bound of time range to select transits from (days)
    t2 : float, optional
        Upper bound of time range to select transits from (days)
    n_tra_syn : int, optional
        Number of synthetic transits to generate
    err_scale : int, optional
        Scale factor for the uncertainties
    seed : int, optional
        Random seed

    Returns
    -------
    tra_syn : array
        Synthetic transit times
    err_tra_syn : array
        Synthetic uncertainties
    tra_syn_noisy : array
        Synthetic transit times with added uncertainties
    """

    np.random.seed(seed=seed)

    sel_tra = tra_index == pl_idx
    all_tra_syn = tra_times[sel_tra][np.logical_and(
        tra_times[sel_tra] >= t1,
        tra_times[sel_tra] <= t2,
    )]
    n_tra = len(all_tra_syn)
    print(
        "{:4d} transits of planet {:s} in {:.2f} days".format(
            n_tra, pl_letter, (t2-t1)
        )
    )

    tra_syn = np.random.choice(all_tra_syn, size=n_tra_syn, replace=False)
    err_tra_syn = np.random.choice(err_pool, size=n_tra_syn, replace=True)
    err_mean, err_std = np.mean(err_pool), np.std(err_pool, ddof=1)
    tra_syn_noisy = tra_syn + np.random.normal(
        loc=err_mean, scale=err_std, size=n_tra_syn
    ) / err_scale

    return tra_syn, err_tra_syn, tra_syn_noisy

def fitting_to_observables(
    t_e, t_s, t_i, 
    m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit, 
    tra_flag, planet_names
    ):

    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = pytrades.kelements_to_observed_t0s(
        t_e,
        t_s,
        t_i,
        m_fit,
        r_fit,
        p_fit,
        e_fit,
        w_fit,
        ma_fit,
        i_fit,
        ln_fit,
        tra_flag
    )

    transits = {}
    for i_pl, pl_letter in enumerate(planet_names):
        pl_number = i_pl + 2
        sel_pl = body_flag_sim == pl_number
        n_transits = np.sum(sel_pl)
        print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_transits, t_i))
        transits[pl_letter] = {
            "n_transits":  n_transits,
            "transit_times": transits_sim[sel_pl],
            "transit_durations": durations_sim[sel_pl],
            "lambda_rm": lambda_rm_sim[sel_pl],
            "kep_elem": kep_elem_sim[sel_pl],
        }
    
    return transits



# ======================== MAIN ========================


def main():

    body_names = ["star", "b", "c"]

    mass   = np.array([0.763, 0.0554*cst.Mjups, 0.525*cst.Mjups])
    radius = np.array([0.757, 7.84*cst.Rears, 10.09*cst.Rears])
    period = np.array([0.0, 17.0988, 34.5508])
    ecc    = np.array([0.0, 0.1593, 0.009])
    argp   = np.array([0.0, 292.0, 236.0])
    meanl  = np.array([0.0, 82.18, 27.50])
    inc    = np.array([0.0, 88.554, 89.801])
    longn  = np.array([0.0, 0.0, -0.80])%360.0
    meana = (meanl-longn-argp) %360.0

    mass_map   = np.array([0.763, 5.404692347894688e-05, 0.0005121577611636013])
    radius_map = np.array([0.757, 0.07184567664367815, 0.09246465272126436])
    period_map = np.array([0.0, 17.098294484920213, 34.55102547856706])
    ecc_map    = np.array([0.0, 0.16118774006272724, 0.005761123253136914])
    argp_map   = np.array([0.0, 291.3880903384844, 227.6994142563319])
    meana_map  = np.array([0.0, 150.8875359258843, 160.46270266177874])
    inc_map    = np.array([0.0, 88.554, 89.801])
    longn_map  = np.array([0.0, 0.0, 359.2])

    n_body = len(mass)
    duration_check = 1 # do computation of the transit duration
    t_epoch = 0.0 # reference time of integration
    t_start = 0.0 # start time of integration
    t_int   = 365.25*5 # duration of integration
    
    transit_flag = [0, 1, 1] # 0 = not transiting (star), 1 transiting (b & c)

    seed = 42
    np.random.seed(seed=seed)
    uncertainty = {
    "b": [0.002, 0.003, 0.004], # Dawson+ 2021
    "c": [0.0004, 0.0005, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0015, 0.002, 0.003], # Dawson+ 2021
    }

    print("", flush=True)

    # ========================= TTVFast =================================
    ttvfast_index, ttvfast_epoch, ttvfast_transits = get_transits_ttvfast(
        mass, period, ecc, argp, meana, inc, longn,
        t_epoch, t_int
    )

    # ========================= Select transits =========================
    n_b, n_c = 20, 10
    tra_b, err_tra_b, tra_noisy_b = select_transit_times(
        "b", 0, 
        ttvfast_index, ttvfast_transits, 
        uncertainty["b"], 
        t1=t_start, t2=t_start + t_int, 
        n_tra_syn=n_b,
        err_scale=5,
        seed=seed
    )
    
    tra_c, err_tra_c, tra_noisy_c = select_transit_times(
        "c", 1, 
        ttvfast_index, ttvfast_transits, 
        uncertainty["c"],
        t1=t_start, t2=t_start + t_int,
        n_tra_syn=n_c,
        err_scale=5,
        seed=seed
    )
    # ========================= END Select transits =========================

    print("", flush=True)

    # ========================= COMPUTE LINEAR EPHEMERIS =========================
    lineph_syn_noisy = {}

    i, pl_letter = 0, "b"
    print("planet {}".format(pl_letter))

    Tr = tra_noisy_b[n_b // 2]
    Pr = 17.16073 # Dawson+ 2021
    print("input: Tref = ", Tr, "Pref = ", Pr)
    Tref_b, Pref_b, chi2_b, epoch_b, Tlin_b, oc_b_days = linear_ephemeris(
        tra_noisy_b, eT0s=err_tra_b, 
        Tref_in = Tr, Pref_in = Pr, 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "noisy",
        *Tref_b, *Pref_b, chi2_b, chi2_b/(n_b-2))
    )
    lineph_syn_noisy[pl_letter] = (Tref_b, Pref_b)

    i, pl_letter = 1, "c"
    print("planet {}".format(pl_letter))

    Tr = tra_noisy_c[n_c // 2]
    Pr = 34.525528 # Dawson+ 2021
    print("input: Tref = ", Tr, "Pref = ", Pr)
    Tref_c, Pref_c, chi2_c, epoch_c, Tlin_c, oc_c_days = linear_ephemeris(
        tra_noisy_c, eT0s=err_tra_c, 
        Tref_in = Tr, Pref_in = Pr, 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "noisy",
        *Tref_c, *Pref_c, chi2_c, chi2_c/(n_c-2))
    )
    lineph_syn_noisy[pl_letter] = (Tref_c, Pref_c)

    print("", flush=True)

    # ======================= INIT TRADES =======================
    pytrades.args_init(
        n_body,
        duration_check,
        t_epoch=t_epoch,
        t_start=t_start,
        t_int=t_int,
        encounter_check=True, # check for close encounters
        do_hill_check=False, # check for stability condition based on Hill radius
        amd_hill_check=False, # check for stability condition based on AMD-Hill criterion
        rv_res_gls=False, # use GLS method on RV residuals to avoid introduction of signals close to the planetary periods
    )
    # ======================= END TRADES =======================

    # ========================= Load transits into memory =========================
    b_sources_id = np.ones(n_b).astype(int)
    pytrades.set_t0_dataset(2, epoch_b, tra_noisy_b, err_tra_b, sources_id=b_sources_id)
    c_sources_id = np.ones(n_c).astype(int)
    pytrades.set_t0_dataset(3, epoch_c, tra_noisy_c, err_tra_c, sources_id=c_sources_id)
    print("\nData info - just loaded transits", flush=True)
    pytrades.get_data_info()
    print("", flush=True)
    # ========================= END Load transits into memory =========================

    
    # print("\nData info - before fit observables DEFAULT")
    # pytrades.get_data_info()

    timer_start = timer.time()
    print("\nFITTING OBSERVABLES - DEFAULT")
    transits_fit_0 = fitting_to_observables(
        t_epoch, t_start, t_int,
        mass, radius, period, ecc, argp, meana, inc, longn,
        transit_flag, body_names[1:]
    )
    timer_end = timer.time()
    print("Time to fit observables: {:.2f} s".format(timer_end - timer_start))

    timer_start = timer.time()
    print("\nRUN AND GET TRANSITS TRADES - DEFAULT")
    transits_full_0 = run_and_get_transits_trades(
        t_epoch, t_start, t_int,
        mass, radius, period, ecc, argp, meana, inc, longn, 
        body_names[1:]
    )
    timer_end = timer.time()
    print("Time to run and get transits trades: {:.2f} s".format(timer_end - timer_start))

    # print("\nData info - before fit observables MAP")
    # pytrades.get_data_info()

    timer_start = timer.time()
    print("\nFITTING OBSERVABLES - MAP")
    transits_fit_1 = fitting_to_observables(
        t_epoch, t_start, t_int,
        mass_map, radius_map, period_map, ecc_map, argp_map, meana_map, inc_map, longn_map,
        transit_flag, body_names[1:]
    )
    timer_end = timer.time()
    print("Time to fit observables: {:.2f} s".format(timer_end - timer_start))

    timer_start = timer.time()
    print("\nRUN AND GET TRANSITS TRADES - MAP")
    transits_full_0 = run_and_get_transits_trades(
        t_epoch, t_start, t_int,
        mass_map, radius_map, period_map, ecc_map, argp_map, meana_map, inc_map, longn_map, 
        body_names[1:]
    )
    timer_end = timer.time()
    print("Time to run and get transits trades: {:.2f} s".format(timer_end - timer_start))

    return

# =============================================================================
# =============================================================================
if __name__ == "__main__":
    main()