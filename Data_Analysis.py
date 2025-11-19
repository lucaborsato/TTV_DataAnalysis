# %% [markdown]
# # Data Analysis

# %% [markdown]
# #### Imports

# %%
import numpy as np
import os
import sys
import pickle

# %%
os.environ["OMP_NUM_THREADS"] = "1"

# %%
from pytrades import pytrades
from pytrades import constants as cst
from pytrades import ancillary as anc
from pytrades import plot_oc as poc
from pytrades.convergence import log_probability_trace, full_statistics

# %%
import rebound

# %%
import ttvfast

# %%
from pytransit.utils import de as pyde

# %%
import emcee

# %%
from scipy.stats import norm, halfnorm, uniform

# %%
import nautilus

# %%
import pygtc

# %%
from multiprocessing import Pool

# %%
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
anc.set_rcParams()
plt.rcParams["figure.dpi"] = 600
plt.rcParams["savefig.dpi"] = 600
for key, value in plt.rcParams.items():
    if "figsize" in key:
        print(key, value)

# %% [markdown]
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

# %%
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

# %% [markdown]
# ## TRADES  
# ### Initialise the simulation  
# Define some parameters needed for `pytrades` code and initialise the simulation

# %%
n_body = len(mass)
duration_check = 1 # do computation of the transit duration
t_epoch = 0.0 # reference time of integration
t_start = 0.0 # start time of integration
t_int   = 5000.0 # duration of integration

# %%
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

# %% [markdown]
# ### Integrates the orbits  
# Integrates the orbits of the planets and obtains the times steps, state vector (position and velocity for all bodies for each time step), stability flag.

# %%
time_steps, orbits, stable = pytrades.kelements_to_orbits_full(
    t_epoch,
    t_start,
    t_int,
    mass,
    radius,
    period,
    ecc,
    argp,
    meana,
    inc,
    longn,
    specific_times=None,
    step_size=None,
    n_steps_smaller_orbits=10.0,
)

# %% [markdown]
# ### Plotting the orbits of the planets in the system  
# Plot the orbits of the planets in the system in three projections:  
# sky plane: $(X-Y)$  
# orbit planet (observer on top): $(X-Z)$  
# side orbit (observer on right side): $(Z-Y)$

# %%
fig= pytrades.base_plot_orbits(
    time_steps,
    orbits,
    radius,
    n_body,
    body_names,
    figsize=(4, 4),
    sky_scale='star',
    side_scale='positive',
    title="TOI-216 - trades",
    show_plot=True,
)
plt.close(fig)

# %% [markdown]
# ### Extracting transit times  
# Let's extract transit times from the orbits.  
# The function returns:  
# - `transits` array of transit times  
# - `durations` array of transit durations (total duration, $T_{14}$) in minutes  
# - `lambda_rm` array of the projected spin-orbit misalignment angles  
# - `kep_elem` array of Keplerian orbital elements (period in days, semi-majoar axis in au, eccentricity, inclination, mean anomaly, argument of pericenter in degrees)  
# - `body_flag` array of flag that identifies the body (2 for planet the first planet b, 3 for the second planet c), it needs to recover the transit times for different planets.

# %%
transiting_body = 1 # Set the transiting body to 1 (all planets)
n_transits = [len(time_steps)-1]*(n_body-1) # prepare a list of the number of transits for each planet
n_all_transits = np.sum(n_transits) # total number of transits for all the planets
# Compute the transits, durations, lambda_rm, Kepler elements, and body flags
transits, durations, lambda_rm, kep_elem, body_flag = pytrades.orbits_to_transits(
    n_all_transits, time_steps, mass, radius, orbits, transiting_body
)

# %% [markdown]
# Let's convert the array into a dictionary.

# %%
trades_transits = {}
for pl_letter, pl_number in zip(body_names[1:], [2,3]):
    sel_pl = body_flag == pl_number
    n_transits = np.sum(sel_pl)
    print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_transits, t_int))
    trades_transits[pl_letter] = {
        "n_transits":  n_transits,
        "transit_times": transits[sel_pl],
        "transit_durations": durations[sel_pl],
        "lambda_rm": lambda_rm[sel_pl],
        "kep_elem": kep_elem[sel_pl],
    }

# %% [markdown]
# ## REBOUND  
# Computes the orbits with `rebound` packages, but get the transits with `trades` package.

# %%
def run_rebound(
    mass,
    radius,
    period,
    ecc,
    argp,
    meana,
    inc,
    longn,
    time_steps
):

    radius_au = radius * cst.RsunAU
    argp_r   = argp * cst.deg2rad
    meana_r  = meana * cst.deg2rad
    inc_r    = inc * cst.deg2rad
    longn_r  = longn * cst.deg2rad

    sim = rebound.Simulation()
    sim.units = ["msun", "au", "days"] # Units of solar mass, au, and days

    sim.add(m=mass[0], r=radius_au[0]) #star
    nbody = len(mass)
    for ib in range(1, nbody):
        sim.add(
            m=mass[ib],
            r=radius_au[ib],
            P=period[ib],
            e=ecc[ib],
            omega=argp_r[ib],
            M=meana_r[ib],
            inc=inc_r[ib],
            Omega=longn_r[ib],
            primary=sim.particles[0]
        )
    
    sim.move_to_com()

    times = (time_steps - time_steps[0])
    ntime = len(times)

    orbits = np.zeros((ntime, 6*nbody))
    for i, t in enumerate(times):
        sim.integrate(t)
        for ib in range(nbody):
            orbits[i, 0+(6*ib)] = sim.particles[ib].x
            orbits[i, 1+(6*ib)] = sim.particles[ib].y
            orbits[i, 2+(6*ib)] = sim.particles[ib].z
            orbits[i, 3+(6*ib)] = sim.particles[ib].vx
            orbits[i, 4+(6*ib)] = sim.particles[ib].vy
            orbits[i, 5+(6*ib)] = sim.particles[ib].vz

    return orbits

# %%
orbits_rebound = run_rebound(
    mass,
    radius,
    period,
    ecc,
    argp,
    meana,
    inc,
    longn,
    time_steps
)

# %%
fig= pytrades.base_plot_orbits(
    time_steps,
    orbits_rebound,
    radius,
    n_body,
    body_names,
    figsize=(4, 4),
    sky_scale='star',
    side_scale='star',
    title="TOI-216 - rebound",
    show_plot=True,
)
plt.close(fig)

# %%
# Compute the transits, durations, lambda_rm, Kepler elements, and body flags
transits_rebound, durations_rebound, lambda_rm_rebound, kep_elem_rebound, body_flag_rebound = pytrades.orbits_to_transits(
    n_all_transits, time_steps, mass, radius, orbits_rebound, transiting_body
)

# %% [markdown]
# Let's convert the array into a dictionary.

# %%
rebound_transits = {}
for pl_letter, pl_number in zip(body_names[1:], [2,3]):
    sel_pl = body_flag_rebound == pl_number
    n_transits = np.sum(sel_pl)
    print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_transits, t_int))
    rebound_transits[pl_letter] = {
        "n_transits":  n_transits,
        "transit_times": transits_rebound[sel_pl],
        "transit_durations": durations_rebound[sel_pl],
        "lambda_rm": lambda_rm_rebound[sel_pl],
        "kep_elem": kep_elem_rebound[sel_pl],
    }

# %% [markdown]
# ## TTVFast
# Tring the `TTVFast` code by [Deck et al., 2014](https://ui.adsabs.harvard.edu/abs/2014ApJ...787..132D/abstract)  
# available at [github@TTVFast](https://github.com/kdeck/TTVFast) and wrapped with [github@ttvfast-python](https://github.com/simonrw/ttvfast-python).  

# %%
planet_b = ttvfast.models.Planet(
    mass = mass[1],# mass: Mplanet in units of M_sun
    period = period[1],# period: Period in days
    eccentricity = ecc[1],# eccentricity: E between 0 and 1
    inclination = inc[1],# inclination: I in units of degrees
    longnode = longn[1],# longnode: Longnode in units of degrees
    argument = argp[1],# argument: Argument in units of degrees
    mean_anomaly = meana[1],# mean_anomaly: mean anomaly in units of degrees
)

planet_c = ttvfast.models.Planet(
    mass = mass[2],# mass: Mplanet in units of M_sun
    period = period[2],# period: Period in days
    eccentricity = ecc[2],# eccentricity: E between 0 and 1
    inclination = inc[2],# inclination: I in units of degrees
    longnode = longn[2],# longnode: Longnode in units of degrees
    argument = argp[2],# argument: Argument in units of degrees
    mean_anomaly = meana[2],# mean_anomaly: mean anomaly in units of degrees
)

planets = [planet_b, planet_c]
gravity = cst.Giau
stellar_mass = mass[0]
dt = np.min(np.diff(time_steps))

# %%
results = ttvfast.ttvfast(
    planets, 
    stellar_mass, 
    t_epoch, 
    dt, 
    t_int, 
    rv_times=None, 
    input_flag=1 # 0 = Jacobi 1 = astrocentric elements 2 = astrocentric cartesian
)

# %%
ttvfast_index, ttvfast_epoch, ttvfast_transits, _, _ = results["positions"]
SEL_OK =  np.array(ttvfast_transits) > -2.0
ttvfast_index = np.array(ttvfast_index)[SEL_OK]
ttvfast_epoch = np.array(ttvfast_epoch)[SEL_OK]
ttvfast_transits = np.array(ttvfast_transits)[SEL_OK]


# %% [markdown]
# ## O-C diagram  
# To visualise an O-C diagram, it has to be computed the transit number (epoch) of each transit with respect to a linear ephemeris.  
# It is needed to define the functions that allows to compute the epoch and the linear ephemeris through the weighted least square method.

# %% [markdown]
# #### functions to compute the linear ephemerism

# %%
def compute_epoch(Tref, Pref, transit_times):

    dt = transit_times - Tref
    epoch = np.rint(dt / Pref)

    return epoch

# %%
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

# %% [markdown]
# #### plot O-C

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

for i, pl_letter in enumerate(body_names[1:]):
    print("planet {}".format(pl_letter))

    ax = axs[i]
    
    # trades
    tra_tr = trades_transits[pl_letter]["transit_times"]
    n_tra_tr = len(tra_tr)
    Tref_tr, Pref_tr, chi2_tr, epoch_tr, Tlin_tr, oc_tr_days = linear_ephemeris(
        tra_tr, eT0s=None, 
        Tref_in = None, Pref_in = None, 
        # Tref_in = tra_tr[0], Pref_in = period[i+1], 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "trades",
        *Tref_tr, *Pref_tr, chi2_tr, chi2_tr/(n_tra_tr-2))
     )

    ax.axhline(0, color="k", lw=0.8)
    ax.plot(
        tra_tr,
        oc_tr_days*u[0],
        marker=markers[i],
        ms=1,
        color="C0",
        label="{} (trades)".format(pl_letter),
        ls='',
    )

    # rebound
    tra_re = rebound_transits[pl_letter]["transit_times"]
    n_tra_re = len(tra_re)
    Tref_re, Pref_re, chi2_re, epoch_re, Tlin_re, oc_re_days = linear_ephemeris(
        tra_re, eT0s=None, Tref_in = Tref_tr, Pref_in = Pref_tr, fit=False
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "rebound",
        *Tref_re, *Pref_re, chi2_re, chi2_re/(n_tra_re-2))
     )

    ax.plot(
        tra_re,
        oc_re_days*u[0],
        marker=markers[i],
        ms=2,
        mfc="None",
        mew=0.3,
        color="C1",
        label="{} (rebound)".format(pl_letter),
        ls='',
    )

    # # TTVFast
    sel_ttvf = ttvfast_index == i
    tra_ttvf = ttvfast_transits[sel_ttvf]
    n_tra_ttvf = len(tra_ttvf)
    Tref_ttvf, Pref_ttvf, chi2_ttvf, epoch_ttvf, Tlin_ttvf, oc_ttvf_days = linear_ephemeris(
        tra_ttvf, eT0s=None, 
        # Tref_in = tra_ttvf[0], 
        # Pref_in = period[i+1], 
        # fit=False
        Tref_in = Tref_tr[0], 
        Pref_in = Pref_tr[0], 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "TTVFast",
        *Tref_ttvf, *Pref_ttvf, chi2_ttvf, chi2_ttvf/(n_tra_ttvf-2))
     )

    ax.plot(
        tra_ttvf,
        oc_ttvf_days*u[0],
        marker=markers[i],
        ms=2.5,
        mfc="None",
        mew=0.3,
        color="C2",
        label="{} (TTVFast)".format(pl_letter),
        ls='',
    )
    

    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# As you can see the two codes reproduces the same orbits and $O-C$ diagrams for both planets.  
# ## Parameters sensitivity
# Let's change some parameters of planet c to see the effects on the TTV amplitude, phase and patterns.

# %%
def run_and_get_transits(
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
        # print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_transits, t_i))
        out_transits[pl_letter] = {
            "n_transits":  n_transits,
            "transit_times": tratime[sel_pl],
            "transit_durations": durmin[sel_pl],
            "lambda_rm": l_rm[sel_pl],
            "kep_elem": kelem[sel_pl],
        }

    return ttt, ooo, stable, out_transits

# %% [markdown]
# #### test eccentricity variations

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

lineph_trades = {}

for i, pl_letter in enumerate(body_names[1:]):
    print("planet {}".format(pl_letter))

    ax = axs[i]
    
    # trades original
    tra_tr = trades_transits[pl_letter]["transit_times"]
    n_tra_tr = len(tra_tr)
    Tref_tr, Pref_tr, chi2_tr, epoch_tr, Tlin_tr, oc_tr_days = linear_ephemeris(
        tra_tr, eT0s=None, 
        Tref_in = None, Pref_in = None, 
        # Tref_in = tra_tr[0], Pref_in = period[i+1], 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "trades",
        *Tref_tr, *Pref_tr, chi2_tr, chi2_tr/(n_tra_tr-2))
     )
    lineph_trades[pl_letter] = (Tref_tr, Pref_tr)

    ax.axhline(0, color="k", lw=0.8)
    ax.plot(
        tra_tr,
        oc_tr_days*u[0],
        marker=markers[0],
        ms=1,
        color="C0",
        label="{} (trades)".format(pl_letter),
        ls='',
    )

par_c_test = [0.0, 0.05, 0.1]
for itest, p_test in enumerate(par_c_test):
    ecc_new = ecc.copy()
    ecc_new[2] = p_test
    
    time_steps_new, orbits_new, stable_new, trades_transits_new = run_and_get_transits(
        t_epoch,
        t_start,
        t_int,
        mass, radius,
        period,
        ecc_new,
        argp, meana,
        inc, longn,
        body_names[1:]
    )
    
    
    for i, pl_letter in enumerate(body_names[1:]):
    
        ax = axs[i]

        Tref_tr, Pref_tr = lineph_trades[pl_letter]
        
        # trades updated ecc
        tra_new = trades_transits_new[pl_letter]["transit_times"]
        n_tra_new = len(tra_new)
        Tref_new, Pref_new, chi2_new, epoch_new, Tlin_new, oc_new_days = linear_ephemeris(
            tra_new, eT0s=None,
            Tref_in = Tref_tr,
            Pref_in = Pref_tr,
            fit=True
        )
        print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
            "rebound",
            *Tref_new, *Pref_new, chi2_new, chi2_new/(n_tra_new-2))
         )
    
        ax.plot(
            tra_new,
            oc_new_days*u[0],
            marker=markers[1+itest],
            ms=2,
            mfc="None",
            mew=0.3,
            color="C{}".format(1+itest),
            label="{} $e_\mathrm{{c}} = {:.2f}$".format(pl_letter, p_test),
            ls='',
        )

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# #### test inclination variations

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

lineph_trades = {}

for i, pl_letter in enumerate(body_names[1:]):
    print("planet {}".format(pl_letter))

    ax = axs[i]
    
    # trades original
    tra_tr = trades_transits[pl_letter]["transit_times"]
    n_tra_tr = len(tra_tr)
    Tref_tr, Pref_tr, chi2_tr, epoch_tr, Tlin_tr, oc_tr_days = linear_ephemeris(
        tra_tr, eT0s=None, 
        Tref_in = None, Pref_in = None, 
        # Tref_in = tra_tr[0], Pref_in = period[i+1], 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "trades",
        *Tref_tr, *Pref_tr, chi2_tr, chi2_tr/(n_tra_tr-2))
     )
    lineph_trades[pl_letter] = (Tref_tr, Pref_tr)

    ax.axhline(0, color="k", lw=0.8)
    ax.plot(
        tra_tr,
        oc_tr_days*u[0],
        marker=markers[0],
        ms=1,
        color="C0",
        label="{} (trades)".format(pl_letter),
        ls='',
    )

par_c_test = inc[2] + np.array([-3, -1, +1, 3])
for itest, p_test in enumerate(par_c_test):
    inc_new = inc.copy()
    inc_new[2] = p_test
    
    time_steps_new, orbits_new, stable_new, trades_transits_new = run_and_get_transits(
        t_epoch,
        t_start,
        t_int,
        mass, radius,
        period,
        ecc,
        argp, meana,
        inc_new, longn,
        body_names[1:],
    )
    
    
    for i, pl_letter in enumerate(body_names[1:]):
    
        ax = axs[i]

        Tref_tr, Pref_tr = lineph_trades[pl_letter]
        
        # trades updated inc
        tra_new = trades_transits_new[pl_letter]["transit_times"]
        n_tra_new = len(tra_new)
        if n_tra_new > 0:
            Tref_new, Pref_new, chi2_new, epoch_new, Tlin_new, oc_new_days = linear_ephemeris(
                tra_new, eT0s=None,
                Tref_in = Tref_tr,
                Pref_in = Pref_tr,
                fit=True
            )
            print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
                "rebound",
                *Tref_new, *Pref_new, chi2_new, chi2_new/(n_tra_new-2))
             )
        
            ax.plot(
                tra_new,
                oc_new_days*u[0],
                marker=markers[1+itest],
                ms=2,
                mfc="None",
                mew=0.3,
                color="C{}".format(1+itest),
                label="{} inc(c) = {:.2f}".format(pl_letter, p_test),
                ls='',
            )

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# #### test argument of pericentre variations

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

lineph_trades = {}

for i, pl_letter in enumerate(body_names[1:]):
    print("planet {}".format(pl_letter))

    ax = axs[i]
    
    # trades original
    tra_tr = trades_transits[pl_letter]["transit_times"]
    n_tra_tr = len(tra_tr)
    Tref_tr, Pref_tr, chi2_tr, epoch_tr, Tlin_tr, oc_tr_days = linear_ephemeris(
        tra_tr, eT0s=None, 
        Tref_in = None, Pref_in = None, 
        # Tref_in = tra_tr[0], Pref_in = period[i+1], 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "trades",
        *Tref_tr, *Pref_tr, chi2_tr, chi2_tr/(n_tra_tr-2))
     )
    lineph_trades[pl_letter] = (Tref_tr, Pref_tr)

    ax.axhline(0, color="k", lw=0.8)
    ax.plot(
        tra_tr,
        oc_tr_days*u[0],
        marker=markers[0],
        ms=1,
        color="C0",
        label="{} (trades)".format(pl_letter),
        ls='',
    )

# seed = 42
# np.random.seed(seed)
# argpc_test = 360.0*np.random.random(4)
par_c_test = [0.0, 90.0, 180.0, 270.0]
for itest, p_test in enumerate(par_c_test):
    argp_new = argp.copy()
    argp_new[2] = p_test
    
    time_steps_new, orbits_new, stable_new, trades_transits_new = run_and_get_transits(
        t_epoch,
        t_start,
        t_int,
        mass, radius,
        period,
        ecc,
        argp_new, meana,
        inc, longn,
        body_names[1:],
    )
    
    
    for i, pl_letter in enumerate(body_names[1:]):
    
        ax = axs[i]

        Tref_tr, Pref_tr = lineph_trades[pl_letter]
        
        # trades updated argp
        tra_new = trades_transits_new[pl_letter]["transit_times"]
        n_tra_new = len(tra_new)
        Tref_new, Pref_new, chi2_new, epoch_new, Tlin_new, oc_new_days = linear_ephemeris(
            tra_new, eT0s=None,
            Tref_in = Tref_tr,
            Pref_in = Pref_tr,
            fit=True
        )
        print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
            "rebound",
            *Tref_new, *Pref_new, chi2_new, chi2_new/(n_tra_new-2))
         )
    
        ax.plot(
            tra_new,
            oc_new_days*u[0],
            marker=markers[1+itest],
            ms=2,
            mfc="None",
            mew=0.3,
            color="C{}".format(1+itest),
            label="{} $\omega_\mathrm{{c}} = {:.2f}^{{\circ}}$".format(pl_letter, p_test),
            ls='',
        )

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# #### test mass variations

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

lineph_trades = {}

for i, pl_letter in enumerate(body_names[1:]):
    print("planet {}".format(pl_letter))

    ax = axs[i]
    
    # trades original
    tra_tr = trades_transits[pl_letter]["transit_times"]
    n_tra_tr = len(tra_tr)
    Tref_tr, Pref_tr, chi2_tr, epoch_tr, Tlin_tr, oc_tr_days = linear_ephemeris(
        tra_tr, eT0s=None, 
        Tref_in = None, Pref_in = None, 
        # Tref_in = tra_tr[0], Pref_in = period[i+1], 
        fit=True
    )
    print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
        "trades",
        *Tref_tr, *Pref_tr, chi2_tr, chi2_tr/(n_tra_tr-2))
     )
    lineph_trades[pl_letter] = (Tref_tr, Pref_tr)

    ax.axhline(0, color="k", lw=0.8)
    ax.plot(
        tra_tr,
        oc_tr_days*u[0],
        marker=markers[0],
        ms=1,
        color="C0",
        label="{} (trades)".format(pl_letter),
        ls='',
    )

label = ["M_\oplus", "M_\mathrm{Nep}", "M_\mathrm{Sat}", "M_\mathrm{Jup}"]
par_c_test = [cst.Mears, cst.Mneps, cst.Msats, cst.Mjups]
radiusc_test = [cst.Rears, cst.Rneps, cst.Rsat/cst.Rsun, cst.Rjups]
for itest, p_test in enumerate(par_c_test):
    mass_new = mass.copy()
    mass_new[2] = p_test

    radius_new = radius.copy()
    radius_new[2] = radiusc_test[itest]

    time_steps_new, orbits_new, stable_new, trades_transits_new = run_and_get_transits(
        t_epoch,
        t_start,
        t_int,
        mass_new,
        radius_new,
        period,
        ecc,
        argp,
        meana,
        inc,
        longn,
        body_names[1:],
    )

    for i, pl_letter in enumerate(body_names[1:]):
    
        ax = axs[i]

        Tref_tr, Pref_tr = lineph_trades[pl_letter]
        
        # trades updated mass
        tra_new = trades_transits_new[pl_letter]["transit_times"]
        n_tra_new = len(tra_new)
        Tref_new, Pref_new, chi2_new, epoch_new, Tlin_new, oc_new_days = linear_ephemeris(
            tra_new, eT0s=None,
            # Tref_in = Tref_tr,
            # Pref_in = Pref_tr,
            Tref_in = None,
            Pref_in = None,
            fit=True
        )
        print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
            "rebound",
            *Tref_new, *Pref_new, chi2_new, chi2_new/(n_tra_new-2))
         )
    
        ax.plot(
            tra_new,
            oc_new_days*u[0],
            marker=markers[1+itest],
            ms=2,
            mfc="None",
            mew=0.3,
            color="C{}".format(1+itest),
            label="{} $M_\mathrm{{c}} = {:s}$".format(pl_letter, label[itest]),
            ls='',
        )

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ## Sythetic transit times
# Let's select a number of transit times for both planets and add uncertainty.  
# First, define a time range to mimic the observations and take a subset of the transits in this time range.  
# Select transits from `TTVFast`, but the fit will be done with `trades`.  

# %%
time_sel_start = 0.0
time_sel_end   = 365.25 * 5
seed = 42
# np.random.seed(seed=seed)
uncertainty = {
    "b": [0.002, 0.003, 0.004], # Dawson+ 2021
    "c": [0.0004, 0.0005, 0.0007, 0.0008, 0.0009, 0.001, 0.0011, 0.0015, 0.002, 0.003], # Dawson+ 2021
}

# %% [markdown]
# ### Select transits

# %%
def select_transit_times(
    pl_letter, pl_idx, tra_index, tra_times, err_pool, 
    t1=0.0, t2=365.25, n_tra_syn=20, 
    err_scale_mult=1.0,
    noise_scale_mult=1.0,
    seed=42
):

    np.random.seed(seed=seed)
    
    sel_tra = tra_index == pl_idx
    all_tra_syn = tra_times[sel_tra][np.logical_and(
    tra_times[sel_tra] >= t1,
    tra_times[sel_tra] <= t2,
)]
    n_tra = len(all_tra_syn)
    # print("{:4d} transits of planet {:s} in {:.2f} days".format(n_tra, pl_letter, (t2-t1)))

    tra_syn = np.random.choice(all_tra_syn, size=n_tra_syn, replace=False)
    err_tra_syn = np.random.choice(err_pool, size=n_tra_syn, replace=True)
    err_mean, err_std = np.mean(err_pool), np.std(err_pool, ddof=1)
    tra_syn_noisy = tra_syn + np.random.normal(loc=err_mean, scale=err_std, size=n_tra_syn)*noise_scale_mult
    
    return tra_syn, err_tra_syn, tra_syn_noisy

n_tra_syn_b, n_tra_syn_c = 20, 10

err_scale_mult_b = 1 + np.random.random(n_tra_syn_b)*3
tra_syn_b, err_tra_syn_b, tra_syn_noisy_b = select_transit_times(
    "b", 0, 
    ttvfast_index, ttvfast_transits, 
    uncertainty["b"], 
    t1=time_sel_start, t2=time_sel_end, 
    n_tra_syn=n_tra_syn_b,
    err_scale_mult=err_scale_mult_b,
    noise_scale_mult=1.0,
    seed=seed
)


err_scale_mult_c = 1 + np.random.random(n_tra_syn_c)*3
tra_syn_c, err_tra_syn_c, tra_syn_noisy_c = select_transit_times(
    "c", 1, 
    ttvfast_index, ttvfast_transits, 
    uncertainty["c"],
    t1=time_sel_start, t2=time_sel_end,
    n_tra_syn=n_tra_syn_c,
    err_scale_mult=err_scale_mult_c,
    noise_scale_mult=1.0,
    seed=seed
)

# %% [markdown]
# ### Compute linear ephemeris and plot O-C w errors

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

lineph_syn = {}
lineph_syn_noisy = {}

i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

Tr = tra_syn_b[n_tra_syn_b // 2]
Pr = 17.16073 # Dawson+ 2021
print("input: Tref = ", Tr, "Pref = ", Pr)
Tref_b, Pref_b, chi2_b, epoch_b, Tlin_b, oc_b_days = linear_ephemeris(
    tra_syn_b, eT0s=None, 
    Tref_in = Tr, Pref_in = Pr, 
    fit=True
)
print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
    "synthetic",
    *Tref_b, *Pref_b, chi2_b, chi2_b/(n_tra_syn_b-2))
 )
lineph_syn[pl_letter] = (Tref_b, Pref_b)

ax.plot(
    tra_syn_b,
    oc_b_days*u[0],
    marker=markers[0],
    ms=1,
    color="C0",
    label="{} (synth.)".format(pl_letter),
    ls='',
)

Tr = tra_syn_noisy_b[n_tra_syn_b // 2]
Pr = 17.16073 # Dawson+ 2021
print("input: Tref = ", Tr, "Pref = ", Pr)
Tref_b, Pref_b, chi2_b, epoch_b, Tlin_b, oc_b_days = linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, 
    Tref_in = Tr, Pref_in = Pr, 
    fit=True
)
print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
    "noisy",
    *Tref_b, *Pref_b, chi2_b, chi2_b/(n_tra_syn_b-2))
 )
lineph_syn_noisy[pl_letter] = (Tref_b, Pref_b)

ax.errorbar(
    tra_syn_b,
    oc_b_days*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[1],
    ms=2,
    mfc='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

Tr = tra_syn_c[n_tra_syn_c // 2]
# Tr = tra_syn_c[0]
Pr = 34.525528 # Dawson+ 2021
print("input: Tref = ", Tr, "Pref = ", Pr)
Tref_c, Pref_c, chi2_c, epoch_c, Tlin_c, oc_c_days = linear_ephemeris(
    tra_syn_c, eT0s=None, 
    Tref_in = Tr, Pref_in = Pr, 
    fit=True
)
print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
    "synthetic",
    *Tref_c, *Pref_c, chi2_c, chi2_c/(n_tra_syn_c-2))
 )
lineph_syn[pl_letter] = (Tref_c, Pref_c)


ax.plot(
    tra_syn_c,
    oc_c_days*u[0],
    marker=markers[0],
    ms=1,
    color="C1",
    label="{} (synth.)".format(pl_letter),
    ls='',
)

Tr = tra_syn_noisy_c[n_tra_syn_c // 2]
Pr = 34.525528 # Dawson+ 2021
print("input: Tref = ", Tr, "Pref = ", Pr)
Tref_c, Pref_c, chi2_c, epoch_c, Tlin_c, oc_c_days = linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, 
    Tref_in = Tr, Pref_in = Pr, 
    fit=True
)
print("{:13s}: Tref = {:.5f} +/- {:.5f}, Pref = {:.5f} +/- {:.5f} with chi^2 = {:.2f} ==> chi^2_reduce = {:.4f}".format(
    "noisy",
    *Tref_c, *Pref_c, chi2_c, chi2_c/(n_tra_syn_c-2))
 )
lineph_syn_noisy[pl_letter] = (Tref_c, Pref_c)

ax.errorbar(
    tra_syn_c,
    oc_c_days*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2,
    mfc='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ### Set default-common parameters for `TRADES`

# %%
# pytrades.args_init(
#     n_body,
#     duration_check,
#     t_epoch=t_epoch,
#     t_start=t_start,
#     t_int=t_int,
#     encounter_check=True, # check for close encounters
#     do_hill_check=False, # check for stability condition based on Hill radius
#     amd_hill_check=False, # check for stability condition based on AMD-Hill criterion
#     rv_res_gls=False, # use GLS method on RV residuals to avoid introduction of signals close to the planetary periods
# )

transit_flag = [0, 1, 1] # 0 = not transiting (star), 1 transiting (b & c)

# load transits into memory
b_sources_id = np.ones(n_tra_syn_b).astype(int)
pytrades.set_t0_dataset(2, epoch_b, tra_syn_noisy_b, err_tra_syn_b, sources_id=b_sources_id)
c_sources_id = np.ones(n_tra_syn_c).astype(int)
pytrades.set_t0_dataset(3, epoch_c, tra_syn_noisy_c, err_tra_syn_c, sources_id=c_sources_id)

# %% [markdown]
# ## TTV analysis with basic parameters

# %% [markdown]
# ### define fitting parameters  
# Parameters as physical parameters, but masses as $M_\mathrm{p}/M_\star$

# %%
# this will be a global variable, it will not be passed to log-like/prob function
fit_labels = [
    "M_b/M_star", 
    "M_c/M_star",
    "P_b", 
    "P_c",
    "e_b", 
    "e_c",
    "w_b", 
    "w_c",
    "meana_b", 
    "meana_c",
]
n_fit = len(fit_labels)
print("Number of fitting parameters = {}".format(n_fit))

# let's define an initial set of parameters to test next functions
fit_pars_initial = [
    mass[1]/mass[0],
    mass[2]/mass[0],
    period[1],
    period[2],
    ecc[1],
    ecc[2],
    argp[1],
    argp[2],
    meana[1],
    meana[2]
]

# this will be a global variable, it will not be passed to log-like/prob function
# using tight boundaries just for simplicity
fit_boundaries =[
    [0.01*cst.Mjups, 1.0*cst.Mjups],
    [0.01*cst.Mjups, 1.0*cst.Mjups],
    [period[1]-1.0, period[1]+1.0],
    [period[2]-1.0, period[2]+1.0],
    [0.0, 0.5],
    [0.0, 0.5],
    [0.0, 360.0],
    [0.0, 360.0],
    [0.0, 360.0],
    [0.0, 360.0],
]

# let's define a set of parameter to test with random values
np.random.seed(seed=123456)
fit_pars_test = [
    uniform.rvs(loc=fit_boundaries[i][0], scale=np.ptp(fit_boundaries[i])) for i in range(0, n_fit)
]

# %% [markdown]
# #### define functions to convert:  
# - fitting parameters to physical
# - check boundaries  
# - computes priors, if needed  
# - computes the log-likelihood
# - computes the log-prior, if needed  

# %%
def fitting_to_physical_params(fit_pars):

    m_x = np.zeros((n_body))
    r_x = radius.copy() # fixed
    p_x = m_x.copy()
    e_x = m_x.copy()
    w_x = m_x.copy()
    ma_x = m_x.copy()
    i_x = inc.copy() # fixed
    ln_x = longn.copy() # fixed
    
    m_x[0] = mass[0] # not fitting stellar mass, global variable
    ifit = 0
    m_x[1] = fit_pars[ifit] * mass[0]
    ifit += 1
    m_x[2] = fit_pars[ifit] * mass[0]
    
    ifit += 1
    p_x[1] = fit_pars[ifit]
    ifit += 1
    p_x[2] = fit_pars[ifit]
    
    ifit += 1
    e_x[1] = fit_pars[ifit]
    ifit += 1
    e_x[2] = fit_pars[ifit]
    
    ifit += 1
    w_x[1] = fit_pars[ifit]
    ifit += 1
    w_x[2] = fit_pars[ifit]
    
    ifit += 1
    ma_x[1] = fit_pars[ifit]
    ifit += 1
    ma_x[2] = fit_pars[ifit]
    
    return m_x, r_x, p_x, e_x, w_x, ma_x, i_x, ln_x 

# %%
def check_fitting_boundaries(fit_pars):

    for ifit, ibound in enumerate(fit_boundaries):
        p = fit_pars[ifit]
        if (p < ibound[0]) or ( p > ibound[1]):
            return False

    return True

# %%
def fitting_to_observables(fit_pars):

    m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit = fitting_to_physical_params(fit_pars)
    
    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = pytrades.kelements_to_observed_t0s(
        t_epoch,
        t_start,
        t_int,
        m_fit,
        r_fit,
        p_fit,
        e_fit,
        w_fit,
        ma_fit,
        i_fit,
        ln_fit,
        transit_flag
    )
    # print("kel to T0s")
    return (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    )

def fitting_to_observables_dict(fit_pars):

    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = fitting_to_observables(fit_pars)

    transits = {}
    for pl_letter, pl_number in zip(body_names[1:], [2,3]):
        sel_pl = body_flag_sim == pl_number
        n_tra = np.sum(sel_pl)
        print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_tra, t_int))
        transits[pl_letter] = {
            "n_transits":  n_tra,
            "transit_times": transits_sim[sel_pl],
            "transit_durations": durations_sim[sel_pl],
            "lambda_rm": lambda_rm_sim[sel_pl],
            "kep_elem": kep_elem_sim[sel_pl],
        }
    
    return transits

# %% [markdown]
# MCMC algorithms and MAP optimization are concerned with the shape of the posterior distribution and finding its mode(s), 
# not its absolute normalization.  
# When you work with log-probabilities, 
# adding a constant term to the log-posterior does not change the location of the maximum or 
# the relative probabilities between different points in parameter space.  
# So, in case of emcee the uniform priors (ln P(\theta) = ln ( 1 / (max - min) )) can be discarded,
# and only Normal-Gaussian (or other types) prios can be computed in log-Probability.  
# It is mandatory to compute the log-prior for uniform priors in case of
# Nested Sampling and/or Model selection (Bayes factor) analysis

# %%
ln_const = -0.5*(n_tra_syn_b+n_tra_syn_c)*np.log(2.0*np.pi) # let's compute this only once

def log_boundaries(fit_pars):

    check_bounds = check_fitting_boundaries(fit_pars)
    if not check_bounds:
        return -np.inf
    return 0.0

def log_likelihood(fit_pars):

    lnL = ln_const
    
    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = fitting_to_observables(fit_pars)
    if not stable:
        return -np.inf

    res_b = tra_syn_noisy_b - transits_sim[body_flag_sim ==2]
    wres_b = res_b / err_tra_syn_b
    lnL_b = -0.5*np.sum(np.log(err_tra_syn_b)) - 0.5*np.sum(wres_b*wres_b)
    
    res_c = tra_syn_noisy_c - transits_sim[body_flag_sim ==3]
    wres_c = res_c / err_tra_syn_c
    lnL_c = -0.5*np.sum(np.log(err_tra_syn_c)) - 0.5*np.sum(wres_c*wres_c)
    
    lnL += lnL_b + lnL_c
    
    return lnL

def log_probability(fit_pars):

    ln_prior = log_boundaries(fit_pars)
    if np.isinf(ln_prior):
        return ln_prior

    lnP = log_likelihood(fit_pars)
    if np.isinf(lnP):
        return lnP
    lnP += ln_prior
    return lnP

# %% [markdown]
# #### test functions

# %%
print("fit_pars_initial")
print(*fit_labels)
print(*fit_pars_initial)
m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit = fitting_to_physical_params(fit_pars_initial)
print("mass (Msun)    = ",*m_fit)
print("radius (Rsun)  = ",*r_fit)
print("period (days)  = ",*p_fit)
print("ecc            = ",*e_fit)
print("arg. peri. (°) = ",*w_fit)
print("mean anom. (°) = ",*ma_fit)
print("inc (°)        = ",*i_fit)
print("long. node (°) = ",*ln_fit)
check = check_fitting_boundaries(fit_pars_initial)
print("check boundaries - fitting pars within boundaries?", check)
lnP_0 = log_probability(fit_pars_initial)
print("logP = ",lnP_0)
(
    body_flag_initial,
    epo_initial,
    transits_initial,
    _,
    _,
    _,
    stable_initial,
) = fitting_to_observables(fit_pars_initial)

# %%
print("fit_pars_test")
print(*fit_labels)
print(*fit_pars_test)
m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit = fitting_to_physical_params(fit_pars_test)
print("mass (Msun)    = ",*m_fit)
print("radius (Rsun)  = ",*r_fit)
print("period (days)  = ",*p_fit)
print("ecc            = ",*e_fit)
print("arg. peri. (°) = ",*w_fit)
print("mean anom. (°) = ",*ma_fit)
print("inc (°)        = ",*i_fit)
print("long. node (°) = ",*ln_fit)
check = check_fitting_boundaries(fit_pars_test)
print("check boundaries - fitting pars within boundaries?", check)
lnP = log_probability(fit_pars_test)
print("logP = ",lnP)
(
    body_flag_test,
    epo_test,
    transits_test,
    _,
    _,
    _,
    stable_test,
    ) = fitting_to_observables(fit_pars_test)

# %% [markdown]
# #### plot tests

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]
_, _, chi2_b, epoch_b, Tlin_b, oc_b= linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, Tref_in = Tref_b, Pref_in = Pref_b, fit=False
)
tra_initial = transits_initial[body_flag_initial == i+2]
oc_initial_b = tra_initial - (Tref_b[0] + epo_initial[body_flag_initial == i+2]*Pref_b[0])
tra_test = transits_test[body_flag_test == i+2]
oc_test_b = tra_test - (Tref_b[0] + epo_test[body_flag_test == i+2]*Pref_b[0])

ax.errorbar(
    tra_syn_noisy_b,
    oc_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2,
    mec='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_initial,
    oc_initial_b*u[0],
    color="black",
    marker=markers[0],
    ms=2,
    mfc='None',
    mew=0.4,
    label="{} (initial)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_test,
    oc_test_b*u[0],
    color="C0",
    marker=markers[0],
    ms=2.3,
    mfc='None',
    mew=0.45,
    label="{} (test)".format(pl_letter),
    ls='',
)


i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]
_, _, chi2_c, epoch_c, Tlin_c, oc_c= linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, Tref_in = Tref_c, Pref_in = Pref_c, fit=False
)
tra_initial = transits_initial[body_flag_initial == i+2]
oc_initial_c = tra_initial - (Tref_c[0] + epo_initial[body_flag_initial == i+2]*Pref_c[0])
tra_test = transits_test[body_flag_test == i+2]
oc_test_c = tra_test - (Tref_c[0] + epo_test[body_flag_test == i+2]*Pref_c[0])

ax.errorbar(
    tra_syn_noisy_c,
    oc_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2,
    mec='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_initial,
    oc_initial_c*u[0],
    color="black",
    marker=markers[1],
    ms=2,
    mfc='None',
    mew=0.4,
    label="{} (fit)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_test,
    oc_test_c*u[0],
    color="C1",
    marker=markers[1],
    ms=2.3,
    mfc='None',
    mew=0.45,
    label="{} (test)".format(pl_letter),
    ls='',
)

for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ### Run a Differential Evolution (DE)  
# with `pyDE` to search the parameter space

# %% [markdown]
# #### Define parameters for `pyDE`

# %%
# f: the difference amplification factor. Values of 0.5-0.8 are good in most cases.
de_f = 0.5
# c: The cross-over probability. Use 0.9 to test for fast convergence, and smaller values (~0.1) for a more elaborate search.
de_c = 0.5
# -maximise (True) or minimise (False)
de_maximize = True
de_fit_type = -1 if de_maximize else 1
# n_pop: number of population, that is the number of different configuration to test at each generation (a step, or a run)
n_pop = n_fit * 4 # suggested n_pop(min) = n_fit * 2, n_pop(ok) = n_fit * 4, n_pop(good)= n_fit * 10
# n_gen: numer of generations, that is the number of evolution of the configurations, it stops when reach this number
n_gen = 5000 # 100-1000 are very low number of generation, just to show how it works
iter_print = n_gen // 10

seed = 42
n_threads = min(n_pop // 1, len(os.sched_getaffinity(0))) # just a test, usually // 2 is ok
# probably using newer version of python and os has the function os.process_cpu_count()

load_de = True

print(
    "number of the population = {}".format(n_pop),
    "number of the generation = {}".format(n_gen),
    "number of the threads = {}".format(n_threads),
    "seed = {}".format(seed),
    sep="\n"
)

# %% [markdown]
# ### Single run of `pyDE`  
# the output will be sent to `emcee`

# %%
skip_this = True
if not skip_this:
    with Pool(n_threads) as pool:
        de_obj = pyde.DiffEvol(
            log_probability,
            fit_boundaries,
            n_pop,
            f=de_f,
            c=de_c,
            seed=seed,
            maximize=de_fit_type,
            pool=pool
        )
        de_obj.optimize(n_gen)

# %%
# np.shape(de_obj.population)  # n_pop x n_fit

# %% [markdown]
# ### run of `pyDE` as iterator, to save DE evolution

# %%
de_pop = np.zeros((n_gen, n_pop, n_fit))
de_fitness = np.zeros((n_gen, n_pop)) - np.inf
de_pop_best = np.zeros((n_gen, n_fit))
de_fitness_best = np.zeros((n_gen)) - np.inf

if load_de:
    # de_obj = pickle.load(open('de_obj.pkl', 'rb'))
    de_pop = pickle.load(open('de_pop.pkl', 'rb'))
    de_fitness = pickle.load(open('de_fitness.pkl', 'rb'))
    de_pop_best = pickle.load(open('de_pop_best.pkl', 'rb'))
    de_fitness_best = pickle.load(open('de_fitness_best.pkl', 'rb'))
else:
    # pool = Pool(n_threads)
    with Pool(n_threads) as pool:
        de_obj = pyde.DiffEvol(
            log_probability,
            fit_boundaries,
            n_pop,
            f=de_f,
            c=de_c,
            seed=seed,
            maximize=de_fit_type,
            pool=pool
        )
        for iter_de, res_de in enumerate(de_obj(n_gen)):
            de_pop[iter_de, :, :]    = de_obj.population.copy()
            de_fitness[iter_de, :]   = de_fit_type * de_obj._fitness.copy()
            de_pop_best[iter_de, :]  = de_obj.minimum_location.copy()
            de_fitness_best[iter_de] = de_fit_type * de_obj.minimum_value
        
            if ((iter_de + 1) % iter_print) == 0:
                print("Completed iter {:5d} / {:5d} ({:5.1f}%)".format(iter_de+1, n_gen, 100*(iter_de+1)/n_gen))
    
    # pool.close()
    # # pool.terminate()
    # pool.join()
if not load_de:
    # pickle.dump(de_obj, open('de_obj.pkl', 'wb'))
    pickle.dump(de_pop, open('de_pop.pkl', 'wb'))
    pickle.dump(de_fitness, open('de_fitness.pkl', 'wb'))
    pickle.dump(de_pop_best, open('de_pop_best.pkl', 'wb'))
    pickle.dump(de_fitness_best, open('de_fitness_best.pkl', 'wb'))

# %%
# np.shape(de_obj.population)  # n_pop x n_fit

# %%
fit_pars_de = de_pop_best[-1]
for n, p, pd in zip(fit_labels, fit_pars_initial, fit_pars_de):
    print(n, p, pd)
print("lnP", lnP_0, de_fitness_best[-1])

# %%
(
    body_flag_de,
    epo_de,
    transits_de,
    _,
    _,
    _,
    stable_de,
) = fitting_to_observables(fit_pars_de)

# %% [markdown]
# #### plot `pyDE`

# %%
skip_this = False

if not skip_this:
    xx = np.repeat(np.arange(n_gen), n_pop)
    cc = de_fitness.reshape((n_gen * n_pop))
    
    for ifit in range(0,n_fit):
    # ifit = 2
        fig = plt.figure(figsize=(5,3))
        yy = de_pop[:,:,ifit].reshape((n_gen * n_pop))
        plt.scatter(
            xx,
            yy,
            s=2,
            c=cc,
            vmin=np.percentile(cc, 50),
            vmax=np.max(cc),
            alpha=0.5,
            edgecolors='None',
            linewidths=0.0,
        )
        plt.colorbar(label="fitness")
        plt.ylabel(fit_labels[ifit])
        plt.xlabel("n_gen")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]
_, _, chi2_b, epoch_b, Tlin_b, oc_b= linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, Tref_in = Tref_b, Pref_in = Pref_b, fit=False
)
tra_de = transits_de[body_flag_de == i+2]
oc_de_b = tra_de - (Tref_b[0] + epo_de[body_flag_de == i+2]*Pref_b[0])

ax.errorbar(
    tra_syn_noisy_b,
    oc_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2,
    mec='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_de,
    oc_de_b*u[0],
    color="black",
    marker=markers[0],
    ms=2,
    mfc='None',
    mew=0.4,
    label="{} (DE)".format(pl_letter),
    ls='',
)

i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]
_, _, chi2_c, epoch_c, Tlin_c, oc_c= linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, Tref_in = Tref_c, Pref_in = Pref_c, fit=False
)
tra_de = transits_de[body_flag_de == i+2]
oc_de_c = tra_de - (Tref_c[0] + epo_de[body_flag_de == i+2]*Pref_c[0])


ax.errorbar(
    tra_syn_noisy_c,
    oc_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2,
    mec='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_de,
    oc_de_c*u[0],
    color="black",
    marker=markers[1],
    ms=2,
    mfc='None',
    mew=0.4,
    label="{} (DE)".format(pl_letter),
    ls='',
)


for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ### `pyDE` to `emcee`  

# %% [markdown]
# #### Define parameters for `emcee`  

# %%
n_walkers = n_pop # same as `pyde`
n_steps = 1000
thin_by = 10 # apply a thinning factor while running, it means that emcee will run for n_steps x thin_by, but it will keep/return only n_steps values

load_emcee = True

# define the move / sampling step
# default is the Affine-Invariant Ensemble MCMC (A.I.E.M.):
# emcee_move = [
#     (emcee.moves.StretchMove(), 1.0),
# ]
#
# for complex problems the emcee's author suggested sampling with DE80%+DESnooker20%
emcee_move = [
    (emcee.moves.DEMove(), 0.8),
    (emcee.moves.DESnookerMove(), 0.2),
]

# just for progress bar
pka = {
    'ncols': 75,
    'dynamic_ncols': False,
    'position': 0
}

# %% [markdown]
# #### run of `emcee`

# %%
backend_filename = os.path.join(os.path.abspath("./"), "sampler.hdf5")

if load_emcee:
    sampler = emcee.backends.HDFBackend(backend_filename, read_only=True)
else:
    backend = emcee.backends.HDFBackend(backend_filename, compression="gzip")
    backend.reset(n_walkers, n_fit)
    
    with Pool(n_threads) as pool:
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_fit,
            log_probability,
            pool=pool,
            moves=emcee_move,
            backend=backend
        )
    
        sampler.run_mcmc(
            de_pop[-1, :, :], # last population of pyde
            n_steps,
            thin_by=thin_by,
            store=True,
            tune=True,
            skip_initial_state_check=False,
            progress=True,
            progress_kwargs=pka
        )

# %% [markdown]
# #### statistics of `emcee` run

# %%
n_burnin = 400 # remove first n_burnin steps

# %%
full_chains = sampler.get_chain()
full_chains_flat = sampler.get_chain(flat=True)
post_chains = sampler.get_chain(discard=n_burnin)
post_chains_flat = sampler.get_chain(discard=n_burnin, flat=True)
full_lnprob = sampler.get_log_prob()
full_lnprob_flat = sampler.get_log_prob(flat=True)
post_lnprob = sampler.get_log_prob(discard=n_burnin)
post_lnprob_flat = sampler.get_log_prob(discard=n_burnin, flat=True)
n_post, _ = np.shape(post_chains_flat)

# %%
map_idx = np.argmax(post_lnprob_flat)
map_lnprob = post_lnprob_flat[map_idx]
map_pars = post_chains_flat[map_idx, :]

# %%
# credible intervals
perc = np.array([68.27, 95.44, 99.74]) / 100.0
for i in range(n_fit):
    fitn, fitp, fitpost = fit_labels[i], map_pars[i], post_chains_flat[:, i]
    credint = [anc.hpd(fitpost, c) for c in perc]
    l = "{:12s}: MAP {:10.6f} ".format(fitn, fitp)
    for j, pc in enumerate(credint):
        l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
    print(l)
print()

# conversion of parameters

err_Mstar = 0.021 # Msun
Mstar_gaussian = norm.rvs(loc=mass[0], scale=err_Mstar, size=n_post)

post_Mb2s_flat = post_chains_flat[:, 0]
post_Mb_Me = post_Mb2s_flat * mass[0] * cst.Msear # needed to take the MAP value
map_Mb_Me = post_Mb_Me[map_idx]
post_Mb_Me_noisy = post_Mb2s_flat * Mstar_gaussian * cst.Msear # needed to take into account the uncertainty on Mstar for the credible interval
credint = [anc.hpd(post_Mb_Me_noisy, c) for c in perc]
l = "{:12s}: MAP {:10.6f} ".format("M_b", map_Mb_Me)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_Mb_Me = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_Mc2s_flat = post_chains_flat[:, 1]
post_Mc_Me = post_Mc2s_flat * mass[0] * cst.Msear # needed to take the MAP value
map_Mc_Me = post_Mc_Me[map_idx]
post_Mc_Me_noisy = post_Mc2s_flat * Mstar_gaussian * cst.Msear # needed to take into account the uncertainty on Mstar for the credible interval
credint = [anc.hpd(post_Mc_Me_noisy, c) for c in perc]
l = "{:12s}: MAP {:10.6f} ".format("M_c", map_Mc_Me)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_Mc_Me = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

print()
print("This example:")
print("M_b = {:10.6f} +/- {:10.6f} Mearth".format(map_Mb_Me, err_Mb_Me))
print("M_c = {:10.6f} +/- {:10.6f} Mearth".format(map_Mc_Me, err_Mc_Me))

print("McKee values:")
print("M_b = {:10.6f} +/- {:10.6f} Mearth".format(0.0554*cst.Mjups*cst.Msear, 0.0020*cst.Mjups*cst.Msear))
print("M_c = {:10.6f} +/- {:10.6f} Mearth".format(0.525*cst.Mjups*cst.Msear, 0.019*cst.Mjups*cst.Msear))

# %% [markdown]
# Perfectly consistent!

# %% [markdown]
# #### trace plot

# %%
# log-Probability
# same but using `trades` function
log_probability_trace(
    full_lnprob,
    post_lnprob_flat,
    None,
    n_burn=n_burnin,
    n_thin=thin_by,
    show_plot=True,
    figsize=(5, 5),
    olog=None,
)

# %%
skip_this = False

if not skip_this:
    # let's use `trades` function to plot all the chains/trace plot of each parameter
    exp_acf_fit, exp_steps_fit = full_statistics(
        full_chains,
        post_chains_flat,
        fit_labels,
        map_pars,
        post_lnprob_flat,
        None,
        olog=None,
        ilast=0,
        n_burn=n_burnin,
        n_thin=thin_by,
        show_plot=True,
        figsize=(5, 5),
    )

# %% [markdown]
# #### corner plot

# %%
ticklabel_size = 4
label_separation = -1.1
label_size = 6
k = anc.get_auto_bins(post_chains_flat)

GTC = pygtc.plotGTC(
    chains=post_chains_flat,
    paramNames=fit_labels,
    nContourLevels=3,
    nBins=k,
    truths=map_pars,
    truthLabels=("MAP"),
    figureSize=plt.rcParams["figure.figsize"][0],
    mathTextFontSet=plt.rcParams["mathtext.fontset"],
    customLabelFont={"family": plt.rcParams["font.family"], "size": label_size},
    customTickFont={"family": plt.rcParams["font.family"], "size": ticklabel_size},
    customLegendFont={"family": plt.rcParams["font.family"], "size": label_size},
    legendMarker='All',
    labelRotation=(True, False),
)
axs = GTC.axes
for ax in axs:
    ax.tick_params(
        direction='inout',
        pad=4,
        size=3,
        labelsize=ticklabel_size
    )
    lb = ax.get_xlabel()
    if lb != "":
        ax.xaxis.set_label_coords(0.5, label_separation)
        ax.set_xlabel(lb, fontsize=label_size, rotation=45.0)
    
    lb = ax.get_ylabel()
    if lb != "":
        ax.yaxis.set_label_coords(label_separation, 0.5)
        ax.set_ylabel(lb, fontsize=label_size, rotation=45.0)

    for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.6)

plt.show()
plt.close(GTC)

# %% [markdown]
# #### O-C plot w/ samples  
# Let's plot the O-C plot with the samples with shades at different credible intervals.

# %%
map_transits = fitting_to_observables_dict(map_pars)

# %%
(
    mass_map, 
    radius_map,
    period_map, 
    ecc_map, 
    argp_map, 
    meana_map,
    inc_map,
    longn_map,
) = fitting_to_physical_params(map_pars)
print("m ", *mass_map  , "(", *mass, ")")
print("r ", *radius_map, "(", *radius, ")")
print("p ", *period_map, "(", *period, ")")
print("e ", *ecc_map   , "(", *ecc, ")")
print("w ", *argp_map  , "(", *argp, ")")
print("ma", *meana_map , "(", *meana, ")")
print("i ", *inc_map   , "(", *inc, ")")
print("ln", *longn_map , "(", *longn, ")")

# %%
t_int_syn = time_sel_end

_, _, _, map_transits_full = run_and_get_transits(
    t_epoch, 
    t_start, 
    t_int_syn,
    mass_map,
    radius_map,
    period_map, 
    ecc_map,
    argp_map, 
    meana_map,
    inc_map,
    longn_map,
    body_names[1:],
)

# %%
n_samples = 33
smp_tra = {}
smp_idx = np.random.choice(n_post, n_samples, replace=False)
for ismp in smp_idx:
    smp_pars = post_chains_flat[ismp, :]
    (
        mass_smp, 
        radius_smp,
        period_smp, 
        ecc_smp, 
        argp_smp, 
        meana_smp,
        inc_smp,
        longn_smp,
    ) = fitting_to_physical_params(smp_pars)
    _, _, _, smp_transits_full = run_and_get_transits(
        t_epoch, 
        t_start, 
        t_int_syn,
        mass_smp,
        radius_smp,
        period_smp, 
        ecc_smp,
        argp_smp, 
        meana_smp,
        inc_smp,
        longn_smp,
        body_names[1:],
    )
    smp_tra[ismp] = smp_transits_full

# %%
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(hspace=0.07, wspace=0.25)

c1, c2, c3 = 0.6827, 0.9544, 0.9974
hc1, hc2, hc3 = c1*0.5, c2*0.5, c3*0.5

lfont = 8
tfont = 6

zo_map = 10
zo_obs = zo_map-1
zo_mod = zo_obs -1
zo_1s = zo_mod - 1
zo_2s = zo_1s - 1
zo_3s = zo_2s - 1

cfsm = plt.get_cmap("gray")
gval = 0.6
dg = 0.1

axs = []
nrows = 6 # (2 + 1) * 2
ncols = 1

u = [1.0, "days"]
markers = anc.filled_markers

all_xlims = []

# =================================================================
i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))

ax = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=2)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("O-C ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]
_, _, chi2_b, epoch_b, Tlin_b, oc_b= linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, Tref_in = Tref_b, Pref_in = Pref_b, fit=False
)
print("Tref_b: {:12.6f} Pref_b: {:12.6f}".format(Tref_b[0], Pref_b[0]))

tra_map_b = map_transits[pl_letter]["transit_times"]
epo_map_b = compute_epoch(Tref_b[0], Pref_b[0], tra_map_b)
tln_map_b = Tref_b[0] + epo_map_b*Pref_b[0]
oc_map_b = tra_map_b - tln_map_b
res_map_b = tra_syn_noisy_b - tra_map_b

tra_map_full_b = map_transits_full[pl_letter]["transit_times"]
epo_map_full_b = compute_epoch(Tref_b[0], Pref_b[0], tra_map_full_b)
tln_map_full_b = Tref_b[0] + epo_map_full_b*Pref_b[0]
oc_map_full_b = tra_map_full_b - tln_map_full_b

ax.errorbar(
    tra_syn_noisy_b,
    oc_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2.5,
    mec='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

ax.plot(
    tra_syn_noisy_b,
    oc_map_b*u[0],
    color="black",
    marker=markers[0],
    ms=2.5,
    mfc='None',
    mew=0.4,
    label="{} (map)".format(pl_letter),
    ls='',
    zorder=zo_map
)

ax.plot(
    tra_map_full_b,
    oc_map_full_b*u[0],
    color="black",
    marker='o',
    ms=0.6,
    ls='-',
    lw=0.3,
    label="{} (full map)".format(pl_letter),
    zorder=zo_mod
)

oc_smp = []
for ksmp, vsmp in smp_tra.items():
    tra_xxx = vsmp[pl_letter]["transit_times"]
    epo_xxx = compute_epoch(Tref_b[0], Pref_b[0], tra_xxx)
    tln_xxx = Tref_b[0] + epo_xxx*Pref_b[0]
    oc_xxx = tra_xxx - tln_xxx
    oc_smp.append(oc_xxx)
oc_smp = np.array(oc_smp).T * u[0]
hdi1 = np.percentile(oc_smp, [50 - (100*hc1), 50 + (100*hc1)], axis=1).T
hdi2 = np.percentile(oc_smp, [50 - (100*hc2), 50 + (100*hc2)], axis=1).T
hdi3 = np.percentile(oc_smp, [50 - (100*hc3), 50 + (100*hc3)], axis=1).T
ax.fill_between(
    tra_map_full_b,
    hdi1[:, 0],
    hdi1[:, 1],
    color=cfsm(gval),
    alpha=1.0,
    lw=0.0,
    zorder=zo_1s,
)
ax.fill_between(
    tra_map_full_b,
    hdi2[:, 0],
    hdi2[:, 1],
    color=cfsm(gval+dg),
    alpha=1.0,
    lw=0.0,
    zorder=zo_2s,
)
ax.fill_between(
    tra_map_full_b,
    hdi3[:, 0],
    hdi3[:, 1],
    color=cfsm(gval+(dg*2)),
    alpha=1.0,
    lw=0.0,
    zorder=zo_3s,
)

all_xlims.append(ax.get_xlim())

axs.append(ax)

# ---
ax = plt.subplot2grid((nrows, ncols), (2, 0), rowspan=1)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("res. ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

ax.errorbar(
    tra_syn_noisy_b,
    res_map_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2.5,
    mec='black',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    # label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

axs.append(ax)

# =================================================================
i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = plt.subplot2grid((nrows, ncols), (3, 0), rowspan=2)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("O-C ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]
_, _, chi2_c, epoch_c, Tlin_c, oc_c= linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, Tref_in = Tref_c, Pref_in = Pref_c, fit=False
)
print("Tref_c: {:12.6f} Pref_c: {:12.6f}".format(Tref_c[0], Pref_c[0]))

tra_map_c = map_transits[pl_letter]["transit_times"]
epo_map_c = compute_epoch(Tref_c[0], Pref_c[0], tra_map_c)
tln_map_c = Tref_c[0] + epo_map_c*Pref_c[0]
oc_map_c = tra_map_c - tln_map_c
res_map_c = tra_syn_noisy_c - tra_map_c

tra_map_full_c = map_transits_full[pl_letter]["transit_times"]
epo_map_full_c = compute_epoch(Tref_c[0], Pref_c[0], tra_map_full_c)
tln_map_full_c = Tref_c[0] + epo_map_full_c*Pref_c[0]
oc_map_full_c = tra_map_full_c - tln_map_full_c

ax.errorbar(
    tra_syn_noisy_c,
    oc_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2.5,
    mec='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

ax.plot(
    tra_syn_noisy_c,
    oc_map_c*u[0],
    color="black",
    marker=markers[1],
    ms=2.5,
    mfc='None',
    mew=0.4,
    label="{} (map)".format(pl_letter),
    ls='',
    zorder=zo_map
)

ax.plot(
    tra_map_full_c,
    oc_map_full_c*u[0],
    color="black",
    marker='o',
    ms=0.6,
    ls='-',
    lw=0.3,
    label="{} (full map)".format(pl_letter),
    zorder=zo_mod
)

oc_smp = []
for ksmp, vsmp in smp_tra.items():
    tra_xxx = vsmp[pl_letter]["transit_times"]
    epo_xxx = compute_epoch(Tref_c[0], Pref_c[0], tra_xxx)
    tln_xxx = Tref_c[0] + epo_xxx*Pref_c[0]
    oc_xxx = tra_xxx - tln_xxx
    oc_smp.append(oc_xxx)
oc_smp = np.array(oc_smp).T * u[0]
hdi1 = np.percentile(oc_smp, [50 - (100*hc1), 50 + (100*hc1)], axis=1).T
hdi2 = np.percentile(oc_smp, [50 - (100*hc2), 50 + (100*hc2)], axis=1).T
hdi3 = np.percentile(oc_smp, [50 - (100*hc3), 50 + (100*hc3)], axis=1).T
ax.fill_between(
    tra_map_full_c,
    hdi1[:, 0],
    hdi1[:, 1],
    color=cfsm(gval),
    alpha=1.0,
    lw=0.0,
    zorder=zo_1s,
)
ax.fill_between(
    tra_map_full_c,
    hdi2[:, 0],
    hdi2[:, 1],
    color=cfsm(gval+dg),
    alpha=1.0,
    lw=0.0,
    zorder=zo_2s,
)
ax.fill_between(
    tra_map_full_c,
    hdi3[:, 0],
    hdi3[:, 1],
    color=cfsm(gval+(dg*2)),
    alpha=1.0,
    lw=0.0,
    zorder=zo_3s,
)

all_xlims.append(ax.get_xlim())

axs.append(ax)

# ---
ax = plt.subplot2grid((nrows, ncols), (5, 0), rowspan=1)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=True)
ax.set_ylabel("res. ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

ax.errorbar(
    tra_syn_noisy_c,
    res_map_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2.5,
    mec='black',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    ls='',
    zorder=zo_obs
)

axs.append(ax)

all_xlims = np.concatenate(all_xlims)
for ax in axs:
    ax.set_xlim(np.min(all_xlims), np.max(all_xlims))
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=lfont, frameon=False)

ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ## TTV analysis with reparameterisation

# %% [markdown]
# ### define fitting parameters  
# Parameters as:  
# - masses as $M_\star, \log_{10} M_\mathrm{b}/M_\star$ and $\log_{10} M_\mathrm{c}/M_\mathrm{b}$ (in $\log_{10}$ for better sampling small values)  
# - $(e, \omega) \rightarrow (\sqrt{e} \cos \omega, \sqrt{e} \sin \omega)$  
# - $\mathcal{M} \rightarrow \lambda = \mathcal{M} + \omega + \Omega = $ mean longitude
# 
# In this test I will fit the mass of the star with a normal prior.

# %%
# this will be a global variable, it will not be passed to log-like/prob function
fit_labels_repar = [
    "M_star",
    "log10(M_b/M_star)", 
    "log10(M_c/M_b)",
    "P_b", 
    "P_c",
    "secosw_b", 
    "sesinw_b", 
    "secosw_c",
    "sesinw_c",
    "meanl_b", 
    "meanl_c",
]
n_fit_repar = len(fit_labels_repar)
print("Number of fitting parameters = {}".format(n_fit_repar))

# let's define an initial set of parameters to test next functions
fit_pars_initial_repar = [
    mass[0],
    np.log10(mass[1]/mass[0]),
    np.log10(mass[2]/mass[1]),
    period[1],
    period[2],
    np.sqrt(ecc[1])*np.cos(argp[1]*cst.deg2rad),
    np.sqrt(ecc[1])*np.sin(argp[1]*cst.deg2rad),
    np.sqrt(ecc[2])*np.cos(argp[2]*cst.deg2rad),
    np.sqrt(ecc[2])*np.sin(argp[2]*cst.deg2rad),
    meanl[1],
    meanl[2]
]

# let's define physical boundaries
M_s_boundaries = [0.01, 2.0]
M_p_boundaries = [0.01*cst.Mjups, 1.0*cst.Mjups]
ecc_boundaries = [0.0, 0.5]
ang_boundaries = [0.0, 360.0]

# this will be a global variable, it will not be passed to log-like/prob function
# using tight boundaries just for simplicity
fit_boundaries_repar =[
    M_s_boundaries, # Msun
    [-10, 0], # 10^-15, 1 as Mb/Ms
    [-10, 3], # 10^-15, 1 as Mc/Mb
    [period[1]-2.0, period[1]+2.0],
    [period[2]-2.0, period[2]+2.0],
    [-np.sqrt(np.max(ecc_boundaries[1])), +np.sqrt(np.max(ecc_boundaries[1]))],
    [-np.sqrt(np.max(ecc_boundaries[1])), +np.sqrt(np.max(ecc_boundaries[1]))],
    [-np.sqrt(np.max(ecc_boundaries[1])), +np.sqrt(np.max(ecc_boundaries[1]))],
    [-np.sqrt(np.max(ecc_boundaries[1])), +np.sqrt(np.max(ecc_boundaries[1]))],
    ang_boundaries, # meanl_b
    ang_boundaries, # meanl_b
]

priors_repar = [
    norm(loc=0.763, scale=0.021)
] + [
    uniform(loc=bd[0], scale=np.ptp(bd)) for bd in fit_boundaries_repar[1:]
]

# let's define a set of parameter to test with random values
np.random.seed(seed=123456)
# fit_pars_repar_test = [
#     pp.rvs() for pp in priors_repar
# ]
fit_pars_repar_test = fit_pars_initial_repar

# %% [markdown]
# #### define functions to convert:  
# - fitting parameters to physical
# - check boundaries  
# - computes priors, if needed  
# - computes the log-likelihood
# - computes the log-prior, if needed  

# %%
def fitting_to_physical_params_repar(fit_pars):

    m_x = np.zeros((n_body))
    r_x = radius.copy() # fixed
    p_x = np.zeros((n_body))
    e_x = np.zeros((n_body))
    w_x = np.zeros((n_body))
    ma_x = np.zeros((n_body))
    i_x = inc.copy() # fixed
    ln_x = longn.copy() # fixed
    
    ifit = 0
    m_x[0] = fit_pars[ifit]
    ifit += 1
    m_x[1] = 10**(fit_pars[ifit]) * m_x[0]
    ifit += 1
    m_x[2] = 10**(fit_pars[ifit]) * m_x[1]
    
    ifit += 1
    p_x[1] = fit_pars[ifit]
    ifit += 1
    p_x[2] = fit_pars[ifit]
    
    ifit += 1
    e_x[1] = fit_pars[ifit]**2 + fit_pars[ifit+1]**2
    w_x[1] = (np.arctan2(fit_pars[ifit+1], fit_pars[ifit])*cst.rad2deg)%360.0
    
    ifit += 2
    e_x[2] = fit_pars[ifit]**2 + fit_pars[ifit+1]**2
    w_x[2] = (np.arctan2(fit_pars[ifit+1], fit_pars[ifit])*cst.rad2deg)%360.0
    
    ifit += 2
    ma_x[1] = (fit_pars[ifit] - w_x[1] - ln_x[1])%360.0
    ifit += 1
    ma_x[2] = (fit_pars[ifit] - w_x[2] - ln_x[2])%360.0
    
    return m_x, r_x, p_x, e_x, w_x, ma_x, i_x, ln_x 

# %%
def check_fitting_boundaries_repar(fit_pars):

    for ifit, ibound in enumerate(fit_boundaries_repar):
        p = fit_pars[ifit]
        if (p < ibound[0]) or ( p > ibound[1]):
            return False
    
    return True

# %%
def fitting_to_observables_repar(fit_pars):

    m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit = fitting_to_physical_params_repar(fit_pars)
    
    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = pytrades.kelements_to_observed_t0s(
        t_epoch,
        t_start,
        t_int,
        m_fit,
        r_fit,
        p_fit,
        e_fit,
        w_fit,
        ma_fit,
        i_fit,
        ln_fit,
        transit_flag
    )
    # print("kel to T0s")
    return (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    )

def fitting_to_observables_repar_dict(fit_pars):

    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = fitting_to_observables_repar(fit_pars)

    transits = {}
    for pl_letter, pl_number in zip(body_names[1:], [2,3]):
        sel_pl = body_flag_sim == pl_number
        n_tra = np.sum(sel_pl)
        print("planet {} (id {}) with {} transits in {:.0f} days of integration".format(pl_letter, pl_number, n_tra, t_int))
        transits[pl_letter] = {
            "n_transits":  n_tra,
            "transit_times": transits_sim[sel_pl],
            "transit_durations": durations_sim[sel_pl],
            "lambda_rm": lambda_rm_sim[sel_pl],
            "kep_elem": kep_elem_sim[sel_pl],
        }
    
    return transits

# %% [markdown]
# MCMC algorithms and MAP optimization are concerned with the shape of the posterior distribution and finding its mode(s), 
# not its absolute normalization.  
# When you work with log-probabilities, 
# adding a constant term to the log-posterior does not change the location of the maximum or 
# the relative probabilities between different points in parameter space.  
# So, in case of emcee the uniform priors (ln P(\theta) = ln ( 1 / (max - min) )) can be discarded,
# and only Normal-Gaussian (or other types) prios can be computed in log-Probability.  
# It is mandatory to compute the log-prior for uniform priors in case of
# Nested Sampling and/or Model selection (Bayes factor) analysis

# %%
ln_const = -0.5*(n_tra_syn_b+n_tra_syn_c)*np.log(2.0*np.pi) # let's compute this only once

def log_boundaries_repar(fit_pars):

    check_bounds = check_fitting_boundaries_repar(fit_pars)
    if not check_bounds:
        return -np.inf

    m_x, r_x, p_x, e_x, w_x, ma_x, i_x, ln_x = fitting_to_physical_params_repar(fit_pars)
    if not M_s_boundaries[0] <= m_x[0] <= M_s_boundaries[1]:
        return -np.inf
    if not M_p_boundaries[0] <= m_x[1] <= M_p_boundaries[1]:
        return -np.inf
    if not M_p_boundaries[0] <= m_x[2] <= M_p_boundaries[1]:
        return -np.inf
    if not ecc_boundaries[0] <= e_x[1] <= ecc_boundaries[1]:
        return -np.inf
    if not ecc_boundaries[0] <= e_x[2] <= ecc_boundaries[1]:
        return -np.inf
    
    return 0.0

def log_priors_repar(fit_pars): # this can be extended for uniform priors when using Nested Sampling

    ln_prior = 0.0

    ms = fit_pars[0]
    
    return priors_repar[0].logpdf(ms)

def log_likelihood_repar(fit_pars):

    lnL = ln_const
    
    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = fitting_to_observables_repar(fit_pars)
    if not stable:
        return -np.inf

    res_b = tra_syn_noisy_b - transits_sim[body_flag_sim ==2]
    wres_b = res_b / err_tra_syn_b
    lnL_b = -0.5*np.sum(np.log(err_tra_syn_b)) - 0.5*np.sum(wres_b*wres_b)
    
    res_c = tra_syn_noisy_c - transits_sim[body_flag_sim ==3]
    wres_c = res_c / err_tra_syn_c
    lnL_c = -0.5*np.sum(np.log(err_tra_syn_c)) - 0.5*np.sum(wres_c*wres_c)
    
    lnL += lnL_b + lnL_c
    
    return lnL

def log_probability_repar(fit_pars):

    ln_prior = log_boundaries_repar(fit_pars)
    if np.isinf(ln_prior):
        return ln_prior

    ln_prior += log_priors_repar(fit_pars)

    lnP = log_likelihood_repar(fit_pars)
    if np.isinf(lnP):
        return lnP
    lnP += ln_prior
    return lnP

# %% [markdown]
# #### plot tests

# %%
print("fit_pars_repar_test")
print(*fit_labels_repar)
print(*fit_pars_repar_test)
m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit = fitting_to_physical_params_repar(fit_pars_repar_test)
print("mass (Msun)    = ",*m_fit)
print("radius (Rsun)  = ",*r_fit)
print("period (days)  = ",*p_fit)
print("ecc            = ",*e_fit)
print("arg. peri. (°) = ",*w_fit)
print("mean anom. (°) = ",*ma_fit)
print("inc (°)        = ",*i_fit)
print("long. node (°) = ",*ln_fit)
check = check_fitting_boundaries_repar(fit_pars_repar_test)
print("check boundaries - fitting pars within boundaries?", check)
lnP = log_probability_repar(fit_pars_repar_test)
print("logP = ",lnP)
(
    body_flag_test,
    epo_test,
    transits_test,
    _,
    _,
    _,
    stable_test,
    ) = fitting_to_observables_repar(fit_pars_repar_test)
print("Is it stable? {}".format(bool(stable_test)))

# %% [markdown]
# ### Run a Differential Evolution (DE)  
# with `pyDE` to search the parameter space

# %% [markdown]
# #### Define parameters for `pyDE`

# %%
# f: the difference amplification factor. Values of 0.5-0.8 are good in most cases.
de_f = 0.5
# c: The cross-over probability. Use 0.9 to test for fast convergence, and smaller values (~0.1) for a more elaborate search.
de_c = 0.5
# -maximise (True) or minimise (False)
de_maximize = True
de_fit_type = -1 if de_maximize else 1
# n_pop: number of population, that is the number of different configuration to test at each generation (a step, or a run)
n_pop = n_fit_repar * 4 # suggested n_pop(min) = n_fit * 2, n_pop(ok) = n_fit * 4, n_pop(good)= n_fit * 10
# n_gen: numer of generations, that is the number of evolution of the configurations, it stops when reach this number
n_gen = 5000 # 100-1000 are very low number of generation, just to show how it works
iter_print = n_gen // 10

seed = 42
n_threads = min(n_pop // 1, len(os.sched_getaffinity(0))) # just a test, usually // 2 is ok
# probably using newer version of python and os has the function os.process_cpu_count()
                
load_de = True

print(
    "number of the population = {}".format(n_pop),
    "number of the generation = {}".format(n_gen),
    "number of the threads = {}".format(n_threads),
    "seed = {}".format(seed),
    sep="\n"
)

# %% [markdown]
# ### run of `pyDE` as iterator, to save DE evolution

# %%
de_pop_repar = np.zeros((n_gen, n_pop, n_fit_repar))
de_fitness_repar = np.zeros((n_gen, n_pop)) - np.inf
de_pop_best_repar = np.zeros((n_gen, n_fit_repar))
de_fitness_best_repar = np.zeros((n_gen)) - np.inf

if load_de:
    de_pop_repar = pickle.load(open('de_pop_repar.pkl', 'rb'))
    de_fitness_repar = pickle.load(open('de_fitness_repar.pkl', 'rb'))
    de_pop_best_repar = pickle.load(open('de_pop_best_repar.pkl', 'rb'))
    de_fitness_best_repar = pickle.load(open('de_fitness_best_repar.pkl', 'rb'))
else:
    with Pool(n_threads) as pool_repar:
        de_obj_repar = pyde.DiffEvol(
            log_probability_repar,
            fit_boundaries_repar,
            n_pop,
            f=de_f,
            c=de_c,
            seed=seed,
            maximize=de_fit_type,
            pool=pool_repar
        )
        for iter_de, res_de in enumerate(de_obj_repar(n_gen)):
            de_pop_repar[iter_de, :, :]    = de_obj_repar.population.copy()
            de_fitness_repar[iter_de, :]   = de_fit_type * de_obj_repar._fitness.copy()
            de_pop_best_repar[iter_de, :]  = de_obj_repar.minimum_location.copy()
            de_fitness_best_repar[iter_de] = de_fit_type * de_obj_repar.minimum_value
        
            if ((iter_de + 1) % iter_print) == 0:
                print("Completed iter {:5d} / {:5d} ({:5.1f}%)".format(iter_de+1, n_gen, 100*(iter_de+1)/n_gen))
            # print("{:5d}/{:5d}".format(iter_de+1, n_gen), end='\r')

# %%
if not load_de:
    pickle.dump(de_pop_repar, open('de_pop_repar.pkl', 'wb'))
    pickle.dump(de_fitness_repar, open('de_fitness_repar.pkl', 'wb'))
    pickle.dump(de_pop_best_repar, open('de_pop_best_repar.pkl', 'wb'))
    pickle.dump(de_fitness_best_repar, open('de_fitness_best_repar.pkl', 'wb'))

# %%
fit_pars_de_repar = de_pop_best_repar[-1]
for n, p, pd in zip(fit_labels_repar, fit_pars_initial_repar, fit_pars_de_repar):
    print(n, p, pd)
print("lnP", lnP_0, de_fitness_best_repar[-1])

# %%
(
    body_flag_de,
    epo_de,
    transits_de,
    _,
    _,
    _,
    stable_de,
) = fitting_to_observables_repar(fit_pars_de_repar)

# %% [markdown]
# #### plot `pyDE`

# %%
skip_this = False

if not skip_this:
    xx = np.repeat(np.arange(n_gen), n_pop)
    cc = de_fitness_repar.reshape((n_gen * n_pop))
    
    for ifit in range(0,n_fit_repar):
    # ifit = 2
        fig = plt.figure(figsize=(5,3))
        yy = de_pop_repar[:,:,ifit].reshape((n_gen * n_pop))
        plt.scatter(
            xx,
            yy,
            s=2,
            c=cc,
            vmin=np.percentile(cc, 50),
            vmax=np.max(cc),
            alpha=0.5,
            edgecolors='None',
            linewidths=0.0,
        )
        plt.colorbar(label="fitness")
        plt.ylabel(fit_labels_repar[ifit])
        plt.xlabel("n_gen")
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)

# %%
fig, axs = plt.subplots(2, 1, sharex=True, figsize=(5,3))

u = [1.0, "days"]
markers = anc.filled_markers

i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]
_, _, chi2_b, epoch_b, Tlin_b, oc_b= linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, Tref_in = Tref_b, Pref_in = Pref_b, fit=False
)
tra_de = transits_de[body_flag_de == i+2]
oc_de_b = tra_de - (Tref_b[0] + epo_de[body_flag_de == i+2]*Pref_b[0])

ax.errorbar(
    tra_syn_noisy_b,
    oc_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2,
    mec='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_de,
    oc_de_b*u[0],
    color="black",
    marker=markers[0],
    ms=2,
    mfc='None',
    mew=0.4,
    label="{} (DE)".format(pl_letter),
    ls='',
)

i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = axs[i]
ax.axhline(0, color="k", lw=0.8)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]
_, _, chi2_c, epoch_c, Tlin_c, oc_c= linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, Tref_in = Tref_c, Pref_in = Pref_c, fit=False
)
tra_de = transits_de[body_flag_de == i+2]
oc_de_c = tra_de - (Tref_c[0] + epo_de[body_flag_de == i+2]*Pref_c[0])


ax.errorbar(
    tra_syn_noisy_c,
    oc_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2,
    mec='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
)

ax.plot(
    tra_de,
    oc_de_c*u[0],
    color="black",
    marker=markers[1],
    ms=2,
    mfc='None',
    mew=0.4,
    label="{} (DE)".format(pl_letter),
    ls='',
)


for ax in axs:
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=8, frameon=False)
    ax.set_ylabel("O-C ({})".format(u[1]))

axs[0].xaxis.set_tick_params(labelbottom=False)
# ax.set_xlabel("Time $(\mathrm{BJD_{TDB}} - 2457000)$")
ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ### `pyDE` to `emcee`  

# %% [markdown]
# #### Define parameters for `emcee`  

# %%
n_walkers = n_pop # same as `pyde`
n_steps = 1000
thin_by = 10 # apply a thinning factor while running, it means that emcee will run for n_steps x thin_by, but it will keep/return only n_steps values

load_emcee = True

# define the move / sampling step
# default is the Affine-Invariant Ensemble MCMC (A.I.E.M.):
# emcee_move = [
#     (emcee.moves.StretchMove(), 1.0),
# ]
#
# for complex problems the emcee's author suggested sampling with DE80%+DESnooker20%
emcee_move = [
    (emcee.moves.DEMove(), 0.8),
    (emcee.moves.DESnookerMove(), 0.2),
]

# just for progress bar
pka = {
    'ncols': 75,
    'dynamic_ncols': False,
    'position': 0
}

# %% [markdown]
# #### run of `emcee`

# %%
backend_filename_repar = os.path.join(os.path.abspath("./"), "sampler_repar.hdf5")

if load_emcee:
    sampler_repar = emcee.backends.HDFBackend(backend_filename_repar, read_only=True)
else:
    backend_repar = emcee.backends.HDFBackend(backend_filename_repar, compression="gzip")
    backend_repar.reset(n_walkers, n_fit_repar)
    
    with Pool(n_threads) as pool:
        sampler_repar = emcee.EnsembleSampler(
            n_walkers,
            n_fit_repar,
            log_probability_repar,
            pool=pool,
            moves=emcee_move,
            backend=backend_repar
        )
    
        sampler_repar.run_mcmc(
            de_pop_repar[-1, :, :], # last population of pyde
            n_steps,
            thin_by=thin_by,
            store=True,
            tune=True,
            skip_initial_state_check=False,
            progress=True,
            progress_kwargs=pka
        )

# %% [markdown]
# #### statistics of `emcee` run

# %%
n_burnin = 400 # remove first n_burnin steps

# %%
full_chains_repar = sampler_repar.get_chain()
full_chains_flat_repar = sampler_repar.get_chain(flat=True)
post_chains_repar = sampler_repar.get_chain(discard=n_burnin)
post_chains_flat_repar = sampler_repar.get_chain(discard=n_burnin, flat=True)
full_lnprob_repar = sampler_repar.get_log_prob()
full_lnprob_flat_repar = sampler_repar.get_log_prob(flat=True)
post_lnprob_repar = sampler_repar.get_log_prob(discard=n_burnin)
post_lnprob_flat_repar = sampler_repar.get_log_prob(discard=n_burnin, flat=True)
n_post, _ = np.shape(post_chains_flat_repar)

# %%
map_idx_repar = np.argmax(post_lnprob_flat_repar)
map_lnprob_repar = post_lnprob_flat_repar[map_idx_repar]
map_pars_repar = post_chains_flat_repar[map_idx_repar, :]

# %%
# credible intervals
perc = np.array([68.27, 95.44, 99.74]) / 100.0
for i in range(n_fit):
    fitn, fitp, fitpost = fit_labels_repar[i], map_pars_repar[i], post_chains_flat_repar[:, i]
    credint = [anc.hpd(fitpost, c) for c in perc]
    l = "{:18s}: MAP {:10.6f} ".format(fitn, fitp)
    for j, pc in enumerate(credint):
        l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
    print(l)
print()

# conversion of parameters

post_Ms_flat_repar = post_chains_flat_repar[:, 0]

post_l10Mb2s_flat_repar = post_chains_flat_repar[:, 1]
post_Mb_Me_repar = 10**(post_l10Mb2s_flat_repar) * post_Ms_flat_repar * cst.Msear
map_Mb_Me_repar = post_Mb_Me_repar[map_idx_repar]

credint = [anc.hpd(post_Mb_Me_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("M_b", map_Mb_Me_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_Mb_Me_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_l10Mc2b_flat_repar = post_chains_flat_repar[:, 2]
post_Mc_Me_repar = 10**(post_l10Mc2b_flat_repar) * post_Mb_Me_repar
map_Mc_Me_repar = post_Mc_Me_repar[map_idx_repar]

credint = [anc.hpd(post_Mc_Me_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("M_c", map_Mc_Me_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_Mc_Me_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

icw = fit_labels_repar.index("secosw_b")
isw = icw + 1
post_secw_b_flat_repar = post_chains_flat_repar[:, icw]
post_sesw_b_flat_repar = post_chains_flat_repar[:, isw]
post_ecc_b_flat_repar = post_secw_b_flat_repar**2 + post_sesw_b_flat_repar**2
map_ecc_b_repar = post_ecc_b_flat_repar[map_idx_repar]

credint = [anc.hpd(post_ecc_b_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("e_b", map_ecc_b_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_b_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_argp_b_flat_repar = np.arctan2(post_sesw_b_flat_repar, post_secw_b_flat_repar)*cst.rad2deg
map_argp_b_repar = post_argp_b_flat_repar[map_idx_repar]

credint = [anc.hpd(post_argp_b_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("w_b", map_argp_b_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_b_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution


icw = fit_labels_repar.index("secosw_c")
isw = icw + 1
post_secw_c_flat_repar = post_chains_flat_repar[:, icw]
post_sesw_c_flat_repar = post_chains_flat_repar[:, isw]
post_ecc_c_flat_repar = post_secw_c_flat_repar**2 + post_sesw_c_flat_repar**2
map_ecc_c_repar = post_ecc_c_flat_repar[map_idx_repar]

credint = [anc.hpd(post_ecc_c_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("e_c", map_ecc_c_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_c_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_argp_c_flat_repar = np.arctan2(post_sesw_c_flat_repar, post_secw_c_flat_repar)*cst.rad2deg
map_argp_c_repar = post_argp_c_flat_repar[map_idx_repar]

credint = [anc.hpd(post_argp_c_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("w_c", map_argp_c_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_c_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

print()
print("This example:")
print("M_b = {:10.6f} +/- {:10.6f} Mearth".format(map_Mb_Me_repar, err_Mb_Me_repar))
print("M_c = {:10.6f} +/- {:10.6f} Mearth".format(map_Mc_Me_repar, err_Mc_Me_repar))

print("McKee values:")
print("M_b = {:10.6f} +/- {:10.6f} Mearth".format(0.0554*cst.Mjups*cst.Msear, 0.0020*cst.Mjups*cst.Msear))
print("M_c = {:10.6f} +/- {:10.6f} Mearth".format(0.525*cst.Mjups*cst.Msear, 0.019*cst.Mjups*cst.Msear))

# %% [markdown]
# Perfectly consistent!

# %% [markdown]
# #### trace plot

# %%
# log-Probability
# same but using `trades` function
log_probability_trace(
    full_lnprob_repar,
    post_lnprob_flat_repar,
    None,
    n_burn=n_burnin,
    n_thin=thin_by,
    show_plot=True,
    figsize=(5, 5),
    olog=None,
)

# %%
skip_this = False

if not skip_this:
    # let's use `trades` function to plot all the chains/trace plot of each parameter
    exp_acf_fit, exp_steps_fit = full_statistics(
        full_chains_repar,
        post_chains_flat_repar,
        fit_labels_repar,
        map_pars_repar,
        post_lnprob_flat_repar,
        None,
        olog=None,
        ilast=0,
        n_burn=n_burnin,
        n_thin=thin_by,
        show_plot=True,
        figsize=(5, 5),
    )

# %% [markdown]
# #### corner plot

# %%
ticklabel_size = 4
label_separation = -1.1
label_size = 6
k = anc.get_auto_bins(post_chains_flat_repar)

GTC = pygtc.plotGTC(
    chains=post_chains_flat_repar,
    paramNames=fit_labels_repar,
    nContourLevels=3,
    nBins=k,
    truths=map_pars_repar,
    truthLabels=("MAP"),
    figureSize=plt.rcParams["figure.figsize"][0],
    mathTextFontSet=plt.rcParams["mathtext.fontset"],
    customLabelFont={"family": plt.rcParams["font.family"], "size": label_size},
    customTickFont={"family": plt.rcParams["font.family"], "size": ticklabel_size},
    customLegendFont={"family": plt.rcParams["font.family"], "size": label_size},
    legendMarker='All',
    labelRotation=(True, False),
)
axs = GTC.axes
for ax in axs:
    ax.tick_params(
        direction='inout',
        pad=4,
        size=3,
        labelsize=ticklabel_size
    )
    lb = ax.get_xlabel()
    if lb != "":
        ax.xaxis.set_label_coords(0.5, label_separation+0.1)
        ax.set_xlabel(lb, fontsize=label_size, rotation=45.0)
    
    lb = ax.get_ylabel()
    if lb != "":
        ax.yaxis.set_label_coords(label_separation-0.5, 0.5)
        ax.set_ylabel(lb, fontsize=label_size, rotation=45.0)

    for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.6)

plt.show()
plt.close(GTC)

# %% [markdown]
# #### O-C plot w/ samples  
# Let's plot the O-C plot with the samples with shades at different credible intervals.

# %%
map_transits = fitting_to_observables_repar_dict(map_pars_repar)

# %%
(
    mass_map, 
    radius_map,
    period_map, 
    ecc_map, 
    argp_map, 
    meana_map,
    inc_map,
    longn_map,
) = fitting_to_physical_params_repar(map_pars_repar)
print("m ", *mass_map  , "(", *mass, ")")
print("r ", *radius_map, "(", *radius, ")")
print("p ", *period_map, "(", *period, ")")
print("e ", *ecc_map   , "(", *ecc, ")")
print("w ", *argp_map  , "(", *argp, ")")
print("ma", *meana_map , "(", *meana, ")")
print("i ", *inc_map   , "(", *inc, ")")
print("ln", *longn_map , "(", *longn, ")")

# %%
t_int_syn = time_sel_end

_, _, _, map_transits_full = run_and_get_transits(
    t_epoch, 
    t_start, 
    t_int_syn,
    mass_map,
    radius_map,
    period_map, 
    ecc_map,
    argp_map, 
    meana_map,
    inc_map,
    longn_map,
    body_names[1:],
)

# %%
n_samples = 33
smp_tra = {}
smp_idx = np.random.choice(n_post, n_samples, replace=False)
for ismp in smp_idx:
    smp_pars = post_chains_flat_repar[ismp, :]
    (
        mass_smp, 
        radius_smp,
        period_smp, 
        ecc_smp, 
        argp_smp, 
        meana_smp,
        inc_smp,
        longn_smp,
    ) = fitting_to_physical_params_repar(smp_pars)
    _, _, _, smp_transits_full = run_and_get_transits(
        t_epoch, 
        t_start, 
        t_int_syn,
        mass_smp,
        radius_smp,
        period_smp, 
        ecc_smp,
        argp_smp, 
        meana_smp,
        inc_smp,
        longn_smp,
        body_names[1:],
    )
    smp_tra[ismp] = smp_transits_full

# %%
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(hspace=0.07, wspace=0.25)

c1, c2, c3 = 0.6827, 0.9544, 0.9974
hc1, hc2, hc3 = c1*0.5, c2*0.5, c3*0.5

lfont = 8
tfont = 6

zo_map = 10
zo_obs = zo_map-1
zo_mod = zo_obs -1
zo_1s = zo_mod - 1
zo_2s = zo_1s - 1
zo_3s = zo_2s - 1

cfsm = plt.get_cmap("gray")
gval = 0.6
dg = 0.1

axs = []
nrows = 6 # (2 + 1) * 2
ncols = 1

u = [1.0, "days"]
markers = anc.filled_markers

all_xlims = []

# =================================================================
i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))

ax = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=2)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("O-C ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]
_, _, chi2_b, epoch_b, Tlin_b, oc_b= linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, Tref_in = Tref_b, Pref_in = Pref_b, fit=False
)
print("Tref_b: {:12.6f} Pref_b: {:12.6f}".format(Tref_b[0], Pref_b[0]))

tra_map_b = map_transits[pl_letter]["transit_times"]
epo_map_b = compute_epoch(Tref_b[0], Pref_b[0], tra_map_b)
tln_map_b = Tref_b[0] + epo_map_b*Pref_b[0]
oc_map_b = tra_map_b - tln_map_b
res_map_b = tra_syn_noisy_b - tra_map_b

tra_map_full_b = map_transits_full[pl_letter]["transit_times"]
epo_map_full_b = compute_epoch(Tref_b[0], Pref_b[0], tra_map_full_b)
tln_map_full_b = Tref_b[0] + epo_map_full_b*Pref_b[0]
oc_map_full_b = tra_map_full_b - tln_map_full_b

ax.errorbar(
    tra_syn_noisy_b,
    oc_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2.5,
    mec='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

ax.plot(
    tra_syn_noisy_b,
    oc_map_b*u[0],
    color="black",
    marker=markers[0],
    ms=2.5,
    mfc='None',
    mew=0.4,
    label="{} (map)".format(pl_letter),
    ls='',
    zorder=zo_map
)

ax.plot(
    tra_map_full_b,
    oc_map_full_b*u[0],
    color="black",
    marker='o',
    ms=0.6,
    ls='-',
    lw=0.3,
    label="{} (full map)".format(pl_letter),
    zorder=zo_mod
)

oc_smp = []
for ksmp, vsmp in smp_tra.items():
    tra_xxx = vsmp[pl_letter]["transit_times"]
    epo_xxx = compute_epoch(Tref_b[0], Pref_b[0], tra_xxx)
    tln_xxx = Tref_b[0] + epo_xxx*Pref_b[0]
    oc_xxx = tra_xxx - tln_xxx
    oc_smp.append(oc_xxx)
oc_smp = np.array(oc_smp).T * u[0]
hdi1 = np.percentile(oc_smp, [50 - (100*hc1), 50 + (100*hc1)], axis=1).T
hdi2 = np.percentile(oc_smp, [50 - (100*hc2), 50 + (100*hc2)], axis=1).T
hdi3 = np.percentile(oc_smp, [50 - (100*hc3), 50 + (100*hc3)], axis=1).T
ax.fill_between(
    tra_map_full_b,
    hdi1[:, 0],
    hdi1[:, 1],
    color=cfsm(gval),
    alpha=1.0,
    lw=0.0,
    zorder=zo_1s,
)
ax.fill_between(
    tra_map_full_b,
    hdi2[:, 0],
    hdi2[:, 1],
    color=cfsm(gval+dg),
    alpha=1.0,
    lw=0.0,
    zorder=zo_2s,
)
ax.fill_between(
    tra_map_full_b,
    hdi3[:, 0],
    hdi3[:, 1],
    color=cfsm(gval+(dg*2)),
    alpha=1.0,
    lw=0.0,
    zorder=zo_3s,
)

all_xlims.append(ax.get_xlim())

axs.append(ax)

# ---
ax = plt.subplot2grid((nrows, ncols), (2, 0), rowspan=1)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("res. ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

ax.errorbar(
    tra_syn_noisy_b,
    res_map_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2.5,
    mec='black',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    # label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

axs.append(ax)

# =================================================================
i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = plt.subplot2grid((nrows, ncols), (3, 0), rowspan=2)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("O-C ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]
_, _, chi2_c, epoch_c, Tlin_c, oc_c= linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, Tref_in = Tref_c, Pref_in = Pref_c, fit=False
)
print("Tref_c: {:12.6f} Pref_c: {:12.6f}".format(Tref_c[0], Pref_c[0]))

tra_map_c = map_transits[pl_letter]["transit_times"]
epo_map_c = compute_epoch(Tref_c[0], Pref_c[0], tra_map_c)
tln_map_c = Tref_c[0] + epo_map_c*Pref_c[0]
oc_map_c = tra_map_c - tln_map_c
res_map_c = tra_syn_noisy_c - tra_map_c

tra_map_full_c = map_transits_full[pl_letter]["transit_times"]
epo_map_full_c = compute_epoch(Tref_c[0], Pref_c[0], tra_map_full_c)
tln_map_full_c = Tref_c[0] + epo_map_full_c*Pref_c[0]
oc_map_full_c = tra_map_full_c - tln_map_full_c

ax.errorbar(
    tra_syn_noisy_c,
    oc_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2.5,
    mec='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

ax.plot(
    tra_syn_noisy_c,
    oc_map_c*u[0],
    color="black",
    marker=markers[1],
    ms=2.5,
    mfc='None',
    mew=0.4,
    label="{} (map)".format(pl_letter),
    ls='',
    zorder=zo_map
)

ax.plot(
    tra_map_full_c,
    oc_map_full_c*u[0],
    color="black",
    marker='o',
    ms=0.6,
    ls='-',
    lw=0.3,
    label="{} (full map)".format(pl_letter),
    zorder=zo_mod
)

oc_smp = []
for ksmp, vsmp in smp_tra.items():
    tra_xxx = vsmp[pl_letter]["transit_times"]
    epo_xxx = compute_epoch(Tref_c[0], Pref_c[0], tra_xxx)
    tln_xxx = Tref_c[0] + epo_xxx*Pref_c[0]
    oc_xxx = tra_xxx - tln_xxx
    oc_smp.append(oc_xxx)
oc_smp = np.array(oc_smp).T * u[0]
hdi1 = np.percentile(oc_smp, [50 - (100*hc1), 50 + (100*hc1)], axis=1).T
hdi2 = np.percentile(oc_smp, [50 - (100*hc2), 50 + (100*hc2)], axis=1).T
hdi3 = np.percentile(oc_smp, [50 - (100*hc3), 50 + (100*hc3)], axis=1).T
ax.fill_between(
    tra_map_full_c,
    hdi1[:, 0],
    hdi1[:, 1],
    color=cfsm(gval),
    alpha=1.0,
    lw=0.0,
    zorder=zo_1s,
)
ax.fill_between(
    tra_map_full_c,
    hdi2[:, 0],
    hdi2[:, 1],
    color=cfsm(gval+dg),
    alpha=1.0,
    lw=0.0,
    zorder=zo_2s,
)
ax.fill_between(
    tra_map_full_c,
    hdi3[:, 0],
    hdi3[:, 1],
    color=cfsm(gval+(dg*2)),
    alpha=1.0,
    lw=0.0,
    zorder=zo_3s,
)

all_xlims.append(ax.get_xlim())

axs.append(ax)

# ---
ax = plt.subplot2grid((nrows, ncols), (5, 0), rowspan=1)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=True)
ax.set_ylabel("res. ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

ax.errorbar(
    tra_syn_noisy_c,
    res_map_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2.5,
    mec='black',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    ls='',
    zorder=zo_obs
)

axs.append(ax)

all_xlims = np.concatenate(all_xlims)
for ax in axs:
    ax.set_xlim(np.min(all_xlims), np.max(all_xlims))
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=lfont, frameon=False)

ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# ### Check $P_\mathrm{b}$ hole in the posterior  
# The posterior of $P_\mathrm{b}$ shows a hole in the distribution around 17.099 and 17.100.  
# We would investigate this behavior, to determine the source of this issue.

# %%
# Define a number of tests (simulations)
n_test = 5

test_fit = np.copy(map_pars_repar)
# Define the range to investigate
prange = [17.09938, 17.09957]
# Create a vector of values to test
pvalues = np.linspace(prange[0], prange[1], n_test, endpoint=True)

stable_tests = np.zeros((n_test))

test_orbits = []
output = []

for i in range(n_test):
    idx = fit_labels_repar.index("P_b")
    test_fit[idx] = pvalues[i]
    print("\n====Test P_b = ", test_fit[idx], " (Pc/Pb = ",test_fit[idx+1]/test_fit[idx], ")")
    (
        mass_test, 
        radius_test,
        period_test, 
        ecc_test, 
        argp_test, 
        meana_test,
        inc_test,
        longn_test,
    ) = fitting_to_physical_params_repar(test_fit)
    print("mass = ", mass_test)
    # The next lines are to test the effect of the mass on the transit events
    # mass_test[1:] /= 2.0
    # print("mass = ", mass_test)
    # test_fit[1] = np.log10(mass_test[1]/mass_test[0])
    # test_fit[2] = np.log10(mass_test[2]/mass_test[1])
    
    print("period = ", period_test)

    # Split the log-probability to check if there is a problem in the boundaries, priors, log-likelihood and posterior
        
    check_fit = check_fitting_boundaries_repar(test_fit)
    print("Check fitting boundaries = ", check_fit)
    
    lnbd = log_boundaries_repar(test_fit)
    print("Log boundaries = ", lnbd)
    
    lnpr = log_priors_repar(test_fit)
    print("Log priors = ", lnpr)
    
    lnlo = log_likelihood_repar(test_fit)
    print("Log likelihood = ", lnlo)
    
    ln_pr = lnbd + lnpr + lnlo
    print("Log posterior = ", ln_pr)
    
    (
        _,
        _,
        _,
        _,
        _,
        _,
        stable,
    ) = fitting_to_observables_repar(test_fit)
    print("Stable? ", bool(stable))
    stable_tests[i] = stable
    
    test_orbits.append([mass_test, radius_test, period_test, ecc_test, argp_test, meana_test, inc_test, longn_test])
    
    tra_test = fitting_to_observables_repar_dict(test_fit)
    output.append(tra_test)

print("\n===== {:.2f}% tests are stable".format(100*np.sum(stable_tests)/n_test))

# %%
# Let's plot the residuals between the synthetic/observed transit times and the test ones

lfont = 6
tfont = 4

zo_map = 10
zo_obs = zo_map-1
zo_mod = zo_obs -1
zo_1s = zo_mod - 1
zo_2s = zo_1s - 1
zo_3s = zo_2s - 1

cfsm = plt.get_cmap("gray")
gval = 0.6
dg = 0.1

nrows = n_test # (2 + 1) * 2
ncols = 1

u = [1.0, "days"]
markers = anc.filled_markers

lbdown = [False] * (nrows-1) + [True]

nspan=1

# =================================================================
i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))
axs = []
all_xlims = []

fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(hspace=0.07, wspace=0.25)


(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]

all_tb = tra_syn_noisy_b

for i, val in enumerate(output):
    ax = plt.subplot2grid((nrows, ncols), (i, 0), rowspan=nspan)
    poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=lbdown[i])
    ax.set_ylabel("O-C ({:s}) - {} -".format(u[1], pl_letter), fontsize=lfont)
    ax.axhline(0, color="k", lw=0.8)

    tra_xxx = val[pl_letter]["transit_times"]
    all_tb = np.column_stack((all_tb, tra_xxx))
    epo_xxx = compute_epoch(Tref_b[0], Pref_b[0], tra_xxx)
    tln_xxx = Tref_b[0] + epo_xxx*Pref_b[0]
    oc_xxx = tra_xxx - tln_xxx
    res = tra_syn_noisy_b - tra_xxx
    ax.errorbar(
        tra_syn_noisy_b,
        res,
        yerr=err_tra_syn_b,
        marker=markers[0],
        ms=2,
        mec='None',
        ls='',
        elinewidth=0.5,
        capsize=0,
        label="P = {:.6f} days".format(pvalues[i])
    )
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=lfont, frameon=False)

all_xlims.append(ax.get_xlim())

axs.append(ax)

all_xlims = np.concatenate(all_xlims)
for ax in axs:
    ax.set_xlim(np.min(all_xlims), np.max(all_xlims))

axs[-1].set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)




all_tc = tra_syn_noisy_c
# =================================================================
i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
axs = []
all_xlims = []

fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(hspace=0.07, wspace=0.25)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]

for i, val in enumerate(output):
    ax = plt.subplot2grid((nrows, ncols), (i, 0), rowspan=nspan)
    poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=lbdown[i])
    ax.set_ylabel("O-C ({:s}) - {} -".format(u[1], pl_letter), fontsize=lfont)
    ax.axhline(0, color="k", lw=0.8)
    
    tra_xxx = val[pl_letter]["transit_times"]
    all_tc = np.column_stack((all_tc, tra_xxx))
    epo_xxx = compute_epoch(Tref_c[0], Pref_c[0], tra_xxx)
    tln_xxx = Tref_c[0] + epo_xxx*Pref_c[0]
    oc_xxx = tra_xxx - tln_xxx
    res = tra_syn_noisy_c - tra_xxx
    ax.errorbar(
        tra_syn_noisy_c,
        res,
        yerr=err_tra_syn_c,
        marker=markers[1],
        ms=2,
        mec='None',
        ls='',
        elinewidth=0.5,
        capsize=0,
        label="P = {:.6f} days".format(pvalues[i])
    )
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=lfont, frameon=False)

all_xlims.append(ax.get_xlim())

axs.append(ax)

all_xlims = np.concatenate(all_xlims)
for ax in axs:
    ax.set_xlim(np.min(all_xlims), np.max(all_xlims))

axs[-1].set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %% [markdown]
# Planet b shows a suddend value change of about 1000 days for the same transit time for three values of the period.  
# From previous loop, we did not find any issue in stability, boundaries, but the log-likelihood assumes very low value for those three configurations.  
# Let's check the transit times compute for all the tests (we already store the synthetic transit times and the test ones).

# %%
print("i synthetic ", *["test_{:d}".format(i) for i in range(n_test)])
for it, t0s in enumerate(all_tb):
    print(it, *t0s)

# %% [markdown]
# The tests with id 1, 2, and 3 have zero values for the transit with index 5.
# Given that the configurations are stable, the parameters are within the fitting and physical boundaries,
# a possible explanation is that the planet b is not transiting in those particular configurations.  
# We can compute the orbits of each configuration and plot the orbits only for time range span by the synthetic observations.  

# %%
t0s_all = np.concatenate([tra_syn_noisy_b, tra_syn_noisy_c])
tmin, tmax = t0s_all.min(), t0s_all.max()

for i, (mass_test, radius_test, period_test, ecc_test, argp_test, meana_test, inc_test, longn_test) in enumerate(test_orbits):
    print("\n\n ==== i = {} ====".format(i))
    time_xxx, orbits_xxx, stable_xxx = pytrades.kelements_to_orbits_full(
        t_epoch,
        t_start,
        t_int,
        mass_test, radius_test, period_test, ecc_test, argp_test, meana_test, inc_test, longn_test,
        specific_times=None,
        step_size=1.0,
        n_steps_smaller_orbits=None,
    )
    selected_orbits = np.logical_and(time_xxx >= tmin, time_xxx <= tmax)
    fig= pytrades.base_plot_orbits(
        time_xxx[selected_orbits],
        orbits_xxx[selected_orbits],
        radius_test,
        n_body,
        body_names,
        figsize=(4, 4),
        sky_scale='star',
        side_scale= 'pos',
        title="Pb = {:.5f} days".format(period_test[1]),
        show_plot=True,
    )
    plt.close(fig)

# %% [markdown]
# A quick view of the orbits and different projections would suggest that planet b in some specific configurations,  
# as for the values of the period of the range 17.099 and 17.100 days, is not transiting.  
# Hence, the code is returning a zero value for that transit.  

# %% [markdown]
# ## TTV analysis with Nested Sampling  

# %% [markdown]
# ### define priors  
# In this case, we will use the reparametization of masses, eccentricity, argument of pericenter and mean longitude.  
# However, we will re-define the priors in a slighty different way.

# %%
priors_nautilus = nautilus.Prior()
for ilab, lab in enumerate(fit_labels_repar):
    priors_nautilus.add_parameter(lab, dist=priors_repar[ilab])

boundaries_physical = [
    M_s_boundaries,
    M_p_boundaries,
    ecc_boundaries,
]

priors_physical = nautilus.Prior()
priors_physical.add_parameter("M_star", dist=uniform(loc=M_s_boundaries[0], scale=np.ptp(M_s_boundaries)))
priors_physical.add_parameter("M_b", dist=uniform(loc=M_p_boundaries[0], scale=np.ptp(M_p_boundaries)))
priors_physical.add_parameter("M_c", dist=uniform(loc=M_p_boundaries[0], scale=np.ptp(M_p_boundaries)))
priors_physical.add_parameter("ecc_b", dist=uniform(loc=ecc_boundaries[0], scale=np.ptp(ecc_boundaries)))
priors_physical.add_parameter("ecc_c", dist=uniform(loc=ecc_boundaries[0], scale=np.ptp(ecc_boundaries)))

# %% [markdown]
# #### define functions to convert:  
# - computes priors, if needed  
# - computes the log-likelihood
# - computes the log-prior, if needed  

# %% [markdown]
# MCMC algorithms and MAP optimization are concerned with the shape of the posterior distribution and finding its mode(s), 
# not its absolute normalization.  
# When you work with log-probabilities, 
# adding a constant term to the log-posterior does not change the location of the maximum or 
# the relative probabilities between different points in parameter space.  
# So, in case of emcee the uniform priors (ln P(\theta) = ln ( 1 / (max - min) )) can be discarded,
# and only Normal-Gaussian (or other types) prios can be computed in log-Probability.  
# It is mandatory to compute the log-prior for uniform priors in case of
# Nested Sampling and/or Model selection (Bayes factor) analysis

# %%
def log_boundaries_nautilus_repar(fit_dict):

    fit_pars = np.array(list(fit_dict.values()))
    
    m_x, r_x, p_x, e_x, w_x, ma_x, i_x, ln_x = fitting_to_physical_params_repar(fit_pars)
    phys = {
        "M_star":m_x[0],
        "M_b":m_x[1],
        "M_c":m_x[2],
        "ecc_b":e_x[1],
        "ecc_c":e_x[2],
    }

    for k, v in phys.items():
        lnphy = priors_physical.dists[priors_physical.keys.index(k)].logpdf(v)
        if np.isinf(lnphy):
            return -np.inf
    return 0.0

def log_priors_nautulis_repar(fit_dict): # this can be extended for uniform priors when using Nested Sampling

    ln_prior = np.sum(
        [
            priors_nautilus.dists[priors_nautilus.keys.index(k)].logpdf(p) for k, p in fit_dict.items()
        ]
    )
    return ln_prior
                      
def log_likelihood_nautilus_repar(fit_dict):

    ln_prior = log_boundaries_nautilus_repar(fit_dict)
    if np.isinf(ln_prior):
        return ln_prior
    
    lnL = ln_const + ln_prior
    fit_pars = np.array(list(fit_dict.values()))
    (
        body_flag_sim,
        epo_sim,
        transits_sim,
        durations_sim,
        lambda_rm_sim,
        kep_elem_sim,
        stable,
    ) = fitting_to_observables_repar(fit_pars)
    if not stable:
        return -np.inf

    res_b = tra_syn_noisy_b - transits_sim[body_flag_sim ==2]
    wres_b = res_b / err_tra_syn_b
    lnL_b = -0.5*np.sum(np.log(err_tra_syn_b)) - 0.5*np.sum(wres_b*wres_b)
    
    res_c = tra_syn_noisy_c - transits_sim[body_flag_sim ==3]
    wres_c = res_c / err_tra_syn_c
    lnL_c = -0.5*np.sum(np.log(err_tra_syn_c)) - 0.5*np.sum(wres_c*wres_c)
    
    lnL += lnL_b + lnL_c
    
    return lnL

# %%
fit_dict_pars = {k:p for k, p in zip(fit_labels_repar, fit_pars_repar_test)}
m_fit, r_fit, p_fit, e_fit, w_fit, ma_fit, i_fit, ln_fit = fitting_to_physical_params_repar(fit_pars_repar_test)
print("mass (Msun)    = ",*m_fit)
print("radius (Rsun)  = ",*r_fit)
print("period (days)  = ",*p_fit)
print("ecc            = ",*e_fit)
print("arg. peri. (°) = ",*w_fit)
print("mean anom. (°) = ",*ma_fit)
print("inc (°)        = ",*i_fit)
print("long. node (°) = ",*ln_fit)
ln_prior = log_priors_nautulis_repar(fit_dict_pars)
print("ln-priors = ",ln_prior)
ln_bound = log_boundaries_nautilus_repar(fit_dict_pars)
print("ln-bounds = ", ln_bound)
lnL = log_likelihood_nautilus_repar(fit_dict_pars)
print("logL = ",lnL)
lnP = lnL + ln_prior + ln_bound
print("lnP = ",lnP)
(
    body_flag_test,
    epo_test,
    transits_test,
    _,
    _,
    _,
    stable_test,
    ) = fitting_to_observables_repar(fit_pars_repar_test)
print("Is it stable? {}".format(bool(stable_test)))

# %% [markdown]
# ### `nautilus`

# %%
n_live = 3000
n_threads = 100
seed = 42578
delete_file = True
resume = True
nautilus_filename = os.path.join(os.path.abspath("."), "sampler_nautilus.hdf5")
if delete_file:
    if os.path.exists(nautilus_filename):
        os.remove(nautilus_filename)

# %%
sampler_nautilus = nautilus.Sampler(
    # log_priors_nautulis_repar,
    priors_nautilus,
    log_likelihood_nautilus_repar,
    # likelihood_args=[fit_boundaries_repar, boundaries_physical],
    n_dim=n_fit_repar,
    n_live=n_live,
    vectorized=False,
    pass_dict=True,
    pool=n_threads,
    seed=seed,
    filepath=nautilus_filename,
    resume=resume,
)

# %%
sampler_nautilus.run(discard_exploration=True, verbose=True)

# %%
log_z = sampler_nautilus.log_z
n_eff = sampler_nautilus.n_eff
points, log_w, log_l = sampler_nautilus.posterior()

# %%
print("log_Z = {:.4f} +/- {:.4f}".format(log_z, 1.0/np.sqrt(n_eff)))
print("N_eff = {:.0f}".format(n_eff))

# %%
def compute_log_prior_nautilus(pars, labels):
    pars_dict = {lab:val for lab, val in zip(labels, pars)}
    ln_prior = log_priors_nautulis_repar(pars_dict)
    return ln_prior

n_calls, _ = np.shape(points)
log_priors_nau, log_prob_nau = np.zeros(n_calls), np.zeros(n_calls)
for icall in range(n_calls):
    log_priors_nau[icall] = compute_log_prior_nautilus(points[icall], fit_labels_repar)
    log_prob_nau[icall] = log_l[icall] + log_priors_nau[icall]

# %%
weights = np.exp(log_w)
sel_w = weights > 0.0
pos_points = points[sel_w, :]
pos_weights = weights[sel_w]
pos_log_like = log_l[sel_w]
pos_log_prior = log_priors_nau[sel_w]
pos_log_prob = log_prob_nau[sel_w]
n_nautilus = len(pos_weights)
print("posterior size n_nautilus = ", n_nautilus)

# %%
idx_mle_nautilus = np.argmax(pos_log_like)
mle_points = pos_points[idx_mle_nautilus, :]
mle_fit_dict = {lab:val for lab, val in zip(fit_labels_repar, mle_points)}

# %%
idx_map_nautilus = np.argmax(pos_log_prob)
map_points = pos_points[idx_map_nautilus, :]
map_fit_dict = {lab:val for lab, val in zip(fit_labels_repar, map_points)}

# %%
for n, mle in mle_fit_dict.items():
    print(n, mle, map_fit_dict[n])
print("lnL", pos_log_like[idx_mle_nautilus], pos_log_like[idx_map_nautilus])
print("lnP", pos_log_prob[idx_mle_nautilus], pos_log_prob[idx_map_nautilus])

# %% [markdown]
# #### Statistics of `nautilus` posterior

# %%
# credible intervals
perc = np.array([68.27, 95.44, 99.74]) / 100.0
for i in range(n_fit):
    fitn, fitp, fitpost = fit_labels_repar[i], map_points[i], pos_points[:, i]
    credint = [anc.hpd(fitpost, c) for c in perc]
    l = "{:18s}: MAP {:10.6f} ".format(fitn, fitp)
    for j, pc in enumerate(credint):
        l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
    print(l)
print()

# conversion of parameters

post_Ms_flat_repar = pos_points[:, 0]

post_l10Mb2s_flat_repar = pos_points[:, 1]
post_Mb_Me_repar = 10**(post_l10Mb2s_flat_repar) * post_Ms_flat_repar * cst.Msear
map_Mb_Me_repar = post_Mb_Me_repar[idx_map_nautilus]

credint = [anc.hpd(post_Mb_Me_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("M_b", map_Mb_Me_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_Mb_Me_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_l10Mc2b_flat_repar = pos_points[:, 2]
post_Mc_Me_repar = 10**(post_l10Mc2b_flat_repar) * post_Mb_Me_repar
map_Mc_Me_repar = post_Mc_Me_repar[idx_map_nautilus]

credint = [anc.hpd(post_Mc_Me_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("M_c", map_Mc_Me_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_Mc_Me_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

icw = fit_labels_repar.index("secosw_b")
isw = icw + 1
post_secw_b_flat_repar = pos_points[:, icw]
post_sesw_b_flat_repar = pos_points[:, isw]
post_ecc_b_flat_repar = post_secw_b_flat_repar**2 + post_sesw_b_flat_repar**2
map_ecc_b_repar = post_ecc_b_flat_repar[idx_map_nautilus]

credint = [anc.hpd(post_ecc_b_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("e_b", map_ecc_b_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_b_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_argp_b_flat_repar = np.arctan2(post_sesw_b_flat_repar, post_secw_b_flat_repar)*cst.rad2deg
map_argp_b_repar = post_argp_b_flat_repar[idx_map_nautilus]

credint = [anc.hpd(post_argp_b_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("w_b", map_argp_b_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_b_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution


icw = fit_labels_repar.index("secosw_c")
isw = icw + 1
post_secw_c_flat_repar = pos_points[:, icw]
post_sesw_c_flat_repar = pos_points[:, isw]
post_ecc_c_flat_repar = post_secw_c_flat_repar**2 + post_sesw_c_flat_repar**2
map_ecc_c_repar = post_ecc_c_flat_repar[idx_map_nautilus]

credint = [anc.hpd(post_ecc_c_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("e_c", map_ecc_c_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_c_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

post_argp_c_flat_repar = np.arctan2(post_sesw_c_flat_repar, post_secw_c_flat_repar)*cst.rad2deg
map_argp_c_repar = post_argp_c_flat_repar[idx_map_nautilus]

credint = [anc.hpd(post_argp_c_flat_repar, c) for c in perc]
l = "{:18s}: MAP {:10.6f} ".format("w_c", map_argp_c_repar)
for j, pc in enumerate(credint):
    l += "HDI@{:.2f}% [{:10.6f} , {:10.6f}] ".format(perc[j]*100, pc[0], pc[1])
print(l)
err_ecc_c_repar = np.ptp(credint[0])*0.5 # semi-interval, but it is not the only solution

print()
print("This example:")
print("M_b = {:10.6f} +/- {:10.6f} Mearth".format(map_Mb_Me_repar, err_Mb_Me_repar))
print("M_c = {:10.6f} +/- {:10.6f} Mearth".format(map_Mc_Me_repar, err_Mc_Me_repar))

print("McKee values:")
print("M_b = {:10.6f} +/- {:10.6f} Mearth".format(0.0554*cst.Mjups*cst.Msear, 0.0020*cst.Mjups*cst.Msear))
print("M_c = {:10.6f} +/- {:10.6f} Mearth".format(0.525*cst.Mjups*cst.Msear, 0.019*cst.Mjups*cst.Msear))

# %% [markdown]
# #### corner plot

# %%
ticklabel_size = 3
label_separation = -1.1
label_size = 4
k = anc.get_auto_bins(pos_points)

GTC = pygtc.plotGTC(
    chains=pos_points,
    paramNames=fit_labels_repar,
    nContourLevels=3,
    sigmaContourLevels=True,
    nBins=k,
    truths=(mle_points, map_points),
    truthLabels=("MLE", "MAP"),
    figureSize=plt.rcParams["figure.figsize"][0],
    mathTextFontSet=plt.rcParams["mathtext.fontset"],
    customLabelFont={"family": plt.rcParams["font.family"], "size": label_size},
    customTickFont={"family": plt.rcParams["font.family"], "size": ticklabel_size},
    customLegendFont={"family": plt.rcParams["font.family"], "size": label_size},
    legendMarker='All',
    labelRotation=(True, False),
)
axs = GTC.axes
for ax in axs:
    ax.tick_params(
        direction='inout',
        pad=4,
        size=3,
        labelsize=ticklabel_size
    )
    lb = ax.get_xlabel()
    if lb != "":
        ax.xaxis.set_label_coords(0.5, label_separation+0.2)
        ax.set_xlabel(lb, fontsize=label_size, rotation=45.0)
    
    lb = ax.get_ylabel()
    if lb != "":
        ax.yaxis.set_label_coords(label_separation, 0.5)
        ax.set_ylabel(lb, fontsize=label_size, rotation=45.0)

    for axis in ['top', 'bottom', 'left', 'right']:
            ax.spines[axis].set_linewidth(0.6)

plt.show()
plt.close(GTC)

# %% [markdown]
# #### O-C plot w/ samples  
# Let's plot the O-C plot with the samples with shades at different credible intervals.

# %%
map_transits_nautilus = fitting_to_observables_repar_dict(map_points)

# %%
(
    mass_map, 
    radius_map,
    period_map, 
    ecc_map, 
    argp_map, 
    meana_map,
    inc_map,
    longn_map,
) = fitting_to_physical_params_repar(map_points)
print("m ", *mass_map  , "(", *mass, ")")
print("r ", *radius_map, "(", *radius, ")")
print("p ", *period_map, "(", *period, ")")
print("e ", *ecc_map   , "(", *ecc, ")")
print("w ", *argp_map  , "(", *argp, ")")
print("ma", *meana_map , "(", *meana, ")")
print("i ", *inc_map   , "(", *inc, ")")
print("ln", *longn_map , "(", *longn, ")")

# %%
t_int_syn = time_sel_end

_, _, _, map_transits_full_nautilus = run_and_get_transits(
    t_epoch, 
    t_start, 
    t_int_syn,
    mass_map,
    radius_map,
    period_map, 
    ecc_map,
    argp_map, 
    meana_map,
    inc_map,
    longn_map,
    body_names[1:],
)

# %%
smp_tra = {}
smp_idx = np.random.choice(n_nautilus, n_samples, replace=False, p=pos_weights)
for ismp in smp_idx:
    smp_pars = pos_points[ismp, :]
    (
        mass_smp, 
        radius_smp,
        period_smp, 
        ecc_smp, 
        argp_smp, 
        meana_smp,
        inc_smp,
        longn_smp,
    ) = fitting_to_physical_params_repar(smp_pars)
    _, _, _, smp_transits_full = run_and_get_transits(
        t_epoch, 
        t_start, 
        t_int_syn,
        mass_smp,
        radius_smp,
        period_smp, 
        ecc_smp,
        argp_smp, 
        meana_smp,
        inc_smp,
        longn_smp,
        body_names[1:],
    )
    smp_tra[ismp] = smp_transits_full

# %%
fig = plt.figure(figsize=(5,5))
fig.subplots_adjust(hspace=0.07, wspace=0.25)

c1, c2, c3 = 0.6827, 0.9544, 0.9974
hc1, hc2, hc3 = c1*0.5, c2*0.5, c3*0.5

lfont = 8
tfont = 6

zo_map = 10
zo_obs = zo_map-1
zo_mod = zo_obs -1
zo_1s = zo_mod - 1
zo_2s = zo_1s - 1
zo_3s = zo_2s - 1

cfsm = plt.get_cmap("gray")
gval = 0.6
dg = 0.1

axs = []
nrows = 6 # (2 + 1) * 2
ncols = 1

u = [1.0, "days"]
markers = anc.filled_markers

all_xlims = []

# =================================================================
i, pl_letter = 0, "b"
print("planet {}".format(pl_letter))

ax = plt.subplot2grid((nrows, ncols), (0, 0), rowspan=2)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("O-C ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

(Tref_b, Pref_b) = lineph_syn_noisy[pl_letter]
_, _, chi2_b, epoch_b, Tlin_b, oc_b= linear_ephemeris(
    tra_syn_noisy_b, eT0s=err_tra_syn_b, Tref_in = Tref_b, Pref_in = Pref_b, fit=False
)
print("Tref_b: {:12.6f} Pref_b: {:12.6f}".format(Tref_b[0], Pref_b[0]))

tra_map_b = map_transits_nautilus[pl_letter]["transit_times"]
epo_map_b = compute_epoch(Tref_b[0], Pref_b[0], tra_map_b)
tln_map_b = Tref_b[0] + epo_map_b*Pref_b[0]
oc_map_b = tra_map_b - tln_map_b
res_map_b = tra_syn_noisy_b - tra_map_b

tra_map_full_b = map_transits_full_nautilus[pl_letter]["transit_times"]
epo_map_full_b = compute_epoch(Tref_b[0], Pref_b[0], tra_map_full_b)
tln_map_full_b = Tref_b[0] + epo_map_full_b*Pref_b[0]
oc_map_full_b = tra_map_full_b - tln_map_full_b

ax.errorbar(
    tra_syn_noisy_b,
    oc_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2.5,
    mec='None',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

ax.plot(
    tra_syn_noisy_b,
    oc_map_b*u[0],
    color="black",
    marker=markers[0],
    ms=2.5,
    mfc='None',
    mew=0.4,
    label="{} (map)".format(pl_letter),
    ls='',
    zorder=zo_map
)

ax.plot(
    tra_map_full_b,
    oc_map_full_b*u[0],
    color="black",
    marker='o',
    ms=0.6,
    ls='-',
    lw=0.3,
    label="{} (full map)".format(pl_letter),
    zorder=zo_mod
)

oc_smp = []
for ksmp, vsmp in smp_tra.items():
    tra_xxx = vsmp[pl_letter]["transit_times"]
    epo_xxx = compute_epoch(Tref_b[0], Pref_b[0], tra_xxx)
    tln_xxx = Tref_b[0] + epo_xxx*Pref_b[0]
    oc_xxx = tra_xxx - tln_xxx
    oc_smp.append(oc_xxx)
oc_smp = np.array(oc_smp).T * u[0]
hdi1 = np.percentile(oc_smp, [50 - (100*hc1), 50 + (100*hc1)], axis=1).T
hdi2 = np.percentile(oc_smp, [50 - (100*hc2), 50 + (100*hc2)], axis=1).T
hdi3 = np.percentile(oc_smp, [50 - (100*hc3), 50 + (100*hc3)], axis=1).T
ax.fill_between(
    tra_map_full_b,
    hdi1[:, 0],
    hdi1[:, 1],
    color=cfsm(gval),
    alpha=1.0,
    lw=0.0,
    zorder=zo_1s,
)
ax.fill_between(
    tra_map_full_b,
    hdi2[:, 0],
    hdi2[:, 1],
    color=cfsm(gval+dg),
    alpha=1.0,
    lw=0.0,
    zorder=zo_2s,
)
ax.fill_between(
    tra_map_full_b,
    hdi3[:, 0],
    hdi3[:, 1],
    color=cfsm(gval+(dg*2)),
    alpha=1.0,
    lw=0.0,
    zorder=zo_3s,
)

all_xlims.append(ax.get_xlim())

axs.append(ax)

# ---
ax = plt.subplot2grid((nrows, ncols), (2, 0), rowspan=1)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("res. ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

ax.errorbar(
    tra_syn_noisy_b,
    res_map_b*u[0],
    yerr=err_tra_syn_b*u[0],
    marker=markers[0],
    ms=2.5,
    mec='black',
    mew=0.4,
    color="C0",
    ecolor="C0",
    elinewidth=0.4,
    capsize=0,
    # label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

axs.append(ax)

# =================================================================
i, pl_letter = 1, "c"
print("planet {}".format(pl_letter))
ax = plt.subplot2grid((nrows, ncols), (3, 0), rowspan=2)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=False)
ax.set_ylabel("O-C ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

(Tref_c, Pref_c) = lineph_syn_noisy[pl_letter]
_, _, chi2_c, epoch_c, Tlin_c, oc_c= linear_ephemeris(
    tra_syn_noisy_c, eT0s=err_tra_syn_c, Tref_in = Tref_c, Pref_in = Pref_c, fit=False
)
print("Tref_c: {:12.6f} Pref_c: {:12.6f}".format(Tref_c[0], Pref_c[0]))

tra_map_c = map_transits_nautilus[pl_letter]["transit_times"]
epo_map_c = compute_epoch(Tref_c[0], Pref_c[0], tra_map_c)
tln_map_c = Tref_c[0] + epo_map_c*Pref_c[0]
oc_map_c = tra_map_c - tln_map_c
res_map_c = tra_syn_noisy_c - tra_map_c

tra_map_full_c = map_transits_full_nautilus[pl_letter]["transit_times"]
epo_map_full_c = compute_epoch(Tref_c[0], Pref_c[0], tra_map_full_c)
tln_map_full_c = Tref_c[0] + epo_map_full_c*Pref_c[0]
oc_map_full_c = tra_map_full_c - tln_map_full_c

ax.errorbar(
    tra_syn_noisy_c,
    oc_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2.5,
    mec='None',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    label="{} (noisy)".format(pl_letter),
    ls='',
    zorder=zo_obs
)

ax.plot(
    tra_syn_noisy_c,
    oc_map_c*u[0],
    color="black",
    marker=markers[1],
    ms=2.5,
    mfc='None',
    mew=0.4,
    label="{} (map)".format(pl_letter),
    ls='',
    zorder=zo_map
)

ax.plot(
    tra_map_full_c,
    oc_map_full_c*u[0],
    color="black",
    marker='o',
    ms=0.6,
    ls='-',
    lw=0.3,
    label="{} (full map)".format(pl_letter),
    zorder=zo_mod
)

oc_smp = []
for ksmp, vsmp in smp_tra.items():
    tra_xxx = vsmp[pl_letter]["transit_times"]
    epo_xxx = compute_epoch(Tref_c[0], Pref_c[0], tra_xxx)
    tln_xxx = Tref_c[0] + epo_xxx*Pref_c[0]
    oc_xxx = tra_xxx - tln_xxx
    oc_smp.append(oc_xxx)
oc_smp = np.array(oc_smp).T * u[0]
hdi1 = np.percentile(oc_smp, [50 - (100*hc1), 50 + (100*hc1)], axis=1).T
hdi2 = np.percentile(oc_smp, [50 - (100*hc2), 50 + (100*hc2)], axis=1).T
hdi3 = np.percentile(oc_smp, [50 - (100*hc3), 50 + (100*hc3)], axis=1).T
ax.fill_between(
    tra_map_full_c,
    hdi1[:, 0],
    hdi1[:, 1],
    color=cfsm(gval),
    alpha=1.0,
    lw=0.0,
    zorder=zo_1s,
)
ax.fill_between(
    tra_map_full_c,
    hdi2[:, 0],
    hdi2[:, 1],
    color=cfsm(gval+dg),
    alpha=1.0,
    lw=0.0,
    zorder=zo_2s,
)
ax.fill_between(
    tra_map_full_c,
    hdi3[:, 0],
    hdi3[:, 1],
    color=cfsm(gval+(dg*2)),
    alpha=1.0,
    lw=0.0,
    zorder=zo_3s,
)

all_xlims.append(ax.get_xlim())

axs.append(ax)

# ---
ax = plt.subplot2grid((nrows, ncols), (5, 0), rowspan=1)
poc.set_axis_default(ax, ticklabel_size=tfont, aspect="auto", labeldown=True)
ax.set_ylabel("res. ({:s})".format(u[1]), fontsize=lfont)
ax.axhline(0, color="k", lw=0.8)

ax.errorbar(
    tra_syn_noisy_c,
    res_map_c*u[0],
    yerr=err_tra_syn_c*u[0],
    marker=markers[1],
    ms=2.5,
    mec='black',
    mew=0.4,
    color="C1",
    ecolor="C1",
    elinewidth=0.4,
    capsize=0,
    ls='',
    zorder=zo_obs
)

axs.append(ax)

all_xlims = np.concatenate(all_xlims)
for ax in axs:
    ax.set_xlim(np.min(all_xlims), np.max(all_xlims))
    ax.legend(loc='center left', bbox_to_anchor =(1.01, 0.5), fontsize=lfont, frameon=False)

ax.set_xlabel("Time (days)")

fig.align_ylabels(axs)

plt.show()

plt.close(fig)

# %%


# %%



