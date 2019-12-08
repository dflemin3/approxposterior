#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Example script for Bayesian optimization of a synthetic function

@author: David P. Fleming [University of Washington, Seattle], 2019
@email: dflemin3 (at) uw (dot) edu

"""

from approxposterior import approx, gpUtils, likelihood as lh, utility as ut
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.optimize import minimize
matplotlib.rcParams.update({"font.size": 15})


# Define algorithm parameters
m0 = 3                           # Size of initial training set
bounds = [[-1, 2]]               # Prior bounds
algorithm = "jones"              # Expected Utility from Jones et al. (1998)
numNewPoints = 5                 # Number of new design points to find
seed = 27                        # RNG seed
np.random.seed(seed)

# First, directly minimize the function to see about how many evaluations it takes
fn = lambda x : -lh.testBOFn(x)
trueSoln = minimize(fn, lh.testBOFnSample(1), method="nelder-mead")

# Sample design points from prior to create initial training set
theta = lh.testBOFnSample(m0)

# Evaluate forward model log likelihood + lnprior for each point
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.testBOFn(theta[ii])

# Initialize default gp with an ExpSquaredKernel
gp = gpUtils.defaultGP(theta, y, white_noise=-10)

# Initialize object using the Wang & Li (2017) Rosenbrock function example
ap = approx.ApproxPosterior(theta=theta,
                            y=y,
                            gp=gp,
                            lnprior=lh.testBOFnLnPrior,
                            lnlike=lh.testBOFn,
                            priorSample=lh.testBOFnSample,
                            bounds=bounds,
                            algorithm=algorithm)

# Run the Bayesian optimization!
soln = ap.bayesOpt(nmax=10, tol=1.0e-3, seed=seed, verbose=False,
                   cache=False, gpMethod="powell", optGPEveryN=1, nGPRestarts=3,
                   nMinObjRestarts=5, initGPOpt=True, minObjMethod="nelder-mead",
                   gpHyperPrior=gpUtils.defaultHyperPrior)

# Plot objective function
fig, ax = plt.subplots(figsize=(6,5))
x = np.linspace(-1, 2, 100)

ax.plot(x, lh.testBOFn(x), lw=2.5, color="k")

# Format
ax.set_xlim(-1.1, 2.1)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"fn($\theta$)")

# Hide top, right axes
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

fig.tight_layout()
fig.savefig("objFn.png", dpi=200)

# Plot the solution path and function value convergence
fig, axes = plt.subplots(ncols=2, figsize=(12,6))

iters = [ii for ii in range(soln["nev"]+1)]

# Left: solution
axes[0].plot(iters, soln["thetas"], "o-", color="C0", lw=3)
axes[0].axhline(trueSoln["x"], ls="--", color="k", lw=2)
axes[0].text(0, trueSoln["x"] + 0.025, r"$\theta_{\mathrm{max}}$", color="k",
             fontsize=18)
st = "true thetaMax: %.3e, \nestimated thetaMax : %.3e" % (trueSoln["x"], soln["thetaBest"])
axes[0].text(1, 1, st, color="k", fontsize=12)

# Format
axes[0].set_ylabel("Theta")

# Right: solution value (- true soln since we minimized -fn)
axes[1].plot(iters, soln["vals"], "o-", color="C0", lw=3)
axes[1].axhline(-trueSoln["fun"], ls="--", color="k", lw=2)
axes[1].text(0, -trueSoln["fun"] + 0.005, "Global Maximum", color="k",
             fontsize=13)
st = "true Maximum: %.3e,\nestimated maximum : %.3e" % (-trueSoln["fun"], soln["valBest"])
axes[1].text(2, 0.1, st, color="k", fontsize=12)

# Format
axes[1].set_ylabel("fn(theta)")

# Format both axes
for ax in axes:
    ax.set_xlabel("Iteration")
    ax.set_xlim(-0.2, soln["nev"]-0.8)

    # Hide top, right axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

fig.tight_layout()
fig.savefig("bo.png", dpi=200, bbox_inches="tight")
