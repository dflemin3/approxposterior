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
numNewPoints = 10                # Number of new design points to find
seed = 91                        # RNG seed
np.random.seed(seed)

# First, directly minimize the function to see about how many evaluations it takes
fn = lambda x : -(lh.testBOFn(x) + lh.testBOFnLnPrior(x))
trueSoln = minimize(fn, lh.testBOFnSample(1), method="nelder-mead")

# Sample design points from prior to create initial training set
theta = lh.testBOFnSample(m0)

# Evaluate forward model log likelihood + lnprior for each point
y = np.zeros(len(theta))
for ii in range(len(theta)):
    y[ii] = lh.testBOFn(theta[ii]) + lh.testBOFnLnPrior(theta[ii])

# Initialize default gp with an ExpSquaredKernel
gp = gpUtils.defaultGP(theta, y, white_noise=-12, fitAmp=True)

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
soln = ap.bayesOpt(nmax=numNewPoints, tol=1.0e-3, seed=seed, verbose=False,
                   cache=False, gpMethod="powell", optGPEveryN=1, nGPRestarts=2,
                   nMinObjRestarts=5, initGPOpt=True, minObjMethod="nelder-mead",
                   gpHyperPrior=gpUtils.defaultHyperPrior, findMAP=True)

# Compare truth to approximate Bayesian optimization solution
print("Truth:")
print("True maximum:", float(trueSoln["x"]))
print("True function value at maximum:", -trueSoln["fun"])
print()
print("BayesOpt solution using GP surrogate model:")
print("BayesOpt theta at maximum:", soln["thetaBest"])
print("BayesOpt function value at  MAP:", soln["valBest"])
print("BayesOpt forward model evaluations:", soln["nev"])
print()
print("GP MAP solution:")
print("Theta at approximate MAP:", soln["thetaMAPBest"])
print("GP predictive conditional function value at approximate MAP:", soln["valMAPBest"])

# Plot objective function
fig, ax = plt.subplots(figsize=(6,5))

x = np.linspace(-1, 2, 100)
ax.plot(x, lh.testBOFn(x), lw=2.5, color="k")

# Format
ax.set_xlim(-1.1, 2.1)
ax.set_xlabel(r"$\theta$")
ax.set_ylabel(r"f($\theta$)")

# Hide top, right axes
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.yaxis.set_ticks_position("left")
ax.xaxis.set_ticks_position("bottom")

fig.savefig("objFn.png", bbox_inches="tight", dpi=200)

# Plot the solution path and function value convergence
fig, axes = plt.subplots(ncols=2, figsize=(12,6))

# Extract number of iterations ran by bayesopt routine
iters = [ii for ii in range(soln["nev"])]

# Left: solution
axes[0].axhline(trueSoln["x"], ls="--", color="k", lw=2)
axes[0].plot(iters, soln["thetas"], "o-", color="C0", lw=2.5, label="GP BayesOpt")
axes[0].plot(iters, soln["thetasMAP"], "o-", color="C1", lw=2.5, label="GP approximate MAP")

# Format
axes[0].set_ylabel(r"$\theta$")
axes[0].legend(loc="best", framealpha=0, fontsize=14)

# Right: solution value (- true soln since we minimized -fn)
axes[1].axhline(-trueSoln["fun"], ls="--", color="k", lw=2)
axes[1].plot(iters, soln["vals"], "o-", color="C0", lw=2.5)
axes[1].plot(iters, soln["valsMAP"], "o-", color="C1", lw=2.5)

# Format
axes[1].set_ylabel(r"$f(\theta)$")

# Format both axes
for ax in axes:
    ax.set_xlabel("Iteration")
    ax.set_xlim(-0.2, soln["nev"]-0.8)

    # Hide top, right axes
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

fig.savefig("bo.png", dpi=200, bbox_inches="tight")
