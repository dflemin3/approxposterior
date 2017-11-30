"""

DOCS

August 2017

@author: David P. Fleming [University of Washington, Seattle]
@email: dflemin3 (at) uw (dot) edu

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

# Tell module what it's allowed to import
__all__ = ["lnlike", "lnprior", "lnprob"]

import numpy as np
import emcee
import corner
from approxposterior import likelihood as lh
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

# Plot what the posterior looks like (approximately)

num = 100
run_mcmc = False

xx = np.linspace(-10, 10, num)
yy = np.linspace(-10, 10, num)

ll = np.zeros((len(xx),len(yy)))
for ii in range(len(xx)):
    for jj in range(len(yy)):
        ll[ii,jj] = lh.bimodal_normal_lnprob([xx[ii],yy[jj]], mus=None, icovs=None)

fig, ax = plt.subplots()
im = ax.pcolormesh(xx, yy, np.fabs(ll.T),
                   norm=LogNorm(vmin=np.fabs(ll).min(), vmax=np.fabs(ll).max()))
cb = fig.colorbar(im)

# Format
cb.set_label("Abs(-LogProb)", rotation=270, labelpad=20)
ax.set_xlim(-10,10)
ax.set_ylim(-10,10)

plt.show()

# Run MCMC (which will suck since bimodal, but whatever)
if run_mcmc:
    ndim = 2
    nwalkers = 100 * ndim

    x0 = [np.random.uniform(low=-10, high=10, size=ndim) for ii in range(nwalkers)]

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lh.bimodal_normal_lnprob)
    sampler.run_mcmc(x0, 1000)
    samples = sampler.chain[:, :, :].reshape((-1, ndim))

    fig, ax = plt.subplots()
    fig = corner.corner(samples)
    fig.savefig("triangle.png")
