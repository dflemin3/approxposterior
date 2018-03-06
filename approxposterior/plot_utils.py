# -*- coding: utf-8 -*-
"""

Diagnostic plot utility functions useful for debugging and to examine what the
Gaussian processes and Gaussian mixture models are doing/learning.

"""

from __future__ import (print_function, division, absolute_import,
                        unicode_literals)

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


__all__ = ["plot_gp", "plot_GMM_loglike"]


def plot_gp(gp, theta, y, xmin=-5, xmax=5, ymin=-5, ymax=5, n=100,
            return_type="mean", save_plot=None, log=False, **kw):
    """
    Plot a 2D slice of the Gaussian process's predictions.

    Parameters
    ----------
    theta : array
        input coordinates
    y : array
        observations corresponding to theta (data on which GP is conditioned)
    xmin : float (optional)
        Defaults to -5
    xmax : float (optional)
        Defaults to 5
    ymin : float (optional)
        Defaults to -5
    ymax : float (optional)
        Defaults to 5
    n : int (optional)
        number of grid points.  Defaults to 100
    return_type : str (optional)
        Whether to plot GP's mean, std, variance.  Defaults to mean.
    save_plot : str (optional)
        If not none, saves the plot as save_plot (name of figure)
    log : bool (optional)
        Whether or not to use a log colorbar.  Defaults to False

    Returns
    -------
    fig : matplotlib figure object
    ax : matplotlib axis object
    """

    xx = np.linspace(xmin, xmax, n)
    yy = np.linspace(ymin, ymax, n)

    zz = np.zeros((len(xx),len(yy)))
    for ii in range(len(xx)):
        for jj in range(len(yy)):
            mu, var = gp.predict(y, np.array([xx[ii],yy[jj]]).reshape(1,-1), return_var=True)
            if return_type.lower() == "var":
                zz[ii,jj] = var
            elif return_type.lower() == "mean":
                zz[ii,jj] = mu
            elif return_type.lower() == "utility":
                zz[ii,jj] = np.fabs(-(2.0*mu + var) - ut.logsubexp(var, 0.0))
            else:
                raise IOError("Invalid return_type : %s" % return_type)

    norm = None
    if log:
        if return_type.lower() == "mean" or return_type.lower() == "utility":
            zz = np.fabs(zz)
            zz[zz <= 1.0e-5] = 1.0e-5

        if return_type.lower() == "var":
            zz[zz <= 1.0e-8] = 1.0e-8

        norm = LogNorm(vmin=zz.min(), vmax=zz.max())

    # Plot what the GP thinks the function looks like
    fig, ax = plt.subplots(**kw)
    im = ax.pcolormesh(xx, yy, zz.T, norm=norm)
    cb = fig.colorbar(im)

    if return_type.lower() == "var":
        cb.set_label("GP Posterior Variance", labelpad=20, rotation=270)
    elif return_type.lower() == "mean":
        cb.set_label("|Mean GP Posterior Density (smaller better)|", labelpad=20, rotation=270)
    elif return_type.lower() == "utility":
        cb.set_label("|Utility Function (smaller better)|", labelpad=20, rotation=270)

    # Scatter plot where the points are
    ax.scatter(theta[:,0], theta[:,1], color="red")

    # Format
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    if save_plot is not None:
        fig.savefig(save_plot, bbox_inches="tight")

    return fig, ax
# end function


def plot_GMM_loglike(GMM, theta, save_plot=None, xmin=-5, xmax=5, ymin=-5, ymax=5):
    """
    Plot a 2D slice of a Gaussian Mixture Model's predicted log likelihood.

    Parameters
    ----------
    GMM : sklearn.mixture.GaussianMixture
    theta : array
        input coordinates
    save_plot : str (optional)
        If not none, saves the plot as save_plot (name of figure)
    xmin : float (optional)
        Defaults to -5
    xmax : float (optional)
        Defaults to 5
    ymin : float (optional)
        Defaults to -5
    ymax : float (optional)
        Defaults to 5

    Returns
    -------
    fig : matplotlib figure object
    ax : matplotlib axis object
    """

    # display predicted scores by the model as a contour plot
    x = np.linspace(xmin, xmax)
    y = np.linspace(ymin, ymax)
    X, Y = np.meshgrid(x, y)
    XX = np.array([X.ravel(), Y.ravel()]).T
    Z = -GMM.score_samples(XX)
    Z = Z.reshape(X.shape)

    fig, ax = plt.subplots(figsize=(9,8))
    CS = ax.contourf(X, Y, Z, norm=LogNorm(vmin=1.0e-1, vmax=1.0e2),
                   levels=np.logspace(-1, 2, 10), lw=3)
    cb = fig.colorbar(CS, shrink=0.8, extend='both')
    cb.set_label("|GMM LogLike|", labelpad=20, rotation=270)
    ax.scatter(theta[:,0], theta[:,1], color="r", zorder=20)
    ax.set_xlim(-5,5)
    ax.set_ylim(-5,5)

    if save_plot is not None:
        fig.savefig(save_plot)

    return fig, ax
# end function
