# -*- coding: utf-8 -*-
"""
:py:mod:`regression.py` - Utility Functions
-----------------------------------

Functionality for training and tuning hyperparameters of sklearn-like regression
algorithms used for predicting the lnlikelihood.

"""

# Tell module what it's allowed to import
__all__ = ["trainXGBoost"]

import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV


################################################################################
#
# Function for training and using an XGBoost regressor
#
################################################################################


def trainXGBoost(reg, theta, y, hyperparams=None, k=5):
    """
    Train an XGBoost regressor and optimize its hyperparameters using k-fold
    cross validation on the training set, theta and y.

    Parameters
    ----------
    reg : xgboost.XGBRegressor
    theta : array-like
        Input features (n_samples x n_features)
    y : array-like
        Input result of forward model (n_samples,)
    hyperparams : dict (optional)
        Algorithm hyperparameters to optimize with k-fold cross-validation. If
        None, defaults to {'max_depth': [2,4,6], 'n_estimators': [50,100,200]}
    k : int (optional)
        Number of folds to use for k-fold cross-validation. Defaults to k = 5.

    Returns
    -------
    reg : xgboost.XGBRegressor
        Trained regressor with optimized hyperparameters
    best_params : dict
        Dictionary of best hyperparameters found via cross-validation
    best_score : int
        Score (mean squared error) of best regressor found via cross-validation
    """

    # Need a model!
    if reg is None:
        reg = xgb.XGBRegressor(objective='reg:squarederror')

    # Optimize hyperparameters with k-folds cross-validation

    # Init default hyperparameters that one should tune
    if hyperparams is None:
        hyperparams = {'max_depth': [20], 'n_estimators': [250, 1000],
                       'reg_lambda' : np.logspace(-5, 2, 10),
                       'learning_rate' : [0.01, 0.1]}

    # Perform the cv!
    gscv = GridSearchCV(reg, param_grid=hyperparams, cv=k).fit(theta, y)

    # Extract best estimator, refit
    reg.set_params(**gscv.best_params_)
    reg.fit(theta, y)

    return reg, gscv.best_params_, gscv.best_score_
# end function
