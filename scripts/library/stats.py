# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import numpy as np
import mne

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression


def lm_bcast(x, psds):
    """broadcast linear model outputs for efficient correlation"""
    return np.transpose((x * np.ones((psds.shape[-1], 1, 1))), [1, 2, 0])


def compute_corr(x, y):
    """Correlate 2 matrices along last axis"""
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) *
                    np.sum(ym * ym, axis=-1))
    r = np.sum(xm * ym, axis=-1) / r_den
    return r
from sklearn.linear_model import LinearRegression


def compute_log_linear_fit(psds, freqs, sfmin, sfmax, reg=None, log_fun=None):
    sfmask = mne.utils._time_mask(freqs, sfmin, sfmax)
    x = freqs[sfmask, None]
    if log_fun is not None:
        x = log_fun(x)
    if reg is None:
        reg = LinearRegression()

    coefs = list()
    intercepts = list()
    msq = list()
    r2 = list()
    if log_fun is not None:
        psds = log_fun(psds)
    for i_epoch, this_psd in enumerate(psds):

        Y = this_psd[:, sfmask].T
        reg.fit(x, Y)
        pred = reg.predict(x)
        msq.append(np.array(
            [mean_squared_error(b, a) for a, b in zip(pred.T, Y.T)]))
        r2.append(np.array(
            [r2_score(b, a) for a, b in zip(pred.T, Y.T)]))
        coefs.append(reg.coef_[:, 0])
        intercepts.append(reg.intercept_)

    coefs = np.array(coefs)
    intercepts = np.array(intercepts)
    msq = np.array(msq)
    r2 = np.array(r2)
    return coefs, intercepts, msq, r2
