# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import numpy as np

def compute_corr(x, y):
    """Correlate 2 matrices along last axis"""
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) *
                    np.sum(ym * ym, axis=-1))
    r = np.sum(xm * ym, axis=-1) / r_den
    return r
