# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os
import os.path as op
import pandas as pd
import numpy as np
from scipy.stats import gmean

from meeg_preprocessing.utils import (
    get_data_picks, setup_provenance, handle_mkl)

from mne.time_frequency import multitaper_psd
from mne.minimum_norm import make_inverse_operator
from mne.minimum_norm import apply_inverse_epochs
from mne.minimum_norm import compute_source_psd_epochs

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor

import library as lib
from library.externals import h5io
import mne
import config as cfg


args = cfg.get_argparse().parse_known_args()[0]
subjects = [args.subject] if args.subject is not None else cfg.subjects
mkl_max_threads = (args.mkl_max_threads if args.mkl_max_threads is not None
                   else cfg.mkl_max_threads)
handle_mkl(max_threads=mkl_max_threads)

# configure logging + provenance tracking magic
report, run_id, results_dir, logger = setup_provenance(
    script=locals().get('__file__', op.curdir), results_dir='results')

fnames = [c for c in os.listdir(cfg.results_sensor) if 'hdf5' in c]

"""
h5io.write_hdf5(
    op.join(results_dir, run_id, '%s-results.hdf5' % subject),
    X_alphas,
    title='dyn/alphas/run%i' % i_run, overwrite='update')
h5io.write_hdf5(
    op.join(results_dir, run_id, '%s-results.hdf5' % subject),
    X_intercepts,
    title='dyn/intercpets/run%i' % i_run, overwrite='update')
h5io.write_hdf5(
    op.join(results_dir, run_id, '%s-results.hdf5' % subject),
    X_coefs,
    title='dyn/alphas/run%i' % i_run, overwrite='update')
h5io.write_hdf5(
    op.join(results_dir, run_id, '%s-results.hdf5' % subject),
    X_psds,
    title='dyn/psd/run%i' % i_run, overwrite='update')
"""

regressions = dict(
    # robust=RANSACRegressor(LinearRegression()),
    glm=LinearRegression()
)

sfmin, sfmax = dict(cfg.scale_free_windows)['low']
ofmin, ofmax = dict(cfg.frequency_windows)['alpha']


for fname in fnames:
    subject = fname.replace('-results.hdf5', '')
    group = [k for k, v in cfg.subject_map.items() if subject in v][0]
    fname = op.join(cfg.results_sensor, fname)

    psds = list()
    for i_run in range(6):
        psds.append(
            h5io.read_hdf5(fname, title='dyn/psd/run%i' % i_run))
    psds = np.concatenate(psds, axis=0)
    freqs = h5io.read_hdf5(fname, title='dyn/freqs')

    X_coefs = dict(glm=list(), robust=list())
    X_intercepts = dict(glm=list(), robust=list())

    for i_epoch, this_psd in enumerate(psds):

        sfmask = mne.utils._time_mask(freqs, sfmin, sfmax)
        for reg_kind, mod in regressions.items():
            mod.fit(np.log10(freqs[sfmask, None]),
                    np.log10(this_psd[:, sfmask].T))
            X_coefs[reg_kind].append(mod.coef_[:, 0])
            X_intercepts[reg_kind].append(mod.intercept_)
            import pdb; pdb.set_trace()

        alpha_mask = mne.utils._time_mask(freqs, ofmin, ofmax)
        # X_alphas.append(
        #     dict(mean=np.mean(this_psd[:, alpha_mask], axis=1),
        #          gmean=gmean(this_psd[:, alpha_mask], axis=1),
        #          mean_norm=np.mean(
        #              this_psd[:, alpha_mask] /
        #              this_psd[:, alpha_mask].sum(-1)[:, None], axis=1),
        #          gmean_norm=gmean(
        #              this_psd[:, alpha_mask] /
        #              this_psd[:, alpha_mask].sum(-1)[:, None], axis=1)))
        # # fig = lib.viz.plot_loglog(this_psd, freqs, sfmask,
        # #                           coefs=X_coefs[-1],
        # #                           intercepts=X_intercepts[-1])
        #
        #

    # report.save(op.join(results_dir, run_id, '%s-report.html' % subject))
    # XXX continuen here
    # X_coefs = np.array(X_coefs)
    # X_alpha = np.array([a['mean'] for a in alphas])
    # X_coefs = np.array(coefs)
    # corr = lib.stats.compute_corr(X_alpha.T, X_coefs.T)
    # corr = np.array([spearmanr(zscore(x), zscore(y)).correlation for x, y in zip(np.log10(X_alpha.T), X_coefs.T)])
    # corr = mne.EvokedArray(
    #     data=corr[:, None],
    #     info=mne.io.pick.pick_info(epochs.info, picks),
    #     tmin=0, nave=1)
    #
    #
    # coefs = mne.EvokedArray(
    #     data=np.median(X_coefs,0, keepdims=True).T,
    #     info=mne.io.pick.pick_info(epochs.info, picks),
    #     tmin=0, nave=1)
    #
    #
    # alpha = mne.EvokedArray(
    #     data=np.median(X_alpha, 0, keepdims=True).T,
    #     info=mne.io.pick.pick_info(epochs.info, picks),
    #     tmin=0, nave=1)
    #
    # alpha = mne.EvokedArray(
    #     data=X_alpha.T,
    #     info=mne.io.pick.pick_info(epochs.info, picks),
    #     tmin=0, nave=1)
    #
    # break  # XXX continue here
    #
