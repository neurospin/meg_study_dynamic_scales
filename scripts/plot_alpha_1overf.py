# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os
import os.path as op
import pandas as pd
import numpy as np
from scipy.stats import trim_mean

from meeg_preprocessing.utils import (
    get_data_picks, setup_provenance, handle_mkl)

from mne.time_frequency import multitaper_psd
from mne.minimum_norm import make_inverse_operator
from mne.minimum_norm import apply_inverse_epochs
from mne.minimum_norm import compute_source_psd_epochs
from sklearn.linear_model import LinearRegression

import library as lib
from library.externals import h5io
import mne
import config as cfg
import matplotlib.pyplot as plt

args = cfg.get_argparse().parse_known_args()[0]
subjects = [args.subject] if args.subject is not None else cfg.subjects
mkl_max_threads = (args.mkl_max_threads if args.mkl_max_threads is not None
                   else cfg.mkl_max_threads)
handle_mkl(max_threads=mkl_max_threads)

# configure logging + provenance tracking magic
report, run_id, results_dir, logger = setup_provenance(
    script=locals().get('__file__', op.curdir), results_dir='results')

fnames = [c for c in os.listdir(cfg.results_sensor) if 'hdf5' in c]

"""XXX repair saving frequencies
for fname in fnames:
    h5io.write_hdf5(op.join(cfg.results_sensor, fname),
                    np.linspace(0.1, 15, 418), title='dyn/freqs',
                    overwrite='update')
"""

lm = LinearRegression()

sfmin, sfmax = dict(cfg.scale_free_windows)['low']
ofmin, ofmax = dict(cfg.frequency_windows)['alpha']

lout = mne.channels.read_layout('Vectorview-mag')
info = mne.create_info(
    lout.names, sfreq=1000, ch_types=['mag'] * len(lout.names))

coefs_list = list()
r2_list = list()
psds_list = list()
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

    coefs, intercepts, msq, r2 = lib.stats.compute_log_linear_fit(
        psds, freqs, sfmin, sfmax, lm)

    lm_props = [
        (coefs, 'betas', coefs_list),
        (r2, 'r2', r2_list),
    ]
    if subject != 'ak_130184-130619':
        continue

    mne_ga_psds = mne.EvokedArray(
        data=10 * np.log10(psds[0]), info=info, tmin=0, nave=1)
    mne_ga_psds.times = freqs
    fig_psds_topos = mne_ga_psds.plot_topomap(
        times=freqs[freqs >= 8][::20], ch_type='mag',
        layout=lout, scale=1, cmap='magma', scale_time=1, unit='dB',
        time_format='%0.1f Hz', contours=0, vmin=np.min, vmax=np.max)

    psds_list.append(mne_ga_psds)
    report.add_figs_to_section(
        fig_psds_topos, subject, section='alpha-power')

    for lm_prop, kind, my_list in lm_props:
        lm_prop = lib.stats.lm_bcast(lm_prop, psds)
        X_corr = lib.stats.compute_corr(lm_prop.T, np.log10(psds).T)
        fig_lines = lib.viz.plot_loglog_corr(freqs, X_corr)
        fig_lines.set_dpi(cfg.dpi)
        plt.ylabel('correlation (%s)' % kind)

        mne_coefs_corr = mne.EvokedArray(
            data=X_corr.T, info=info, tmin=0, nave=1)
        mne_coefs_corr.times = freqs
        my_list.append(mne_coefs_corr)

        fig_topos = mne_coefs_corr.plot_topomap(
            times=freqs[freqs >= 8][::20], ch_type='mag',
            layout=lout, scale=1, cmap='RdYlBu_r', scale_time=1, unit='r',
            time_format='%1d Hz', contours=0, vmin=-0.6, vmax=0.6)
        fig_topos.set_dpi(cfg.dpi)

        fig_image = plt.figure(dpi=cfg.dpi)
        plt.imshow(
            mne_coefs_corr.data[np.argsort(mne_coefs_corr.data.var(1))],
            cmap='RdYlBu_r', vmin=-0.9, vmax=0.9, norm=None,
            interpolation='nearest',
            aspect=len(freqs) / float(len(mne_coefs_corr.data)))
        cb = plt.colorbar()
        cb.set_label('r')
        plt.xticks(range(len(freqs))[50::100], freqs[50::100].astype(int))
        plt.xlabel('Frequency [Hz]')
        plt.ylabel('Channels [by variance]')

        for fig, section in zip(
            [fig_lines, fig_topos, fig_image],
                ['parallel ' + kind, 'topos ' + kind, 'image ' + kind]):
            report.add_figs_to_section(fig, subject, section=section)

grand_ave = coefs_list[0].copy()
grand_ave.data = trim_mean([c.data for c in coefs_list], 0.1, axis=0)

fig_topos_ga = grand_ave.plot_topomap(
    times=freqs[freqs >= 8][::20], ch_type='mag',
    layout=lout, scale=1, cmap='RdYlBu_r', scale_time=1, unit='r',
    time_format='%0.1f Hz', contours=0,
    # vmin=-0.6, vmax=0.6
)
fig_topos_ga.set_dpi(cfg.dpi)

fig_lines_ga = lib.viz.plot_loglog_corr(freqs, grand_ave.data.T)
fig_lines_ga.set_dpi(cfg.dpi)
plt.ylabel('correlation (%s)' % 'beta')

for fig, section in zip(
    [fig_lines_ga, fig_topos_ga],
        ['parallel ' + kind, 'topos ' + kind]):
    report.add_figs_to_section(fig, subject, section='grand_ave_coefs')

report.save()
