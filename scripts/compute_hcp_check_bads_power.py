# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import matplotlib
import os.path as op

from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mne.time_frequency import psd_multitaper

from meeg_preprocessing.utils import (
    setup_provenance)

import config as cfg
import library as lib
import mne
from mne.report import Report
import hcp
from hcp.preprocessing import apply_ica_hcp

# configure logging + provenance tracking magic
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir='results')

hcp_path = '/Volumes/MEG-HUB/HCP'


sfmin, sfmax = dict(cfg.scale_free_windows)['low']
ofmin, ofmax = dict(cfg.frequency_windows)['alpha']


def mad_detect(y, thresh=3.5):
    # warning: this function does not check for NAs
    # nor does it address issues when
    # more than 50% of your data have identical values
    m = np.median(y)
    abs_dev = np.abs(y - m)
    left_mad = np.median(abs_dev[y <= m])
    right_mad = np.median(abs_dev[y >= m])
    y_mad = np.zeros(len(y))
    y_mad[y < m] = left_mad
    y_mad[y > m] = right_mad
    modified_z_score = 0.6745 * abs_dev / y_mad
    modified_z_score[y == m] = 0
    return modified_z_score > thresh, left_mad, right_mad

hcp_info = pd.read_csv('./hcp_unrestricted_data.csv')
results = list()
subjects = hcp_info['Subject'].astype(str).tolist()
my_iter = [(x, y) for x in subjects for y in range(3)]
import mkl
mkl.set_num_threads(1)
for subject, run_idx in my_iter:
    if not op.exists(op.join(hcp_path, subject)):
        continue
    if run_idx == 0:
        report = Report(title=subject)
    print(subject, run_idx)
    annots = hcp.io.read_annot_hcp(
        subject, hcp_path=hcp_path, data_type='rest',
        run_index=run_idx)
    ica_mat = hcp.io.read_ica_hcp(
        subject, hcp_path=hcp_path, data_type='rest',
        run_index=run_idx)
    raw = hcp.io.read_raw_hcp(
        subject, hcp_path=hcp_path, data_type='rest',
        run_index=run_idx)
    raw_noise = hcp.io.read_raw_hcp(
        subject, hcp_path=hcp_path, data_type='noise_empty_room',
        run_index=0)
    exclude = np.array(annots['ica']['ecg_eog_ic']) - 1
    for this_raw in (raw_noise, raw):
        this_raw.info['bads'] = annots['channels']['all']
        this_raw.pick_types(meg='mag', ref_meg=False)
        if len(exclude) > 0:
            apply_ica_hcp(this_raw, ica_mat=ica_mat, exclude=exclude)
    projs_noise = mne.compute_proj_raw(raw_noise, n_mag=12)
    raw.add_proj(projs_noise)
    raw.filter(None, 500, n_jobs=8, method='iir',
               iir_params=dict(order=4, ftype='butter'))
    events = mne.make_fixed_length_events(raw, 3000, duration=1)
    epochs = mne.Epochs(
        raw, events=events, event_id=3000, tmin=0, tmax=1, detrend=1,
        reject=dict(mag=5e-12), preload=True, decim=4, proj=True)
    # del raw, raw_noise

    X_psds = 0.0
    freq = 0
    for i_epoch in np.arange(len(epochs.events)):
        print i_epoch
        this_psd, freqs = psd_multitaper(epochs[i_epoch],
                                         fmin=5,
                                         fmax=35, bandwidth=4,
                                         n_jobs=1)
        X_psds += this_psd[0]
    X_psds /= i_epoch

    med_power = np.median(np.log10(X_psds), 1)
    outlier_mask, left_mad, right_mad = mad_detect(med_power)

    plot_range = np.arange(len(epochs.ch_names))

    outlier_label = np.array(epochs.ch_names)
    fig_mad = plt.figure()
    plt.axhline(np.median(med_power), color='#225ee4')
    plt.plot(
        plot_range[~outlier_mask],
        med_power[~outlier_mask], linestyle='None', marker='o', color='white')
    plt.ylim(med_power.min() - 1, med_power.min() + 2)
    for inds in np.where(outlier_mask)[0]:
        plt.plot(
            plot_range[inds],
            med_power[inds], linestyle='None', marker='o', color='red',
            label=outlier_label[inds])
    plt.ylabel('median log10(PSD)')
    plt.xlabel('channel index')
    plt.legend()
    report.add_figs_to_section(
        fig_mad, '%s: run-%i' % (subject, run_idx + 1), 'MAD power')

    mne_med_power = mne.EvokedArray(
        data=med_power[:, None], info=deepcopy(epochs.info), tmin=0, nave=1)
    hcp.preprocessing.transform_sensors_to_mne(mne_med_power)
    fig = mne_med_power.plot_topomap(
        [0], scale=1, cmap='viridis', vmin=np.min, vmax=np.max,
        show_names=True, mask=outlier_mask[:, None], contours=1,
        time_format='', unit='dB')

    for tt in fig.findobj(matplotlib.text.Text):
        if tt.get_text().startswith('A'):
            tt.set_color('red')

    fig.set_dpi(cfg.dpi)
    report.add_figs_to_section(
        fig, '%s: run-%i' % (subject, run_idx + 1), 'topo')
    results.append(
        {'subject': subject, 'run': run_idx,
         'labels': outlier_label[outlier_mask]})

    fig_log = lib.viz.plot_loglog(X_psds, freqs[(freqs >= 5) & (freqs <= 50)])
    fig_log.set_dpi(cfg.dpi)
    report.add_figs_to_section(
        fig_log, '%s: run-%i' % (subject, run_idx + 1), 'loglog')

    if run_idx == 2:
        report.save(
            op.join(results_dir, run_id,
                    'hcp_bads_%s.html' % subject), overwrite=True)
    plt.close('all')

# 
# write_hdf5(op.join(results_dir, run_id, 'bad_labels.hdf5'), all_results)
# 
# raw = hcp.io.read_raw_hcp(
#     subject, hcp_path=hcp_path,
#     data_type='rest', run_index=0) 
# 
# all_labels = np.array([k for k in raw.bti_ch_labels if k.startswith('A')])
# bad_info = mne.pick_info(
#     deepcopy(raw.info), mne.pick_types(raw.info, meg='mag'))
# 
# bad_dist = np.sum(
#     [np.in1d(all_labels, k['labels']) for k in all_results], 0)[:, None]
# mne_bad_dist = mne.EvokedArray(bad_dist, bad_info, tmin=0, nave=1)
# hcp.preprocessing.transform_sensors_to_mne(mne_bad_dist)
# 
# for ii, show_names in enumerate((True, False)):
#     fig = mne_bad_dist.plot_topomap(
#         [0], scale=1, cmap='viridis', vmin=np.min, vmax=np.max,
#         show_names=show_names, contours=0,
#         mask={1: np.array([b > 0 for b in bad_dist]), 0: None}[show_names],
#         time_format='', unit='sum', size=3)
#     for tt in fig.findobj(matplotlib.text.Text):
#         if tt.get_text().startswith('A'):
#             tt.set_color('red')
#     fig.set_dpi(300)
#     fig.savefig(op.join(results_dir, run_id, 'all_bads_%i.png' % ii),
#                 dpi=300)
