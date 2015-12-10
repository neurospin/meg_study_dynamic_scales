# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os.path as op
import glob
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

import library as lib
import mne
from mne import io
from mne.preprocessing import read_ica
from config import (
    get_argparse,
    recordings_path,
    subjects,
    sss_raw_name_tmp,
    subjects_to_group,
    frequency_windows,
    scale_free_windows,
    current_ica_solution_path,
    ica_meg_combined,
    ica_fname_tmp,
    fwd_fname,
    noise_cov_fname,
    mkl_max_threads
)

args = get_argparse().parse_known_args()[0]
subjects = [args.subject] if args.subject is not None else subjects
mkl_max_threads = (args.mkl_max_threads if args.mkl_max_threads is not None
                   else mkl_max_threads)
handle_mkl(max_threads=mkl_max_threads)

# configure logging + provenance tracking magic
report, run_id, results_dir, logger = setup_provenance(
    script=locals().get('__file__', op.curdir), results_dir='results')


def make_overlapping_events(raw, event_id, start, stop, step, duration):
    """Create overlapping events"""
    events = [
        mne.make_fixed_length_events(
            raw, id=event_id, start=float(ii), duration=duration)
        for ii in np.arange(start, stop, step)]
    events = [e for e in events if len(e) > 0]
    events = [e for e in events if
              ((e[:, 0].min() - raw.first_samp) / raw.info['sfreq']) >=
              duration]
    events = np.concatenate(events, axis=0)
    events = events[events[:, 0].argsort()]

    return events

ica_infos = list()
for subject in subjects:
    report = mne.report.Report(title=subject)

    # read files
    raw_fnames = glob.glob(op.join(
        recordings_path, subject, sss_raw_name_tmp.format(
            group='*', run='*')))[:1]

    raw = io.Raw(raw_fnames, preload=True)
    for picks, ch_type in get_data_picks(raw, meg_combined=ica_meg_combined):
        this_ica_fname = op.join(
            current_ica_solution_path,
            ica_fname_tmp.format(subject=subject, ch_type=ch_type))
        ica = read_ica(this_ica_fname)
        ica.apply(raw)
    fwd = mne.read_forward_solution(
        op.join(recordings_path, subject, fwd_fname))
    noise_cov = mne.read_cov(
        op.join(recordings_path, subject, noise_cov_fname))

    invop = make_inverse_operator(raw.info, forward=fwd, noise_cov=noise_cov)

    events = make_overlapping_events(
        raw, 3000, 0, 70., step=15, duration=28.0)
    epochs = mne.Epochs(raw, events=events, event_id=3000, tmin=0, tmax=28)
    picks = mne.pick_types(raw.info, meg='mag')
    X_psd = None
    alphas = list()
    coefs = list()
    regression = LinearRegression()
    sfmin, sfmax = dict(scale_free_windows)['low']
    ofmin, ofmax = dict(frequency_windows)['alpha']

    for ii, epoch in enumerate(epochs):
        this_psd, freqs = multitaper_psd(epoch[picks],
                                         sfreq=epochs.info['sfreq'], fmin=0.1,
                                         fmax=15, bandwidth=0.5)
        if ii == 0:
            X_psd = np.empty_like(this_psd)
        sfmask = mne.utils._time_mask(freqs, sfmin, sfmax)
        X_psd += this_psd
        coefs.append(regression.fit(
            freqs[sfmask, None], this_psd[:, sfmask].T).coef_[:, 0])
        alpha_mask = mne.utils._time_mask(freqs, ofmin, ofmax)
        alphas.append(
            dict(mean=np.mean(this_psd[:, alpha_mask], axis=1),
                 gmean=gmean(this_psd[:, alpha_mask], axis=1),
                 mean_norm=np.mean(this_psd[:, alpha_mask] /
                                   this_psd[:, alpha_mask].sum(-1)[:, None],
                                   axis=1),
                 gmean_norm=gmean(this_psd[:, alpha_mask] /
                                  this_psd[:, alpha_mask].sum(-1)[:, None],
                                  axis=1)))
    break  # XXX continue here
