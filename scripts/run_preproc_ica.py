"""
Preprocessing MEG and EEG data using filters and ICA
====================================================

This examples filters MEG and EEG data and subsequently computes separate
ICA solutions for MEG and EEG. An html report is created, which includes
diagnostic plots of the preprocessing steps.
"""
# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os.path as op
import glob
import pandas as pd

from meeg_preprocessing import compute_ica
from meeg_preprocessing.utils import (
    get_data_picks, setup_provenance, handle_mkl)

import library as lib
import mne
from mne import io
from config import (
    get_argparse,
    recordings_path,
    subjects,
    sss_raw_name_tmp,
    subjects_to_group,
    n_components,
    ica_reject,
    n_max_ecg,
    n_max_eog,
    ica_decim,
    ica_meg_combined,
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

ica_infos = list()
for subject in subjects:
    report = mne.report.Report(title=subject)

    # read files
    raw_fnames = glob.glob(op.join(
        recordings_path, subject, sss_raw_name_tmp.format(
            group='*', run='*')))

    raw = io.Raw(raw_fnames, preload=True)
    lib.decimate_raw(raw, decim=ica_decim)

    # get picks and iterate over channels
    artifact_stats = dict()
    ica_info = dict()
    for picks, ch_type in get_data_picks(raw, meg_combined=ica_meg_combined):
        ica, _ = compute_ica(raw, picks=picks,
                             subject='%s (%s)' % (
                                 subject, subjects_to_group[subject]),
                             n_components=n_components,
                             n_max_ecg=n_max_ecg, n_max_eog=n_max_eog,
                             reject=ica_reject,
                             random_state=42,
                             artifact_stats=artifact_stats,
                             decim=1, report=report)
        ica.save(
            op.join(
                results_dir, run_id, '{}-{}-ica.fif'.format(subject, ch_type)))
        labels = getattr(ica, 'labels_', 0)
        if 'ecg' not in labels:
            labels['ecg'] = []
        if 'eog' not in labels:
            labels['ecg'] = []
        ica_info.update(
            {'%s_n_components' % ch_type: ica.n_components_,
             '%s_n_components_ecg' % ch_type: len(labels['ecg']) if labels
             else labels,
             '%s_n_components_eog' % ch_type: len(labels['eog']) if labels
             else labels})
    ica_info.update(artifact_stats)
    ica_infos.append(ica_info)
    report.save(  # save in automatically generated folder
        op.join(results_dir, run_id,
                'preprocessing-report-{}.html'.format(subject)),
        open_browser=True, overwrite=True)

ica_df = pd.DataFrame(ica_infos)
ica_df.to_csv(op.join(results_dir, run_id, 'ica_info.csv'))
