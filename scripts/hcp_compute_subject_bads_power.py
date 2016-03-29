# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os
import os.path as op
from argparse import ArgumentParser
import time
from copy import deepcopy


import numpy as np
import pandas as pd
from mne.time_frequency import psd_multitaper

from meeg_preprocessing.utils import (
    setup_provenance)

import config as cfg
import mne
from mne.report import Report
import hcp
from hcp.preprocessing import apply_ica_hcp
import mkl
import aws_hacks

mkl.set_num_threads(1)


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


def compute_power_sepctra_and_bads(subject, run_index, recordings_path,
                                   fmin, fmax,
                                   hcp_path, report, n_jobs=1):
    import matplotlib
    matplolib.use('Agg')
    import matplotlib.pyplot as plt
    annots = hcp.io.read_annot_hcp(
        subject, hcp_path=hcp_path, data_type='rest',
        run_index=run_index)
    ica_mat = hcp.io.read_ica_hcp(
        subject, hcp_path=hcp_path, data_type='rest',
        run_index=run_index)
    raw = hcp.io.read_raw_hcp(
        subject, hcp_path=hcp_path, data_type='rest',
        run_index=run_index)
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
    raw.filter(None, 500, n_jobs=n_jobs, method='iir',
               iir_params=dict(order=4, ftype='butter'))
    events = mne.make_fixed_length_events(raw, 3000, duration=1)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    epochs = mne.Epochs(
        raw, picks=picks, events=events, event_id=3000, tmin=0, tmax=1,
        detrend=1,
        reject=dict(mag=5e-12), preload=True, decim=16, proj=True)
    del raw, raw_noise
    X_psds = 0.0
    for ii in range(len(epochs.events)):
        psds, freqs = psd_multitaper(
            epochs[ii], fmin=fmin, fmax=fmax, bandwidth=4, n_jobs=1)
        X_psds += psds
    X_psds /= (ii + 1)
    X_psds = X_psds[0]
    written_files = list()

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
        fig_mad, '%s: run-%i' % (subject, run_index + 1), 'MAD power')

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

    if report is not None:
        report.add_figs_to_section(
            fig, '%s: run-%i' % (subject, run_index + 1), 'topo')

    results = {'subject': subject, 'run': run_index,
               'labels': outlier_label[outlier_mask]}

    fig_log = lib.viz.plot_loglog(X_psds, freqs[(freqs >= 5) & (freqs <= 50)])
    fig_log.set_dpi(cfg.dpi)
    if report is not None:
        report.add_figs_to_section(
            fig_log, '%s: run-%i' % (subject, run_index + 1), 'loglog')
    plt.close('all')
    return written_files, results


if __name__ == '__main__':
    storage_dir = '/mnt'
    start_time_global = time.time()
    aws_details = pd.read_csv('aws_details.csv')
    aws_access_key_id = aws_details['Access Key Id'].values[0]
    aws_secret_access_key = aws_details['Secret Access Key'].values[0]

    aws_details = pd.read_csv('aws_hcp_details.csv')
    hcp_aws_access_key_id = aws_details['Access Key Id'].values[0]
    hcp_aws_secret_access_key = aws_details['Secret Access Key'].values[0]

    parser = ArgumentParser(description='tell subject')
    parser.add_argument('--subject', metavar='subject', type=str, nargs='?',
                        default=None,
                        help='the subject to extract')
    parser.add_argument('--storage_dir', metavar='storage_dir', type=str,
                        nargs='?', default=storage_dir,
                        help='the storage dir')
    parser.add_argument('--keep_files',
                        action='store_true',
                        help='delete files that were written')
    parser.add_argument('--n_jobs', metavar='n_jobs', type=int,
                        nargs='?', default=1,
                        help='the number of jobs to run in parallel')
    parser.add_argument('--s3', action='store_true',
                        help='skip s3')

    args = parser.parse_args()
    subject = args.subject
    storage_dir = args.storage_dir

    hcp_path = op.join(storage_dir, 'HCP')
    recordings_path = op.join(storage_dir, 'hcp-meg')

    if not op.exists(storage_dir):
        os.makedirs(storage_dir)

    # configure logging + provenance tracking magic
    report, run_id, results_dir, logger = setup_provenance(
        script=__file__, results_dir=op.join(recordings_path, subject))

    s3_meg_files = hcp.io.file_mapping.get_s3_keys_meg(
        subject, data_types=('rest',), processing=('unprocessed'),
        run_inds=(0, 1, 2))

    if args.s3 is True:
        start_time = time.time()
        for key in s3_meg_files:
            fname = op.join(hcp_path, key.split('HCP_900')[1].lstrip('/'))
            if not op.exists(op.split(fname)[0]):
                os.makedirs(op.split(fname)[0])
            if not op.exists(fname):
                aws_hacks.download_from_s3(
                    aws_access_key_id=hcp_aws_access_key_id,
                    aws_secret_access_key=hcp_aws_secret_access_key,
                    fname=fname,
                    bucket='hcp-openaccess', key=key)
        elapsed_time = time.time() - start_time
        print('Elapsed time downloading from s3 {}'.format(
            time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    written_files = list()
    results = list()
    report = Report(subject)
    for run_index in range(3):
        files, this_result = compute_power_sepctra_and_bads(
            subject=subject, run_index=run_index,
            recordings_path=recordings_path,
            fmin=5, fmax=50, hcp_path=hcp_path,
            report=report, n_jobs=1)
        written_files.extend(files)
        results.append(this_result)

    df = pd.DataFrame(results)
    written_files.append(
        op.join(results_dir, run_id, 'bads_by_power.csv'))
    df.to_csv(written_files[-1])

    results_path = op.join(results_dir, run_id)
    written_files.extend([op.join(results_path, f) for f in
                          os.listdir(results_path)])
    written_files.append(op.join(results_path, 'bads-report.html'))
    report.save(written_files[-1], open_browser=False)

    written_files.append(op.join(results_path, 'written_files.txt'))
    with open(written_files[-1], 'w') as fid:
        fid.write('\n'.join(written_files))

    if args.s3 is True:
        start_time = time.time()
        for fname in written_files:
            key = fname.split(storage_dir)[-1].lstrip('/')
            aws_hacks.upload_to_s3(
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key,
                fname=fname,
                bucket='hcp-meg-data', key=key, host='s3.amazonaws.com')

        elapsed_time = time.time() - start_time
        print('Elapsed time uploading to s3 {}'.format(
            time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

    if args.keep_files and args.s3 is True:
        my_files_to_clean = list()
        my_files_to_clean += written_files
        my_files_to_clean += [op.join(recordings_path,
                                      f.replace('HCP_900/', ''))
                              for f in s3_meg_files]
        for fname in my_files_to_clean:
            if op.exists(fname):
                os.remove(fname)

    elapsed_time_global = time.time() - start_time_global
    print('Elapsed time for running scripts {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time_global))))
