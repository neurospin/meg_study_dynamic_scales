import os.path as op
from copy import deepcopy

import numpy as np
import pandas as pd

import mne
from mne.time_frequency import psd_multitaper
import hcp

from .hcp_utils import hcp_preprocess_ssp_ica
from .utils import mad_detect
from .viz import plot_loglog


def dummy(subject):
    print(subject)
    return list()


def _psd_average_sensor_space(
        subject, run_index, recordings_path, fmin, fmax,
        hcp_path, n_ssp, decim,  mt_bandwidth, duration, n_jobs):

    raw = hcp_preprocess_ssp_ica(
        subject=subject, run_index=run_index, recordings_path=recordings_path,
        hcp_path=hcp_path, fmin=fmin, fmax=fmax, n_jobs=n_jobs, n_ssp=n_ssp)

    events = mne.make_fixed_length_events(raw, 3000, duration=duration)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    epochs = mne.Epochs(
        raw, picks=picks, events=events, event_id=3000, tmin=0, tmax=duration,
        detrend=1, baseline=None,
        reject=dict(mag=5e-12), preload=True, decim=decim, proj=True)

    X_psds = 0.0
    for ii in range(len(epochs.events)):
        psds, freqs = psd_multitaper(
            epochs[ii], fmin=fmin, fmax=fmax, bandwidth=mt_bandwidth, n_jobs=1)
        X_psds += psds
    X_psds /= (ii + 1)
    X_psds = X_psds[0]
    mne_psds = mne.EvokedArray(
        data=X_psds, info=deepcopy(epochs.info), tmin=0, nave=1)
    hcp.preprocessing.transform_sensors_to_mne(mne_psds)
    return mne_psds, freqs


def _psd_epochs_sensor_space(
        subject, run_index, recordings_path, fmin, fmax,
        hcp_path, n_ssp, decim,  mt_bandwidth, duration, n_jobs):

    raw = hcp_preprocess_ssp_ica(
        subject=subject, run_index=run_index, recordings_path=recordings_path,
        hcp_path=hcp_path, fmin=fmin, fmax=fmax, n_jobs=n_jobs, n_ssp=n_ssp)

    events = mne.make_fixed_length_events(raw, 3000, duration=duration)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    epochs = mne.Epochs(
        raw, picks=picks, events=events, event_id=3000, tmin=0, tmax=duration,
        detrend=1, baseline=None,
        reject=dict(mag=5e-12), preload=True, decim=decim, proj=True)
    epochs.apply_proj()

    X_psds, freqs = psd_multitaper(
        epochs, fmin=fmin, fmax=fmax, bandwidth=mt_bandwidth, n_jobs=1)

    info = deepcopy(epochs.info)
    info['projs'] = list()
    events = epochs.events.copy()
    del epochs
    mne_psds = mne.EpochsArray(data=X_psds, info=info, events=events, tmin=0)
    hcp.preprocessing.transform_sensors_to_mne(mne_psds)
    return mne_psds, freqs


def compute_power_sepctra(
        subject, recordings_path, fmin=None, fmax=200,
        hcp_path=op.curdir, n_ssp=12, decim=1, mt_bandwidth=4, duration=30,
        report=None, dpi=300, n_jobs=1, results_dir=None, run_inds=(0, 1, 2),
        run_id=None, out='average'):
    if out == 'average':
        fun = _psd_average_sensor_space
        file_tag = 'ave'
    elif out == 'epochs':
        fun = _psd_epochs_sensor_space
        file_tag = 'epo'
    else:
        raise ValueError('"out" must be "epochs" or "average"')

    for run_index in run_inds:
        mne_psds, freqs = fun(
            subject=subject, run_index=run_index,
            recordings_path=recordings_path, fmin=fmin, fmax=fmax,
            hcp_path=hcp_path, n_ssp=n_ssp, decim=decim,
            mt_bandwidth=mt_bandwidth,
            duration=duration, n_jobs=n_jobs)

        written_files = list()
        written_files.append(
            op.join(recordings_path, subject, 'psds-r%i-%i-%i-%s.fif' % (
                run_index, int(0 if fmin is None else fmin), int(fmax),
                file_tag)))
        fig_log = plot_loglog(
            mne_psds.average().data if out == 'epochs' else mne_psds.data,
            freqs[(freqs >= fmin) & (freqs <= fmax)],
            xticks=(0.1, 1, 10, 100))
        fig_log.set_dpi(dpi)
        if report is not None:
            report.add_figs_to_section(
                fig_log, '%s: run-%i' % (subject, run_index + 1), 'loglog')
        mne_psds.save(written_files[-1])

    return written_files


def compute_power_sepctra_and_bads(
        subject, recordings_path, fmin=None, fmax=200,
        hcp_path=op.curdir, n_ssp=12, decim=16,  mt_bandwidth=4, duration=1,
        report=None, dpi=300, n_jobs=1, run_inds=(0, 1, 2), results_dir=None,
        run_id=None):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    written_files = list()
    results = list()
    if not isinstance(run_inds, (list, tuple)):
        run_inds = [run_inds]

    for run_ind in run_inds:
        mne_psds, freqs, info = _psd_average_sensor_space(
            subject=subject, run_index=run_ind,
            recordings_path=recordings_path, fmin=fmin, fmax=fmax,
            hcp_path=hcp_path, n_ssp=n_ssp, decim=decim,
            mt_bandwidth=mt_bandwidth,
            duration=duration, n_jobs=n_jobs)

        written_files.append(
            op.join(recordings_path, subject, 'psds-bads-%i-%i-ave.fif' % (
                int(fmin), int(fmax))))
        mne_psds.save(written_files[-1])

        med_power = np.median(np.log10(mne_psds.data), 1)
        outlier_mask, left_mad, right_mad = mad_detect(med_power)

        plot_range = np.arange(len(info['ch_names']))

        outlier_label = np.array(info['ch_names'])

        fig_mad = plt.figure()
        plt.axhline(np.median(med_power), color='#225ee4')
        plt.plot(
            plot_range[~outlier_mask],
            med_power[~outlier_mask], linestyle='None', marker='o',
            color='white')
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
            fig_mad, '%s: run-%i' % (subject, run_ind + 1), 'MAD power')

        mne_med_power = mne.EvokedArray(
            data=med_power[:, None], info=deepcopy(info), tmin=0, nave=1)
        hcp.preprocessing.transform_sensors_to_mne(mne_med_power)
        fig = mne_med_power.plot_topomap(
            [0], scale=1, cmap='viridis', vmin=np.min, vmax=np.max,
            show_names=True, mask=outlier_mask[:, None], contours=1,
            time_format='', unit='dB')

        for tt in fig.findobj(matplotlib.text.Text):
            if tt.get_text().startswith('A'):
                tt.set_color('red')

        fig.set_dpi(dpi)

        if report is not None:
            report.add_figs_to_section(
                fig, '%s: run-%i' % (subject, run_ind + 1), 'topo')

        results.append({'subject': subject, 'run': run_ind,
                        'labels': outlier_label[outlier_mask]})

        fig_log = plot_loglog(
            mne_psds.data, freqs[(freqs >= fmin) & (freqs <= fmax)])
        fig_log.set_dpi(dpi)
        if report is not None:
            report.add_figs_to_section(
                fig_log, '%s: run-%i' % (subject, run_ind + 1), 'loglog')
        plt.close('all')

    if run_id is not None and results_dir is not None:
        df = pd.DataFrame(results)
        written_files.append(
            op.join(results_dir, run_id, 'bads_by_power.csv'))
        df.to_csv(written_files[-1])

    return written_files
