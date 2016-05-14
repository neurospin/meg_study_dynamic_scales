import glob
import os
import os.path as op
from copy import deepcopy

import numpy as np
import pandas as pd

import mne
from mne.time_frequency import psd_multitaper
import hcp
from hcp.workflows.anatomy import make_mne_anatomy
from hcp import io
from hcp.preprocessing import set_eog_ecg_channels
from .hcp_utils import hcp_preprocess_ssp_ica
from .hcp_utils import hcp_compute_noise_cov

from .utils import mad_detect
from .event import make_overlapping_events
from .viz import plot_loglog
from .stats import compute_log_linear_fit
from mne.externals.h5io import write_hdf5
from .downloaders import get_single_trial_source_psd


def dummy(subject):
    print(subject)
    return list()


def compute_alpha_fluctuations(
        subject, recordings_path, alpha_peak, fmin=None, fmax=200,
        hcp_path=op.curdir, n_ssp=16, decim=1,
        report=None, dpi=600, n_jobs=1, results_dir=None, run_inds=(0, 1, 2),
        run_id=None, verbose=None):
    for run_index in run_inds:
        raw = hcp_preprocess_ssp_ica(
            subject=subject, run_index=run_index,
            recordings_path=recordings_path,
            hcp_path=hcp_path, fmin=fmin, fmax=fmax, n_jobs=n_jobs,
            n_ssp=n_ssp)
        pass


def compute_ecg_events(
        subject, recordings_path, hcp_path, results_dir=None,
        run_inds=(0, 1, 2), run_id=None, report=None):

    written_files = list()
    for run_index in run_inds:
        raw = io.read_raw_hcp(subject=subject, data_type='rest',
                              run_index=run_index, hcp_path=hcp_path)
        set_eog_ecg_channels(raw)
        raw.pick_channels(['ECG'])
        assert len(raw.ch_names) == 1
        if report is not None:
            fig = raw.plot()
            report.add_figs_to_section(fig, 'ECG', subject)
        written_files.append(op.join(recordings_path, subject,
                                     'ecg-r%i-raw.fif' % run_index))
        raw.save(written_files[-1])
        events, _, _ = mne.preprocessing.find_ecg_events(raw)
        written_files.append(op.join(recordings_path, subject,
                             'ecg-r%i-eve.fif' % run_index))
        mne.write_events(written_files[-1], events)

    return written_files


def _get_epochs_for_subject(subject, recordings_path, pattern):
    epochs_list = list()
    for run_index in range(2):
        psd_fname = op.join(recordings_path, subject,
                            pattern.format(run=run_index))
        if op.isfile(psd_fname):
            epochs = mne.read_epochs(psd_fname)
            if epochs.info['nchan'] == 248:
                epochs_list.append(epochs)
    if epochs_list:
        epochs = mne.epochs.concatenate_epochs(epochs_list)
    else:
        epochs = None
    return epochs


def compute_log_linear_fit_epochs(
        subject, recordings_path, run_id=None,
        pattern='psds-r{run}-0-150-epo.fif',
        pattern_times='100307/psds-r0-0-150-times.npy',
        sfmin=0.1, sfmax=1, n_jobs=1, log_fun=np.log10):
    written_files = list()
    epochs = _get_epochs_for_subject(
        subject=subject, recordings_path=recordings_path, pattern=pattern)

    freqs = np.load(op.join(recordings_path, pattern_times))
    out = dict(info=epochs.info.to_dict())
    fun = compute_log_linear_fit
    out['coefs'], out['intercepts'], out['msq'], out['r2'] = fun(
        epochs.get_data(), freqs=freqs, sfmin=sfmin, sfmax=sfmax,
        log_fun=log_fun)
    written_files.append(
        op.join(recordings_path, subject,
                'psds-loglinear-fit-{}-{}.h5'.format(
                    str(sfmin).replace('.', 'p'),
                    str(sfmax).replace('.', 'p'))))
    write_hdf5(written_files[-1], out, overwrite=True)
    return written_files


def compute_source_power_spectra(
        subject, recordings_path,
        events_duration=30, events_overlap=None,
        fmin=None, fmax=200,
        hcp_path=op.curdir, n_ssp=16, decim=1, mt_bandwidth=4, duration=30,
        report=None, dpi=600, n_jobs=1, results_dir=None, run_inds=(0, 1, 2),
        run_id=None, out='epochs', spacing='oct3', add_dist=True,
        surface='white', covariance_fmin=None, covariance_fmax=None,
        lambda2=1. / 3. ** 2, method='MNE',
        pick_ori=None, label=None,
        nave=1, pca=True, adaptive=False,
        inv_split=1,
        compute_sensor_psds=False,
        low_bias=True, prepared=False, verbose=None):

    written_files = list()
    inv_fname = op.join(recordings_path, subject, (
        '{spacing}_{surface}_dist-{add_dist}-{fmin}-{fmax}-inv.fif'.format(
            surface='white', spacing=spacing, add_dist=add_dist,
            fmin=covariance_fmin, fmax=covariance_fmax).lower()))

    inverse_operator = mne.minimum_norm.prepare_inverse_operator(
        orig=mne.minimum_norm.read_inverse_operator(inv_fname), nave=nave,
        lambda2=lambda2, method=method)

    for run_ii, run_index in enumerate(run_inds):
        raw = hcp_preprocess_ssp_ica(
            subject=subject, run_index=run_index,
            recordings_path=recordings_path,
            decim=decim,
            hcp_path=hcp_path, fmin=fmin, fmax=fmax, n_jobs=n_jobs,
            n_ssp=n_ssp, return_noise=False)

        epochs = get_epochs(raw, events_duration=events_duration,
                            events_overlap=events_overlap)

        written_files.extend(
            _compute_source_psd(
                epochs=epochs, method=method, run_ii=run_ii,
                fmax=fmax, fmin=fmin, pick_ori=pick_ori, label=label,
                pca=pca, inv_split=inv_split,
                mt_bandwidth=mt_bandwidth, adaptive=adaptive,
                low_bias=low_bias,
                inverse_operator=inverse_operator,
                run_index=run_index,
                n_jobs=n_jobs, recordings_path=recordings_path,
                spacing=spacing,
                subject=subject, run_id=run_id)
        )
        if compute_sensor_psds:
            mne_psds, freqs = _psd_epochs_sensor_space(
                epochs=epochs, fmin=fmin, fmax=fmax, mt_bandwidth=mt_bandwidth,
                n_jobs=n_jobs)
            written_files.append(
                op.join(recordings_path, subject, 'psds-r%i-%i-%i-%s.fif' % (
                    run_index, int(0 if fmin is None else fmin), int(fmax),
                    'epo')))
    return written_files


def _compute_source_psd(epochs, method, fmax, fmin, pick_ori,
                        label, pca, inv_split, inverse_operator, run_ii,
                        mt_bandwidth, adaptive, low_bias, run_index,
                        spacing,
                        n_jobs, recordings_path, subject, run_id):

    gen_stc_epochs = mne.minimum_norm.compute_source_psd_epochs(
        epochs, inverse_operator,
        fmin=(0 if fmin is None else fmin),
        method=method,
        fmax=(np.inf if fmax is None else fmax), pick_ori=pick_ori,
        label=label, pca=pca, inv_split=inv_split, bandwidth=mt_bandwidth,
        adaptive=adaptive, low_bias=low_bias, return_generator=True,
        n_jobs=n_jobs, prepared=True)
    written_files = list()
    for ii, stc in enumerate(gen_stc_epochs):
        print('stc epoch %s run %s' % (ii, run_index))
        written_files.append(
            op.join(recordings_path, subject,
                    ('psds-e%s-r%s-%s-%s-%s' % (
                        ii, run_index, int(0 if fmin is None else fmin),
                        int(fmax), spacing)).lower()))
        stc.save(written_files[-1])
        written_files[-1] += '-lh.stc'
        written_files.append(written_files[-1].replace('-lh.stc', '-rh.stc'))
    return written_files


def _psd_average_sensor_space(epochs, fmin, fmax, mt_bandwidth, n_jobs):

    X_psds = 0.0
    for ii in range(len(epochs.events)):
        psds, freqs = psd_multitaper(
            epochs[ii], fmin=fmin, fmax=fmax, bandwidth=mt_bandwidth,
            n_jobs=n_jobs)
        X_psds += psds
    X_psds /= (ii + 1)
    X_psds = X_psds[0]
    mne_psds = mne.EvokedArray(
        data=X_psds, info=deepcopy(epochs.info), tmin=0, nave=1)
    hcp.preprocessing.transform_sensors_to_mne(mne_psds)
    return mne_psds, freqs


def _psd_epochs_sensor_space(epochs, fmin, fmax, mt_bandwidth, n_jobs):

    X_psds, freqs = psd_multitaper(
        epochs, fmin=fmin, fmax=fmax, bandwidth=mt_bandwidth, n_jobs=1)

    info = deepcopy(epochs.info)
    info['projs'] = list()
    events = epochs.events.copy()
    del epochs
    mne_psds = mne.EpochsArray(data=X_psds, info=info, events=events, tmin=0)
    hcp.preprocessing.transform_sensors_to_mne(mne_psds)
    return mne_psds, freqs


def get_epochs(raw, events_duration, events_overlap=None,
               reject_mag=5e-12):
    if events_overlap is None:
        events = mne.make_fixed_length_events(
            raw, 3000, duration=events_duration)
    else:
        events = make_overlapping_events(
            raw, 3000, stop=raw.times[raw.last_samp],
            overlap=events_overlap, duration=events_duration)
    picks = mne.pick_types(raw.info, meg=True, ref_meg=False)
    epochs = mne.Epochs(
        raw, picks=picks, events=events, event_id=3000, tmin=0,
        tmax=events_duration,
        detrend=1, baseline=None, decim=1,
        reject=dict(mag=reject_mag), preload=False, proj=True)
    return epochs


def compute_power_spectra(
        subject, recordings_path, fmin=None, fmax=200,
        hcp_path=op.curdir, n_ssp=16, decim=1, mt_bandwidth=4,
        report=None, dpi=600, n_jobs=1, results_dir=None, run_inds=(0, 1, 2),
        events_duration=30, events_overlap=None,
        run_id=None, out='average'):
    if out == 'average':
        fun = _psd_average_sensor_space
        file_tag = 'ave'
    elif out == 'epochs':
        fun = _psd_epochs_sensor_space
        file_tag = 'epo'
    else:
        raise ValueError('"out" must be "epochs" or "average"')

    written_files = list()
    for run_ii, run_index in enumerate(run_inds):

        raw = hcp_preprocess_ssp_ica(
            subject=subject, run_index=run_index,
            recordings_path=recordings_path,
            decim=decim,
            hcp_path=hcp_path, fmin=fmin, fmax=fmax, n_jobs=n_jobs,
            n_ssp=n_ssp)

        epochs = get_epochs(raw, events_duration=events_duration,
                            events_overlap=events_overlap)

        mne_psds, freqs = fun(epochs=epochs, fmin=fmin, fmax=fmax,
                              mt_bandwidth=mt_bandwidth, n_jobs=n_jobs)
        if run_ii == 0:
            written_files.append(
                op.join(recordings_path, subject,
                        'psds-r%i-%i-%i-times.npy' % (
                            run_index, int(0 if fmin is None else fmin),
                            int(fmax))))
            np.save(written_files[-1], freqs)

        written_files.append(
            op.join(recordings_path, subject, 'psds-r%i-%i-%i-%s.fif' % (
                run_index, int(0 if fmin is None else fmin), int(fmax),
                file_tag)))
        mne_psds.save(written_files[-1])
        fig_log = plot_loglog(
            mne_psds.average().data if out == 'epochs' else mne_psds.data,
            freqs[(freqs >= fmin) & (freqs <= fmax)],
            xticks=(0.1, 1, 10, 100))

        fig_log.set_dpi(dpi)
        if report is not None:
            report.add_figs_to_section(
                fig_log, '%s: run-%i' % (subject, run_index + 1), 'loglog')

    return written_files


def average_power_spectra(subject, recordings_path, fmin=None, fmax=200,
                          results_dir=None,
                          run_inds=(0, 1, 2), run_id=None):
    written_files = list()
    for run_index in run_inds:
        fname = op.join(recordings_path, subject, 'psds-r%i-%i-%i-epo.fif' % (
                        run_index, fmin, fmax))
        written_files.append(
            fname.replace('psds-r', 'psds-ave-r').replace('epo', 'ave'))
        mne.read_epochs(fname).average().save(written_files[-1])
    return written_files


def compute_power_spectra_and_bads(
        subject, recordings_path, fmin=None, fmax=200,
        hcp_path=op.curdir, n_ssp=16, decim=16,  mt_bandwidth=4, duration=1,
        report=None, dpi=600, n_jobs=1, run_inds=(0, 1, 2), results_dir=None,
        run_id=None):

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    written_files = list()
    results = list()
    if not isinstance(run_inds, (list, tuple)):
        run_inds = [run_inds]

    for run_ind in run_inds:
        mne_psds, freqs = _psd_average_sensor_space(
            subject=subject, run_index=run_ind,
            recordings_path=recordings_path, fmin=fmin, fmax=fmax,
            hcp_path=hcp_path, n_ssp=n_ssp, decim=decim,
            mt_bandwidth=mt_bandwidth,
            duration=duration, n_jobs=n_jobs)
        info = mne_psds.info
        written_files.append(
            op.join(recordings_path, subject, 'psds-bads-r%i-%i-%i-ave.fif' % (
                run_ind, int(fmin), int(fmax))))
        mne_psds.save(written_files[-1])
        written_files.append(
            op.join(recordings_path, subject,
                    'psds-bads-r%i-%i-%i-times.npy' % (
                        run_ind, int(fmin), int(fmax))))
        np.save(written_files[-1], freqs)
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


def compute_covariance(
        subject, recordings_path, filter_freq_ranges=None,
        report=None, hcp_path=op.curdir, inverse_params=None, decim=1,
        n_jobs=1, n_ssp=16, results_dir=None, run_id=None):
    if filter_freq_ranges is None:
        filter_freq_ranges = ((None, None),)
    elif filter_freq_ranges == 'full':
        filter_freq_ranges = (
            (None, None), (None, 3), (4, 7), (8, 12), (13, 20),
            (20, 30), (30, 60), (60, 120))

    kwargs = dict(subject=subject, recordings_path=recordings_path,
                  hcp_path=hcp_path,
                  filter_freq_ranges=filter_freq_ranges,
                  decim=decim,
                  n_jobs=n_jobs, n_ssp=n_ssp)
    written_files = list()
    for fmin, fmax, noise_cov, info in hcp_compute_noise_cov(**kwargs):
        if report is not None:
            report.add_figs_to_section(
                mne.viz.plot_cov(info=info, cov=noise_cov),
                ['cov', 'eig'], 'subject-%s-%s' % (fmin, fmax))
        written_files.append(
            op.join(recordings_path, subject, 'noise-%s-%s-cov.fif' % (
                    fmin, fmax)).lower())
        noise_cov.save(written_files[-1])
    return written_files


def compute_inverse_solution(subject, recordings_path, spacings=None,
                             results_dir=None,
                             filter_freq_ranges=None, inverse_params=None,
                             hcp_path=op.curdir,
                             run_id=None):
    if filter_freq_ranges is None:
        filter_freq_ranges = ((None, None),)
    elif filter_freq_ranges == 'full':
        filter_freq_ranges = (
            (None, None), (None, 3), (4, 7), (8, 12), (13, 20),
            (20, 30), (30, 60), (60, 120))
    if spacings is None:
        spacings = ['ico2', 'oct3', 'oct4', 'oct5', 'oct5', 'oct6']
    if inverse_params is None:
        inverse_params = dict(
            loose=0.2, depth=0.8, fixed=False, limit_depth_chs=True)
    inv_fname_tmp = '{spacing}_{surface}_dist-{add_dist}-{fmin}-{fmax}-inv.fif'
    written_files = list()
    info = hcp.io.read_info_hcp(
        subject, data_type='rest', run_index=0, hcp_path=hcp_path)
    for this_spacing in spacings:
        src_params = dict(spacing=this_spacing, surface='white',
                          add_dist=True)
        fwd_fname = '{spacing}_{surface}_dist-{add_dist}-fwd.fif'.format(
            **src_params).lower()
        forward = mne.read_forward_solution(
            op.join(recordings_path, subject, fwd_fname))
        for fmin, fmax in filter_freq_ranges:
            inv_params = deepcopy(src_params)
            inv_params.update(fmin=fmin, fmax=fmax)
            noise_cov = mne.read_cov(
                op.join(recordings_path, subject, 'noise-%s-%s-cov.fif' % (
                    fmin, fmax)).lower())

            inv_fname = inv_fname_tmp.format(**inv_params).lower()
            inverse_operator = mne.minimum_norm.make_inverse_operator(
                info=info, forward=forward, noise_cov=noise_cov,
                **inverse_params)
            written_files.append(
                op.join(recordings_path, subject, inv_fname))
            mne.minimum_norm.write_inverse_operator(
                written_files[-1], inverse_operator)
    return written_files


def compute_covariance_and_inverse(
        subject, recordings_path, filter_freq_ranges=None,
        report=None, hcp_path=op.curdir, spacings=None,
        inverse_params=None, decim=1,
        n_jobs=1, n_ssp=16, results_dir=None, run_id=None):
    written_files = list()
    written_files.extend(
        compute_covariance(
            subject=subject, recordings_path=recordings_path,
            filter_freq_ranges=filter_freq_ranges,
            report=report, hcp_path=hcp_path, decim=decim,
            n_jobs=n_jobs, n_ssp=n_ssp, results_dir=results_dir,
            run_id=run_id))
    written_files.extend(
        compute_inverse_solution(
            subject=subject, recordings_path=recordings_path,
            spacings=spacings, results_dir=results_dir,
            filter_freq_ranges=filter_freq_ranges,
            inverse_params=inverse_params,
            hcp_path=hcp_path, run_id=run_id)
    )
    return written_files


def get_brodmann_labels(spacing, subjects_dir):
    src_orig = mne.setup_source_space(
        subject='fsaverage', subjects_dir=subjects_dir, fname=None,
        spacing=spacing, add_dist=False)

    rh_brodman = op.join(
        subjects_dir, 'fsaverage', 'label', 'rh.PALS_B12_Brodmann.annot')
    lh_brodman = op.join(
        subjects_dir, 'fsaverage', 'label', 'lh.PALS_B12_Brodmann.annot')

    labels = list()
    labels += mne.read_labels_from_annot(
        subject='fsaverage', subjects_dir=subjects_dir,
        annot_fname=lh_brodman, regexp="Brodmann*")
    labels += mne.read_labels_from_annot(
        subject='fsaverage', subjects_dir=subjects_dir,
        annot_fname=rh_brodman, regexp="Brodmann*")

    out_labels = list()
    for label in labels:
        label.values.fill(1.0)
        out_labels.append(
            label.morph('fsaverage', 'fsaverage', copy=False,
                        subjects_dir=subjects_dir,
                        grade=[ss['vertno'] for ss in src_orig]))
    written_files = [
        op.join(subjects_dir, 'fsaverage', 'label',
                label.name.replace('Brodmann', 'Brodmann-%s' % spacing)) for
        label in out_labels]
    for label, fname in zip(out_labels, written_files):
        label.save(fname)
    return written_files


def _morph_stcs_in_fsaverage(epochs, subjects_dir, src_orig, inverse_operator,
                             method, lambda2=1./9., prepared=True):
    stcs = mne.minimum_norm.apply_inverse_epochs(
        epochs=epochs, inverse_operator=inverse_operator, lambda2=lambda2,
        return_generator=True, prepared=prepared, method=method)
    for stc in stcs:
        yield stc.to_original_src(src_orig, subject_orig='fsaverage',
                                  subjects_dir=subjects_dir)


def compute_source_outputs(subject, recordings_path, anatomy_path,
                           hcp_path=op.curdir,
                           fmin=0, fmax=150,
                           spacing='oct5',
                           subjects_dir='/home/ubuntu/freesurfer/subjects',
                           debug=False):

    make_mne_anatomy(subject=subject, anatomy_path=anatomy_path,
                     recordings_path=recordings_path, hcp_path=hcp_path)
    # just use this as reference subject
    freqs = np.load(
        op.join(recordings_path, '100307', 'psds-r0-{}-{}-times.npy'.format(
            fmin, fmax)))
    if not op.exists(anatomy_path):
        os.mkdir(anatomy_path)

    if not op.exists(anatomy_path + '/fsaverage'):
        os.symlink(subjects_dir + '/fsaverage',
                   anatomy_path + '/fsaverage')

    src_orig = mne.setup_source_space(
        subject='fsaverage', fname=None, spacing=spacing, add_dist=False,
        subjects_dir=anatomy_path)

    stc_files = dict(r0=list(), r1=list(), r2=list())
    i_find = 0
    for fname in glob.glob(op.join(recordings_path, subject, '*stc')):
        if i_find >= 3 and debug is True:
            break
        for pattern in get_single_trial_source_psd(subject)['key_list']:
            if i_find >= 3 and debug is True:
                break
            if glob.fnmatch.fnmatch(fname, '*' + pattern):
                if fname.endswith('lh.stc'):
                    if 'r1' in fname:
                        key = 'r1'
                    elif 'r2' in fname:
                        key = 'r2'
                    else:
                        key = 'r0'
                    stc_files[key].append(fname)
                    i_find += 1
    tmp = 'Brodmann-{spacing}.{num}-{hemi}.label'
    brodmann_label_names = [tmp.format(spacing=spacing, num=num, hemi=hemi)
                            for num in range(1, 48, 1) for hemi in ('lh', 'rh')
                            if num not in [12, 13, 14, 15, 16, 34]]
    labels = [mne.read_label(
              op.join(anatomy_path, 'fsaverage', 'label', fname))
              for fname in brodmann_label_names]

    X = 0.
    label_tcs = list()
    stc = None
    for ii, fname in enumerate(sum(stc_files.values(), []), 1):
        stc = mne.read_source_estimate(fname)
        if stc.data.shape[0] != sum(len(sp['vertno']) for sp in src_orig):
            continue
        stc.subject = subject
        stc = stc.to_original_src(
            src_orig=src_orig, subject_orig='fsaverage',
            subjects_dir=anatomy_path)
        stc.subject = 'fsaverage'
        X += np.log10(stc.data)
        label_tcs.append(
            np.array([stc.extract_label_time_course(label,
                                                    src_orig,
                                                    mode='mean')
                      for label in labels]))
    label_tcs = np.array(label_tcs)

    if stc is None:
        raise RuntimeError('could not find stc files for %s' % subject)
    mean_power_stc = stc.copy()
    mean_power_stc._data = X
    mean_power_stc._data /= ii
    mean_power_stc.times[:] = freqs

    def stc_gen(stc_files):
        for ii, fname in enumerate(sum(stc_files.values(), [])):
            stc = mne.read_source_estimate(fname)
            stc.subject = fname.split('/')[-2]
            stc = stc.to_original_src(
                src_orig=src_orig, subject_orig='fsaverage',
                subjects_dir=anatomy_path)
            stc.subject = 'fsaverage'
            yield stc.data

    coefs_, _, mse_, _ = compute_log_linear_fit(
        stc_gen(stc_files), freqs=freqs, sfmin=0.1, sfmax=1.,
        log_fun=np.log10)

    mean_coefs_stc = stc.copy()
    mean_coefs_stc._data = coefs_.T
    mean_coefs_stc.times = np.arange(len(coefs_))
    mean_mse_stc = stc.copy()
    mean_mse_stc._data = mse_.T
    mean_mse_stc.times = np.arange(len(mse_))
    written_files = list()
    for stc, kind in zip((mean_power_stc, mean_coefs_stc, mean_mse_stc),
                         ('power', 'coefs', 'mse')):
        written_files.append(
            op.join(recordings_path, subject,
                    '{}-{}-{}'.format(kind, fmin, fmax)))
        stc.save(written_files[-1])
        written_files[-1] += '-lh.stc'
        written_files.append(written_files[-1].replace('-lh', '-rh'))
    written_files.append(
        op.join(recordings_path, subject,
                r'{}-{}-{}_label_tcs.npy'.format('power', fmin, fmax)))
    np.save(written_files[-1], label_tcs)
    return written_files
