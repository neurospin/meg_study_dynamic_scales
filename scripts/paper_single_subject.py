import os
import os.path as op

import numpy as np

import mne
import hcp

from hcp import preprocessing as preproc
from mne.time_frequency.psd import _psd_welch

from mne.minimum_norm import apply_inverse_epochs, make_inverse_operator

import sys


def clean_psd(psd, freqs, normalize=True, dB=20):
    """Crop zero frequency and noamrliaze to first freq"""
    psd = psd[:, freqs > 0]
    freqs = freqs[freqs > 0]
    # inplace on psd gives bug
    psd = dB * np.log10(psd / psd[:, 0:1] if normalize else psd)
    return psd, freqs


def _preprocess_raw(raw, hcp_params, ica_sel):
    # construct MNE annotations
    annots = hcp.read_annot(**hcp_params)
    bad_seg = (annots['segments']['all']) / raw.info['sfreq']
    annotations = mne.Annotations(
        bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
        description='bad')

    raw.annotations = annotations
    raw.info['bads'] += annots['channels']['all']
    raw.pick_types(meg=True, ref_meg=False)

    # read ICA and remove EOG ECG or keep brain components
    ica_mat = hcp.read_ica(**hcp_params)
    if ica_sel == 'ecg_eog':
        exclude = annots['ica']['ecg_eog_ic']
    elif ica_sel == 'brain':
        exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
                   if ii not in annots['ica']['brain_ic_vs']]

    preproc.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)
    # add back missing channels
    raw = preproc.interpolate_missing(raw, **hcp_params)
    return raw


def _compute_source_psd(epochs, noise_cov, fwd, n_fft=2 ** 15, method='MNE',
                        lambda2=1./1.**2., fmax=150):
    inv_op = make_inverse_operator(
        info=epochs.info, forward=fwd, noise_cov=noise_cov, loose=0.2,
        depth=0.8, fixed=False, limit_depth_chs=True, rank=None, verbose=None)

    stc_gen = apply_inverse_epochs(
        epochs, inv_op, lambda2=lambda2, method=method, label=None, nave=1,
        pick_ori='normal',
        return_generator=False, prepared=False, verbose=None)

    psd_src = 0.
    for ii, this_stc in enumerate(stc_gen, 1):
        psd, freqs = _psd_welch(this_stc.data, epochs.info['sfreq'], fmin=0,
                                fmax=fmax,
                                n_fft=n_fft, n_overlap=0, n_jobs=1)
        psd_src += psd
    psd_src /= ii

    stc_psd = this_stc
    stc_psd._data = psd_src
    stc_psd.times = freqs
    return stc_psd


def hcp_band_pass(
        raw, fmin, fmax, order=4, notch=True, ftype='butter', n_jobs=1):
    if notch is True:
        raw.notch_filter(
            freqs=np.arange(60, 241, 60), method='iir',
            iir_params=dict(order=order, ftype=ftype))
    raw.filter(fmin, None, n_jobs=n_jobs, method='iir',
               iir_params=dict(order=order, ftype=ftype))
    raw.filter(None, fmin, n_jobs=n_jobs, method='iir',
               iir_params=dict(order=order, ftype=ftype))


def compute_noise_cov(subject, recordings_path, data_type, hcp_path,
                      methods=('empirical',)):

    data_type = 'noise_empty_room'

    raw_noise = hcp.read_raw(
        run_index=0, subject=subject, hcp_path=hcp_path, data_type=data_type)
    raw_noise.load_data()
    preproc.apply_ref_correction(raw_noise)
    raw_noise.pick_types(meg=True, ref_meg=False)
    noise_cov = mne.compute_raw_covariance(
        raw_noise, method=list(methods), return_estimators=False)
    return noise_cov


def make_fwd_stack(subject, subjects_dir, hcp_path, recordings_path,
                   surface='white', add_dist=True,
                   src_type='subject_on_fsaverage',
                   spacings=('oct6', 'oct5', 'oct4', 'ico4')):

    hcp.make_mne_anatomy(subject=subject, subjects_dir=subjects_dir,
                         hcp_path=hcp_path,
                         recordings_path=recordings_path)

    for spacing in spacings:
        out = hcp.compute_forward_stack(
            subjects_dir=subjects_dir, subject=subject,
            recordings_path=recordings_path,
            src_params=dict(spacing=spacing,
                            add_dist=add_dist, surface=surface),
            hcp_path=hcp_path)

        mne.write_forward_solution(
            op.join(recordings_path, subject, '%s-%s-%s-%s-fwd.fif' % (
                    surface, spacing, add_dist, src_type)),
            out['fwd'], overwrite=True)

        mne.write_source_spaces(
            op.join(recordings_path, subject, '%s-%s-%s-%s-src.fif' % (
                    surface, spacing, add_dist, src_type)),
            out['src_subject'])

        fname_bem = op.join(
            subjects_dir, subject, 'bem', '%s-%i-bem.fif' % (
                subject, out['bem_sol']['solution'].shape[0]))
        mne.write_bem_solution(
            fname_bem, out['bem_sol'])


def make_psds(subject, hcp_path, recordings_path, project_path, run_inds=(0,),
              spacing='ico4', data_type='rest', decim=1, n_fft=2 ** 15):
    hcp_params = dict(subject=subject, hcp_path=hcp_path,
                      data_type='rest')

    noise_cov = compute_noise_cov(
        recordings_path=recordings_path, methods=('empirical',), **hcp_params)

    fwd_fname = op.join(
        recordings_path, subject,
        '{surface}-{spacing}-{add_dist}-{src_type}-fwd.fif'.format(
            surface='white', spacing=spacing, add_dist=True,
            src_type='subject_on_fsaverage'))
    if not op.exists(op.join(project_path, subject)):
        os.makedirs(op.join(project_path, subject))

    fwd = mne.read_forward_solution(fwd_fname)
    written_files = list()
    for run in run_inds:
        fname = op.join(recordings_path, subject,
                        'rest-run%i-preproc-raw.fif' % run)
        raw = mne.io.read_raw_fif(fname)
        raw.load_data()
        if len(raw.info['bads']) > 0:
            raw.interpolate_bads()

        epochs_psd, stc_psd = _compute_psds(
            raw, noise_cov, fwd, fmax=110, n_fft=n_fft, decim=decim, lambda2=1.)

        written_files.append(
            op.join(project_path, subject, 'psd-broad-%i-epo.fif' % run))
        epochs_psd.save(written_files[-1])

        written_files.append(
            op.join(project_path, subject, 'psd-broad-%s-spacing-%s' % (
                    run, spacing)))
        stc_psd.save(written_files[-1], ftype='h5')
        del epochs_psd, stc_psd

        raw.filter(8, 12)
        raw.apply_hilbert(picks=list(range(248)), envelope=False)

        epochs_psd_alpha, stc_psd_alpha = _compute_psds(
            raw, noise_cov, fwd, fmax=50, n_fft=n_fft, decim=decim, lambda2=1.)

        written_files.append(
            op.join(project_path, subject, 'psd-alpha-%i-epo.fif' % run))
        epochs_psd_alpha.save(written_files[-1])

        written_files.append(
            op.join(project_path, subject, 'psd-alpha-%s-spacing-%s' % (
                    run, spacing)))
        stc_psd_alpha.save(written_files[-1], ftype='h5')
    return written_files


def _compute_psds(raw, noise_cov, fwd, fmax, n_fft=2 ** 15, decim=1,
                  lambda2=1.):

    duration = n_fft * (1. / raw.info['sfreq'])
    events = mne.make_fixed_length_events(raw, 42, duration=duration)
    epochs = mne.Epochs(raw, events=events, event_id=42, tmin=0,
                        tmax=duration, baseline=None, preload=True,
                        decim=decim)
    if np.iscomplexobj(epochs.get_data()):
        epochs._data = np.abs(epochs.get_data())

    psd_epochs, freqs = mne.time_frequency.psd_welch(epochs, n_fft=n_fft,
                                                     n_overlap=0, fmax=fmax)
    stc_psd = _compute_source_psd(
        epochs, noise_cov=noise_cov, fwd=fwd, method='MNE',
        lambda2=lambda2, fmax=fmax)

    epochs._data = psd_epochs
    epochs.times = freqs
    epochs.info['description'] = 'n_fft=%i;fmax=%i' % (n_fft, fmax)
    del psd_epochs

    return epochs, stc_psd


def run_all(subject, recordings_path, hcp_path='/mnt1/HCP',
            project_path='/mnt2/dynamic-scales', run_inds=(0, 1, 2)):
    written_files = list()
    written_files += make_psds(subject, hcp_path=hcp_path,
                               project_path=project_path,
                               recordings_path=recordings_path,
                               run_inds=run_inds)
    return written_files

if __name__ == '__main__':
    print('yes')
    subject = sys.argv[-1]
    run_all(subject)
