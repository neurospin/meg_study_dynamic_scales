import os.path as op

import numpy as np

import mne

import hcp
from hcp.preprocessing import apply_ica_hcp, interpolate_missing_channels


def hcp_preprocess_ssp_ica(subject, run_index, recordings_path,
                           fmin=None, fmax=200, hcp_path=op.curdir, n_jobs=1,
                           n_ssp=12):

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
        this_raw.pick_types(meg='mag', ref_meg=False)
        this_raw.info['bads'] = annots['channels']['all']
        if len(exclude) > 0:
            apply_ica_hcp(this_raw, ica_mat=ica_mat, exclude=exclude)
    projs_noise = mne.compute_proj_raw(raw_noise, n_mag=n_ssp)
    raw.add_proj(projs_noise)
    raw.filter(fmin, fmax, n_jobs=n_jobs, method='iir',
               iir_params=dict(order=4, ftype='butter'))
    interpolate_missing_channels(raw, subject=subject, data_type='rest',
                                 hcp_path=hcp_path)
    return raw


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


def hcp_compute_noise_cov(subject, recordings_path,
                          hcp_path=op.curdir, n_jobs=1,
                          filter_freq_ranges=tuple((None, None)),
                          method=('shrunk', 'empirical'),
                          n_ssp=12):

    raw_noise = hcp.io.read_raw_hcp(
        subject, hcp_path=hcp_path, data_type='noise_empty_room',
        run_index=0)

    if n_ssp:
        projs_noise = mne.compute_proj_raw(raw_noise, n_mag=n_ssp)
        raw_noise.add_proj(projs_noise)
    out = list()
    for fmin, fmax in filter_freq_ranges:
        raw_noise_ = raw_noise.copy()
        hcp_band_pass(raw_noise_, fmin, fmax, notch=True, n_jobs=n_jobs)
        noise_cov = mne.compute_raw_covariance(
            raw_noise_, tstep=10, reject=dict(mag=5e-12), method=method)
        out.append((fmin, fmax, noise_cov, raw_noise_.info))
    return out
