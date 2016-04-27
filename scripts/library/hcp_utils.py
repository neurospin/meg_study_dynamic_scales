import os.path as op

import numpy as np

import mne

import hcp
from hcp.preprocessing import apply_ica_hcp, interpolate_missing_channels

from .raw import decimate_raw


def hcp_preprocess_ssp_ica(subject, run_index, recordings_path, decim=1,
                           fmin=None, fmax=200, hcp_path=op.curdir,
                           n_jobs=1,
                           n_ssp=16, return_noise=False):

    raws = dict()
    ica_mat = None
    annots = None
    if not return_noise:
        annots = hcp.io.read_annot_hcp(
            subject, hcp_path=hcp_path, data_type='rest',
            run_index=run_index)
        ica_mat = hcp.io.read_ica_hcp(
            subject, hcp_path=hcp_path, data_type='rest',
            run_index=run_index)
        raws['raw'] = hcp.io.read_raw_hcp(
            subject, hcp_path=hcp_path, data_type='rest',
            run_index=run_index)
    raws['noise'] = hcp.io.read_raw_hcp(
        subject, hcp_path=hcp_path, data_type='noise_empty_room',
        run_index=0)
    exclude = list()
    if ica_mat is not None:
        exclude = np.array(annots['ica']['ecg_eog_ic']) - 1
    for this_raw in raws.values():
        decimate_raw(this_raw, decim=decim)
        if annots is not None:
            this_raw.info['bads'] = annots['channels']['all']

        this_raw.pick_types(meg='mag', ref_meg=False)
        this_raw.filter(fmin, fmax, n_jobs=n_jobs, method='iir',
                        iir_params=dict(order=4, ftype='butter'))
        if len(exclude) > 0:
            print('applying ICA')
            apply_ica_hcp(this_raw, ica_mat=ica_mat, exclude=exclude)

    projs_noise = mne.compute_proj_raw(raws['noise'], n_mag=n_ssp)
    if return_noise:
        raw = raws['noise']
    else:
        raw = raws['raw']

    raw.add_proj(projs_noise)
    raw.apply_proj()
    if not return_noise:
        raw = interpolate_missing_channels(
            raw, subject=subject, run_index=run_index, data_type='rest',
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
                          decim=1,
                          method=('shrunk', 'empirical'),
                          n_ssp=16):

    raw_noise = hcp_preprocess_ssp_ica(
        subject=subject, run_index=0, recordings_path=recordings_path,
        decim=decim, hcp_path=hcp_path, n_jobs=n_jobs,
        n_ssp=n_ssp, return_noise=True)
    if decim > 1:
        decimate_raw(raw_noise, decim=decim)
    out = list()
    for fmin, fmax in filter_freq_ranges:
        raw_noise_ = raw_noise.copy()
        hcp_band_pass(raw_noise_, fmin, fmax, notch=False, n_jobs=n_jobs)
        noise_cov = mne.compute_raw_covariance(
            raw_noise_, tstep=10, reject=dict(mag=5e-12), method=method)
        out.append((fmin, fmax, noise_cov, raw_noise_.info))
    return out
