import os.path as op

import numpy as np

import mne

import hcp
from hcp.preprocessing import apply_ica_hcp


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
        this_raw.info['bads'] = annots['channels']['all']
        this_raw.pick_types(meg='mag', ref_meg=False)
        if len(exclude) > 0:
            apply_ica_hcp(this_raw, ica_mat=ica_mat, exclude=exclude)
    projs_noise = mne.compute_proj_raw(raw_noise, n_mag=n_ssp)
    raw.add_proj(projs_noise)
    raw.filter(fmin, fmax, n_jobs=n_jobs, method='iir',
               iir_params=dict(order=4, ftype='butter'))
    return raw
