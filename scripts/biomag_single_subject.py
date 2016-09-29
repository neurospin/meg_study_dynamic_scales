import os
import os.path as op

import mne
import hcp
import h5io
import pandas as pd

from hcp import io
from hcp import preprocessing as preproc
from hcp_central.s3_utils import upload_to_s3, get_aws_credentials

import sys


def _preprocess_raw(raw, hcp_params, ica_sel):
    preproc.apply_ref_correction(raw)

    # construct MNE annotations
    annots = io.read_annot_hcp(**hcp_params)
    bad_seg = (annots['segments']['all']) / raw.info['sfreq']
    annotations = mne.Annotations(
        bad_seg[:, 0], (bad_seg[:, 1] - bad_seg[:, 0]),
        description='bad')

    raw.annotations = annotations
    raw.info['bads'] += annots['channels']['all']
    raw.pick_types(meg=True, ref_meg=False)

    # read ICA and remove EOG ECG or keep brain components
    ica_mat = hcp.io.read_ica_hcp(**hcp_params)
    if ica_sel == 'ecg_eog':
        exclude = annots['ica']['ecg_eog_ic']
    elif ica_sel == 'brain':
        exclude = [ii for ii in range(annots['ica']['total_ic_number'][0])
                   if ii not in annots['ica']['brain_ic_vs']]
    preproc.apply_ica_hcp(raw, ica_mat=ica_mat, exclude=exclude)
    # add back missing channels
    raw = preproc.interpolate_missing(raw, **hcp_params)
    return raw


def make_psd_broadband_epochs(subject, hcp_path, project_path,
                              data_type='rest', run_inds=(0, 1, 2),
                              n_fft=2**15):

    hcp_params = dict(subject=subject, hcp_path=hcp_path,
                      data_type=data_type)
    written_files = list()
    for run_index in run_inds:
        hcp_params['run_index'] = run_index
        raw = io.read_raw_hcp(**hcp_params)
        raw.load_data()

        raw = _preprocess_raw(raw, hcp_params, ica_sel='brain')

        duration = n_fft * (1 / raw.info['sfreq'])
        events = mne.make_fixed_length_events(raw, 42, duration=duration)
        epochs = mne.Epochs(raw, events=events, event_id=42, tmin=0,
                            tmax=duration,
                            baseline=None, preload=False)
        psd, freqs = mne.time_frequency.psd_welch(
            epochs, n_fft=n_fft, fmax=150)

        out_dir = op.join(project_path, subject)
        if not op.exists(out_dir):
            os.makedirs(out_dir)
        written_files.append(
            op.join(out_dir,
                    '{}-psd-broadband-epochs-run{}.hdf5'.format(
                        data_type, run_index)))

        h5io.write_hdf5(written_files[-1],
                        dict(psd=psd, freqs=freqs),
                        overwrite=True)

    return written_files


def make_psd_alpha_epochs(subject, hcp_path, project_path,
                          data_type='rest', run_inds=(0, 1, 2), n_fft=2**13):
    hcp_params = dict(subject=subject, hcp_path=hcp_path,
                      data_type=data_type)
    
    written_files = list()
    for run_index in run_inds:
        hcp_params['run_index'] = run_index
        raw = io.read_raw_hcp(**hcp_params)
        raw.load_data()
        raw = _preprocess_raw(raw, hcp_params, ica_sel='brain')

        duration = n_fft * (1 / raw.info['sfreq'])
        events = mne.make_fixed_length_events(raw, 42, duration=duration)
        epochs = mne.Epochs(raw, events=events, event_id=42, tmin=0,
                            tmax=duration, baseline=None, preload=True)
        psd, freqs = mne.time_frequency.psd_welch(epochs, n_fft=n_fft, fmin=5,
                                                  fmax=15)
        out_dir = op.join(project_path, subject)
        if not op.exists(out_dir):
            os.makedirs(out_dir)
        written_files.append(
            op.join(out_dir,
                    '{}-psd-alpha-epochs-run{}.hdf5'.format(
                        data_type, run_index)))
        h5io.write_hdf5(written_files[-1], dict(psd=psd, freqs=freqs), overwrite=True)

    return written_files


def run_all(subject, hcp_path='/mnt1/HCP', project_path='/mnt2/dynamic-scales'):
    written_files = list()
    written_files += make_psd_alpha_epochs(
        subject, hcp_path, project_path)
    written_files += make_psd_broadband_epochs(
        subject, hcp_path, project_path)
    # aws_up_key, aws_up_secret = get_aws_credentials(
    #     op.join(op.expanduser("~"), "mne-hcp-aws", 'aws_details.csv'))
    # for fname in written_files:
    #     key = op.join('hcp-meg', '/'.join(fname.split('/')[-2:]))
    #     upload_to_s3(aws_key=aws_up_key,
    #                  aws_secret=aws_up_secret, bucket='dynamic-scales', key=key,
    #                  fname=fname)
    return written_files

if __name__ == '__main__':
    print('yes')
    subject = sys.argv[-1]
    run_all(subject)