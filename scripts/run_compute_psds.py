# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os.path as op
import glob
import numpy as np

from meeg_preprocessing.utils import (
    get_data_picks, setup_provenance, handle_mkl)

from mne.time_frequency import multitaper_psd

from library.externals import h5io
import mne
from mne import io
from mne.preprocessing import read_ica
import config as cfg

args = cfg.get_argparse().parse_known_args()[0]
subjects = [args.subject] if args.subject is not None else cfg.subjects
mkl_max_threads = (args.mkl_max_threads if args.mkl_max_threads is not None
                   else cfg.mkl_max_threads)
handle_mkl(max_threads=mkl_max_threads)

# configure logging + provenance tracking magic
report, run_id, results_dir, logger = setup_provenance(
    script=locals().get('__file__', op.curdir), results_dir='results')


for subject in subjects:
    report = mne.report.Report(title=subject)

    # read files
    raw_fnames = glob.glob(op.join(
        cfg.recordings_path, subject, cfg.sss_raw_name_tmp.format(
            group='*', run='*')))

    written_fields = list()
    for i_run, fname in enumerate(raw_fnames):
        raw = io.Raw(fname, preload=True)
        raw.resample(250., n_jobs=4)
        for picks, ch_type in get_data_picks(
                raw, meg_combined=cfg.ica_meg_combined):
            this_ica_fname = op.join(
                cfg.current_ica_solution_path,
                cfg.ica_fname_tmp.format(subject=subject, ch_type=ch_type))
            ica = read_ica(this_ica_fname)
            ica.apply(raw)
        fwd = mne.read_forward_solution(
            op.join(cfg.recordings_path, subject, cfg.fwd_fname))
        noise_cov = mne.read_cov(
            op.join(cfg.recordings_path, subject, cfg.noise_cov_fname))

        events = mne.make_fixed_length_events(raw, 3000, duration=28)

        epochs = mne.Epochs(raw, events=events, event_id=3000, tmin=0, tmax=28)
        picks = mne.pick_types(raw.info, meg=True)

        X_psd = None
        X_psds = list()

        for i_epoch, epoch in enumerate(epochs):
            this_psd, freqs = multitaper_psd(epoch[picks],
                                             sfreq=epochs.info['sfreq'],
                                             fmin=cfg.fmin,
                                             fmax=cfg.fmax, bandwidth=0.5)
            X_psds.append(this_psd)

        X_psds = np.array(X_psds)

        h5io.write_hdf5(
            op.join(results_dir, run_id, '%s-results.hdf5' % subject),
            X_psds,
            title='dyn/psd/run%i' % i_run, overwrite='update')

        h5io.write_hdf5(
            op.join(results_dir, run_id, '%s-results.hdf5' % subject),
            X_psds,
            title='dyn/freqs', overwrite='update')
