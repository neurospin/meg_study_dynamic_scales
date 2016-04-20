import glob
import os.path as op

import numpy as np
import pandas as pd
from scipy.signal import detrend, argrelmax
from scipy.stats import kendalltau
import mne
import matplotlib.pyplot as plt

from meeg_preprocessing.utils import (
    setup_provenance)
# configure logging + provenance tracking magic
report, run_id, results_dir, logger = setup_provenance(
    script=__file__, results_dir='results')

recordings_path = '/Volumes/MEG-HUB/HCP-MEG/'

psds_files = glob.glob(op.join(recordings_path, '*',
                               'psds-bads-r?-1-50-ave.fif'))
df = pd.read_csv('hcp_unrestricted_data.csv')
hcp_alpha_peaks = dict(
    (str(int(k)), v) for k, v in zip(*df[['Subject', 'Alpha_Peak']].values.T))

my_alpha_peaks = list()
times = np.load(psds_files[0].replace('-ave.fif', '-times.npy'))

for fname in psds_files:
    subject = fname.split('/')[-2]
    alpha_peak = hcp_alpha_peaks[subject]
    this_run = fname.split('/')[-1].split('-')[2]
    psd_ave = mne.read_evokeds(fname)[0]
    log_detrend = detrend(np.log10(psd_ave.data))
    # log_detrend = np.log10(psd_ave.data)
    tmask = (times >= 5) & (times <= 16)
    search_times = times[tmask]
    argmaxima = argrelmax(log_detrend[:, tmask].max(0))[0]
    my_alpha_peak = search_times[argmaxima]
    my_alpha_peak = my_alpha_peak[
        (my_alpha_peak >= 8) & (my_alpha_peak <= 13)]
    if len(my_alpha_peak) == 0:
        print('skipping %s' % subject)
        continue
    fig = plt.figure()
    plt.plot(
        # times, np.max(10 ** log_detrend, axis=0))
        times, np.log10(psd_ave.data.T))
        # times, np.max(log_detrend, axis=0))
    plt.axvline(alpha_peak, color='red', label='HCP=%0.3f' % alpha_peak)
    for pp in my_alpha_peak:
        plt.axvline(pp, color='violet',
                    label='MNE-HCP=%0.3f' % pp)
    plt.legend()
    report.add_figs_to_section(
        fig, 'alpha-%s' % this_run, subject
    )
    my_alpha_peaks.append(dict(subject=subject, alpha=my_alpha_peak[0]))
    plt.close('all')

my_alpha_peaks = pd.DataFrame(my_alpha_peaks)
my_alpha_peaks['hcp_alpha'] = None
for subject, alpha in hcp_alpha_peaks.items():
    if subject in my_alpha_peaks['subject'].values:
        mask = (my_alpha_peaks['subject'] == subject)
        my_alpha_peaks.loc[mask, 'hcp_alpha'] = alpha

        print(kendalltau(np.nan_to_num(list(my_alpha_peaks['alpha'])),
      np.nan_to_num(list(my_alpha_peaks['hcp_alpha']))))

report.save(op.join(results_dir, run_id, 'report.html'))
