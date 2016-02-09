import os
import os.path as op
from argparse import ArgumentParser

default_args = dict(
    description='Run script for MEG-Gaze study',
    add_args=[dict(args=['--subject'],
                   kwargs=dict(metavar='subject', nargs='?', type=str,
                               help='Subject name', default=None)),
              dict(args=['--n-jobs'],
                   kwargs=dict(metavar='n_jobs', type=int, nargs='?',
                               default=None, help='Number of CPUs')),
              dict(args=['--mkl-max-threads'],
                   kwargs=dict(metavar='mkl_max_threads',
                               type=int, nargs='?',
                               default=None, help='Number of MKL threads')),
              dict(args=['--run-id'],
                   kwargs=dict(metavar='run_id', type=str, nargs='?',
                               default=None,
                               help='Run id (default = generated)')),
              dict(args=['--noise_cov_band'],
                   kwargs=dict(metavar='noise_cov_band', type=str, nargs='+',
                               default=None,
                               help='The band-range of the nosie cov')),
              dict(args=['--input-run-id'],
                   kwargs=dict(metavar='input_run_id', type=str, nargs='?',
                               default=None,
                               help='Run id to read (default = generated)'))
              ])


def get_argparse():
    parser = ArgumentParser(description=default_args['description'])
    for args in default_args['add_args']:
        parser.add_argument(*args['args'], **args['kwargs'])
    return parser


subject_map = {
    'AV': [
        'pc_110210-110517',
        'ks_110142-110609',
        'mp_110340-110908',
        'pe_110338-110912',
        'fp_110067-110301',
        'nc_110174-110421',
        'ld_110370-110922',
        'na_110353-111006',
        'kr_080082-111011',
        'bl_110396-111025',
        'da_110453-111124',
        'mb_110421-111206'
        ],
    'AVr': [
        'ga_130053-130610',
        'jm_100042-120402',
        'cd_100449-120403',
        'ap_110299-120405',
        'ma_130185-130529',
        'rg_110386-130606',
        'mj_130216-130529',
        'bd_120417-130625',
        'mr_080072-130620',
        'sa_130042-130618',
        'ak_130184-130619',
        'jd_110235-130625'
        ],
    'V': [
        'jh_100405-110308',
        'gc_100388-110309',
        'jm_100109-110412',
        'vr_100551-110510',
        'fb_110137-110512',
        'aa_100234-110531',
        'jh_110224-110630',
        'cl_100240-110623',
        'mn_080208-110718',
        'in_110286-110719',
        'cm_110222-110809',
        'tl_110313-110906'
        ]}

subjects = list()
for v in subject_map.values():
    subjects.extend(v)

subjects_to_group = dict()
for group, subs in subject_map.items():
    for sub in subs:
        subjects_to_group[sub] = group


###############################################################################
# FILE templates and variables

data_path = 'data'
recordings_path = data_path + '/' + 'MEG'
er_path = data_path + '/' + 'MEG-ER'
anatomy_path = data_path + '/' + 'subjects'

sss_raw_name_tmp = '{group}_res{run}_raw_trans_sss.fif'

results_sensor = 'results/run_compute_alpha_1overf/current-2015-12-15_17-59-36'

# inverse

noise_cov_fname = 'empty_sss-cov.fif'
fwd_fname = 'forward-oct-6.fif'


###############################################################################
# resampling

f_resample = 250.

###############################################################################
# Segmentation
epochs_tmin = 0.0
epochs_tmax = 28.0

###############################################################################
# PSD

fmin = 0.03
fmax = 200

# ICA #########################################################################
use_ica = False
n_components = 'rank'
n_max_ecg = 4
n_max_eog = 2
ica_reject = dict(grad=4000e-13, mag=5.5e-12)
ica_decim = 4
max_ecg_epochs = 300
ica_meg_combined = True

ica_fname_tmp = '{subject}-{ch_type}-ica.fif'

current_ica_solution_path = op.join(
    'results',
    'run_preproc_ica',
    'current-2015-12-02_12-26-31')

######################################################################
# Inverse

snr = 1
inverse_method = 'MNE'

######################################################################
# Report
scale = 1.5


######################################################################
# Frequencies
# HCP
# delta
# theta
# alpha
# beta low beta high gamma low gamma mid gamma high wide band
# 1.5-4 4-8 8-15 15-26 26-35 35-50 50-76 76-120 1.5-150

scale_free_windows = [
    ('low', (0.1, 1))
]

frequency_windows = [
    ('delta', (1.5, 4.)),
    ('theta', (4., 8.)),
    ('alpha', (8., 15.)),
    ('beta-low', (15., 26.)),
    ('beta-high', (26., 35.)),
    ('gamma-low', (35., 50.)),
    ('gamma-mid', (50., 76.)),
    ('gamma-high', (76., 120.)),
]

# as steep as possible
filter_orders = {
    'delta': (6, 7),
    'theta': (6, 7),
    'alpha': (7, 8),
    'beta-low': (9, 9),
    'beta-high': (9, 11),
    'gamma-low': (9, 12),
    'gamma-mid': (11, 14),
    'gamma-high': (13, 14),
}

noise_cov_filter = list()
for band, (l_freq, h_freq) in frequency_windows:
    filter_params = list()
    if band == 'delta':
        filter_params += [
            {'l_freq': None, 'h_freq': h_freq, 'method': 'iir',
             'iir_params': {'ftype': 'butter',
                            'order': filter_orders[band][0]}}]
    else:
        filter_params += [
            {'l_freq': None, 'h_freq': h_freq, 'method': 'iir',
             'iir_params': {'ftype': 'butter',
                            'order': filter_orders[band][0]}},
            {'l_freq': l_freq, 'h_freq': None, 'method': 'iir',
             'iir_params': {'ftype': 'butter',
                            'order': filter_orders[band][1]}}]
    noise_cov_filter.append((band, filter_params))

######################################################################
# stats

# stat_method = ['ranksums', 'tfce', 'cluster']
# stat_method = ['ranksums', 'tfce']
stat_method = ['tfce']

# Performance ########################################################

mkl_max_threads = 2

######################################################################
# viz

dpi = 100
