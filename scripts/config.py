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
