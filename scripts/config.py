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
