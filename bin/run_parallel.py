#!/usr/bin/env python
# License: simplified BSD (3 clause)
# Author: Denis A. Engemann <denis.engemann@gmail.com>

"""
Simple wrapper around GNU parallel
----------------------------------
Use this script to dispatch distributed processes and
Take care of n_jobs for child process.

Example:

```bash
run_parallel.py my_script --par_args subject1 subject2 subject3 \\
    --par_target subject --args --n_jobs 2
```
"""

import shlex
import subprocess
from argparse import ArgumentParser, REMAINDER
import multiprocessing
n_cpus = multiprocessing.cpu_count()
try:
    import mkl
    n_threads = mkl.get_max_threads()
except ImportError:
    n_threads = 1
n_par = n_cpus / n_threads

parser = ArgumentParser(description='Run script in distributed fashion')
parser.add_argument(
    '--script', metavar='script', type=str, nargs='?',
    help='The name of the script to launch', required=True)
parser.add_argument(
    '--par_args', metavar='par_args', type=str, nargs='+',
    help='multiple values to parallelize over', required=True)
parser.add_argument(
    '--par_target', metavar='par_target', type=str, nargs='?',
    help='the target variable to which parallel values are passed',
    required=True)
parser.add_argument('--args', nargs=REMAINDER,
                    help='Commands passed to the script')
parser.add_argument('--interpreter', metavar='interpreter',
                    default='python', nargs='?',
                    help='the interpreter to use.'
                         'Defaults to python')
parser.add_argument('--n_par', metavar='n_par', type=int,
                    default=n_par, nargs='?',
                    help='the number of jobs to run in parallel')
parser.add_argument('--par_sep', metavar='par_sep', type=str,
                    default='colon', nargs='?', choices=['colon', 'dash'],
                    help='the parallel syntax. used to be -- on older version')

input_args = parser.parse_args()
script = input_args.script
n_par = input_args.n_par
if input_args.par_sep == 'colon':
    par_sep = ':::'
elif input_args.par_sep == 'dash':
    par_sep = '--'
args = input_args.args
if args is None:
    args = []
par_args = input_args.par_args
par_target = input_args.par_target
if input_args.interpreter is not None:
    interpreter = input_args.interpreter


def run_parallel(script, args, par_args, par_target):
    """Run GNU parallel"""
    cmd = 'parallel -j {}'.format(n_par)
    cmd += ' --progress'
    cmd += ' "{interpreter} {script} {args} --{target} '.format(
        interpreter=interpreter,
        script=script, args=' '.join(args), target=par_target)
    cmd += "'{}'; sleep 2" + '"'
    cmd += ' {par_sep} {par_args}'.format(
        par_sep=par_sep, par_args=' '.join(par_args))
    command = shlex.split(cmd)
    print('Running command: ')
    print(cmd)
    subprocess.call(command, shell=False)

if __name__ == '__main__':
    run_parallel(script=script, args=args, par_args=par_args,
                 par_target=par_target)
