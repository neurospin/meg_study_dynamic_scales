# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import os
import sys
import os.path as op
from argparse import ArgumentParser, REMAINDER
import time
import importlib
import inspect

import pandas as pd

from meeg_preprocessing.utils import setup_provenance

from mne.report import Report
import hcp
import mkl
import aws_hacks

mkl.set_num_threads(1)

storage_dir = '/mnt'
start_time_global = time.time()
aws_details = pd.read_csv('aws_details.csv')
aws_access_key_id = aws_details['Access Key Id'].values[0]
aws_secret_access_key = aws_details['Secret Access Key'].values[0]

aws_details = pd.read_csv('aws_hcp_details.csv')
hcp_aws_access_key_id = aws_details['Access Key Id'].values[0]
hcp_aws_secret_access_key = aws_details['Secret Access Key'].values[0]

parser = ArgumentParser(description='tell subject')
parser.add_argument('--subject', metavar='subject', type=str, nargs='?',
                    default=None,
                    help='the subject to extract')
parser.add_argument('--storage_dir', metavar='storage_dir', type=str,
                    nargs='?', default=storage_dir,
                    help='the storage dir')
parser.add_argument('--keep_files',
                    action='store_true',
                    help='delete files that were written')
parser.add_argument('--n_jobs', metavar='n_jobs', type=int,
                    nargs='?', default=1,
                    help='the number of jobs to run in parallel')
parser.add_argument('--s3', action='store_true',
                    help='skip s3')
parser.add_argument('--run_id', metavar='run_id', type=str,
                    nargs='?', default=None,
                    help='the run_id')
parser.add_argument('--fun_path', metavar='fun_path', type=str,
                    nargs='?',
                    help=(
                        'the function path, e.g. '
                        '"/home/ubuntu/gihtub/mylib:algos.math"'
                        'The colon is used as split point, "mylib" is added'
                        ' to sys.path'
                        'and the subsequent string will be imported where'
                        'the last element is considered the funtion'))
parser.add_argument('--fun_args', nargs=REMAINDER,
                    help='Commands passed to the script')

args = parser.parse_args()
subject = args.subject
storage_dir = args.storage_dir
run_id = args.run_id
fun_path = args.fun_path
fun_args = args.fun_args

hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
for this_dir in [storage_dir, hcp_path, recordings_path]:
    if not op.exists(this_dir):
        os.makedirs(this_dir)

s3_meg_files = hcp.io.file_mapping.get_s3_keys_meg(
    subject, data_types=('rest', 'noise_empty_room'),
    processing=('unprocessed'), run_inds=(0, 1, 2))
s3_meg_files += hcp.io.file_mapping.get_s3_keys_meg(
    subject, data_types=('rest',), processing=('preprocessed'),
    outputs=('bads', 'ica'), run_inds=(0, 1, 2))

if args.s3 is True:
    start_time = time.time()
    for key in s3_meg_files:
        fname = op.join(hcp_path, key.split('HCP_900')[1].lstrip('/'))
        if not op.exists(op.split(fname)[0]):
            os.makedirs(op.split(fname)[0])
        if not op.exists(fname):
            aws_hacks.download_from_s3(
                aws_access_key_id=hcp_aws_access_key_id,
                aws_secret_access_key=hcp_aws_secret_access_key,
                fname=fname,
                bucket='hcp-openaccess', key=key)
    elapsed_time = time.time() - start_time
    print('Elapsed time downloading from s3 {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

# parse module and function
module_path, fun_path = args.fun_path.split(':')
sys.path.append(module_path)
library_name = op.split(module_path)[-1]
fun_path = fun_path.split('.')
fun_name = fun_path.pop(-1)
fun_path = '.'.join([library_name] + fun_path)
fun = getattr(importlib.import_module(fun_path), fun_name, None)
if fun is None:
    raise ValueError('could not find %s in module %s' % (fun_path, fun_name))

# configure logging + provenance tracking magic
# use fun_name for script name
report, run_id, results_dir, logger = setup_provenance(
    script=fun_name, results_dir=op.join(recordings_path, subject),
    run_id=run_id)

written_files = list()
report = Report(subject)

# check that only keyword arguments are used
assert len(fun_args) % 2 == 0
assert [k.endswith('--') for k in fun_args[0::2]]


def guess_type(string):
    if '.' in string and ''.join(string.split('.')).isdigit():
        return float(string)
    if string.isdigit():
        return int(string)
    else:
        return string

fun_args = {k.replace('--', ''): guess_type(v) for k, v in
            zip(fun_args[0::2], fun_args[1::2])}

# now add locals as function arguments if they are supported
argspec = inspect.getargspec(fun)
for arg in ['report', 'hcp_path', 'recordings_path', 'run_id', 'subject']:
    if arg in argspec.args:
        fun_args[arg] = locals()[arg]

print('calling "%s" with:\n\t%s' % (
    fun_path + '.' + fun_name,
    '\n\t'.join(['{}: {}'.format(k, v) for k, v in fun_args.items()])
))
fun_args['n_jobs'] = 1  # for now

written_files = fun(**fun_args)
# write the report into the directory with run id.
# and add files from provenance tracking to written files
results_path = op.join(results_dir, run_id)
written_files.extend(list({op.join(results_path, f) for f in
                           os.listdir(results_path)}))
written_files.append(op.join(results_path, 'report.html'))
report.save(written_files[-1], open_browser=False)
written_files.append(op.join(results_path, 'written_files.txt'))

with open(written_files[-1], 'w') as fid:
    fid.write('\n'.join(written_files))

if args.s3 is True:
    start_time = time.time()
    for fname in written_files:
        key = fname.split(storage_dir)[-1].lstrip('/')
        aws_hacks.upload_to_s3(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            fname=fname,
            bucket='hcp-meg-data', key=key, host='s3.amazonaws.com')

    elapsed_time = time.time() - start_time
    print('Elapsed time uploading to s3 {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

if not args.keep_files and args.s3 is True:
    my_files_to_clean = list()
    my_files_to_clean += written_files
    my_files_to_clean += [op.join(hcp_path,
                                  f.replace('HCP_900/', ''))
                          for f in s3_meg_files]
    for fname in my_files_to_clean:
        if op.exists(fname):
            os.remove(fname)

elapsed_time_global = time.time() - start_time_global
print('Elapsed time for running scripts {}'.format(
    time.strftime('%H:%M:%S', time.gmtime(elapsed_time_global))))
