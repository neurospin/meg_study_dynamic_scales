# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)
"""example usage
run hcp_compute_subject.py --subject 100307 --storage_dir \
    /Volumes/MEG-HUB \
    --fun_path '~/scripts/library:workflows.compute_power_sepctra_and_bads' \
    --keep_files --fun_args --fmin 5 --fmax 35 --decim 16 --run_index 1
"""

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
parser.add_argument('--fun_path', metavar='fun_path', type=str,
                    nargs='?',
                    help=(
                        'the function path, e.g. '
                        '"/home/ubuntu/gihtub/mylib:algos.math.sine"'
                        'The colon is used as split point, "mylib" is added'
                        ' to sys.path'
                        'and the subsequent string will be imported where'
                        'the last element is considered the function'))
parser.add_argument('--fun_args', nargs=REMAINDER,
                    help='Commands passed to the script')
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
parser.add_argument('--out_bucket', metavar='out_bucket', type=str,
                    nargs='?', default=storage_dir,
                    help='the out bucket')
parser.add_argument('--hcp_run_inds', nargs='+', type=int, default=(0, 1, 2))
parser.add_argument('--hcp_data_types', nargs='+', type=str, default=('rest',))
parser.add_argument('--hcp_preprocessed_outputs', nargs='+', type=str,
                    default=('bads', 'ica'))
parser.add_argument('--hcp_anatomy_output', type=str,
                    default='minimal')
parser.add_argument('--hcp_no_anat', action='store_true',
                    help='skip anat')
parser.add_argument('--hcp_no_meg', action='store_true',
                    help='skip anat')

additional_downloaders_doc = """custom s3 downloaders
should take subject parameter and return a dict with the keys:
bucket : str
    The bucket name
key_list : list
    The s3 keys to download

The same pattern as in --fun_path is expected:
    ~/scripts/library:downloaders.get_from_my_bucket'
"""
parser.add_argument('--downloaders', nargs='+', type=str,
                    help=additional_downloaders_doc)


def download_from_s3_bucket(bucket, hcp_path, key_list, prefix=''):
    start_time = time.time()
    files_written = list()
    for key in key_list:
        if prefix:
            key_path = key.split(prefix)[1]
        else:
            key_path = key
        fname = op.join(hcp_path, key_path.lstrip('/'))
        files_written.append(fname)
        if not op.exists(op.split(fname)[0]):
            os.makedirs(op.split(fname)[0])
        if not op.exists(fname):
            aws_hacks.download_from_s3(
                aws_access_key_id=hcp_aws_access_key_id,
                aws_secret_access_key=hcp_aws_secret_access_key,
                fname=fname,
                bucket=bucket, key=key)
    elapsed_time = time.time() - start_time
    print('Elapsed time downloading {} from s3 {}'.format(
        bucket,
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))
    return files_written


def guess_type(string):
    if '.' in string and ''.join(string.split('.')).isdigit():
        return float(string)
    if string.isdigit():
        return int(string)
    else:
        return string


def get_function(fun_path):
    module_path, fun_path = fun_path.split(':')
    sys.path.append(module_path)
    library_name = op.split(module_path)[-1]
    fun_path = fun_path.split('.')
    fun_name = fun_path.pop(-1)
    fun_path = '.'.join([library_name] + fun_path)
    fun = getattr(importlib.import_module(fun_path), fun_name, None)
    if fun is None:
        raise ValueError('could not find %s in module %s' % (
            fun_path, fun_name))
    return fun, fun_path, fun_name

args = parser.parse_args()
subject = args.subject
storage_dir = args.storage_dir
run_id = args.run_id
fun_path = args.fun_path
fun_args = args.fun_args
hcp_run_inds = tuple(args.hcp_run_inds)
hcp_data_types = tuple(args.hcp_data_types)
hcp_preprocessed_outputs = tuple(args.hcp_preprocessed_outputs)
hcp_anatomy_output = args.hcp_anatomy_output
hcp_data_types = tuple(args.hcp_data_types)


hcp_path = op.join(storage_dir, 'HCP')
recordings_path = op.join(storage_dir, 'hcp-meg')
for this_dir in [storage_dir, hcp_path, recordings_path]:
    if not op.exists(this_dir):
        os.makedirs(this_dir)

s3_files = list()
if not args.hcp_no_anat:
    s3_files += hcp.io.file_mapping.get_s3_keys_anatomy(
        subject, hcp_path_bucket='HCP_900', mode=hcp_anatomy_output)
if not args.hcp_no_meg:
    s3_files += hcp.io.file_mapping.get_s3_keys_meg(
        subject, data_types=hcp_data_types + ('noise_empty_room',),
        processing=('unprocessed'), run_inds=hcp_run_inds)
    s3_files += hcp.io.file_mapping.get_s3_keys_meg(
        subject, data_types=hcp_data_types, processing=('preprocessed'),
        outputs=hcp_preprocessed_outputs, run_inds=hcp_run_inds)

written_files = list()
if args.s3 is True:
    written_files.extend(download_from_s3_bucket(
        bucket='hcp-openaccess', hcp_path=hcp_path, key_head='HCP_900',
        key_list=s3_files))

if args.downloaders is not None:
    for downloader in args.downloaders:
        written_files.extend(
            download_from_s3_bucket(
                hcp_path=hcp_path,
                **get_function(downloader)[0](subject=subject)))

# parse module and function
fun, fun_name, fun_path = get_function(args.fun_path)

# configure logging + provenance tracking magic
# use fun_name for script name
report, run_id, results_dir, logger = setup_provenance(
    script=fun, results_dir=op.join(recordings_path, subject),
    config=__file__, run_id=run_id)

results_path = op.join(results_dir, run_id)
written_files.append(op.join(results_path, 'call.txt'))
with open(written_files[-1], 'w') as fid:
    fid.write(repr(args))
report = Report(subject)

# check that only keyword arguments are used
assert len(fun_args) % 2 == 0
assert [k.endswith('--') for k in fun_args[0::2]]

fun_args = {k.replace('--', ''): guess_type(v) for k, v in
            zip(fun_args[0::2], fun_args[1::2])}

# now add locals as function arguments if they are supported
argspec = inspect.getargspec(fun)
for arg in ['report', 'hcp_path', 'recordings_path', 'run_id', 'subject',
            'hcp_run_inds']:
    if arg in argspec.args:
        fun_args[arg] = locals()[arg]

print('calling "%s" with:\n\t%s' % (
    fun_path + '.' + fun_name,
    '\n\t'.join(['{}: {}'.format(k, v) for k, v in fun_args.items()])
))
fun_args['n_jobs'] = 1  # for now

written_files.extend(fun(**fun_args))
# write the report into the directory with run id.
# and add files from provenance tracking to written files
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
            bucket=args.out_bucket, key=key, host='s3.amazonaws.com')

    elapsed_time = time.time() - start_time
    print('Elapsed time uploading to s3 {}'.format(
        time.strftime('%H:%M:%S', time.gmtime(elapsed_time))))

if not args.keep_files and args.s3 is True:
    my_files_to_clean = list()
    my_files_to_clean += written_files
    my_files_to_clean += [op.join(hcp_path,
                                  f.replace('HCP_900/', ''))
                          for f in s3_files]
    for fname in my_files_to_clean:
        if op.exists(fname):
            os.remove(fname)

elapsed_time_global = time.time() - start_time_global
print('Elapsed time for running scripts {}'.format(
    time.strftime('%H:%M:%S', time.gmtime(elapsed_time_global))))
