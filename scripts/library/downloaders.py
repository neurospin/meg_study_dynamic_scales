import os.path as op
import pandas as pd

aws_details = pd.read_csv(
    op.join(op.dirname(op.dirname(__file__)), 'aws_details.csv'))
aws_access_key_id = aws_details['Access Key Id'].values[0]
aws_secret_access_key = aws_details['Secret Access Key'].values[0]

src_decims = ['ico2', 'oct3', 'oct4', 'oct5', 'oct5', 'oct6']

for decim in src_decims:

    def src_forward():
        key_list = (
            '%s/{}_white_dist-true-fwd.fif'.format(decim),
            '%s/{}_white_dist-true-src.fif'.format(decim),
        )
        out = dict(
            bucket='hcp-meg-data',
            key_list=key_list,
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            out_path='recordings_path'
        )

        def get_files(subject):
            out['key_list'] = [f % op.join('hcp-meg', subject) if '%' in f
                               else f for f in out['key_list']]
            return out

        return get_files

    locals()['src_fwd_%s' % decim] = src_forward()


def get_bads_psd(subject):
    key_list = ['hcp-meg/{}/'.format(subject) + k for k in (
        'psds-bads-r0-1-50-times.npy',
        'psds-bads-r0-1-50-ave.fif',
        # 'psds-bads-r1-1-50-times.npy',
        'psds-bads-r1-1-50-ave.fif',
        # 'psds-bads-r2-1-50-times.npy',
        'psds-bads-r2-1-50-ave.fif',
        'compute_power_spectra_and_bads/bads-psds-2/report.html'
    )]
    out = dict(
        bucket='hcp-meg-data',
        key_list=key_list,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        out_path='recordings_path',
        prefix='hcp-meg/{}'.format(subject)
    )
    return out


def get_single_trial_psd(subject):
    key_list = ['hcp-meg/{}/'.format(subject) + k for k in (
        'psds-r0-0-150-epo.fif',
        'psds-r1-0-150-epo.fif',
        'psds-r2-0-150-epo.fif'
    )]
    out = dict(
        bucket='hcp-meg-data',
        key_list=key_list,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        out_path='recordings_path',
        prefix='hcp-meg/{}'.format(subject)
    )
    return out
