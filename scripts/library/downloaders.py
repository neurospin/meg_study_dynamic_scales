import os.path as op

src_decims = ['ico2', 'oct3', 'oct4', 'oct5', 'oct5', 'oct6']

for decim in src_decims:

    def src_forward():
        key_list = [
            '%s_white_dist-true-fwd.fif' % decim,
            '%s_white_dist-true-src.fif' % decim,
        ]
        out = dict(bucket='hcp-meg-data', key_list=key_list)

        def get_files(subject):
            out['key_list'] = [
                op.join('hcp-meg', subject, f) for f in out['key_list']]
            return out

        return get_files

    locals()['src_fwd_%s' % decim] = src_forward()
