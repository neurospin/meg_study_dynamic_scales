# Author: Denis A. Engemann <denis.engemann@gmail.com>
# License: BSD (3-clause)

import warnings
import os.path as op
import os
import glob
import itertools as itt
import zipfile

import numpy as np
import mne
from mne.transforms import Transform
from mne.io.bti.bti import _get_bti_info
from mne.io import _loc_to_coil_trans
from mne.io.bti.bti import (
    _get_bti_dev_t, _convert_coil_trans, _coil_trans_to_loc)
from mne.utils import logger
from mne.externals import six

import scipy.io as scio
from scipy import linalg
from StringIO import StringIO


def _parse_trans(string):
    return np.array(string.replace('\n', '')
                          .strip('[] ')
                          .split(' '), dtype=float).reshape(4, 4)


def read_trans(fname, convert_to_meter):
    """Read + parse transforms

    subject_MEG_anatomy_transform.txt
    """

    transforms = dict()
    with open(fname) as fid:
        _parse_hcp_trans(fid, transforms, convert_to_meter)
    return transforms


def _parse_hcp_trans(fid, transforms, convert_to_meter):
    contents = fid.read()
    for trans in contents.split(';'):
        if 'filename' in trans or trans == '\n':
            continue
        key, trans = trans.split(' = ')
        key = key.lstrip('\ntransform.')
        transforms[key] = _parse_trans(trans)
        if convert_to_meter:
            transforms[key][:3, 3] *= 1e-3  # mm to m
    if not transforms:
        raise RuntimeError('Could not parse the transforms.')


def read_landmarks(fname):
    out = dict()
    with open(fname) as fid:
        for line in fid:
            kind, data = line.split(' = ')
            kind = kind.split('.')[1]
            if kind == 'coordsys':
                out['coord_frame'] = data.split(';')[0].replace("'", "")
            else:
                data = data.split()
                for c in ('[', '];'):
                    if c in data:
                        data.remove(c)
                out[kind] = np.array(data, dtype=int) * 1e-3  # mm to m
    return out


HCP_FNAME_TEMP = [
    '3T_Structural_preproc_extended',
    'MEG_Restin_preproc',
    'MEG_anatomy',
    'MEG_Restin_unproc',
    'MEG_noise_unproc',
    'MEG_Wrkmem_preproc',
    'MEG_Wrkmem_unproc'
]

UNPROC_RUN_MAP = {
    'meg-restin-unproc': 3,
    'meg-noise-unproc': 1,
    'meg-wrkmem-unproc': 2,
    'meg-motor-unproc': 2,
    'meg-story-unproc': 2
}

KIND_ID_MAP = {
    'meg-restin-unproc': 'Restin',
    'meg-noise-unproc': 'Rnoise',
    'meg-wrkmem-unproc': 'Wrkmem',
    'meg-motor-unproc': 'Motor',
    'meg-story-unproc': 'Story'
}

HCP_PREPROC_TEMP = dict(
    bads='{subject}_MEG_{run}-{kind}_baddata_bad{kind}.txt',
    rest='{subject}_MEG_{run}-Restin_rmegpreproc.mat',
    # task_working_memory='{subject}_MEG_'
)
FS_ANATOMY_DIR = [
    'label',
    'surf',
    'mri'
]


def _kind_from_str(string):
    """get kind from string"""
    if 'Structural' in string:
        kind = 'anatomy'
    elif 'anatomy' in string:
        kind = 'anatomy'
    elif 'Restin' in string:
        kind = 'rest'
    elif 'Motor' in string:
        kind = 'task_motor'
    elif 'noise' in string:
        kind = 'empty_room'
    elif 'Wrkmem' in string:
        kind = 'task_working_memory'
    elif 'Story' in string:
        kind = 'task_story_math'
    else:
        raise RuntimeError('Cannot classify the candidate file.')

    return kind


def _map_kind_to_idchar(kind):
    """get kind from fname"""
    id_char = None
    if kind == 'anatomy':
        pass
    elif kind == 'anatomy':
        pass
    elif kind == 'rest':
        id_char = '-Restin'
    elif kind == 'task_motor':
        pass
    elif kind == 'empty_room':
        id_char = '-Rnoise'
    elif kind == 'task_working_memory':
        id_char = '_Wrkmem_'
    if id_char is None:
        raise RuntimeError('Cannot map kind to split point.')

    return id_char


def _status_from_fname(fname):
    """Mapper function to get splitting or id character strings from kind"""
    if '_preproc' in fname or '_preproc_extended' in fname:
        status = 'preprocessed'
    elif '_unproc' in fname:
        status = 'unprocessed'
    elif fname.endswith('anatomy.zip'):
        status = 'final'  # XXX MEG can never get final. Anatomy
    else:
        raise RuntimeError('Cannot recognize the processing status of '
                           'the candidate file.')
    return status


def _classify_hcp_path(path):
    out = False
    if not _is_hcp_folder(path):
        return out
    subject = path.strip('./').split('/')[-1]
    out = list()
    path_contents = os.listdir(path)

    # anatomy freesurfer
    anatomy = ['label', 'mri', 'stats', 'surf', 'touch']
    anat_path = '{}/T1w/{}'.format(path, subject)
    has_anat_extended = True
    if 'T1w' in path_contents:
        anatomy_contents = os.listdir(anat_path)
        has_all_dirs = all(an in anatomy and op.isdir(op.join(anat_path, an))
                           for an in anatomy_contents)
        if not has_all_dirs:
            has_anat_extended = False
        for an in anatomy_contents:
            if os.stat(op.join(anat_path, an)).st_size == 0:
                has_anat_extended = False
                break
    else:
        has_anat_extended = False

    if has_anat_extended:
        out.append({
            'subject': subject,
            'label': '3t-structural-preproc-extended',
            'modality': 'mri',
            'kind': 'anatomy',
            'status': 'processed',
            'file': None,
            'root': anat_path
        })

    # meg anatomy
    has_meg_anat = True
    meg_anat_dir = op.join(path, 'MEG', 'anatomy')
    essential_files = [
        '{}_MEG_anatomy_fiducials.txt',
        '{}_MEG_anatomy_headmodel.mat',
        '{}_MEG_anatomy_landmarks.txt',
        '{}_MEG_anatomy_transform.txt',
    ]
    if op.isdir(meg_anat_dir):
        meg_anat_contents = os.listdir(meg_anat_dir)
        for fname in essential_files:
            if not fname.format(subject) in meg_anat_contents:
                warnings.warn('Essential MEG anatomy is not complete '
                              'for %s' % subject)
                has_meg_anat = False
                break
    if has_meg_anat:
        out.append({
            'subject': subject,
            'label': 'meg-anatomy',
            'modality': 'meg',
            'kind': 'anatomy',
            'status': 'preprocessed',
            'file': None,
            'root': meg_anat_dir
        })

    unprocessed_dir = op.join(path, 'unprocessed', 'MEG')
    expected_files = {'c,rfDC', 'config'}
    if os.path.isdir(unprocessed_dir):
        for orig_kind in KIND_ID_MAP.values():
            globbed = glob.glob(unprocessed_dir + '/*' + orig_kind)
            kind = _kind_from_str(orig_kind)
            label = '-'.join(
                ['meg', orig_kind.lower().replace('rnoise', 'noise'),
                 'unproc'])

            # check if all runs are there
            if len(globbed) == 0:
                continue
            elif len(globbed) != UNPROC_RUN_MAP[label]:
                warnings.warn('could not find all "{}" folders for {},'
                              ' skipping.'
                              .format(kind, subject))
                continue

            # check if all files are there
            for gl in globbed:
                data_files = glob.glob(gl + '/4D/*')
                data_files = set(expected_files).intersection(data_files)
                if data_files != expected_files:
                    warnings.warn('could not find all "{}" files for {}, '
                                  'skipping.'
                                  .format(kind, subject))
                    continue

            out.append({
                'subject': subject,
                'label': label,
                'modality': 'meg',
                'kind': kind,
                'status': 'unprocessed',
                'file': None,
                'root': unprocessed_dir
            })

    for orig_kind in KIND_ID_MAP.values():
        if orig_kind == 'anatomy':
            continue
        this_dir = op.join(path, 'MEG', orig_kind)
        label = '-'.join(
            ['meg', orig_kind.lower().replace('rnoise', 'noise'),
             'preproc'])

        if op.isdir(this_dir):
            # XXX improve samity checks later
            kind = _kind_from_str(orig_kind)
            out.append({
                'subject': subject,
                'label': label,
                'modality': 'meg',
                'kind': kind,
                'status': 'preprocessed',
                'file': None,
                'root': this_dir
            })

    return out


def _is_hcp_folder(path):
    """helper to check if path is HCP path"""
    n_folders = 0
    is_hcp_folder = False
    main_dirs = ['MEG', 'T1w', 'unprocessed', 'release-notes']
    if op.isdir(path):
        contents = os.listdir(path)
        for cc in contents:
            if not op.isdir(op.join(path, cc)):
                continue
            elif cc in main_dirs:
                n_folders += 1

    isdigit = path.strip('./').split('/')[-1].isdigit()
    if n_folders > 0 and isdigit:
        is_hcp_folder = True
    return is_hcp_folder


def _classify_hcp_zip(path):
    """check if file belongs to hcp"""
    out = False
    root, fname = op.split(path)
    for e in HCP_FNAME_TEMP:
        if e in fname:
            break
    else:
        return out
    split = op.splitext(fname)[0].split('_')
    out = {
        'subject': split[0],
        'label': '-'.join(split[1:]).lower(),
        'modality': split[1],
        'kind': _kind_from_str(fname),
        'status': _status_from_fname(fname),
        'file': fname,
        'root': root
    }
    return out


def _check_subject(subject, include_subjects, exclude_subjects):
    """Check subject selection"""
    out = True
    if include_subjects:
        if subject not in include_subjects:
            out = False
    if exclude_subjects:
        if subject in exclude_subjects:
            out = False
    return out


def parse_hcp_dir(hcp_path, exclude_subjects=None, include_subjects=None,
                  required_fields=None):
    if required_fields is None:
        required_fields = [
            'meg-restin-unproc',
            'meg-anatomy',
            'meg-noise-unproc',
            '3t-structural-preproc-extended',
            'meg-restin-preproc',
        ]
    paths = (op.join(hcp_path, s) for s in os.listdir(hcp_path) if s.isdigit()
             and len(s) == 6)
    hcp_files = (_classify_hcp_path(f) for f in paths)
    hcp_files = sum([m for m in hcp_files if m], [])
    file_map = _assemble_file_map(hcp_files, include_subjects=include_subjects,
                                  exclude_subjects=exclude_subjects,
                                  required_fields=required_fields)
    return file_map


def _assemble_file_map(hcp_files, include_subjects, exclude_subjects,
                       required_fields):

    # important: groupby won't work if iterable is not sorted already by key
    hcp_files.sort(key=lambda m: m['subject'])
    file_map = list()
    for subject, records in itt.groupby(hcp_files, lambda m: m['subject']):
        if _check_subject(subject=subject, include_subjects=include_subjects,
                          exclude_subjects=exclude_subjects):
            records = {r['label']: r for r in records}
            is_complete = True
            if required_fields is not None:
                for field in required_fields:
                    if field not in records:
                        is_complete = False
            if is_complete:
                file_map.append((subject, records))
    file_map.sort(key=lambda x: x[0])
    return file_map


def parse_hcp_zips(hcp_path, exclude_subjects=None, include_subjects=None,
                   required_fields=None):
    """ Traverse and map zip files

    Parameters
    ----------
    hcp_path : str
        The directory containing the HCP files (flat, as downloaded).
    exclude_subjects : list of str
        Subjects to exclude.
    include_subjects : list of str
        Subjects to include.
    required_fields : list of str
        The required file type that are needed for each subject. Different
        analyses necessitate access to various files. For example, in order
        to read the preprocessed data into an MNE container, not only the
        preprocessed files are needed but also the headers from the raw files.

        Defaults to:
            required_fields = [
                'meg-restin-unproc',
                'meg-anatomy',
                'meg-noise-unproc',
                '3t-structural-preproc-extende'
                'meg-restin-preproc',
            ]

        Note. The labels are derived from concatenating the file name common
        patterns of the given zip file using dashes and putting them in
        lowercase. For example, 'MEG_Wrkmem_unproc' gets 'meg-wrkmem-unproc'.
    """
    if required_fields is None:
        required_fields = [
            'meg-restin-unproc',
            'meg-anatomy',
            'meg-noise-unproc',
            '3t-structural-preproc-extended',
            'meg-restin-preproc',
        ]

    hcp_files = (_classify_hcp_zip(f)
                 for f in glob.glob(op.join(hcp_path, '*.zip')))

    hcp_files = [m for m in hcp_files if m]
    file_map = _assemble_file_map(hcp_files, include_subjects=include_subjects,
                                  exclude_subjects=exclude_subjects,
                                  required_fields=required_fields)

    return file_map


def _filter_fmap(file_map, include_subjects=None, exclude_subjects=None):
    """Convenience function to iterate subject entries in file map.
    """
    if not exclude_subjects:
        exclude_subjects = list()
    if not include_subjects:
        include_subjects = list()
    for subject, records in file_map:
        if subject in exclude_subjects:
            continue
        elif include_subjects:
            if subject not in include_subjects:
                continue
        yield subject, records


def _recursive_create_dir(path, start):
    """Recursive directory expansion"""
    if start not in path:
        raise RuntimeError('Start value is not valid for force create dir.'
                           'Please ping @dengemann')
    out_path = start
    rest_path = path.split(start)[1]
    path_split = rest_path.lstrip(op.sep).split(op.sep)
    for this_dir in path_split:
        out_path = op.join(out_path, this_dir)
        if not op.isdir(out_path):
            os.mkdir(out_path)


def _write_target(fname, data, start_path, out_path, prefix=None):
    """Recursive directory expansion and writing"""
    if prefix is None:
        prefix = start_path
    # XXX handle paths / fnames which start with '/'
    prefix = [prefix] if prefix else []
    out_name_list = fname.split(op.sep)
    out_name = op.join(*prefix +
                       out_name_list[1 + out_name_list.index(start_path):])
    out_name = op.join(out_path, out_name)
    _recursive_create_dir(op.split(out_name)[0], out_path)
    with open(out_name, 'wb') as fid:
        fid.write(data)


def _get_head_model(head_model_fid, hcp_trans, ras_trans):
    head_mat = scio.loadmat(head_model_fid, squeeze_me=False)
    pnts = head_mat['headmodel']['bnd'][0][0][0][0][0]
    faces = head_mat['headmodel']['bnd'][0][0][0][0][1]
    faces -= 1  # correct matlab index

    pnts = mne.transforms.apply_trans(
        linalg.inv(ras_trans).dot(hcp_trans['bti2spm']), pnts)
    return pnts, faces


def extract_anatomy(subject, hcp_path, anatomy_path, recordings_path=None):
    """Extract relevant anatomy and create MNE friendly directory layout"""
    if isinstance(subject, six.string_types):
        _, records = parse_hcp_dir(
            hcp_path=hcp_path, include_subjects=[subject])[0]
    elif isinstance(subject, dict):
        records, subject = subject, subject.values()[0]['subject']
    else:
        raise ValueError('subject must be dict or str.')
    this_anatomy_path = op.join(anatomy_path, subject)
    _recursive_create_dir(this_anatomy_path, op.split(anatomy_path)[0])
    if not recordings_path:
        recordings_path = anatomy_path

    this_recordings_path = op.join(recordings_path, subject)
    _recursive_create_dir(this_recordings_path, op.split(recordings_path)[0])

    rec = records.get('3t-structural-preproc-extended', None)
    ras_trans_fname = ''
    if rec:
        logger.info('reading extended structural processing ...')
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf = zipfile.ZipFile(fid)
            for info in zf.infolist():  # let's extract data!

                if not info.file_size:  # test if there is data == a file.
                    print 'skiping %s' % info.filename
                    continue

                fname = info.filename
                if fname.endswith('c_ras.mat'):
                    ras_trans_fname += fname

                sel = np.where(  # check + handle anatomy paths
                    # test if '/subpatath/' is in fname, that is how you filter
                    ['/' + k + '/' in fname for k in FS_ANATOMY_DIR])[0]
                if len(sel):  # hit!
                    start_path = FS_ANATOMY_DIR[sel[0]]
                    logger.debug(fname)
                    _write_target(
                        fname=fname, data=zf.read(fname),
                        start_path=start_path, out_path=this_anatomy_path)
            logger.info('done.')
            if not ras_trans_fname:
                raise ValueError(
                    'Could not find the Freesurfer RAS transform for'
                    'subject %s' % subject)
            logger.info('reading RAS freesurfer transform')
            ras_trans = np.array([
                r.split() for r in zf.read(ras_trans_fname)
                                     .split('\n') if r], dtype=np.float64)

    else:
        ValueError('Could not find extended anatomy for %s' % subject)

    rec = records.get('meg-anatomy', None)
    if rec:
        logger.info('reading MEG anatomy')
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf = zipfile.ZipFile(fid)
            head_model_fname = _zip_get_fnames(
                zf=zf, kind='headmodel', extension='.mat', run=[1],
                max_runs=1)[0]
            transforms_fname = _zip_get_fnames(
                zf=zf, kind='anatomy_transform', extension='.txt', run=[1],
                max_runs=1)[0]

            hcp_trans = dict()
            _parse_hcp_trans(
                fid=StringIO(zf.read(transforms_fname)),
                transforms=hcp_trans, convert_to_meter=False)

            pnts, faces = _get_head_model(
                head_model_fid=StringIO(zf.read(head_model_fname)),
                hcp_trans=hcp_trans,
                ras_trans=ras_trans)

        tri_fname = op.join(this_anatomy_path, 'bem', 'inner_skull.surf')

        logger.info('extracting head model')
        _recursive_create_dir(
            op.join(this_anatomy_path, 'bem'), this_anatomy_path)
        mne.write_surface(tri_fname, pnts, faces)

        logger.info('Combining RAS transform and coregistration')
        ras_trans_m = ras_trans.copy()
        ras_trans_m = linalg.inv(ras_trans_m)  # and the inversion

        # now convert to meter too here
        ras_trans_m[:3, 3] *= 1e-3
        bti2spm = hcp_trans['bti2spm']
        bti2spm[:3, 3] *= 1e-3
        head_mri_t = Transform(
            'ctf_head', 'mri', np.dot(ras_trans_m, bti2spm))

        logger.info('extracting coregistration')
        mne.write_trans(op.join(
            this_recordings_path, '%s-head_mri-trans.fif') % subject,
            head_mri_t)
    else:
        logger.info('done.')
        ValueError('Could not find MEG anatomy for %s' % subject)


def _read_bti_info(zf, config):
    raw_fid = None
    if zf is not None:
        config_fid = StringIO(zf.read(config))
    else:
        config_fid = config
    info, bti_info = _get_bti_info(
        pdf_fname=raw_fid, config_fname=config_fid, head_shape_fname=None,
        rotation_x=0.0, translation=(0.0, 0.02, 0.11), convert=False,
        ecg_ch='E31', eog_ch=('E63', 'E64'),
        rename_channels=False, sort_by_ch_name=False)
    return info


def _read_raw_bti(raw_fid, config_fid, convert):
    """Convert and raw file from HCP input"""
    raw = mne.io.read_raw_bti(
        raw_fid, config_fid, convert=convert, head_shape_fname=None,
        sort_by_ch_name=False, rename_channels=False, preload=True)

    return raw


def _zip_get_fnames(zf, kind, extension, run, max_runs=3):
    """Zip content filter"""
    candidates = [f for f in zf.namelist() if f.endswith(extension) and
                  kind in f]
    assert len(candidates) == max_runs
    if run is not None:
        candidates = [k for ii, k in enumerate(sorted(candidates), 1)
                      if ii in run]
    return candidates


def _check_raw_config_runs(raws, configs):
    # XXX still needed?
    for raw, config in zip(raws, configs):
        assert op.split(raw)[0] == op.split(config)[0]
    run_str = set([configs[0].split('/')[-3]])
    for config in configs[1:]:
        assert set(configs[0].split('/')) - set(config.split('/')) == run_str


def _check_infos_trans(infos):
    """check info extraction"""
    chan_max_idx = np.argmax([c['nchan'] for c in infos])
    chan_template = infos[chan_max_idx]['ch_names']
    channels = [c['ch_names'] for c in infos]
    common_channels = set(chan_template).intersection(*channels)

    common_chs = [[c['chs'][c['ch_names'].index(ch)] for ch in common_channels]
                  for c in infos]
    dev_ctf_trans = [i['dev_ctf_t']['trans'] for i in infos]
    cns = [[c['ch_name'] for c in cc] for cc in common_chs]
    for cn1, cn2 in itt.combinations(cns, 2):
        assert cn1 == cn2
    # BTI stores data in head coords, as a consequence the coordinates
    # change across run, we apply the ctf->ctf_head transform here
    # to check that all transforms are correct.
    cts = [np.array([linalg.inv(_loc_to_coil_trans(c['loc'])).dot(t)
                    for c in cc])
           for t, cc in zip(dev_ctf_trans, common_chs)]
    for ct1, ct2 in itt.combinations(cts, 2):
        np.testing.assert_array_almost_equal(ct1, ct2, 12)


def _handle_records(subject, hcp_path, required_fields):
    if isinstance(subject, six.string_types):
        _, records = parse_hcp_dir(
            hcp_path=hcp_path, include_subjects=[subject])[0]
    elif isinstance(subject, dict):
        records, subject = subject, subject.values()[0]['subject']
    else:
        raise ValueError('subject must be dict or str.')

    for field in required_fields:
        if field not in records:
            raise ValueError('I need "%s" to get you the data' % field)
    return records, subject


def read_meg_noise(subject, hcp_path, kind='empty_room'):
    noise_map = {
        'empty_room': 'Rnoise',
        'subject': 'Pnoise'
    }

    if kind not in noise_map:
        raise ValueError('I only know `empty_room` or `subject` noise, sorry.')
    noise_kind = noise_map[kind]
    required_fields = ['meg-noise-unproc']
    records, subject = _handle_records(
        subject=subject, hcp_path=hcp_path, required_fields=required_fields)
    rec = records['meg-noise-unproc']
    logger.info('Reading empty room data')

    if rec['file'] is not None:
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf_er_noise = zipfile.ZipFile(fid)
            config = _zip_get_fnames(
                zf=zf_er_noise, kind=noise_kind, extension='config',
                max_runs=1,
                run=None)[0]
            pdf_fname = _zip_get_fnames(
                zf=zf_er_noise, kind=noise_kind, extension='c,rfDC',
                max_runs=1,
                run=None)[0]
            config_fid = StringIO(zf_er_noise.read(config))
            raw_er = _read_raw_bti(
                StringIO(zf_er_noise.read(pdf_fname)), config_fid,
                convert=False)
    else:
        dirs = list(sorted(
            f for f in os.listdir(rec['root']) if f.endswith(noise_kind)))
        location = dirs[0]
        pdf_fname = op.join(rec['root'], location, '4D', 'c,rfDC')
        config_fname = op.join(rec['root'], location, '4D', 'config')
        raw_er = _read_raw_bti(pdf_fname, config_fname, convert=False)

    return raw_er


def read_meg_unprocessed(subject, hcp_path, kind='restin', run=0):
    key = 'meg-%s-unproc' % kind
    required_fields = [key]
    records, subject = _handle_records(
        subject=subject, hcp_path=hcp_path, required_fields=required_fields)
    rec = records[key]
    max_runs = UNPROC_RUN_MAP[key]
    logger.info('Reading %s data' % key)
    my_kind = KIND_ID_MAP[key]
    if rec['file'] is not None:
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf = zipfile.ZipFile(fid)
            config = _zip_get_fnames(
                zf=zf, kind=my_kind, extension='config', max_runs=max_runs,
                run=run)[0]
            pdf_fname = _zip_get_fnames(
                zf=zf, kind=my_kind, extension='c,rfDC', max_runs=max_runs,
                run=run)[0]
            config_fid = StringIO(zf.read(config))
            raw = _read_raw_bti(
                StringIO(zf.read(pdf_fname)), config_fid, convert=False)
    else:
        dirs = list(sorted(
            f for f in os.listdir(rec['root']) if f.endswith(my_kind)))
        location = dirs[run]
        pdf_fname = op.join(rec['root'], location, '4D', 'c,rfDC')
        config_fname = op.join(rec['root'], location, '4D', 'config')
        raw = _read_raw_bti(pdf_fname, config_fname, convert=False)

    return raw


def read_meg_info(subject, hcp_path, run=0):
    required_fields = ['meg-restin-unproc']
    records, subject = _handle_records(
        subject=subject, hcp_path=hcp_path, required_fields=required_fields)
    rec = records['meg-restin-unproc']
    logger.info('Reading infos from MEG data')
    if rec['file'] is None:
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf_meg = zipfile.ZipFile(fid)
            config = _zip_get_fnames(
                zf=zf_meg, kind='4D', extension='config', max_runs=3,
                run=run)[0]
            meg_info = _read_bti_info(zf_meg, config)
    else:
        raise NotImplementedError('For now only zips supported')
        # XXX
        # my_kind = kind.capitalize()
        # dirs = list(sorted(
        #     f for f in os.listdir(rec['root']) if f.endswith(my_kind)))
        # location = dirs[run]
        # config_fname = op.join(rec['root'], location, '4D', 'config')
        # meg_info = _read_bti_info(None, config_fname)

    return meg_info


def read_meg_preprocessed(subject, hcp_path, kind, onset='TIM', run=0):
    """Read HCP processed data

    Parameters
    ----------
    subject : str, dict
        The subject or the record from the directory parser
    hcp_path : str
        The directory containing the HCP data.
    kind : str
        The type of epoched data, e.g. 'meg-wrkmem' or 'meg-motor', see
        `required_fields` in `parse_hcp_dir`.
    onset : str
        Depends on task data, e.g., 'TRESP' or 'TIM'. Defaults to 'TIM'.
    run : int
        The run number (not an index). Defaults to the first run.
    """

    required_fields = [kind + '-' + state for state in ('preproc', 'unproc')]
    records, subject = _handle_records(subject, hcp_path, required_fields)
    rec = records['meg-%s-unproc' % kind]
    logger.info('creating measurement info structure from config for '
                'runs: %s' % run)
    max_runs = UNPROC_RUN_MAP['%s-unproc' % kind]
    if rec['file'] is not None:
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf_unproc = zipfile.ZipFile(fid)
            config = _zip_get_fnames(
                zf=zf_unproc, kind='4D', extension='config', run=run,
                max_runs=max_runs)[0]
            info = _read_bti_info(zf_unproc, config)
    else:
        my_kind = kind.capitalize()
        dirs = list(sorted(f for f in os.listdir(rec['root'])
                           if f.endswith(my_kind)))
        location = dirs[run]
        config_fname = op.join(rec['root'], location, '4D', 'config')
        info = _read_bti_info(None, config_fname)

    rec = records['%s-preproc' % kind]
    if onset is not None:
        filter_ = 'preproc_' + onset
    else:
        filter_ = 'preproc'

    if rec['file'] is not None:
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf_preproc = zipfile.ZipFile(fid)
            preproc = _zip_get_fnames(
                zf=zf_preproc, kind=filter_, extension='.mat', run=run,
                max_runs=max_runs)[0]
            epochs = _read_epochs(preproc, info, zf_preproc)
    else:
        preproc_type = 'rmegpreproc' if 'rest' in kind else 'tmegpreproc'
        my_kind = kind.capitalize()
        dirs = list(sorted(
            f for f in os.listdir(op.join(rec['root'], preproc_type))
            if f.endswith('.mat') and my_kind in f))
        location = dirs[run]
        preproc = op.join(rec['root'], preproc_type, location)
        epochs = _read_epochs(preproc, info, None)

    return epochs


def read_trial_info(subject, hcp_path, kind, run=0):

    required_fields = ['meg-%s-preproc' % kind]
    records, subject = _handle_records(subject, hcp_path, required_fields)
    rec = records['%s-preproc' % kind]
    max_runs = UNPROC_RUN_MAP['%s-unproc' % kind]
    if rec['file'] is not None:
        with open(op.join(rec['root'], rec['file'])) as fid:
            zf_preproc = zipfile.ZipFile(fid)
            trl_info_fname = _zip_get_fnames(
                zf=zf_preproc, kind='preproc',
                extension='trialinfo.mat', run=run,
                max_runs=max_runs)[0]
            trl_infos = _read_trial_info(zf_preproc, trl_info_fname)
    else:
        dirs = list(sorted(f for f in
                           os.listdir(op.join(rec['root'], 'tmegpreproc'))
                           if f.endswith('trialinfo.mat')))
        location = dirs[run]
        fname = op.join(rec['root'], 'tmegpreproc', location)
        trl_infos = _read_trial_info(None, fname)
    return trl_infos


def _read_trial_info(zf_preproc, fname):
    """ helper to read trial info """
    if zf_preproc is not None:
        finput = StringIO(zf_preproc.read(fname))
    else:
        finput = fname
    data = scio.loadmat(finput, squeeze_me=True)['trlInfo']
    out = dict()

    for idx, lock_name in enumerate(data['lockNames'].tolist()):
        out[lock_name] = dict(
            comments=data['trlColDescr'].tolist()[idx],
            codes=data['lockTrl'].tolist().tolist()[idx])

    return out


def _read_epochs(preproc, info, zf_preproc):
    if zf_preproc is not None:
        finput = StringIO(zf_preproc.read(preproc))
    else:
        finput = preproc
    data = scio.loadmat(finput,
                        squeeze_me=True)['data']
    ch_names = [ch for ch in data['label'].tolist()]
    info['sfreq'] = data['fsample'].tolist()
    data = np.array([data['trial'].tolist()][0].tolist())
    events = np.zeros((len(data), 3), dtype=np.int)
    events[:, 0] = np.arange(len(data))
    events[:, 2] = 99
    this_info = mne.pick_info(
        info, [info['ch_names'].index(ch) for ch in ch_names],
        copy=True)
    return mne.EpochsArray(data=data, info=this_info, events=events,
                           tmin=0)


def _check_sorting_runs(candidates, id_char):
    """helper to ensure correct run-parsing and mapping"""
    run_idx = [f.find(id_char) for f in candidates]
    for config, idx in zip(candidates, run_idx):
        assert config[idx - 1].isdigit()
        assert not config[idx - 2].isdigit()
    runs = [int(f[idx - 1]) for f, idx in zip(candidates, run_idx)]
    return runs, candidates


def _parse_annotations_segments(segment_strings):
    """Read bad segments defintions from text file"""
    split = segment_strings.split(';')
    out = dict()
    for entry in split:
        if len(entry) == 1 or entry == '\n':
            continue
        key, rest = entry.split(' = ')
        val = np.array([e.split(' ') for e in rest.split() if
                        e.replace(' ', '').isdigit()], dtype=int)
        # reindex and reshape
        val = val.reshape(-1, 2) - 1
        out[key.split('.')[1]] = val
    return out


def read_annotations(subject, hcp_path, kind='restin', run=0):
    """
    Parameters
    ----------
    subject : str, file_map
        The subject
    hcp_path : str
        The HCP directory
    kind : str
        the data type

    Returns
    -------
    out : dict
        The annotations.
    """
    key = 'meg-%s-preproc' % kind
    required_fields = [key]
    records, subject = _handle_records(subject, hcp_path, required_fields)
    rec = records[key]
    logger.info('creating measurement info structure from config for '
                'runs: %s' % run)

    my_kind = kind.capitalize()
    out = dict()
    call = {'channels': _parse_annotations_bad_channels,
            'segments': _parse_annotations_segments}
    preproc_type = 'baddata'
    for subtype in ['channels', 'segments']:
        dirs = list(sorted(f for f in
                           os.listdir(op.join(rec['root'], preproc_type))
                           if my_kind in f and
                           f.endswith('baddata_bad%s.txt' % subtype)))
        location = dirs[run]
        fname = op.join(rec['root'], preproc_type, location)
        with open(fname, 'r') as fid:
            out[subtype] = call[subtype](fid.read())
    dirs = list(sorted(f for f in
                       os.listdir(op.join(rec['root'], 'icaclass'))
                       if my_kind in f and
                       f.endswith('icaclass_vs.txt')))
    location = dirs[run]
    fname = op.join(rec['root'], 'icaclass', location)
    with open(fname, 'r') as fid:
        out['ica'] = _parse_annotations_ica(fid.read())

    return out


def read_ica(subject, hcp_path, kind='restin', run=0):
    """
    Parameters
    ----------
    subject : str, file_map
        The subject
    hcp_path : str
        The HCP directory
    kind : str
        the data type

    Returns
    -------
    out : numpy structured array
        The ICA mat struct.
    """
    key = 'meg-%s-preproc' % kind
    required_fields = [key]
    records, subject = _handle_records(subject, hcp_path, required_fields)
    rec = records[key]
    logger.info('creating measurement info structure from config for '
                'runs: %s' % run)

    my_kind = kind.capitalize()
    dirs = list(sorted(f for f in
                       os.listdir(op.join(rec['root'], 'icaclass'))
                       if my_kind in f and
                       f.endswith('icaclass_vs.mat')))
    location = dirs[run]
    fname = op.join(rec['root'], 'icaclass', location)
    mat = scio.loadmat(fname, squeeze_me=True)['comp_class']
    return mat


def apply_ica(raw, ica_mat, exclude):
    """
    Parameters
    ----------
    raw : instance of Raw
        the hcp raw data.
    ica_mat : numpy structured array
        The hcp ICA solution
    exclude : array-like
        the components to be excluded
    """
    assert ica_mat['topolabel'].tolist().tolist() == raw.ch_names[:]

    unmixing_matrix = np.array(ica_mat['unmixing'].tolist())

    n_components, n_channels = unmixing_matrix.shape
    mixing = np.array(ica_mat['topo'].tolist())
    exclude = np.array(exclude)
    proj_mat = (np.eye(n_channels) -
                mixing[:, exclude].dot(unmixing_matrix[exclude]))
    raw._data *= 1e15
    raw._data[:] = np.dot(proj_mat, raw._data)
    raw._data /= 1e15


def _parse_annotations_bad_channels(bads_strings):
    """Read bad channel definitions from text file"""
    split = bads_strings.split(';')
    out = dict()
    for entry in split:
        if len(entry) == 1 or entry == '\n':
            continue
        key, rest = entry.split(' = ')
        val = [ch for ch in rest.split("'") if ch.isalnum()]
        out[key.split('.')[1]] = val
    return out


def _parse_annotations_ica(ica_strings):
    """Read bad channel definitions from text file"""
    split = ica_strings.split(';')
    out = dict()
    for entry in split:
        if len(entry) == 1 or entry == '\n':
            continue
        key, rest = entry.split(' = ')
        if '[' in rest:
            sep = ' '
        else:
            sep = "'"
        val = [(int(ch) if ch.isdigit() else ch) for ch in
               rest.split(sep) if ch.isalnum()]
        out[key.split('.')[1]] = val
    return out


def transform_sensors_to_mne(inst):
    """ transform sensors to MNE coordinates """
    bti_dev_t = mne.transforms.Transform('ctf_meg', 'meg', _get_bti_dev_t())
    dev_ctf_t = inst.info['dev_ctf_t']
    for ch in inst.info['chs']:
        loc = ch['loc'][:]
        if loc is not None:
            print 'converting %s' % ch['ch_name']
            t = _loc_to_coil_trans(loc)
            t = _convert_coil_trans(t, dev_ctf_t, bti_dev_t)
            loc = _coil_trans_to_loc(t)
            ch['loc'] = loc
