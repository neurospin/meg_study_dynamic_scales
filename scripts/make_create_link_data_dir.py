import os
import os.path as op
import glob
from itertools import groupby
base_path = '/neurospin/meg/meg_tmp/'


orig_path = op.join(base_path, 'Dynacomp_Ciuciu_2011/Results/data/SSS')
orig_er_path = op.join(
    base_path, 'Dynacomp_Ciuciu_2011/Results/data/raw/empty_room')

dest_path = op.join(base_path, 'meg_study_dynamic_scales')
meg_dir = op.join(dest_path, 'data', 'MEG')
subjects_dir = op.join(dest_path, 'data', 'subjects')
meg_er_dir = op.join(dest_path, 'data', 'MEG-ER')

#
# MEG
#

orig_subjects = list()
for group in ['AV', 'AVr', 'V']:
    orig_subjects.extend(glob.glob(op.join(orig_path, group, '*/*/')))

new_names = list()
for subject in orig_subjects:
    a, b = subject[:-1].split('/')[-2:]
    new_names.append('-'.join([a, b]))

assert len(orig_subjects) == len(set(new_names))

for this_dir in [meg_dir, meg_er_dir, subjects_dir]:
    if not op.exists(this_dir):
        os.makedirs(this_dir)

for orig, new in zip(orig_subjects, new_names):
    dest = op.join(meg_dir, new)
    if not op.islink(dest):
        print 'linking %s ->\n %s' % (orig, dest)
        os.symlink(orig, dest)
    else:
        print('doing nothing')


# create subject map
subjects_mapping = {k: [s[-1] for s in v] for k, v in
                    groupby(zip(orig_subjects, new_names),
                            key=lambda x: x[0][:-1].split('/')[-3])}
# print subjects_mapping

#
# empty room
#

orig_empty_room = glob.glob(op.join(orig_er_path, '*'))

subject_to_er = {k: k.split('-')[-1] for k in new_names}
er_to_subject = {v: k for k, v in subject_to_er.items()}
er_map = dict()
for orig in orig_empty_room:
    key = orig.split('/')[-1]
    if key in er_to_subject:
        er_map[orig] = op.join(meg_er_dir,  er_to_subject[key])


for orig, dest in er_map.items():
    if not op.islink(dest):
        print 'linking %s ->\n %s' % (orig, dest)
        os.symlink(orig, dest)
    else:
        print('doing nothing')

#
# anatomy
#

orig_subjects_folder = glob.glob(op.join(
    base_path, 'Dynacomp_Ciuciu_2011', 'yousra/subjects/*'))

subjects_orgig_new_map = dict()
new_names_old_format = [''.join(n.split('-')[0].split('_')) for n in new_names]
for f in orig_subjects_folder:
    name = op.split(f)[-1]
    key = None
    if 'landmark' in name:
        name = name.replace('_landmark',  '')
        if name in new_names_old_format:
            index = new_names_old_format.index(name)
            key = op.join(subjects_dir, new_names[index])

    if key is None:
        key = op.join(subjects_dir, name)
    subjects_orgig_new_map[f] = key


for orig, dest in subjects_orgig_new_map.items():
    if not op.islink(dest):
        print 'linking %s ->\n %s' % (orig, dest)
        os.symlink(orig, dest)
    else:
        print('doing nothing')
