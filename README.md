# meg_study_dynamic_scales

Code of study on the relationship between 1/f and periodic brain dynamics (alpha-band oscillations).

# organization of the repository

## tracked files

### ./library

The modularized code that is used here and there. No need to install,
assuming that you run your python interpreter from the root directory.

In a script, simply

```Ptyhon
import library as lib
from library.stats import compute_corr
```

### ./scripts

The actual scripts to achieve defined steps. Don't cd here,
call `run scripts/run_XXX.py` from the root directory.

Please follow the following convention:

For computing stuff do: `run_{step}_{aspect}_{subaspect}_{subsubaspect}.py`

For plotting stuff do: `run_step_aspect_subaspect_subsubaspect.py`

Examples:

`run_preprocessing_get-filtered-data.py`
`run_preprocessing_compute-ica.py`
`run_main_inverse_compute-1of.py`
`plot_main_inverse_1of-grand-average.py`
`...`

#### ./scripts/config.py file

The config.py file has it. The variables and parameters, but also a command
line parser.

### ./examples

playground, room for trying things

### ./bin

command line programs, mostly for parallelization

## untracked files

# ./data

Here you have the data, as symbolic links. Please do it yourself.

## ./data/subjects

in `data/subject` you should have a link to the freesurfer dynacomp directory

## ./data/MEG

in `data/MEG` you should have a link to the MEG dynacomp directory

# ./results

The provenance tracking routines from [The meeg_preprocessing package](https://github.com/dengemann/meeg-preprocessing/)
will save the labled output script-wise here. Will not be commited. It's local.
Don't commit results, ok?

# dependencies

You will need the following:

- current mne master branch
- [The meeg_preprocessing package](https://github.com/dengemann/meeg-preprocessing/)

