"""
==============================================================
Compute coherence in source space using a MNE inverse solution
==============================================================

This examples computes the coherence between a seed in the left
auditory cortex and the rest of the brain based on single-trial
MNE-dSPM inverse solutions.

"""
# Author: Martin Luessi <mluessi@nmr.mgh.harvard.edu>
#
# License: BSD (3-clause)

import numpy as np

from slkearn.linear_model import LinearRegression

import mne
from mne.datasets import sample
from mne.io import Raw
from mne.minimum_norm import (apply_inverse, apply_inverse_epochs,
                              read_inverse_operator)
from mne.minimum_norm.inverse import _check_method, _check_ori
from mne.connectivity import seed_target_indices
# spectral_connectivity
from mne.minimum_norm.time_frequency import (
    _prepare_source_params, _prepare_tfr)
from mne.utils import logger, _time_mask
from mne.time_frequency.tfr import cwt, morlet


def compute_corr(x, y):
    """Correlate 2 matrices along last axis"""
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    r_den = np.sqrt(np.sum(xm * xm, axis=-1) *
                    np.sum(ym * ym, axis=-1))
    r = np.sum(xm * ym, axis=-1) / r_den
    return r


def source_tfr_gen(epochs, inverse_operator, frequencies, label=None,
                   lambda2=1.0 / 9.0, method="dSPM", nave=1, n_cycles=5,
                   decim=1, use_fft=False, pick_ori=None,
                   tmin=None, tmax=None,
                   baseline=None, baseline_mode=None, pca=True,
                   n_jobs=1, zero_mean=False, prepared=False, verbose=None):
    """Compute induced power and phase lock

    Computation can optionaly be restricted in a label.

    Parameters
    ----------
    epochs : instance of Epochs
        The epochs.
    inverse_operator : instance of InverseOperator
        The inverse operator.
    frequencies : array
        Array of frequencies of interest.
    label : Label
        Restricts the source estimates to a given label.
    lambda2 : float
        The regularization parameter of the minimum norm.
    method : "MNE" | "dSPM" | "sLORETA"
        Use mininum norm, dSPM or sLORETA.
    nave : int
        The number of averages used to scale the noise covariance matrix.
    n_cycles : float | array of float
        Number of cycles. Fixed number or one per frequency.
    decim : int
        Temporal decimation factor.
    use_fft : bool
        Do convolutions in time or frequency domain with FFT.
    pick_ori : None | "normal"
        If "normal", rather than pooling the orientations by taking the norm,
        only the radial component is kept. This is only implemented
        when working with loose orientations.
    baseline : None (default) or tuple of length 2
        The time interval to apply baseline correction.
        If None do not apply it. If baseline is (a, b)
        the interval is between "a (s)" and "b (s)".
        If a is None the beginning of the data is used
        and if b is None then b is set to the end of the interval.
        If baseline is equal ot (None, None) all the time
        interval is used.
    baseline_mode : None | 'logratio' | 'zscore'
        Do baseline correction with ratio (power is divided by mean
        power during baseline) or zscore (power is divided by standard
        deviation of power during baseline after subtracting the mean,
        power = [power - mean(power_baseline)] / std(power_baseline)).
    pca : bool
        If True, the true dimension of data is estimated before running
        the time frequency transforms. It reduces the computation times
        e.g. with a dataset that was maxfiltered (true dim is 64).
    n_jobs : int
        Number of jobs to run in parallel.
    zero_mean : bool
        Make sure the wavelets are zero mean.
    prepared : bool
        If True, do not call `prepare_inverse_operator`.
    verbose : bool, str, int, or None
        If not None, override default verbose level (see mne.verbose).
    """
    # XXX improve if picks is different from inverse operator
    epochs_data = epochs.get_data()
    method = _check_method(method)
    pick_ori = _check_ori(pick_ori)

    K, sel, Vh, vertno, is_free_ori, noise_norm = _prepare_source_params(
        inst=epochs, inverse_operator=inverse_operator, label=label,
        lambda2=lambda2, method=method, nave=nave, pca=pca, pick_ori=pick_ori,
        prepared=prepared, verbose=verbose)

    Fs = epochs.info['sfreq']  # sampling in Hz

    logger.info('Computing source TFR ...')
    Ws = morlet(Fs, frequencies, n_cycles=n_cycles, zero_mean=zero_mean)
    shape, _ = _prepare_tfr(
        data=epochs_data, decim=decim, pick_ori=pick_ori, Ws=Ws, K=K,
        source_ori=inverse_operator['source_ori'])
    time_mask = _time_mask(epochs.times, tmin, tmax)
    shape = (shape[0], shape[1], min(shape[2], np.sum(time_mask)))
    for epoch in epochs_data:
        if Vh is not None:
            epoch = np.dot(Vh, epoch)  # reducing data rank
        sol = _compute_tfr_wavelet(
            data=epoch, K=K, Ws=Ws, use_fft=use_fft, is_free_ori=is_free_ori,
            time_mask=time_mask, decim=decim, shape=shape)
        yield sol


def _compute_tfr_wavelet(data, shape, K, Ws, use_fft, is_free_ori,
                         decim, time_mask, verbose=None):
    """Aux function for source power using Wavelets"""

    tfr_out = np.zeros(shape, dtype=np.complex)
    for f, w in enumerate(Ws):
        print('convoluting %0.3f' % f)
        tfr = cwt(data, [w], use_fft=use_fft, decim=decim)
        tfr = np.asfortranarray(tfr.reshape(len(data), -1))
        sol = np.dot(K, tfr)
        if time_mask is not None:
            sol = sol[..., time_mask]
        sol = sol[2::3]
        tfr_out[:, f, :] = sol
    return tfr_out


print(__doc__)

data_path = sample.data_path()
subjects_dir = data_path + '/subjects'
fname_inv = data_path + '/MEG/sample/sample_audvis-meg-oct-6-meg-inv.fif'
fname_raw = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw.fif'
fname_event = data_path + '/MEG/sample/sample_audvis_filt-0-40_raw-eve.fif'
label_name_lh = 'Aud-lh'
fname_label_lh = data_path + '/MEG/sample/labels/%s.label' % label_name_lh

event_id, tmin, tmax = 1, -0.2, 1.0
method = "dSPM"  # use dSPM method (could also be MNE or sLORETA)

# Load data
inverse_operator = read_inverse_operator(fname_inv)
label_lh = mne.read_label(fname_label_lh)
raw = Raw(fname_raw)
events = mne.read_events(fname_event)

# Add a bad channel
raw.info['bads'] += ['MEG 2443']

# pick MEG channels
picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, eog=True,
                       exclude='bads')

# Read epochs
epochs = mne.Epochs(raw, events, event_id, tmin, tmax, picks=picks,
                    baseline=(None, 0), reject=dict(mag=4e-12, grad=4000e-13,
                                                    eog=150e-6))

# First, we find the most active vertex in the left auditory cortex, which
# we will later use as seed for the connectivity computation
snr = 3.0
lambda2 = 1.0 / snr ** 2
evoked = epochs.average()
stc = apply_inverse(evoked, inverse_operator, lambda2, method,
                    pick_ori="normal")

# Restrict the source estimate to the label in the left auditory cortex
stc_label = stc.in_label(label_lh)

# Find number and index of vertex with most power
src_pow = np.sum(stc_label.data ** 2, axis=1)
seed_vertno = stc_label.vertices[0][np.argmax(src_pow)]
seed_idx = np.searchsorted(stc.vertices[0], seed_vertno)  # index in orig stc

# Generate index parameter for seed-based connectivity analysis
n_sources = stc.data.shape[0]
indices = seed_target_indices([seed_idx], np.arange(n_sources))

# Compute inverse solution and for each epoch. By using "return_generator=True"
# stcs will be a generator object instead of a list. This allows us so to
# compute the coherence without having to keep all source estimates in memory.

snr = 1.0  # use lower SNR for single epochs
lambda2 = 1.0 / snr ** 2
# stcs = apply_inverse_epochs(epochs, inverse_operator, lambda2, method,
#                             pick_ori="normal", return_generator=True)

epochs.load_data()
epochs.pick_types(meg=True)

freqs = np.array([8., 13, 15., 18., 20., 25., 30.])
tfr_gen = source_tfr_gen(
    epochs, inverse_operator, frequencies=freqs,
    label=None, n_cycles=5)

n_epochs = len(epochs.events)
n_sources = np.sum([len(s['vertno']) for s in inverse_operator['src']])
regression = LinearRegression()
coefs = np.zeros([n_epochs, n_sources], dtype=np.float)
n_freqs = _time_mask(freqs, fmin, fmax).sum()
alpha_psds = np.zeros([n_epochs, n_sources, n_freqs], dtype=np.float)
fmin, fmax = 8., 15.
for ii, tfr in enumerate(tfr_gen):
    psd = np.abs(tfr)
    psd **= 2
    psd = psd.mean(-1)
    alpha_psds[ii] = psd[:, _time_mask(freqs, fmin, fmax)]
    coefs[ii] = regression.fit(freqs[:, None], psd.T).coef_[:, 0]

# # Now we are ready to compute the coherence in the alpha and beta band.
# # fmin and fmax specify the lower and upper freq. for each band, resp.
# fmin = (8., 13.)
# fmax = (13., 30.)
# sfreq = raw.info['sfreq']  # the sampling frequency
#
# # Now we compute connectivity. To speed things up, we use 2 parallel jobs
# # and use mode='fourier', which uses a FFT with a Hanning window
# # to compute the spectra (instead of multitaper estimation, which has a
# # lower variance but is slower). By using faverage=True, we directly
# # average the coherence in the alpha and beta band, i.e., we will only
# # get 2 frequency bins
# coh, freqs, times, n_epochs, n_tapers = spectral_connectivity(
#     stcs, method='coh', mode='fourier', indices=indices,
#     sfreq=sfreq, fmin=fmin, fmax=fmax, faverage=True, n_jobs=2)
#
# print('Frequencies in Hz over which coherence was averaged for alpha: ')
# print(freqs[0])
# print('Frequencies in Hz over which coherence was averaged for beta: ')
# print(freqs[1])
#
# # Generate a SourceEstimate with the coherence. This is simple since we
# # used a single seed. For more than one seeds we would have to split coh.
# # Note: We use a hack to save the frequency axis as time
# tmin = np.mean(freqs[0])
# tstep = np.mean(freqs[1]) - tmin
# coh_stc = mne.SourceEstimate(coh, vertices=stc.vertices, tmin=1e-3 * tmin,
#                              tstep=1e-3 * tstep, subject='sample')
#
# # Now we can visualize the coherence using the plot method
# brain = coh_stc.plot('sample', 'inflated', 'both',
#                      time_label='Coherence %0.1f Hz',
#                      subjects_dir=subjects_dir,
#                      clim=dict(kind='value', lims=(0.25, 0.4, 0.65)))
# brain.show_view('lateral')
