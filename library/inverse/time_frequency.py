import numpy as np

from mne.minimum_norm.inverse import _check_method, _check_ori
from mne.minimum_norm.time_frequency import (
    _prepare_source_params, _prepare_tfr)
from mne.utils import logger, _time_mask
from mne.time_frequency.tfr import cwt, morlet


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
    time_mask = None
    if any([tmin is not None, tmax is not None]):
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
