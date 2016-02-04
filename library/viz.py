import numpy as np


def plot_loglog(psd, freqs, reg_fmask=None, coefs=None, intercepts=None,
                log='dec', xticks=(0.1, 1, 10)):
    """Plot loglog

    Note. you will see some workarounds here that replace matplotlib
    functions such as semilogx and loglog.
    """
    from matplotlib import pyplot as plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    my_log = _get_my_log(log)
    scaled_freqs = my_log(freqs)
    fig = plt.figure()
    plt.plot(scaled_freqs, my_log(psd).T, color='black', alpha=0.1)
    plt.plot(scaled_freqs,
             np.median(my_log(psd), axis=0), color='blue', alpha=1,
             linewidth=2, label='median PSD')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('log power')
    plt.xlim(scaled_freqs[0], scaled_freqs[-1])

    if coefs is not None and intercepts is not None and reg_fmask is not None:
        median_beta = np.median(coefs, 0)
        slope_line = (median_beta * scaled_freqs[reg_fmask] +
                      np.median(intercepts, 0))
        plt.plot(scaled_freqs[reg_fmask], slope_line, color='red', linewidth=2,
                 linestyle='--', label=r'$\beta = %0.1f$' % median_beta)
    plt.legend(loc='best')

    # Draw labels and map back to original scale
    _cleanup_logscale(fig, scaled_freqs, freqs, xticks)
    return fig


def plot_loglog_corr(freqs, corr, log='dec', xticks=(0.1, 1, 10)):
    """Plot loglog

    Note. you will see some workarounds here that replace matplotlib
    functions such as semilogx and loglog.
    """
    from matplotlib import pyplot as plt

    my_log = _get_my_log(log)
    scaled_freqs = my_log(freqs)
    fig = plt.figure()
    plt.plot(scaled_freqs, corr, color='black', alpha=0.1)
    plt.plot(scaled_freqs,
             np.median(corr, axis=1), color='blue', alpha=1,
             linewidth=2, label='median PSD')

    plt.xlabel('Frequency [Hz]')
    plt.ylabel('correlation')
    plt.xlim(scaled_freqs[0], scaled_freqs[-1])
    fig.canvas.draw()
    # # Draw labels and map back to original scale
    _cleanup_logscale(fig, scaled_freqs, freqs, xticks)
    return fig


def _get_my_log(log):
    """helper"""
    if log == 'dec':
        my_log = np.log10
    elif log == 'ln':
        my_log = np.log
    elif log is None:
        def my_log(x):
            return x
    return my_log


def _cleanup_logscale(fig, scaled_freqs, freqs, xticks):
    """helper"""
    ax = fig.gca()
    new_ticklabels = list()
    new_pos = list()
    for pos in xticks:
        # import pdb; pdb.set_trace()
        freq_idx = np.abs(freqs - pos).argmin()
        text = str(round(freqs[freq_idx], 1))
        scaled_pos = scaled_freqs[freq_idx]
        new_ticklabels.append(text)
        new_pos.append(scaled_pos)
    ax.set_xticks(new_pos)
    ax.set_xticklabels(new_ticklabels)
    fig.canvas.draw()
