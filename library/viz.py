import numpy as np


def plot_loglog(psd, freqs, reg_fmask=None, coefs=None, intercepts=None,
                log='dec'):
    """Plot loglog

    Note. you will see some workarounds here that replace matplotlib
    functions such as semilogx and loglog.
    """
    from matplotlib import pyplot as plt
    if log == 'dec':
        my_log = np.log10
    elif log == 'ln':
        my_log = np.log
    elif log is None:
        def my_log(x):
            return x

    scaled_freqs = my_log(freqs)
    fig = plt.figure()
    plt.plot(scaled_freqs, my_log(psd).T, color='black', alpha=0.1)
    plt.plot(scaled_freqs,
             np.median(my_log(psd), axis=0), color='blue', alpha=1,
             linewidth=2)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('log power')
    plt.xlim(scaled_freqs[0], scaled_freqs[-1])

    if coefs is not None and intercepts is not None and reg_fmask is not None:
        slope_line = (np.median(coefs, 0) * scaled_freqs[reg_fmask] +
                      np.median(intercepts, 0))
        plt.plot(scaled_freqs[reg_fmask], slope_line, color='red', linewidth=2,
                 linestyle='--')

    # Draw labels and map back to original scale
    fig.canvas.draw()
    ax = plt.gca()
    new_ticklabels = list()
    for pos, text in zip(ax.get_xticks(), ax.get_xticklabels()):
        text = text.get_text()
        if text:
            text = str(round(freqs[np.abs(scaled_freqs - pos).argmin()], 1))
        new_ticklabels.append(text)
    ax.set_xticklabels(new_ticklabels)
    fig.canvas.draw()
    return fig
