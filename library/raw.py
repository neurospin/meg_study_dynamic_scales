

def decimate_raw(raw, decim):
    """operates in place"""
    raw._data = raw._data[:, ::decim]
    raw._times = raw._times[::decim]
    raw.info['sfreq'] /= decim
