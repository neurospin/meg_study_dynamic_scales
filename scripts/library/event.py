import numpy as np
import mne


def make_overlapping_events(raw, event_id, overlap, duration,
                            stop=None):
    """Create overlapping events"""
    if stop is None:
        stop = raw.times[raw.last_samp]
    events = list()
    for start in np.arange(0, duration, overlap):
        events.append(mne.make_fixed_length_events(
            raw, id=event_id, start=start, stop=stop, duration=duration))
    events_max = events[0][:, 0].max()
    # events = [events + (ii, 0, 0) for ii in
    #           range(0, int(duration)]
    events = [e[np.where(e[:, 0] <= events_max)] for e in events]
    events = np.concatenate(events, axis=0)
    events = events[events[:, 0].argsort()]

    return events
