import numpy as np
import mne


def make_overlapping_events(raw, event_id, start, stop, step, duration):
    """Create overlapping events"""
    events = [
        mne.make_fixed_length_events(
            raw, id=event_id, start=float(ii), duration=duration)
        for ii in np.arange(start, stop, step)]
    events = [e for e in events if len(e) > 0]
    events = [e for e in events if
              ((e[:, 0].min() - raw.first_samp) / raw.info['sfreq']) >=
              duration]
    events = np.concatenate(events, axis=0)
    events = events[events[:, 0].argsort()]

    return events
