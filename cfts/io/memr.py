from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from psi.data.io.api import Recording

# Max size of LRU cache
MAXSIZE = 1024


# Columns to potentially rename.
RENAME = {
    'probe_chirp_start_frequency': 'probe_fl',
    'probe_chirp_end_frequency': 'probe_fh',
    'probe_bandlimited_click_flb': 'probe_fl',
    'probe_bandlimited_click_fub': 'probe_fh',
    'probe_chirp_n': 'probe_n',
    'probe_click_n': 'probe_n',
    'probe_chirp_delay': 'probe_delay',
    'probe_click_delay': 'probe_delay',
    'probe_bandlimited_click_window': 'probe_duration',
    'elicitor_bandlimited_noise_fl': 'elicitor_fl',
    'elicitor_bandlimited_noise_fh': 'elicitor_fh',
    'elicitor_bandlimited_noise_polarity': 'elicitor_polarity',
    'elicitor_bandlimited_noise_level': 'elicitor_level',
}


class BaseMEMRFile(Recording):

    def __init__(self, base_path, setting_table='memr_metadata'):
        if 'memr' not in Path(base_path).stem:
            raise ValueError(f'{base_path} is not a MEMR recording')
        super().__init__(base_path, setting_table)

    @property
    @lru_cache(maxsize=MAXSIZE)
    def memr_metadata(self):
        data = self.__getattr__('memr_metadata')
        # We need to check what needs to be renamed since an update to the MEMR
        # paradigm now includes the renamed column names.
        rename = {k: v for k, v in RENAME.items() if v not in data}
        return data.rename(columns=rename)

    @lru_cache(maxsize=MAXSIZE)
    def get_epochs(self, columns='auto', signal_name='microphone'):
        signal = getattr(self, signal_name)
        return signal.get_epochs(
            self.memr_metadata, 0, self.trial_duration,
            columns=columns).sort_index()

    @lru_cache(maxsize=MAXSIZE)
    def get_repeats(self, columns='auto', signal_name='microphone'):
        signal = getattr(self, signal_name)
        epochs = self.get_epochs(columns, signal_name).copy()
        t = epochs.columns.values
        s = np.round(t * signal.fs).astype('i')
        s_repeat = int(round(self.repeat_period * signal.fs))
        n_probe, s_probe = np.divmod(s, s_repeat)
        t_probe = s_probe / signal.fs

        epochs.columns = pd.MultiIndex.from_arrays(
            [n_probe, t_probe], names=['repeat', 'time']
        )
        return epochs.stack('repeat').sort_index()

    @property
    def trial_duration(self):
        raise NotImplementedError

    @property
    def repeat_period(self):
        raise NotImplementedError


class InterleavedMEMRFile(BaseMEMRFile):

    @property
    def trial_duration(self):
        return self.get_setting('probe_n') * self.get_setting('repeat_period')

    @property
    def repeat_period(self):
        return self.get_setting('repeat_period')

class SimultaneousMEMRFile(BaseMEMRFile):

    @property
    def trial_duration(self):
        return self.get_setting('trial_duration')

    @property
    def repeat_period(self):
        return 1 / self.get_setting('probe_rate')

    @lru_cache(maxsize=MAXSIZE)
    def get_repeats(self, columns='auto', signal_name='microphone'):
        repeats = super().get_repeats(columns, signal_name)

        probe_n = self.get_setting('probe_n')
        onset = self.get_setting('elicitor_onset')
        duration = self.get_setting('elicitor_duration')
        rise = self.get_setting('elicitor_noise_rise_time')

        probe_map =pd.Series('', index=range(probe_n))

        def to_repeat(x, p):
            return int(round(x / p))

        rp = self.repeat_period
        e_start = to_repeat(onset, rp)
        e_ss_start = to_repeat(onset + rise, rp)
        e_ss_end = to_repeat(onset + duration - rise, rp)
        e_end =  to_repeat(onset + duration, rp)

        nw_start = to_repeat(onset - duration + rise * 2, rp)
        nw_end = to_repeat(onset, rp)

        probe_map[e_start:e_end] = 'elicitor'
        probe_map[e_ss_start:e_ss_end] = 'elicitor_ss'
        probe_map[nw_start:nw_end] = 'baseline'
        probe_map[e_end:] = 'recovery'

        ix = repeats.index.to_frame(index=False)
        ix['group'] = ix['repeat'].map(probe_map)
        new_names = repeats.index.names[:-2] + ['group', 'repeat', 't0']
        repeats.index = ix.set_index(new_names).index
        return repeats
