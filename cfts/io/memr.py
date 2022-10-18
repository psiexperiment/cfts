from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from psi.data.io.api import Recording

# Max size of LRU cache
MAXSIZE = 1024


class MEMRFile(Recording):

    def __init__(self, base_path, setting_table='memr_metadata',
                 probe='chirp'):
        if 'memr' not in Path(base_path).stem:
            raise ValueError(f'{base_path} is not a MEMR recording')
        self._probe = probe
        super().__init__(base_path, setting_table)

    @property
    @lru_cache(maxsize=MAXSIZE)
    def memr_metadata(self):
        data = self.__getattr__('memr_metadata')
        data = data.rename(columns=lambda x: x.replace('_bandlimited_noise', '')) \
            .rename(columns={
                'probe_chirp_start_frequency': 'probe_fl',
                'probe_chirp_end_frequency': 'probe_fh',
            })
        return data

    @lru_cache(maxsize=MAXSIZE)
    def get_epochs(self, columns='auto'):
        duration = self.get_setting(f'probe_{self._probe}_n') * \
            self.get_setting('repeat_period')
        return self.microphone.get_epochs(self.memr_metadata, 0, duration,
                                          columns=columns).sort_index()

    @lru_cache(maxsize=MAXSIZE)
    def get_repeats(self, columns='auto'):
        epochs = self.get_epochs(columns)
        t = epochs.columns.values
        repeat_period = self.get_setting('repeat_period')
        s = np.round(t * self.microphone.fs).astype('i')
        s_repeat = int(round(repeat_period * self.microphone.fs))
        n_probe, s_probe = np.divmod(s, s_repeat)
        t_probe = s_probe / self.microphone.fs

        epochs.columns = pd.MultiIndex.from_arrays(
            [n_probe, t_probe], names=['repeat', 'time']
        )
        return epochs.stack('repeat').sort_index()
