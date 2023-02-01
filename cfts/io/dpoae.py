import logging
log = logging.getLogger(__name__)

from functools import lru_cache

from psi.data.io.api import Recording

# Max size of LRU cache
MAXSIZE = 1024


def dpoae_renamer(x):
    if x in ('f1_level', 'f2_level', 'dpoae_level'):
        return f'meas_{x}'
    if x in ('f2_frequency'):
        return f'req_{x}'
    return x.replace('primary_tone', 'f1') \
        .replace('secondary_tone', 'f2')


class DPOAEFile(Recording):

    def __init__(self, base_path, setting_table='dpoae_store'):
        super().__init__(base_path, setting_table)

    @property
    @lru_cache(maxsize=MAXSIZE)
    def results(self):
        data = getattr(self, 'dpoae_store')
        data = data.rename(columns=dpoae_renamer)

        # Add in the start/stop time of the actual stimulus itself. The
        # ts_start and ts_end timestamps indicate what was captured for the
        # online analysis.
        ts = self.event_log.query('event in ("dpoae_start", "experiment_end")')['timestamp'].values
        ts_start = ts[:-1]
        ts_end = ts[1:]
        if len(ts_start) != len(data):
            raise ValueError('Mismatch between event log and DPOAE metadata')
        data['dp_start'] = ts_start
        data['dp_end'] = ts_end
        return data


def load(filename):
    return DPOAEFile(filename)
