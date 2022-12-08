import os
from pathlib import Path

RAW_DIR = os.environ['RAW_DATA_DIR']
PROC_DIR = os.environ['PROC_DATA_DIR']


def add_trial(df, grouping):
    def _add_trial(df):
        df['trial'] = range(len(df))
        return df.set_index('trial', append=True)
    result = df.groupby(grouping, group_keys=False).apply(_add_trial)
    return result


def cal_from_epl(name, base_path=None):
    if base_path is None:
        base_path = Path('c:/Data/Probe Tube Calibrations')
    filename = base_path / f'{name}_ProbeTube.calib'
    with filename.open('r') as fh:
        for line in fh:
            if line.startswith('Freq(Hz)'):
                break
        cal = pd.read_csv(fh, sep='\t',
                          names=['freq', 'SPL', 'phase'])
    return InterpCalibration.from_spl(cal['freq'], cal['SPL'],
                                      phase=cal['phase'])


class DatasetManager:

    def __init__(self, path, raw_dir=RAW_DIR, proc_dir=PROC_DIR):
        self.path = Path(path)
        self.raw_dir = Path(raw_dir)
        self.proc_dir = Path(proc_dir)

    def get_proc_path(self):
        return self.proc_dir / self.path.parent.relative_to(self.raw_dir) / self.path.stem

    def get_proc_filename(self, suffix, mkdir=True):
        proc_path = self.get_proc_path()
        proc_path.mkdir(exist_ok=True, parents=True)
        return proc_path / f'{self.path.stem} {suffix}'

    def is_processed(self, suffixes):
        for suffix in suffixes:
            if not self.get_proc_filename(suffix).exists():
                return False
        return True


if __name__ == '__main__':
    filename = '/mnt/nutshell/work/OHSU/R01/data_raw/mouse/pilot/20221101-135109 Brad B009-2 right  memr_interleaved_chirp.md5'
    manager = DatasetManager(filename)
    print(manager.get_proc_filename('foo.pdf'))
