import importlib
import json
from pathlib import Path
import re

import pandas as pd

from psiaudio.calibration import FlatCalibration, InterpCalibration
from psi import get_config
from psi.application import get_default_io
from psi.core.enaml.api import load_manifest


NO_STARSHIP_ERROR = '''
No starship could be found in the IO manifest. To use this plugin, you must
have an analog input channel named starship_ID_microphone and two analog output
channels named starship_ID_primary and starship_ID_secondary. ID is the name of
the starship that will appear in any drop-down selectors where you can select
which starship to use (assuming your system is configured for more than one
starship).
'''


class CalibrationManager:

    P_NAME = re.compile('^(.*)\((.*)\)$')

    def __init__(self):
        self.loaders = {}

    def register(self, name):
        module, klass = name.rsplit('.', 1)
        loader = getattr(importlib.import_module(module), klass)()
        loader.qualname = name
        self.loaders[name] = loader

    def get_calibration(self, name):
        path = self.base_path / name
        return sorted(list(path.iterdir()))[-1]

    def list_choices(self):
        names = {}
        for loader_name, loader in self.loaders.items():
            for name in loader.list_choices():
                names[f'{name} ({loader.label})'] = f'{loader_name}::{name}'
        return names

    def load(self, name):
        loader_name, obj_name = name.split('::', 1)
        return self.loaders[loader_name].load(obj_name)


class BaseLoader:
    pass


class StarshipLoader(BaseLoader):
    pass


class EPLStarshipLoader(StarshipLoader):

    label = 'EPL'
    base_path = Path(r'C:\Data\Probe Tube Calibrations')

    @classmethod
    def load_file(self, filename, attrs=None):
        '''
        Load calibration from EPL probe tube calibration file

        Parameters
        ----------
        filename : {str, Path}
            File containing calibration data
        attrs : {None, dict}
            Attributes to attach to the calibration. These automatically get
            saved with the psiexperiment data. Useful for verifying proper
            calibration was loaded.
        '''
        with Path(filename).open('r') as fh:
            for line in fh:
                if line.startswith('Freq(Hz)'):
                    break
            cal = pd.read_csv(fh, sep='\t', header=None)
            return InterpCalibration.from_spl(cal[0], cal[1], attrs=attrs)

    def list_choices(self):
        for calfile in self.base_path.glob('*_ProbeTube.calib'):
            yield calfile.stem.rsplit('_', 1)[0]

    def load(self, name):
        path = self.base_path / f'{name}_ProbeTube.calib'
        attrs = {
            'calibration_file': str(path),
            'loader': self.qualname,
            'name': name,
        }
        return self.load_file(path, attrs)


class CFTSStarshipLoader(StarshipLoader):

    label = 'CFTS'
    base_path = get_config('CAL_ROOT') / 'starship'

    def list_choices(self):
        names = {}
        for path in self.base_path.iterdir():
            yield path.stem


class MicrophoneLoader(BaseLoader):
    pass


class CFTSMicrophoneLoader(MicrophoneLoader):

    label = 'CFTS'
    base_path = get_config('CAL_ROOT') / 'microphone'

    def list_choices(self):
        names = {}
        for path in self.base_path.iterdir():
            yield path.stem

    def current_calibration(self, name):
        path = self.base_path / name
        candidates = sorted(path.glob(f'* {name}'))
        return(candidates[-1])

    def load(self, name):
        current_cal = self.current_calibration(name)
        sens_file = current_cal / 'microphone_sensitivity.json'
        cal = json.loads(sens_file.read_text())
        sens = cal['mic sens overall (mV/Pa)']
        attrs = {
            'name': name,
            'calibration_file': str(current_cal),
            'calibration': cal,
            'loader': self.qualname,
        }
        return FlatCalibration.from_mv_pa(sens, attrs=attrs)


def list_starship_connections():
    '''
    List all starships found in the IO Manifest
    '''
    starships = {}
    manifest = load_manifest(f'{get_default_io()}.IOManifest')()
    for channel in manifest.find_all('starship', regex=True):
        # Strip quotation marks off 
        _, starship_id, starship_output = channel.name.split('_')
        starships.setdefault(starship_id, []).append(starship_output)

    choices = {}
    for name, channels in starships.items():
        for c in ('microphone', 'primary', 'secondary'):
            if c not in channels:
                raise ValueError(f'Must define starship_{name}_{c} channel')
        choices[name] = f'starship_{name}'

    if len(choices) == 0:
        raise ValueError(NO_STARSHIP_ERROR)

    return choices


starship_manager = CalibrationManager()
starship_manager.register('cfts.util.EPLStarshipLoader')
#starship_manager.register('cfts.util.CFTSStarshipLoader')

microphone_manager = CalibrationManager()
microphone_manager.register('cfts.util.CFTSMicrophoneLoader')


if __name__ == '__main__':
    print(microphone_manager.list_choices())
    print(microphone_manager.load('cfts.util.CFTSMicrophoneLoader::GRAS-40DP'))
