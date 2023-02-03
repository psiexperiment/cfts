from pathlib import Path

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


class StarshipCalManager:

    base_path = get_config('CAL_ROOT') / 'starship'

    def get_calibration(self, starship_name):
        path = self.base_path / starship_name
        return sorted(list(path.iterdir()))[-1]

    def list_starships(self):
        starships = {}
        for path in self.base_path.iterdir():
            starships[path.stem] = {
                'filename': self.get_calibration(path.stem),
                'loader': 'cfts'
            }
        return starships


starship_cal_manager = StarshipCalManager()


def list_epl_starships():
    base_path = Path(r'C:\Data\Probe Tube Calibrations')
    calibrations = {}
    for calfile in base_path.glob('*_ProbeTube.calib'):
        name = calfile.stem.rsplit('_', 1)[0]
        calibrations[f'{name} (EPL)'] = {
            'filename': calfile,
            'loader': 'EPL'
        }
    return calibrations


def list_starship_calibrations():
    result = list_epl_starships()
    result.update(starship_cal_manager.list_starships())
    return result


def list_starship_connections():
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


if __name__ == '__main__':
    #print(list_starships())
    pass
