'''
Run a paradigm through the Fireface in a subprocess, configured the way the
cfts launcher would configure it.
'''
from dataclasses import dataclass
import functools
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from rig import ENGINE_BLOCKSIZE, ENGINE_FS, IO_MANIFEST, measure_latency


HERE = Path(__file__).parent

#: Every output is set to this calibration by the `debug_calibration` plugin:
#: 1 Vrms is 130 dB SPL.
OUTPUT_CAL_DB = 130

UNITY_CALIBRATION = 'cftscal.objects.UnityInputCalibration'


@dataclass
class RunResult:
    path: Path
    log: Path
    returncode: int
    #: What psi reported when it wrapped up the experiment (None if it never
    #: got that far).
    wrapup: dict

    @property
    def completed(self):
        '''
        True if the experiment ran to the end on its own, rather than being
        stopped by an error or never starting.
        '''
        return self.wrapup is not None and not self.wrapup['stop_reason'] \
            and not self.wrapup['error_message']

    def describe(self):
        if self.wrapup is None:
            return f'psi exited ({self.returncode}) before wrapping up; ' \
                f'see {self.log}'
        if self.wrapup['error_message']:
            return f'psi stopped with an error: ' \
                f'{self.wrapup["error_message"]}\n{self.wrapup["traceback"]}'
        return f'psi stopped early: {self.wrapup["stop_reason"]}'


def starship_env(starship=None, **sides):
    '''
    Environment selecting Starships by side, as the launcher sets it, each
    with a unity microphone calibration and no gain.

    `starship_env('A')` is the test Starship; `starship_env(left='A',
    right='B')` sets `CFTSCAL_LEFT_STARSHIP` and `CFTSCAL_RIGHT_STARSHIP`.
    '''
    if starship is not None:
        sides['test'] = starship
    env = {}
    for side, starship in sides.items():
        name = f'starship_{starship}'
        key = f'CFTSCAL_STARSHIP_{name.upper()}'
        env[f'CFTSCAL_{side.upper()}_STARSHIP'] = name
        env[key] = UNITY_CALIBRATION
        env[f'{key}_GAIN'] = '0'
    return env


def eeg_env(gain=1):
    return {
        'CFTSCAL_INPUT_AMPLIFIER': 'eeg',
        'CFTSCAL_INPUT_AMPLIFIER_EEG_GAIN': str(gain),
        'CFTSCAL_INPUT_AMPLIFIER_EEG_FREQ_LB': '10',
        'CFTSCAL_INPUT_AMPLIFIER_EEG_FREQ_UB': '10000',
    }


def run_paradigm(paradigm, path, settings, env=None, plugins=None,
                 timeout=300):
    '''
    Run `paradigm` until it finishes, saving its data to `path`.

    Parameters
    ----------
    paradigm : str
        Name of the paradigm (e.g., 'abr_io').
    path : Path
        Folder for the data, the settings and the log. Must not exist yet.
    settings : dict
        Context settings, see `autorun_manifest.apply_settings`.
    env : dict
        Extra environment variables (e.g., from `starship_env`).
    plugins : list of str
        Optional plugins to enable, in addition to `debug_calibration`.
    timeout : float
        Seconds to wait before killing a run that hasn't finished.
    '''
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    run_dir = path.parent / f'{path.name} run'
    run_dir.mkdir()
    settings_file = run_dir / 'settings.json'
    settings_file.write_text(json.dumps(settings, indent=2))
    result_file = run_dir / 'result.json'
    log_file = run_dir / 'psi.log'

    plugins = ['debug_calibration'] + list(plugins or [])
    cmd = [
        sys.executable, str(HERE / 'autorun.py'), paradigm, str(path),
        '--io', IO_MANIFEST,
        '--plugins', *plugins,
        '--settings', str(settings_file),
        '--result', str(result_file),
        '--log', str(log_file),
    ]
    full_env = {**os.environ, 'CFTSCAL_IO': IO_MANIFEST, **(env or {})}
    # Not connected to a launcher, and on the sound card the IO manifest
    # names rather than whichever one the launcher last selected.
    for name in ('PSI_WEBSOCKETS_URI', 'PSI_SOUND_DEVICE_NAME',
                 'PSI_SOUND_DEVICE_FS'):
        full_env.pop(name, None)

    try:
        proc = subprocess.run(cmd, env=full_env, timeout=timeout, cwd=HERE,
                              capture_output=True, text=True)
        returncode = proc.returncode
        (run_dir / 'stdout.txt').write_text(proc.stdout + proc.stderr)
    except subprocess.TimeoutExpired as e:
        returncode = None
        out = [o.decode(errors='replace') if isinstance(o, bytes) else o
               for o in (e.stdout, e.stderr) if o]
        (run_dir / 'stdout.txt').write_text(''.join(out) + '\nTIMED OUT\n')

    wrapup = json.loads(result_file.read_text()) \
        if result_file.exists() else None
    return RunResult(path, log_file, returncode, wrapup)


@functools.cache
def paradigm_available(paradigm):
    '''
    True if the installed cfts (or the one `--src` points to) has
    `paradigm`.
    '''
    code = ('import cfts.paradigms\n'
            'from psi.experiment.api import paradigm_manager\n'
            f'paradigm_manager.get_paradigm({paradigm!r})\n')
    proc = subprocess.run([sys.executable, '-c', code], cwd=HERE,
                          capture_output=True)
    return proc.returncode == 0


def require_paradigm(paradigm):
    if not paradigm_available(paradigm):
        pytest.skip(f'{paradigm} is not in this cfts; point --src at a '
                    'checkout that has it')


def find_starship(routing, input_name):
    '''
    The first Starship whose primary output TotalMix routes to `input_name`.
    '''
    for starship in 'AB':
        if routing.get(f'starship_{starship}_primary', input_name):
            return starship
    pytest.skip(f'No Starship primary is routed to {input_name} in TotalMix')


def require_routes(routing, required):
    '''
    Skip unless TotalMix has every route in `required` ({input: [outputs]}).
    '''
    missing = routing.missing(required)
    if missing:
        pytest.skip('TotalMix is missing ' + ', '.join(missing))


def measure_run_latency(output, input):
    '''
    Latency at the engine's sampling rate and blocksize. The first stream
    after the Fireface changes rate comes back thousands of samples off from
    the rest, so measure until two agree.
    '''
    previous = None
    for _ in range(4):
        latency = measure_latency(output, input, ENGINE_FS, ENGINE_BLOCKSIZE)
        if latency == previous:
            return latency
        previous = latency
    pytest.fail(f'Latency from {output} to {input} is not stable')
