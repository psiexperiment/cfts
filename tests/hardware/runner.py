'''
Run a cfts paradigm through the Fireface in a subprocess, configured the way
the cfts launcher would configure it.
'''
from pathlib import Path

import pytest

from psi.loopback import rig as loopback_rig
from psi.loopback import runner as loopback_runner
from psi.loopback.pytest_plugin import require_routes  # noqa: F401

from rig import RIG


HERE = Path(__file__).parent

#: Every output is set to this calibration by the `debug_calibration` plugin:
#: 1 Vrms is 130 dB SPL.
OUTPUT_CAL_DB = 130

UNITY_CALIBRATION = 'cftscal.objects.UnityInputCalibration'

PARADIGM_MODULES = ('cfts.paradigms',)


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
                 manifests=None, timeout=300):
    '''
    Run `paradigm` until it finishes, saving its data to `path`, with every
    output calibrated by `debug_calibration` (see
    `psi.loopback.runner.run_paradigm`).
    '''
    env = {'CFTSCAL_IO': RIG.io_manifest, **(env or {})}
    plugins = ['debug_calibration'] + list(plugins or [])
    return loopback_runner.run_paradigm(
        RIG, paradigm, path, settings, modules=PARADIGM_MODULES, env=env,
        plugins=plugins, manifests=manifests, timeout=timeout, cwd=HERE)


def require_paradigm(paradigm):
    if not loopback_runner.paradigm_available(paradigm, PARADIGM_MODULES,
                                              HERE):
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


def measure_run_latency(output, input):
    try:
        return loopback_rig.measure_run_latency(RIG, output, input)
    except RuntimeError as e:
        pytest.fail(str(e))
