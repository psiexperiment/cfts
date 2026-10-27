'''
MEMR paradigms through the loopback.

The elicitor plays on the test Starship's secondary output and the probe on
the nontest Starship's primary; each is recorded on its own Starship's
microphone (both on Starship A for ipsilateral). The paradigms need the
turntable, which `rig_io.IOManifest` supplies as stationary.

For now these only check that each paradigm runs to the end on its own.
None of them currently start: their grouped MEMR plots override `_y`, which
psi's grouped epoch plots no longer call, and the sweep's
`elicitor_noise_transform` lambda fails psi's context validation. Stimulus
checks (probe and elicitor timing and level, with the elicitor regenerated
from its saved seed) belong here once they do.
'''
import pytest

from runner import (measure_run_latency, require_routes, run_paradigm,
                    starship_env)


#: Short trials so each run takes seconds.
INTERLEAVED = {'elicitor_n': 3, 'trial_iti': 0.2, 'trial_n': 2}
SIMULTANEOUS = {'trial_duration': 2, 'elicitor_onset': 1,
                'elicitor_duration': 0.5, 'norm_window': 0.5,
                'probe_rate': 40, 'trial_iti': 0.2, 'trial_n': 2}
SWEEP = {'min_level': 60, 'max_level': 80, 'ramp_rate': 40, 'trial_n': 2,
         'trial_iti': 0.2}
ELICITOR_LEVELS = [50, 60]

RUNS = [
    ('memr_interleaved_click', 'ipsi', INTERLEAVED),
    ('memr_interleaved_click', 'contra', INTERLEAVED),
    ('memr_interleaved_chirp', 'contra', INTERLEAVED),
    ('memr_simultaneous_click', 'contra', SIMULTANEOUS),
    ('memr_simultaneous_chirp', 'contra', SIMULTANEOUS),
    ('memr_sweep_click', 'contra', SWEEP),
]

#: A healthy run takes well under a minute.
TIMEOUT = 120


@pytest.mark.parametrize('paradigm, mode, parameters', RUNS,
                         ids=[f'{p} {m}' for p, m, _ in RUNS])
def test_runs(paradigm, mode, parameters, routing, results_dir):
    nontest = 'A' if mode == 'ipsi' else 'B'
    require_routes(routing, {
        'starship_A_microphone': ['starship_A_secondary'],
        f'starship_{nontest}_microphone': [f'starship_{nontest}_primary'],
    })
    latency = measure_run_latency('starship_A_primary',
                                  'starship_A_microphone')
    settings = {
        'engine': {'latency_samples': latency},
        # Flat elicitor, so its level is simple to check.
        'parameters': {**parameters, 'elicitor_audiogram_weighting': 'no'},
    }
    if 'sweep' not in paradigm:
        settings['selectors'] = {'default': {
            'elicitor_bandlimited_noise_level': ELICITOR_LEVELS,
        }}
    path = results_dir / f'{paradigm} {mode}' / 'data'
    run = run_paradigm(paradigm, path, settings,
                       env=starship_env(test='A', nontest=nontest),
                       timeout=TIMEOUT)
    if not run.completed:
        pytest.fail(run.describe())
