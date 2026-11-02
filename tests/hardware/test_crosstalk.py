'''
Binaural crosstalk (`binaural_crosstalk`) through the loopback.

The left ear is Starship A and the right ear Starship B, each on its primary
output, each routed only to its own microphone. So the other ear gets
nothing: every band of the interaural attenuation should be a lower bound
(the other microphone at its noise floor), far above the criterion, while
each ear's own chirp arrives at the level requested.
'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from psidata.api import Recording
from psi.loopback.report import Checks, write_report

from cfts.paradigms.crosstalk_util import EARS
import plots
from rig import INPUT_DBFS
from runner import (OUTPUT_CAL_DB, measure_run_latency, require_paradigm,
                    require_routes, run_paradigm, starship_env)


PARAMETERS = {
    'left_output': 'primary',
    'right_output': 'primary',
    'crosstalk_start_frequency': 4e3,
    'crosstalk_end_frequency': 40e3,
    'crosstalk_level': 90,
    'crosstalk_chirp_duration': 20e-3,
    'crosstalk_averages': 32,
    'crosstalk_rate': 20,
    'crosstalk_criterion': 30,
}

#: Epoch after the chirp (`crosstalk.EPOCH_TAIL`).
EPOCH_TAIL = 5e-3

LEVEL_TOLERANCE = 0.5

#: Without a crossed route, the other microphone has only the Fireface's
#: noise, far below the chirp.
MIN_ATTENUATION = 80

ROUTES = {
    'starship_A_microphone': ['starship_A_primary'],
    'starship_B_microphone': ['starship_B_primary'],
}

STARSHIPS = {'left': 'A', 'right': 'B'}


def run_crosstalk(routing, results_dir):
    require_paradigm('binaural_crosstalk')
    require_routes(routing, ROUTES)
    crossed = [f'{o} -> {i}' for o, i in (
        ('starship_A_primary', 'starship_B_microphone'),
        ('starship_B_primary', 'starship_A_microphone'))
        if routing.get(o, i) is not None]
    if crossed:
        pytest.skip('The mixer crosses the Starships: ' + ', '.join(crossed))

    latency = measure_run_latency('starship_A_primary',
                                  'starship_A_microphone')
    path = results_dir / 'binaural_crosstalk' / 'data'
    run = run_paradigm('binaural_crosstalk', path,
                       {'engine': {'latency_samples': latency},
                        'parameters': PARAMETERS},
                       env=starship_env(left='A', right='B'), timeout=120)
    if not run.completed:
        pytest.fail(run.describe())

    bands = pd.read_csv(path / 'crosstalk_bands.csv')
    summary = pd.read_csv(path / 'crosstalk_summary.csv')
    recording = Recording(path)
    p = PARAMETERS
    checks = Checks()

    # How many chirps played: each is one burst on its own microphone.
    n = int(p['crosstalk_averages'])
    counts = {}
    for ear, starship in STARSHIPS.items():
        signal = getattr(recording, f'{ear}_microphone')
        x = np.asarray(signal[:])[0]
        fs = signal.fs
        hop = int(fs / p['crosstalk_rate'])
        frames = len(x) // hop
        rms = np.sqrt((x[:frames * hop].reshape(frames, hop) ** 2).mean(axis=1))
        counts[ear] = int((rms > rms.max() / 10).sum())
    checks.add('chirps played', all(c == n for c in counts.values()),
               ', '.join(f'{e} {c}' for e, c in counts.items()),
               f'{n} per ear',
               'Chirp-length frames on each microphone: the run stops once '
               'each ear has its averages, without extra chirps.')

    # Each ear's chirp, at its own microphone: the bands add up to the
    # chirp's level over the epoch.
    duration = p['crosstalk_chirp_duration']
    for ear, starship in STARSHIPS.items():
        gain = routing.get(f'starship_{starship}_primary',
                           f'starship_{starship}_microphone').gain_db
        expected = p['crosstalk_level'] - OUTPUT_CAL_DB + INPUT_DBFS + gain \
            + 10 * np.log10(duration / (duration + EPOCH_TAIL))
        g = bands[bands.ear == ear]
        total = 10 * np.log10(np.sum(10 ** (g.ipsi_level / 10)))
        checks.add(f'{ear} level', abs(total - expected) <= LEVEL_TOLERANCE,
                   f'{total - expected:+.3f} dB', f'± {LEVEL_TOLERANCE} dB',
                   f'Sum of the {ear} chirp\'s bands on the {ear} microphone '
                   're requested (through the route, over the epoch).')

    worst = bands.attenuation.min()
    checks.add('attenuation', worst >= MIN_ATTENUATION
               and bands.lower_bound.all(),
               f'at least {worst:.1f} dB; '
               f'{bands.lower_bound.sum()} of {len(bands)} bands at the '
               'noise floor', f'≥ {MIN_ATTENUATION} dB, all at the noise '
               'floor', 'No crossed route, so the other ear gets only noise.')
    checks.add('summary', len(summary) == 2 and summary.passed.all()
               and np.allclose(sorted(summary.min_attenuation),
                               sorted(bands.groupby('ear').attenuation.min())),
               ', '.join(f'{r.ear} {r.min_attenuation:.1f} dB'
                         for r in summary.itertuples()),
               'both ears pass, worst band as in the bands table')

    run_dir = path.parent
    figures = [plot_bands(bands, run_dir / 'crosstalk.png')]
    info = {'Paradigm': 'binaural_crosstalk', 'Left': 'Starship A primary',
            'Right': 'Starship B primary', 'Data': path, 'Log': run.log}
    write_report(run_dir, 'Binaural crosstalk', checks, figures, info)
    return checks


def plot_bands(bands, path):
    figure, (ax_level, ax_att) = plt.subplots(1, 2, figsize=(9, 3.2))
    for i, ear in enumerate(EARS):
        g = bands[bands.ear == ear]
        f = g.frequency / 1e3
        ax_level.semilogx(f, g.ipsi_level, f'C{i}-o', ms=3,
                          label=f'{ear} chirp, {ear} mic')
        ax_level.semilogx(f, g.contra_level, f'C{i}--',
                          label=f'{ear} chirp, other mic')
        ax_level.semilogx(f, g.contra_noise_floor, f'C{i}:',
                          label='noise floor, other mic')
        ax_att.semilogx(f, g.attenuation, f'C{i}-o', ms=3, mfc='none',
                        label=f'from {ear} (lower bound)')
    ax_level.set_ylabel('dB re 1 V per band')
    ax_att.set_ylabel('Interaural attenuation (dB)')
    for ax in (ax_level, ax_att):
        ax.set_xlabel('Frequency (kHz)')
        ax.legend(fontsize=7)
    figure.tight_layout()
    return plots.save(figure, path)


@pytest.fixture(scope='module')
def checks(routing, results_dir):
    return run_crosstalk(routing, results_dir)


class TestCrosstalk:

    def test_chirps(self, checks):
        checks.assert_passed('chirps played')

    def test_levels(self, checks):
        checks.assert_passed('left level', 'right level')

    def test_attenuation(self, checks):
        checks.assert_passed('attenuation', 'summary')
