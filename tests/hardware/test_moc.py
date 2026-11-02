'''
MOC suppression of DPOAEs (`moc_dpoae`) through the loopback.

The probe Starship (A, the test ear) plays f1 on its primary and f2 on its
secondary, continuously, into its microphone; the elicitor Starship (B, the
nontest ear) plays the noise on its secondary into its own microphone.
`moc_dpoae contra` checks the stimuli and what psi measured with no DP (a
loopback has no cochlea). `moc_dpoae synthetic` adds a DP that drops by a
known amount while the elicitor plays (`moc_synthetic.enaml`), and checks
that psi's time course, average and suppression recover it.
'''
import numpy as np
import pandas as pd
import pytest

from psidata.api import Recording
from psi.loopback.report import write_report

from moc_analysis import analyze_moc
import plots
from runner import (measure_run_latency, require_routes, run_paradigm,
                    starship_env)


#: Short trials so each run takes seconds; block edges fall on the window
#: edges.
PARAMETERS = {
    'f2_frequency': 8000,
    'f2_level': 60,
    'f2_f1_ratio': 1.2,
    'l1_l2_difference': 10,
    'response_window': 0.1,
    'baseline_duration': 1.0,
    'elicitor_duration': 1.0,
    'elicitor_rise_time': 20e-3,
    'recovery_duration': 1.0,
    'trial_n': 3,
    # Not the default, to check that each trial adds its number to it.
    'elicitor_seed': 7,
    'elicitor_frozen': False,
    'elicitor_level': 70,
    'elicitor_fl': 4e3,
    'elicitor_fh': 32e3,
    # Flat elicitor, so it is simple to regenerate.
    'elicitor_max_equalize': 0,
    'elicitor_audiogram_weighting': 'no',
}

SYNTHETIC = {'level': 30, 'suppression': 3}

#: A healthy run takes well under a minute.
TIMEOUT = 120

ROUTES = {
    'starship_A_microphone': ['starship_A_primary', 'starship_A_secondary'],
    'starship_B_microphone': ['starship_B_secondary'],
}


def run_moc(name, routing, results_dir, synthetic=None):
    require_routes(routing, ROUTES)
    crossed = [f'{o} -> {i}' for o, i in (
        ('starship_B_secondary', 'starship_A_microphone'),
        ('starship_A_primary', 'starship_B_microphone'),
        ('starship_A_secondary', 'starship_B_microphone'))
        if routing.get(o, i) is not None]
    if crossed:
        pytest.skip('The mixer crosses the Starships: ' + ', '.join(crossed))
    latency = measure_run_latency('starship_A_primary',
                                  'starship_A_microphone')
    path = results_dir / name / 'data'
    manifests = None
    if synthetic is not None:
        manifests = {'moc_synthetic.SyntheticDPManifest': synthetic}
    run = run_paradigm('moc_dpoae', path,
                       {'engine': {'latency_samples': latency},
                        'parameters': PARAMETERS},
                       env=starship_env(test='A', nontest='B'),
                       manifests=manifests, timeout=TIMEOUT)
    if not run.completed:
        pytest.fail(run.describe())

    recording = Recording(path)
    probe = recording.probe_microphone
    fs = probe.fs
    probe = np.asarray(probe[:])[0]
    elicitor = np.asarray(recording.elicitor_microphone[:])[0]
    trials = pd.read_csv(path / 'moc_trial_store.csv')
    blocks = pd.read_csv(path / 'moc_block_store.csv')
    p = {**PARAMETERS, 'trial_duration': PARAMETERS['baseline_duration']
         + PARAMETERS['elicitor_duration']
         + PARAMETERS['recovery_duration']}
    gain = {
        'primary': routing.get('starship_A_primary',
                               'starship_A_microphone').gain_db,
        'secondary': routing.get('starship_A_secondary',
                                 'starship_A_microphone').gain_db,
        'elicitor': routing.get('starship_B_secondary',
                                'starship_B_microphone').gain_db,
    }
    engine, = run.wrapup['engines'].values()
    a = analyze_moc(probe, elicitor, fs, trials, blocks, p, gain, latency,
                    engine, synthetic)

    run_dir = path.parent
    figures = [
        plots.plot_moc_timeline(probe, elicitor, fs, a, p,
                                run_dir / 'timeline.png'),
        plots.plot_moc_blocks(a, p, run_dir / 'blocks.png'),
        plots.plot_timing(a.elicitor_fit, run_dir / 'elicitor_timing.png'),
    ]
    info = {'Paradigm': 'moc_dpoae', 'Probe Starship': 'A',
            'Elicitor Starship': 'B', 'Data': path, 'Log': run.log,
            'Sampling rate': f'{fs:.0f} Hz',
            'Route gain': ', '.join(f'{k} {v:.3f} dB'
                                    for k, v in gain.items())}
    if synthetic is not None:
        info['Synthetic DP'] = f'{synthetic["level"]} dB SPL, ' \
            f'{synthetic["suppression"]} dB suppression'
    title = 'MOC suppression' + (', synthetic DP' if synthetic else '')
    write_report(run_dir, title, a.checks, figures, info)
    return a.checks


@pytest.fixture(scope='module')
def moc(routing, results_dir):
    return run_moc('moc_dpoae contra', routing, results_dir)


@pytest.fixture(scope='module')
def moc_synthetic(routing, results_dir):
    return run_moc('moc_dpoae synthetic', routing, results_dir, SYNTHETIC)


class MOCTests:
    '''
    Tests both runs share; subclasses provide `checks`.
    '''
    def test_latency(self, checks):
        checks.assert_passed('engine latency', 'tones start on time')

    def test_primaries(self, checks):
        checks.assert_passed('primaries continuous', 'primary level',
                             'elicitor isolated')

    def test_elicitor(self, checks):
        checks.assert_passed('elicitor timing', 'elicitor level',
                             'elicitor waveform')

    def test_sequence(self, checks):
        checks.assert_passed('trial sequence', 'psi blocks')

    def test_psi_primaries(self, checks):
        checks.assert_passed('psi measures primaries')


class TestMOC(MOCTests):
    @pytest.fixture
    def checks(self, moc):
        return moc

    def test_no_dp(self, checks):
        checks.assert_passed('no distortion', 'psi sees no DP')


class TestMOCSynthetic(MOCTests):
    @pytest.fixture
    def checks(self, moc_synthetic):
        return moc_synthetic

    def test_synthetic_dp(self, checks):
        checks.assert_passed('synthetic DP played', 'psi DP time course',
                             'psi average', 'psi suppression')
