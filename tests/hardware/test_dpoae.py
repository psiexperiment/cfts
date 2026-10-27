'''
DPOAE (input-output, DPgram and dual) through the loopback: each
Starship's primary (f1) and secondary (f2) outputs are summed into its
microphone, so each setting's segment of the recording should be exactly
the two tones psi logs, starting on the logged sample, at the requested
levels, with nothing at 2f1 - f2.
'''
import itertools

import numpy as np
import pandas as pd
import pytest

from psidata.api import Recording

from dpoae_analysis import analyze_dpoae
import plots
from report import write_report
from runner import (measure_run_latency, require_routes, run_paradigm,
                    starship_env)


F2_FREQUENCIES = [4000, 8000]
L2_LEVELS = [50, 70]
F2_F1_RATIO = 1.2
L1_L2_DIFFERENCE = 10
N_FFT = 2
N_TIME = 4

#: Plugins the DPOAE paradigms need besides the required ones: their setup
#: looks up the microphone FFT plot.
PLUGINS = ['microphone_fft_view']

#: The paradigm records each setting for well under a minute, so a run that
#: takes longer than this has hung (psi logs errors in a synchronized start
#: without stopping).
TIMEOUT = 120


def imul(x, y):
    '''
    `psi.context` rounding: `x` to the nearest multiple of `y`.
    '''
    return np.round(np.asarray(x) / y) * y


def load_segments(path, store, microphone, side=''):
    '''
    One row per setting from the DPOAE store CSV, starting at its capture and
    ending at the next one (or the end of the recording). `side` is the
    suffix the dual paradigm adds to output names (e.g., '_left').
    '''
    recording = Recording(path)
    mic = getattr(recording, microphone)
    x = np.asarray(mic[:])[0]
    table = pd.read_csv(path / f'{store}.csv')
    # A side that finishes first in the dual paradigm keeps analyzing until
    # the other is done, so it can save more than one row per setting.
    n_rows = len(table)
    table = table.drop_duplicates('capture', keep='first') \
        .sort_values('capture').reset_index(drop=True)
    p, s = f'primary{side}_tone', f'secondary{side}_tone'
    segments = pd.DataFrame({
        'start': table.capture,
        'end': np.r_[table.capture.values[1:], len(x) / mic.fs],
        'analyzed_end': table.ts_end,
        'f1': table[f'{p}_frequency'],
        'f2': table[f'{s}_frequency'],
        'l1': table[f'{p}_level'],
        'l2': table[f'{s}_level'],
        'requested_f2': table.f2_frequency if not side
        else table[f'{s}_frequency'],
        'rise_time': table[f'{s}_rise_time'],
        'response_window': table.response_window,
        'n_fft': table.n_fft,
        'n_time': table.n_time,
        'psi_f1': table.f1_level,
        'psi_f2': table.f2_level,
        'psi_dp': table.dpoae_level,
        'psi_nf': table.dpoae_noise_floor,
    })
    return x, mic.fs, segments, n_rows - len(table)


def check_settings(checks, segments, order):
    '''
    Every requested f2 and L2 played once, in the order the selector lists
    them, with f1 and L1 derived from them.
    '''
    m = segments
    if order == 'frequency':
        expected = list(itertools.product(F2_FREQUENCIES, L2_LEVELS))
    else:
        expected = [(f, l) for l, f in
                    itertools.product(L2_LEVELS, F2_FREQUENCIES)]
    rw = m.response_window.iloc[0]
    expected = [(imul(f, 1 / rw), l) for f, l in expected]
    played = list(zip(m.f2, m.l2))
    checks.add('settings', played == expected,
               ', '.join(f'{f / 1e3:g}k/{l:g}' for f, l in played),
               ', '.join(f'{f / 1e3:g}k/{l:g}' for f, l in expected),
               f'(f2/L2, in order; sorted by {order} first)')
    f1 = imul(m.f2 / F2_F1_RATIO, 1 / rw)
    checks.add('f1 and L1', np.allclose(m.f1, f1)
               and np.allclose(m.l1 - m.l2, L1_L2_DIFFERENCE),
               f'f1 {", ".join(f"{f:g}" for f in m.f1)} Hz; '
               f'L1 - L2 {", ".join(f"{d:g}" for d in m.l1 - m.l2)} dB',
               f'f2 / {F2_F1_RATIO} to the nearest {1 / rw:g} Hz; '
               f'L2 + {L1_L2_DIFFERENCE}')


def settings():
    return {
        'parameters': {
            'n_fft': N_FFT,
            'n_time': N_TIME,
            'f2_f1_ratio': F2_F1_RATIO,
            'l1_l2_difference': L1_L2_DIFFERENCE,
        },
        'selectors': {'default': {
            'f2_frequency': F2_FREQUENCIES,
            'f2_level': L2_LEVELS,
        }},
    }


def analyze_side(x, fs, segments, routing, starship, latency, engine,
                 order):
    mic = f'starship_{starship}_microphone'
    gain_db = {o: routing.get(f'starship_{starship}_{o}', mic).gain_db
               for o in ('primary', 'secondary')}
    a = analyze_dpoae(x, fs, segments, gain_db, latency, engine)
    check_settings(a.checks, a.segments, order)
    return a, gain_db


def write_dpoae_report(run_dir, title, x, fs, a, info):
    figures = [
        plots.plot_dpoae_timeline(x, fs, a, run_dir / 'timeline.png'),
        plots.plot_dpoae_timing(a, run_dir / 'timing.png'),
        plots.plot_dpoae_levels(a, run_dir / 'levels.png'),
        plots.plot_dpoae_spectra(x, fs, a, run_dir / 'spectra.png'),
    ]
    write_report(run_dir, title, a.checks, figures, info)


def run_single(paradigm, title, order, routing, results_dir):
    require_routes(routing, {'starship_A_microphone': [
        'starship_A_primary', 'starship_A_secondary']})
    latency = measure_run_latency('starship_A_primary',
                                  'starship_A_microphone')
    path = results_dir / paradigm / 'data'
    run = run_paradigm(paradigm, path,
                       {'engine': {'latency_samples': latency}, **settings()},
                       env=starship_env('A'), plugins=PLUGINS,
                       timeout=TIMEOUT)
    if not run.completed:
        pytest.fail(run.describe())
    x, fs, segments, duplicates = load_segments(path, 'dpoae_store',
                                                'system_microphone')
    engine, = run.wrapup['engines'].values()
    a, gain_db = analyze_side(x, fs, segments, routing, 'A', latency, engine,
                              order)
    info = {'Paradigm': paradigm, 'Starship': 'A', 'Data': path,
            'Log': run.log, 'Sampling rate': f'{fs:.0f} Hz',
            'Route gain': ', '.join(f'{k} {v:.3f} dB'
                                    for k, v in gain_db.items()),
            'Duplicate rows dropped': duplicates}
    write_dpoae_report(path.parent, title, x, fs, a, info)
    return a.checks


@pytest.fixture(scope='module')
def dpoae_io(routing, results_dir):
    return run_single('dpoae_io', 'DPOAE (input-output)', 'frequency',
                      routing, results_dir)


@pytest.fixture(scope='module')
def dpgram(routing, results_dir):
    return run_single('dpgram', 'DPgram', 'level', routing, results_dir)


@pytest.fixture(scope='module')
def dual_dpoae_io(routing, results_dir):
    '''
    Left is Starship A and right is Starship B; each microphone should hear
    only its own Starship.
    '''
    require_routes(routing, {
        'starship_A_microphone': ['starship_A_primary',
                                  'starship_A_secondary'],
        'starship_B_microphone': ['starship_B_primary',
                                  'starship_B_secondary'],
    })
    latency = measure_run_latency('starship_A_primary',
                                  'starship_A_microphone')
    path = results_dir / 'dual_dpoae_io' / 'data'
    run = run_paradigm('dual_dpoae_io', path,
                       {'engine': {'latency_samples': latency}, **settings()},
                       env=starship_env(left='A', right='B'),
                       plugins=PLUGINS, timeout=TIMEOUT)
    if not run.completed:
        pytest.fail(run.describe())
    engine, = run.wrapup['engines'].values()
    checks = {}
    for side, starship in (('left', 'A'), ('right', 'B')):
        x, fs, segments, duplicates = load_segments(
            path, f'dpoae_{side}_store', f'{side}_microphone', f'_{side}')
        a, gain_db = analyze_side(x, fs, segments, routing, starship,
                                  latency, engine, 'frequency')
        info = {'Paradigm': 'dual_dpoae_io', 'Side': side,
                'Starship': starship, 'Data': path, 'Log': run.log,
                'Sampling rate': f'{fs:.0f} Hz',
                'Route gain': ', '.join(f'{k} {v:.3f} dB'
                                        for k, v in gain_db.items()),
                'Duplicate rows dropped': duplicates}
        run_dir = path.parent.parent / f'dual_dpoae_io {side}'
        run_dir.mkdir(exist_ok=True)
        write_dpoae_report(run_dir, f'Dual DPOAE, {side}', x, fs, a, info)
        checks[side] = a.checks
    return checks


class DPOAETests:
    '''
    Tests every DPOAE paradigm shares; subclasses provide `checks`.
    '''
    def test_latency(self, checks):
        checks.assert_passed('engine latency', 'tones start on time')

    def test_stimuli(self, checks):
        checks.assert_passed('level', 'waveform', 'nothing unlogged',
                             'whole cycles')

    def test_analysis(self, checks):
        checks.assert_passed('psi measures level', 'no distortion',
                             'settings long enough', 'tones cover analysis')

    def test_sequence(self, checks):
        checks.assert_passed('settings', 'f1 and L1')


class TestDPOAEIO(DPOAETests):
    @pytest.fixture
    def checks(self, dpoae_io):
        return dpoae_io


class TestDPgram(DPOAETests):
    @pytest.fixture
    def checks(self, dpgram):
        return dpgram


class TestDualDPOAEIO(DPOAETests):
    @pytest.fixture(params=['left', 'right'])
    def checks(self, request, dual_dpoae_io):
        return dual_dpoae_io[request.param]
