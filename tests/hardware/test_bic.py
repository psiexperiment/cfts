'''
Binaural interaction component of the ABR (`abr_bic`, `abr_bic_click`)
through the loopback.

The left ear is Starship A and the right ear Starship B, each on its primary
output. Each Starship's microphone records its own ear and the EEG records
both (see `bic_analysis`). Each paradigm runs `sequential` with jittered
timing and `interleaved_random` with steady timing; all alternate polarity.
'''
import numpy as np
import pandas as pd
import pytest

from psidata.api import Recording
from psi.loopback.report import write_report

from abr_analysis import check_intervals, check_sequence
from bic_analysis import analyze_bic, CONDITIONS
from cfts.paradigms.bic_util import CLICK, ITD, TONE
import plots
from runner import (OUTPUT_CAL_DB, eeg_env, measure_run_latency,
                    require_paradigm, require_routes, run_paradigm,
                    starship_env)


#: Kept short so each run takes seconds.
FREQUENCIES = [2000, 8000]
LEVELS = [60, 80]
#: ITDs: whole samples at 48 and 96 kHz, both signs, and one (125 µs) that
#: is a whole cycle at 8 kHz, so only the envelope shows it there.
ITDS = [0, 125, -1000]
AVERAGES = 16
RATE = 40
BURST_DURATION = 5e-3
BURST_RISE_TIME = 0.5e-3
CLICK_DURATION = 100e-6

#: Paradigm and stimulus of each variant.
VARIANTS = {
    'tone': ('abr_bic', TONE),
    'click': ('abr_bic_click', CLICK),
}

#: Rate jitter (ms) for the sequential run.
JITTER_MS = 2

RUNS = [(variant, ordering, jitter)
        for variant in VARIANTS
        for ordering, jitter in (('sequential', JITTER_MS),
                                 ('interleaved_random', 0))]

ROUTES = {
    'eeg': ['starship_A_primary', 'starship_B_primary'],
    'starship_A_microphone': ['starship_A_primary'],
    'starship_B_microphone': ['starship_B_primary'],
}


def run_bic(variant, ordering, jitter_ms, routing, results_dir):
    paradigm, st = VARIANTS[variant]
    require_paradigm(paradigm)
    require_routes(routing, ROUTES)
    crossed = [f'{o} -> {i}' for o, i in (
        ('starship_A_primary', 'starship_B_microphone'),
        ('starship_B_primary', 'starship_A_microphone'))
        if routing.get(o, i) is not None]
    if crossed:
        pytest.skip('The mixer crosses the Starships: ' + ', '.join(crossed))

    latency = measure_run_latency('starship_A_primary', 'eeg')
    settings = {
        'engine': {'latency_samples': latency},
        'parameters': {
            'left_output': 'primary',
            'right_output': 'primary',
            'averages': AVERAGES,
            'rate': RATE,
            'rate_jitter_ms': jitter_ms,
            'ordering': ordering,
            'alternate_polarity': True,
            'reject_threshold': float('inf'),
        },
        'selectors': {'default': {
            'bic_level': LEVELS,
            ITD: ITDS,
        }},
    }
    if st is TONE:
        settings['parameters'].update({
            'bic_burst_duration': BURST_DURATION,
            'bic_burst_rise_time': BURST_RISE_TIME,
        })
        settings['selectors']['default']['bic_frequency'] = FREQUENCIES
    else:
        settings['parameters']['bic_click_duration'] = CLICK_DURATION
    path = results_dir / f'{paradigm} {ordering}' / 'data'
    run = run_paradigm(paradigm, path, settings,
                       env={**starship_env(left='A', right='B'), **eeg_env()},
                       manifests={'bic_microphones.BICMicrophonesManifest': {}})
    if not run.completed:
        pytest.fail(run.describe())

    recording = Recording(path)
    eeg = recording.eeg
    left_mic = recording.left_microphone
    right_mic = recording.right_microphone
    md = pd.read_csv(path / 'erp_metadata.csv')
    eeg_fs, mic_fs = eeg.fs, left_mic.fs
    epoch_n = int(round(md.epoch_size[0] * eeg_fs))
    gains = {
        'left_eeg': routing.get('starship_A_primary', 'eeg').gain_db,
        'right_eeg': routing.get('starship_B_primary', 'eeg').gain_db,
        'left_mic': routing.get('starship_A_primary',
                                'starship_A_microphone').gain_db,
        'right_mic': routing.get('starship_B_primary',
                                 'starship_B_microphone').gain_db,
    }
    engine, = run.wrapup['engines'].values()
    a = analyze_bic(np.asarray(eeg[:])[0], eeg_fs,
                    np.asarray(left_mic[:])[0], np.asarray(right_mic[:])[0],
                    mic_fs, md, gains, latency, engine, epoch_n,
                    OUTPUT_CAL_DB)
    checks = a.checks
    stimuli = a.stimuli

    if jitter_ms:
        ioi = np.diff(md.t0.values)
        lb = 1 / RATE - jitter_ms * 1e-3 - 1 / eeg_fs
        ub = 1 / RATE + jitter_ms * 1e-3 + 1 / eeg_fs
        inside = (ioi >= lb) & (ioi <= ub)
        checks.add('repetition rate', inside.all() and ioi.std() > 1 / eeg_fs,
                   f'{(~inside).sum()} of {len(ioi)} intervals outside; '
                   f'{ioi.min() * 1e3:.3f} to {ioi.max() * 1e3:.3f} ms',
                   f'{1e3 / RATE:g} ± {jitter_ms} ms, and varying')
        expected_ioi = None
    else:
        check_intervals(checks, md.t0.values, eeg_fs, 1 / RATE)
        expected_ioi = 1 / RATE

    n_values = len(ITDS)
    n_groups = len(LEVELS) * (len(FREQUENCIES) if st is TONE else 1)
    n_conditions = n_groups * (2 + n_values)
    check_sequence(checks, stimuli, CONDITIONS, n_conditions, AVERAGES)
    if ordering == 'sequential':
        check_grouping(checks, stimuli, n_values, n_groups)

    run_dir = path.parent
    epoch_span = epoch_n / eeg_fs
    figures = [
        plots.plot_timeline(a.eeg.fit, stimuli.onset.values, epoch_span,
                            run_dir / 'timeline.png'),
        plots.plot_timing(a.eeg.fit, run_dir / 'timing.png', expected_ioi),
        plots.plot_waveforms(stimuli, a.eeg.fit, a.templates,
                             run_dir / 'waveforms.png', ['stimulus'],
                             ['frequency', 'level']),
        plots.plot_epochs(a.eeg.epochs, a.eeg.alignment, a.eeg.cancellation,
                          run_dir / 'epochs.png', ['stimulus'],
                          ['frequency', 'level']),
        plots.plot_bic(a.bic, run_dir / 'bic.png'),
        plots.plot_itd(a.waveforms, a.itd, run_dir / 'itd.png'),
    ]
    info = {'Paradigm': paradigm, 'Left': 'Starship A primary',
            'Right': 'Starship B primary', 'Ordering': ordering,
            'Rate jitter': f'{jitter_ms} ms', 'Data': path, 'Log': run.log,
            'EEG sampling rate': f'{eeg_fs:.0f} Hz',
            'Microphone sampling rate': f'{mic_fs:.0f} Hz',
            'Route gain': ', '.join(f'{k} {v:.3f} dB'
                                    for k, v in gains.items()),
            'Stimuli': len(stimuli)}
    write_report(run_dir, f'ABR binaural interaction ({st.name}), '
                 f'{ordering}', checks,
                 figures, info)
    return checks


def check_grouping(checks, stimuli, n_values, n_groups):
    '''
    `sequential` presents each frequency and level (clicks: level) in turn,
    with all of its conditions randomly interleaved.
    '''
    block = list(zip(stimuli.frequency, stimuli.level, strict=True))
    starts = [i for i in range(len(block)) if i == 0 or block[i] != block[i - 1]]
    checks.add('frequency and level in turn', len(starts) == n_groups,
               f'{len(starts)} runs of one frequency and level',
               f'{n_groups}')
    # The first block of each group has every condition and polarity once.
    size = (2 + n_values) * 2
    complete = all(
        len(set(zip(stimuli.stimulus[s:s + size],
                    stimuli.polarity[s:s + size], strict=True))) == size
        for s in starts)
    checks.add('conditions interleaved', complete,
               'every condition and polarity in the first '
               f'{size} trials of each group' if complete else
               'some group starts with a condition repeated',
               f'all {size} in the first {size}')


@pytest.fixture(scope='module', params=RUNS,
                ids=[f'{c}-{o}' for c, o, _ in RUNS])
def checks(request, routing, results_dir):
    variant, ordering, jitter_ms = request.param
    return run_bic(variant, ordering, jitter_ms, routing, results_dir)


class TestBIC:

    def test_latency(self, checks):
        checks.assert_passed('engine latency', 'latency compensated')

    def test_timing(self, checks):
        checks.assert_passed('onset jitter', 'repetition rate')

    def test_alignment(self, checks):
        checks.assert_passed('trials aligned', 'trials identical')

    def test_polarity_cancels(self, checks):
        checks.assert_passed('polarity cancels')

    def test_stimuli(self, checks):
        checks.assert_passed('level', 'polarity', 'waveform',
                             'nothing unlogged')

    def test_sequence(self, checks):
        checks.assert_passed('conditions', 'averages', 'polarity alternates')

    def test_grouping(self, checks):
        if 'frequency and level in turn' not in checks:
            pytest.skip('Only the sequential ordering groups')
        checks.assert_passed('frequency and level in turn',
                             'conditions interleaved')

    def test_ears(self, checks):
        checks.assert_passed('left level', 'left waveform', 'left ear only',
                             'right level', 'right waveform',
                             'right ear only', 'ears in step')

    def test_itd(self, checks):
        checks.assert_passed('ITD')

    def test_itd_shift(self, checks):
        checks.assert_passed('ITD shifts')

    def test_psi_bic(self, checks):
        checks.assert_passed('psi BIC')
