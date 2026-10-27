'''
ABR (tone and click) through the loopback: the EEG input records the
Starship's primary output, so every stimulus psi logs should be in the EEG
exactly as specified, and in the window psi averages.
'''
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cftsdata import abr

from abr_analysis import analyze_stimuli, check_intervals, check_sequence
from checks import click, tone_burst
import plots
from report import write_report
from runner import (OUTPUT_CAL_DB, eeg_env, find_starship,
                    measure_run_latency, require_paradigm, run_paradigm,
                    starship_env)


ORDERINGS = ['sequential', 'interleaved_plateau', 'interleaved_ramp',
             'interleaved_random']

#: Kept short so each run takes seconds. Frequencies are an octave apart so
#: the interleaved orderings can space them (and see `interleave_octaves`:
#: two frequencies two octaves apart currently fail).
FREQUENCIES = [2000, 4000, 8000]
LEVELS = [60, 80]
AVERAGES = 16
RATE = 80
BURST_DURATION = 5e-3
BURST_RISE_TIME = 0.5e-3

CLICK_DURATION = 100e-6

#: Rate jitter (ms) for the jittered click run.
JITTER_MS = 2


@dataclass
class EEGRun:
    '''
    A finished run that recorded its stimuli on the EEG input, and what the
    analysis needs to know about it.
    '''
    data: object
    #: One row per trial.
    md: pd.DataFrame
    eeg: np.ndarray
    fs: float
    run_dir: Path
    info: dict
    latency: int
    engine: dict
    route_gain_db: float

    @property
    def epoch_n(self):
        return int(round(self.md.epoch_size[0] * self.fs))


def load_abr(path):
    data = abr.load(path)
    return data, data.erp_metadata.reset_index(drop=True), data.eeg


def run_eeg_paradigm(paradigm, name, settings, routing, results_dir,
                     load=load_abr):
    '''
    Run `paradigm` on the Starship routed to the EEG input, with the latency
    measured just before. `load(path)` returns the data, the trial metadata
    and the EEG signal.
    '''
    require_paradigm(paradigm)
    starship = find_starship(routing, 'eeg')
    output = f'starship_{starship}_primary'
    latency = measure_run_latency(output, 'eeg')
    settings = {'engine': {'latency_samples': latency}, **settings}
    path = results_dir / name / 'data'
    run = run_paradigm(paradigm, path, settings,
                       env={**starship_env(starship), **eeg_env()})
    if not run.completed:
        pytest.fail(run.describe())
    data, md, eeg = load(path)
    route_gain_db = routing.get(output, 'eeg').gain_db
    info = {'Paradigm': paradigm, 'Output': output, 'Data': path,
            'Log': run.log, 'Sampling rate': f'{eeg.fs:.0f} Hz',
            'Route gain': f'{route_gain_db:.3f} dB'}
    engine, = run.wrapup['engines'].values()
    return EEGRun(data, md, np.asarray(eeg[:])[0], eeg.fs, path.parent, info,
                  latency, engine, route_gain_db)



def stimulus_table(md, fs):
    '''
    One stimulus per trial, with the epoch starting at its onset.
    '''
    return md.assign(onset=md.t0,
                     epoch_start=np.round(md.t0 * fs).astype(int))


def write_abr_report(run, title, stimuli, templates, analysis, row, col,
                     expected_ioi=None, info=None):
    a = analysis
    epoch_span = run.epoch_n / run.fs
    figures = [
        plots.plot_timeline(a.fit, stimuli.onset.values, epoch_span,
                            run.run_dir / 'timeline.png'),
        plots.plot_timing(a.fit, run.run_dir / 'timing.png', expected_ioi),
        plots.plot_levels(stimuli, a.fit, a.expected_gain_db,
                          run.run_dir / 'levels.png'),
        plots.plot_waveforms(stimuli, a.fit, templates,
                             run.run_dir / 'waveforms.png', row, col),
        plots.plot_epochs(a.epochs, a.alignment, a.cancellation,
                          run.run_dir / 'epochs.png', row, col),
    ]
    info = {**run.info, **(info or {}), 'Stimuli': len(stimuli)}
    write_report(run.run_dir, title, a.checks, figures, info)


@pytest.fixture(scope='module', params=ORDERINGS)
def abr_tone(request, routing, results_dir):
    ordering = request.param
    settings = {
        'parameters': {
            'averages': AVERAGES,
            'rate': RATE,
            'ordering': ordering,
            'alternate_polarity': True,
            'reject_threshold': float('inf'),
            'target_tone_burst_duration': BURST_DURATION,
            'target_tone_burst_rise_time': BURST_RISE_TIME,
        },
        'selectors': {'default': {
            'target_tone_frequency': FREQUENCIES,
            'target_tone_level': LEVELS,
        }},
    }
    run = run_eeg_paradigm('abr_io', f'abr_io {ordering}', settings,
                           routing, results_dir)
    fs, md = run.fs, run.md
    stimuli = stimulus_table(md, fs)
    templates = [
        tone_burst(fs, r.frequency, r.level, r.burst_duration,
                   r.burst_rise_time, r.burst_envelope, r.polarity, r.phase,
                   cal_db=OUTPUT_CAL_DB)
        for r in md.itertuples()
    ]
    conditions = ['frequency', 'level']
    a = analyze_stimuli(run.eeg, fs, stimuli, templates, conditions,
                        run.epoch_n, run.route_gain_db, run.latency,
                        run.engine)
    check_intervals(a.checks, md.t0.values, fs, 1 / RATE)
    check_sequence(a.checks, md, conditions,
                   len(FREQUENCIES) * len(LEVELS), AVERAGES)
    write_abr_report(run, f'ABR (tone), {ordering}', stimuli, templates, a,
                     ['level'], ['frequency'], 1 / RATE,
                     {'Ordering': ordering})
    return a.checks


@pytest.fixture(scope='module', params=[0, JITTER_MS],
                ids=['steady', 'jittered'])
def abr_click(request, routing, results_dir):
    jitter_ms = request.param
    settings = {
        'parameters': {
            'averages': AVERAGES,
            'rate': RATE,
            'rate_jitter_ms': jitter_ms,
            'click_duration': CLICK_DURATION,
            'click_polarity': 'alternating',
            'reject_threshold': float('inf'),
        },
        'selectors': {'default': {'target_click_level': LEVELS}},
    }
    label = 'jittered' if jitter_ms else 'steady'
    run = run_eeg_paradigm('abr_io_click', f'abr_io_click {label}',
                           settings, routing, results_dir)
    fs, md = run.fs, run.md
    stimuli = stimulus_table(md, fs)
    templates = [click(fs, r.level, r.click_duration, r.polarity,
                       cal_db=OUTPUT_CAL_DB) for r in md.itertuples()]
    conditions = ['level']
    a = analyze_stimuli(run.eeg, fs, stimuli, templates, conditions,
                        run.epoch_n, run.route_gain_db, run.latency,
                        run.engine)
    if jitter_ms:
        ioi = np.diff(md.t0.values)
        lb = 1 / RATE - jitter_ms * 1e-3 - 1 / fs
        ub = 1 / RATE + jitter_ms * 1e-3 + 1 / fs
        inside = (ioi >= lb) & (ioi <= ub)
        a.checks.add('repetition rate', inside.all() and ioi.std() > 1 / fs,
                     f'{(~inside).sum()} of {len(ioi)} intervals outside; '
                     f'{ioi.min() * 1e3:.3f} to {ioi.max() * 1e3:.3f} ms',
                     f'{1e3 / RATE:g} ± {jitter_ms} ms, and varying')
        expected_ioi = None
    else:
        check_intervals(a.checks, md.t0.values, fs, 1 / RATE)
        expected_ioi = 1 / RATE
    check_sequence(a.checks, md, conditions, len(LEVELS), AVERAGES)
    write_abr_report(run, f'ABR (click), {label}', stimuli, templates, a,
                     [], ['level'], expected_ioi,
                     {'Rate jitter': f'{jitter_ms} ms'})
    return a.checks


class ABRTests:
    '''
    Tests every ABR paradigm shares; subclasses provide the `checks` fixture.
    '''
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


class TestTone(ABRTests):
    @pytest.fixture
    def checks(self, abr_tone):
        return abr_tone


class TestClick(ABRTests):
    @pytest.fixture
    def checks(self, abr_click):
        return abr_click
