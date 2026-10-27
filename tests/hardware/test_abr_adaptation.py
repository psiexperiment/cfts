'''
ABR adaptation (tone and click trains) through the loopback.

Each trial is a train of bursts followed by a pause; psi averages the whole
train and also splits it into one epoch per burst (`SplitBursts`). Every
burst has to land where psiaudio's `repeat` puts it, and every per-burst
epoch has to start on its burst.

These paradigms are on the `abr-adaptation` branch. Run with `--src`
pointing at a checkout of it (its `src` folder), otherwise they are skipped.
'''
import numpy as np
import pandas as pd
import pytest

from psiaudio.pipeline import PipelineData

from abr_analysis import analyze_stimuli, check_intervals, check_sequence
from checks import click, tone_burst
from runner import OUTPUT_CAL_DB
from test_abr import ABRTests, run_eeg_paradigm, write_abr_report


FREQUENCIES = [2000, 4000]
LEVELS = [60, 80]
AVERAGES = 4
N_BURSTS = 5
#: Not a whole number of samples per period at 96 kHz (1371.43), so the
#: burst positions and the per-burst epochs have to agree on rounding.
RATE = 70
PAUSE = 0.05
BURST_DURATION = 5e-3
BURST_RISE_TIME = 0.5e-3
CLICK_DURATION = 100e-6

#: Per-burst epoch duration (`ABRAdaptationBaseManifest.burst_span`).
BURST_SPAN = 8.5e-3


def split_starts(n, rate, sequence_n, fs):
    '''
    Sample, within the train's epoch, at which psi starts each per-burst
    epoch, found by running the paradigm's own `split_bursts` on an epoch
    whose samples are their own indices.
    '''
    import enaml
    with enaml.imports():
        from cfts.paradigms.abr_adaptation import split_bursts
    result = []
    split = split_bursts(n, rate, BURST_SPAN, result.append)
    epoch = PipelineData(np.arange(sequence_n, dtype=float)[np.newaxis],
                         fs=fs, s0=0, metadata={})
    split.send([epoch])
    return [int(e[0]) for e in np.asarray(result[0])[:, 0, :]]


def burst_table(md, fs):
    '''
    One row per burst. Burst `i` is played `i` periods into the train, where
    psiaudio rounds the period to whole samples; psi's per-burst epoch starts
    wherever `split_bursts` puts it.
    '''
    rows = []
    for r in md.itertuples(index=False):
        s_period = int(round(fs / r.rate))
        s0 = int(round(r.t0 * fs))
        starts = split_starts(int(r.n_bursts), r.rate,
                              int(round(r.epoch_size * fs)), fs)
        for i, start in enumerate(starts):
            rows.append({
                **r._asdict(),
                'burst': i,
                'onset': r.t0 + i * s_period / fs,
                'epoch_start': s0 + start,
            })
    return pd.DataFrame(rows)


def analyze_trains(run, stimuli, templates, conditions, n_conditions, title,
                   row, info):
    fs, md = run.fs, run.md
    epoch_n = int(round(BURST_SPAN * fs))
    a = analyze_stimuli(run.eeg, fs, stimuli, templates,
                        conditions + ['burst'], epoch_n, run.route_gain_db,
                        run.latency, run.engine)
    offset = (stimuli.epoch_start - np.round(stimuli.onset * fs)) \
        .groupby(stimuli.burst).agg(lambda x: x.abs().max()).astype(int)
    a.checks.add('burst epochs start on bursts', (offset == 0).all(),
                 'offset by burst: ' + ', '.join(f'{o:+d}' for o in offset),
                 '0 samples for every burst',
                 'Where psi starts each per-burst epoch (`split_bursts`) re '
                 'where the burst is (i * round(fs / rate)).')
    s_period = int(round(fs / RATE))
    burst_period = s_period / fs
    train_period = N_BURSTS * burst_period + PAUSE
    check_intervals(a.checks, md.t0.values, fs, train_period,
                    'sequence rate')
    check_sequence(a.checks, md, conditions, n_conditions, AVERAGES)
    info = {**info, 'Bursts per train': N_BURSTS,
            'Burst period': f'{s_period} samples ({fs / s_period:.3f}/s; '
                            f'{RATE}/s requested)',
            'Train period': f'{train_period * 1e3:.4f} ms'}
    # Between bursts in a train, and from the last burst of one train to the
    # first of the next.
    expected_ioi = [burst_period,
                    train_period - (N_BURSTS - 1) * burst_period]
    write_abr_report(run, title, stimuli, templates, a, row, ['burst'],
                     expected_ioi, info)
    return a.checks


@pytest.fixture(scope='module', params=['sequential', 'interleaved_plateau'])
def adaptation_tone(request, routing, results_dir):
    ordering = request.param
    settings = {
        'parameters': {
            'averages': AVERAGES,
            'n_bursts': N_BURSTS,
            'rate': RATE,
            'sequence_pause': PAUSE,
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
    run = run_eeg_paradigm('abr_adaptation', f'abr_adaptation {ordering}',
                           settings, routing, results_dir)
    stimuli = burst_table(run.md, run.fs)
    templates = [
        tone_burst(run.fs, r.frequency, r.level, r.burst_duration,
                   r.burst_rise_time, r.burst_envelope, r.polarity, r.phase,
                   cal_db=OUTPUT_CAL_DB)
        for r in stimuli.itertuples()
    ]
    return analyze_trains(run, stimuli, templates, ['frequency', 'level'],
                          len(FREQUENCIES) * len(LEVELS),
                          f'ABR adaptation (tone), {ordering}',
                          ['frequency', 'level'], {'Ordering': ordering})


@pytest.fixture(scope='module')
def adaptation_click(routing, results_dir):
    settings = {
        'parameters': {
            'averages': AVERAGES,
            'n_bursts': N_BURSTS,
            'rate': RATE,
            'sequence_pause': PAUSE,
            'click_duration': CLICK_DURATION,
            'click_polarity': 'alternating',
            'reject_threshold': float('inf'),
        },
        'selectors': {'default': {'target_click_level': LEVELS}},
    }
    run = run_eeg_paradigm('abr_adaptation_click', 'abr_adaptation_click',
                           settings, routing, results_dir)
    stimuli = burst_table(run.md, run.fs)
    templates = [click(run.fs, r.level, r.click_duration, r.polarity,
                       cal_db=OUTPUT_CAL_DB) for r in stimuli.itertuples()]
    return analyze_trains(run, stimuli, templates, ['level'], len(LEVELS),
                          'ABR adaptation (click)', ['level'], {})


class AdaptationTests(ABRTests):

    def test_timing(self, checks):
        checks.assert_passed('onset jitter', 'sequence rate')

    def test_alignment(self, checks):
        checks.assert_passed('burst epochs start on bursts', 'trials aligned',
                             'trials identical')


class TestAdaptationTone(AdaptationTests):
    @pytest.fixture
    def checks(self, adaptation_tone):
        return adaptation_tone


class TestAdaptationClick(AdaptationTests):
    @pytest.fixture
    def checks(self, adaptation_click):
        return adaptation_click
