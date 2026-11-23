'''
Forward-masking ABR through the loopback.

Each trial is a masker tone burst, a gap and a probe tone burst. Every pair
has to be in the EEG as specified, at the requested rate whatever the gap,
and the probe-aligned epoch psi cuts out (`align_to_probe`) has to start on
the probe.
'''
import numpy as np
import pytest

from psiaudio.calibration import FlatCalibration
from psiaudio.pipeline import concat, PipelineData
from psiaudio.stim import EnvelopeFactory, ToneFactory

from abr_analysis import ALIGNMENT_LIMIT, analyze_stimuli, check_intervals, \
    check_sequence
from runner import OUTPUT_CAL_DB
from test_abr import ABRTests, run_eeg_paradigm, write_abr_report


MASKER_FREQUENCY = 4000
PROBE_FREQUENCY = 8000
MASKER_DURATION = 20e-3
PROBE_DURATION = 5e-3
RISE_TIME = 0.5e-3
#: Gaps (ms, as the selector takes them). Not whole numbers of samples at
#: every rate, so the ITI has to follow the samples played.
GAPS_MS = [0.51, 2, 8.33]
MASKER_LEVELS = [60, 80]
PROBE_LEVELS = [60]
AVERAGES = 4
RATE = 20

#: Probe window (`ABRForwardMaskingManifest.probe_span`).
PROBE_SPAN = 8.5e-3

MASKER = 'target_masker'
PROBE = 'target_probe'
GAP = 'target_forward_masking_gap'
CONDITIONS = ['masker_level', 'probe_level', 'gap']


def burst(fs, prefix, r, calibration):
    tone = ToneFactory(fs, getattr(r, f'{prefix}_tone_frequency'),
                       getattr(r, f'{prefix}_tone_level'),
                       polarity=getattr(r, f'{prefix}_tone_polarity'),
                       calibration=calibration)
    return EnvelopeFactory(getattr(r, f'{prefix}_envelope'), fs,
                           getattr(r, f'{prefix}_duration'),
                           getattr(r, f'{prefix}_rise_time'), tone)


def pair_template(fs, r, cal_db=OUTPUT_CAL_DB):
    '''
    The masker, gap and probe psi plays for trial `r`, and the probe alone.
    '''
    calibration = FlatCalibration.from_db(cal_db)
    masker = burst(fs, MASKER, r, calibration).get_samples_remaining()
    probe = burst(fs, PROBE, r, calibration).get_samples_remaining()
    gap = np.zeros(int(round(getattr(r, GAP) * fs)))
    return np.concatenate((masker, gap, probe)), probe


def probe_starts(md, fs, epoch_n):
    '''
    Sample, within each trial's epoch, at which psi starts the probe-aligned
    epoch, found by running the paradigm's own `align_to_probe` on epochs
    whose samples are their own indices.
    '''
    from cfts.paradigms.forward_masking_util import align_to_probe
    epochs = [PipelineData(np.arange(epoch_n, dtype=float)[np.newaxis], fs=fs,
                           s0=0, metadata=r._asdict())
              for r in md.itertuples(index=False)]
    result = []
    align_to_probe(PROBE_SPAN, result.append).send(
        concat(epochs, axis='epoch'))
    return np.asarray(result[0])[:, 0, 0].astype(int)


@pytest.fixture(scope='module', params=['sequential', 'interleaved_random'])
def forward_masking(request, routing, results_dir):
    ordering = request.param
    settings = {
        'parameters': {
            'averages': AVERAGES,
            'rate': RATE,
            'ordering': ordering,
            'alternate_polarity': True,
            'include_probe_alone': True,
            'reject_threshold': float('inf'),
            f'{MASKER}_duration': MASKER_DURATION,
            f'{MASKER}_rise_time': RISE_TIME,
            f'{MASKER}_tone_frequency': MASKER_FREQUENCY,
            f'{PROBE}_duration': PROBE_DURATION,
            f'{PROBE}_rise_time': RISE_TIME,
            f'{PROBE}_tone_frequency': PROBE_FREQUENCY,
        },
        'selectors': {'default': {
            GAP: GAPS_MS,
            f'{MASKER}_tone_level': MASKER_LEVELS,
            f'{PROBE}_tone_level': PROBE_LEVELS,
        }},
    }
    run = run_eeg_paradigm('abr_forward_masking',
                           f'abr_forward_masking {ordering}', settings,
                           routing, results_dir)
    fs = run.fs
    # Keep the token's names for the templates; add short ones for the
    # checks and plots.
    md = run.md.assign(
        masker_level=run.md[f'{MASKER}_tone_level'],
        probe_level=run.md[f'{PROBE}_tone_level'],
        gap=run.md[GAP],
        level=run.md[f'{PROBE}_tone_level'],
        polarity=run.md[f'{PROBE}_tone_polarity'],
        onset=run.md.t0,
        epoch_start=np.round(run.md.t0 * fs).astype(int),
    )
    pairs = [pair_template(fs, r) for r in md.itertuples(index=False)]
    templates = [p for p, _ in pairs]

    a = analyze_stimuli(run.eeg, fs, md, templates, CONDITIONS, run.epoch_n,
                        run.route_gain_db, run.latency, run.engine)
    check_intervals(a.checks, md.t0.values, fs, 1 / RATE)
    n_conditions = len(GAPS_MS) * len(MASKER_LEVELS) * len(PROBE_LEVELS) \
        + len(PROBE_LEVELS)
    check_sequence(a.checks, md, CONDITIONS, n_conditions, AVERAGES)

    # The probe-aligned epoch has to hold the probe at lag 0.
    starts = md.epoch_start.values + probe_starts(run.md, fs, run.epoch_n)
    span_n = int(round(PROBE_SPAN * fs))
    gain = 10 ** (a.expected_gain_db / 20)
    errors = []
    for start, (_, probe) in zip(starts, pairs):
        expected = np.zeros(span_n)
        expected[:len(probe)] = probe[:span_n] * gain
        actual = run.eeg[start:start + span_n]
        errors.append(10 * np.log10(np.sum((actual - expected) ** 2)
                                    / np.sum(expected ** 2)))
    worst = max(errors)
    a.checks.add('probe epochs start on probe', worst <= ALIGNMENT_LIMIT,
                 f'worst epoch {worst:.1f} dB from its probe',
                 f'≤ {ALIGNMENT_LIMIT} dB',
                 'Each probe-aligned epoch (`align_to_probe`) against the '
                 'probe at lag 0.')

    write_abr_report(run, f'ABR forward masking, {ordering}', md, templates,
                     a, ['masker_level'], ['gap'], CONDITIONS, 1 / RATE,
                     {'Ordering': ordering, 'Rate': f'{RATE}/s'})
    return a.checks


class TestForwardMasking(ABRTests):

    @pytest.fixture
    def checks(self, forward_masking):
        return forward_masking

    def test_probe_alignment(self, checks):
        checks.assert_passed('probe epochs start on probe')
