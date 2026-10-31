'''
`moc_pipeline.MOCAnalyzer` on a synthetic recording: continuous primaries,
plus a DP that drops by a known amount while the elicitor plays.
'''
from collections import deque

import numpy as np
import pytest

from psiaudio.calibration import FlatCalibration
from psiaudio.pipeline import PipelineData

from cfts.paradigms.moc_pipeline import (MOCAnalyzer, moc_frequencies,
                                         tone_phasors)


FS = 48000
RW = 0.1
F1, F2 = 3330, 4000
DP = 2 * F1 - F2
BASELINE, ELICITOR, RECOVERY = 1.0, 1.0, 1.0
TRIAL = BASELINE + ELICITOR + RECOVERY
#: Primaries and first trial start here (s re acquisition start).
START = 0.5
DP_AMPLITUDE = 1e-3
SUPPRESSION = 3


def db(x):
    return 20 * np.log10(x / np.sqrt(2))


def recording(n_trials, noise=0, seed=0):
    n = int((START + TRIAL * n_trials) * FS)
    t = np.arange(n) / FS
    x = 0.1 * np.cos(2 * np.pi * F1 * t) + 0.1 * np.cos(2 * np.pi * F2 * t)
    trial_time = (t - START) % TRIAL
    suppressed = (trial_time >= BASELINE) & (trial_time < BASELINE + ELICITOR)
    dp_amplitude = np.where(suppressed,
                            DP_AMPLITUDE * 10 ** (-SUPPRESSION / 20),
                            DP_AMPLITUDE)
    # The DP's phase is arbitrary, but fixed re the primaries.
    x += dp_amplitude * np.cos(2 * np.pi * DP * t + 1.0)
    x[t < START] = 0
    x += noise * np.random.default_rng(seed).normal(size=n)
    return x


def run(x, n_trials, requested_n, reject=()):
    trials = deque({'t0': START + i * TRIAL, 'key': 'k', 'metadata': {}}
                   for i in range(n_trials))
    blocks, summaries = [], []
    analyzer = MOCAnalyzer(
        fs=FS, f1=F1, f2=F2, response_window=RW, trial_duration=TRIAL,
        baseline=(0, BASELINE), elicitor=(BASELINE, BASELINE + ELICITOR),
        calibration=FlatCalibration.unity(), requested_n=requested_n,
        trials=trials, block_cb=blocks.extend,
        reject_cb=lambda info, s: 'test' if s['trial'] in reject else None,
        trial_cb=lambda info, s: summaries.append(s),
        discard=0.1)
    # Odd-sized chunks so blocks straddle them.
    for lb in range(0, len(x), 3001):
        chunk = x[lb:lb + 3001][np.newaxis]
        analyzer.send(PipelineData(chunk, fs=FS, s0=lb))
    return analyzer, blocks, summaries


def test_tone_phasors_reference_acquisition_start():
    # Two blocks of a continuous tone taken at different times have the same
    # phasor.
    t = np.arange(FS) / FS
    x = np.cos(2 * np.pi * 1000 * t + 0.3)
    n = int(RW * FS)
    a = tone_phasors(x[:n], 0, FS, [1000])
    b = tone_phasors(x[3 * n + 17:4 * n + 17], 3 * n + 17, FS, [1000])
    assert a == pytest.approx(b)
    assert abs(a[0]) == pytest.approx(1)
    assert np.angle(a[0]) == pytest.approx(0.3)


def test_moc_frequencies():
    f = moc_frequencies(F1, F2, 1 / RW)
    assert list(f) == [F1, F2, DP, DP - 20, DP - 10, DP + 10, DP + 20]


def test_blocks_and_suppression():
    analyzer, blocks, summaries = run(recording(3), 3, requested_n=3)
    n_blocks = int(TRIAL / RW)
    # Discarded blocks (primaries ramping on) aren't reported.
    assert len(blocks) == 3 * n_blocks - 1
    assert [b['block'] for b in blocks[:3]] == [1, 2, 3]

    dp = np.array([b['dpoae_level'] for b in blocks])
    in_elicitor = np.array([b['elicitor'] for b in blocks])
    assert dp[~in_elicitor] == pytest.approx(db(DP_AMPLITUDE), abs=1e-6)
    assert dp[in_elicitor] == \
        pytest.approx(db(DP_AMPLITUDE) - SUPPRESSION, abs=1e-6)
    f1 = np.array([b['f1_level'] for b in blocks])
    assert f1 == pytest.approx(db(0.1), abs=1e-6)

    assert len(summaries) == 3
    for s in summaries:
        assert s['accepted']
        assert s['suppression'] == pytest.approx(SUPPRESSION, abs=1e-6)
        assert s['avg_suppression'] == pytest.approx(SUPPRESSION, abs=1e-6)
        assert s['f1_change'] == pytest.approx(0, abs=1e-6)
    assert [s['n_accepted'] for s in summaries] == [1, 2, 3]
    # Only the first trial's first block is missing from the average.
    assert list(analyzer.count) == [2] + [3] * (n_blocks - 1)


def test_average_reduces_noise_floor():
    _, blocks, summaries = run(recording(8, noise=1e-3), 8, requested_n=8)
    first = [b['avg_dpoae_noise_floor'] for b in blocks if b['trial'] == 0]
    last = [b['avg_dpoae_noise_floor'] for b in blocks if b['trial'] == 7]
    # Eight trials coherently averaged: the floor drops by about 9 dB.
    assert np.mean(first) - np.mean(last) == pytest.approx(9, abs=2)
    assert summaries[-1]['avg_suppression'] == \
        pytest.approx(SUPPRESSION, abs=0.5)


def test_rejected_and_extra_trials():
    # Trial 1 is rejected, so a fourth is needed; the fifth (already queued)
    # is ignored.
    analyzer, blocks, summaries = run(recording(5), 5, requested_n=3,
                                      reject={1})
    assert [s['trial'] for s in summaries] == [0, 1, 2, 3]
    assert [s['accepted'] for s in summaries] == [True, False, True, True]
    assert summaries[1]['reject_reason'] == 'test'
    assert analyzer.n_accepted == 3
    assert not any(b['trial'] == 4 for b in blocks)
