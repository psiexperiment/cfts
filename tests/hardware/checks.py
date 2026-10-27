'''
Compare what a paradigm recorded through the loopback with what it says it
played.

Each stimulus the paradigm logs (onset `t0` plus the token's settings) is
regenerated with psiaudio and fit to the recording. A digital loopback is
exact, so after allowing for the sound card's round-trip delay and the
route's gain, every stimulus should match its template to well below the
noise floor, start at the same delay after its logged onset, and nothing
should be in the recording that wasn't logged.
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from psiaudio.calibration import FlatCalibration
from psiaudio import stim


@dataclass
class Fit:
    #: Delay from logged onset to the stimulus in the recording, in samples,
    #: common to every stimulus.
    latency: int
    #: One row per stimulus: `lag` (samples, relative to `latency`),
    #: `gain` (recorded / template; negative if the polarity is inverted),
    #: `residual_db` (what the template leaves unexplained, re the stimulus)
    #: and `onset` (seconds, in the recording).
    epochs: pd.DataFrame
    #: The recording, and the sum of every fitted template.
    recorded: np.ndarray
    reconstructed: np.ndarray
    fs: float

    @property
    def unexplained_db(self):
        '''
        Energy in the recording that the fitted stimuli don't account for
        (e.g., stimuli that were played but not logged), re the recording.
        '''
        error = self.recorded - self.reconstructed
        return 10 * np.log10(np.sum(error ** 2) / np.sum(self.recorded ** 2))


def click(fs, level, duration, polarity=1, cal_db=130):
    '''
    The click psi plays for these settings (see `psiaudio.stim.ClickFactory`).
    '''
    calibration = FlatCalibration.from_db(cal_db)
    n = int(fs * duration)
    return polarity * calibration.get_sf(0, level) * np.ones(n)


def sam_tone(fs, fc, fm, level, duration, rise_time, depth=1, polarity=1,
             equalize=True, cal_db=130):
    '''
    The SAM tone psi plays in the SAM EFR paradigms (a `SAMTone` in a
    `Cos2Envelope`).
    '''
    calibration = FlatCalibration.from_db(cal_db)
    carrier = stim.SAMToneFactory(fs, fc, fm, level, depth=depth,
                                  polarity=polarity, equalize=equalize,
                                  calibration=calibration)
    factory = stim.Cos2EnvelopeFactory(fs, duration, rise_time, carrier)
    return factory.get_samples_remaining()


def ram_tone(fs, fc, fm, level, duration, duty_cycle, alpha, depth=1,
             phase=0, polarity=1, cal_db=130):
    '''
    The RAM tone psi plays in the RAM EFR paradigm (a `Tone` in a
    `SquareWaveEnvelope` in a `Gate`).
    '''
    calibration = FlatCalibration.from_db(cal_db)
    carrier = stim.ToneFactory(fs, fc, level, phase=phase, polarity=polarity,
                               calibration=calibration)
    modulated = stim.SquareWaveEnvelopeFactory(fs, depth, fm, duty_cycle,
                                               calibration, carrier,
                                               alpha=alpha)
    factory = stim.GateFactory(fs, 0, duration, modulated)
    return factory.get_samples_remaining()


def tone_burst(fs, frequency, level, duration, rise_time, envelope='cosine-squared',
               polarity=1, phase=0, cal_db=130):
    '''
    The tone burst psi plays for these settings, scaled for an output
    calibrated at `cal_db` dB SPL per volt RMS.
    '''
    calibration = FlatCalibration.from_db(cal_db)
    n = int(round(duration * fs))
    carrier = stim.tone(fs, frequency, level, phase=phase, polarity=polarity,
                        calibration=calibration, samples=n)
    window = stim.envelope(envelope, fs, duration, rise_time, samples=n)
    return carrier * window


def find_latency(x, fs, onsets, templates, max_latency=0.5):
    '''
    Delay, in samples, from the logged onsets to where the stimuli appear in
    `x`.

    Found by correlating the recording against all the templates placed at
    their logged onsets, which a single periodic stimulus could not pin down
    but the onset of the whole train does. The search starts at the first
    logged onset, so a delay longer than the silence before the train is
    still found.
    '''
    max_lag = int(max_latency * fs)
    expected = np.zeros(len(x) + max_lag)
    for t0, template in zip(onsets, templates):
        i = int(round(t0 * fs))
        expected[i:i + len(template)] += template
    expected = expected[:len(x)]
    c = signal.correlate(x, expected, 'full', 'fft')
    zero = len(expected) - 1
    c = c[zero:zero + max_lag]
    return int(np.argmax(c))


def fit_stimuli(x, fs, onsets, templates, latency=None, search=0.002):
    '''
    Fit each template to the recording near its logged onset.

    Parameters
    ----------
    x : array
        The recording.
    onsets : array
        Logged onset of each stimulus, in seconds.
    templates : list of arrays
        What each stimulus should be.
    latency : int or None
        Delay in samples; found with `find_latency` if None.
    search : float
        How far either side of the expected position to look, in seconds.
    '''
    if latency is None:
        latency = find_latency(x, fs, onsets, templates)
    n_search = int(search * fs)
    reconstructed = np.zeros_like(x)
    rows = []
    for t0, template in zip(onsets, templates):
        i = int(round(t0 * fs)) + latency
        lb = i - n_search
        ub = i + len(template) + n_search
        if lb < 0 or ub > len(x):
            rows.append({'lag': np.nan, 'gain': np.nan, 'residual_db': np.nan,
                         'onset': np.nan})
            continue
        segment = x[lb:ub]
        c = signal.correlate(segment, template, 'valid')
        k = int(np.argmax(np.abs(c)))
        s = x[lb + k:lb + k + len(template)]
        gain = np.dot(s, template) / np.dot(template, template)
        error = s - gain * template
        with np.errstate(divide='ignore'):
            residual_db = 10 * np.log10(np.sum(error ** 2) / np.sum(s ** 2))
        reconstructed[lb + k:lb + k + len(template)] += gain * template
        rows.append({
            'lag': k - n_search,
            'gain': gain,
            'residual_db': residual_db,
            'onset': (lb + k) / fs,
        })
    return Fit(latency, pd.DataFrame(rows), x, reconstructed, fs)


def gain_db(gain):
    return 20 * np.log10(np.abs(gain))


def db_ratio(a, b):
    '''
    RMS of `a` re RMS of `b`, in dB (along the last axis).
    '''
    rms_a = np.sqrt(np.mean(np.square(a), axis=-1))
    rms_b = np.sqrt(np.mean(np.square(b), axis=-1))
    with np.errstate(divide='ignore'):
        return 20 * np.log10(rms_a / rms_b)


def epoch_alignment(epochs, expected):
    '''
    How well each epoch psi averages matches its stimulus, and how well it
    matches the other trials of the same condition and polarity.

    `expected` holds, for each epoch, the stimulus as it should appear in that
    epoch (at lag 0, scaled by the route's gain). Returns one row per epoch
    with `template_db` (what the expected stimulus leaves unexplained) and
    `trial_db` (the difference from the condition's first trial). Both are re
    the stimulus, so an empty epoch has a `template_db` of 0 dB, not -inf.
    '''
    x = epochs.values
    groups = list(epochs.index.names)
    groups.remove('t0')
    first = epochs.groupby(groups).transform('first').values
    return pd.DataFrame({
        'template_db': db_ratio(x - expected, expected),
        'trial_db': db_ratio(x - first, expected),
    }, index=epochs.index)


def polarity_cancellation(epochs):
    '''
    For each condition, the average of its + and - averages re the +
    average, in dB. Equal counts of each polarity are used, as an averaged
    ABR would.
    '''
    groups = [n for n in epochs.index.names if n not in ('t0', 'polarity')]
    result = {}
    for key, g in epochs.groupby(groups):
        pos = g.xs(1, level='polarity').values
        neg = g.xs(-1, level='polarity').values
        k = min(len(pos), len(neg))
        pos, neg = pos[:k].mean(axis=0), neg[:k].mean(axis=0)
        result[key] = db_ratio((pos + neg) / 2, pos)
    return pd.Series(result, name='cancellation_db') \
        .rename_axis(groups)
