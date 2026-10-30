'''
The stimuli cfts plays, regenerated with psiaudio so they can be fit to what
the loopback recorded (with `psi.loopback.fit`), and checks on the epochs psi
averages.
'''
import numpy as np
import pandas as pd

from psiaudio.calibration import FlatCalibration
from psiaudio import stim
from psi.loopback.fit import db_ratio


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
