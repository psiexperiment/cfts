'''
Checks shared by the ABR paradigms (tone, click and adaptation trains).

The paradigm-specific part is turning the saved metadata into a table of
stimuli: one row per tone burst or click, with where psi played it (`onset`,
seconds), where the epoch psi averages for it starts (`epoch_start`,
samples), its condition, and the waveform it should be. Everything else
(timing, level, waveform, alignment and polarity cancellation) is the same.
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd

from checks import (db_ratio, epoch_alignment, fit_stimuli, gain_db,
                    polarity_cancellation)
from report import Checks
from rig import INPUT_DBFS


#: Largest acceptable error in level, in dB.
LEVEL_TOLERANCE = 0.1

#: How far below the stimulus whatever the template doesn't explain must be.
RESIDUAL_LIMIT = -40

#: How far below the stimulus each epoch psi averages may differ from the
#: expected stimulus, or from the other trials of its condition. A digital
#: loopback is exact, so this only allows for the converters' resolution;
#: being off by one sample is about -18 dB at 2 kHz and -6 dB at 8 kHz.
ALIGNMENT_LIMIT = -60

#: How far below the + average the mean of the + and - averages must be.
CANCELLATION_LIMIT = -60


@dataclass
class Analysis:
    checks: Checks
    fit: object
    epochs: pd.DataFrame
    alignment: pd.DataFrame
    cancellation: pd.Series
    expected_gain_db: float


def extract_epochs(x, stimuli, n, fs, conditions):
    '''
    The epoch psi averages for each stimulus, indexed by its condition and
    `t0` (the stimulus onset).
    '''
    values = np.stack([x[s:s + n] for s in stimuli.epoch_start])
    index = pd.MultiIndex.from_frame(stimuli[conditions + ['onset']]
                                     .rename(columns={'onset': 't0'}))
    return pd.DataFrame(values, index=index,
                        columns=pd.Index(np.arange(n) / fs, name='time'))


def analyze_stimuli(x, fs, stimuli, templates, conditions, epoch_n,
                    route_gain_db, latency, engine):
    '''
    Parameters
    ----------
    x : array
        The EEG recording.
    stimuli : DataFrame
        One row per stimulus: `onset`, `epoch_start`, `level`, `polarity`
        and the `conditions` columns.
    templates : list of arrays
        Each stimulus as played.
    conditions : list of str
        Columns that, with polarity, make up a condition psi averages
        separately.
    epoch_n : int
        Samples in each epoch psi averages.
    latency : int
        Latency measured just before the run.
    engine : dict
        The sound card engine's settings as the run recorded them.
    '''
    checks = Checks()
    fit = fit_stimuli(x, fs, stimuli.onset.values, templates)
    e = fit.epochs

    checks.add('engine latency', engine['latency_samples'] == latency,
               f'latency_samples {engine["latency_samples"]} at '
               f'{engine["fs"]} Hz, blocksize {engine["blocksize"]}',
               f'{latency} (measured just before the run)')
    checks.add('latency compensated', abs(fit.latency) <= 1,
               f'{fit.latency} samples ({fit.latency / fs * 1e3:.2f} ms)',
               '0 ± 1 samples',
               'Delay from logged onset to stimulus in the EEG. Nonzero '
               'means psi averages the wrong window; set the engine\'s '
               'latency_samples.')
    jitter = np.nanmax(np.abs(e.lag))
    checks.add('onset jitter', jitter <= 1, f'{jitter:.0f} samples',
               '≤ 1 sample', 'Each stimulus re where its logged onset puts it.')

    expected_gain_db = route_gain_db + INPUT_DBFS
    level_error = gain_db(e.gain.values) - expected_gain_db
    worst = np.nanmax(np.abs(level_error))
    checks.add('level', worst <= LEVEL_TOLERANCE, f'worst error {worst:.3f} dB',
               f'± {LEVEL_TOLERANCE} dB')
    inverted = int((e.gain < 0).sum())
    checks.add('polarity', inverted == 0,
               f'{inverted} of {len(e)} inverted', '0 inverted')
    residual = np.nanmax(e.residual_db)
    checks.add('waveform', residual <= RESIDUAL_LIMIT,
               f'worst residual {residual:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               'Energy each stimulus\'s template leaves unexplained.')
    checks.add('nothing unlogged', fit.unexplained_db <= RESIDUAL_LIMIT,
               f'{fit.unexplained_db:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               'Energy in the whole recording the logged stimuli don\'t '
               'account for (e.g., a stimulus played but not logged).')

    epochs = extract_epochs(x, stimuli, epoch_n, fs,
                            conditions + ['polarity'])
    gain = 10 ** (expected_gain_db / 20)
    # Each epoch should start exactly on its stimulus.
    expected = np.zeros_like(epochs.values)
    for i, template in enumerate(templates):
        t = template[:epoch_n] * gain
        expected[i, :len(t)] = t
    alignment = epoch_alignment(epochs, expected)
    worst = alignment.template_db.max()
    checks.add('trials aligned', worst <= ALIGNMENT_LIMIT,
               f'worst epoch {worst:.1f} dB from its stimulus',
               f'≤ {ALIGNMENT_LIMIT} dB',
               'Each epoch psi averages against its stimulus at lag 0. '
               'An empty epoch reads 0 dB.')
    worst = alignment.trial_db.max()
    checks.add('trials identical', worst <= ALIGNMENT_LIMIT,
               f'worst epoch {worst:.1f} dB from the first of its condition',
               f'≤ {ALIGNMENT_LIMIT} dB')
    cancellation = polarity_cancellation(epochs)
    worst = cancellation.max()
    checks.add('polarity cancels', worst <= CANCELLATION_LIMIT,
               f'worst condition {worst:.1f} dB', f'≤ {CANCELLATION_LIMIT} dB',
               'Mean of the + and - averages re the + average.')

    return Analysis(checks, fit, epochs, alignment, cancellation,
                    expected_gain_db)


def check_intervals(checks, onsets, fs, expected, name='repetition rate'):
    '''
    Intervals between logged onsets, each against the nearest of `expected`
    (seconds). Each may be off by up to a sample when the period isn't a
    whole number of samples, but they must not drift: on average, they have
    to be within 0.1 sample.
    '''
    ioi = np.diff(onsets)
    expected = np.atleast_1d(expected)
    i = np.abs(ioi[:, None] - expected).argmin(axis=1)
    error = (ioi - expected[i]) * fs
    worst = np.abs(error).max()
    drift = error.mean()
    label = ', '.join(f'{e * 1e3:.4f}' for e in expected)
    checks.add(name, worst <= 1 and abs(drift) <= 0.1,
               f'worst interval off by {worst:.2f} samples, '
               f'{drift:+.2f} on average',
               f'{label} ms; each ± 1 sample, on average ± 0.1')


def check_sequence(checks, md, conditions, n_conditions, averages):
    '''
    Every condition was presented, at least `averages` times, with + and -
    polarity balanced. `md` has one row per trial.
    '''
    counts = md.groupby(conditions).size() if conditions \
        else pd.Series([len(md)])
    checks.add('conditions', len(counts) == n_conditions,
               f'{len(counts)} conditions', f'{n_conditions}')
    checks.add('averages', counts.min() >= averages,
               f'{counts.min()}–{counts.max()} per condition',
               f'≥ {averages}',
               f'{counts.sum() - averages * len(counts)} extra in total '
               'from trials already queued when a condition completed.')
    by_polarity = md.groupby(conditions + ['polarity']).size() \
        .unstack('polarity')
    imbalance = int((by_polarity[1] - by_polarity[-1]).abs().max())
    checks.add('polarity alternates', imbalance <= 1,
               f'largest +/- imbalance {imbalance}', '≤ 1')
