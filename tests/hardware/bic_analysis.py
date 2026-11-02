'''
Checks for the BIC paradigms (`abr_bic`, `abr_bic_click`) through the
loopback.

The left ear is Starship A and the right ear Starship B. Each Starship's
primary output is routed to its own microphone and both to the EEG input, so
the microphones show what each ear got and the EEG gets their sum (as if the
two ears' ABRs added linearly, with no binaural interaction). That makes the
BIC of the loopback known exactly: the monaural sum is shifted as each ear's
burst (or click) was, so the BIC vanishes at every ITD.

Clicks have no frequency; their `frequency` is 0 in the tables here.
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from psiaudio import stim
from psiaudio.calibration import FlatCalibration
from psi.loopback.fit import db_ratio, fit_stimuli, gain_db

from cfts.paradigms.bic_util import bic_traces, CLICK, ITD, LEFT, RIGHT, TONE

from abr_analysis import (ALIGNMENT_LIMIT, analyze_stimuli, LEVEL_TOLERANCE,
                          RESIDUAL_LIMIT)
from rig import INPUT_DBFS


#: Largest acceptable error in the ITD between the ears' microphones (µs).
ITD_TOLERANCE = 0.5

CONDITIONS = ['frequency', 'level', 'stimulus']


@dataclass
class BICAnalysis:
    checks: object
    eeg: object
    left: object
    right: object
    stimuli: pd.DataFrame
    templates: list
    bic: list
    #: Measured ITD of each binaural burst.
    itd: pd.DataFrame
    #: Both microphones for each frequency and ITD (see `itd_waveforms`).
    waveforms: list


def tone_burst(fs, frequency, level, duration, rise_time, polarity, phase,
               start_time=0, cal_db=130):
    '''
    One ear's tone burst (a `Tone` in a `Cos2Envelope`), scaled for an
    output calibrated at `cal_db` dB SPL per volt RMS. Silent at -inf dB.
    '''
    calibration = FlatCalibration.from_db(cal_db)
    carrier = stim.ToneFactory(fs, frequency, level, phase=phase,
                               polarity=polarity, calibration=calibration)
    factory = stim.Cos2EnvelopeFactory(fs, duration, rise_time, carrier,
                                       start_time=start_time)
    return factory.get_samples_remaining()


def click(fs, level, duration, polarity, window, delay, cal_db=130):
    '''
    One ear's click (a `Click` played once, `delay` into a `Repeat` window),
    scaled as `tone_burst` is.
    '''
    calibration = FlatCalibration.from_db(cal_db)
    factory = stim.ClickFactory(fs, duration, level, polarity, calibration)
    return stim.RepeatFactory(fs, 1, 0, 1 / window, delay, factory) \
        .get_samples_remaining()


def stimulus_of(md):
    return CLICK if f'{LEFT}_click_level' in md else TONE


def ear_templates(md, fs, ear, cal_db):
    if stimulus_of(md) is CLICK:
        return [
            click(fs, r[f'{ear}_click_level'], r[f'{ear}_click_duration'],
                  r[f'{ear}_click_polarity'], 1 / r[f'{ear}_click_train_rate'],
                  r[f'{ear}_click_train_delay'], cal_db)
            for _, r in md.iterrows()
        ]
    return [
        tone_burst(fs, r[f'{ear}_tone_frequency'], r[f'{ear}_tone_level'],
                   r[f'{ear}_tone_burst_duration'],
                   r[f'{ear}_tone_burst_rise_time'],
                   r[f'{ear}_tone_polarity'], r[f'{ear}_tone_phase'],
                   r[f'{ear}_tone_burst_start_time'], cal_db)
        for _, r in md.iterrows()
    ]


def with_frequency(md):
    '''
    `md` with a `bic_frequency` (0 for clicks) and the ears' polarity and
    level under names that don't depend on the stimulus.
    '''
    st = stimulus_of(md)
    md = md.copy()
    if 'bic_frequency' not in md:
        md['bic_frequency'] = 0
    for ear in (LEFT, RIGHT):
        for name in ('polarity', 'level', 'delay'):
            md[f'{ear}_{name}'] = md[st.ear_name(ear, name)]
    return md


def add_padded(a, b):
    n = max(len(a), len(b))
    return np.pad(a, (0, n - len(a))) + np.pad(b, (0, n - len(b)))


def stimulus_label(condition, value):
    if condition != 'binaural':
        return condition
    return f'binaural {value:g} µs'


def stimulus_table(md, fs):
    '''
    One row per trial: its condition, polarity, where psi logged it and
    where the epoch psi averages for it starts.
    '''
    values = md[ITD]
    stimulus = [stimulus_label(c, v)
                for c, v in zip(md.bic_condition, values, strict=True)]
    return pd.DataFrame({
        'frequency': md.bic_frequency,
        'level': md.bic_level,
        'condition': md.bic_condition,
        'itd': values,
        'stimulus': stimulus,
        'polarity': md[f'{LEFT}_polarity'].astype(int),
        'onset': md.t0,
        'epoch_start': np.round(md.t0 * fs).astype(int),
    })


def check_ear(checks, name, x, fs, onsets, templates, levels,
              route_gain_db):
    '''
    Fit one ear's microphone with that ear's bursts (silent ones excluded),
    and check that nothing else is there (in particular, nothing on the
    trials where this ear is silent).
    '''
    fit = fit_stimuli(x, fs, onsets, templates)
    playing = np.isfinite(levels)
    e = fit.epochs[playing]
    error = gain_db(e.gain.values) - (route_gain_db + INPUT_DBFS)
    worst = np.nanmax(np.abs(error))
    checks.add(f'{name} level', worst <= LEVEL_TOLERANCE,
               f'worst error {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
               f'{playing.sum()} bursts in the {name} ear.')
    residual = np.nanmax(e.residual_db)
    checks.add(f'{name} waveform', residual <= RESIDUAL_LIMIT,
               f'worst residual {residual:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               'Includes the carrier phase and the burst\'s delay, so a '
               'wrong ITD fails here.')
    checks.add(f'{name} ear only', fit.unexplained_db <= RESIDUAL_LIMIT,
               f'{fit.unexplained_db:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               f'Energy on the {name} microphone the {name} ear\'s bursts '
               'don\'t explain, e.g., a burst on a trial where this ear '
               'should be silent.')
    return fit


def carrier_phase(left, right, fs, start, n, frequency):
    '''
    Phase (degrees) of the right microphone re the left at the tone
    frequency, over `n` samples from `start`.
    '''
    t = np.arange(n) / fs
    basis = np.exp(-2j * np.pi * frequency * t)
    zl = left[start:start + n] @ basis
    zr = right[start:start + n] @ basis
    return np.degrees(np.angle(zr * np.conj(zl)))


def measure_itd(left, right, fs, start, n, frequency):
    '''
    How far (µs) the right microphone leads the left over `n` samples from
    `start`: the lag of the cross-correlation peak, which the envelopes make
    unambiguous to within a carrier cycle, refined with the carrier phase (a
    click, frequency 0, has none: to the nearest sample).
    '''
    l = left[start:start + n]
    r = right[start:start + n]
    c = signal.correlate(l, r)
    lags = signal.correlation_lags(len(l), len(r))
    # left[i] ≈ right[i - lag]: the left lags by `lag` samples.
    coarse = lags[np.argmax(c)] / fs
    if not frequency:
        return coarse * 1e6
    period = 1 / frequency
    fine = carrier_phase(left, right, fs, start, n, frequency) / 360 * period
    cycles = np.round((coarse - fine) / period)
    return (fine + cycles * period) * 1e6


def measure_itds(left, right, fs, md, latency):
    '''
    The ITD of each binaural burst, as the microphones show it, over the
    whole of both bursts.
    '''
    binaural = md[md.bic_condition == 'binaural']
    rows = []
    for _, r in binaural.iterrows():
        if stimulus_of(md) is CLICK:
            # The click is delayed within its window.
            n = int(round(fs / r[f'{LEFT}_click_train_rate']))
        else:
            delay = max(r[f'{LEFT}_delay'], r[f'{RIGHT}_delay'])
            n = int(round((r.bic_burst_duration + delay) * fs))
        start = int(round(r.t0 * fs)) + latency
        rows.append({
            'frequency': r.bic_frequency,
            'level': r.bic_level,
            'requested': r[ITD],
            'measured': measure_itd(left, right, fs, start, n,
                                    r.bic_frequency),
        })
    return pd.DataFrame(rows)


def check_itd(checks, table):
    error = table.measured - table.requested
    worst = np.abs(error).max()
    checks.add('ITD', worst <= ITD_TOLERANCE,
               f'worst error {worst:.3f} µs', f'± {ITD_TOLERANCE} µs',
               'ITD of the right microphone re the left, over each binaural '
               'burst.')


def check_itd_shift(checks, table):
    '''
    The microphones' ITD moves by the requested amount (re the smallest
    ITD), at each frequency and level, and actually moves.
    '''
    errors, moved = [], []
    for _, g in table.groupby(['frequency', 'level']):
        mean = g.groupby('requested').measured.mean()
        base = mean.index.min()
        shift = mean - mean[base]
        wanted = mean.index - base
        errors.extend(np.abs(shift.values - wanted))
        moved.extend(np.abs(shift.values[wanted != 0]))
    worst = max(errors)
    least = min(moved) if moved else 0
    checks.add('ITD shifts', worst <= ITD_TOLERANCE and least > 10,
               f'worst error {worst:.3f} µs; smallest shift {least:.1f} µs',
               f'± {ITD_TOLERANCE} µs of the requested shift',
               'Mean ITD of the right microphone re the left at each ITD, '
               'minus at the smallest ITD, per frequency and level.')


def itd_waveforms(left, right, fs, md, latency):
    '''
    Both microphones, averaged over the binaural bursts of each frequency
    and ITD (positive polarity, highest level) and scaled by the left
    microphone's peak, from the start of the trial to 1.5 ms after the
    later burst starts, so the burst onsets show.
    '''
    binaural = md[(md.bic_condition == 'binaural')
                  & (md[f'{LEFT}_polarity'] == 1)]
    binaural = binaural[binaural.bic_level == binaural.bic_level.max()]
    result = []
    for (frequency, value), g in binaural.groupby(['bic_frequency', ITD]):
        delay = abs(value) * 1e-6
        n = int(round((delay + 1.5e-3) * fs))
        x = np.arange(n) / fs
        starts = np.round(g.t0.values * fs).astype(int) + latency
        l = np.mean([left[i:i + n] for i in starts], axis=0)
        r = np.mean([right[i:i + n] for i in starts], axis=0)
        scale = np.abs(l).max()
        result.append({'frequency': frequency, 'value': value, 'x': x,
                       'left': l / scale, 'right': r / scale})
    return result


def psi_bic(epochs, expected, stimuli, fs):
    '''
    The BIC the plot shows, computed from the epochs psi averages, for each
    frequency, level and polarity (alternating polarity cancels the
    stimulus, so each polarity is checked on its own), and what it should be.
    '''
    index = epochs.index.to_frame(index=False)
    keys = stimuli.set_index('stimulus')[['condition', 'itd']] \
        .drop_duplicates()
    results = []
    for (frequency, level, polarity), g in \
            index.groupby(['frequency', 'level', 'polarity']):
        measured, wanted = {}, {}
        for stimulus, s in g.groupby('stimulus'):
            row = keys.loc[stimulus]
            key = ('binaural', row.itd) if row.condition == 'binaural' \
                else (row.condition,)
            measured[key] = epochs.values[s.index].mean(axis=0)
            wanted[key] = expected[s.index].mean(axis=0)
        sums, bic = bic_traces(measured, fs)
        e_sums, e_bic = bic_traces(wanted, fs)
        error = {v: db_ratio(bic[v] - e_bic[v], e_sums[v]) for v in bic}
        results.append({
            'frequency': frequency, 'level': level, 'polarity': polarity,
            'time': epochs.columns.values, 'binaural': {
                k[1]: v for k, v in measured.items() if k[0] == 'binaural'},
            'sums': sums, 'bic': bic, 'expected_bic': e_bic,
            'error_db': error,
        })
    return results


def analyze_bic(eeg, eeg_fs, left_mic, right_mic, mic_fs, md, gains,
                latency, engine, epoch_n, cal_db):
    '''
    Parameters
    ----------
    gains : dict
        Route gain (dB) of each ear's output to the EEG (`left_eeg`,
        `right_eeg`) and to its own microphone (`left_mic`, `right_mic`).
    '''
    md = with_frequency(md)
    stimuli = stimulus_table(md, eeg_fs)

    # EEG: the sum of both ears' bursts, each through its route.
    g = {k: 10 ** (v / 20) for k, v in gains.items()}
    left = ear_templates(md, eeg_fs, LEFT, cal_db)
    right = ear_templates(md, eeg_fs, RIGHT, cal_db)
    templates = [add_padded(g['left_eeg'] * l, g['right_eeg'] * r)
                 for l, r in zip(left, right, strict=True)]
    a = analyze_stimuli(eeg, eeg_fs, stimuli, templates, CONDITIONS, epoch_n,
                        0, latency, engine)
    checks = a.checks

    # Each ear on its own microphone.
    onsets = md.t0.values
    fits = {}
    for ear, name, x in ((LEFT, 'left', left_mic), (RIGHT, 'right', right_mic)):
        fits[name] = check_ear(checks, name, x, mic_fs, onsets,
                               ear_templates(md, mic_fs, ear, cal_db),
                               md[f'{ear}_level'].values,
                               gains[f'{name}_mic'])
    binaural = (md.bic_condition == 'binaural').values
    lag_diff = np.abs(fits['left'].epochs.lag.values[binaural]
                      - fits['right'].epochs.lag.values[binaural])
    same_latency = fits['left'].latency == fits['right'].latency
    checks.add('ears in step', same_latency and lag_diff.max() == 0,
               f'latency {fits["left"].latency} and {fits["right"].latency} '
               f'samples; worst binaural burst {lag_diff.max():.0f} samples '
               'apart', 'same latency, 0 samples apart',
               'Each binaural burst in the left ear re the right, after '
               'allowing for any requested delay.')

    table = measure_itds(left_mic, right_mic, mic_fs, md,
                         fits['left'].latency)
    check_itd(checks, table)
    check_itd_shift(checks, table)
    waveforms = itd_waveforms(left_mic, right_mic, mic_fs, md,
                              fits['left'].latency)

    # The BIC psi plots, from the epochs it averages.
    expected = np.zeros_like(a.epochs.values)
    gain = 10 ** (a.expected_gain_db / 20)
    for i, template in enumerate(templates):
        t = template[:epoch_n] * gain
        expected[i, :len(t)] = t
    bic = psi_bic(a.epochs, expected, stimuli, eeg_fs)
    worst = max(max(r['error_db'].values()) for r in bic)
    checks.add('psi BIC', worst <= ALIGNMENT_LIMIT,
               f'worst {worst:.1f} dB re the monaural sum',
               f'≤ {ALIGNMENT_LIMIT} dB',
               'Binaural minus the sum of the monaural averages (shifted as '
               'each ear\'s burst was), per polarity, against what the '
               'loopback should give.')
    return BICAnalysis(checks, a, fits['left'], fits['right'], stimuli,
                       templates, bic, table, waveforms)
