'''
Checks for the DPOAE paradigms.

Each setting plays f1 on the Starship's primary output and f2 on its
secondary as continuous tones that start, at phase 0 with a cos² ramp, on
the sample psi logs as the setting's start. They stop (unramped, and at a
time psi doesn't log) once psi has analyzed enough of them, and the next
setting starts a little later. Both outputs are summed into the Starship's
microphone, so each segment of the recording should be exactly those two
tones, covering the window psi analyzed, and nothing at 2f1 - f2 (a digital
loopback has no distortion).
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd

from report import Checks
from rig import INPUT_DBFS
from runner import OUTPUT_CAL_DB


#: Largest acceptable error in level, in dB.
LEVEL_TOLERANCE = 0.1

#: Largest acceptable lag of a tone re the logged start, in samples
#: (measured from its phase, so a fraction of a sample is resolved).
LAG_TOLERANCE = 0.5

#: How far below the tones whatever they leave unexplained must be.
RESIDUAL_LIMIT = -60

#: How far below f2 the level at 2f1 - f2 must be.
DISTORTION_LIMIT = -90

#: Discarded at the start of each setting before psi analyzes it, as a
#: multiple of the rise time (`dpoae_base` Discard).
DISCARD_RISE_TIMES = 4


@dataclass
class Analysis:
    checks: Checks
    #: One row per setting with the measurements.
    segments: pd.DataFrame
    #: Fitted tones over the whole recording.
    reconstructed: np.ndarray


def ramp_on(n, rise_n):
    '''
    The onset of psi's cos² envelope over `n` samples.
    '''
    env = np.ones(n)
    k = np.arange(min(rise_n, n))
    env[:len(k)] = np.sin(np.pi * k / (2 * rise_n)) ** 2
    return env


def fit_tones(x, fs, frequencies, env):
    '''
    Least-squares fit of a cosine and sine at each frequency, under `env`.
    Returns the amplitude, phase (radians, re a cosine starting at sample 0)
    and the fitted signal.
    '''
    t = np.arange(len(x)) / fs
    basis = []
    for f in frequencies:
        basis += [env * np.cos(2 * np.pi * f * t),
                  env * np.sin(2 * np.pi * f * t)]
    basis = np.stack(basis, axis=1)
    coefs = np.linalg.lstsq(basis, x, rcond=None)[0]
    a, b = coefs[0::2], coefs[1::2]
    return np.hypot(a, b), np.arctan2(b, a), basis @ coefs


def level_db(amplitude, gain_db):
    '''
    dB SPL of a tone with peak `amplitude` in the recording.
    '''
    return 20 * np.log10(amplitude / np.sqrt(2)) + OUTPUT_CAL_DB \
        - INPUT_DBFS - gain_db


def analyze_dpoae(x, fs, segments, gain_db, latency, engine):
    '''
    Parameters
    ----------
    x : array
        The microphone recording.
    segments : DataFrame
        One row per setting: `start` and `end` (s, the next setting's
        start), `analyzed_end` (s, the end of what psi analyzed), `f1`, `f2`,
        `l1`, `l2`,
        `rise_time`, `response_window`, `n_fft`, `n_time`, and psi's own
        measurements `psi_f1`, `psi_f2`, `psi_dp`, `psi_nf` (dB re the
        microphone calibration, here 1 V).
    gain_db : dict
        Route gain from the 'primary' and 'secondary' outputs to the
        microphone.
    '''
    checks = Checks()
    reconstructed = np.zeros_like(x)
    rows = []
    for s in segments.itertuples():
        s0 = int(round(s.start * fs))
        s1 = min(int(round(s.end * fs)), len(x))
        # Where the tones stop: the last sample above the floor.
        peak = np.abs(x[s0:s1]).max()
        s1 = s0 + np.flatnonzero(np.abs(x[s0:s1]) > 1e-4 * peak)[-1] + 1
        segment = x[s0:s1]
        env = ramp_on(len(segment), int(round(s.rise_time * fs)))
        dp = 2 * s.f1 - s.f2
        amplitude, phase, fitted = fit_tones(segment, fs, [s.f1, s.f2, dp],
                                             env)
        # The DP isn't in the template; only the two primaries are.
        _, _, primaries = fit_tones(segment, fs, [s.f1, s.f2], env)
        reconstructed[s0:s1] = primaries
        error = segment - primaries
        with np.errstate(divide='ignore'):
            residual_db = 10 * np.log10(np.sum(error ** 2)
                                        / np.sum(primaries ** 2))
        # A tone delayed by d samples has phase 2 pi f d / fs; wrap to the
        # nearest period.
        lags = [(p / (2 * np.pi * f) * fs + fs / f / 2) % (fs / f) - fs / f / 2
                for p, f in zip(phase[:2], [s.f1, s.f2])]
        rows.append({
            'f1_level': level_db(amplitude[0], gain_db['primary']),
            'f2_level': level_db(amplitude[1], gain_db['secondary']),
            'dp_re_f2': 20 * np.log10(amplitude[2] / amplitude[1]),
            'f1_lag': lags[0],
            'f2_lag': lags[1],
            'residual_db': residual_db,
            'duration': (s1 - s0) / fs,
            'tone_end': s1 / fs,
            # Level of the last millisecond re the plateau; 0 dB if the tones
            # are cut off rather than ramped down.
            'offset_db': 20 * np.log10(
                np.abs(segment[-int(fs * 1e-3):]).max()
                / np.abs(segment[len(segment) // 2:]).max()),
        })
    m = pd.concat([segments.reset_index(drop=True), pd.DataFrame(rows)],
                  axis=1)

    error = x - reconstructed
    unexplained = 10 * np.log10(np.sum(error ** 2) / np.sum(x ** 2))
    checks.add('nothing unlogged', unexplained <= RESIDUAL_LIMIT,
               f'{unexplained:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               'Energy in the whole recording the logged settings don\'t '
               'account for.')

    checks.add('engine latency', engine['latency_samples'] == latency,
               f'latency_samples {engine["latency_samples"]} at '
               f'{engine["fs"]} Hz, blocksize {engine["blocksize"]}',
               f'{latency} (measured just before the run)')
    worst = np.abs(m[['f1_lag', 'f2_lag']].values).max()
    checks.add('tones start on time', worst <= LAG_TOLERANCE,
               f'worst {worst:.2f} samples',
               f'≤ {LAG_TOLERANCE} samples',
               'Lag of each tone re the logged start of its setting, from '
               'its phase (tones start at phase 0).')

    error = np.r_[m.f1_level - m.l1, m.f2_level - m.l2]
    worst = np.abs(error).max()
    checks.add('level', worst <= LEVEL_TOLERANCE,
               f'worst error {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
               'f1 and f2 levels in the recording re L1 and L2.')
    # What psi's own analysis should report, in dB re 1 V (the microphone
    # calibration is unity).
    m['psi_f1_error'] = m.psi_f1 \
        - (m.l1 - OUTPUT_CAL_DB + INPUT_DBFS + gain_db['primary'])
    m['psi_f2_error'] = m.psi_f2 \
        - (m.l2 - OUTPUT_CAL_DB + INPUT_DBFS + gain_db['secondary'])
    worst = np.abs(np.r_[m.psi_f1_error, m.psi_f2_error]).max()
    checks.add('psi measures level', worst <= LEVEL_TOLERANCE,
               f'worst error {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
               'f1 and f2 levels psi saved, re what the loopback delivers.')
    worst = m.residual_db.max()
    checks.add('waveform', worst <= RESIDUAL_LIMIT,
               f'worst residual {worst:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               'What f1 and f2 leave unexplained in each setting (e.g., the '
               'previous tones still playing).')
    worst = max(m.dp_re_f2.max(), (m.psi_dp - m.psi_f2).max())
    checks.add('no distortion', worst <= DISTORTION_LIMIT,
               f'worst 2f1-f2 {worst:.1f} dB re f2',
               f'≤ {DISTORTION_LIMIT} dB',
               'Both in the recording and as psi measured it.')

    on_bin = [np.isclose(f * w, round(f * w)) for f, w in
              zip(np.r_[m.f1, m.f2], np.r_[m.response_window,
                                           m.response_window])]
    checks.add('whole cycles', all(on_bin),
               f'{sum(on_bin)} of {len(on_bin)} tones',
               'f1 and f2 are whole multiples of 1 / response window')
    needed = DISCARD_RISE_TIMES * m.rise_time \
        + m.n_fft * m.n_time * m.response_window
    short = int((m.duration < needed).sum())
    checks.add('settings long enough', short == 0,
               f'{short} of {len(m)} shorter than psi analyzes',
               'discard + n_fft × n_time × response window')
    margin = (m.tone_end - m.analyzed_end).min()
    gap = (m.end - m.tone_end).iloc[:-1]
    checks.add('tones cover analysis', margin >= 0,
               f'tones stop {margin * 1e3:.1f} ms or more after the end of '
               'what psi analyzed', 'after',
               f'Tones are then cut off (last ms '
               f'{m.offset_db.max():.1f} dB re plateau) and the next setting '
               f'starts {gap.min() * 1e3:.0f}–{gap.max() * 1e3:.0f} ms later.')
    return Analysis(checks, m, reconstructed)
