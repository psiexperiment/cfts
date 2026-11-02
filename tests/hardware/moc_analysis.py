'''
Checks for the MOC suppression paradigm (`moc_dpoae`).

f1 and f2 play continuously on the probe Starship from the first trial's
onset until the experiment stops, and are summed into its microphone. The
elicitor (frozen bandlimited noise, the same every trial) plays on the
elicitor Starship as back-to-back trials, each `baseline` s of silence, then
the noise, then `recovery` s of silence, and only its own microphone should
hear it. A digital loopback has no cochlea, so there is no DP unless the
`moc_synthetic` manifest adds one: a tone at 2f1 - f2 that drops by a known
amount while the elicitor plays (see `suppression_envelope`).
'''
from dataclasses import dataclass

import numpy as np
import pandas as pd

from psiaudio import stim
from psiaudio.calibration import FlatCalibration
from psi.loopback.fit import fit_stimuli, gain_db
from psi.loopback.report import Checks

from dpoae_analysis import fit_tones, level_db, ramp_on
from rig import INPUT_DBFS
from runner import OUTPUT_CAL_DB
from synthetic_dp import suppression_envelope


#: Largest acceptable error in level, in dB.
LEVEL_TOLERANCE = 0.1

#: Largest acceptable lag of a tone re the logged start, in samples.
LAG_TOLERANCE = 0.5

#: How far below the stimuli whatever they leave unexplained must be.
RESIDUAL_LIMIT = -60

#: How far below f2 the level at 2f1 - f2 must be when there is no DP.
DISTORTION_LIMIT = -90

#: Rise time of the primaries (`moc.enaml`).
PRIMARY_RISE_TIME = 25e-3


@dataclass
class Analysis:
    checks: Checks
    #: The paradigm's trial and block tables, with what each should be.
    trials: pd.DataFrame
    blocks: pd.DataFrame
    #: Fit of the elicitor template to each trial on the elicitor microphone.
    elicitor_fit: object
    #: Fitted primaries (and synthetic DP) over the probe recording.
    reconstructed: np.ndarray
    #: First sample of the primaries and the sample after they stop.
    span: tuple


def elicitor_template(fs, p, seed):
    '''
    One trial of the elicitor, as psi plays it (a `BandlimitedFIRNoise` in a
    `Cos2Envelope`, with the trial's seed), up to the end of the noise. The
    silence after it is left out: the run stops partway through a trial.
    '''
    calibration = FlatCalibration.from_db(OUTPUT_CAL_DB)
    noise = stim.BandlimitedFIRNoiseFactory(
        fs, p['elicitor_fl'], p['elicitor_fh'], p['elicitor_level'],
        seed=seed, equalize=False, calibration=calibration,
        correct_level=True)
    factory = stim.Cos2EnvelopeFactory(fs, p['elicitor_duration'],
                                       p['elicitor_rise_time'], noise,
                                       start_time=p['baseline_duration'])
    return factory.get_samples_remaining()


def expected_db_re_v(level, gain):
    '''
    What psi should report (dB re 1 V, as the microphone calibration is
    unity) for a tone at `level` dB SPL through a route of `gain` dB.
    '''
    return level - OUTPUT_CAL_DB + INPUT_DBFS + gain


def analyze_moc(probe, elicitor, fs, trials, blocks, p, gain, latency,
                engine, synthetic=None):
    '''
    Parameters
    ----------
    probe, elicitor : array
        The two microphone recordings.
    trials, blocks : DataFrame
        The paradigm's `moc_trial_store` and `moc_block_store` tables.
    p : dict
        The paradigm's settings: `baseline_duration`, `elicitor_duration`,
        `recovery_duration`, `elicitor_rise_time`, `response_window`,
        `trial_n`, `elicitor_level`, `elicitor_fl`, `elicitor_fh`,
        `elicitor_seed`, plus
        `trial_duration`.
    gain : dict
        Route gain (dB) of the 'primary', 'secondary' and 'elicitor' outputs
        to their microphones.
    synthetic : dict or None
        `level` and `suppression` of the synthetic DP, if it played.
    '''
    checks = Checks()
    trials = trials.sort_values('trial').reset_index(drop=True)
    f1 = trials.primary_tone_frequency.iloc[0]
    f2 = trials.secondary_tone_frequency.iloc[0]
    l1 = trials.primary_tone_level.iloc[0]
    l2 = trials.secondary_tone_level.iloc[0]
    dp = 2 * f1 - f2
    rw = p['response_window']
    period = p['trial_duration']
    start = trials.t0.iloc[0]

    checks.add('engine latency', engine['latency_samples'] == latency,
               f'latency_samples {engine["latency_samples"]} at '
               f'{engine["fs"]} Hz, blocksize {engine["blocksize"]}',
               f'{latency} (measured just before the run)')

    # Trials.
    n_ok = trials.accepted.sum()
    steps = np.diff(trials.t0.values)
    checks.add('trial sequence',
               (n_ok == p['trial_n']) and trials.accepted.all()
               and np.allclose(steps, period, atol=1 / fs)
               and list(trials.trial) == list(range(len(trials))),
               f'{n_ok} of {len(trials)} accepted; onsets '
               f'{", ".join(f"{s:.4f}" for s in steps)} s apart',
               f'{p["trial_n"]} accepted, every {period:g} s',
               'A loopback has nothing to reject; trials play back to back.')

    # The primaries (and synthetic DP) over the whole run, from the first
    # trial's onset to where they stop.
    s0 = int(round(start * fs))
    peak = np.abs(probe[s0:]).max()
    s1 = s0 + np.flatnonzero(np.abs(probe[s0:]) > 1e-4 * peak)[-1] + 1
    x = probe[s0:s1]
    t = np.arange(len(x)) / fs
    env = ramp_on(len(x), int(round(PRIMARY_RISE_TIME * fs)))
    amplitude, phase, fitted = fit_tones(x, fs, [f1, f2, dp], env)
    if synthetic is not None:
        dp_env = env * suppression_envelope(
            t, period, p['baseline_duration'], p['elicitor_duration'],
            p['elicitor_rise_time'], synthetic['suppression'])
        dp_amp, dp_phase, dp_fitted = fit_tones(x, fs, [dp], dp_env)
        _, _, primaries = fit_tones(x, fs, [f1, f2], env)
        fitted = primaries + dp_fitted
    else:
        _, _, fitted = fit_tones(x, fs, [f1, f2], env)
    reconstructed = np.zeros_like(probe)
    reconstructed[s0:s1] = fitted
    error = x - fitted
    residual = 10 * np.log10(np.sum(error ** 2) / np.sum(fitted ** 2))
    checks.add('primaries continuous', residual <= RESIDUAL_LIMIT,
               f'{residual:.1f} dB over {len(x) / fs:.1f} s',
               f'≤ {RESIDUAL_LIMIT} dB',
               'One ramp on and constant f1 and f2 from the first trial to '
               'the end of the run explain the probe microphone: no '
               'restarts or gaps between trials.')

    in_elicitor = ((t % period) >= p['baseline_duration']) \
        & ((t % period) < p['baseline_duration'] + p['elicitor_duration'])
    leak = 10 * np.log10(np.sum(error[in_elicitor] ** 2)
                         / np.sum(fitted[in_elicitor] ** 2))
    checks.add('elicitor isolated', leak <= RESIDUAL_LIMIT,
               f'{leak:.1f} dB while the elicitor plays',
               f'≤ {RESIDUAL_LIMIT} dB',
               'Nothing but the probe stimuli on the probe microphone.')

    lags = [(ph / (2 * np.pi * f) * fs + fs / f / 2) % (fs / f) - fs / f / 2
            for ph, f in zip(phase[:2], [f1, f2], strict=True)]
    worst = np.abs(lags).max()
    checks.add('tones start on time', worst <= LAG_TOLERANCE,
               f'worst {worst:.2f} samples', f'≤ {LAG_TOLERANCE} samples',
               'Lag of f1 and f2 re the first trial\'s logged onset, from '
               'their phase.')

    errors = [level_db(amplitude[0], gain['primary']) - l1,
              level_db(amplitude[1], gain['secondary']) - l2]
    worst = np.abs(errors).max()
    checks.add('primary level', worst <= LEVEL_TOLERANCE,
               f'f1 {errors[0]:+.3f}, f2 {errors[1]:+.3f} dB',
               f'± {LEVEL_TOLERANCE} dB')

    if synthetic is None:
        dp_re_f2 = 20 * np.log10(amplitude[2] / amplitude[1])
        checks.add('no distortion', dp_re_f2 <= DISTORTION_LIMIT,
                   f'{dp_re_f2:.1f} dB re f2', f'≤ {DISTORTION_LIMIT} dB')
    else:
        err = level_db(dp_amp[0], gain['primary']) - synthetic['level']
        checks.add('synthetic DP played', abs(err) <= LEVEL_TOLERANCE,
                   f'{err:+.3f} dB', f'± {LEVEL_TOLERANCE} dB',
                   'Level of the synthetic DP in the recording re requested.')

    # The elicitor, once per trial for as long as the recording lasts.
    end = p['baseline_duration'] + p['elicitor_duration']
    onsets = start + period * np.arange(int((len(elicitor) / fs - start
                                             - end) // period) + 1)
    # Trial i's noise is seeded with the starting seed plus i.
    seeds = p['elicitor_seed'] + np.arange(len(onsets))
    checks.add('elicitor seeds',
               list(trials.elicitor_bandlimited_noise_seed)
               == list(seeds[:len(trials)]),
               ', '.join(str(s) for s in trials.elicitor_bandlimited_noise_seed),
               f'{p["elicitor_seed"]} plus the trial number')
    checks.add('no extra trial', len(onsets) == p['trial_n'],
               f'{len(onsets)} elicitors played', f'{p["trial_n"]}',
               'The run stops once the last trial is accepted, before the '
               "next trial's elicitor.")
    templates = [elicitor_template(fs, p, s) for s in seeds]
    fit = fit_stimuli(elicitor, fs, onsets, templates, latency=0)
    e = fit.epochs
    worst = np.nanmax(np.abs(e.lag))
    checks.add('elicitor timing', worst == 0,
               f'worst lag {worst:g} samples over {len(e)} trials',
               '0 samples', 'Each trial\'s noise re its onset (the first '
               'trial\'s logged onset plus whole trials).')
    error = gain_db(e.gain) - INPUT_DBFS - gain['elicitor']
    worst = np.nanmax(np.abs(error))
    checks.add('elicitor level', worst <= LEVEL_TOLERANCE,
               f'worst {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
               f'Noise re {p["elicitor_level"]:g} dB SPL, regenerated from '
               'its seed.')
    worst = np.nanmax(e.residual_db)
    unexplained = fit.unexplained_db
    checks.add('elicitor waveform', (worst <= RESIDUAL_LIMIT)
               and (unexplained <= RESIDUAL_LIMIT),
               f'worst trial {worst:.1f} dB; whole recording '
               f'{unexplained:.1f} dB', f'≤ {RESIDUAL_LIMIT} dB',
               'Each trial\'s noise from its seed, and nothing else on the '
               'elicitor microphone.')

    # What psi reported for each block.
    b = blocks.merge(trials[['trial', 't0']], on='trial')
    n_blocks = int(np.floor(period / rw + 1e-9))
    expected_blocks = len(trials) * n_blocks - 1
    contiguous = np.allclose(b.ts, b.t0 + b.block * rw, atol=1 / fs) \
        and np.allclose(b.time, (b.block + 0.5) * rw)
    checks.add('psi blocks', len(b) == expected_blocks and contiguous,
               f'{len(b)} blocks', f'{expected_blocks} (all but the first, '
               'while the primaries ramp on)',
               f'{rw * 1e3:g} ms blocks from each trial\'s onset.')

    b['f1_expected'] = expected_db_re_v(l1, gain['primary'])
    b['f2_expected'] = expected_db_re_v(l2, gain['secondary'])
    worst = np.abs(np.r_[b.f1_level - b.f1_expected,
                         b.f2_level - b.f2_expected]).max()
    checks.add('psi measures primaries', worst <= LEVEL_TOLERANCE,
               f'worst error {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
               'f1 and f2 in every block.')

    if synthetic is None:
        worst = (b.dpoae_level - b.f2_level).max()
        checks.add('psi sees no DP', worst <= DISTORTION_LIMIT,
                   f'worst {worst:.1f} dB re f2', f'≤ {DISTORTION_LIMIT} dB')
    else:
        # Each block's DP is the synthetic DP times its envelope averaged
        # over the block.
        n = int(round(rw * fs))
        tt = (b.ts.values - start)[:, np.newaxis] + np.arange(n) / fs
        e_mean = suppression_envelope(
            tt, period, p['baseline_duration'], p['elicitor_duration'],
            p['elicitor_rise_time'], synthetic['suppression']).mean(axis=1)
        b['dpoae_expected'] = expected_db_re_v(synthetic['level'],
                                               gain['primary']) \
            + 20 * np.log10(e_mean)
        err = b.dpoae_level - b.dpoae_expected
        worst = np.abs(err).max()
        checks.add('psi DP time course', worst <= LEVEL_TOLERANCE,
                   f'worst error {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
                   'DP in every block re the synthetic DP\'s envelope, '
                   'including the ramps.')
        err = b.avg_dpoae_level - b.dpoae_expected
        worst = np.abs(err).max()
        checks.add('psi average', worst <= LEVEL_TOLERANCE,
                   f'worst error {worst:.3f} dB', f'± {LEVEL_TOLERANCE} dB',
                   'Running average across trials, block by block.')
        s = synthetic['suppression']
        worst = np.abs(np.r_[trials.suppression, trials.avg_suppression]
                       - s).max()
        checks.add('psi suppression', worst <= LEVEL_TOLERANCE,
                   f'{", ".join(f"{v:.3f}" for v in trials.suppression)} dB '
                   f'(average {trials.avg_suppression.iloc[-1]:.3f})',
                   f'{s:g} ± {LEVEL_TOLERANCE} dB',
                   'Baseline re elicitor window, per trial and averaged.')

    return Analysis(checks, trials, b, fit, reconstructed, (s0, s1))
