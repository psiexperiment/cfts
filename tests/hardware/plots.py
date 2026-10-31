'''
Figures for the cfts run reports (see `psi.loopback.plots` for the shared
ones).
'''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from psi.loopback.fit import gain_db
# Shared with the other psi programs; plot_timing is used by the tests.
from psi.loopback.plots import plot_timing, save  # noqa: F401


def format_key(names, values):
    '''
    Label for a panel, e.g. "4 kHz, 60 dB".
    '''
    parts = []
    for name, value in zip(names, values):
        if name in ('frequency', 'fc'):
            parts.append(f'{value / 1e3:g} kHz')
        elif name in ('level', 'l2', 'elicitor_level'):
            parts.append(f'{value:g} dB')
        elif name == 'burst':
            parts.append(f'burst {value + 1}')
        else:
            parts.append(f'{name} {value}')
    return ', '.join(parts)


def panel_grid(keys, row, col, panel_size=(3, 2.2), sharex=True):
    '''
    A grid of axes, one per unique (row, col) key, where `row` and `col` are
    lists of column names. Returns the figure and a function mapping a key
    (row values + col values) to its axes.
    '''
    n_row = len(row)
    rows = sorted({k[:n_row] for k in keys})
    cols = sorted({k[n_row:] for k in keys})
    figure, axes = plt.subplots(len(rows), len(cols), squeeze=False,
                                figsize=(panel_size[0] * len(cols),
                                         panel_size[1] * len(rows)),
                                sharex=sharex)

    def get_axes(key):
        return axes[rows.index(key[:n_row]), cols.index(key[n_row:])]
    return figure, axes, get_axes


def unique_keys(table, columns):
    return list(table[columns].drop_duplicates()
                .itertuples(index=False, name=None))


def as_tuple(key):
    return key if isinstance(key, tuple) else (key,)


def plot_timeline(fit, onsets, epoch_span, path, n=6):
    '''
    The start of the recording, with the logged onsets and the window psi
    averages for each, to show where the stimuli landed relative to them.
    '''
    fs = fit.fs
    t_start = onsets[0] - 0.005
    t_end = max(onsets[min(n, len(onsets) - 1)] + epoch_span,
                onsets[0] + fit.latency / fs) + 0.005
    lb, ub = int(t_start * fs), int(t_end * fs)
    t = np.arange(lb, ub) / fs
    figure, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t * 1e3, fit.recorded[lb:ub], 'k', lw=0.6, label='Recorded')
    for i, t0 in enumerate(onsets[:n + 1]):
        ax.axvspan(t0 * 1e3, (t0 + epoch_span) * 1e3, color='C0', alpha=0.15,
                   label='Epoch psi averages' if i == 0 else None)
        ax.axvline(t0 * 1e3, color='C0', lw=1,
                   label='Logged onset' if i == 0 else None)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Signal (V)')
    ax.set_title(f'Stimuli arrive {fit.latency} samples '
                 f'({fit.latency / fs * 1e3:.2f} ms) after their logged onset')
    ax.legend(loc='upper right', fontsize=8)
    return save(figure, path)


def plot_levels(stimuli, fit, expected_gain_db, path):
    '''
    Level each stimulus came back at, against the level requested.
    '''
    measured = stimuli.level.values + gain_db(fit.epochs.gain.values) \
        - expected_gain_db
    table = stimuli.assign(measured=measured)
    figure, axes = plt.subplots(1, 2, figsize=(10, 3.2))
    ax = axes[0]
    if 'frequency' in table:
        for frequency, group in table.groupby('frequency'):
            ax.plot(group.level, group.measured, 'o', ms=4,
                    label=f'{frequency / 1e3:g} kHz')
        ax.legend(fontsize=8)
    else:
        ax.plot(table.level, table.measured, 'o', ms=4)
    levels = np.unique(table.level)
    ax.plot(levels, levels, 'k:', lw=1)
    ax.set_xlabel('Requested (dB SPL)')
    ax.set_ylabel('Measured (dB SPL)')
    ax.set_title('Level')
    ax = axes[1]
    error = measured - stimuli.level.values
    ax.plot(fit.epochs.onset, error, 'o', ms=3)
    ax.set_xlabel('Onset (s)')
    ax.set_ylabel('Error (dB)')
    ax.set_title('Level error per stimulus')
    lim = max(0.2, np.nanmax(np.abs(error)) * 1.2)
    ax.set_ylim(-lim, lim)
    figure.tight_layout()
    return save(figure, path)


def plot_waveforms(stimuli, fit, templates, path, row=('level',),
                   col=('frequency',)):
    '''
    The first stimulus of each polarity for every condition, as recorded and
    as expected.
    '''
    fs = fit.fs
    row, col = list(row), list(col)
    groups = stimuli.groupby(row + col)
    figure, axes, get_axes = panel_grid(unique_keys(stimuli, row + col),
                                        row, col, panel_size=(3, 2))
    for key, group in groups:
        key = as_tuple(key)
        ax = get_axes(key)
        for polarity, color in ((1, 'C0'), (-1, 'C3')):
            g = group[group.polarity == polarity]
            if not len(g):
                continue
            i = g.index[0]
            template = templates[i] * fit.epochs.gain[i]
            lb = int(round(fit.epochs.onset[i] * fs))
            recorded = fit.recorded[lb:lb + len(template)]
            t = np.arange(len(template)) / fs * 1e3
            ax.plot(t, recorded, color=color, lw=1,
                    label=f'Recorded ({polarity:+d})')
            ax.plot(t, template, 'k:', lw=1)
        ax.set_title(format_key(row + col, key), fontsize=9)
    axes[0, 0].legend(fontsize=7)
    for ax in axes[-1]:
        ax.set_xlabel('Time re onset (ms)')
    figure.tight_layout()
    return save(figure, path)


def plot_epochs(epochs, alignment, cancellation, path, row=('level',),
                col=('frequency',)):
    '''
    Every epoch psi averages, overlaid by polarity, with the mean of the + and
    - averages (which alternating polarity should cancel to nothing).
    '''
    row, col = list(row), list(col)
    groups = epochs.groupby(row + col)
    figure, axes, get_axes = panel_grid(
        unique_keys(epochs.index.to_frame(index=False), row + col), row, col)
    t = epochs.columns.values * 1e3
    for key, group in groups:
        key = as_tuple(key)
        ax = get_axes(key)
        averages = []
        for polarity, color in ((1, 'C0'), (-1, 'C3')):
            g = group.xs(polarity, level='polarity')
            ax.plot(t, g.values.T, color=color, lw=0.5, alpha=0.4)
            averages.append(g.values.mean(axis=0))
        # Mean of the two polarity averages, as `polarity_cancellation` does,
        # so an odd number of trials doesn't leave a burst behind.
        ax.plot(t, np.mean(averages, axis=0), 'k', lw=1.2)
        worst = alignment.loc[group.index, 'trial_db'].max()
        # Worst cancellation among the conditions shown in this panel.
        mask = np.ones(len(cancellation), dtype=bool)
        for name, value in zip(row + col, key):
            if name in cancellation.index.names:
                mask &= cancellation.index.get_level_values(name) == value
        cancel = cancellation[mask].max()
        ax.set_title(f'{format_key(row + col, key)} (n={len(group)})\n'
                     f'+/− cancel {cancel:.0f} dB, '
                     f'trials differ {worst:.0f} dB', fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Time re epoch start (ms)')
    figure.suptitle('Epochs psi averages: + (blue), − (red), mean of the + '
                    'and − averages (black)', fontsize=10)
    figure.tight_layout()
    return save(figure, path)


def plot_dpoae_timeline(x, fs, analysis, path, n=3):
    '''
    The microphone over the first settings, and a close-up of each change of
    setting: the previous tones should stop and the new ones ramp on at the
    logged start.
    '''
    m = analysis.segments
    figure = plt.figure(figsize=(10, 5.5))
    ax = figure.add_subplot(2, 1, 1)
    end = m.end.iloc[min(n, len(m)) - 1]
    ub = min(int(end * fs), len(x))
    t = np.arange(ub) / fs
    ax.plot(t, x[:ub], 'k', lw=0.3)
    for s in m.start.iloc[:n]:
        ax.axvline(s, color='C0', lw=1)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Microphone (V)')
    ax.set_title('Recording, with the logged start of each setting')

    starts = m.start.iloc[:n].values
    for i, s in enumerate(starts):
        ax = figure.add_subplot(2, len(starts), len(starts) + i + 1)
        s0 = int(round(s * fs))
        lb, ub = max(0, s0 - int(0.005 * fs)), s0 + int(0.03 * fs)
        t = (np.arange(lb, ub) - s0) / fs * 1e3
        ax.plot(t, x[lb:ub], 'k', lw=0.6, label='Recorded')
        ax.plot(t, analysis.reconstructed[lb:ub], 'C1:', lw=1,
                label='Expected')
        ax.axvline(0, color='C0', lw=1)
        ax.set_xlabel('Time re start (ms)')
        r = m.iloc[i]
        ax.set_title(f'f2 {r.f2 / 1e3:g} kHz, L2 {r.l2:g} dB', fontsize=9)
    ax.legend(fontsize=7)
    figure.tight_layout()
    return save(figure, path)


def plot_dpoae_levels(analysis, path):
    m = analysis.segments
    figure, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    for ax, tone in zip(axes[:2], ('f1', 'f2')):
        requested = m[f'l{tone[1]}']
        ax.plot(requested, m[f'{tone}_level'] - requested, 'o',
                label='Recording')
        ax.plot(requested, m[f'psi_{tone}_error'], 'x', label='psi online')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_xlabel(f'Requested {tone} level (dB SPL)')
        ax.set_ylabel('Error (dB)')
        ax.set_title(f'{tone} level')
        lim = max(0.2, np.nanmax(np.abs(np.r_[m[f'{tone}_level'] - requested,
                                              m[f'psi_{tone}_error']])) * 1.2)
        ax.set_ylim(-lim, lim)
    axes[0].legend(fontsize=8)
    ax = axes[2]
    i = np.arange(len(m))
    ax.plot(i, m.psi_f2, 'o', label='f2 (psi)')
    ax.plot(i, m.psi_dp, 's', label='2f1-f2 (psi)')
    ax.plot(i, m.psi_nf, 'v', label='noise floor (psi)')
    ax.set_xticks(i, [f'{f / 1e3:g}k/{l:g}' for f, l in zip(m.f2, m.l2)],
                  rotation=45, fontsize=7)
    ax.set_ylabel('dB re 1 V')
    ax.set_title('What psi measured')
    ax.legend(fontsize=7)
    figure.tight_layout()
    return save(figure, path)


def plot_dpoae_timing(analysis, path):
    m = analysis.segments
    figure, ax = plt.subplots(figsize=(6, 3.2))
    i = np.arange(len(m))
    ax.plot(i, m.f1_lag, 'o', label='f1')
    ax.plot(i, m.f2_lag, 's', label='f2')
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xticks(i, [f'{f / 1e3:g}k/{l:g}' for f, l in zip(m.f2, m.l2)],
                  rotation=45, fontsize=7)
    ax.set_ylabel('Lag (samples)')
    ax.set_title('Tone lag re logged start (from phase)')
    lim = max(1, np.nanmax(np.abs(m[['f1_lag', 'f2_lag']].values)) * 1.2)
    ax.set_ylim(-lim, lim)
    ax.legend(fontsize=8)
    figure.tight_layout()
    return save(figure, path)


def plot_dpoae_spectra(x, fs, analysis, path):
    '''
    Spectrum of each setting as psi analyzes it (after the discard), with
    f1, f2 and 2f1 - f2 marked.
    '''
    m = analysis.segments
    n_col = min(4, len(m))
    n_row = int(np.ceil(len(m) / n_col))
    figure, axes = plt.subplots(n_row, n_col, squeeze=False,
                                figsize=(3 * n_col, 2.4 * n_row))
    for ax, s in zip(axes.ravel(), m.itertuples()):
        s0 = int(round((s.start + 4 * s.rise_time) * fs))
        n = int(round(s.n_fft * s.n_time * s.response_window * fs))
        segment = x[s0:s0 + n]
        spectrum = np.abs(np.fft.rfft(segment)) / len(segment) * 2
        freq = np.fft.rfftfreq(len(segment), 1 / fs)
        with np.errstate(divide='ignore'):
            ax.plot(freq / 1e3, 20 * np.log10(spectrum), 'k', lw=0.4)
        for f, color in ((s.f1, 'C0'), (s.f2, 'C1'),
                         (2 * s.f1 - s.f2, 'C3')):
            ax.axvline(f / 1e3, color=color, lw=0.8, alpha=0.6)
        ax.set_xlim(0, s.f2 * 1.5 / 1e3)
        ax.set_title(f'f2 {s.f2 / 1e3:g} kHz, L2 {s.l2:g} dB', fontsize=9)
        ax.set_xlabel('Frequency (kHz)')
    for ax in axes.ravel()[len(m):]:
        ax.axis('off')
    axes[0, 0].set_ylabel('dB re 1 V peak')
    figure.suptitle('f1 (blue), f2 (orange), 2f1-f2 (red)', fontsize=10)
    figure.tight_layout()
    return save(figure, path)


def plot_moc_timeline(probe, elicitor, fs, analysis, p, path, n=2):
    '''
    Both microphones over the first trials, with the elicitor windows shaded.
    '''
    t0 = analysis.trials.t0.iloc[0]
    period = p['trial_duration']
    lb = max(0, int((t0 - 0.2) * fs))
    ub = min(len(probe), int((t0 + n * period) * fs))
    t = np.arange(lb, ub) / fs
    figure, axes = plt.subplots(2, 1, figsize=(10, 4.5), sharex=True)
    for ax, x, label in ((axes[0], probe, 'Probe microphone (V)'),
                         (axes[1], elicitor, 'Elicitor microphone (V)')):
        ax.plot(t, x[lb:ub], 'k', lw=0.3)
        ax.set_ylabel(label, fontsize=8)
        for i in range(n):
            onset = t0 + i * period + p['baseline_duration']
            ax.axvspan(onset, onset + p['elicitor_duration'], color='C1',
                       alpha=0.15, lw=0)
            ax.axvline(t0 + i * period, color='C0', lw=1)
    axes[-1].set_xlabel('Time (s)')
    axes[0].set_title('Trial onsets (blue) and elicitor (shaded)')
    figure.tight_layout()
    return save(figure, path)


def plot_moc_blocks(analysis, p, path):
    '''
    What psi reported for each block against time in the trial: the DP (each
    trial and the running average) and noise floor, and the error in f1 and
    f2.
    '''
    b = analysis.blocks
    figure, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    ax = axes[0]
    for trial, g in b.groupby('trial'):
        ax.plot(g.time, g.dpoae_level, color='lightcoral', lw=0.8,
                label='DP (each trial)' if trial == 0 else None)
    last = b[b.trial == b.trial.max()]
    ax.plot(last.time, last.avg_dpoae_level, color='darkred', lw=2,
            label='DP (average)')
    ax.plot(last.time, last.avg_dpoae_noise_floor, color='lightsteelblue',
            lw=2, label='Noise floor (average)')
    if 'dpoae_expected' in b:
        ax.plot(last.time, last.dpoae_expected, 'k:', lw=1, label='Expected')
    onset = p['baseline_duration']
    for a in axes:
        a.axvspan(onset, onset + p['elicitor_duration'], color='C1',
                  alpha=0.15, lw=0)
        a.set_xlabel('Time re trial onset (s)')
    ax.set_ylabel('dB re 1 V')
    ax.set_title('DPOAE as psi measured it')
    ax.legend(fontsize=7)

    ax = axes[1]
    ax.plot(b.time, b.f1_level - b.f1_expected, '.', ms=3, label='f1')
    ax.plot(b.time, b.f2_level - b.f2_expected, '.', ms=3, label='f2')
    ax.axhline(0, color='k', lw=0.5)
    lim = max(0.2, np.abs(np.r_[b.f1_level - b.f1_expected,
                                b.f2_level - b.f2_expected]).max() * 1.2)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel('Error (dB)')
    ax.set_title('Primaries as psi measured them')
    ax.legend(fontsize=7)
    figure.tight_layout()
    return save(figure, path)
