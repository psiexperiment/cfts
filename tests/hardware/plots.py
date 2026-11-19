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


def format_key(names, values, sep=', '):
    '''
    Label for a panel, e.g. "4 kHz, 60 dB". A frequency of 0 is a click.
    '''
    parts = []
    for name, value in zip(names, values):
        if name in ('frequency', 'fc', 'f2'):
            parts.append(f'{value / 1e3:g} kHz' if value else 'click')
        elif name in ('level', 'l2', 'elicitor_level'):
            parts.append(f'{value:g} dB')
        elif name == 'fm':
            parts.append(f'fm {value:g} Hz')
        elif name == 'burst':
            parts.append(f'burst {value + 1}')
        elif name == 'stimulus':
            parts.append(str(value))
        else:
            parts.append(f'{name} {value}')
    return sep.join(parts)


def polarity_sign(polarity):
    return '+' if polarity > 0 else '−'


def trial_labels(table, columns, sep='\n'):
    '''
    One label per row of `table`: its value of each of `columns`, then its
    polarity (if it has one).
    '''
    labels = []
    for _, r in table.iterrows():
        label = format_key(columns, [r[c] for c in columns], sep)
        if 'polarity' in table:
            label += f'{sep}({polarity_sign(r.polarity)})'
        labels.append(label)
    return labels


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


def plot_timeline(fit, onsets, epoch_span, path, n=6, labels=None,
                  extra=()):
    '''
    The start of the recording, with the logged onsets and the window psi
    averages for each, to show where the stimuli landed relative to them.

    Parameters
    ----------
    labels : list of str, optional
        What each stimulus was (e.g., from `trial_labels`), written above
        its onset.
    extra : list of (name, signal, fs)
        Other inputs recorded alongside (e.g., each ear's microphone), drawn
        below on the same time axis.
    '''
    fs = fit.fs
    t_start = onsets[0] - 0.005
    t_end = max(onsets[min(n, len(onsets) - 1)] + epoch_span,
                onsets[0] + fit.latency / fs) + 0.005
    traces = [('Recorded', fit.recorded, fs), *extra]
    height = 3 + 1.6 * len(extra) + (0.6 if labels is not None else 0)
    figure, axes = plt.subplots(len(traces), 1, figsize=(10, height),
                                sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, (name, x, x_fs) in zip(axes, traces):
        lb, ub = int(t_start * x_fs), min(int(t_end * x_fs), len(x))
        t = np.arange(lb, ub) / x_fs
        ax.plot(t * 1e3, x[lb:ub], 'k', lw=0.6, label=name)
        for i, t0 in enumerate(onsets[:n + 1]):
            ax.axvspan(t0 * 1e3, (t0 + epoch_span) * 1e3, color='C0',
                       alpha=0.15,
                       label='Epoch psi averages' if i == 0 else None)
            ax.axvline(t0 * 1e3, color='C0', lw=1,
                       label='Logged onset' if i == 0 else None)
        ax.set_ylabel(f'{name} (V)' if extra else 'Signal (V)', fontsize=8)
    ax = axes[0]
    ax.legend(loc='upper right', fontsize=8)
    if labels is not None:
        for t0, label in zip(onsets[:n + 1], labels):
            ax.text(t0 * 1e3, 1.02, label, transform=ax.get_xaxis_transform(),
                    ha='left', va='bottom', fontsize=7, color='C0')
    title = (f'Stimuli arrive {fit.latency} samples '
             f'({fit.latency / fs * 1e3:.2f} ms) after their logged onset')
    if labels is not None:
        figure.suptitle(title)
    else:
        ax.set_title(title)
    axes[-1].set_xlabel('Time (ms)')
    figure.tight_layout()
    return save(figure, path)


def plot_sequence(trials, conditions, path, title='Order of presentation',
                  unit='trial'):
    '''
    Every trial in the order played (x), on the row of its condition (y),
    coloured by polarity, so the ordering, interleaving and balance of the
    design can be checked by eye.

    `trials` has one row per trial; it is sorted by `onset`, `t0` or `start`
    if it has one of them.
    '''
    for column in ('onset', 't0', 'start'):
        if column in trials:
            trials = trials.sort_values(column, kind='stable')
            break
    trials = trials.reset_index(drop=True)
    keys = list(trials[conditions].itertuples(index=False, name=None))
    rows = sorted(set(keys))
    y = np.array([rows.index(k) for k in keys])
    x = np.arange(len(trials)) + 1

    figure, ax = plt.subplots(figsize=(10, 1 + 0.28 * max(len(rows), 3)))
    if 'polarity' in trials:
        for polarity, color in ((1, 'C0'), (-1, 'C3')):
            mask = trials.polarity.values == polarity
            ax.plot(x[mask], y[mask], '|', color=color, ms=8, mew=1.2,
                    label=f'{polarity_sign(polarity)} polarity')
        ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), fontsize=8)
    else:
        ax.plot(x, y, '|', color='k', ms=8, mew=1.2)
    # Connect successive trials so the path through the conditions shows.
    ax.plot(x, y, color='0.6', lw=0.4, zorder=0)
    counts = np.bincount(y, minlength=len(rows))
    ax.set_yticks(range(len(rows)),
                  [f'{format_key(conditions, k)} (n={c})'
                   for k, c in zip(rows, counts)], fontsize=7)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.set_xlim(0.5, len(trials) + 0.5)
    ax.set_xlabel(f'{unit.capitalize()} (order played)')
    ax.set_title(f'{title}: {len(trials)} {unit}s, {len(rows)} conditions',
                 fontsize=10)
    figure.tight_layout()
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


def plot_bic(results, path):
    '''
    For each frequency and level (+ polarity), the binaural averages at each
    ITD on the sum of the monaural averages (left column; each has its own
    sum, dashed), and the BIC at each ITD with what the loopback should give
    (right column).
    '''
    results = [r for r in results if r['polarity'] == 1]
    figure, axes = plt.subplots(len(results), 2, squeeze=False,
                                figsize=(9, 2.2 * len(results)), sharex=True)
    for (ax_overlay, ax_bic), r in zip(axes, results, strict=True):
        t = r['time'] * 1e3
        sums = r['sums']
        for i, (value, y) in enumerate(sorted(r['binaural'].items())):
            color = f'C{i}'
            label = f'{value:g} µs'
            ax_overlay.plot(t[:len(y)], y, color=color, lw=1, label=label)
            s = sums[value]
            ax_overlay.plot(t[:len(s)], s, color=color, lw=2, ls='--',
                            alpha=0.6)
            bic = r['bic'][value]
            ax_bic.plot(t[:len(bic)], bic, color=color, lw=1,
                        label=f'{label} ({r["error_db"][value]:.0f} dB)')
            expected = r['expected_bic'][value]
            ax_bic.plot(t[:len(expected)], expected, 'k:', lw=1)
        key = format_key(['frequency', 'level'], [r['frequency'], r['level']])
        ax_overlay.set_title(f'{key}: binaural vs. monaural (sums dashed)',
                             fontsize=9)
        ax_bic.set_title(f'{key}: BIC (expected dotted)', fontsize=9)
        ax_overlay.legend(fontsize=7, loc='upper right')
        ax_bic.legend(fontsize=7, loc='upper right')
    for ax in axes[-1]:
        ax.set_xlabel('Time re epoch start (ms)')
    figure.tight_layout()
    return save(figure, path)


def plot_itd(waveforms, table, path):
    '''
    For each frequency, both microphones at each ITD (left), and the
    measured ITD of every binaural burst against the requested one (right).

    The start of each trial, one row per ITD (left black, right coloured),
    so which ear starts first, and by how much, shows.
    '''
    frequencies = sorted({w['frequency'] for w in waveforms})
    values = sorted({w['value'] for w in waveforms})
    colors = {v: f'C{i}' for i, v in enumerate(values)}
    height = 0.9 * len(values) + 1
    figure, axes = plt.subplots(len(frequencies), 2, squeeze=False,
                                figsize=(9, height * len(frequencies)),
                                gridspec_kw={'width_ratios': [2, 1]})
    for (ax_wave, ax_itd), frequency in zip(axes, frequencies, strict=True):
        rows = [w for w in waveforms if w['frequency'] == frequency]
        for i, w in enumerate(sorted(rows, key=lambda w: w['value'])):
            offset = -2.5 * i
            t = w['x'] * 1e3
            ax_wave.plot(t, w['left'] + offset, 'k', lw=0.8)
            ax_wave.plot(t, w['right'] + offset, color=colors[w['value']],
                         lw=0.8, alpha=0.8)
            ax_wave.text(t[-1], offset, f' {w["value"]:g} µs',
                         va='center', fontsize=7, color=colors[w['value']])
        ax_wave.set_yticks([])
        ax_wave.set_xlabel('Time re trial start (ms)')
        ax_wave.set_title(f'{format_key(["frequency"], [frequency])}: '
                          'onsets, left black, right coloured', fontsize=9)

        g = table[table.frequency == frequency]
        ax_itd.plot([min(values), max(values)], [min(values), max(values)],
                    'k:', lw=1)
        ax_itd.scatter(g.requested, g.measured, s=12,
                       c=[colors[v] for v in g.requested])
        ax_itd.set_xticks(values)
        ax_itd.set_yticks(values)
        ax_itd.tick_params(labelsize=7)
        ax_itd.set_xlabel('Requested ITD (µs)')
        ax_itd.set_ylabel('Measured ITD (µs)')
        ax_itd.set_title(f'{format_key(["frequency"], [frequency])}: '
                         'every binaural burst',
                         fontsize=9)
    figure.tight_layout()
    return save(figure, path)


EAR_COLORS = {'left': 'C0', 'right': 'C1'}


def plot_bic_trials(left, right, fs, onsets, labels, path, n=8, first=0):
    '''
    `n` consecutive trials as each ear's microphone recorded them, sample
    for sample: both ears over the whole stretch (top two rows, with the
    interval from each burst to the ear's next one), then each trial's
    onset close up with both ears overlaid (bottom row).

    Dashed lines are where the requested ITD says each ear's burst should
    start; `onsets` is from `bic_analysis.burst_onsets` and `labels` has one
    entry per trial.
    '''
    trials = sorted(onsets.trial.unique())[first:first + n]
    o = onsets[onsets.trial.isin(trials)]
    # The logged onset (plus latency) of each trial, in samples.
    start = {t: g.expected.min() for t, g in o.groupby('trial')}
    t_ref = onsets[onsets.trial == trials[0]].expected.min()
    lb = t_ref - int(0.003 * fs)
    ub = o.measured.max() + int(0.012 * fs)

    figure = plt.figure(figsize=(max(12, 1.7 * len(trials)), 8.5))
    grid = figure.add_gridspec(3, len(trials), height_ratios=[1, 1, 1.3],
                               hspace=0.55)
    ax_l = figure.add_subplot(grid[0, :])
    ax_r = figure.add_subplot(grid[1, :], sharex=ax_l, sharey=ax_l)
    t = (np.arange(lb, ub) - t_ref) / fs * 1e3
    for ax, name, x in ((ax_l, 'left', left), (ax_r, 'right', right)):
        ax.plot(t, x[lb:ub], color='k', lw=0.4)
        g = o[o.ear == name]
        for r in g.itertuples():
            ax.axvline((r.expected - t_ref) / fs * 1e3,
                       color=EAR_COLORS[name], ls='--', lw=0.8)
        # The interval from each burst to this ear's next one, measured,
        # and its error re what the logged onsets and ITDs call for.
        m, e = g.measured.values, g.expected.values
        for i in range(len(g) - 1):
            interval = (m[i + 1] - m[i]) / fs * 1e3
            error = (m[i + 1] - m[i]) - (e[i + 1] - e[i])
            mid = ((m[i] + m[i + 1]) / 2 - t_ref) / fs * 1e3
            ax.text(mid, 0.97, f'{interval:.3f} ms\n({error:+d} samples)',
                    transform=ax.get_xaxis_transform(), ha='center',
                    va='top', fontsize=7)
        ax.set_ylabel(f'{name.capitalize()} mic (V)', fontsize=8)
    for trial, label in zip(trials, labels):
        ax_l.text((start[trial] - t_ref) / fs * 1e3, 1.03, label,
                  transform=ax_l.get_xaxis_transform(), ha='left',
                  va='bottom', fontsize=7)
    ax_r.set_xlabel('Time re first trial\'s logged onset + latency (ms)')
    ax_l.set_xlim(t[0], t[-1])

    # Close-ups: from just before the trial to just after the later ear
    # starts.
    for j, trial in enumerate(trials):
        ax = figure.add_subplot(grid[2, j])
        g = o[o.trial == trial]
        s0 = start[trial]
        z_lb = s0 - int(0.0003 * fs)
        z_ub = g.expected.max() + int(0.0015 * fs)
        tz = (np.arange(z_lb, z_ub) - s0) / fs * 1e6
        for r in g.itertuples():
            x = left if r.ear == 'left' else right
            ax.plot(tz, x[z_lb:z_ub], color=EAR_COLORS[r.ear], lw=0.8)
            ax.axvline((r.expected - s0) / fs * 1e6,
                       color=EAR_COLORS[r.ear], ls='--', lw=0.8)
        parts = labels[j].split('\n')
        title = ' '.join(parts[:2]) + '\n' + ' '.join(parts[2:])
        if len(g) == 2:
            l, r = g.set_index('ear').measured[['left', 'right']]
            lead = (l - r) / fs * 1e6
            title += f'\nmeasured: R {"leads" if lead >= 0 else "lags"} ' \
                     f'{abs(lead):.0f} µs'
        ax.set_title(title, fontsize=6.5)
        ax.tick_params(labelsize=6)
        ax.set_xlabel('µs re onset', fontsize=7)
        if j == 0:
            ax.set_ylabel('V', fontsize=7)
    figure.suptitle(f'Trials {trials[0] + 1}–{trials[-1] + 1}: left mic '
                    '(blue), right mic (orange); dashed, where each burst '
                    'should start for the requested ITD', fontsize=10)
    return save(figure, path)


def plot_bic_intervals(onsets, fs, path):
    '''
    Every burst of the run, per ear: where it started re where it should
    have, the interval since the ear's previous burst, and that interval's
    error.
    '''
    figure, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for name, g in onsets.groupby('ear'):
        style = dict(color=EAR_COLORS[name], ms=3, label=name,
                     marker='o' if name == 'left' else 'x', ls='none')
        trial = g.trial.values + 1
        axes[0].plot(trial, g.error, **style)
        m, e = g.measured.values, g.expected.values
        axes[1].plot(trial[1:], np.diff(m) / fs * 1e3, **style)
        axes[2].plot(trial[1:], np.diff(m) - np.diff(e), **style)
    axes[0].set_ylabel('Onset error\n(samples)')
    axes[0].set_title('Burst onset on each microphone re the logged onset '
                      '+ requested ITD delay', fontsize=9)
    axes[1].set_ylabel('Interval (ms)')
    axes[1].set_title('Interval from the same ear\'s previous burst',
                      fontsize=9)
    axes[2].set_ylabel('Interval error\n(samples)')
    axes[2].set_title('Interval re the logged onsets and requested ITDs',
                      fontsize=9)
    for ax in (axes[0], axes[2]):
        ax.axhline(0, color='k', lw=0.5)
        lim = max(2, np.nanmax(np.abs(ax.get_ylim())))
        ax.set_ylim(-lim, lim)
    axes[0].legend(fontsize=8, loc='upper right')
    axes[-1].set_xlabel('Trial')
    figure.tight_layout()
    return save(figure, path)
