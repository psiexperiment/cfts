from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import palettable

from psiaudio.plot import waterfall_plot
from psiaudio import util

from .memr import InterleavedMEMRFile, SimultaneousMEMRFile
from .util import DatasetManager


def get_colors(n):
    return getattr(palettable.cmocean.sequential, f'Deep_{n}').mpl_colors


def process_interleaved_file(filename, reprocess=False):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed(['MEMR.pdf']):
        return
    fh = InterleavedMEMRFile(filename)

    # Load variables we need from the file
    probe_cal = fh.probe_microphone.get_calibration()
    elicitor_cal = fh.elicitor_microphone.get_calibration()

    fs = fh.probe_microphone.fs
    period = fh.get_setting('repeat_period')
    probe_delay = fh.get_setting('probe_delay')
    probe_duration = fh.get_setting('probe_duration')
    elicitor_delay = fh.get_setting('elicitor_envelope_start_time')
    elicitor_fl = fh.get_setting('elicitor_fl')
    elicitor_fh = fh.get_setting('elicitor_fh')
    probe_fl = fh.get_setting('probe_fl')
    probe_fh = fh.get_setting('probe_fh')
    elicitor_n = fh.get_setting('elicitor_n')

    # First, plot the entire stimulus train. We only plot the positive polarity
    # because if we average in the negative polarity, the noise will cancel
    # out. If we invert then average in the negative polarity, the chirp will
    # cancel out! We just can't win.
    epochs = fh.get_epochs()
    epochs_pos = epochs.xs(1, level='elicitor_polarity')
    epochs_mean = epochs_pos.groupby('elicitor_level').mean()

    figsize = 6, 1 * len(epochs)
    figure, ax = plt.subplots(1, 1, figsize=figsize)
    waterfall_plot(ax, epochs, 'elicitor_level', scale_method='mean',
                   plotkw={'lw': 0.1, 'color': 'k'},
                   x_transform=lambda x: x*1e3)
    ax.set_xlabel('Time (msec)')
    ax.grid(False)
    # Draw lines showing the repeat boundaries
    for i in range(elicitor_n + 2):
        ax.axvline(i * period * 1e3, zorder=-1, alpha=0.5)
    # Save the figure
    figure.savefig(manager.get_proc_filename('stimulus train.pdf'), bbox_inches='tight')

    # Now, load the repeats. This essentially segments the epochs DataFrame
    # into the individual repeat segments.
    repeats = fh.get_repeats()

    m = repeats.columns >= elicitor_delay
    elicitor = repeats.loc[:, m]
    elicitor_psd = util.psd_df(elicitor, fs=fs)
    elicitor_spl = elicitor_cal.get_db(elicitor_psd)
    # Be sure to throw out the last "repeat" (which has a silent period after
    # it rather than another elicitor).
    elicitor_psd_mean = elicitor_psd.query('repeat < @elicitor_n').groupby('elicitor_level').mean()
    elicitor_spl_mean = elicitor_cal.get_db(elicitor_psd_mean)

    # Plot the elicitor for each level as a waterfall plot
    figure, ax = plt.subplots(1, 1, figsize=figsize)
    waterfall_plot(ax, elicitor_spl_mean.dropna(axis=1), 'elicitor_level', scale_method='mean', plotkw={'lw': 0.1, 'color': 'k'})
    ax.set_xscale('octave')
    ax.axis(xmin=0.5e3, xmax=50e3)
    ax.set_xlabel('Frequency (kHz)')
    figure.savefig(manager.get_proc_filename('elicitor PSD.pdf'), bbox_inches='tight')

    acoustic_delay = 0.75e-3
    lb = acoustic_delay + probe_delay
    ub = acoustic_delay + probe_delay + probe_duration
    m = (repeats.columns >= lb) & (repeats.columns < ub)
    probe = repeats.loc[:, m]

    lb = acoustic_delay + probe_delay + probe_duration
    ub = acoustic_delay + probe_delay + probe_duration * 2
    m = (repeats.columns >= lb) & (repeats.columns < ub)
    silence = repeats.loc[:, m]

    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(probe.columns.values * 1e3, probe.values.T, alpha=0.1, color='k', lw=0.1);
    ax.plot(silence.columns.values * 1e3, silence.values.T, alpha=0.1, color='r', lw=0.1);
    ax.set_xlabel('Time (msec)')
    ax.set_ylabel('Signal (V)')
    figure.savefig(manager.get_proc_filename('probe waveform.pdf'), bbox_inches='tight')

    probe_psd = util.psd_df(probe, fs)
    probe_spl = probe_cal.get_db(probe_psd)
    silence_psd = util.psd_df(silence, fs)
    silence_spl = probe_cal.get_db(silence_psd)
    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(silence_spl.columns, silence_spl.values.T, alpha=0.1, color='r', lw=0.1);
    ax.plot(probe_spl.columns, probe_spl.values.T, alpha=0.1, color='k', lw=0.1);
    ax.set_xscale('octave')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Level (dB SPL)')
    ax.axvline(probe_fl)
    ax.axvline(probe_fh)
    figure.savefig(manager.get_proc_filename('probe PSD.pdf'), bbox_inches='tight')

    probe_level = probe_spl.loc[:, probe_fl:probe_fh].apply(util.rms_rfft_db, axis=1)
    silence_level = silence_spl.loc[:, probe_fl:probe_fh].apply(util.rms_rfft_db, axis=1)

    lb, ub = np.percentile(silence_level, [2.5, 100-2.5])

    figure, ax = plt.subplots(1, 1, figsize=(4, 4))
    m = (silence_level > lb) & (silence_level < ub)
    ax.plot(silence_level.loc[m], probe_level.loc[m], 'k.')
    ax.plot(silence_level.loc[~m], probe_level.loc[~m], 'r.')
    ax.set_xlabel('Silence level (dB SPL)')
    ax.set_ylabel('Probe level (dB SPL)')
    figure.savefig(manager.get_proc_filename('probe level.pdf'), bbox_inches='tight')

    memr = probe_spl - probe_spl.xs(0, level='repeat')
    memr_mean = memr.groupby(['repeat', 'elicitor_level']).mean().loc[:, probe_fl:probe_fh]

    probe_n = elicitor_n + 1
    figure, axes = plt.subplots(1, probe_n, figsize=(8*probe_n, 4))
    memr_mean_end = memr_mean.loc[elicitor_n, probe_fl:probe_fh]

    for p, ax in enumerate(axes.flat):
        m = memr_mean.loc[p]
        colors = get_colors(len(m))
        for c, (level, value) in zip(colors, m.iterrows()):
            ax.plot(value, label=f'{level} dB SPL', color=c)
        ax.set_title(f'Probe {p}')

    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xscale('octave')
    figure.tight_layout()
    figure.savefig(manager.get_proc_filename('MEMR.pdf'), bbox_inches='tight')
    plt.close('all')


def process_simultaneous_file(filename, reprocess=False):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed(['elicitor PSD.pdf']):
        return
    fh = SimultaneousMEMRFile(filename)

    cal = fh.probe_microphone.get_calibration()
    fs = fh.probe_microphone.fs
    repeats = fh.get_repeats()
    probe_window = fh.get_setting('probe_duration') + 1.5e-3
    probes = repeats.loc[:, :probe_window]
    probe_mean = probes.groupby(['elicitor_level', 'group']).mean()
    probe_spl = cal.get_db(util.psd_df(probe_mean, fs=fs))
    probe_spl_mean = probe_spl.groupby(['elicitor_level', 'group']).mean()
    baseline = probe_spl_mean.xs('baseline', level='group')
    elicitor = probe_spl_mean.xs('elicitor_ss', level='group')
    memr = elicitor - baseline

    epochs = fh.get_epochs()
    onset = fh.get_setting('elicitor_onset')
    duration = fh.get_setting('elicitor_duration')
    elicitor = epochs.loc[:, onset:onset+duration]
    elicitor_waveform = elicitor.loc[1].groupby(['elicitor_level']).mean()
    elicitor_spl = cal.get_db(util.psd_df(elicitor, fs=fs)).dropna(axis='columns')
    elicitor_spl_mean = elicitor_spl.groupby('elicitor_level').mean()

    figure, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    t = probe_mean.columns * 1e3
    for (group, g_df), ax in zip(probe_mean.groupby('group'), axes.flat):
        ax.set_title(f'{group}')
        for level, row in g_df.iterrows():
            ax.plot(t, row, lw=1, label=f'{level[0]} dB SPL')
    for ax in axes[1]:
        ax.set_xlabel('Time (ms)')
    for ax in axes[:, 0]:
        ax.set_ylabel('Signal (V)')
    axes[0, 1].legend(bbox_to_anchor=(1, 1), loc='upper left')
    figure.savefig(manager.get_proc_filename('probe_waveform.pdf'), bbox_inches='tight')

    figure, axes = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)
    for (group, g_df), ax in zip(probe_spl_mean.iloc[:, 1:].groupby('group'), axes.flat):
        ax.set_title(f'{group}')
        for level, row in g_df.iterrows():
            ax.plot(row.index, row, lw=1, label=f'{level[0]} dB SPL')
    for ax in axes[1]:
        ax.set_xlabel('Frequency (kHz)')
    for ax in axes[:, 0]:
        ax.set_ylabel('PSD (dB SPL)')
    axes[0, 1].legend(bbox_to_anchor=(1, 1), loc='upper left')
    axes[0, 0].set_xscale('octave')
    axes[0, 0].axis(xmin=4e3, xmax=32e3)
    figure.savefig(manager.get_proc_filename('probe PSD.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = get_colors(len(memr))
    for c, level, row in zip(colors, memr.iloc[:, 1:].iterrows()):
        ax.plot(row, label=f'{level} dB SPL', color=c)
    ax.set_xscale('octave')
    ax.axis(xmin=4e3, xmax=32e3, ymin=-5, ymax=5)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('MEMR (dB re baseline)')
    figure.savefig(manager.get_proc_filename('MEMR.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(6, 1 * len(elicitor_waveform)))
    waterfall_plot(ax, elicitor_waveform, 'elicitor_level',
                   plotkw={'lw': 0.1, 'color': 'k'})
    figure.savefig(manager.get_proc_filename('elicitor waveform.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(6, 1 * len(elicitor_spl_mean)))
    waterfall_plot(ax, elicitor_spl_mean, 'elicitor_level',
                   plotkw={'lw': 0.1, 'color': 'k'}, scale_method='mean')
    figure.savefig(manager.get_proc_filename('elicitor PSD.pdf'), bbox_inches='tight')
    plt.close('all')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='*')
    parser.add_argument('--reprocess', action='store_true')
    args = parser.parse_args()
    for path in tqdm(args.path):
        try:
            path = Path(path)
            if 'interleaved' in path.stem:
                process_interleaved_file(path, args.reprocess)
            elif 'simultaneous' in path.stem:
                process_simultaneous_file(path, args.reprocess)
        except Exception as e:
            print(f'Error processing {path.name}')
            print(f' {e}')


if __name__ == '__main__':
    main()
