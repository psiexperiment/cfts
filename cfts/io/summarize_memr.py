from pathlib import Path

import matplotlib.pyplot as plt
from tqdm import tqdm

from psiaudio.plot import waterfall_plot
from psiaudio import util

from .memr import InterleavedMEMRFile, SimultaneousMEMRFile
from .util import DatasetManager


def process_interleaved_file(filename):
    manager = DatasetManager(filename)
    if manager.is_processed(['MEMR.pdf']):
        return
    fh = InterleavedMEMRFile(filename)

    # Load variables we need from the file
    cal = fh.microphone.get_calibration()
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

    figsize = 6, 1 * len(epochs_mean)
    figure, ax = plt.subplots(1, 1, figsize=figsize)
    waterfall_plot(ax, epochs_mean, 'elicitor_level', scale_method='max',
                   plotkw={'lw': 0.1, 'color': 'k'},
                   x_transform=lambda x: x*1e3)
    ax.set_xlabel('Time (msec)')
    ax.grid(False)
    # Draw lines showing the repeat boundaries
    for i in range(elicitor_n + 2):
        ax.axvline(i * period * 1e3, zorder=-1, alpha=0.5)
    # Save the figure
    figure.savefig(manager.get_proc_filename('stimulus train.pdf'))

    # Now, load the repeats. This essentially segments the epochs DataFrame
    # into the individual repeat segments.
    repeats = fh.get_repeats()

    elicitor = repeats.loc[:, elicitor_delay:]
    elicitor_psd = util.psd_df(elicitor, fs=fh.microphone.fs)
    elicitor_spl = cal.get_db(elicitor_psd)
    # Be sure to throw out the last "repeat" (which has a silent period after
    # it rather than another elicitor).
    elicitor_psd_mean = elicitor_psd.query('repeat < @elicitor_n').groupby('elicitor_level').mean()
    elicitor_spl_mean = cal.get_db(elicitor_psd_mean)

    # Plot the elicitor for each level as a waterfall plot
    figure, ax = plt.subplots(1, 1, figsize=figsize)
    waterfall_plot(ax, elicitor_spl_mean.dropna(axis=1), 'elicitor_level', scale_method='mean', plotkw={'lw': 0.1, 'color': 'k'})
    ax.set_xscale('octave')
    ax.axis(xmin=0.5e3, xmax=50e3)
    ax.set_xlabel('Frequency (kHz)')
    figure.savefig(manager.get_proc_filename('elicitor PSD.pdf'))

    probe = repeats.loc[:, probe_delay:probe_delay+probe_duration*1.5]
    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(probe.columns.values * 1e3, probe.values.T, alpha=0.1, color='k');
    ax.set_xlabel('Time (msec)')
    ax.set_ylabel('Signal (V)')
    figure.savefig(manager.get_proc_filename('probe waveform.pdf'))

    probe_psd = util.psd_df(probe, fh.microphone.fs)
    probe_spl = cal.get_db(probe_psd)
    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(probe_spl.columns, probe_spl.values.T, alpha=0.1, color='k');
    ax.set_xscale('octave')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Level (dB SPL)')
    ax.axvline(probe_fl)
    ax.axvline(probe_fh)
    figure.savefig(manager.get_proc_filename('probe PSD.pdf'))

    memr = probe_spl - probe_spl.xs(0, level='repeat')
    memr_mean = memr.groupby(['repeat', 'elicitor_level']).mean()

    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    memr_mean_end = memr_mean.loc[elicitor_n]
    for level, value in memr_mean_end.iterrows():
        ax.plot(value, label=f'{level} dB SPL')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xscale('octave')
    figure.savefig(manager.get_proc_filename('MEMR.pdf'))


def process_simultaneous_file(filename):
    manager = DatasetManager(filename)
    fh = SimultaneousMEMRFile(filename)

    cal = fh.microphone.get_calibration()
    repeats = fh.get_repeats()
    probe_window = fh.get_setting('probe_duration') + 1.5e-3
    probes = repeats.loc[:, :probe_window]
    probe_mean = probes.groupby(['elicitor_level', 'group']).mean()
    probe_spl = cal.get_db(util.psd_df(probe_mean, fs=fh.microphone.fs))
    probe_spl_mean = probe_spl.groupby(['elicitor_level', 'group']).mean()
    baseline = probe_spl_mean.xs('baseline', level='group')
    elicitor = probe_spl_mean.xs('elicitor_ss', level='group')
    memr = elicitor - baseline

    epochs = fh.get_epochs()
    onset = fh.get_setting('elicitor_onset')
    duration = fh.get_setting('elicitor_duration')
    elicitor = epochs.loc[:, onset:onset+duration]
    elicitor_waveform = elicitor.loc[1].groupby(['elicitor_level']).mean()
    elicitor_spl = cal.get_db(util.psd_df(elicitor, fs=fh.microphone.fs)).dropna(axis='columns')
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
    figure.savefig(manager.get_proc_filename('probe_waveform.pdf'))

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
    figure.savefig(manager.get_proc_filename('probe PSD.pdf'))

    figure, ax = plt.subplots(1, 1, figsize=(6, 6))
    for level, row in memr.iloc[:, 1:].iterrows():
        ax.plot(row, label=f'{level} dB SPL')
    ax.set_xscale('octave')
    ax.axis(xmin=4e3, xmax=32e3, ymin=-5, ymax=5)
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('MEMR (dB re baseline)')
    figure.savefig(manager.get_proc_filename('MEMR.pdf'))

    figure, ax = plt.subplots(1, 1, figsize=(6, 1 * len(elicitor_waveform)))
    waterfall_plot(ax, elicitor_waveform, 'elicitor_level',
                   plotkw={'lw': 0.1, 'color': 'k'})
    figure.savefig(manager.get_proc_filename('elicitor waveform.pdf'))

    figure, ax = plt.subplots(1, 1, figsize=(6, 1 * len(elicitor_spl_mean)))
    waterfall_plot(ax, elicitor_spl_mean, 'elicitor_level',
                   plotkw={'lw': 0.1, 'color': 'k'}, scale_method='mean')
    figure.savefig(manager.get_proc_filename('elicitor PSD.pdf'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='*')
    args = parser.parse_args()
    for path in tqdm(args.path):
        try:
            path = Path(path)
            if 'interleaved' in path.stem:
                process_interleaved_file(path)
            elif 'simultaneous' in path.stem:
                process_simultaneous_file(path)
        except Exception as e:
            print(path)


if __name__ == '__main__':
    main()
