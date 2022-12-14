from pathlib import Path

import matplotlib.pyplot as plt

from psiaudio.plot import waterfall_plot
from psiaudio import util

from .memr import MEMRFile


def process_file(filename):
    fh = MEMRFile(filename)
    output_dir = filename.parent / filename.stem
    output_dir.mkdir(exist_ok=True)
    filename_template = f'{filename.stem} {{}}.pdf'

    # Load variables we need from the file
    cal = fh.microphone.get_calibration()
    period = fh.get_setting('repeat_period')
    probe_delay = fh.get_setting('probe_chirp_delay')
    probe_duration = fh.get_setting('probe_chirp_duration')
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
    figure.savefig(output_dir / filename_template.format('stimulus train'))

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
    figure.savefig(output_dir / filename_template.format('elicitor PSD'))

    probe = repeats.loc[:, probe_delay:probe_delay+probe_duration*1.5]
    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(probe.columns.values * 1e3, probe.values.T, alpha=0.1, color='k');
    ax.set_xlabel('Time (msec)')
    ax.set_ylabel('Signal (V)')
    figure.savefig(output_dir / filename_template.format('probe waveform'))

    probe_psd = util.psd_df(probe, fh.microphone.fs)
    probe_spl = cal.get_db(probe_psd)
    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.plot(probe_spl.columns, probe_spl.values.T, alpha=0.1, color='k');
    ax.set_xscale('octave')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Level (dB SPL)')
    ax.axvline(probe_fl)
    ax.axvline(probe_fh)
    figure.savefig(output_dir / filename_template.format('probe PSD'))

    memr = probe_spl - probe_spl.xs(0, level='repeat')
    memr_mean = memr.groupby(['repeat', 'elicitor_level']).mean()

    figure, ax = plt.subplots(1, 1, figsize=(8, 4))
    memr_mean_end = memr_mean.loc[elicitor_n]
    for level, value in memr_mean_end.iterrows():
        ax.plot(value, label=f'{level} dB SPL')
    ax.legend(bbox_to_anchor=(1, 1), loc='upper left')
    ax.set_xscale('octave')
    figure.savefig(output_dir / filename_template.format('MEMR'))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('path', nargs='*')
    args = parser.parse_args()
    for path in args.path:
        try:
            process_file(Path(path))
        except Exception as e:
            print(e)


if __name__ == '__main__':
    main()

