import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psiaudio import util
from psiaudio.calibration import FlatCalibration
from psiaudio.stim import chirp

from .util import add_default_options, DatasetManager, process_files

from .iec import IEC


def process_file(filename, cb, reprocess=False):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed('psd.csv'):
        return
    manager.clear()
    cb = manager.create_cb(cb)

    cb(0)
    fh = IEC(filename)

    epochs = fh.get_epochs(cb=cb)
    cal = fh.hw_ai.get_calibration()
    freq_start = fh.get_setting('hw_ao_chirp_start_frequency')
    freq_end = fh.get_setting('hw_ao_chirp_end_frequency')
    duration = fh.get_setting('hw_ao_chirp_duration')
    window = fh.get_setting('hw_ao_chirp_window')
    equalize = fh.get_setting('hw_ao_chirp_equalize')
    levels = fh.epoch_metadata['hw_ao_chirp_level'].unique()

    if equalize:
        raise ValueError('Cannot process equalized chirps')

    stim_cal = FlatCalibration.as_attenuation()
    waveforms = {}
    for level in levels:
        w = chirp(fh.hw_ai.fs, freq_start, freq_end, duration, level, stim_cal,
                  window, equalize)
        n_pad = epochs.shape[-1] - w.shape[-1]
        waveforms[level] = np.pad(w, (0, n_pad))

    index = pd.Index(waveforms.keys(), name='hw_ao_chirp_level')
    waveforms = pd.DataFrame(waveforms.values(), index=index,
                             columns=epochs.columns)
    waveforms_psd_db = util.db(util.psd_df(waveforms, fs=fh.hw_ai.fs))

    grouping = epochs.index.names[:-1]
    epochs_mean = epochs.groupby(grouping).mean()
    epochs_psd = util.psd_df(epochs, fs=fh.hw_ai.fs)
    epochs_psd_db_mean = util.db(epochs_psd).groupby(grouping).mean().iloc[:, 1:]
    epochs_spl = cal.get_db(epochs_psd)
    epochs_spl_mean = epochs_spl.groupby(grouping).mean()

    figure, axes = plt.subplots(2, 3, figsize=(15, 8))

    ax = axes[0, 0]
    for index, waveform in waveforms.iterrows():
        ax.plot(waveform.index * 1e3, waveform.values, label=f'{index}')
    ax.set_title('Speaker')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend()

    ax = axes[0, 1]
    for index, waveform in epochs_mean.iterrows():
        ax.plot(waveform.index * 1e3, waveform.values, label=f'{index}')
    ax.set_title('Microphone')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Amplitude (V)')
    ax.legend()

    padding = 0.5

    ax = axes[1, 1]
    for index, psd in epochs_psd_db_mean.iterrows():
        ax.plot(psd.index, psd.values, label=f'{index}')
    ax.axis(xmin=freq_start * padding, xmax=freq_end / padding)
    ax.axvline(freq_start, ls=':', color='k')
    ax.axvline(freq_end, ls=':', color='k')
    ax.set_xscale('octave')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('PSD (V)')
    ax.legend()
    ax.grid()

    ax = axes[1, 0]
    for index, psd in waveforms_psd_db.iterrows():
        ax.plot(psd.index, psd.values, label=f'{index}')
    ax.axis(xmin=freq_start * padding, xmax=freq_end / padding, ymin=-100)
    ax.axvline(freq_start, ls=':', color='k')
    ax.axvline(freq_end, ls=':', color='k')
    ax.set_xscale('octave')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('PSD (V)')
    ax.legend()
    ax.grid()

    ax = axes[0, 2]
    for index, spl in epochs_spl_mean.iterrows():
        ax.plot(spl.index, spl.values, label=f'{index}')
    ax.axis(xmin=freq_start * padding, xmax=freq_end / padding)
    ax.axvline(freq_start, ls=':', color='k')
    ax.axvline(freq_end, ls=':', color='k')
    ax.set_xscale('octave')
    ax.set_title('Calibration')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Stim. level (dB SPL)')
    ax.legend()
    ax.grid()

    norm_spl = epochs_spl_mean - waveforms_psd_db
    ax = axes[1, 2]
    for index, spl in norm_spl.iterrows():
        ax.plot(spl.index, spl.values, label=f'{index}')
    ax.axis(xmin=freq_start * padding, xmax=freq_end / padding, ymin=80, ymax=160)
    ax.axvline(freq_start, ls=':', color='k')
    ax.axvline(freq_end, ls=':', color='k')
    ax.set_xscale('octave')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Normalized stim. level (dB SPL @ 1Vrms)')
    ax.legend()
    ax.grid()

    figure.tight_layout()
    manager.save_fig(figure, 'calibration.pdf')

    waveforms = pd.DataFrame({
        'sig_out': waveforms.stack(),
        'mic_in': epochs_mean.stack()
    })
    manager.save_dataframe(waveforms, 'waveforms.csv')

    psd = pd.DataFrame({
        'sig_out': waveforms_psd_db.stack(),
        'mic_in': epochs_psd_db_mean.stack(),
        'mic_spl': epochs_spl_mean.stack(),
        'norm_spl': norm_spl.stack(),
    })
    manager.save_dataframe(psd, 'psd.csv')


def main_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize IEC data in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*inear_speaker_calibration_chirp*',
                  process_file, reprocess=args.reprocess)
