import matplotlib.pyplot as plt

from psiaudio import util
from psi.data.io.api import Recording

from .util import DatasetManager


class IEC(Recording):

    def __init__(self, base_path, setting_table='epoch_metadata'):
        super().__init__(base_path, setting_table)

    def get_epochs(self, columns='auto', offset=0, extra_duration=5e-3):
        signal = getattr(self, 'hw_ai')
        duration = self.get_setting('hw_ao_chirp_duration')
        return signal.get_epochs(self.epoch_metadata, offset, duration+extra_duration)


def summarize_iec(filename, reprocess=False):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed('chirp normalized SPL.pdf'):
        return
    fh = IEC(filename)

    epochs = fh.get_epochs()
    cal = fh.hw_ai.get_calibration()
    freq_start = fh.get_setting('hw_ao_chirp_start_frequency')
    freq_end = fh.get_setting('hw_ao_chirp_end_frequency')

    grouping = epochs.index.names[:-1]
    epochs_mean = epochs.groupby(grouping).mean()
    epochs_psd = util.psd_df(epochs, fs=fh.hw_ai.fs)
    epochs_psd_db_mean = util.db(epochs_psd).groupby(grouping).mean().iloc[:, 1:]
    epochs_spl = cal.get_db(epochs_psd)
    epochs_spl_mean = epochs_spl.groupby(grouping).mean()

    figure, ax = plt.subplots(1, 1, figsize=(11.5, 8))
    for index, waveform in epochs_mean.iterrows():
        ax.plot(waveform.index * 1e3, waveform.values, label=f'{index}')
    ax.set_title('Chirp waveform')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Microphone amplitude (V)')
    ax.legend()
    figure.savefig(manager.get_proc_filename('chirp waveform.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(11.5, 8))
    for index, psd in epochs_psd_db_mean.iterrows():
        ax.plot(psd.index, psd.values, label=f'{index}')
    ax.axis(xmin=freq_start * 0.9, xmax=freq_end / 0.9)
    ax.axvline(freq_start, ls=':', color='k')
    ax.axvline(freq_end, ls=':', color='k')
    ax.set_xscale('octave')
    ax.set_title('Chirp PSD')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Microphone PSD (V)')
    ax.legend()
    figure.savefig(manager.get_proc_filename('chirp PSD.pdf'), bbox_inches='tight')

    figure, ax = plt.subplots(1, 1, figsize=(11.5, 8))
    for index, spl in epochs_spl_mean.iterrows():
        ax.plot(spl.index, spl.values, label=f'{index}')
    ax.axis(xmin=freq_start * 0.9, xmax=freq_end / 0.9)
    ax.axvline(freq_start, ls=':', color='k')
    ax.axvline(freq_end, ls=':', color='k')
    ax.set_xscale('octave')
    ax.set_title('Chirp SPL')
    ax.set_xlabel('Frequency (kHz)')
    ax.set_ylabel('Stim. level (dB SPL)')
    ax.legend()
    figure.savefig(manager.get_proc_filename('chirp SPL.pdf'), bbox_inches='tight')

    #figure.savefig(manager.get_proc_filename('chirp normalized SPL.pdf'), bbox_inches='tight')

    plt.show()

if __name__ == '__main__':
    filename = r'C:\Users\mmm\projects\psi1\data\data\20230123-112200 Sean B008-1 left test inear_speaker_calibration_chirp'
    summarize_iec(filename)

