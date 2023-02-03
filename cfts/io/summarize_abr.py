import logging
log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

import argparse
import datetime as dt
from functools import partial
from math import ceil
import json
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from psi import get_config
from psiaudio.plot import waterfall_plot
from cfts.io import abr

from .util import add_default_options, DatasetManager, process_files


COLUMNS = ['frequency', 'level', 'polarity']
DATA_ROOT = get_config('DATA_ROOT')


def get_file_template(filename, offset, duration, filter_settings, n_epochs,
                      prefix='ABR', simple_filename=True,
                      include_filename=True):

    if include_filename:
        prefix = f'{filename.stem} {prefix}'
    if simple_filename:
        return f'{prefix}'

    base_string = f'{prefix}{offset*1e3:.1f}ms to {(offset+duration)*1e3:.1f}ms'

    if n_epochs:
        base_string = f'{base_string} {n_epochs} averages'

    fh = abr.load(filename)
    if filter_settings == 'saved':
        settings = _get_filter(fh)
        if not settings['digital_filter']:
            filter_string = 'no filter'
        else:
            lb = settings['lb']
            ub = settings['ub']
            filter_string = f'{lb:.0f}Hz to {ub:.0f}Hz filter'
    elif filter_settings is None:
        filter_string = 'no filter'
    else:
        lb = filter_settings['lb']
        ub = filter_settings['ub']
        filter_string = f'{lb:.0f}Hz to {ub:.0f}Hz filter'
        order = filter_settings['order']
        if order != 1:
            filter_string = f'{order:.0f} order {filter_string}'

    file_string = f'{base_string} with {filter_string}'
    if suffix is not None:
        file_string = f'{file_string} {suffix}'

    return f'{file_string}'


def _get_filter(fh):
    if not isinstance(fh, (abr.ABRFile, abr.ABRSupersetFile)):
        fh = abr.load(fh)
    return {
        'digital_filter': fh.get_setting_default('digital_filter', True),
        'lb': fh.get_setting_default('digital_highpass', 300),
        'ub': fh.get_setting_default('digital_lowpass', 3000),
        # Filter order is not currently an option in the psiexperiment ABR
        # program so it defaults to 1.
        'order': 1,
    }


def _get_epochs(fh, offset, duration, filter_settings, reject_ratio=None,
                downsample=None, cb=None):
    # We need to do the rejects in this code so that we can obtain the
    # information for generating the CSV files. Set reject_threshold to np.inf
    # to ensure that nothing gets rejected.
    kwargs = {'offset': offset, 'duration': duration, 'columns': COLUMNS,
              'reject_threshold': np.inf, 'downsample': downsample, 'cb': cb,
              'bypass_cache': True}

    if filter_settings is None:
        return fh.get_epochs(**kwargs)

    if filter_settings == 'saved':
        settings = _get_filter(fh)
        if not settings['digital_filter']:
            return fh.get_epochs(**kwargs)
        lb = settings['lb']
        ub = settings['ub']
        order = settings['order']
        kwargs.update({'filter_lb': lb, 'filter_ub': ub, 'filter_order': order})
        return fh.get_epochs_filtered(**kwargs)

    lb = filter_settings['lb']
    ub = filter_settings['ub']
    order = filter_settings['order']
    kwargs.update({'filter_lb': lb, 'filter_ub': ub, 'filter_order': order})
    return fh.get_epochs_filtered(**kwargs)


def add_trial(epochs):
    '''
    This adds trial number on a per-stim-level/frequency basis
    '''
    def number_trials(subset):
        subset = subset.sort_index(level='t0')
        idx = subset.index.to_frame()
        i = len(idx.columns) - 1
        idx.insert(i, 'trial', np.arange(len(idx)))
        subset.index = pd.MultiIndex.from_frame(idx)
        return subset

    levels = list(epochs.index.names[:-1])
    if 'polarity' in levels:
        levels.remove('polarity')

    return epochs.groupby(levels, group_keys=False).apply(number_trials)


def plot_waveforms_cb(epochs_mean, filename, name):
    epochs_mean = epochs_mean.reset_index(['epoch_n', 'epoch_reject_ratio'], drop=True)
    grouped = epochs_mean.groupby('frequency')
    n_panels = len(grouped)
    figure, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 8.5))
    for ax, (frequency, data) in zip(axes, grouped):
        waterfall_plot(ax, data)
        ax.set_xlabel('Time (msec)')
        ax.set_title(f'{frequency * 1e-3:0.2f} Hz')
    figure.suptitle(name)
    figure.savefig(filename)


def process_file(filename, offset=-1e-3, duration=10e-3,
                 filter_settings='saved', n_epochs='auto',
                 simple_filename=True, export_single_trial=False, cb=None,
                 file_template=None, target_fs=12.5e3, analysis_window=None,
                 latency_correction=0, gain_correction=1, debug_mode=False,
                 plot_waveforms_cb=plot_waveforms_cb, reprocess=False):
    '''
    Extract ABR epochs, filter and save result to CSV files

    Parameters
    ----------
    filename : path
        Path to ABR experiment. If it's a set of ABR experiments, epochs across
        all experiments will be combined for the analysis.
    offset : sec
        The start of the epoch to extract, in seconds, relative to tone pip
        onset. Negative values can be used to extract a prestimulus baseline.
    duration: sec
        The duration of the epoch to extract, in seconds, relative to the
        offset. If offset is set to -0.001 sec and duration is set to 0.01 sec,
        then the epoch will be extracted from -0.001 to 0.009 sec re tone pip
        onset.
    filter_settings : {None, 'saved', dict}
        If None, no additional filtering is done. If 'saved', uses the digital
        filter settings that were saved in the ABR file. If a dictionary, must
        contain 'lb' (the lower bound of the passband in Hz) and 'ub' (the
        upper bound of the passband in Hz).
    n_epochs : {None, 'auto', int, dict}
        If None, all epochs will be used. If 'auto', use the value defined at
        acquisition time. If integer, will limit the number of epochs per
        frequency and level to this number.
    simple_filename : bool
        If True, do not embed settings used for processing data in filename.
    export_single_trial : bool
        If True, export single trials.
    cb : {None, 'tqdm', callable}
        If a callable, takes one value (the estimate of percent done as a
        fraction). If 'tqdm', progress will be printed to the console.
    file_template : {None, str}
        Template that will be used to determine names (and path) of processed
        files.
    target_fs : float
        Closest sampling rate to target
    analysis_window : Ignored
        This is ignored for now. Primarily used to allow acceptance of the
        queue since we add analysis window for GUI purposes.
    latency_correction : float
        Correction, in seconds, to apply to timing of ABR. This allows us to
        retroactively correct for any ADC or DAC delays that were present in
        the acquisition system.
    gain_correction : float
        Correction to apply to the scaling of the waveform. This allows us to
        retroactively correct for differences in gain that were present in the
        acquisition system.
    debug_mode : bool
        This is reserved for internal use only. This mode will load the epochs
        and return them without saving to disk.
    plot_waveforms_cb : {Callable, None}
        Callback that takes three arguments. Epoch mean dataframe, path to file
        to save figures in, and name of file.
    '''
    settings = locals()
    filename = Path(filename)

    # Cleanup settings so that it is JSON-serializable
    settings.pop('cb')
    settings.pop('plot_waveforms_cb')
    settings['filename'] = str(settings['filename'])
    settings['creation_time'] = dt.datetime.now().isoformat()

    fh = abr.load(filename)
    if len(fh.erp_metadata) == 0:
        raise IOError('No data in file')

    # This is a hack to ensure that native Python types are returned instead of
    # Numpy ones. Newer versions of Pandas have fixed this issue, though.
    md = fh.erp_metadata.iloc[:1].to_dict('records')[0]
    for column in COLUMNS:
        del md[column]
    del md['t0']

    downsample = int(ceil(fh.eeg.fs / target_fs))
    settings['downsample'] = downsample
    settings['actual_fs'] = fh.eeg.fs / downsample

    if n_epochs is not None:
        if n_epochs == 'auto':
            n_epochs = fh.get_setting('averages')

    if file_template is None:
        file_template = get_file_template(
            filename, offset, duration, filter_settings, n_epochs,
            simple_filename=simple_filename, include_filename=True)

    manager = DatasetManager(filename, file_template=file_template)
    files = [
        'average waveforms.csv',
        'processing settings.json',
        'experiment settings.json',
        'waveforms.pdf'
    ]
    if export_single_trial:
        files.append('individual waveforms.csv')

    if not reprocess and manager.is_processed(files):
        return

    # Load the epochs. The callbacks for loading the epochs return a value in
    # the range 0 ... 1. Since this only represents "half" the total work we
    # need to do, rescale to the range 0 ... 0.5.
    cb = manager.create_cb(cb)
    cb(0)

    def cb_rescale(frac):
        nonlocal cb
        cb(frac * 0.5)

    epochs = _get_epochs(fh, offset + latency_correction, duration,
                         filter_settings, cb=cb_rescale, downsample=downsample)

    if gain_correction != 1:
        epochs = epochs * gain_correction

    if latency_correction != 0:
        new_idx = [(*r[:-1], r[-1] - latency_correction) for r in epochs.index]
        new_idx = pd.MultiIndex.from_tuples(new_idx, names=epochs.index.names)
        new_col = epochs.columns - latency_correction
        epochs = pd.DataFrame(epochs.values, index=new_idx, columns=new_col)

    if debug_mode:
        return epochs

    # Apply the reject
    reject_threshold = fh.get_setting('reject_threshold')
    m = np.abs(epochs) < reject_threshold
    m = m.all(axis=1)
    epochs = epochs.loc[m]

    cb(0.6)
    if n_epochs is not None:
        n = int(np.floor(n_epochs / 2))
        epochs = epochs.groupby(COLUMNS, group_keys=False) \
            .apply(lambda x: x.iloc[:n])
    cb(0.7)

    epoch_mean = epochs.groupby(COLUMNS).mean().groupby(COLUMNS[:-1]).mean()

    epoch_reject_ratio = 1-m.groupby(COLUMNS[:-1]).mean()
    epoch_n = epochs.groupby(COLUMNS[:-1]).size()
    epoch_info = pd.DataFrame({
        'epoch_n': epoch_n,
        'epoch_reject_ratio': epoch_reject_ratio,
    })
    if not np.all(epoch_mean.index == epoch_info.index):
        raise ValueError('Programming issue. Please contact developer.')

    # Merge in the N and reject ratio into the index for epoch_mean
    epoch_info = epoch_info.set_index(['epoch_n', 'epoch_reject_ratio'],
                                      append=True)
    epoch_mean.index = epoch_info.index
    epoch_mean.columns.name = 'time'

    manager.get_proc_filename('processing settings.json') \
        .write_text(json.dumps(settings, indent=2))
    manager.get_proc_filename('experiment settings.json') \
        .write_text(json.dumps(md, indent=2))

    epoch_mean.T.to_csv(manager.get_proc_filename('average waveforms.csv'))
    cb(0.8)

    if export_single_trial:
        epochs = add_trial(epochs)
        epochs.columns.name = 'time'
        epochs.T.to_csv(manager.get_proc_filename('individual waveforms.csv'))

    cb(0.9)
    if plot_waveforms_cb is not None:
        plot_waveforms_cb(
            epoch_mean,
            manager.get_proc_filename('waveforms.pdf'),
            filename.stem
        )

    cb(1.0)
    return True


def main_file():
    parser = argparse.ArgumentParser('Filter and summarize ABR data')

    parser.add_argument('filenames', type=str,
                        help='Filename', nargs='+')
    parser.add_argument('--offset', type=float,
                        help='Epoch offset',
                        default=-0.001)
    parser.add_argument('--duration', type=float,
                        help='Epoch duration',
                        default=0.01)
    parser.add_argument('--filter-lb', type=float,
                        help='Highpass filter cutoff',
                        default=None)
    parser.add_argument('--filter-ub', type=float,
                        help='Lowpass filter cutoff',
                        default=None)
    parser.add_argument('--order', type=float,
                        help='Filter order',
                        default=None)
    parser.add_argument('--reprocess',
                        help='Redo existing results',
                        action='store_true')
    args = parser.parse_args()

    if args.filter_lb is not None or args.filter_ub is not None:
        filter_settings = {
            'lb': args.filter_lb,
            'ub': args.filter_ub,
            'order': args.order,
        }
    else:
        filter_settings = None
    process_files(
        filenames=args.filenames,
        offset=args.offset,
        duration=args.duration,
        filter_settings=filter_settings,
        reprocess=args.reprocess,
        cb='tqdm',
    )


def main_folder():
    parser = argparse.ArgumentParser('Filter and summarize ABR files in folder')
    add_default_options(parser)
    args = parser.parse_args()
    fn = partial(process_file, filter_settings='saved', offset=-0.001,
                 duration=0.01)
    process_files(args.folder, '**/*abr_io*', fn, reprocess=args.reprocess)


def main_gui():
    import enaml
    from enaml.qt.qt_application import QtApplication
    with enaml.imports():
        from .summarize_abr_gui import SummarizeABRGui
    app = QtApplication()
    view = SummarizeABRGui()
    view.show()
    app.start()
