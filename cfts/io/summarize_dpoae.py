import logging
log = logging.getLogger(__name__)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from psiaudio import util

from .dpoae import DPOAEFile
from .util import add_default_options, DatasetManager, process_files


def isodp_th(l2, dp, nf, criterion):
    '''
    Computes iso-DP threshold for a level sweep at a single frequency

    Parameters
    ----------
    l2 : array-like
        Requested F2 levels
    dp : array-like
        Measured DPOAE levels
    nf : array-like
        Measured DPOAE noise floor
    criterion : float
        Threshold criterion (e.g., value that the input-output function must
        exceed)

    Returns
    -------
    threshold : float
        If no threshold is identified, NaN is returned
    '''
    # First, discard up to the first level where the DPOAE exceeds one standard
    # deviation from the noisne floor
    nf_crit = np.mean(nf) + np.std(nf)
    i = np.flatnonzero(dp < nf_crit)
    if len(i):
        dp = dp[i[-1]:]
        l2 = l2[i[-1]:]

    # Now, loop through every pair of points until we find the first pair that
    # brackets criterion (for non-Python programmers, this uses chained
    # comparision operators and is not a bug)
    for l_lb, l_ub, d_lb, d_ub in zip(l2[:-1], l2[1:], dp[:-1], dp[1:]):
        if d_lb < criterion <= d_ub:
            return np.interp(criterion, [d_lb, d_ub], [l_lb, l_ub])
    return np.nan


def isodp_th_criterions(df, criterions=None, debug=False):
    '''
    Helper function that takes dataframe containing a single frequency and
    calculates threshold for each criterion.
    '''
    if criterions is None:
        criterions = [-5, 0, 5, 10, 15, 20, 25]

    if ':dB' in df.columns:
        # This is used for thresholding data already in EPL CFTS format
        l2 = df.loc[:, ':dB']
        dp = df.loc[:, '2f1-f2(dB)']
        nf = df.loc[:, '2f1-f2Nse(dB)']
    else:
        # This is used for thresholding data from the psi DPOAE IO.
        if debug:
            # Use a measurable signal to estimate threshold.
            l2 = df.loc[:, 'secondary_tone_level'].values
            dp = df.loc[:, 'f2_level'].values
            nf = df.loc[:, 'dpoae_noise_floor'].values
        else:
            l2 = df.loc[:, 'secondary_tone_level'].values
            dp = df.loc[:, 'dpoae_level'].values
            nf = df.loc[:, 'dpoae_noise_floor'].values

    th = [isodp_th(l2, dp, nf, c) for c in criterions]
    index = pd.Index(criterions, name='criterion')
    return pd.Series(th, index=index, name='threshold')


def process_file(filename, cb, reprocess=False):
    manager = DatasetManager(filename)
    if not reprocess and manager.is_processed('io.csv'):
        return
    manager.clear()
    cb = manager.create_cb(cb)

    cb(0)
    fh = DPOAEFile(filename)
    fs = fh.system_microphone.fs
    ramp_time = fh.get_setting('primary_tone_rise_time')
    n_time = fh.get_setting('n_time')
    n_fft = fh.get_setting('n_fft')
    window = fh.get_setting('response_window')
    f2_f1_ratio = fh.get_setting('f2_f1_ratio')

    n_window = window * fs
    n_trim = (ramp_time * 4) * fs
    if int(n_window) != n_window:
        raise ValueError('n_window is not an integer')
    if int(n_trim) != n_trim:
        raise ValueError('n_trim is not an integer')
    n_trim = int(n_trim)
    n_window = int(n_window)
    resolution = fs / n_window

    step = 25e-3
    n_step = int(step * fs)

    cal = fh.system_microphone.get_calibration()

    psd = {}
    measured = {}
    f2_prev = None
    for i, row in fh.results.iterrows():
        cb(i / len(fh.results))
        lb = row['dp_start']
        ub = row['dp_end']
        f2 = row['f2_frequency']
        f1 = row['f1_frequency']
        l2 = row['f2_level']
        dp = 2 * f1 - f2
        nf_freq = np.array([-2, -1, 1, 2]) * resolution + dp

        s = fh.system_microphone.get_segment(lb, 0, ub-lb, allow_partial=True)
        s = s.values[n_trim:]

        m_set = []
        p_set = []
        for i in range(1):
            n_segments, n_left = divmod(s.shape[-1], n_window)
            s_segmented = s[:-n_left].reshape((n_segments, -1))
            m = np.isfinite(s_segmented).all(axis=1)
            s_segmented = s_segmented[m]
            p = cal.get_db(util.psd_df(s_segmented.mean(axis=0), fs))
            s = s[n_step:]

            p_set.append(p)
            m_set.append({
                'f1_level': p[f1],
                'f2_level': p[f2],
                'dp_level': p[dp],
                'dp_nf': p[nf_freq].mean(),
                'online_dp_level': row['meas_dpoae_level'],
                'online_dp_nf': row['dpoae_noise_floor'],
            })
        measured[f2, l2] = pd.DataFrame(m_set).mean(axis=0)
        psd[f2, l2] = pd.DataFrame(p_set).mean(axis=0)

    freq = fh.results['f2_frequency'].unique()
    level = fh.results['f2_level'].unique()
    n_freq = len(freq)
    n_level = len(level)

    measured = pd.DataFrame(
        measured.values(),
        index=pd.MultiIndex.from_tuples(measured.keys(), names=['f2', 'l2'])
    )

    figure, axes = plt.subplots(2, n_freq, figsize=(4 * n_freq, 8), sharex=True, sharey=True)
    for fi, f2 in enumerate(freq):
        col = axes[:, fi]
        m = measured.loc[f2]

        ax = col[0]
        ax.axhline(0, ls='-', color='k')
        ax.plot(m['f2_level'], marker='o', color='0.5')
        ax.plot(m['f1_level'], marker='o', color='k')
        ax.plot(m['dp_level'], marker='o', color='darkturquoise')
        ax.plot(m['dp_nf'], marker='x', color='lightblue')
        ax.set_title(f'{f2} Hz')
        ax.grid()

        ax = col[1]
        ax.axhline(0, ls='-', color='k')
        ax.plot(m['f2_level'], marker='o', color='0.5')
        ax.plot(m['f1_level'], marker='o', color='k')
        ax.plot(m['online_dp_level'], marker='o', color='darkorange')
        ax.plot(m['online_dp_nf'], marker='x', color='coral')
        ax.grid()
        ax.set_xlabel('F2 level (dB SPL)')

    for ax in axes[:, 0]:
        ax.set_ylabel('Measured level (dB SPL)')

    manager.save_fig(figure, 'io.pdf')

    figure, axes = plt.subplots(n_level, n_freq, figsize=(4 * n_freq, 4 * n_level), sharex=True, sharey=True)
    for fi, f2 in enumerate(freq):
        for li, l2 in enumerate(level[::-1]):
            ax = axes[li, fi]
            f1 = f2 / f2_f1_ratio
            dp = 2 * f1 - f2
            ax.axvline(f2, lw=2, color='lightblue')
            ax.axvline(f1, lw=2, color='lightblue')
            ax.axvline(dp, lw=2, color='darkturquoise')
            ax.axhline(0, lw=2, color='0.5')
            ax.plot(psd[f2, l2].iloc[1:], color='k')

    min_freq = min(2 * (freq / f2_f1_ratio) - freq)
    max_freq = max(freq)
    axes[0, 0].axis(xmin=min_freq * 0.8, xmax=max_freq / 0.8)
    axes[0, 0].set_xscale('octave')

    for ax in axes[-1]:
        ax.set_xlabel('Frequency (kHz)')
    for ax in axes[:, 0]:
        ax.set_xlabel('PSD (dB)')

    manager.save_fig(figure, 'mic spectrum.pdf')

    manager.save_dataframe(measured, 'io.csv')


def main_folder():
    import argparse
    parser = argparse.ArgumentParser('Summarize DPOAE IO data in folder')
    add_default_options(parser)
    args = parser.parse_args()
    process_files(args.folder, '**/*dpoae_io*',
                  process_file, reprocess=args.reprocess)


if __name__ == '__main__':
    cb = lambda x: x
    process_file('c:/users/mmm/projects/psi1/data/data/20230127-122543 Sean Sean left 151122 1 dpoae_io.zip', cb)
