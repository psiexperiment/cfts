'''
Analysis for the binaural crosstalk check: how much of the sound played into
one ear reaches the microphone in the other ear.

Run before a binaural experiment (e.g., the BIC): a chirp plays in one ear at
a time, both ear canals are recorded, and the interaural attenuation is the
level in the stimulated ear minus the level in the other ear, by frequency.
Interaural attenuation sets the largest interaural level difference (ILD) the
setup can present, and how far a "monaural" stimulus is really monaural;
Laumen et al. 2016 (doi:10.1016/j.heares.2016.04.009) required at least
30 dB for the BIC.

The noise floor of each average is estimated from the difference of the
averages of the even and odd presentations, which cancels the stimulus but
keeps the noise. Where the other ear's level is not clearly above its noise
floor, the true attenuation is larger than measured, so the measurement is
only a lower bound there.

Pure Python (no Qt or enaml), so it can be tested without hardware.
'''
import numpy as np
import pandas as pd

from psiaudio import util


EARS = ('left', 'right')

#: Context name of the ear a chirp plays in.
EAR = 'crosstalk_ear'

#: The other ear's level must be this far (dB) above its noise floor for the
#: attenuation to be a measurement rather than a lower bound.
MIN_SNR = 6


def other(ear):
    return EARS[1 - EARS.index(ear)]


class CrosstalkAnalyzer:
    '''
    Accumulates each microphone's epochs for each stimulated ear and computes
    the spectra and interaural attenuation.

    Parameters
    ----------
    fs : float
        Sampling rate of the microphones.
    n : int
        Samples per epoch (longer epochs are truncated).
    calibrations : dict
        Calibration of each ear's microphone (`left`, `right`), to convert
        volts to dB SPL.
    '''
    def __init__(self, fs, n, calibrations):
        self.fs = fs
        self.n = n
        self.calibrations = calibrations
        self.frequency = np.fft.rfftfreq(n, 1 / fs)
        # Sum of the even and odd epochs for each (stimulated ear,
        # microphone), and how many of each.
        self._sums = {}
        self._counts = {}

    def add(self, ear, mic, epoch):
        '''
        Add one epoch of `mic`'s recording with the chirp in `ear`.
        '''
        key = ear, mic
        sums = self._sums.setdefault(key, np.zeros((2, self.n)))
        counts = self._counts.setdefault(key, [0, 0])
        i = (counts[0] + counts[1]) % 2
        sums[i] += np.asarray(epoch)[:self.n]
        counts[i] += 1

    def count(self, ear, mic):
        return sum(self._counts.get((ear, mic), [0, 0]))

    def spectrum(self, ear, mic):
        '''
        Level (dB SPL) in each frequency bin of the average of `mic`'s epochs
        with the chirp in `ear`, and of its noise floor. None if there are
        not yet two epochs.
        '''
        counts = self._counts.get((ear, mic))
        if counts is None or min(counts) == 0:
            return None
        sums = self._sums[ear, mic]
        even, odd = sums[0] / counts[0], sums[1] / counts[1]
        n_total = counts[0] + counts[1]
        average = (sums[0] + sums[1]) / n_total
        # The half-difference has the noise of an average of
        # 4 / (1 / n_even + 1 / n_odd) epochs; scale it to the average's
        # (the same when n_even == n_odd).
        noise = (even - odd) / 2 \
            * np.sqrt(4 / (n_total * (1 / counts[0] + 1 / counts[1])))
        cal = self.calibrations[mic]
        return (self._db(cal, average), self._db(cal, noise))

    def _db(self, cal, x):
        # RMS of the sinusoid in each bin.
        rms = np.abs(np.fft.rfft(x)) / self.n * np.sqrt(2)
        with np.errstate(divide='ignore'):
            return cal.get_db(self.frequency, rms)

    def bands(self, flb, fub, fraction=3):
        '''
        Interaural attenuation in fractional-octave bands (`fraction` per
        octave) from `flb` to `fub`, for each stimulated ear. The levels of
        each band are the power summed over its bins.

        Returns a DataFrame with one row per stimulated ear and band:
        `ear`, `frequency` (band centre), `ipsi_level`, `contra_level`,
        `contra_noise_floor` (dB SPL), `attenuation` (dB) and `lower_bound`
        (True if the other ear's level is within `MIN_SNR` of its noise
        floor, so the attenuation is at least this).
        '''
        centres = util.octave_space(flb / 1e3, fub / 1e3, 1 / fraction) * 1e3
        edges = np.c_[centres * 2 ** (-0.5 / fraction),
                      centres * 2 ** (0.5 / fraction)]
        rows = []
        for ear in EARS:
            ipsi = self.spectrum(ear, ear)
            contra = self.spectrum(ear, other(ear))
            if ipsi is None or contra is None:
                continue
            for centre, (lb, ub) in zip(centres, edges, strict=True):
                mask = (self.frequency >= lb) & (self.frequency < ub)
                if not mask.any():
                    continue
                i_level = band_level(ipsi[0][mask])
                c_level = band_level(contra[0][mask])
                c_noise = band_level(contra[1][mask])
                rows.append({
                    'ear': ear,
                    'frequency': centre,
                    'ipsi_level': i_level,
                    'contra_level': c_level,
                    'contra_noise_floor': c_noise,
                    'attenuation': i_level - c_level,
                    # Also if both are -inf (digital silence).
                    'lower_bound': not (c_level - c_noise >= MIN_SNR),
                })
        return pd.DataFrame(rows)


def band_level(db):
    '''
    Level (dB) of the power summed over bins at levels `db`.
    '''
    with np.errstate(divide='ignore'):
        return 10 * np.log10(np.sum(10 ** (np.asarray(db) / 10)))


def summarize(bands, min_attenuation=30):
    '''
    Worst (smallest) attenuation for each stimulated ear, and whether every
    band reaches `min_attenuation` (a band that is a lower bound and below
    it counts as not reaching it, since the true value is unknown).
    '''
    result = {}
    for ear, g in bands.groupby('ear'):
        worst = g.loc[g.attenuation.idxmin()]
        result[ear] = {
            'min_attenuation': worst.attenuation,
            'min_attenuation_frequency': worst.frequency,
            'min_attenuation_lower_bound': bool(worst.lower_bound),
            'passed': bool((g.attenuation >= min_attenuation).all()),
        }
    return result
