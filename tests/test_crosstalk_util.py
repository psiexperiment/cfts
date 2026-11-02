import numpy as np
import pytest

from psiaudio import stim
from psiaudio.calibration import FlatCalibration

from cfts.paradigms.crosstalk_util import (CrosstalkAnalyzer, EARS, other,
                                           summarize)


FS = 96e3
DURATION = 0.025
FS_BOTH = {ear: FS for ear in EARS}
CAL = {ear: FlatCalibration.from_db(100) for ear in EARS}


def chirp(level, fs=FS):
    x = stim.chirp(fs, 2e3, 40e3, 0.02, level, CAL['left'], window='hann')
    return np.pad(x, (0, int(round(DURATION * fs)) - len(x)))


def run(attenuation, noise_rms, n=64, seed=1, fs=FS_BOTH):
    '''
    Chirps at 80 dB SPL in each ear in turn; the other ear gets the chirp
    `attenuation[ear]` dB down, and both microphones add white noise. `fs`
    is each microphone's sampling rate.
    '''
    rng = np.random.default_rng(seed)
    analyzer = CrosstalkAnalyzer(fs, DURATION, CAL)
    x = {mic: chirp(80, fs[mic]) for mic in EARS}
    for _ in range(n):
        for ear in EARS:
            for mic in EARS:
                gain = 1 if mic == ear else 10 ** (-attenuation[ear] / 20)
                y = gain * x[mic]                     + rng.normal(scale=noise_rms, size=len(x[mic]))
                analyzer.add(ear, mic, y)
    return analyzer


def test_attenuation():
    analyzer = run({'left': 40, 'right': 55}, noise_rms=1e-7)
    bands = analyzer.bands(4e3, 32e3)
    assert set(bands.ear) == set(EARS)
    for ear, expected in (('left', 40), ('right', 55)):
        g = bands[bands.ear == ear]
        np.testing.assert_allclose(g.attenuation, expected, atol=0.2)
        assert not g.lower_bound.any()
    s = summarize(bands, 50)
    assert s['left']['min_attenuation'] == pytest.approx(40, abs=0.2)
    assert not s['left']['passed']
    assert s['right']['passed']


def test_different_sampling_rates():
    # Microphones on different engines, at different rates: the band levels
    # depend on the epoch duration, not the rate.
    fs = {'left': 96e3, 'right': 192e3}
    analyzer = run({'left': 40, 'right': 55}, noise_rms=1e-7, fs=fs)
    bands = analyzer.bands(4e3, 32e3)
    for ear, expected in (('left', 40), ('right', 55)):
        g = bands[bands.ear == ear]
        np.testing.assert_allclose(g.attenuation, expected, atol=0.2)
    same = run({'left': 40, 'right': 55}, noise_rms=1e-7).bands(4e3, 32e3)
    np.testing.assert_allclose(bands.ipsi_level, same.ipsi_level, atol=0.1)


def test_noise_floor_and_lower_bound():
    # No crosstalk at all: the other ear has only noise, so the attenuation
    # is just a lower bound, set by the noise floor.
    analyzer = run({'left': np.inf, 'right': np.inf}, noise_rms=1e-4)
    bands = analyzer.bands(4e3, 32e3)
    assert bands.lower_bound.all()
    # The noise floor estimate matches the level of a noise-only average.
    np.testing.assert_allclose(bands.contra_level, bands.contra_noise_floor,
                               atol=3)
    # Four times the averages: the noise floor drops 6 dB.
    more = run({'left': np.inf, 'right': np.inf}, noise_rms=1e-4, n=256)
    drop = bands.contra_noise_floor.values \
        - more.bands(4e3, 32e3).contra_noise_floor.values
    assert np.median(drop) == pytest.approx(6, abs=1)


def test_unequal_even_odd():
    analyzer = run({'left': 40, 'right': 40}, noise_rms=1e-7, n=5)
    assert analyzer.count('left', 'right') == 5
    assert analyzer.spectrum('left', 'right') is not None
    empty = CrosstalkAnalyzer(FS_BOTH, DURATION, CAL)
    assert empty.spectrum('left', 'left') is None


def test_other():
    assert other('left') == 'right' and other('right') == 'left'
