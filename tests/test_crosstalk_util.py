import numpy as np
import pytest

from psiaudio import stim
from psiaudio.calibration import FlatCalibration

from cfts.paradigms.crosstalk_util import (CrosstalkAnalyzer, EARS, other,
                                           summarize)


FS = 96e3
N = int(0.025 * FS)
CAL = {ear: FlatCalibration.from_db(100) for ear in EARS}


def chirp(level):
    x = stim.chirp(FS, 2e3, 40e3, 0.02, level, CAL['left'], window='hann')
    return np.pad(x, (0, N - len(x)))


def run(attenuation, noise_rms, n=64, seed=1):
    '''
    Chirps at 80 dB SPL in each ear in turn; the other ear gets the chirp
    `attenuation[ear]` dB down, and both microphones add white noise.
    '''
    rng = np.random.default_rng(seed)
    analyzer = CrosstalkAnalyzer(FS, N, CAL)
    x = chirp(80)
    for _ in range(n):
        for ear in EARS:
            for mic in EARS:
                gain = 1 if mic == ear else 10 ** (-attenuation[ear] / 20)
                y = gain * x + rng.normal(scale=noise_rms, size=N)
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
    assert CrosstalkAnalyzer(FS, N, CAL).spectrum('left', 'left') is None


def test_other():
    assert other('left') == 'right' and other('right') == 'left'
