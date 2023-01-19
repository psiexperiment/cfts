from pathlib import Path

import pandas as pd

from psiaudio.calibration import InterpCalibration


def load_epl_calibration(filename):
    with Path(filename).open('r') as fh:
        for line in fh:
            if line.startswith('Freq(Hz)'):
                break
        cal = pd.read_csv(fh, sep='\t', header=None)
        return InterpCalibration.from_spl(cal[0], cal[1])

