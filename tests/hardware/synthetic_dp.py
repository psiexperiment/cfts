'''
The synthetic DP's envelope, shared by `moc_synthetic.enaml` (which plays it)
and `moc_analysis` (which checks what psi measured against it).
'''
import numpy as np

from psiaudio.stim import Modulator


def suppression_envelope(t, period, start_time, duration, rise_time, depth):
    '''
    1 except from `start_time` to `start_time + duration` of every `period`
    (s), where it is `depth` dB lower, with cos² ramps of `rise_time` inside
    that window.
    '''
    tt = t % period - start_time
    ramp = np.clip(np.minimum(tt, duration - tt) / rise_time, 0, 1)
    w = np.sin(np.pi / 2 * ramp) ** 2
    return 1 - (1 - 10 ** (-depth / 20)) * w


class SuppressionEnvelopeFactory(Modulator):

    def __init__(self, fs, period, start_time, duration, rise_time, depth,
                 input_factory):
        vars(self).update(locals())
        self.reset()

    def env(self, samples):
        t = (self.offset + np.arange(samples)) / self.fs
        return suppression_envelope(t, self.period, self.start_time,
                                    self.duration, self.rise_time,
                                    self.depth)

    def max_amplitude(self):
        return self.input_factory.max_amplitude()
