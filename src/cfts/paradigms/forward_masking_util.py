'''
Stimulus and epoch helpers for the forward-masking ABR (`abr_forward_masking`).

Each trial is a masker tone burst, a silent gap and a probe tone burst. The
ABR to the probe is cut out of the trial's epoch starting at probe onset
(`masker duration + gap` after the start of the trial), so that the responses
at different gaps line up.
'''
import numpy as np

from atom.api import Float, Str
from enaml.core.api import d_

from psiaudio.pipeline import concat, coroutine, PipelineData
from psiaudio.stim import FixedWaveform

from psi.controller.api import EpochInput


#: Names of the context items (and epoch metadata) that set the probe onset.
MASKER_DURATION = 'target_masker_duration'
GAP = 'target_forward_masking_gap'
MASKER_LEVEL = 'target_masker_tone_level'
MASKER_FREQUENCY = 'target_masker_tone_frequency'
MASKER_POLARITY = 'target_masker_tone_polarity'
PROBE_LEVEL = 'target_probe_tone_level'
PROBE_FREQUENCY = 'target_probe_tone_frequency'
PROBE_POLARITY = 'target_probe_tone_polarity'
PROBE_DURATION = 'target_probe_duration'


class ForwardMaskingFactory(FixedWaveform):
    '''
    Masker, then `gap` seconds of silence, then probe.

    The gap runs from the end of the masker (including its offset ramp) to
    the start of the probe. The masker and probe are the factories of the
    block's two child blocks, in that order.
    '''

    def __init__(self, fs, gap, input_factories):
        if gap < 0:
            raise ValueError(f'Forward masking gap cannot be negative ({gap} s)')
        if len(input_factories) != 2:
            raise ValueError('Forward masking needs a masker and a probe')
        self.gap = gap
        self.masker_factory, self.probe_factory = input_factories
        masker = self.masker_factory.get_samples_remaining()
        probe = self.probe_factory.get_samples_remaining()
        silence = np.zeros(int(round(gap * fs)))
        #: Sample at which the probe starts.
        self.probe_onset = len(masker) + len(silence)
        super().__init__(fs, np.concatenate((masker, silence, probe)))

    def max_amplitude(self):
        return max(self.masker_factory.max_amplitude(),
                   self.probe_factory.max_amplitude())


def probe_onset(metadata):
    '''
    Time of probe onset (s) relative to the start of the trial.
    '''
    return metadata[MASKER_DURATION] + metadata[GAP]


def pair_duration(setting, fs=None):
    '''
    Duration (s) of the masker, gap and probe. With `fs`, each is rounded to
    whole samples, as `ForwardMaskingFactory` plays them.
    '''
    durations = setting[MASKER_DURATION], setting[GAP], setting[PROBE_DURATION]
    if fs is None:
        return sum(durations)
    return sum(int(round(d * fs)) for d in durations) / fs


@coroutine
def align_to_probe(span, target):
    '''
    Cut `span` seconds out of each epoch starting at probe onset (from the
    epoch's own metadata, so the gap can change from trial to trial).
    '''
    while True:
        new_data = []
        for epoch in (yield):
            lb = int(round(probe_onset(epoch.metadata) * epoch.fs))
            ub = lb + int(round(span * epoch.fs))
            if ub > epoch.shape[-1]:
                raise ValueError(
                    f'Epoch is too short to hold the probe response. Probe '
                    f'window ends at sample {ub} but the epoch has '
                    f'{epoch.shape[-1]} samples.')
            d = PipelineData(np.asarray(epoch)[..., lb:ub], fs=epoch.fs, s0=0,
                             channel=epoch.channel,
                             metadata=epoch.metadata.copy())
            new_data.append(d)
        target(concat(new_data, axis='epoch'))


class AlignToProbe(EpochInput):

    #: Duration of the epoch to keep, starting at probe onset.
    span = d_(Float()).tag(metadata=True)

    def _get_duration(self):
        return self.span

    def configure_callback(self):
        cb = super().configure_callback()
        return align_to_probe(self.span, cb).send


def probe_alone_settings(settings):
    '''
    One probe-alone setting (masker silent, gap 0) per probe level, unless the
    settings already hold that condition. Polarity is set when queued.
    '''
    existing = {(s[MASKER_LEVEL], s[PROBE_LEVEL], s[GAP]) for s in settings}
    result = {}
    for s in settings:
        key = (-np.inf, s[PROBE_LEVEL], 0)
        if key in existing or s[PROBE_LEVEL] in result:
            continue
        result[s[PROBE_LEVEL]] = {**s, MASKER_LEVEL: -np.inf, GAP: 0}
    return list(result.values())
