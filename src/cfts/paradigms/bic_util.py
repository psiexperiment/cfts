'''
Stimulus conditions, queue ordering and averaging for the binaural
interaction component (BIC) of the ABR.

The BIC is the binaural ABR minus the sum of the left and right monaural
ABRs. For each frequency and level (tone bursts) or level (clicks), the left
and right monaural stimuli are presented once, and the binaural stimulus once
per interaural time difference (ITD, µs): the whole stimulus in one ear (for
a tone burst, envelope and carrier) is delayed. Positive means the right ear
leads (the left is delayed).

The two ears are driven by separate outputs whose queues are filled
identically, so they stay in step trial by trial.

Pure Python (no Qt or enaml), so it can be tested without hardware.
'''
from collections import Counter
from dataclasses import dataclass

import numpy as np

from psiaudio.queue import (BlockedRandomSignalQueue, GroupedFIFOSignalQueue,
                            InterleavedFIFOSignalQueue, QueueEmptyError)
from psiaudio.util import interleave_octaves


#: Context names the selector manages.
FREQUENCY = 'bic_frequency'
LEVEL = 'bic_level'
ITD = 'bic_itd'

#: Metadata added to each queued stimulus: which ear(s) play it.
CONDITION = 'bic_condition'
CONDITIONS = ('left', 'right', 'binaural')

#: Outputs driving each ear, and the key of the right ear's stimulus, which
#: is stored with the left ear's (the one epochs are extracted from).
LEFT = 'left_target'
RIGHT = 'right_target'
RIGHT_KEY = 'right_key'

ORDERINGS = ('sequential', 'interleaved_plateau', 'interleaved_ramp',
             'interleaved_random')


@dataclass(frozen=True)
class Stimulus:
    '''
    Context names of an ear's stimulus that the conditions set, and the
    selector's items that group the conditions.
    '''
    name: str
    #: Items the conditions are grouped by (each group has every condition).
    groups: tuple
    #: Context names (after the ear's output name) of the level, polarity and
    #: delay of an ear's stimulus.
    level: str
    polarity: str
    delay: str
    #: Context name of the carrier phase, which a delay must also shift (the
    #: tone starts with the waveform, not the burst). None if no carrier.
    phase: str = None
    #: Whether the delay lengthens the stimulus. A tone burst's start time
    #: does; a click is delayed within a window of fixed length.
    delay_lengthens: bool = False

    def ear_name(self, output, name):
        '''
        Context name of `name` ('level', 'polarity', 'delay' or 'phase') of
        the stimulus in an ear.
        '''
        return f'{output}_{getattr(self, name)}'

    def group_key(self, setting):
        return tuple(setting[g] for g in self.groups)

    def describe(self, setting):
        if FREQUENCY in self.groups:
            return f'{setting[FREQUENCY]:.0f} Hz at {setting[LEVEL]} dB SPL'
        return f'{self.name.capitalize()} at {setting[LEVEL]} dB SPL'


#: Tone bursts: a `Cos2Envelope` named `tone_burst` around a `Tone`.
TONE = Stimulus('tone', (FREQUENCY, LEVEL), level='tone_level',
                polarity='tone_polarity', delay='tone_burst_start_time',
                phase='tone_phase', delay_lengthens=True)

#: Clicks: a `Click` in a `Repeat` named `click_train` that plays it once,
#: `delay` into a window of fixed length.
CLICK = Stimulus('click', (LEVEL,), level='click_level',
                 polarity='click_polarity', delay='click_train_delay')


def ear_delays(itd, fs=None):
    '''
    Delay (s) of the left and right ear's stimulus for a binaural stimulus
    at `itd` (µs): the lagging ear's is delayed by the ITD (to the nearest
    sample if the sampling rate `fs` is given).
    '''
    if itd:
        delay = abs(itd) * 1e-6
        if fs is not None:
            delay = round(delay * fs) / fs
        return (delay, 0) if itd > 0 else (0, delay)
    return (0, 0)


def make_condition(setting, condition, itd=None, stim=TONE, fs=None):
    '''
    Copy of `setting` for one condition: the silent ear is set to -inf dB SPL
    and, for binaural stimuli, the right ear leads the left by `itd` (µs).

    Pass the outputs' sampling rate `fs` to round the ITD to whole samples.
    The ears' queues each round every interval to samples, so a delay that
    is not a whole number of samples would let the ears drift apart (see
    `ear_iti`).
    '''
    if condition not in CONDITIONS:
        raise ValueError(f'Unknown condition {condition!r}')
    binaural = condition == 'binaural'
    s = dict(setting)
    s[CONDITION] = condition
    s[ITD] = itd if binaural else None
    level = setting[LEVEL]
    s[stim.ear_name(LEFT, 'level')] = -np.inf if condition == 'right' \
        else level
    s[stim.ear_name(RIGHT, 'level')] = -np.inf if condition == 'left' \
        else level
    delays = ear_delays(itd, fs) if binaural else (0, 0)
    for ear, delay in zip((LEFT, RIGHT), delays, strict=True):
        s[stim.ear_name(ear, 'delay')] = delay
        if stim.phase is not None:
            # The tone starts with the waveform, not the burst, so a delayed
            # burst needs its carrier delayed too.
            s[stim.ear_name(ear, 'phase')] = \
                -2 * np.pi * setting[FREQUENCY] * delay
    return s


def bic_conditions(settings, stim=TONE, fs=None):
    '''
    Expand the selector's settings (one per group, e.g. frequency and level,
    and ITD) into the stimuli to present, grouped in the order the selector
    first gives them: left, right, then binaural at each ITD. See
    `make_condition` for `fs`.
    '''
    groups = {}
    for s in settings:
        groups.setdefault(stim.group_key(s), []).append(s)
    result = []
    for group in groups.values():
        result.append(make_condition(group[0], 'left', stim=stim))
        result.append(make_condition(group[0], 'right', stim=stim))
        for s in group:
            result.append(make_condition(s, 'binaural', s[ITD], stim, fs))
    return result


def group_size(conditions, stim=TONE):
    '''
    Number of conditions per group (e.g. frequency and level).
    '''
    counts = Counter(stim.group_key(c) for c in conditions)
    sizes = set(counts.values())
    if len(sizes) != 1:
        raise ValueError(f'Every {" and ".join(stim.groups)} must have the '
                         'same ITDs')
    return sizes.pop()


class GroupedRandomSignalQueue(GroupedFIFOSignalQueue):
    '''
    Like `GroupedFIFOSignalQueue`, but presents the stimuli within a group in
    blocked random order (each stimulus once, in a new random order, per
    block). Seeded, so two queues filled the same way stay in step.
    '''
    def __init__(self, group_size, seed=0, **kwargs):
        super().__init__(group_size, **kwargs)
        self._rng = np.random.RandomState(seed)
        self._block = []

    def next_key(self):
        if len(self._ordering) == 0:
            raise QueueEmptyError
        n = min(self._group_size, len(self._ordering))
        if not self._block:
            self._block = self._rng.permutation(n).tolist()
        i = self._block.pop()
        return self._ordering[min(i, n - 1)]

    def decrement_key(self, key, n=1):
        complete = super().decrement_key(key, n)
        if complete:
            # Start the next group on a fresh block.
            self._block = []
        return complete

    def remove_key(self, key):
        super().remove_key(key)
        # The block indexes the group, which has changed.
        self._block = []


def make_queue(ordering, size, seed=0):
    '''
    Queue for one ear. Call once per ear with the same arguments.

    `sequential` presents each group (e.g. frequency and level) in turn, with
    its conditions (and polarities) randomly interleaved; the others
    interleave every group as the ABR IO paradigm does.
    '''
    if ordering == 'sequential':
        return GroupedRandomSignalQueue(group_size=size, seed=seed)
    if ordering == 'interleaved_random':
        return BlockedRandomSignalQueue(seed=seed)
    if ordering in ('interleaved_ramp', 'interleaved_plateau'):
        return InterleavedFIFOSignalQueue()
    raise ValueError(f'Unrecognized ordering {ordering}')


def queue_stimuli(conditions, ordering, alternate_polarity, averages,
                  stim=TONE):
    '''
    The stimuli in the order to add them to each ear's queue, as a list of
    (setting, trials), with the polarity of both ears set. Returns that list
    and the group size `make_queue` needs.

    With alternating polarity, `averages` is split between the two
    polarities and must be even.
    '''
    conditions = list(conditions)
    size = group_size(conditions, stim)
    if ordering.startswith('interleaved'):
        if ordering == 'interleaved_random' or FREQUENCY not in stim.groups:
            key = stim.group_key
        else:
            int_freq = interleave_octaves({c[FREQUENCY] for c in conditions})
            if ordering == 'interleaved_ramp':
                key = lambda c: (int_freq.index(c[FREQUENCY]), c[LEVEL])
            else:
                key = lambda c: (c[LEVEL], int_freq.index(c[FREQUENCY]))
        # Stable, so the conditions stay in order within each group.
        conditions.sort(key=key)
    elif ordering != 'sequential':
        raise ValueError(f'Unrecognized ordering {ordering}')

    def with_polarity(c, polarity):
        c = dict(c)
        c[stim.ear_name(LEFT, 'polarity')] = polarity
        c[stim.ear_name(RIGHT, 'polarity')] = polarity
        return c

    if not alternate_polarity:
        return [(with_polarity(c, 1), averages) for c in conditions], size

    if averages % 2:
        raise ValueError('Averages must be even to alternate polarity')
    n = averages // 2
    if ordering in ('interleaved_ramp', 'interleaved_plateau'):
        # All of one polarity, then the other, as the ABR IO paradigm does.
        stimuli = [(with_polarity(c, p), n) for p in (1, -1)
                   for c in conditions]
        return stimuli, size

    # Both polarities of a group go in it.
    stimuli = []
    for i in range(0, len(conditions), size):
        group = conditions[i:i + size]
        for p in (1, -1):
            stimuli.extend((with_polarity(c, p), n) for c in group)
    return stimuli, size * 2


def ear_iti(iti, delay, fs=None):
    '''
    Interval after a stimulus lengthened by `delay` (a tone burst delayed by
    its start time). It is that much shorter, so both ears' trials take the
    same time. `iti` is a number or an iterator (one per ear, shared by all
    of its stimuli, so both ears draw the same intervals).

    Each queue rounds the stimulus and interval to samples separately, so
    with `fs` the interval is rounded to whole samples first; with `delay`
    also whole samples (see `make_condition`), both ears' trials then take
    exactly the same number of samples.
    '''
    def adjust(i):
        if fs is not None:
            i = round(i * fs) / fs
        return i - delay

    if np.iterable(iti):
        return (adjust(i) for i in iti)
    return adjust(iti)


def click_window(fs, duration, max_delay):
    '''
    Length (s, a whole number of samples at `fs`) of the window a click of
    `duration` is played in, so that it fits at any delay up to `max_delay`.
    One sample to spare, as the click and delay are rounded to samples.
    '''
    return (np.ceil(fs * (duration + max_delay)) + 1) / fs


def plot_key(md):
    '''
    Trace an epoch is averaged into: ('left',), ('right',) or ('binaural',
    ITD).
    '''
    condition = md[CONDITION]
    if condition == 'binaural':
        return (condition, md[ITD])
    return (condition,)


def delay(x, seconds, fs):
    '''
    `x` delayed by `seconds`, which need not be a whole number of samples
    (the EEG is usually decimated, so an ITD rounded to output samples is
    rarely whole EEG samples).

    The shift multiplies the spectrum by exp(-2πif·seconds), a pure delay at
    every frequency. Linear interpolation between samples would instead
    average neighbouring samples, low-pass filtering the trace (most at half
    a sample: a gain of cos(πf/fs), -3 dB at fs/4) and so blunting the sum's
    peaks relative to the binaural ABR, which would leak into the BIC. The
    trace is zero-padded so nothing wraps around, and zeroed before the
    delay (the ear's response cannot start before its stimulus).
    '''
    if not seconds:
        return x
    n = x.shape[-1]
    n_shift = int(np.ceil(seconds * fs))
    n_fft = n + n_shift
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    X = np.fft.rfft(x, n_fft) * np.exp(-2j * np.pi * f * seconds)
    y = np.fft.irfft(X, n_fft)[..., :n]
    # The delayed signal starts at the delay.
    y[..., :int(np.floor(seconds * fs))] = 0
    return y


def bic_traces(means, fs):
    '''
    Given the average for each trace (keyed as `plot_key` returns) and
    their sampling rate `fs`, the sum of the monaural averages and the BIC
    (binaural minus that sum) for each ITD that has a binaural average. Both
    are dicts keyed by the ITD; `sums[None]` is the plain sum. Both are
    empty unless both monaural averages are available.

    Each epoch starts at the onset of the leading ear's stimulus, so in the
    binaural epoch the lagging ear's response comes later by the ITD than in
    its monaural epoch. Each monaural average is therefore delayed as its
    ear's stimulus was before summing, so that the sum is what the binaural
    ABR would be without binaural interaction (see `delay`).
    '''
    left = means.get(('left',))
    right = means.get(('right',))
    if left is None or right is None:
        return {}, {}
    n = min(left.shape[-1], right.shape[-1])
    left, right = left[..., :n], right[..., :n]
    sums = {None: left + right}
    bic = {}
    for key, binaural in means.items():
        if key[0] != 'binaural' or binaural is None:
            continue
        value = key[1]
        left_delay, right_delay = ear_delays(value)
        if left_delay or right_delay:
            sums[value] = delay(left, left_delay, fs) \
                + delay(right, right_delay, fs)
        else:
            sums[value] = sums[None]
        m = min(n, binaural.shape[-1])
        bic[value] = binaural[..., :m] - sums[value][..., :m]
    return sums, bic
