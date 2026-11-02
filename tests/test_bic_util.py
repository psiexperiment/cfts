import numpy as np
import pytest

from psiaudio.queue import QueueEmptyError

from cfts.paradigms.bic_util import (bic_conditions, bic_traces, CLICK,
                                     click_window, CONDITION, delay,
                                     ear_delays, ear_iti, FREQUENCY,
                                     GroupedRandomSignalQueue, ITD, LEFT,
                                     LEVEL, make_queue, ORDERINGS, plot_key,
                                     queue_stimuli, RIGHT, RIGHT_KEY, TONE)


def selector_settings(frequencies, levels, itds):
    return [{FREQUENCY: f, LEVEL: l, ITD: i}
            for f in frequencies for l in levels for i in itds]


def test_conditions():
    conditions = bic_conditions(selector_settings([1000, 2000], [60],
                                                  [0, 500]))
    assert [(c[FREQUENCY], c[CONDITION], c[ITD]) for c in conditions] == [
        (1000, 'left', None), (1000, 'right', None),
        (1000, 'binaural', 0), (1000, 'binaural', 500),
        (2000, 'left', None), (2000, 'right', None),
        (2000, 'binaural', 0), (2000, 'binaural', 500),
    ]
    left, right, b0, b500 = conditions[:4]
    assert left[f'{LEFT}_tone_level'] == 60
    assert left[f'{RIGHT}_tone_level'] == -np.inf
    assert right[f'{LEFT}_tone_level'] == -np.inf
    assert right[f'{RIGHT}_tone_level'] == 60
    assert right[f'{RIGHT}_tone_phase'] == 0
    assert b500[f'{LEFT}_tone_level'] == b500[f'{RIGHT}_tone_level'] == 60


def test_conditions_any_selector_order():
    # ITD varied slowest: still grouped by frequency and level.
    settings = [{FREQUENCY: f, LEVEL: 60, ITD: i}
                for i in [0, 500] for f in [1000, 2000]]
    conditions = bic_conditions(settings)
    assert [(c[FREQUENCY], c[CONDITION], c[ITD]) for c in conditions[:4]] \
        == [(1000, 'left', None), (1000, 'right', None),
            (1000, 'binaural', 0), (1000, 'binaural', 500)]


def fill(ordering, alternate_polarity, averages=4):
    '''
    Two queues filled the way `configure_bic` fills them.
    '''
    conditions = bic_conditions(selector_settings([2000, 4000, 8000],
                                                  [60, 80], [0, 500, -500]))
    stimuli, size = queue_stimuli(conditions, ordering, alternate_polarity,
                                  averages)
    queues = [make_queue(ordering, size) for _ in range(2)]
    keys = []
    for setting, n in stimuli:
        right_key = queues[1].append(np.zeros(10), n, 0, 0.01, metadata=setting)
        left_key = queues[0].append(np.zeros(10), n, 0, 0.01,
                                    metadata={**setting, RIGHT_KEY: right_key})
        keys.append((left_key, right_key))
    return queues, keys, stimuli


def play(queues, reject_every=0):
    '''
    Pop both queues until empty, decrementing both after each trial (the
    way valid epochs do), skipping every `reject_every`th trial as if it
    were rejected. Returns the left queue's metadata and the right queue's
    keys per trial.
    '''
    left, right = queues
    played = []
    i = 0
    while True:
        try:
            lk, ld = left.pop_next(decrement=False)
        except QueueEmptyError:
            with pytest.raises(QueueEmptyError):
                right.pop_next(decrement=False)
            return played
        rk, rd = right.pop_next(decrement=False)
        played.append((ld['metadata'], rk))
        i += 1
        if reject_every and (i % reject_every == 0):
            continue
        for q, k in ((left, lk), (right, rk)):
            try:
                q.decrement_key(k)
            except KeyError:
                pass
        assert i < 10000


@pytest.mark.parametrize('ordering', ORDERINGS)
@pytest.mark.parametrize('alternate_polarity', [True, False])
def test_queues_in_step(ordering, alternate_polarity):
    queues, keys, stimuli = fill(ordering, alternate_polarity)
    played = play(queues, reject_every=7)
    for md, right_key in played:
        assert md[RIGHT_KEY] == right_key
    # Every stimulus got its averages.
    counts = {}
    for i, (md, _) in enumerate(played):
        if (i + 1) % 7:
            counts[md[RIGHT_KEY]] = counts.get(md[RIGHT_KEY], 0) + 1
    for (_, n), (_, right_key) in zip(stimuli, keys, strict=True):
        assert counts.get(right_key, 0) >= n


def test_sequential_groups():
    queues, keys, stimuli = fill('sequential', True, averages=4)
    played = play(queues)
    blocks = [(md[FREQUENCY], md[LEVEL]) for md, _ in played]
    # Frequency and level change only when a group is done.
    changes = [b for i, b in enumerate(blocks) if i == 0 or b != blocks[i-1]]
    assert len(changes) == 6
    # Within a group, conditions and polarities are interleaved: the first
    # block of the group has each (condition, ITD, polarity) once.
    group_size = 5 * 2
    first = played[:group_size]
    assert len({(md[CONDITION], md[ITD], md[f'{LEFT}_tone_polarity'])
                for md, _ in first}) == group_size
    # ... in random rather than queued order.
    assert [md[CONDITION] for md, _ in first] != \
        [s[CONDITION] for s, _ in stimuli[:group_size]]


def test_polarity():
    stimuli, _ = queue_stimuli(
        bic_conditions(selector_settings([1000], [60], [0])), 'sequential',
        True, 4)
    for s, n in stimuli:
        assert n == 2
        assert s[f'{LEFT}_tone_polarity'] == s[f'{RIGHT}_tone_polarity']
    with pytest.raises(ValueError):
        queue_stimuli(bic_conditions(selector_settings([1000], [60], [0])),
                      'sequential', True, 3)


def test_grouped_random_queue_blocks():
    q = GroupedRandomSignalQueue(group_size=3, seed=1)
    keys = [q.append(np.zeros(10), 2, 0, 0.01) for _ in range(6)]
    seen = [q.next_key() for _ in range(3)]
    assert set(seen) == set(keys[:3])


def test_plot_key():
    assert plot_key({CONDITION: 'left', ITD: None}) == ('left',)
    assert plot_key({CONDITION: 'binaural', ITD: -250}) == ('binaural', -250)


def test_bic_traces():
    t = np.arange(10)
    left, right = np.sin(t), np.cos(t)
    means = {('left',): left, ('right',): right,
             ('binaural', 0): left + right + 1}
    sums, bic = bic_traces(means, 25e3)
    np.testing.assert_allclose(sums[None], left + right)
    np.testing.assert_allclose(sums[0], left + right)
    np.testing.assert_allclose(bic[0], 1)
    assert bic_traces({('left',): left}, 25e3) == ({}, {})


@pytest.mark.parametrize('shift', [0.5, 0.25, 3.7])
def test_delay_is_a_pure_delay(shift):
    # A fractional shift moves a sinusoid without changing its amplitude,
    # even near Nyquist, where interpolating between samples would
    # attenuate it (by cos(pi f / fs) at half a sample).
    fs = 25e3
    n = 4096
    t = np.arange(n) / fs
    f = fs / 4 * 1.5
    x = np.sin(2 * np.pi * f * t)
    y = delay(x, shift / fs, fs)
    expected = np.sin(2 * np.pi * f * (t - shift / fs))
    middle = slice(n // 4, 3 * n // 4)
    np.testing.assert_allclose(y[middle], expected[middle], atol=0.02)
    # Nothing before the delay.
    assert np.all(y[:int(shift)] == 0)


def itd_settings(itds, frequency=2000):
    return [{FREQUENCY: frequency, LEVEL: 60, ITD: i} for i in itds]


def test_itd_conditions():
    left, right, b0, b_pos, b_neg = bic_conditions(
        itd_settings([0, 500, -250]))
    start = 'tone_burst_start_time'
    for c in (left, right, b0):
        assert c[f'{LEFT}_{start}'] == c[f'{RIGHT}_{start}'] == 0
        assert c[f'{LEFT}_tone_phase'] == c[f'{RIGHT}_tone_phase'] == 0
    assert left[ITD] is None and b0[ITD] == 0
    # Positive: the right ear leads, so the left burst is delayed, carrier
    # and all (2 kHz for 500 µs is one cycle).
    assert b_pos[f'{LEFT}_{start}'] == pytest.approx(500e-6)
    assert b_pos[f'{RIGHT}_{start}'] == 0
    assert b_pos[f'{LEFT}_tone_phase'] == pytest.approx(-2 * np.pi)
    assert b_pos[f'{RIGHT}_tone_phase'] == 0
    # Negative: the left ear leads.
    assert b_neg[f'{LEFT}_{start}'] == 0
    assert b_neg[f'{RIGHT}_{start}'] == pytest.approx(250e-6)
    assert b_neg[f'{RIGHT}_tone_phase'] == pytest.approx(-np.pi)


def test_itd_rounded_to_samples():
    # 125 µs at 97656.25 Hz is 12.2 samples: rounded to 12.
    fs = 97656.25
    assert ear_delays(125, fs) == (12 / fs, 0)
    assert ear_delays(-125, fs) == (0, 12 / fs)
    assert ear_delays(0, fs) == (0, 0)


def test_itd_burst_is_delayed_tone():
    # The delayed ear's burst is the undelayed burst, shifted: the carrier
    # moves with the envelope.
    from psiaudio import stim
    fs = 100e3
    frequency, d = 2000, 250e-6

    def burst(start_time, phase):
        tone = stim.ToneFactory(fs, frequency, 0, phase=phase)
        return stim.Cos2EnvelopeFactory(fs, 5e-3, 0.5e-3, tone,
                                        start_time=start_time) \
            .get_samples_remaining()

    c = make_itd_condition(frequency, 250)
    delayed = burst(d, c[f'{RIGHT}_tone_phase'])
    reference = burst(0, 0)
    n = int(round(d * fs))
    np.testing.assert_allclose(delayed[n:], reference, atol=1e-12)
    np.testing.assert_allclose(delayed[:n], 0, atol=1e-12)


def make_itd_condition(frequency, itd):
    return bic_conditions(itd_settings([-itd], frequency))[-1]


def test_itd_trials_take_same_time():
    # A delayed burst is longer; its interval is that much shorter.
    duration = 5e-3
    for itd in (-1000, 0, 750):
        c = bic_conditions(itd_settings([itd]))[-1]
        totals = []
        for ear in (LEFT, RIGHT):
            d = c[f'{ear}_tone_burst_start_time']
            totals.append(d + duration + ear_iti(0.025, d))
        assert totals[0] == pytest.approx(totals[1])

    # With jitter, both ears draw the same intervals from their own
    # generators.
    def jitter():
        rng = np.random.default_rng(1)
        while True:
            yield rng.uniform(0.02, 0.03)

    left, right = jitter(), jitter()
    left_iti = ear_iti(left, 1e-3)
    right_iti = ear_iti(right, 0)
    for _ in range(5):
        assert next(left_iti) + 1e-3 == pytest.approx(next(right_iti))


def test_itd_bic_traces():
    # With no binaural interaction, the binaural ABR is each ear's ABR
    # delayed as its burst was, so the BIC is zero at every ITD.
    fs = 25e3
    t = np.arange(300) / fs
    left = np.exp(-((t - 3e-3) / 0.3e-3) ** 2)
    right = -0.5 * np.exp(-((t - 4e-3) / 0.4e-3) ** 2)
    means = {('left',): left, ('right',): right}
    for itd in (-1000, -250, 0, 500):
        dl, dr = ear_delays(itd)
        means[('binaural', itd)] = delay(left, dl, fs) + delay(right, dr, fs)
    sums, bic = bic_traces(means, fs)
    for y in bic.values():
        np.testing.assert_allclose(y, 0, atol=1e-12)
    # ... but not if the monaural ABRs are summed as they are.
    assert np.abs(means[('binaural', 500)] - sums[None]).max() > 0.1


def click_settings(itds, levels=(60,)):
    return [{LEVEL: l, ITD: i} for l in levels for i in itds]


def test_click_conditions():
    conditions = bic_conditions(click_settings([0, 500, -250], [60, 80]),
                                CLICK)
    assert [(c[LEVEL], c[CONDITION], c[ITD]) for c in conditions[:5]] == [
        (60, 'left', None), (60, 'right', None), (60, 'binaural', 0),
        (60, 'binaural', 500), (60, 'binaural', -250)]
    left, right, b0, b_pos, b_neg = conditions[:5]
    assert left[f'{LEFT}_click_level'] == 60
    assert left[f'{RIGHT}_click_level'] == -np.inf
    assert right[f'{LEFT}_click_level'] == -np.inf
    delay_name = 'click_train_delay'
    assert b_pos[f'{LEFT}_{delay_name}'] == pytest.approx(500e-6)
    assert b_pos[f'{RIGHT}_{delay_name}'] == 0
    assert b_neg[f'{LEFT}_{delay_name}'] == 0
    assert b_neg[f'{RIGHT}_{delay_name}'] == pytest.approx(250e-6)
    # No carrier, so no phase, and no tone settings.
    assert not any('tone' in k for k in b_pos)


def test_click_queue_grouped_by_level():
    conditions = bic_conditions(click_settings([0, 500], [60, 80]), CLICK)
    for ordering in ORDERINGS:
        stimuli, size = queue_stimuli(conditions, ordering, True, 4, CLICK)
        assert size == (4 * 2 if not ordering.endswith(('ramp', 'plateau'))
                        else 4)
        for s, _ in stimuli:
            assert s[f'{LEFT}_click_polarity'] == s[f'{RIGHT}_click_polarity']
        if ordering == 'sequential':
            assert [s[LEVEL] for s, _ in stimuli] == [60] * 8 + [80] * 8


def test_click_is_delayed_in_fixed_window():
    # Each ear's click plays once in a window that fits the largest delay,
    # so both ears' trials take the same time; the lagging ear's click
    # starts the ITD later (to the nearest sample).
    from psiaudio import stim
    from psiaudio.calibration import FlatCalibration
    fs = 96e3
    duration = 100e-6
    cal = FlatCalibration.unity()
    itds = [-1000, -250, 0, 125, 1000]
    window = click_window(fs, duration, 1000e-6)
    for c in bic_conditions(click_settings(itds), CLICK)[2:]:
        waveforms = []
        for ear in (LEFT, RIGHT):
            click = stim.ClickFactory(fs, duration, 0, 1, cal)
            factory = stim.RepeatFactory(fs, 1, 0, 1 / window,
                                         c[f'{ear}_click_train_delay'], click)
            assert factory.get_duration() == pytest.approx(window)
            waveforms.append(factory.get_samples_remaining())
        left, right = waveforms
        assert len(left) == len(right) == round(window * fs)
        onset = np.argmax(right > 0) - np.argmax(left > 0)
        assert onset / fs == pytest.approx(-c[ITD] * 1e-6, abs=0.5 / fs)
        assert left.sum() == right.sum() == int(fs * duration)


def play_ears(fs, values, stim_name, frequency=8000, averages=40):
    '''
    Fill both ears' queues the way `configure_bic` does, with real stimuli
    and jittered intervals, play them out sample by sample and return each
    ear's trial onsets (samples) with the left ear's metadata.
    '''
    from psiaudio import stim
    from psiaudio.calibration import FlatCalibration
    cal = FlatCalibration.unity()
    if stim_name == 'tone':
        settings = [{FREQUENCY: frequency, LEVEL: 0, ITD: v}
                    for v in values]
        s = TONE
    else:
        settings = click_settings(values, [0])
        s = CLICK
    conditions = bic_conditions(settings, s, fs)
    stimuli, size = queue_stimuli(conditions, 'sequential', True, averages, s)
    max_delay = max(max(ear_delays(v, fs)) for v in values)
    window = click_window(fs, 100e-6, max_delay)

    def jitter():
        rng = np.random.default_rng(1)
        while True:
            yield rng.uniform(0.020, 0.040)

    base = {LEFT: jitter(), RIGHT: jitter()}
    queues = {ear: make_queue('sequential', size) for ear in (LEFT, RIGHT)}
    for q in queues.values():
        q.set_fs(fs)
    onsets = {LEFT: [], RIGHT: []}
    for ear, q in queues.items():
        q.connect(lambda info, ear=ear: onsets[ear].append(
            (info['t0'], info['metadata'])), 'added')
    for setting, n in stimuli:
        for ear, q in queues.items():
            level = setting[s.ear_name(ear, 'level')]
            polarity = setting[s.ear_name(ear, 'polarity')]
            delay_s = setting[s.ear_name(ear, 'delay')]
            if s is TONE:
                tone = stim.ToneFactory(fs, frequency, level,
                                        phase=setting[s.ear_name(ear, 'phase')],
                                        polarity=polarity, calibration=cal)
                factory = stim.Cos2EnvelopeFactory(fs, 5e-3, 0.5e-3, tone,
                                                   start_time=delay_s)
                iti = ear_iti(base[ear], delay_s, fs)
            else:
                click = stim.ClickFactory(fs, 100e-6, level, polarity, cal)
                factory = stim.RepeatFactory(fs, 1, 0, 1 / window, delay_s,
                                             click)
                iti = ear_iti(base[ear], 0, fs)
            q.append(factory, n, iti, factory.get_duration(), setting)
    waveforms = {}
    for ear, q in queues.items():
        chunks = []
        while True:
            chunk = q.pop_buffer(100_000)
            chunks.append(chunk)
            if q.is_empty():
                break
        waveforms[ear] = np.concatenate(chunks)
        q.notifier.join(timeout=5)
    return onsets, waveforms


@pytest.mark.parametrize('stim_name', ['tone', 'click'])
@pytest.mark.parametrize('fs', [96e3, 97656.25, 195312.5])
def test_ears_stay_in_step(fs, stim_name):
    # Each queue rounds every stimulus and interval to samples on its own,
    # so the ITD (and the interval after a delayed burst) must be whole
    # samples or the ears drift apart over the run.
    onsets, waveforms = play_ears(fs, [0, 250, -500, 1000, -125], stim_name)
    left = [round(t * fs) for t, _ in onsets[LEFT]]
    right = [round(t * fs) for t, _ in onsets[RIGHT]]
    assert len(left) == len(right) > 100
    assert left == right
    # And each binaural stimulus starts in each ear where it should.
    first = {LEFT: [], RIGHT: []}
    for (t, md) in onsets[LEFT]:
        if md[CONDITION] != 'binaural':
            continue
        i = round(t * fs)
        for ear in (LEFT, RIGHT):
            w = waveforms[ear][i:i + int(0.01 * fs)]
            first[ear].append(np.flatnonzero(np.abs(w) > 1e-9)[0])
        expected = -md[ITD] * 1e-6 * fs
        assert first[RIGHT][-1] - first[LEFT][-1] \
            == pytest.approx(expected, abs=0.5 + 1)
