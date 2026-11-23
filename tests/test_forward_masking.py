'''
Stimulus and epoch helpers of the forward-masking ABR (`forward_masking_util`).
'''
import numpy as np
import pytest

from psiaudio.calibration import FlatCalibration
from psiaudio.pipeline import concat, PipelineData
from psiaudio.stim import EnvelopeFactory, ToneFactory

from cfts.paradigms.forward_masking_util import (
    align_to_probe, ForwardMaskingFactory, GAP, MASKER_DURATION,
    MASKER_LEVEL, PROBE_DURATION, PROBE_LEVEL, pair_duration,
    probe_alone_settings, probe_onset
)


FS = 100e3
CAL = FlatCalibration.from_spl(94)
MASKER_DUR = 50e-3
PROBE_DUR = 5e-3
RISE = 1e-3


def burst(frequency, level, duration, polarity=1):
    tone = ToneFactory(FS, frequency, level, polarity=polarity,
                       calibration=CAL)
    return EnvelopeFactory('cosine-squared', FS, duration, RISE, tone)


def factory(gap, masker_level=80, probe_level=60, polarity=1):
    return ForwardMaskingFactory(FS, gap, [
        burst(4000, masker_level, MASKER_DUR, polarity),
        burst(8000, probe_level, PROBE_DUR, polarity),
    ])


@pytest.mark.parametrize('gap', [0, 1e-3, 2.5e-3, 32e-3])
def test_layout(gap):
    f = factory(gap)
    w = f.get_samples_remaining()
    s_masker = int(round(MASKER_DUR * FS))
    s_gap = int(round(gap * FS))
    s_probe = int(round(PROBE_DUR * FS))
    assert len(w) == s_masker + s_gap + s_probe
    assert f.probe_onset == s_masker + s_gap
    assert f.get_duration() == pytest.approx(MASKER_DUR + gap + PROBE_DUR)

    masker = w[:s_masker]
    probe = w[f.probe_onset:]
    np.testing.assert_array_equal(w[s_masker:f.probe_onset], 0)
    np.testing.assert_allclose(masker, burst(4000, 80, MASKER_DUR)
                               .get_samples_remaining())
    np.testing.assert_allclose(probe, burst(8000, 60, PROBE_DUR)
                               .get_samples_remaining())


def test_levels_independent():
    w = factory(5e-3, masker_level=80, probe_level=40).get_samples_remaining()
    s_masker = int(round(MASKER_DUR * FS))
    masker_peak = np.abs(w[:s_masker]).max()
    probe_peak = np.abs(w[s_masker:]).max()
    assert 20 * np.log10(masker_peak / probe_peak) == pytest.approx(40, abs=0.1)


def test_polarity_flips_both():
    pos = factory(5e-3, polarity=1).get_samples_remaining()
    neg = factory(5e-3, polarity=-1).get_samples_remaining()
    np.testing.assert_allclose(neg, -pos)


def test_silent_masker():
    f = factory(0, masker_level=-np.inf)
    w = f.get_samples_remaining()
    np.testing.assert_array_equal(w[:f.probe_onset], 0)
    assert np.abs(w[f.probe_onset:]).max() > 0


def test_max_amplitude():
    f = factory(5e-3, masker_level=80, probe_level=60)
    w = f.get_samples_remaining()
    assert f.max_amplitude() >= np.abs(w).max()


def test_negative_gap():
    with pytest.raises(ValueError):
        factory(-1e-3)


def test_reset_replays():
    f = factory(5e-3)
    first = f.get_samples_remaining()
    f.reset()
    np.testing.assert_array_equal(f.get_samples_remaining(), first)


def md(gap, masker_level=80, probe_level=60):
    return {MASKER_DURATION: MASKER_DUR, GAP: gap, PROBE_DURATION: PROBE_DUR,
            MASKER_LEVEL: masker_level, PROBE_LEVEL: probe_level}


def test_pair_duration():
    assert probe_onset(md(4e-3)) == pytest.approx(MASKER_DUR + 4e-3)
    assert pair_duration(md(4e-3)) == pytest.approx(MASKER_DUR + 4e-3 + PROBE_DUR)


def test_align_to_probe():
    '''
    Epochs whose samples are their own indices, with a different gap each:
    every probe-aligned epoch starts at that epoch's probe onset.
    '''
    eeg_fs = 25e3
    span = 8.5e-3
    gaps = [0, 2e-3, 8e-3, 32e-3]
    n = int(round((MASKER_DUR + max(gaps) + span + 0.5e-3) * eeg_fs))
    epochs = [PipelineData(np.arange(n, dtype=float)[np.newaxis], fs=eeg_fs,
                           s0=0, metadata=md(g)) for g in gaps]
    result = []
    align_to_probe(span, result.append).send(concat(epochs, axis='epoch'))
    aligned = result[0]
    s_span = int(round(span * eeg_fs))
    assert aligned.shape[-1] == s_span
    for epoch, gap in zip(aligned, gaps):
        start = int(round((MASKER_DUR + gap) * eeg_fs))
        np.testing.assert_array_equal(np.asarray(epoch)[0],
                                      np.arange(start, start + s_span))
        assert epoch.metadata[GAP] == gap


def test_align_to_probe_short_epoch():
    eeg_fs = 25e3
    epoch = PipelineData(np.zeros((1, 100)), fs=eeg_fs, s0=0, metadata=md(0))
    with pytest.raises(ValueError):
        align_to_probe(8.5e-3, lambda x: None).send(
            concat([epoch], axis='epoch'))


def test_probe_alone_settings():
    settings = [md(g, ml, pl) for g in (2e-3, 8e-3) for ml in (60, 80)
                for pl in (40, 60)]
    extra = probe_alone_settings(settings)
    assert sorted(s[PROBE_LEVEL] for s in extra) == [40, 60]
    for s in extra:
        assert s[MASKER_LEVEL] == -np.inf
        assert s[GAP] == 0
        assert s[MASKER_DURATION] == MASKER_DUR


def test_probe_alone_not_duplicated():
    settings = [md(2e-3, 80, 40), md(0, -np.inf, 40), md(2e-3, 80, 60)]
    extra = probe_alone_settings(settings)
    assert [s[PROBE_LEVEL] for s in extra] == [60]


def test_token_parameters():
    '''
    The paradigm's token registers the names the paradigm and selector use,
    and psi builds the same waveform from them.
    '''
    from types import SimpleNamespace
    import enaml
    from psi.controller.token_context import initialize_factory, load_items
    with enaml.imports():
        from cfts.paradigms.abr_forward_masking import ForwardMasking
        from psi.token.primitives import Envelope, Tone

    token = ForwardMasking()
    for name in ('masker', 'probe'):
        env = Envelope(parent=token, name=name, label=name)
        Tone(parent=env, name=f'{name}_tone', label=f'{name} tone')

    output = SimpleNamespace(name='target', fs=FS, calibration=CAL)
    names = {p.name: p for p in load_items(output, token)}
    for name in (GAP, MASKER_DURATION, PROBE_DURATION, MASKER_LEVEL,
                 PROBE_LEVEL, 'target_masker_tone_frequency',
                 'target_probe_tone_frequency', 'target_masker_tone_polarity',
                 'target_probe_tone_polarity', 'target_masker_rise_time'):
        assert name in names

    context = {n: p.default for n, p in names.items()}
    context.update({
        GAP: 5e-3, MASKER_DURATION: MASKER_DUR, PROBE_DURATION: PROBE_DUR,
        'target_masker_rise_time': RISE, 'target_probe_rise_time': RISE,
        'target_masker_envelope': 'cosine-squared',
        'target_probe_envelope': 'cosine-squared',
        'target_masker_tone_frequency': 4000, MASKER_LEVEL: 80,
        'target_probe_tone_frequency': 8000, PROBE_LEVEL: 60,
        'target_masker_tone_polarity': -1, 'target_probe_tone_polarity': -1,
    })
    f = initialize_factory(output, token, context)
    np.testing.assert_allclose(f.get_samples_remaining(),
                               factory(5e-3, polarity=-1)
                               .get_samples_remaining())
