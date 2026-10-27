'''
EFR (SAM and RAM tones) through the loopback: the EEG input records the
Starship's primary output, so every tone psi logs should be in the EEG as
specified, separated by the jittered gaps the paradigm draws, with
alternating polarity cancelling.
'''
import numpy as np
import pytest

from cftsdata.efr import EFR

from abr_analysis import analyze_stimuli, check_sequence
from checks import ram_tone, sam_tone
from runner import OUTPUT_CAL_DB
from test_abr import ABRTests, run_eeg_paradigm, stimulus_table, \
    write_abr_report


AVERAGES = 4
DURATION = 0.2
ISI = 0.1
ISI_JITTER = 0.02
FM = 110
FC = 8000
LEVEL = 70
DUTY_CYCLE = 0.25

#: Seed of the gaps between tokens (`efr_epochs.rng_generator`).
ISI_SEED = 123456789


def load_efr(path):
    data = EFR(path)
    md = data.analyze_efr_metadata.reset_index(drop=True)
    return data, md, data.eeg


def check_gaps(checks, md, fs, n_token):
    '''
    The silence after each token is drawn, in the order played, from
    `RandomState(ISI_SEED).uniform(isi, isi + isi_jitter)`, so every interval
    between onsets is known to the sample.
    '''
    onsets = np.round(md.t0.values * fs).astype(int)
    rng = np.random.RandomState(ISI_SEED)
    gaps = rng.uniform(ISI, ISI + ISI_JITTER, len(onsets) - 1)
    expected = n_token + np.round(gaps * fs)
    error = np.diff(onsets) - expected
    worst = np.abs(error).max()
    checks.add('repetition rate', worst <= 1,
               f'worst interval off by {worst:.0f} samples',
               f'token + seeded gap ({ISI * 1e3:g}–'
               f'{(ISI + ISI_JITTER) * 1e3:g} ms) ± 1 sample')


def analyze_efr(run, templates, level_name, title):
    fs, md = run.fs, run.md
    stimuli = stimulus_table(md, fs).assign(level=md[level_name])
    a = analyze_stimuli(run.eeg, fs, stimuli, templates, ['fc', 'fm'],
                        run.epoch_n, run.route_gain_db, run.latency,
                        run.engine)
    check_gaps(a.checks, md, fs, len(templates[0]))
    check_sequence(a.checks, stimuli, ['fc', 'fm'], 1, AVERAGES)
    write_abr_report(run, title, stimuli, templates, a, [], ['fc'])
    return a.checks


def epoch_settings(**parameters):
    return {'parameters': {
        'averages': AVERAGES,
        'duration': DURATION,
        'isi': ISI,
        'isi_jitter': ISI_JITTER,
        'fm': FM,
        'fc': FC,
        'alternate_polarity': True,
        **parameters,
    }}


@pytest.fixture(scope='module')
def efr_sam_epoch(routing, results_dir):
    settings = epoch_settings(target_sam_tone_level=LEVEL)
    run = run_eeg_paradigm('efr_sam_epoch', 'efr_sam_epoch', settings,
                           routing, results_dir, load_efr)
    templates = [
        sam_tone(run.fs, r.fc, r.fm, r.target_sam_tone_level, r.duration,
                 r.target_sam_tone_rise_time, r.target_sam_tone_depth,
                 r.polarity, r.target_sam_tone_equalize, cal_db=OUTPUT_CAL_DB)
        for r in run.md.itertuples()
    ]
    return analyze_efr(run, templates, 'target_sam_tone_level',
                       'SAM EFR (epoch)')


@pytest.fixture(scope='module')
def efr_ram_epoch(routing, results_dir):
    settings = epoch_settings(target_tone_level=LEVEL, duty_cycle=DUTY_CYCLE)
    run = run_eeg_paradigm('efr_ram_epoch', 'efr_ram_epoch', settings,
                           routing, results_dir, load_efr)
    templates = [
        ram_tone(run.fs, r.fc, r.fm, r.target_tone_level,
                 r.target_ram_tone_duration, r.target_mod_duty_cycle,
                 r.target_mod_alpha, r.target_mod_depth, r.target_tone_phase,
                 r.polarity, cal_db=OUTPUT_CAL_DB)
        for r in run.md.itertuples()
    ]
    return analyze_efr(run, templates, 'target_tone_level',
                       'RAM EFR (epoch)')


class TestSAMEpoch(ABRTests):
    @pytest.fixture
    def checks(self, efr_sam_epoch):
        return efr_sam_epoch


class TestRAMEpoch(ABRTests):
    @pytest.fixture
    def checks(self, efr_ram_epoch):
        return efr_ram_epoch

