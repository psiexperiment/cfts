'''
Analysis for contralateral suppression of DPOAEs by the medial olivocochlear
(MOC) reflex.

The primaries play continuously in the probe ear while the elicitor (noise in
the opposite ear) plays as back-to-back trials of baseline, elicitor and
recovery. Each trial of the probe microphone is cut into blocks of
`response_window` starting at the trial's onset, and each block gives the
complex amplitude of f1, f2, the DP (2f1 - f2) and four noise-floor bins on
either side of the DP. Phase is referenced to the start of acquisition rather
than to the block, so blocks of the continuous primaries (and DP) average
coherently across trials no matter where the trials fall.

Kept free of psi so it can be tested on its own; `moc.enaml` wires it into
the paradigm.
'''
import logging

import numpy as np

log = logging.getLogger(__name__)


#: Offsets of the noise-floor bins from the DP, in multiples of the frequency
#: resolution (1 / response window).
NF_BINS = (-2, -1, 1, 2)


def moc_frequencies(f1, f2, resolution):
    '''
    f1, f2, the DP and the noise-floor bins around the DP.
    '''
    dp = 2 * f1 - f2
    return np.array([f1, f2, dp] + [dp + k * resolution for k in NF_BINS])


def tone_phasors(x, s0, fs, frequencies):
    '''
    Complex peak amplitude of each of `frequencies` in `x`, whose first sample
    is sample `s0` of the acquisition. Phase is re a cosine starting at sample
    0 of the acquisition.
    '''
    t = (s0 + np.arange(x.shape[-1])) / fs
    basis = np.exp(-2j * np.pi * np.asarray(frequencies)[:, np.newaxis] * t)
    return 2 * (basis @ x) / x.shape[-1]


def phasor_levels(phasors, frequencies, calibration):
    '''
    Levels of f1, f2 and the DP, and the noise floor (power average of the
    noise-floor bins, at the DP frequency), from `phasors` as returned by
    `tone_phasors`.
    '''
    rms = np.abs(phasors) / np.sqrt(2)
    f1, f2, dp = calibration.get_db(frequencies[:3], rms[:3])
    nf_rms = np.sqrt(np.mean(rms[3:] ** 2))
    nf = calibration.get_db(frequencies[2], nf_rms)
    return float(f1), float(f2), float(dp), float(nf)


class MOCAnalyzer:
    '''
    Accumulates the probe microphone into per-trial blocks and the average
    across accepted trials.

    Parameters
    ----------
    fs : float
        Sampling rate of the probe microphone.
    f1, f2 : float
        Primary frequencies (Hz). Should be whole multiples of 1 /
        `response_window` so that each block holds whole cycles.
    response_window : float
        Duration (s) of each block. This is the time resolution.
    trial_duration : float
        Duration (s) of each trial. Any part shorter than a block at the end
        of the trial is not analyzed.
    baseline : (float, float)
        Window (s re trial onset) used as the trial's baseline.
    elicitor : (float, float)
        Window (s re trial onset) over which suppression is measured.
    calibration : psiaudio calibration
        Probe microphone calibration.
    requested_n : int
        Trials to accept. Trials after these (which the queue may already
        have played) are ignored.
    trials : deque
        Information on each trial (as notified by the elicitor queue, with
        `t0` in seconds re acquisition start), appended as they are queued.
    block_cb : callable
        Called with a list of one record (dict) per block, as soon as the
        block is acquired.
    reject_cb : callable
        Called at the end of each trial with the trial's info and summary
        (dict). Returns the reason to reject the trial, or None to accept it.
    trial_cb : callable
        Called with the trial's info and summary once it has been accepted or
        rejected (the summary then includes the average of the accepted
        trials, prefixed by `avg_`).
    discard : float
        Blocks starting less than this (s) after the first trial's onset are
        not analyzed (the primaries start with the first trial and are still
        ramping on).
    '''

    def __init__(self, fs, f1, f2, response_window, trial_duration,
                 baseline, elicitor, calibration, requested_n, trials,
                 block_cb, reject_cb, trial_cb, discard=0):
        self.fs = fs
        self.response_window = response_window
        self.block_n = int(round(response_window * fs))
        self.n_blocks = int(np.floor(trial_duration / response_window + 1e-9))
        self.frequencies = moc_frequencies(f1, f2, 1 / response_window)
        self.calibration = calibration
        self.requested_n = requested_n
        self.trials = trials
        self.block_cb = block_cb
        self.reject_cb = reject_cb
        self.trial_cb = trial_cb
        self.discard = discard
        self.discard_until = None

        # Center of each block re trial onset. The windows include only the
        # blocks that lie entirely within them.
        self.time = (np.arange(self.n_blocks) + 0.5) * response_window
        self.baseline_mask = self._window_mask(*baseline)
        self.elicitor_mask = self._window_mask(*elicitor)

        n_f = len(self.frequencies)
        self.sum = np.zeros((self.n_blocks, n_f), dtype=complex)
        self.count = np.zeros(self.n_blocks, dtype=int)
        self.n_accepted = 0
        self.n_trials = 0

        self._buffer = np.zeros(0)
        self._buffer_s0 = None
        self._trial = None

    def _window_mask(self, lb, ub):
        # Small tolerance so a window edge on a block edge includes it.
        eps = 1e-9
        half = self.response_window / 2
        return (self.time - half >= lb - eps) & (self.time + half <= ub + eps)

    def send(self, data):
        x = np.asarray(data)
        if x.ndim > 1:
            x = x[0]
        if self._buffer_s0 is None:
            self._buffer_s0 = data.s0
        expected = self._buffer_s0 + len(self._buffer)
        if data.s0 != expected:
            log.warning('MOC analysis expected sample %d but got %d',
                        expected, data.s0)
            self._buffer = np.zeros(0)
            self._buffer_s0 = data.s0
        self._buffer = np.concatenate((self._buffer, x))
        self._process()

    def _start_trial(self):
        if not self.trials:
            return False
        info = self.trials.popleft()
        if self.discard_until is None:
            self.discard_until = info['t0'] + self.discard
        n = len(self.frequencies)
        self._trial = {
            'info': info,
            'index': self.n_trials,
            's0': int(round(info['t0'] * self.fs)),
            'block': 0,
            'phasors': np.full((self.n_blocks, n), np.nan, dtype=complex),
            'nf': np.full(self.n_blocks, np.nan),
            # Trials the queue played after we had all we need.
            'ignore': self.n_accepted >= self.requested_n,
        }
        self.n_trials += 1
        return True

    def _process(self):
        while True:
            if self._trial is None and not self._start_trial():
                # Nothing to analyze until the queue tells us about the next
                # trial; keep a little in case that notification is late.
                keep = int(10 * self.fs)
                if len(self._buffer) > keep:
                    drop = len(self._buffer) - keep
                    self._buffer = self._buffer[drop:]
                    self._buffer_s0 += drop
                return

            trial = self._trial
            lb = trial['s0'] + trial['block'] * self.block_n
            ub = lb + self.block_n
            if ub > self._buffer_s0 + len(self._buffer):
                return
            if lb >= self._buffer_s0:
                i = lb - self._buffer_s0
                self._analyze_block(trial, self._buffer[i:i + self.block_n],
                                    lb)
                self._buffer = self._buffer[i + self.block_n:]
                self._buffer_s0 = ub
            else:
                log.warning('MOC analysis missed block %d of trial %d',
                            trial['block'], trial['index'])
            trial['block'] += 1
            if trial['block'] == self.n_blocks:
                self._end_trial(trial)
                self._trial = None

    def _analyze_block(self, trial, x, s0):
        k = trial['block']
        if trial['ignore'] or (s0 / self.fs) < self.discard_until:
            return
        p = tone_phasors(x, s0, self.fs, self.frequencies)
        trial['phasors'][k] = p
        f1, f2, dp, nf = phasor_levels(p, self.frequencies, self.calibration)
        trial['nf'][k] = nf

        # Average of the accepted trials plus this one.
        avg = (self.sum[k] + p) / (self.count[k] + 1)
        a_f1, a_f2, a_dp, a_nf = phasor_levels(avg, self.frequencies,
                                               self.calibration)
        record = {
            'trial': trial['index'],
            'block': k,
            'time': self.time[k],
            'ts': s0 / self.fs,
            'elicitor': bool(self.elicitor_mask[k]),
            'f1_level': f1,
            'f2_level': f2,
            'dpoae_level': dp,
            'dpoae_noise_floor': nf,
            'dpoae_real': p[2].real,
            'dpoae_imag': p[2].imag,
            'avg_f1_level': a_f1,
            'avg_f2_level': a_f2,
            'avg_dpoae_level': a_dp,
            'avg_dpoae_noise_floor': a_nf,
            'n_average': int(self.count[k] + 1),
        }
        self.block_cb([record])

    def window_levels(self, phasors, mask):
        '''
        Levels from the coherent average of `phasors` over the blocks in
        `mask` (ignoring blocks that weren't analyzed).
        '''
        p = phasors[mask]
        p = p[~np.isnan(p).any(axis=-1)]
        if len(p) == 0:
            return (np.nan,) * 4
        return phasor_levels(p.mean(axis=0), self.frequencies,
                             self.calibration)

    def summarize(self, phasors):
        b_f1, b_f2, b_dp, b_nf = self.window_levels(phasors,
                                                    self.baseline_mask)
        e_f1, e_f2, e_dp, e_nf = self.window_levels(phasors,
                                                    self.elicitor_mask)
        return {
            'baseline_dpoae_level': b_dp,
            'baseline_noise_floor': b_nf,
            'elicitor_dpoae_level': e_dp,
            'elicitor_noise_floor': e_nf,
            'suppression': b_dp - e_dp,
            # A change in the primaries during the elicitor suggests the
            # middle-ear muscle reflex.
            'f1_change': e_f1 - b_f1,
            'f2_change': e_f2 - b_f2,
        }

    def average_phasors(self):
        with np.errstate(invalid='ignore'):
            return self.sum / self.count[:, np.newaxis]

    def _end_trial(self, trial):
        if trial['ignore']:
            return
        summary = {'trial': trial['index'], 't0': trial['info']['t0']}
        summary.update(self.summarize(trial['phasors']))
        nf = trial['nf'][self.baseline_mask]
        summary['max_block_noise_floor'] = \
            np.nanmax(nf) if not np.isnan(nf).all() else np.nan
        reason = self.reject_cb(trial['info'], summary)
        summary['accepted'] = reason is None
        summary['reject_reason'] = reason or ''
        if reason is None:
            valid = ~np.isnan(trial['phasors']).any(axis=-1)
            self.sum[valid] += trial['phasors'][valid]
            self.count[valid] += 1
            self.n_accepted += 1
        summary['n_accepted'] = self.n_accepted
        average = self.summarize(self.average_phasors())
        summary.update({f'avg_{k}': v for k, v in average.items()})
        self.trial_cb(trial['info'], summary)
