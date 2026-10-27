'''
The Fireface test rig: which IO manifest the paradigms run on, and a probe
that measures how TotalMix currently routes outputs back to inputs.

The paradigms run on `cftscal.io.dummy_fireface.IOManifest`, which names
every channel cfts looks for. Which output reaches which input is decided
in TotalMix, not in the manifest, so before any paradigm runs the probe
plays independent noise on every output and finds where each one comes back,
at what gain and with what delay. Tests then check that the routes they need
exist, and use the measured gain as the reference for level checks.
'''
from dataclasses import dataclass, field

import numpy as np
from scipy import signal


#: `dummy_fireface` plus a stationary turntable (see `rig_io.enaml`).
IO_MANIFEST = 'rig_io.IOManifest'
DEVICE_NAME = 'ASIO Fireface USB'

#: Channel numbers in `cftscal.io.dummy_fireface` (0-based).
OUTPUTS = {
    'starship_A_primary': 0,
    'starship_A_secondary': 1,
    'starship_B_primary': 2,
    'starship_B_secondary': 3,
    'speaker_1': 4,
    'speaker_2': 5,
    'ir_emitter': 6,
}

INPUTS = {
    'starship_A_microphone': 0,
    'starship_B_microphone': 1,
    'eeg': 2,
    'cal_microphone': 3,
    'microphone_1': 4,
    'loopback_1': 5,
    'output_monitor': 6,
    'temperature': 7,
}

#: Full-scale input level of the manifest's inputs, `Input.dBFS` in
#: `dummy_fireface`. psi scales the samples it reads by this, while outputs
#: are written unscaled, so a digital loopback at unity reads back this much
#: louder than it was played.
INPUT_DBFS = 19

#: Probe signal level (peak, re full scale). Low enough to be harmless if
#: something is plugged into an output.
PROBE_AMPLITUDE = 0.01

#: Gain below which an output is treated as not reaching an input.
ROUTE_THRESHOLD = 0.05

#: Longest round-trip delay through the sound card to allow for, in seconds.
MAX_LATENCY = 0.5

#: The sound card engine's defaults, which `dummy_fireface` keeps.
ENGINE_FS = 96000
ENGINE_BLOCKSIZE = 8192


@dataclass
class Route:
    output: str
    input: str
    #: Linear gain from the digital value written to the value read.
    gain: float
    #: Delay through the sound card, in samples.
    lag: int

    @property
    def gain_db(self):
        return 20 * np.log10(abs(self.gain))


@dataclass
class Routing:
    fs: float
    routes: list = field(default_factory=list)
    #: gain[input, output], including routes below threshold.
    gain: np.ndarray = None

    def get(self, output, input):
        for r in self.routes:
            if r.output == output and r.input == input:
                return r
        return None

    def sources(self, input):
        return [r.output for r in self.routes if r.input == input]

    def missing(self, required):
        '''
        Routes in `required` ({input: [outputs]}) that TotalMix isn't set up
        for, as readable strings.
        '''
        return [f'{o} -> {i}' for i, outputs in required.items()
                for o in outputs if self.get(o, i) is None]


def measure_routing(fs=96000, duration=1, device=DEVICE_NAME):
    '''
    Play independent noise on every output in `OUTPUTS` and find which
    inputs it reaches.
    '''
    import sounddevice as sd

    n = int(fs * duration)
    n_ch = max(max(OUTPUTS.values()), max(INPUTS.values())) + 1
    rng = np.random.default_rng(0)
    noise = rng.uniform(-1, 1, (n, len(OUTPUTS))).astype('float32')
    noise *= PROBE_AMPLITUDE
    # Keep recording after the noise ends so all of it comes back despite the
    # round-trip delay; otherwise the gain reads low by the fraction missed.
    played = np.zeros((n + int(fs * MAX_LATENCY), n_ch), 'float32')
    for i, ch in enumerate(OUTPUTS.values()):
        played[:n, ch] = noise[:, i]
    mapping = list(range(1, n_ch + 1))
    recorded = sd.playrec(played, fs, input_mapping=mapping,
                          output_mapping=mapping, device=device,
                          blocking=True)

    routing = Routing(fs=fs)
    routing.gain = np.zeros((len(INPUTS), len(OUTPUTS)))
    out_names = list(OUTPUTS)
    for i, (in_name, in_ch) in enumerate(INPUTS.items()):
        x = recorded[:, in_ch]
        lags = {}
        for o, out_name in enumerate(out_names):
            c = signal.correlate(x, noise[:, o], 'full', 'fft')
            peak = np.argmax(np.abs(c))
            routing.gain[i, o] = c[peak] / np.sum(noise[:, o] ** 2)
            if abs(routing.gain[i, o]) > ROUTE_THRESHOLD:
                lags[o] = peak - (n - 1)
        if not lags:
            continue
        # Where several outputs sum into one input, each one's correlation
        # also picks up the others by chance, so fit them jointly for the
        # gains.
        sources = np.zeros((len(x), len(lags)))
        for j, (o, lag) in enumerate(lags.items()):
            sources[lag:lag + n, j] = noise[:, o]
        gains = np.linalg.lstsq(sources, x, rcond=None)[0]
        for (o, lag), gain in zip(lags.items(), gains):
            routing.gain[i, o] = gain
            routing.routes.append(Route(out_names[o], in_name, gain, lag))
    return routing


def measure_latency(output, input, fs=ENGINE_FS, blocksize=ENGINE_BLOCKSIZE,
                    duration=1):
    '''
    Samples from when psi writes a sample to `output` to when it reads it
    back on `input`, which is what the engine's `latency_samples` has to be.

    This goes through psi's own `PlayRec` rather than `sd.playrec`, since the
    delay depends on how the stream is opened and buffered (sampling rate,
    blocksize, and PlayRec dropping the first input block), and on the ASIO
    buffer size set in the Fireface settings.
    '''
    import time
    from psi.controller.engines.soundcard.playrec import PlayRec

    rng = np.random.default_rng(0)
    probe = rng.uniform(-1, 1, int(fs * 0.05)) * PROBE_AMPLITUDE
    recorded = []

    def ao_cb(s0, samples, buffer):
        if s0 < len(probe):
            chunk = probe[s0:s0 + samples]
            buffer[0, :len(chunk)] = chunk

    def ai_cb(s0, samples, data):
        recorded.append(data[0])

    stream = PlayRec(fs, DEVICE_NAME, [INPUTS[input]], [OUTPUTS[output]],
                     ai_cb, ao_cb, blocksize)
    stream.start()
    time.sleep(duration + MAX_LATENCY)
    stream.stop()

    x = np.concatenate(recorded)
    c = signal.correlate(x, probe, 'full', 'fft')
    peak = np.argmax(np.abs(c))
    gain = c[peak] / np.sum(probe ** 2)
    if abs(gain) < ROUTE_THRESHOLD:
        raise ValueError(f'{output} does not reach {input}')
    return int(peak - (len(probe) - 1))
