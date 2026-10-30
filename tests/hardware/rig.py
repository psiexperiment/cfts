'''
The Fireface test rig: which IO manifest the paradigms run on and which
channel is which.

The paradigms run on `cftscal.io.dummy_fireface.IOManifest`, which names
every channel cfts looks for. Which output reaches which input is decided
in TotalMix, not in the manifest; `psi.loopback.rig.measure_routing` finds
out before any paradigm runs.
'''
from psi.loopback.rig import Rig


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
#: `dummy_fireface`.
INPUT_DBFS = 19

#: The sound card engine's defaults, which `dummy_fireface` keeps.
ENGINE_FS = 96000
ENGINE_BLOCKSIZE = 8192

RIG = Rig(
    io_manifest=IO_MANIFEST,
    device=DEVICE_NAME,
    outputs=OUTPUTS,
    inputs=INPUTS,
    input_dbfs=INPUT_DBFS,
    engine_fs=ENGINE_FS,
    engine_blocksize=ENGINE_BLOCKSIZE,
)
