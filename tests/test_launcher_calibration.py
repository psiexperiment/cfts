'''
Which calibrations the cfts launcher accepts for a starship (from its
microphone's entry under File > Settings > Input Channels) and a speaker
(from its entry under Outputs).
'''
import json
from types import SimpleNamespace

import enaml
import pytest

from cftscal.objects import (
    NominalInputCalibration, NominalSpeakerCalibration, speaker_manager,
    starship_manager,
)
from cftscal.plugins.settings import (
    MultiTypeSensorReference, SpeakerCalibrationReference,
)

with enaml.imports():
    from cfts.app.exp_launcher_gui import Settings


def resolve(device_type, sensitivity=1.0):
    sensor = MultiTypeSensorReference(device_type=device_type,
                                      sensitivity=sensitivity)
    settings = SimpleNamespace(
        monitor_input_sensors={'starship_K_microphone': sensor},
        output_sensors={},
    )
    return Settings.equipment_calibration(settings, 'starship_K', 'Starship')


def test_unconfigured_starship_rejected():
    # Unity is what an input channel nobody has configured defaults to.
    assert resolve('Unity') is None
    assert resolve('Meas. Mic.') is None


def test_nominal_starship_accepted():
    # E.g., the digital loopback in cftscal.io.keyboard, which has no
    # starship calibration on file.
    sensor = resolve('Nominal', 50)
    assert sensor is not None
    # The paradigm loads it from the string the launcher passes on.
    cal_str = sensor.get_calibration().to_string()
    cal = starship_manager.from_string(cal_str)
    assert isinstance(cal, NominalInputCalibration)
    assert cal.sensitivity == 50


def test_nominal_starship_needs_sensitivity():
    assert resolve('Nominal', 0) is None


def resolve_speaker(device_type, sensitivity=110):
    sensor = SpeakerCalibrationReference(device_type=device_type,
                                         sensitivity=sensitivity)
    settings = SimpleNamespace(monitor_input_sensors={},
                               output_sensors={'speaker_1': sensor})
    return Settings.equipment_calibration(settings, 'speaker_1', 'Speaker')


def test_uncalibrated_speaker_rejected():
    # No speaker calibration picked.
    assert resolve_speaker('Speaker') is None
    assert resolve_speaker('Nominal', 0) is None


def test_nominal_speaker_accepted():
    # E.g., the sound card outputs in cftscal.io.keyboard.
    sensor = resolve_speaker('Nominal', 120)
    assert sensor is not None
    cal_str = sensor.get_calibration().to_string()
    cal = speaker_manager.from_string(cal_str)
    assert isinstance(cal, NominalSpeakerCalibration)
    assert cal.spl == 120


def test_hardware_config_with_starship_output_loads(monkeypatch, tmp_path):
    # Outputs used to offer starship calibrations. A hardware config saved
    # with one still loads, leaving that output for the user to set.
    monkeypatch.setenv('CFTSCAL_IO', 'cftscal.io.keyboard.IOManifest')
    filename = tmp_path / 'hardware.json'
    filename.write_text(json.dumps({'output_sensors': {
        'speaker_1': {'name': 'A/B', 'device_type': 'Starship',
                      'sensitivity': 1.0},
        'speaker_2': {'name': '', 'device_type': 'Nominal',
                      'sensitivity': 120.0},
    }}))
    settings = Settings()
    settings.load_hardware_config(filename)
    speaker_1 = settings.output_sensors['speaker_1']
    assert (speaker_1.device_type, speaker_1.name) == ('Speaker', '')
    assert settings.equipment_calibration('speaker_1', 'Speaker') is None
    speaker_2 = settings.equipment_calibration('speaker_2', 'Speaker')
    assert speaker_2.get_calibration().spl == 120
