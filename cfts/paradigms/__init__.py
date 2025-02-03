import numpy as np
from psi.experiment.api import ParadigmDescription


CAL_PATH = 'cftscal.paradigms.'
CFTS_PATH = 'cfts.paradigms.'
CORE_PATH = 'psi.paradigms.core.'


COMMON_PLUGINS = [
    {'manifest': CFTS_PATH + 'metadata_mixins.MetadataManifest', 'required': True},
    {'manifest': CORE_PATH + 'websocket_mixins.WebsocketClientManifest', 'required': True},
    {'manifest': CFTS_PATH + 'cfts_mixins.DebugCalibration', 'required': False, 'info': {'hide': True}},
    {'manifest': CFTS_PATH + 'cfts_mixins.CalibrationMicrophone', 'required': False, 'info': {'hide': True}},
    {'manifest': CFTS_PATH + 'cfts_mixins.OutputMonitor', 'required': False, 'info': {'hide': True}},
    {'manifest': CFTS_PATH + 'cfts_mixins.OutputMonitorView', 'required': False, 'info': {'hide': True}},
    {'manifest': CFTS_PATH + 'cfts_mixins.SwapOutputs', 'required': False, 'info': {'hide': True}},
    {'manifest': CFTS_PATH + 'video.PSIVideo', 'required': False},
]


################################################################################
# Single-starship paradigms (ABR, EFR, DPOAE, IEC)
################################################################################
eeg_dec_mixin = {
    'manifest': CAL_PATH + 'objects.InputAmplifier',
    'required': True,
    'attrs': {'id': 'eeg', 'title': 'EEG', 'target_fs': 25e3},
}


eeg_mixin = {
    'manifest': CAL_PATH + 'objects.InputAmplifier',
    'required': True,
    'attrs': {'id': 'eeg', 'title': 'EEG', 'target_fs': None},
}


selectable_starship_mixin = {
    'manifest': CAL_PATH + 'objects.Starship',
    'required': True,
    'attrs': {'id': 'system', 'title': 'Starship', 'output_mode': 'select'}
}


dual_starship_mixin = {
    'manifest': CAL_PATH + 'objects.Starship',
    'required': True,
    'attrs': {'id': 'system', 'title': 'Starship', 'output_mode': 'dual'}
}



microphone_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'microphone_signal_view',
        'title': 'Microphone (time)',
        'time_span': 4,
        'time_delay': 0.125,
        'source_name': 'system_microphone',
        'y_label': 'Microphone (V)'
    },
}


microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'microphone_fft_view',
        'title': 'Microphone (PSD)',
        'fft_time_span': 0.25,
        'fft_freq_lb': 500,
        'fft_freq_ub': 50000,
        'source_name': 'system_microphone',
        'y_label': 'Microphone (dB)',
        'apply_calibration': True,
    }
}


eeg_view_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'eeg_view',
        'title': 'EEG display',
        'time_span': 2,
        'time_delay': 0.125,
        'source_name': 'eeg_filtered',
        'y_label': 'EEG (V)'
    }
}
eeg_view_mixin_required = eeg_view_mixin.copy()
eeg_view_mixin_required['required'] = True


eeg_fft_view_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'eeg_fft_view',
        'title': 'EEG PSD',
        'fft_time_span': 1,
        'fft_freq_lb': 10,
        'fft_freq_ub': 10000,
        'source_name': 'eeg',
        'y_label': 'EEG (dB re 1Vrms)',
        # Show 60 Hz plus harmonics.
        'vlines': [
            {
                'position': 60,
                'color': 'LightPink',
                'width': 5,
            },
            {
                'position': 120,
                'color': 'LightPink',
                'width': 5,
            },
            {
                'position': 180,
                'color': 'LightPink',
                'width': 5,
            }
        ]
    }
}
eeg_fft_view_mixin_required = eeg_fft_view_mixin.copy()
eeg_fft_view_mixin_required['required'] = True


temperature_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
    'required': False,
    'info': {'hide': True},
}


turntable_mixin = {
    'manifest': CFTS_PATH + 'turntable.TurntableManifest',
    'required': True,
}


turntable_angular_velocity_view_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'turntable_angular_velocity_view_mixin',
        'title': 'Turntable (Ω⃗)',
        'time_span': 30,
        'time_delay': 0.125,
        'source_name': 'turntable_angular_velocity',
        'y_label': 'Turntable velocity (radians/sec)',
        'y_min': -np.pi,
        'y_max': np.pi,
        'decimate_mode': 'none',
    }
}
turntable_angular_velocity_view_mixin_required = turntable_angular_velocity_view_mixin.copy()
turntable_angular_velocity_view_mixin_required['required'] = True


turntable_linear_velocity_view_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'turntable_linear_velocity_view_mixin',
        'title': 'Turntable',
        'time_span': 30,
        'time_delay': 0.125,
        'source_name': 'turntable_linear_velocity',
        'y_label': 'Turntable velocity (cm/sec)',
        'y_min': -5,
        'y_max': 5,
        'decimate_mode': 'none',
    }
}
turntable_linear_velocity_view_mixin_required = turntable_linear_velocity_view_mixin.copy()
turntable_linear_velocity_view_mixin_required['required'] = True


################################################################################
# Paradigm descriptions
################################################################################
ParadigmDescription(
    # This is for monitoring the incoming EEG trace
    'monitor', 'EEG Monitor', 'ear', [
        temperature_mixin,
        eeg_dec_mixin,
        eeg_view_mixin_required,
        eeg_fft_view_mixin_required,
        {'manifest': CFTS_PATH + 'monitor.MonitorManifest', 'selected': True},
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


################################################################################
ParadigmDescription(
    # This is for testing the turntable
    'turntable_monitor', 'Turntable Monitor', 'ear', [
        turntable_angular_velocity_view_mixin_required,
        turntable_linear_velocity_view_mixin_required,
        turntable_mixin,
        {'manifest': CFTS_PATH + 'monitor.MonitorManifest', 'selected': True},
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io', 'ABR (tone)', 'ear', [
        eeg_dec_mixin,
        selectable_starship_mixin,
        {'manifest': CFTS_PATH + 'abr_io.ABRIOToneSimpleManifest'},
        temperature_mixin,
        eeg_view_mixin,
        eeg_fft_view_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest',
         'info': {'hide': True}
         },
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io_click', 'ABR (click)', 'ear', [
        eeg_dec_mixin,
        selectable_starship_mixin,
        {'manifest': CFTS_PATH + 'abr_io.ABRIOClickSimpleManifest'},
        temperature_mixin,
        eeg_view_mixin,
        eeg_fft_view_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRClickInEarCalibrationMixinManifest',
         'info': {'hide': True}
         },
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)



ParadigmDescription(
    'dpoae_io', 'DPOAE', 'ear', [
        dual_starship_mixin,
        {'manifest': CFTS_PATH + 'dpoae_base.SingleDPOAEManifest'},
        {'manifest': CFTS_PATH + 'dpoae_io.DPOAEIOSimpleManifest', 'required': True},
        {'manifest': CFTS_PATH + 'dpoae_io.SingleDPOAEIO', 'required': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest',
         'info': {'hide': True}},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    'dpgram', 'DPgram', 'ear', [
        dual_starship_mixin,
        {'manifest': CFTS_PATH + 'dpoae_base.SingleDPOAEManifest'},
        {'manifest': CFTS_PATH + 'dpgram.DPGramSimpleManifest', 'required': True},
        {'manifest': CFTS_PATH + 'dpgram.SingleDPGram', 'required': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest',
         'info': {'hide': True}},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    'efr_sam', 'SAM EFR (continuous)', 'ear', [
        selectable_starship_mixin,
        eeg_mixin,
        {'manifest': CFTS_PATH + 'efr.SAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.SAMEFRInEarCalibrationMixinManifest',
         'info': {'hide': True}},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        eeg_view_mixin,
        eeg_fft_view_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    'efr_sam_epoch', 'SAM EFR (continuous)', 'ear', [
        selectable_starship_mixin,
        eeg_mixin,
        {'manifest': CFTS_PATH + 'efr_epochs.SAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.SAMEFRInEarCalibrationMixinManifest',
         'info': {'hide': True}},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        eeg_view_mixin,
        eeg_fft_view_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    'efr_sam_epoch', 'SAM EFR (epoch)', 'ear', [
        selectable_starship_mixin,
        eeg_mixin,
        {'manifest': CFTS_PATH + 'efr_epochs.SAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.SAMEFRInEarCalibrationMixinManifest',
         'info': {'hide': True}},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        eeg_view_mixin,
        eeg_fft_view_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    'efr_ram_epoch', 'RAM EFR (epoch)', 'ear', [
        selectable_starship_mixin,
        eeg_mixin,
        {'manifest': CFTS_PATH + 'efr_epochs.RAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.RAMEFRInEarCalibrationMixinManifest',
         'info': {'hide': True}},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
        eeg_view_mixin,
        eeg_fft_view_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'IEC (chirp)', 'ear', [
        selectable_starship_mixin,
        {
            'manifest': CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest',
            'attrs': {'mic_source_name': 'system_microphone'},
        },
        {'manifest': CAL_PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': CAL_PATH + 'calibration_mixins.ToneValidateMixin'},
    ] + COMMON_PLUGINS,
    info={'modes': ['run']},
)


################################################################################
# Two-starship paradigms for MEMR
################################################################################
multi_microphone_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.MultiSignalViewManifest',
    'attrs': {
        'id': 'microphone_signal_view',
        'title': 'Microphone (time)',
        'time_span': 4,
        'time_delay': 0.125,
        'sources': {
            'left_microphone': {'color': 'DarkCyan'},
            'right_microphone': {'color': 'DarkMagenta'}
        },
        'y_label': 'Microphone (V)'
    },
}


multi_microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.MultiSignalFFTViewManifest',
    'selected': True,
    'attrs': {
        'id': 'microphone_fft_view',
        'title': 'Microphone (PSD)',
        'fft_time_span': 0.25,
        'fft_freq_lb': 500,
        'fft_freq_ub': 50000,
        'sources': {
            'left_microphone': {'color': 'DarkCyan', 'apply_calibration': True},
            'right_microphone': {'color': 'DarkMagenta', 'apply_calibration': True}
        },
        'y_label': 'Microphone (dB)'
    }
}


multi_memr_microphone_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.MultiSignalViewManifest',
    'selected': True,
    'attrs': {
        'id': 'memr_microphone_signal_view',
        'title': 'Microphone (time)',
        'time_span': 4,
        'time_delay': 0.125,
        'sources': {
            'elicitor_microphone': {'color': 'DarkCyan'},
            'probe_microphone': {'color': 'DarkMagenta'}
        },
        'y_label': 'Microphone (V)'
    },
}


multi_memr_microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.MultiSignalFFTViewManifest',
    'attrs': {
        'id': 'memr_microphone_signal_fft_view',
        'title': 'Microphone (PSD)',
        'fft_time_span': 0.25,
        'fft_freq_lb': 500,
        'fft_freq_ub': 50000,
        'y_min': -10,
        'y_max': 100,
        'y_mode': 'mouse',
        'save_limits': True,
        'sources': {
            'probe_microphone': {'color': 'DarkMagenta', 'apply_calibration': True},
            'elicitor_microphone': {'color': 'DarkCyan', 'apply_calibration': True},
        },
        'y_label': 'Signal (dB)'
    }
}

ParadigmDescription(
    'dual_dpoae_io', 'Dual DPOAE (input-output)', 'ear', [
        {'manifest': CAL_PATH + 'objects.Starship', 'required': True,
         'attrs': {'id': 'left', 'title': 'Left starship', 'side': 'left'}},
        {'manifest': CAL_PATH + 'objects.Starship', 'required': True,
         'attrs': {'id': 'right', 'title': 'Right starship', 'side': 'right'}},
        {'manifest': CFTS_PATH + 'dpoae_base.DualDPOAEManifest'},
        {'manifest': CFTS_PATH + 'dpoae_io.DualDPOAEIOSimpleManifest', 'required': True},
        {'manifest': CFTS_PATH + 'dpoae_io.DualDPOAEIO'},
        {'manifest': CFTS_PATH + 'cfts_mixins.BinauralDPOAEInEarCalibrationMixinManifest', 'info': {'hide': True}},
        temperature_mixin,
        multi_microphone_mixin,
        multi_microphone_fft_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['binaural']},
)


elicitor_starship_mixin = {
    'manifest': CAL_PATH + 'objects.Starship',
    'required': True,
    'attrs': {'id': 'elicitor', 'title': 'Elicitor starship', 'side': 'test'}
}


probe_starship_mixin = {
    'manifest': CAL_PATH + 'objects.Starship',
    'required': True,
    'attrs': {'id': 'probe', 'title': 'Probe starship', 'side': 'nontest'}
}


ParadigmDescription(
    'memr_interleaved_click', 'MEMR (Keefe, click)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.InterleavedClickMEMRManifest'},
        {'manifest': CFTS_PATH + 'memr.InterleavedElicitorMixin', 'required': True},
        {'manifest': CFTS_PATH + 'memr.InterleavedClickProbeMixin', 'required': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'info': {'hide': True}},
        multi_memr_microphone_mixin,
        multi_memr_microphone_fft_mixin,
        turntable_linear_velocity_view_mixin_required,
        turntable_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['ipsi', 'contra']},
)


ParadigmDescription(
    'memr_interleaved_chirp', 'MEMR (Keefe, chirp)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.InterleavedChirpMEMRManifest'},
        {'manifest': CFTS_PATH + 'memr.InterleavedElicitorMixin', 'required': True},
        {'manifest': CFTS_PATH + 'memr.InterleavedChirpProbeMixin', 'required': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'info': {'hide': True}},
        multi_memr_microphone_mixin,
        multi_memr_microphone_fft_mixin,
        turntable_linear_velocity_view_mixin_required,
        turntable_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['ipsi', 'contra']},
)


ParadigmDescription(
    'memr_simultaneous_click', 'MEMR (Valero, click)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.SimultaneousMEMRManifest', 'attrs': {'probe': 'click'}},
        {'manifest': CFTS_PATH + 'memr.SimultaneousClickProbeMixin', 'required': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'info': {'hide': True}},
        multi_memr_microphone_mixin,
        multi_memr_microphone_fft_mixin,
        turntable_linear_velocity_view_mixin_required,
        turntable_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['contra']},
)


ParadigmDescription(
    'memr_simultaneous_chirp', 'MEMR (Valero, chirp)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.SimultaneousMEMRManifest', 'attrs': {'probe': 'chirp'}},
        {'manifest': CFTS_PATH + 'memr.SimultaneousChirpProbeMixin', 'required': True},
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'info': {'hide': True}},
        multi_memr_microphone_mixin,
        multi_memr_microphone_fft_mixin,
        turntable_linear_velocity_view_mixin_required,
        turntable_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['contra']},
)


ParadigmDescription(
    'memr_sweep_click', 'MEMR (sweep, click)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.SweptMEMRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'info': {'hide': True}},
        multi_memr_microphone_mixin,
        multi_memr_microphone_fft_mixin,
        turntable_linear_velocity_view_mixin_required,
        turntable_mixin,
    ] + COMMON_PLUGINS,
    info={'modes': ['contra']},
)
