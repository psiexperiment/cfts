from psi.experiment.api import ParadigmDescription


CAL_PATH = 'psi.paradigms.calibration.'
CFTS_PATH = 'cfts.paradigms.'
CORE_PATH = 'psi.paradigms.core.'


################################################################################
# Single-starship paradigms (ABR, EFR, DPOAE, IEC)
################################################################################
selectable_starship_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.Starship',
    'required': True,
    'attrs': {'id': 'system', 'title': 'Starship', 'output_mode': 'select'}
}


dual_starship_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.Starship',
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
        'y_label': 'Microphone (dB)'
    }
}


eeg_view_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'eeg_view_mixin',
        'name': 'eeg_view',
        'title': 'EEG display',
        'time_span': 2,
        'time_delay': 0.125,
        'source_name': 'eeg_filtered',
        'y_label': 'EEG (V)'
    }
}
eeg_view_mixin_required = eeg_view_mixin.copy()
eeg_view_mixin_required['required'] = True


temperature_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.TemperatureMixinManifest',
    'selected': True,
}


efr_microphone_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'microphone_fft_view',
        'title': 'Microphone view (PSD)',
        'fft_time_span': 5,
        'source_name': 'system_microphone',
        'y_label': 'Microphone (dB)'
    }
}


ParadigmDescription(
    'monitor', 'Monitor', 'ear', [
        temperature_mixin,
        eeg_view_mixin_required,
        {'manifest': CFTS_PATH + 'monitor.MonitorManifest', 'selected': True},
    ],
)


ParadigmDescription(
    # This is the default, simple ABR experiment that most users will want.  
    'abr_io', 'ABR (input-output)', 'ear', [
        selectable_starship_mixin,
        {'manifest': CFTS_PATH + 'abr_io.ABRIOSimpleManifest'},
        temperature_mixin,
        eeg_view_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.ABRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'dpoae_io', 'DPOAE (input-output)', 'ear', [
        dual_starship_mixin,
        {'manifest': CFTS_PATH + 'dpoae_io.DPOAEIOSimpleManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.DPOAEInEarCalibrationMixinManifest', 'selected': True},
        temperature_mixin,
        microphone_mixin,
        microphone_fft_mixin,
    ],
)


ParadigmDescription(
    'efr_sam', 'SAM EFR', 'ear', [
        selectable_starship_mixin,
        {'manifest': CFTS_PATH + 'efr.SAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.SAMEFRInEarCalibrationMixinManifest', 'selected': True},
        temperature_mixin,
        microphone_mixin,
        efr_microphone_fft_mixin,
        eeg_view_mixin,
    ]
)


ParadigmDescription(
    'efr_ram', 'RAM EFR', 'ear', [
        selectable_starship_mixin,
        {'manifest': CFTS_PATH + 'efr.RAMEFRManifest'},
        {'manifest': CFTS_PATH + 'cfts_mixins.RAMEFRInEarCalibrationMixinManifest', 'selected': True},
        temperature_mixin,
        microphone_mixin,
        efr_microphone_fft_mixin,
        eeg_view_mixin,
    ]
)


ParadigmDescription(
    'inear_speaker_calibration_chirp', 'In-ear speaker calibration (chirp)', 'ear', [
        selectable_starship_mixin,
        {'manifest': CAL_PATH + 'speaker_calibration.BaseSpeakerCalibrationManifest'},
        {'manifest': CAL_PATH + 'calibration_mixins.ChirpMixin'},
        {'manifest': CAL_PATH + 'calibration_mixins.ToneValidateMixin'},
    ]
)


################################################################################
# Two-starship paradigms for MEMR
################################################################################
elicitor_mic_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'attrs': {
        'id': 'elicitor_microphone_signal_view',
        'title': 'Elicitor microphone (time)',
        'time_span': 4,
        'time_delay': 0.125,
        'source_name': 'elicitor_microphone',
        'y_label': 'Signal (V)'
    },
}


elicitor_mic_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'attrs': {
        'id': 'elicitor_microphone_fft_view',
        'title': 'Elicitor microphone (PSD)',
        'fft_time_span': 0.25,
        'fft_freq_lb': 500,
        'fft_freq_ub': 50000,
        'source_name': 'elicitor_microphone',
        'y_label': 'Signal (dB)'
    }
}


probe_mic_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalViewManifest',
    'required': True,
    'attrs': {
        'id': 'probe_microphone_signal_view',
        'name': 'probe_microphone_signal_view',
        'title': 'Probe microphone (time)',
        'time_span': 4,
        'time_delay': 0.125,
        'source_name': 'probe_microphone',
        'y_label': 'Signal (V)'
    },
}


probe_mic_fft_mixin = {
    'manifest': CORE_PATH + 'signal_mixins.SignalFFTViewManifest',
    'required': True,
    'attrs': {
        'id': 'probe_microphone_fft_view',
        'name': 'probe_microphone_fft_view',
        'title': 'Probe microphone (PSD)',
        'fft_time_span': 0.25,
        'fft_freq_lb': 500,
        'fft_freq_ub': 50000,
        'source_name': 'probe_microphone',
        'y_label': 'Signal (dB)'
    }
}


elicitor_starship_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.Starship',
    'required': True,
    'attrs': {'id': 'elicitor', 'title': 'Elicitor starship'}
}


probe_starship_mixin = {
    'manifest': CFTS_PATH + 'cfts_mixins.Starship',
    'required': True,
    'attrs': {'id': 'probe', 'title': 'Probe starship'}
}


ParadigmDescription(
    'memr_interleaved_click', 'MEMR (interleaved click)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.InterleavedMEMRManifest', 'attrs': {'probe': 'click'}},
        {'manifest': CFTS_PATH + 'memr.InterleavedElicitorMixin', 'required': True},
        {'manifest': CFTS_PATH + 'memr.InterleavedClickProbeMixin', 'required': True},
        temperature_mixin,
        elicitor_mic_mixin,
        elicitor_mic_fft_mixin,
        probe_mic_mixin,
        probe_mic_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'memr_interleaved_chirp', 'MEMR (interleaved chirp)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.InterleavedMEMRManifest', 'attrs': {'probe': 'chirp'}},
        {'manifest': CFTS_PATH + 'memr.InterleavedElicitorMixin', 'required': True},
        {'manifest': CFTS_PATH + 'memr.InterleavedChirpProbeMixin', 'required': True},
        temperature_mixin,
        elicitor_mic_mixin,
        elicitor_mic_fft_mixin,
        probe_mic_mixin,
        probe_mic_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'memr_simultaneous_click', 'MEMR (simultaneous click)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.SimultaneousMEMRManifest', 'attrs': {'probe': 'click'}},
        {'manifest': CFTS_PATH + 'memr.SimultaneousClickProbeMixin', 'required': True},
        temperature_mixin,
        elicitor_mic_mixin,
        elicitor_mic_fft_mixin,
        probe_mic_mixin,
        probe_mic_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'selected': True},
    ]
)


ParadigmDescription(
    'memr_simultaneous_chirp', 'MEMR (simultaneous chirp)', 'ear', [
        elicitor_starship_mixin,
        probe_starship_mixin,
        {'manifest': CFTS_PATH + 'memr.SimultaneousMEMRManifest', 'attrs': {'probe': 'chirp'}},
        {'manifest': CFTS_PATH + 'memr.SimultaneousChirpProbeMixin', 'required': True},
        temperature_mixin,
        elicitor_mic_mixin,
        elicitor_mic_fft_mixin,
        probe_mic_mixin,
        probe_mic_fft_mixin,
        {'manifest': CFTS_PATH + 'cfts_mixins.MEMRInEarCalibrationMixinManifest', 'selected': True},
    ]
)
