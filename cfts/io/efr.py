from psi.data.io.api import Recording


class EFR(Recording):

    def __init__(self, filename, setting_table='analyze_efr_metadata'):
        super().__init__(filename, setting_table)

    def get_eeg_epochs(self, **kwargs):
        pass

    def get_mic_epochs(self, **kwargs):
        pass
