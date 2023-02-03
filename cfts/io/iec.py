from psi.data.io.api import Recording


class IEC(Recording):

    def __init__(self, base_path, setting_table='epoch_metadata'):
        super().__init__(base_path, setting_table)

    def get_epochs(self, columns='auto', offset=0, extra_duration=5e-3, cb=None):
        signal = getattr(self, 'hw_ai')
        duration = self.get_setting('hw_ao_chirp_duration')
        return signal.get_epochs(self.epoch_metadata, offset,
                                 duration+extra_duration, cb=cb)
