'''
Default values for every cfts setting.

Registered with :func:`psi.config.register_defaults` when ``cfts`` is
imported, so ``get_config('CFTS_ROOT')`` resolves the same way as any psi
setting: default, then ``config.toml``, then the environment.
'''
from psi import get_config


DEFAULTS = {
    #: Where the cfts launcher keeps the experiment and hardware presets
    #: the user saves and loads through its File menu (under a
    #: `cfts-launcher` subfolder). Distinct from CFTSCAL_ROOT, which is
    #: where calibration data lives, and from the CFTSCAL_* handoff
    #: variables, which are a process contract rather than settings.
    'CFTS_ROOT': lambda: get_config('PSI_BASE_DIRECTORY') / 'cfts',
}
