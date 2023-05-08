import logging
log = logging.getLogger(__name__)


def connect_trigger(event):
    '''
    Utility function that verifies that start trigger is set appropriately
    since some channels may be disabled.

    This needs to be explicitly hooked up in your IOManifest.
    '''
    # Since we want to make sure timing across all engines in the task are
    # synchronized properly, we need to inspect for the active channels and
    # then determine which device task is the one to syncrhonize all other
    # tasks to. We prioritize the last engine (the one that is started last)
    # and prioritize the analog output over the analog input. This logic may
    # change in the future.
    controller = event.workbench.get_plugin('psi.controller')

    ai_channels = []
    ao_channels = []
    for engine in list(controller._engines.values())[::-1]:
        hw_ai = engine.get_channels(direction='in', timing='hw', active=True)
        hw_ao = engine.get_channels(direction='out', timing='hw', active=True)
        ai_channels.extend(hw_ai)
        ao_channels.extend(hw_ao)

    channels = ai_channels + ao_channels

    # If no channels are active, we don't have any sync issues.
    log.info('No channels are active.')
    if len(channels) == 0:
        return

    # If only one channel is active, we don't have any sync issues.
    if len(channels) == 1:
        log.info('Only one channel active. Disabling start trigger.')
        channels[0].start_trigger = ''
        return

    if ao_channels:
        c = ao_channels[0]
        direction = 'ao'
    else:
        c = ai_channels[0]
        direction = 'ai'

    dev = c.channel.split('/', 1)[0]
    trigger = f'/{dev}/{direction}/StartTrigger'
    master_engine = None
    for c in channels:
        if dev in c.channel and direction in c.channel:
            log.info(f'Setting {c} start_trigger to ""')
            c.start_trigger = ''
            if master_engine is None:
                master_engine = c.engine
        else:
            log.info(f'Setting {c} start_trigger to "{trigger}"')
            c.start_trigger = trigger

    # Now, make sure the master engine is set to the one that controls the
    # start trigger.
    log.info('Setting master engine to %s', master_engine.name)
    controller._master_engine = master_engine
