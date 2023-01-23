import logging
log = logging.getLogger(__name__)

from pathlib import Path

import pandas as pd

from psiaudio.calibration import InterpCalibration


def load_epl_calibration(filename):
    with Path(filename).open('r') as fh:
        for line in fh:
            if line.startswith('Freq(Hz)'):
                break
        cal = pd.read_csv(fh, sep='\t', header=None)
        return InterpCalibration.from_spl(cal[0], cal[1])


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
    if len(channels) == 0:
        return

    # If only one channel is active, we don't have any sync issues.
    if len(channels) == 1:
        channels[0].start_trigger = None
        return

    if ao_channels:
        c = ao_channels[0]
        direction = 'ao'
    else:
        c = ai_channels[0]
        direction = 'ai'

    dev = c.channel.split('/', 1)[0]
    trigger = f'/{dev}/{direction}/StartTrigger'
    for c in channels:
        if dev in c.channel and direction in c.channel:
            log.info(f'Setting {c} start_trigger to ""')
            c.start_trigger = ''
        else:
            log.info(f'Setting {c} start_trigger to "{trigger}"')
            c.start_trigger = trigger

    # Now, make sure the master engine is set to the one that controls the
    # start trigger.
    controller._master_engine = c.engine
