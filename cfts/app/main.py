from enaml.qt.qt_application import QtApplication
import enaml
with enaml.imports():
    from .exp_launcher_gui import Main as ExpLauncherMain


from cfts import paradigms
from cftscal import paradigms


def cfts():
    import argparse
    parser = argparse.ArgumentParser('cfts')
    parser.add_argument('experiment_config', nargs='?', help=(
        'Path to the experiment config file (animal/experimenter defaults, '
        'the experiment/monitor sequences and saved sequences, and each '
        'row\'s plot/save channel settings) to load. Can also be loaded '
        'afterward via File > Load config.'
    ))
    parser.add_argument('hardware_config', nargs='?', help=(
        'Path to the shared hardware config file (input/output channel '
        'calibration and gain) to load alongside the experiment config. '
        'Can also be loaded afterward via File > Load hardware config.'
    ))

    args = parser.parse_args()
    app = QtApplication()
    view = ExpLauncherMain()

    # This needs to be loaded to ensure that some defaults are set properly.
    view.settings.load_config(args.experiment_config)
    view.settings.load_hardware_config(args.hardware_config)

    view.show()
    app.start()
    return True
