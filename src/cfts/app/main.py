from enaml.qt.QtCore import Qt
from enaml.qt.qt_application import QtApplication
import enaml
# psi.core.app_id rather than psi.launcher.api: the latter pulls in .enaml modules.
from psi.core.app_id import set_app_id
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

    # Before the Qt application is created, and distinct from the `psi.psi`
    # that the experiment subprocesses claim, so cfts gets its own taskbar
    # button showing its icon rather than python.exe's.
    set_app_id('psi.cfts')
    app = QtApplication()
    view = ExpLauncherMain()

    # This needs to be loaded to ensure that some defaults are set properly.
    view.settings.load_config(args.experiment_config)
    view.settings.load_hardware_config(args.hardware_config)

    # Settings are usually saved from within the experiment subprocess (or
    # copied in via Explorer), so rescan the available preference files each
    # time the user switches back to the launcher.
    def refresh_preferences(state):
        if state == Qt.ApplicationState.ApplicationActive:
            view.settings.refresh_preferences()
    app._qapp.applicationStateChanged.connect(refresh_preferences)

    view.show()
    app.start()
    return True
