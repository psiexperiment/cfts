from enaml.qt.qt_application import QtApplication
import enaml
with enaml.imports():
    from .exp_launcher_gui import Main as ExpLauncherMain


from cfts import paradigms
from cftscal import paradigms


def cfts():
    import argparse
    parser = argparse.ArgumentParser('cfts')
    parser.add_argument('config', nargs='?')

    args = parser.parse_args()
    app = QtApplication()
    view = ExpLauncherMain()

    # This needs to be loaded to ensure that some defaults are set properly.
    view.settings.load_config(args.config)

    view.show()
    app.start()
    return True
