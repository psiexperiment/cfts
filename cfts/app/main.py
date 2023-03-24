from enaml.qt.qt_application import QtApplication
import enaml
with enaml.imports():
    from .exp_launcher_gui import Main as ExpLauncherMain


from psi.application import load_paradigm_descriptions


def cfts():
    import argparse
    parser = argparse.ArgumentParser('cfts')
    parser.add_argument('config')

    args = parser.parse_args()
    load_paradigm_descriptions()
    app = QtApplication()
    view = ExpLauncherMain()

    view.settings.load_config(args.config)
    view.show()
    app.start()
    return True
