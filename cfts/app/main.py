from enaml.qt.qt_application import QtApplication
import enaml
with enaml.imports():
    from .exp_launcher_gui import Main as ExpLauncherMain


from psi.application import load_paradigm_descriptions


def cfts():
    load_paradigm_descriptions()
    app = QtApplication()
    view = ExpLauncherMain()
    view.show()
    app.start()
    return True
