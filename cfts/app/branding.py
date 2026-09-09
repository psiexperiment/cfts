import importlib.resources


def load_app_icon():
    '''
    Load cfts/icons/main-icon.png as an Enaml `Icon` for window branding.

    Uses `importlib.resources` rather than a `__file__`-relative path so
    this keeps working if cfts is ever installed as a zipped wheel.
    '''
    from enaml.icon import Icon, IconImage
    from enaml.image import Image

    data = importlib.resources.files('cfts').joinpath(
        'icons', 'main-icon.png').read_bytes()
    return Icon(images=[IconImage(image=Image(data=data, format='png'))])
