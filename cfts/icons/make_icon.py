import matplotlib as mp
import matplotlib.pyplot as plt
from matplotlib import patheffects as pe
import numpy as np
from PIL import Image


def make_main_icon():
    fig = plt.figure(frameon=False)
    fig.set_size_inches(1, 1)
    ax = plt.Axes(fig, [0, 0, 1, 1])
    ax.set_axis_off()
    fig.add_axes(ax)

    background = mp.patches.Rectangle([0, 0], width=1, height=1, facecolor='midnightblue',
                                       edgecolor='none', transform=ax.transAxes)
    ax.add_patch(background)

    # DPOAE-style spectral peaks: f1, f2 primaries plus the smaller
    # 2*f1-f2 distortion product, evoking the otoacoustic emission
    # measurements this suite records.
    spline_effect = [
        pe.Stroke(linewidth=5, foreground="white"),
    ]
    xs = [0.25, 0.5, 0.75]
    heights = [0.25, 1.25, 1.0]
    widths = [0.075, 0.075, 0.075]
    for x, h, w in zip(xs, heights, widths):
        poly = np.array([[x - w, -1.5], [x, h], [x + w, -1.5]])
        patch = mp.patches.Polygon(poly, closed=True, facecolor='cornflowerblue',
                                    edgecolor='none')
        ax.add_patch(patch)
        ax.plot(poly[:, 0], poly[:, 1], color='none', solid_capstyle='round',
                 solid_joinstyle='round', path_effects=spline_effect)
    ax.axhline(-1.5, color='white', linewidth=3)

    ax.axis(xmin=-0.05, xmax=1.05, ymin=-1.5, ymax=2.0)

    border = mp.patches.Rectangle([0, 0], width=1, height=1, facecolor='none',
                                   edgecolor='white', linewidth=10,
                                   transform=ax.transAxes, zorder=3)
    ax.add_patch(border)
    fig.savefig('main-icon.png', transparent=False, bbox_inches='tight')


def make_ico():
    im = Image.open('main-icon.png').convert('RGBA')
    im.save('main-icon.ico', sizes=[(16, 16), (24, 24), (32, 32), (48, 48), (64, 64)])


make_main_icon()
make_ico()
