import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# presentation colors
cdarkprime = '#0E1E97'
cmediumprime = '#3246e0'
clightprime = '#e2e5fe'
cdarkcomp = '#9b7c0f'
cmediumcomp = '#f6c51c'
clightcomp = '#fef8e2'
cdarkgrey = '#303030'
cmediumgrey = '#888888'
clightgrey = '#e5e5e5'
cdarkthree = '#11990f'
cmediumthree = '#2dea2a'
clightthree = '#e4fce3'
cdarkfour = '#950e0e'
cmediumfour = '#eb2828'
clightfour = '#fce3e3'

almostblack = cdarkgrey
color_cycle = ["#001C7F", "#017517", "#8C0900", "#7600A1", "#B8860B",
               "#006374"]

# Aesthetic ratio
golden_mean = (5**0.5-1.0)/2.0
# Convert pt to inch
inches_per_pt = 1.0/72.27
# Convert cm to inch
inches_per_cm = 0.393701

# three different types of figures (1, 1.5, 2 columns)
# textwidth in cm
width = 8.7
onehalfwidth = 11.4
doublewidth = 17.8
figsize = np.array([width * inches_per_cm,
                    width * inches_per_cm * golden_mean])
figsize_onehalf = np.array([onehalfwidth * inches_per_cm,
                            onehalfwidth * inches_per_cm * golden_mean])
figsize_double = np.array([doublewidth * inches_per_cm,
                           doublewidth * inches_per_cm * golden_mean])

fontsize = 9
linewidth = 0.75
# unfortunately the labelpad parameter can not be set via rcparams currently
axis_labelpad = 2
tight_layout_pad = 0.75

linedotstyle = dict(linestyle='-', linewidth=0.75 * linewidth,
                    marker='.', ms=1.25)

rcparams = {
    'figure.figsize': tuple(figsize),
    'font.size': fontsize,
    'font.family': 'serif',
    'font.serif': 'cm',
    'lines.linewidth': linewidth,
    'axes.linewidth': linewidth,
    'lines.markeredgewidth': linewidth,
    'xtick.major.pad': 3,
    'ytick.major.pad': 3,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
    'lines.markersize': 2,
    'axes.labelsize': 'medium',
    'legend.fontsize': 'medium',
    'legend.handletextpad': 0.3,
    'legend.handlelength': 1.5,
    'legend.numpoints': 3,
    'legend.frameon': False,
    'legend.columnspacing': 1.,
    'text.usetex': True,
    'text.latex.preamble': [r'\usepackage{amsmath}'],
    'savefig.dpi': 600,
    'axes.color_cycle': color_cycle,
    # set normally black figure elements to grey for nicer plot
    'ytick.color': almostblack,
    'xtick.color': almostblack,
    'axes.edgecolor': almostblack,
    'axes.labelcolor': almostblack,
    'text.color': almostblack,
    'grid.color': almostblack,
    }
matplotlib.rcParams.update(rcparams)
